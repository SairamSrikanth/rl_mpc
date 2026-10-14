import torch
from isaaclab_assets.robots.spot import SPOT_CFG

LEGS = ("fl", "fr", "hl", "hr")
JOINTS = ("hx", "hy", "kn")
GRAVITY = 9.81
TAU_LIM = (45.0, 45.0, 90.0)      # hip abduction, hip flexion, knee limits used by the MPC [N m]
SPOT_INERTIA_B = ((0.583, 0.0, -0.0781), (0.0, 2.2076, 0.0), (-0.0781, 0.0, 2.3596))


def to_t(x):
    if isinstance(x, torch.Tensor):
        return x
    t = getattr(x, "torch", None)
    if t is not None:
        return t() if callable(t) else t
    import warp as wp
    return wp.to_torch(x)


def pick(data, *names):
    """First attribute of `data` that exists, as a torch tensor (the names changed between Isaac Lab versions)."""
    for n in names:
        if hasattr(data, n):
            return to_t(getattr(data, n))
    raise AttributeError(f"none of {names} found on {type(data).__name__}")


def physx_view(robot):
    for n in ("root_physx_view", "root_view"):
        if hasattr(robot, n):
            return getattr(robot, n)
    raise AttributeError("articulation has no physx view")


def quat_to_rotmat(q):
    w, x, y, z = q.unbind(-1)
    r = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1)
    return r.reshape(*q.shape[:-1], 3, 3)


def roll_pitch_yaw(R):
    pitch = torch.asin((-R[..., 2, 0]).clamp(-1.0, 1.0))
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return roll, pitch, yaw


def make_spot_torque_cfg():
    """Spot with every actuator group turned into a pure torque source (the stock torque-limit models are kept)."""
    actuators = {}
    for name, act in SPOT_CFG.actuators.items():
        kw = dict(stiffness=0.0, damping=0.0)
        if hasattr(act, "max_delay"):
            kw.update(min_delay=0, max_delay=0)
        actuators[name] = act.replace(**kw)
    return SPOT_CFG.replace(actuators=actuators)


class SpotLegs:
    """Batched foot kinematics and body state read from the articulation."""

    def __init__(self, robot, foot_suffix="foot"):
        assert not robot.is_fixed_base, "expected a floating-base Spot"
        self.robot, self.data, self.dev = robot, robot.data, robot.device
        jids, _ = robot.find_joints([f"{l}_{j}" for l in LEGS for j in JOINTS], preserve_order=True)
        bids, _ = robot.find_bodies([f"{l}_{foot_suffix}" for l in LEGS], preserve_order=True)
        self.jids = torch.as_tensor(jids, device=self.dev, dtype=torch.long)
        self.bids = [int(b) for b in bids]
        self.masses = to_t(physx_view(robot).get_masses()).to(self.dev).float()
        self.total_mass = self.masses.sum(1)

    def _p(self, *names):
        return pick(self.data, *names).float()

    def root_rot(self):
        return quat_to_rotmat(self._p("root_link_quat_w", "root_quat_w"))

    def root_pos_w(self):
        return self._p("root_link_pos_w", "root_pos_w")

    def root_lin_vel_w(self):
        return self._p("root_link_lin_vel_w", "root_lin_vel_w")

    def root_ang_vel_w(self):
        return self._p("root_link_ang_vel_w", "root_ang_vel_w")

    def q(self):
        return self._p("joint_pos")[:, self.jids]

    def qd(self):
        return self._p("joint_vel")[:, self.jids]

    def foot_pos_w(self):
        return self._p("body_link_pos_w", "body_pos_w")[:, self.bids]

    def foot_vel_w(self):
        return self._p("body_link_lin_vel_w", "body_lin_vel_w")[:, self.bids]

    def foot_ang_vel_w(self):
        return self._p("body_link_ang_vel_w", "body_ang_vel_w")[:, self.bids]

    def com_state(self):
        """Mass-weighted CoM position and velocity (world frame)."""
        m = self.masses.unsqueeze(-1)
        M = self.total_mass.unsqueeze(-1)
        com = (m * self._p("body_com_pos_w", "body_pos_w")).sum(1) / M
        vel = (m * self._p("body_com_lin_vel_w", "body_lin_vel_w")).sum(1) / M
        return com, vel

    def leg_jacobians(self):
        """(N,4,3,3): world-frame linear Jacobian of each foot with respect to its own three joints."""
        J = to_t(physx_view(self.robot).get_jacobians()).float()
        cols = (6 + self.jids).view(4, 3)          # the first 6 columns belong to the floating base
        return torch.stack([J[:, b][:, :3][:, :, cols[i]] for i, b in enumerate(self.bids)], 1)


def joint_pd(legs, q_des, kp=250.0, kd=5.0):
    return kp * (q_des - legs.q()) - kd * legs.qd()


def apply_torque(robot, legs, tau12):
    full = torch.zeros_like(pick(robot.data, "joint_pos")).float()
    full[:, legs.jids] = tau12
    robot.set_joint_effort_target(full)


def randomize_material(robot, env_ids, friction_range):
    """One random friction value per env, applied to every collider of the articulation (host-side tensor API)."""
    view = physx_view(robot)
    ids = env_ids.cpu()
    mat = to_t(view.get_material_properties()).clone()
    lo, hi = friction_range
    fric = lo + (hi - lo) * torch.rand(len(ids), device=mat.device, dtype=mat.dtype)
    mat[ids, :, 0] = fric[:, None]
    mat[ids, :, 1] = fric[:, None]
    view.set_material_properties(mat, ids)


def randomize_base_mass(robot, legs, env_ids, nominal_mass, added_mass_range, base_bid):
    """Add a fresh random payload to the base link (drawn from the nominal mass, never compounding) and keep the cached masses in sync."""
    view = physx_view(robot)
    ids = env_ids.cpu()
    masses = to_t(view.get_masses()).clone()
    lo, hi = added_mass_range
    delta = lo + (hi - lo) * torch.rand(len(ids), device=masses.device, dtype=masses.dtype)
    masses[ids, base_bid] = nominal_mass[env_ids].to(masses.device, masses.dtype) + delta
    view.set_masses(masses, ids)
    legs.masses[env_ids, base_bid] = masses[ids, base_bid].to(legs.masses.device, legs.masses.dtype)
    legs.total_mass[env_ids] = legs.masses[env_ids].sum(1)
