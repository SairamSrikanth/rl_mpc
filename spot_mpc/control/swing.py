import torch

from .footholds import foothold
from .gait import command_world

BIG = 1e9


def _delay_like(p0, delay):
    # a tensor, not a float: dividing by a python scalar on the GPU multiplies by the reciprocal and changes the last bit
    return torch.full_like(p0[..., 2], delay)


def _smoothstep(u):
    return 3 * u**2 - 2 * u**3


def swing_apex(p0, p_td, terrain, clearance=0.06, base_apex=0.08, delay=0.2, n=80, ahead=0.04, behind=0.02, slack=0.02, edge_tol=0.02):
    """Lowest apex height (B,4) for a swing p0 -> p_td that clears the terrain by `clearance`.

    The vertical profile is linear in the apex, so every sample along the swing gives a lower bound directly. Clearance is only
    demanded where the foot's neighbourhood contains a height change above edge_tol.
    """
    dev, dt = p0.device, p0.dtype
    s = torch.linspace(0.0, 1.0, n, device=dev, dtype=dt).expand(*p0.shape[:-1], n)
    d0 = _delay_like(p0, delay)[..., None]
    sh = ((s - d0) / (1.0 - d0)).clamp(0.0, 1.0)
    sig = 10 * sh**3 - 15 * sh**4 + 6 * sh**5
    d = torch.where(p_td[..., 0] >= p0[..., 0], torch.ones_like(p0[..., 0]), -torch.ones_like(p0[..., 0]))
    x = p0[..., 0:1] + (p_td[..., 0:1] - p0[..., 0:1]) * sig
    y = p0[..., 1:2] + (p_td[..., 1:2] - p0[..., 1:2]) * sig
    u6 = torch.linspace(0.0, 1.0, 6, device=dev, dtype=dt)
    xs = x[..., None] - (d * behind)[..., None, None] + (d * (ahead + behind))[..., None, None] * u6
    hs = getattr(terrain, "swing_height", terrain.height)(xs, y[..., None].expand_as(xs))
    hmax, hmin = hs.amax(-1), hs.amin(-1)
    need_z = hmax + (clearance if isinstance(clearance, (int, float)) else clearance.reshape(-1, 1, 1)) + slack
    up = s <= 0.5
    u = torch.where(up, s / 0.5, (s - 0.5) / 0.5)
    w = _smoothstep(u)
    coeff = torch.where(up, w, 1.0 - w)
    const = torch.where(up, p0[..., 2:3] * (1.0 - w), p_td[..., 2:3] * w)
    need = ((hmax - hmin) > edge_tol) & (coeff > 1e-3)
    zreq = torch.where(need, (need_z - const) / coeff.clamp(min=1e-3), torch.full_like(need_z, -BIG))
    zmax = zreq.amax(-1) + 1e-3
    base_extra = base_apex if isinstance(base_apex, (int, float)) else base_apex.reshape(-1, 1)
    base = torch.maximum(p0[..., 2], p_td[..., 2]) + base_extra
    return torch.where(zmax > -BIG / 2, torch.maximum(base, zmax), base)


def plan_swing(P, terrain, gait, x0, cmd, anchor, p0, clearance=None, base_apex=None, foot_res=None, apex_res=None):
    """Touchdown point (B,4,3) and apex height (B,4) for every leg (only the swinging ones are used)."""
    v_w, wz = command_world(x0, cmd, anchor[0])
    td = gait.time_to_touchdown()
    com_td = x0[:, None, 3:5] + v_w[:, None, :] * td[..., None]
    yaw_td = x0[:, 2:3] + wz[:, None] * td
    p_td = foothold(P, terrain, com_td, yaw_td, x0[:, None, 9:11], v_w[:, None, :], gait.stance_duration[:, None], foot_res)
    apex = swing_apex(p0, p_td, terrain, P.clearance if clearance is None else clearance, P.base_apex if base_apex is None else base_apex,
                      delay=P.swing_delay, edge_tol=P.edge_tol)
    top = torch.maximum(p0[..., 2], p_td[..., 2])
    if apex_res is not None:
        apex = torch.maximum(apex + apex_res, top + 0.02)
    return p_td, torch.minimum(apex, top + P.max_apex)


def swing_target(p0, p_td, s, t_sw, apex, delay=0.2):
    """Foot position and velocity (B,4,3) at swing progress s: minimum-jerk in the plane, smoothstep up to the apex and down."""
    s = s.clamp(0.0, 1.0)
    d0 = _delay_like(p0, delay)
    sh = ((s - d0) / (1.0 - d0)).clamp(0.0, 1.0)
    sig, dsig = 10 * sh**3 - 15 * sh**4 + 6 * sh**5, 30 * sh**2 * (1 - sh) ** 2
    tsw = t_sw[:, None]
    pos_xy = p0[..., :2] + (p_td[..., :2] - p0[..., :2]) * sig[..., None]
    vel_xy = (p_td[..., :2] - p0[..., :2]) * dsig[..., None] / ((1.0 - d0[..., None]) * tsw[..., None])
    up = s <= 0.5
    u = torch.where(up, s / 0.5, (s - 0.5) / 0.5)
    w, dw = _smoothstep(u), 6 * u - 6 * u**2
    z0, zt = p0[..., 2], p_td[..., 2]
    z = torch.where(up, z0 + (apex - z0) * w, apex + (zt - apex) * w)
    vz = torch.where(up, apex - z0, zt - apex) * dw / 0.5 / tsw
    return torch.cat([pos_xy, z[..., None]], -1), torch.cat([vel_xy, vz[..., None]], -1)


class SwingPlanner:
    """Per-leg swing bookkeeping: lift-off point, touchdown plan and apex, frozen once a leg is past mid-swing."""

    def __init__(self, num_envs, device, delay=0.2, dtype=torch.float32):
        z3 = torch.zeros(num_envs, 4, 3, device=device, dtype=dtype)
        self.delay = delay
        self.p0, self.p_td = z3.clone(), z3.clone()
        self.apex = torch.zeros(num_envs, 4, device=device, dtype=dtype)
        self.frozen = torch.zeros(num_envs, 4, device=device, dtype=torch.bool)
        self.was_c = torch.ones(num_envs, 4, device=device, dtype=torch.bool)

    def reset(self, env_ids, feet):
        self.p0[env_ids] = feet[env_ids]
        self.p_td[env_ids] = feet[env_ids]
        self.apex[env_ids] = feet[env_ids, :, 2]
        self.frozen[env_ids] = False
        self.was_c[env_ids] = True

    def observe(self, c_now, feet):
        """Latch the lift-off point of legs that just left the ground (every physics step)."""
        liftoff = self.was_c & ~c_now
        self.p0 = torch.where(liftoff[..., None], feet, self.p0)
        self.was_c = c_now

    def update(self, P, terrain, gait, x0, cmd, anchor, feet, c_now, **plan_kw):
        """Re-plan legs that swing now or lift off before the next tick (every MPC tick)."""
        s_now = gait.swing_progress()
        will_swing = gait.phase_ahead(torch.tensor([P.dt], device=x0.device, dtype=x0.dtype))[:, 0] >= gait.duty[:, None]
        need = (~c_now | will_swing) & ~self.frozen
        p0_use = torch.where(c_now[..., None], feet, self.p0)
        p_td, apex = plan_swing(P, terrain, gait, x0, cmd, anchor, p0_use, **plan_kw)
        self.p_td = torch.where(need[..., None], p_td, self.p_td)
        self.apex = torch.where(need, apex, self.apex)
        self.frozen = (self.frozen | (need & ~c_now & (s_now >= 0.5))) & ~c_now

    def target(self, gait):
        return swing_target(self.p0, self.p_td, gait.swing_progress(), gait.swing_duration, self.apex, self.delay)
