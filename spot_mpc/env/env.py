import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera, RayCasterCameraCfg, RayCasterCfg, patterns

from ..control import gait as gs
from ..control.footholds import surface_normals
from ..control.mpc import BatchedSrbdMpc, MpcParams
from ..control.swing import SwingPlanner
from ..control.torque_interface import GRAVITY, LEGS, SPOT_INERTIA_B, TAU_LIM, SpotLegs, apply_torque, joint_pd, roll_pitch_yaw
from ..perception.elevation_map import ElevationMap
from ..perception.sensors import depth_points, lidar_points
from ..worlds.generated.world import GeneratedWorld
from ..worlds.test_world import World
from ..worlds.warehouse_static.world import Warehouse
from .cfg import ACT_ACCEL, ACT_APEX, ACT_FOOT, ACT_FREQ, ACT_META, DECIMATION, NUM_ACTIONS, PHYSICS_DT, SpotMpcEnvCfg  # noqa: F401
from .scheduler import blend, previewed_slopes


def _quat_mul(a, b):
    return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


def _tilted(rot, pitch_deg):
    """Camera rotation tilted about the body y axis by pitch_deg."""
    th = math.radians(pitch_deg)
    return _quat_mul((math.cos(th / 2), 0.0, math.sin(th / 2), 0.0), rot)


def _step(cur, target, max_step):
    return cur + (target - cur).clamp(-max_step, max_step)


class SpotMpcEnv(DirectRLEnv):
    """Spot driven by the batched SRB-MPC, a clock gait and terrain-aware swing legs. One env step is one 40 Hz MPC tick
    (ten 400 Hz physics steps). The policy action is a set of residuals; a zero action is the pure MPC.
    Set env.cmd_target to command a body-frame velocity (vx, vy, wz)."""

    cfg: SpotMpcEnvCfg

    def __init__(self, cfg, render_mode=None, **kw):
        self.world = None
        self._yaw_prev = None
        super().__init__(cfg, render_mode, **kw)
        dev, B = self.device, self.num_envs
        self.world.to(dev)
        self.legs = SpotLegs(self._robot, "foot")
        self.elev_map = None
        if cfg.terrain_source == "sensed":
            self.elev_map = ElevationMap(B, dev, cfg.map_half_x, cfg.map_half_y, cfg.map_res, cfg.ceiling_exclude, cfg.foot_z_offset, flat_tol=cfg.map_flat_tol,
                                         ema_alpha=cfg.map_ema_alpha, upper_fill=cfg.upper_fill, upper_reach=cfg.upper_reach, stair_prior=cfg.stair_prior)
        self._age = torch.zeros(B, dtype=torch.long, device=dev)         # MPC ticks since the last reset
        self._fresh = torch.ones(B, dtype=torch.bool, device=dev)        # first tick after a reset: the sensors still show the old pose
        self._ep_t = torch.zeros(B, device=dev)
        self._substep = 0
        self.motor_gain = torch.ones(B, 1, device=dev)
        self._floor_last = torch.zeros(B, device=dev)
        self._q0 = self._robot.data.default_joint_pos.clone().float()
        self._link_ids = [i for i, n in enumerate(self._contact.body_names) if not n.endswith("foot")]
        self._foot_ids = self._contact.find_bodies([f"{l}_foot" for l in LEGS], preserve_order=True)[0]

        self.P = gs.PlannerParams(kd=cfg.raibert_kd, margin=cfg.foot_margin, edge_tol=cfg.edge_tol, dt=self.step_dt, clearance=cfg.swing_clearance,
                                  base_apex=cfg.swing_base_apex, H=cfg.mpc_horizon, swing_delay=cfg.swing_delay)
        mass = float(self.legs.total_mass[0])
        self.mpc_params = MpcParams(dt=self.step_dt, mass=mass, inertia_b=np.array(SPOT_INERTIA_B), mu=cfg.mu, fmax=cfg.fmax, horizon=cfg.mpc_horizon)
        self.mpc_params.q_diag = self.mpc_params.q_diag.copy()
        self.mpc_params.q_diag[[3, 4]] = cfg.q_pos_xy
        self.mpc = BatchedSrbdMpc(self.mpc_params, iters=cfg.mpc_iters, rho=cfg.mpc_rho, dtype=torch.float32, device=dev)
        self.tau_lim = np.array(TAU_LIM)
        self.gait = gs.BatchedGait(B, dev, freq=cfg.gait_freq, duty=cfg.gait_duty, offsets=cfg.gait_offsets,
                                   freq_range=(cfg.gait_freq - cfg.freq_range, cfg.gait_freq + cfg.freq_range))
        self.gov = gs.ReferenceGovernor(B, dev)
        self.swing = SwingPlanner(B, dev, delay=cfg.swing_delay)

        zeros = lambda *shape: torch.zeros(*shape, device=dev)
        self.h_ref = torch.full((B,), cfg.com_height, device=dev)
        self.u_prev0 = zeros(B, 4, 3)
        self.u_prev0[..., 2] = mass * GRAVITY / 4
        self.u_prev0 = self.u_prev0.reshape(B, 12)
        self.u_prev = self.u_prev0.clone()
        self.F_cmd = zeros(B, 4, 3)
        self.accel_resid, self.foot_res, self.apex_res = zeros(B, 6), zeros(B, 4, 2), zeros(B, 4)
        self.pitch_ref, self.vz_ref = zeros(B), zeros(B)
        self._steep, self._steep_down = zeros(B), zeros(B)          # previewed ascent / descent, 0..1
        self._wup, self._wback = zeros(B), zeros(B)                 # weights of the ascent and backward-descent operating points
        self._duty_blend = zeros(B)
        self.cmd, self.cmd_target = zeros(B, 3), zeros(B, 3)
        self.x0 = zeros(B, 13)
        self.xe = self.x0                # the state the controller works from (the simulator state)
        self._yaw_prev = zeros(B)
        self.state_nonfinite = torch.zeros(B, dtype=torch.bool, device=dev)
        self.grav_col = torch.full((B, 1), -GRAVITY, device=dev)
        self.nav = None                  # a PathFollower that drives cmd_target from a route
        self.nav_drives = True
        self.heading_target = None       # (B,) world heading to hold, or None
        self.lane_y = None               # (B,) world y of the lane to hold (NaN: no hold), lanes run along +x
        self.spawn_floor_z = torch.full((B,), float("nan"), device=dev)   # NaN: spawn on the surface under the spawn xy
        self.spawn_pose = torch.tensor([[-8.0, 0.0, 0.0]], device=dev).repeat(B, 1)
        self.reset()

    def _setup_scene(self):
        cfg = self.cfg
        if cfg.world_type == "warehouse":
            self.world = Warehouse(cfg.warehouse, "cpu")
        elif cfg.world_type == "generated":
            self.world = GeneratedWorld(cfg.generated, self.device, self.scene.cfg.num_envs, self.scene.cfg.env_spacing)
        else:
            self.world = World(cfg.world, "cpu")
        self.world.spawn()
        if getattr(cfg.warehouse, "visual", False) and hasattr(self.world, "dress"):
            self.world.dress()
        self._robot = Articulation(cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact = ContactSensor(cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact
        if cfg.world_type == "generated":           # true floor height: one ray straight down from above the base
            self._floor_ray = RayCaster(RayCasterCfg(prim_path="/World/envs/env_.*/Robot/body", update_period=PHYSICS_DT * DECIMATION,
                                                     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)), ray_alignment="world",
                                                     pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0)), max_distance=6.0,
                                                     mesh_prim_paths=[self.world.RAY_MESH_PRIM]))
            self.scene.sensors["floor_ray"] = self._floor_ray
        if cfg.terrain_source == "sensed":
            self._setup_sensors()
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=list(self.world.collision_roots))
        sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))

    def _setup_sensors(self):
        cfg = self.cfg
        mesh = [self.world.RAY_MESH_PRIM]
        self.world.spawn_ray_mesh()
        base = cfg.depth_camera.offset
        front = cfg.depth_camera.replace(mesh_prim_paths=mesh, offset=RayCasterCameraCfg.OffsetCfg(pos=base.pos, rot=_tilted(base.rot, cfg.depth_pitch_deg), convention="ros"))
        self._lidar = RayCaster(cfg.lidar.replace(mesh_prim_paths=mesh))
        self.scene.sensors["lidar"] = self._lidar
        self._depth = RayCasterCamera(front)
        self.scene.sensors["depth_camera"] = self._depth
        self._depth_rear = []
        if not cfg.rear_depth:
            return
        for k, pitch in enumerate(cfg.rear_depth_pitches):           # the front camera turned half a turn about z
            q = _tilted(base.rot, pitch)
            rear_rot = (-q[3], -q[2], q[1], q[0])
            offset = RayCasterCameraCfg.OffsetCfg(pos=(-base.pos[0], base.pos[1], base.pos[2]), rot=rear_rot, convention="ros")
            cam = RayCasterCamera(cfg.depth_camera.replace(mesh_prim_paths=mesh, offset=offset))
            self._depth_rear.append(cam)
            self.scene.sensors[f"depth_camera_rear{k}"] = cam

    def _read_state(self):
        com, vel = self.legs.com_state()
        roll, pitch, yaw = roll_pitch_yaw(self.legs.root_rot())
        # atan2 jumps by 2 pi at +-pi, which the MPC would read as a full-turn error: unwrap against the last reading
        yaw = self._yaw_prev + torch.remainder(yaw - self._yaw_prev + math.pi, 2 * math.pi) - math.pi
        self._yaw_prev = yaw
        raw = torch.cat([roll[:, None], pitch[:, None], yaw[:, None], com, self.legs.root_ang_vel_w(), vel, self.grav_col], 1)
        self.state_nonfinite = ~torch.isfinite(raw).all(1)
        return torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    def floor_height(self):
        """True floor height under each robot (a downward ray on generated terrain, the analytic surface otherwise)."""
        x = self.x0
        if self.cfg.world_type != "generated":
            return self.world.surface(x[:, 3], x[:, 4], z_ref=x[:, 5])
        h = self._floor_ray.data.ray_hits_w[:, 0, 2]
        h = torch.where(torch.isfinite(h) & (self._age >= 2), h, self._floor_last)
        h = torch.where(self._age >= 2, h, self.spawn_floor_z.nan_to_num(nan=0.0))     # the ray still shows the previous tile just after a reset
        self._floor_last = h
        return h

    def _late_feet(self, c_now):
        """(B,4) feet the clock has put in stance that carry no load yet (only while trotting)."""
        loaded = self._contact.data.net_forces_w[:, self._foot_ids].norm(dim=-1) > self.cfg.load_threshold
        return c_now & ~loaded & self.gait.trotting[:, None]

    def link_contact(self):
        """Summed contact force [N] on every link except the feet (a leg or the body hitting the terrain)."""
        return self._contact.data.net_forces_w[:, self._link_ids].norm(dim=-1).sum(-1)

    @property
    def terrain(self):
        return self.elev_map if self.elev_map is not None else self.world

    def _update_perception(self):
        if self.elev_map is None:
            return
        cfg, em = self.cfg, self.elev_map
        em.recenter(self.xe[:, 3:5])
        feet = self.legs.foot_pos_w()
        loaded = self._contact.data.net_forces_w[:, self._foot_ids].norm(dim=-1) > cfg.load_threshold
        foot_pts = torch.cat([feet[..., :2], feet[..., 2:3] - cfg.foot_z_offset], -1)       # a loaded foot writes the surface it stands on
        keep = ~self._fresh[:, None]

        def noisy(points, valid):
            if cfg.sensor_noise_std > 0.0:
                points = torch.where(valid[..., None], points + torch.randn_like(points) * cfg.sensor_noise_std, points)
            return points

        lp, lv = lidar_points(self._lidar)
        dp, dv = depth_points(self._depth)
        lp, dp = noisy(lp, lv), noisy(dp, dv)
        clouds = [(lp, lv & keep), (dp, dv & keep), (foot_pts, loaded & keep)]
        rears = []
        for cam in self._depth_rear:
            rp, rv = depth_points(cam)
            rp = noisy(rp, rv)
            rears.append((cam, rp, rv & keep))
            clouds.append((rp, rv & keep))
        em.update(clouds, self.xe[:, 5])
        if cfg.stair_prior:
            direction = torch.where(self.cmd[:, 0] < -0.02, -1.0, 1.0)
            yaw = self.xe[:, 2]
            em.update_stair_prior(self.xe[:, 3:5], torch.stack([torch.cos(yaw) * direction, torch.sin(yaw) * direction], 1))
        if cfg.upper_fill:
            em.update_upper(self._depth.data.pos_w, dp, dv & keep)
            for cam, rp, rv in rears:
                em.update_upper(cam.data.pos_w, rp, rv)
        self._fresh[:] = False
        n = loaded.float().sum(1)
        mean_z = (feet[..., 2] * loaded.float()).sum(1) / n.clamp(min=1.0)
        em.fallback = torch.where(n > 0, mean_z, em.fallback)

    def _shield(self, a):
        """Fade the residual out and slow the command as the body nears a tip-over or runs away. Returns the faded action."""
        cfg = self.cfg
        self.risk = torch.zeros(self.num_envs, device=self.device)
        if not (cfg.shield and hasattr(self, "x0")):
            return a
        x = self.xe
        tilt = torch.maximum(x[:, 0].abs(), (x[:, 1] - self.pitch_ref).abs())
        over = x[:, 9:11].norm(dim=-1) - self.cmd[:, :2].norm(dim=-1)
        risk_tilt = ((tilt - cfg.shield_tilt[0]) / (cfg.shield_tilt[1] - cfg.shield_tilt[0])).clamp(0.0, 1.0)
        risk_over = ((over - cfg.shield_over[0]) / (cfg.shield_over[1] - cfg.shield_over[0])).clamp(0.0, 1.0)
        self.risk = torch.maximum(risk_tilt, risk_over)
        return a * (1.0 - self.risk)[:, None]

    def _update_command(self, trot, meta):
        """Lane keeping or the path follower sets cmd_target; the speed governors and the command ramp turn it into cmd."""
        cfg = self.cfg
        xc = self.xe
        if self.heading_target is not None:
            err = torch.remainder(self.heading_target - xc[:, 2] + math.pi, 2 * math.pi) - math.pi
            self.cmd_target[:, 2] = (cfg.k_yaw * err).clamp(-cfg.wz_max, cfg.wz_max)
            if self.lane_y is not None:
                vy_w = (cfg.k_lane * (self.lane_y - xc[:, 4])).clamp(-cfg.vy_max, cfg.vy_max)
                vy_w = torch.where(torch.isnan(vy_w), torch.zeros_like(vy_w), vy_w)
                self.cmd_target[:, 1] = torch.cos(xc[:, 2]) * vy_w
        if self.nav is not None:
            nav_cmd = self.nav.command(xc[:, 3:5], xc[:, 2])
            if self.nav_drives:
                self.cmd_target = nav_cmd
        target = self.cmd_target * trot[:, None].float()
        fwd = target[:, 0].clone()
        mag = fwd.abs()
        moving = mag > 0.05
        target[:, 0] = torch.where(moving, torch.sign(fwd) * (mag + (cfg.stair_speed - mag) * self._steep), fwd)          # on a previewed ascent
        fwd = target[:, 0].clone()
        mag = fwd.abs()
        descent_speed = torch.where(self.cmd[:, 0] < -0.02, torch.full_like(mag, cfg.stair_speed_back), torch.full_like(mag, cfg.stair_speed_down))
        target[:, 0] = torch.where(mag > 0.05, torch.sign(fwd) * (mag + (descent_speed - mag) * self._steep_down), fwd)    # and on a descent
        target[:, :2] = target[:, :2] * ((1.0 + cfg.meta_speed * meta[:, 0]) * (1.0 - cfg.shield_slow * self.risk))[:, None]
        limit = cfg.cmd_accel * self.step_dt
        self.cmd = self.cmd + (target - self.cmd).clamp(-limit, limit)

    def _pre_physics_step(self, actions):
        cfg = self.cfg
        trot = self._ep_t >= cfg.t_start
        self.gait.set_trotting(trot)
        a = actions.clamp(-1.0, 1.0) * trot[:, None].float()
        a = self._shield(a)
        meta = a[:, ACT_META]

        # operating point of the planner, blended by last tick's previewed slope
        wu, wb = self._wup, self._wback
        self.P.clearance = blend(cfg.swing_clearance, wu, cfg.swing_clearance_up, wb, cfg.swing_clearance_back)
        self.P.base_apex = blend(cfg.swing_base_apex, wu, cfg.swing_base_apex_up, wb, cfg.swing_base_apex_back)
        self.P.margin = blend(cfg.foot_margin, wu, cfg.foot_margin_up, wb, cfg.foot_margin_back)
        freq = blend(cfg.gait_freq, wu, cfg.gait_freq_up, wb, cfg.gait_freq_back)
        self.gait.set_freq(_step(self.gait.freq, freq + cfg.freq_range * a[:, ACT_FREQ.start], 0.5))
        accel = torch.cat([cfg.accel_lin_scale * a[:, ACT_ACCEL][:, :3], cfg.accel_ang_scale * a[:, ACT_ACCEL][:, 3:]], 1)
        self.accel_resid = _step(self.accel_resid, accel, cfg.accel_rate)
        self.foot_res = _step(self.foot_res, cfg.foot_scale * a[:, ACT_FOOT].view(-1, 4, 2), cfg.foot_rate)
        self.apex_res = _step(self.apex_res, cfg.apex_scale * a[:, ACT_APEX], cfg.apex_rate)

        self.x0 = self._read_state()
        self.xe = self.x0
        self._update_command(trot, meta)
        self._age += 1
        self._update_perception()
        feet = self.legs.foot_pos_w()
        c_now = self.gait.contact()
        self.swing.observe(c_now, feet)
        terrain = self.terrain
        anchor = self.gov.update(self.xe, self.cmd)
        self.anchor = anchor
        x_ref, r, contact = gs.build_horizon(self.P, terrain, self.gait, self.xe, self.cmd, feet, anchor, self.h_ref, self.foot_res)
        if cfg.contact_aware:           # no force on a foot that has not landed yet: the others carry the body this tick
            contact = torch.cat([contact[:, :1] * (~self._late_feet(c_now))[:, None].to(contact.dtype), contact[:, 1:]], 1)

        # the previewed slope along the direction of travel picks the operating point for the next tick
        backward = (self.cmd[:, 0] < -0.02)
        travel = torch.where(backward, -1.0, 1.0)[:, None]
        up_p, dn_p, up, down = previewed_slopes(x_ref[:, :, 1], travel, cfg.steep_pitch)
        crouch_down = torch.where(backward, torch.full_like(dn_p, cfg.descent_crouch_back), torch.full_like(dn_p, cfg.descent_crouch))
        self.h_ref = cfg.com_height - cfg.slope_crouch * (up_p / cfg.slope_crouch_pitch).clamp(0.0, 1.0) - crouch_down * (dn_p / cfg.slope_crouch_pitch).clamp(0.0, 1.0)
        self._steep, self._steep_down = up, down
        self._wup = up * (~backward).float()
        self._wback = down * backward.float()
        self._duty_blend = self._duty_blend + min(1.0, self.step_dt / cfg.duty_tau) * (down - self._duty_blend)
        duty_down = torch.where(backward, torch.full_like(down, cfg.gait_duty_back), torch.full_like(down, cfg.gait_duty_down))
        duty = cfg.gait_duty + (duty_down - cfg.gait_duty) * self._duty_blend + self._wup * (cfg.gait_duty_up - cfg.gait_duty) + cfg.meta_duty * meta[:, 2]
        self.gait.set_duty(duty.clamp(0.5, 0.8))

        q_diag = torch.as_tensor(self.mpc_params.q_diag, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        q_down = torch.where(backward, torch.full_like(down, cfg.q_vel_xy_back), torch.full_like(down, cfg.q_vel_xy_down))
        q_vel = cfg.q_vel_xy + (cfg.q_vel_xy_steep - cfg.q_vel_xy) * up + (q_down - cfg.q_vel_xy) * down
        q_vel = q_vel * torch.exp(cfg.meta_q * meta[:, 1])
        q_diag[:, 9], q_diag[:, 10] = q_vel, q_vel
        q_pitch = torch.where(backward, torch.full_like(down, cfg.q_pitch_back), torch.full_like(down, cfg.q_pitch_down))
        q_diag[:, 1] = (q_diag[:, 1] + (q_pitch - q_diag[:, 1]) * down) * torch.exp(cfg.meta_q * meta[:, 3])

        jac = self.legs.leg_jacobians()
        f0, info = self.mpc.solve(self.xe, x_ref, r, contact, u_prev=self.u_prev, torque_map=-jac.transpose(-1, -2), tau_lim=self.tau_lim,
                                  q_diag=q_diag, accel_resid=self.accel_resid,
                                  normals=surface_normals(terrain.height, feet[..., 0], feet[..., 1]) if cfg.terrain_cone else None)
        self.F_cmd, self.u_prev = f0, f0.reshape(self.num_envs, 12)
        self.f_raw, self.mpc_resid = info["x"][:, :12].reshape(-1, 4, 3), info["resid"]
        self.pitch_ref = x_ref[:, 0, 1]
        self.vz_ref = x_ref[:, 0, 11]
        self.r_h0 = r[:, 0]
        self.swing.update(self.P, terrain, self.gait, self.xe, self.cmd, anchor, feet, c_now, foot_res=self.foot_res, apex_res=self.apex_res)
        self._substep = 0

    def _apply_action(self):
        cfg = self.cfg
        feet, vfoot, jac = self.legs.foot_pos_w(), self.legs.foot_vel_w(), self.legs.leg_jacobians()
        c_now = self.gait.contact()
        self.swing.observe(c_now, feet)
        pos, vel = self.swing.target(self.gait)
        wu, wb = self._wup, self._wback
        kp = blend(cfg.swing_kp, wu, cfg.swing_kp_up, wb, cfg.swing_kp_back)[:, None, None]
        kd = blend(cfg.swing_kd, wu, cfg.swing_kd_up, wb, cfg.swing_kd_back)[:, None, None]
        stance = c_now
        if cfg.contact_aware:           # a late foot keeps reaching down past its planned touchdown point instead of pushing on air
            late = self._late_feet(c_now)
            stance = c_now & ~late
            probe = self.swing.p_td - torch.tensor([0.0, 0.0, cfg.touchdown_probe], device=self.device)
            pos = torch.where(late[..., None], probe, pos)
            vel = torch.where(late[..., None], torch.zeros_like(vel), vel)
        f_swing = kp * (pos - feet) + kd * (vel - vfoot)
        tau_stance = -torch.einsum("nfij,nfi->nfj", jac, self.F_cmd)
        tau_swing = torch.einsum("nfij,nfi->nfj", jac, f_swing)
        tau_leg = torch.where(stance[..., None], tau_stance, tau_swing).reshape(self.num_envs, 12)
        alpha = ((self._ep_t + self._substep * self.physics_dt) / cfg.t_blend).clamp(0.0, 1.0)[:, None]
        tau = (1.0 - alpha) * joint_pd(self.legs, self._q0[:, self.legs.jids], cfg.pd_kp, cfg.pd_kd) + alpha * tau_leg
        tau = torch.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        self.tau_cmd = tau
        apply_torque(self._robot, self.legs, tau * self.motor_gain)
        self.gait.step(self.physics_dt)
        self._substep += 1
        if self._substep == DECIMATION:
            self._ep_t = self._ep_t + self.step_dt

    def _get_observations(self):
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        x = self.x0 = self._read_state()
        under = x[:, 5] < self.floor_height() + 0.2
        fell = (x[:, 0].abs() > self.cfg.fall_tilt) | (x[:, 1].abs() > self.cfg.fall_tilt) | under | ~torch.isfinite(x).all(1)
        return fell, self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if not hasattr(self, "legs"):
            return
        dev, n, cfg = self.device, len(env_ids), self.cfg
        x, y, yaw = self.spawn_pose[env_ids].unbind(-1)
        if cfg.world_type == "generated":
            floor = torch.nan_to_num(self.spawn_floor_z[env_ids], nan=0.0)
        else:
            floor = torch.where(torch.isnan(self.spawn_floor_z[env_ids]), self.world.surface(x, y), self.spawn_floor_z[env_ids])
        root = self._robot.data.default_root_state[env_ids].clone()
        root[:, 0], root[:, 1], root[:, 2] = x, y, floor + cfg.root_height
        root[:, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
        root[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(self._q0[env_ids], torch.zeros_like(self._q0[env_ids]), None, env_ids)
        self._yaw_prev[env_ids] = yaw
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        mask[env_ids] = True
        self.mpc.reset(mask)
        self.gait.reset(env_ids)
        self.gov.reset(env_ids)
        feet = torch.zeros(self.num_envs, 4, 3, device=dev)
        feet[env_ids] = torch.cat([torch.stack([x, y], -1)[:, None, :].expand(n, 4, 2), (floor + cfg.foot_z_offset)[:, None, None].expand(n, 4, 1)], -1)
        self.swing.reset(env_ids, feet)
        for t in (self.F_cmd, self.accel_resid, self.foot_res, self.apex_res, self.cmd, self.cmd_target):
            t[env_ids] = 0.0
        self.u_prev[env_ids] = self.u_prev0[env_ids]
        self.h_ref[env_ids] = cfg.com_height
        self._duty_blend[env_ids] = 0.0
        self._fresh[env_ids] = True
        self._age[env_ids] = 0
        if self.elev_map is not None:
            self.elev_map.reset(env_ids, torch.stack([x, y], -1), floor + cfg.foot_z_offset)
        self._ep_t[env_ids] = 0.0
        self._substep = 0
