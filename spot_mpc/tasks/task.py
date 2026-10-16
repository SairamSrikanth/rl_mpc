import math

import torch

from isaaclab.utils import configclass

from ..control.gait import rot2
from ..control.torque_interface import GRAVITY
from ..env.cfg import NUM_ACTIONS, SpotMpcEnvCfg
from ..env.env import SpotMpcEnv

SCAN_X, SCAN_Y, SCAN_RES = (-0.6, 1.2), (-0.5, 0.5), 0.1
NUM_SCAN = (round((SCAN_X[1] - SCAN_X[0]) / SCAN_RES) + 1) * (round((SCAN_Y[1] - SCAN_Y[0]) / SCAN_RES) + 1)   # 209
NUM_POLICY = 3 + 3 + 3 + 3 + 12 + 12 + 4 + 4 + 4 + 4 + 1 + NUM_ACTIONS + 3 + NUM_SCAN       # ... + (pitch ref, ascent steepness, descent steepness)
NUM_CRITIC = NUM_POLICY + 12 + 12 + 1
STAIRS, RAMP, RUBBLE, FLAT, DESCENT, RAMP_DOWN = range(6)


@configclass
class SpotTaskCfg(SpotMpcEnvCfg):
    episode_length_s = 45.0
    observation_space = NUM_POLICY
    state_space = NUM_CRITIC
    lane_probs: tuple = (0.15, 0.05, 0.05, 0.05, 0.4, 0.3)   # stairs, ramp, rubble, flat, stair descent, ramp descent
    accel_lin_scale: float = 0.5
    accel_ang_scale: float = 1.0
    foot_scale: float = 0.02
    apex_scale: float = 0.02
    cmd_speed: float = 0.45
    flat_goal: float = 6.0
    # reward weights (per second unless noted)
    w_progress: float = 1.0          # forward distance plus height gained
    height_gain_weight: float = 3.0
    w_track: float = 1.0
    w_orient: float = 0.5
    w_action_rate: float = 0.02
    # terms aimed at the descent failures
    w_link: float = 0.5              # penalty for legs or body touching the terrain
    link_ref: float = 100.0
    link_cap: float = 5.0
    w_slope_track: float = 0.5       # extra tracking reward on steep slopes
    w_overspeed: float = 5.0         # penalty for speeding up on a descent
    w_vz: float = 0.5                # reward for the vertical speed the slope asks for
    vz_sigma2: float = 0.04
    gate_on_steps: bool = True       # the residual only acts where the scan shows a riser
    step_thresh: float = 0.10        # height jump between scan samples that counts as a riser
    w_alive: float = 0.0             # (0.2 let the policy stall: survive 45 s slowly instead of finishing)
    w_fall: float = 10.0
    w_timeout: float = 0.0           # a time limit is a truncation (bootstrapped by PPO), so it carries no penalty; a penalty would also need the time left in the observation
    w_success: float = 20.0
    track_sigma2: float = 0.04       # (0.25 paid ~45% of the tracking reward for standing still)


class SpotTask(SpotMpcEnv):
    cfg: SpotTaskCfg

    def __init__(self, cfg: SpotTaskCfg, render_mode=None, **kw):
        super().__init__(cfg, render_mode, **kw)
        B, dev = self.num_envs, self.device
        self.scan_offsets = self._scan_offsets()
        self._scan_ys = torch.arange(SCAN_Y[0], SCAN_Y[1] + 1e-6, SCAN_RES)
        self._gate = torch.ones(B, device=dev)
        self.lane = torch.zeros(B, dtype=torch.long, device=dev)
        self.spawn_xy = torch.zeros(B, 2, device=dev)
        self.spawn_yaw = torch.zeros(B, device=dev)
        self.prev_actions = torch.zeros(B, cfg.action_space, device=dev)
        self.actions = torch.zeros(B, cfg.action_space, device=dev)
        self._foot_ids = self._contact.find_bodies([f"{l}_foot" for l in ("fl", "fr", "hl", "hr")], preserve_order=True)[0]
        self.success = torch.zeros(B, dtype=torch.bool, device=dev)
        self.fell = torch.zeros(B, dtype=torch.bool, device=dev)
        self.n_nonfinite = 0
        self.phi_prev = torch.zeros(B, device=dev)
        self.spawn_z = torch.zeros(B, device=dev)
        self.hsign = torch.ones(B, device=dev)          # +1: reward height gained; -1 on the descent lane: reward height lost
        self.heading_target = None      # set by the spawn sampler; eval scripts set their own
        self.lane_y = None
        self.fixed_spawn = False     # eval scripts set True to keep their own spawn_pose / command
        self._ep_sums = {k: torch.zeros(B, device=dev) for k in ("progress", "track", "link", "slope_track", "overspeed", "vz", "orient", "action_rate", "alive", "timeout", "fall", "success")}
        self.reset()

    def _scan_offsets(self):
        xs = torch.arange(SCAN_X[0], SCAN_X[1] + 1e-6, SCAN_RES, device=self.device)
        ys = torch.arange(SCAN_Y[0], SCAN_Y[1] + 1e-6, SCAN_RES, device=self.device)
        return torch.stack(torch.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)

    def _scan(self):
        x0 = self.x0
        px, py = rot2(x0[:, 2:3], self.scan_offsets[:, 0], self.scan_offsets[:, 1])
        h = self.terrain.height(x0[:, 3:4] + px, x0[:, 4:5] + py)           # what the robot can perceive (the map when sensed)
        return (x0[:, 5:6] - h - self.cfg.com_height - self.cfg.foot_z_offset).clamp(-1.0, 1.0)
    def _steps_ahead(self):
        """(B,) float: 1 where the body-frame scan contains a riser (stairs), 0 on ramps, rough floor and flat ground"""
        s = self._scan().view(self.num_envs, -1, len(self._scan_ys))                         # (B, Nx, Ny), x-major like scan_offsets
        centre = (self._scan_ys.abs() <= 0.2 + 1e-6).to(s.device)                             # only the lane straight ahead: a ramp's SIDE drop is not a riser
        s = s[:, :, centre]
        return ((s[:, 1:] - s[:, :-1]).abs().amax((1, 2)) > self.cfg.step_thresh).float()

    def _pre_physics_step(self, actions):
        a = actions.clamp(-1.0, 1.0)
        self._gate = self._steps_ahead() if self.cfg.gate_on_steps else torch.ones(self.num_envs, device=self.device)
        a = a * self._gate[:, None]
        self.prev_actions, self.actions = self.actions, a
        super()._pre_physics_step(a)
    def _get_observations(self):
        if not hasattr(self, "lane"):
            return {"policy": torch.zeros(self.num_envs, NUM_POLICY, device=self.device),
                    "critic": torch.zeros(self.num_envs, NUM_CRITIC, device=self.device)}
        cfg, x0 = self.cfg, self._read_state()
        self.x0 = x0
        R = self.legs.root_rot()
        Rt = R.transpose(1, 2)
        lin_b = (Rt @ x0[:, 9:12, None])[..., 0]
        ang_b = (Rt @ x0[:, 6:9, None])[..., 0]
        grav_b = -Rt[:, :, 2]
        q_rel = self.legs.q() - self._q0[:, self.legs.jids]
        ph = 2 * math.pi * self.gait.phase
        freq_n = ((self.gait.freq - cfg.gait_freq) / cfg.freq_range)[:, None]
        forces = self._contact.data.net_forces_w[:, self._foot_ids]
        loaded = (forces.norm(dim=-1) > cfg.load_threshold).float()
        policy = torch.cat([lin_b, ang_b, grav_b, self.cmd, q_rel, self.legs.qd(), torch.sin(ph), torch.cos(ph),
                            self.gait.contact().float(), loaded, freq_n, self.actions,
                            torch.stack([self.pitch_ref, self._steep, self._steep_down], 1), self._scan()], 1)
        mg = self.mpc_params.mass * GRAVITY
        h_above = (x0[:, 5] - self.floor_height())[:, None]      # critic only (privileged): height above the true floor
        critic = torch.cat([policy, forces.reshape(self.num_envs, 12) / mg, self.F_cmd.reshape(self.num_envs, 12) / mg, h_above], 1)
        return {"policy": torch.nan_to_num(policy, nan=0.0, posinf=0.0, neginf=0.0), "critic": torch.nan_to_num(critic, nan=0.0, posinf=0.0, neginf=0.0)}
    def _is_success(self, x0):
        c = self.cfg.world
        ca = math.cos(math.radians(c.ramp_deg))
        x = x0[:, 3]
        stairs = x > c.stair_x + c.n_steps * c.run + 0.5
        descent = x > self.world.bottom_x + 0.2
        ramp_dn = x > self.world.ramp_bottom_x + 0.5
        ramp = x > c.stair_x + c.ramp_len * ca + 0.5
        rubble = x > c.stair_x + 24 * self.world.rubble_cell - 0.5
        walked = ((x0[:, 3:5] - self.spawn_xy) * torch.stack([torch.cos(self.spawn_yaw), torch.sin(self.spawn_yaw)], -1)).sum(-1) > self.cfg.flat_goal
        lane = self.lane
        return torch.where(lane == STAIRS, stairs, torch.where(lane == RAMP, ramp, torch.where(lane == RUBBLE, rubble, torch.where(lane == DESCENT, descent, torch.where(lane == RAMP_DOWN, ramp_dn, walked)))))

    def _get_dones(self):
        x0 = self.x0 = self._read_state()
        under = x0[:, 5] < self.floor_height() + 0.2
        fell = (x0[:, 0].abs() > self.cfg.fall_tilt) | (x0[:, 1].abs() > self.cfg.fall_tilt) | under
        self.n_nonfinite = self.n_nonfinite + int(self.state_nonfinite.sum())
        # eval scripts spawn anywhere and keep their own success logic
        self.success = (torch.zeros_like(fell) if self.fixed_spawn else self._is_success(x0) & ~fell & (self._ep_t > self.cfg.t_start))
        self.fell = fell
        # reaching the goal ends the task (terminated, no bootstrap); only running out of time is a truncation
        done = fell | self.success
        return done, (self.episode_length_buf >= self.max_episode_length - 1) & ~done

    def _get_rewards(self):
        cfg, dt = self.cfg, self.step_dt
        x0 = self.x0
        R = self.legs.root_rot()
        v_b = (R.transpose(1, 2) @ x0[:, 9:12, None])[..., 0]
        trot = (self._ep_t >= cfg.t_start).float()
        track = torch.exp(-((v_b[:, :2] - self.cmd[:, :2]) ** 2).sum(1) / cfg.track_sigma2)
        orient = x0[:, 0] ** 2 + (x0[:, 1] - self.pitch_ref) ** 2
        fell = self.fell
        steep = self._steep + self._steep_down                               # previewed slope steepness 0..2
        link = (self.link_contact() / cfg.link_ref).clamp(0.0, cfg.link_cap)
        slope_track = torch.exp(-((v_b[:, 0] - self.cmd[:, 0]) ** 2) / 0.01)
        over = torch.relu(v_b[:, 0] - self.cmd[:, 0] - 0.05) ** 2
        vz_track = torch.exp(-((x0[:, 11] - self.vz_ref) ** 2) / cfg.vz_sigma2)
        heading = torch.stack([torch.cos(self.spawn_yaw), torch.sin(self.spawn_yaw)], -1)
        phi = ((x0[:, 3:5] - self.spawn_xy) * heading).sum(-1) + cfg.height_gain_weight * self.hsign * (self.world.surface(x0[:, 3], x0[:, 4]) - self.spawn_z)
        progress = (phi - self.phi_prev) * trot
        self.phi_prev = phi
        terms = {
            "progress": cfg.w_progress * progress,
            "track": cfg.w_track * track * trot * dt,
            "link": -cfg.w_link * link * trot * dt,
            "slope_track": cfg.w_slope_track * steep * slope_track * trot * dt,
            "overspeed": -cfg.w_overspeed * over * self._steep_down * trot * dt,
            "vz": cfg.w_vz * steep * vz_track * trot * dt,
            "orient": -cfg.w_orient * orient * dt,
            "action_rate": -cfg.w_action_rate * ((self.actions - self.prev_actions) ** 2).sum(1) * dt,
            "alive": cfg.w_alive * trot * dt,
            "timeout": -cfg.w_timeout * self.reset_time_outs.float(),
            "fall": -cfg.w_fall * fell.float(),
            "success": cfg.w_success * self.success.float(),
        }
        for k, v in terms.items():
            self._ep_sums[k] += v
        return torch.nan_to_num(torch.stack(list(terms.values()), 0).sum(0), nan=0.0, posinf=0.0, neginf=0.0)
    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        if hasattr(self, "lane") and len(env_ids) > 0:
            log = {f"Episode_Reward/{k}": v[env_ids].mean() / self.max_episode_length_s for k, v in self._ep_sums.items()}
            log["Episode/fall_rate"] = self.fell[env_ids].float().mean()
            log["Episode/success_rate"] = self.success[env_ids].float().mean()
            log["Episode/nonfinite_resets_total"] = torch.tensor(float(self.n_nonfinite), device=self.device)
            for i, nm in enumerate(("stairs", "ramp", "rubble", "flat", "descent", "ramp_down")):
                m = self.lane[env_ids] == i
                if bool(m.any()):
                    log[f"Episode/success_{nm}"] = self.success[env_ids][m].float().mean()
            self.extras["log"] = log
            for v in self._ep_sums.values():
                v[env_ids] = 0.0
            if not self.fixed_spawn and hasattr(self.world, "descent_x"):       # (the warehouse sets its own spawns)
                self._sample_spawn(env_ids)
            self.actions[env_ids] = 0.0
            self.prev_actions[env_ids] = 0.0
            self.success[env_ids] = False
        super()._reset_idx(env_ids)               # zeroes cmd_target, so the command is set AFTER it
        if hasattr(self, "lane") and not self.fixed_spawn and hasattr(self.world, "descent_x"):
            self.cmd_target[env_ids, 0] = self.cfg.cmd_speed

    def _sample_spawn(self, env_ids):
        dev, n, c = self.device, len(env_ids), self.cfg.world
        U = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=dev)  # noqa: E731
        probs = torch.tensor(self.cfg.lane_probs, device=dev)
        lane = torch.multinomial(probs, n, replacement=True)
        run_up = U(1.0, 3.0)
        x = torch.where(lane == DESCENT, self.world.descent_x - run_up, torch.where(lane == RAMP_DOWN, self.world.ramp_down_x - run_up, c.stair_x - run_up))
        ylane = torch.stack([torch.full((n,), c.stair_y), torch.full((n,), c.ramp_y), torch.full((n,), c.rubble_y),
                             torch.zeros(n), torch.full((n,), c.stair_y), torch.full((n,), c.ramp_y)], 1).to(dev)
        y = ylane.gather(1, lane[:, None])[:, 0] + U(-0.3, 0.3)
        yaw = U(-0.15, 0.15)
        flat = lane == FLAT
        x = torch.where(flat, U(-12.0, -8.0), x)
        y = torch.where(flat, U(-3.0, 3.0), y)
        yaw = torch.where(flat, U(-math.pi, math.pi), yaw)
        self.lane[env_ids] = lane
        self.spawn_pose[env_ids] = torch.stack([x, y, yaw], 1)
        self.spawn_xy[env_ids] = torch.stack([x, y], 1)
        self.spawn_yaw[env_ids] = yaw
        self.spawn_z[env_ids] = self.world.surface(x, y)
        self.phi_prev[env_ids] = 0.0
        self.hsign[env_ids] = torch.where((lane == DESCENT) | (lane == RAMP_DOWN), -1.0, 1.0)
        # lane keeping: heading 0 and the lane line on a lane, the spawn heading on open floor
        if self.heading_target is None:
            self.heading_target = torch.zeros(self.num_envs, device=dev)
            self.lane_y = torch.zeros(self.num_envs, device=dev)
        self.heading_target[env_ids] = torch.where(flat, yaw, torch.zeros_like(yaw))
        self.lane_y[env_ids] = torch.where(flat, torch.full_like(y, float("nan")), ylane.gather(1, lane[:, None])[:, 0])
