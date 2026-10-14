import math

import torch

from isaaclab.utils import configclass

from ..control.torque_interface import randomize_base_mass, randomize_material
from ..navigation import PathFollower
from .task import SpotTask, SpotTaskCfg


@configclass
class GeneralTaskCfg(SpotTaskCfg):
    world_type = "generated"
    rear_depth = True                    # backward descent needs the rear camera
    terrain_source = "sensed"
    episode_length_s = 60.0              # the 15.5 m goal at the governed 0.45 m/s needs 40+ s
    gate_on_steps: bool = False          # the policy must work on every terrain: no hand-made gate
    sensor_noise_std: float = 0.01
    # commands
    speed_range: tuple = (0.35, 0.70)     # the goal is 15.5 m away: below ~0.3 m/s it is unreachable in time
    yaw_start_prob: float = 0.3          # fraction of episodes that start facing away from the goal
    yaw_start_range: float = 1.4         # +- rad
    goal_radius: float = 0.6
    # domain randomisation
    domain_rand: bool = True
    friction_range: tuple = (0.45, 1.25)
    added_mass_range: tuple = (-2.0, 4.0)
    motor_strength_range: tuple = (0.9, 1.1)
    push_interval_s: tuple = (4.0, 8.0)
    push_speed: float = 0.35
    # curriculum
    curriculum: bool = True
    progress_for_move_down: float = 0.5
    w_progress: float = 2.0              # per metre of reduced distance to the goal
    w_alive: float = 0.0
    back_descent: bool = True            # stairs_down tiles are travelled BACKWARD (the deployed way down)
    # safety costs with multiplicative Lagrange multipliers, which grow while a running cost exceeds its budget
    lag_init: tuple = (1.0, 1.0, 5.0, 1.0)         # unloaded-stance, tilt barrier, overspeed, link contact
    lag_budget: tuple = (0.0008, 0.003, 0.0007, 0.012)
    lag_eta: float = 5e-4
    lag_range: tuple = (0.3, 10.0)
    cbf_gamma: float = 0.2                         # CBF decrease condition on h = tilt_lim - tilt
    tilt_lim: float = 0.6
    unload_grace: float = 0.05


class GeneralTask(SpotTask):
    cfg: GeneralTaskCfg
    def _init_general(self):
        B, dev, cfg = self.num_envs, self.device, self.cfg
        self.importer = self.world.importer
        self.goal_xy = torch.zeros(B, 2, device=dev)
        self.start_dist = torch.ones(B, device=dev)
        self.prog = torch.zeros(B, device=dev)               # fraction of the start->goal distance covered so far
        self.dist_prev = torch.ones(B, device=dev)
        self.has_run = torch.zeros(B, dtype=torch.bool, device=dev)
        self.push_timer = torch.empty(B, device=dev).uniform_(*cfg.push_interval_s)
        self.eval_mode = False               # no DR, no pushes, no start-heading randomisation, fixed speed
        self.fixed_command = False           # fixed start heading and speed
        self.eval_speed = 0.45
        self.nav = PathFollower(B, torch.zeros(B, 1, 2, device=dev), dev, arrive=cfg.goal_radius, v_max=0.45)
        self._lam = torch.tensor(self.cfg.lag_init, device=dev)
        self._cost_ema = torch.zeros(4, device=dev)
        self._h_prev = torch.full((B,), self.cfg.tilt_lim, device=dev)
        self._unl_t = torch.zeros(B, 4, device=dev)
        self._dr_base_bid = self._robot.find_bodies(["body"])[0][0]
        self._dr_nominal_mass = self.legs.masses[:, self._dr_base_bid].clone()
        self._gen_ready = True

    def assign_tiles(self, levels: torch.Tensor, types: torch.Tensor):
        """Pin env i to tile (levels[i], types[i]) -- for evaluation. Takes effect at the next reset"""
        imp = self.importer
        imp.terrain_levels[:] = levels.to(self.device)
        imp.terrain_types[:] = types.to(self.device)
        imp.env_origins[:] = imp.terrain_origins[imp.terrain_levels, imp.terrain_types]
    def _pre_physics_step(self, actions):
        cfg = self.cfg
        if cfg.domain_rand and not self.eval_mode and hasattr(self, "push_timer"):
            self.push_timer -= self.step_dt
            due = torch.nonzero(self.push_timer <= 0.0).flatten()
            if len(due) > 0:
                vel = self._robot.data.root_vel_w[due].clone()
                vel[:, :2] += (torch.rand(len(due), 2, device=self.device) * 2 - 1) * cfg.push_speed
                self._robot.write_root_velocity_to_sim(vel, due)
                self.push_timer[due] = torch.empty(len(due), device=self.device).uniform_(*cfg.push_interval_s)
        super()._pre_physics_step(actions)
    def _get_dones(self):
        x0 = self.x0 = self._read_state()
        fell = ((x0[:, 0].abs() > self.cfg.fall_tilt) | (x0[:, 1].abs() > self.cfg.fall_tilt) | (x0[:, 5] < self.floor_height() + 0.2)
                | self.state_nonfinite)
        dist = (self.goal_xy - x0[:, 3:5]).norm(dim=-1)
        self.success = (dist < self.cfg.goal_radius) & ~fell & (self._ep_t > self.cfg.t_start)
        self.prog = (1.0 - dist / self.start_dist).clamp(-1.0, 1.0)
        timeout = (self.episode_length_buf >= self.max_episode_length - 1) | self.success
        return fell, timeout & ~fell

    def _get_rewards(self):
        cfg, dt, x0 = self.cfg, self.step_dt, self.x0
        R = self.legs.root_rot()
        v_b = (R.transpose(1, 2) @ x0[:, 9:12, None])[..., 0]
        trot = (self._ep_t >= cfg.t_start).float()
        dist = (self.goal_xy - x0[:, 3:5]).norm(dim=-1)
        progress = (self.dist_prev - dist) * trot
        self.dist_prev = dist
        track = torch.exp(-((v_b[:, :2] - self.cmd[:, :2]) ** 2).sum(1) / cfg.track_sigma2)
        steep = self._steep + self._steep_down
        sgn = torch.where(self.cmd[:, 0] < -0.02, -1.0, 1.0)
        slope_track = torch.exp(-((v_b[:, 0] - self.cmd[:, 0]) ** 2) / 0.01)
        vz_track = torch.exp(-((x0[:, 11] - self.vz_ref) ** 2) / cfg.vz_sigma2)
        # ---- safety costs ----
        tilt = torch.maximum(x0[:, 0].abs(), (x0[:, 1] - self.pitch_ref).abs())
        h = cfg.tilt_lim - tilt
        c_tilt = 10.0 * torch.relu((1.0 - cfg.cbf_gamma) * self._h_prev - h) + 5.0 * torch.relu(tilt - 0.35) ** 2       # CBF decrease violation + barrier
        self._h_prev = h
        sched = self.gait.contact()
        loaded = (self._contact.data.net_forces_w[:, self._foot_ids].norm(dim=-1) > cfg.load_threshold)
        self._unl_t = torch.where(sched & ~loaded, self._unl_t + dt, torch.zeros_like(self._unl_t))
        c_unl = ((self._unl_t > cfg.unload_grace).float().mean(1)) * (self.gait.trotting.float())
        c_over = torch.relu(v_b[:, 0] * sgn - self.cmd[:, :2].norm(dim=-1) - 0.05) ** 2
        c_link = (self.link_contact() / cfg.link_ref).clamp(0.0, cfg.link_cap)
        costs = torch.stack([c_unl, c_tilt, c_over, c_link], 1) * trot[:, None]
        with torch.no_grad():                                   # multiplicative dual ascent on the running mean cost
            self._cost_ema = 0.99 * self._cost_ema + 0.01 * costs.mean(0)
            bud = torch.tensor(cfg.lag_budget, device=self.device)
            self._lam = (self._lam * torch.exp(cfg.lag_eta * (self._cost_ema / bud - 1.0))).clamp(*cfg.lag_range)
        pen = (costs * self._lam[None]).sum(1)
        orient = x0[:, 0] ** 2 + (x0[:, 1] - self.pitch_ref) ** 2
        terms = {
            "progress": cfg.w_progress * progress,
            "track": cfg.w_track * track * trot * dt,
            "safety": -pen * dt,
            "slope_track": cfg.w_slope_track * steep * slope_track * trot * dt,
            "vz": cfg.w_vz * steep * vz_track * trot * dt,
            "orient": -cfg.w_orient * orient * dt,
            "action_rate": -cfg.w_action_rate * ((self.actions - self.prev_actions) ** 2).sum(1) * dt,
            "alive": cfg.w_alive * trot * dt,
            "timeout": -cfg.w_timeout * (self.reset_time_outs & ~self.success).float(),
            "fall": -cfg.w_fall * self.reset_terminated.float(),
            "success": cfg.w_success * self.success.float(),
        }
        for k, v in terms.items():
            self._ep_sums.setdefault(k, torch.zeros(self.num_envs, device=self.device))
            self._ep_sums[k] += v
        for i, nm in enumerate(("unl", "tilt", "over", "link")):
            self._ep_sums.setdefault("cost_" + nm, torch.zeros(self.num_envs, device=self.device))
            self._ep_sums["cost_" + nm] += costs[:, i] * dt
        return torch.nan_to_num(torch.stack(list(terms.values()), 0).sum(0), nan=0.0, posinf=0.0, neginf=0.0)
    def _reset_idx(self, env_ids):
        if not hasattr(self, "legs"):
            return super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        if not hasattr(self, "_gen_ready"):
            self._init_general()
        cfg, imp, dev, n = self.cfg, self.importer, self.device, len(env_ids)
        ran = self.has_run[env_ids]
        if cfg.curriculum and not self.eval_mode and bool(ran.any()):
            up = self.success[env_ids] & ran
            down = (self.reset_terminated[env_ids] | (self.prog[env_ids] < cfg.progress_for_move_down)) & ~up & ran
            imp.update_env_origins(env_ids, up, down)
        o = imp.env_origins[env_ids]
        fixed = self.eval_mode or self.fixed_command
        yaw = torch.zeros(n, device=dev)
        if not fixed:
            turn = torch.rand(n, device=dev) < cfg.yaw_start_prob
            yaw = torch.where(turn, (torch.rand(n, device=dev) * 2 - 1) * cfg.yaw_start_range, (torch.rand(n, device=dev) - 0.5) * 0.2)
        back = torch.zeros(n, dtype=torch.bool, device=dev)
        if cfg.back_descent:
            fams = self.world.families
            back = torch.tensor([fams[int(c)] == "stairs_down" for c in imp.terrain_types[env_ids].tolist()], device=dev)
        self.nav.back_env[env_ids] = back
        yaw = yaw + math.pi * back.float()
        self.spawn_pose[env_ids] = torch.stack([o[:, 0], o[:, 1], yaw], 1)
        self.spawn_floor_z[env_ids] = o[:, 2]
        goal_dx = self.world.goal_dx
        self.goal_xy[env_ids] = torch.stack([o[:, 0] + goal_dx, o[:, 1]], 1)
        self.start_dist[env_ids] = goal_dx
        self.dist_prev[env_ids] = goal_dx
        self.prog[env_ids] = 0.0
        self.nav.wp[env_ids, 0] = self.goal_xy[env_ids]
        self.nav.v_max[env_ids] = self.eval_speed if fixed else torch.empty(n, device=dev).uniform_(*cfg.speed_range)
        if cfg.domain_rand and not self.eval_mode:
            randomize_material(self._robot, env_ids, cfg.friction_range)
            randomize_base_mass(self._robot, self.legs, env_ids, self._dr_nominal_mass, cfg.added_mass_range, self._dr_base_bid)
            lo, hi = cfg.motor_strength_range
            self.motor_gain[env_ids] = lo + (hi - lo) * torch.rand(n, 1, device=dev)
        else:
            self.motor_gain[env_ids] = 1.0
        super()._reset_idx(env_ids)
        self.nav.reset(env_ids, self.spawn_pose[env_ids, :2], 0)
        self.has_run[env_ids] = True
        if "log" in self.extras:
            self.extras["log"].update({f"Lagrange/lambda_{nm}": float(self._lam[i]) for i, nm in enumerate(("unl", "tilt", "over", "link"))})
            self.extras["log"].update({f"Lagrange/cost_{nm}": float(self._cost_ema[i]) for i, nm in enumerate(("unl", "tilt", "over", "link"))})
            self.extras["log"]["Curriculum/level_mean"] = imp.terrain_levels.float().mean()
            self.extras["log"]["Curriculum/level_max"] = imp.terrain_levels.float().max()
