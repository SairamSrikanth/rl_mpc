import math

import torch


class PathFollower:
    """Turns a waypoint route into body-frame velocity commands (vx, vy, wz).

    Heading follows the current segment, bent back toward the line by the cross-track error. The robot turns on the spot while the
    heading error is large and walks at v_max * cos(error) otherwise. waypoints is (W,2) for a shared route or (B,W,2) for one route
    per env; back[i] marks the segment into waypoint i as travelled backward (W,) or per env (B,W). n says how many of the W waypoints a robot's route really has.
    """

    def __init__(self, num_envs, waypoints, device, back=None, arrive=0.5, v_max=0.45, k_yaw=0.8, w_max=0.8, turn_only=math.pi / 4, k_cross=1.0, cross_max=0.5):
        wp = torch.as_tensor(waypoints, device=device, dtype=torch.float32)[..., :2]
        self.per_env = wp.ndim == 3
        self.wp = wp
        self.W = wp.shape[-2]
        self.n = torch.full((num_envs,), self.W, dtype=torch.long, device=device)
        self.idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.seg0 = torch.zeros(num_envs, 2, device=device)
        self.arrive, self.k_yaw, self.w_max, self.turn_only = arrive, k_yaw, w_max, turn_only
        self.k_cross, self.cross_max = k_cross, cross_max
        self.v_max = torch.as_tensor(v_max, device=device, dtype=torch.float32).expand(num_envs).clone()
        self.back_env = torch.zeros(num_envs, dtype=torch.bool, device=device)     # travel backward on every segment of that env
        self.back = torch.zeros(self.W, dtype=torch.bool, device=device) if back is None else torch.as_tensor(back, dtype=torch.bool, device=device)
        self.arange = torch.arange(num_envs, device=device)

    @property
    def complete(self):
        return self.idx >= self.n

    def reset(self, env_ids, start_xy, start_idx=0):
        self.idx[env_ids] = start_idx
        self.seg0[env_ids] = start_xy

    def _target(self):
        i = torch.minimum(self.idx, (self.n - 1).clamp(min=0))
        return self.wp[self.arange, i] if self.per_env else self.wp[i]

    def command(self, xy, yaw):
        """Body-frame command (B,3). Advances the waypoint index as a side effect."""
        target = self._target()
        reached = ((target - xy).norm(dim=-1) < self.arrive) & ~self.complete
        self.seg0 = torch.where(reached[:, None], target, self.seg0)
        self.idx = self.idx + reached.long()
        target = self._target()
        d = target - self.seg0
        u = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rel = xy - self.seg0
        cross = u[:, 0] * rel[:, 1] - u[:, 1] * rel[:, 0]               # > 0: the robot is left of the line
        i = self.idx.clamp(max=self.W - 1)
        backward = (self.back[i] if self.back.ndim == 1 else self.back[self.arange, i]) | self.back_env
        heading = torch.atan2(u[:, 1], u[:, 0]) + math.pi * backward.float() - (self.k_cross * cross).clamp(-self.cross_max, self.cross_max)
        err = torch.remainder(heading - yaw + math.pi, 2 * math.pi) - math.pi
        wz = (self.k_yaw * err).clamp(-self.w_max, self.w_max)
        vx = torch.where(err.abs() < self.turn_only, self.v_max * torch.cos(err) * (1.0 - 2.0 * backward.float()), torch.zeros_like(err))
        live = (~self.complete).float()
        return torch.stack([vx * live, torch.zeros_like(vx), wz * live], -1)
