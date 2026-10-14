import torch

from ..control.footholds import safe_foothold


class ElevationMap:
    """Robot-centred height grid fused from LiDAR, depth and foot-contact points, one grid per environment.

    Heights are stored as surface + foot_z_offset, so the map doubles as the planner's terrain. Cells nobody has seen read
    'flat at my feet' (the mean loaded-foot height), lowered to the ray ceiling where a ray flew over them."""

    def __init__(self, num_envs, device, half_x=4.0, half_y=3.0, res=0.03, ceiling_exclude=1.2, foot_z_offset=0.035,
                 flat_tol=0.02, reach=0.08, ema_alpha=0.10, ema_gate=0.06, upper_fill=False, upper_reach=1.0, upper_step=0.05,
                 stair_prior=False, prior_reach=1.2):
        self.B, self.device, self.res = num_envs, device, res
        self.ceiling_exclude, self.foot_z_offset, self.flat_tol, self.reach = ceiling_exclude, foot_z_offset, flat_tol, reach
        self.ema_alpha, self.ema_gate = ema_alpha, ema_gate
        self.upper_fill, self.upper_reach, self.upper_step = upper_fill, upper_reach, upper_step
        self.stair_prior, self.prior_reach = stair_prior, prior_reach
        self._sp = None
        self.nx, self.ny = int(round(2 * half_x / res)) + 1, int(round(2 * half_y / res)) + 1
        self._ncell = self.nx * self.ny
        # one trailing sentinel cell absorbs rejected points, so no masking (and no host sync) is needed
        self._grid = torch.zeros(num_envs * self._ncell + 1, device=device)
        self._seen = torch.zeros(num_envs * self._ncell + 1, device=device, dtype=torch.bool)
        self._upper = torch.full((num_envs * self._ncell + 1,), torch.inf, device=device)
        self._views()
        self.half = torch.tensor([half_x, half_y], device=device)
        self.origin = torch.zeros(num_envs, 2, device=device) - self.half
        self.fallback = torch.zeros(num_envs, device=device)

    def _views(self):
        shape = (self.B, self.ny, self.nx)
        self.grid, self.seen, self.upper = self._grid[:-1].view(shape), self._seen[:-1].view(shape), self._upper[:-1].view(shape)

    def reset(self, env_ids, center_xy, fallback_z):
        self.origin[env_ids] = center_xy - self.half
        self.grid[env_ids] = 0.0
        self.seen[env_ids] = False
        self.upper[env_ids] = torch.inf
        self.fallback[env_ids] = fallback_z

    def recenter(self, xy, margin_frac=0.4):
        """Shift each window to follow the robot once it drifts too far from the centre; only the newly exposed strip becomes unseen."""
        drift = xy - (self.origin + self.half)
        need = (drift.abs() > margin_frac * self.half).any(-1)
        if not bool(need.any()):
            return
        for i in need.nonzero(as_tuple=True)[0].tolist():
            shift = torch.round(drift[i] / self.res).long()
            sx, sy = int(shift[0]), int(shift[1])
            if abs(sx) >= self.nx or abs(sy) >= self.ny:
                self.grid[i], self.seen[i], self.upper[i] = 0.0, False, torch.inf
            else:
                for layer, empty in ((self.grid, 0.0), (self.seen, False), (self.upper, torch.inf)):
                    rolled = torch.roll(layer[i], shifts=(-sy, -sx), dims=(0, 1))
                    if sy > 0:
                        rolled[-sy:, :] = empty
                    elif sy < 0:
                        rolled[:-sy, :] = empty
                    if sx > 0:
                        rolled[:, -sx:] = empty
                    elif sx < 0:
                        rolled[:, :-sx] = empty
                    layer[i] = rolled
            self.origin[i] = self.origin[i] + shift.to(self.origin.dtype) * self.res

    def _cell_index(self, pts):
        o = self.origin[:, None, :]
        ix = torch.floor((pts[..., 0] - o[..., 0]) / self.res + 0.5).long()
        iy = torch.floor((pts[..., 1] - o[..., 1]) / self.res + 0.5).long()
        return ix, iy, (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

    def update(self, clouds, robot_z):
        """clouds: list of (points (B,K,3), valid mask or None). Writes the highest point per cell, blended into the old value
        unless it differs by more than the gate (a real step overwrites at once)."""
        pts = torch.cat([p for p, _ in clouds], 1)
        valid = torch.cat([torch.ones(p.shape[:2], dtype=torch.bool, device=p.device) if v is None else v for p, v in clouds], 1)
        valid = valid & torch.isfinite(pts).all(-1)
        pts = torch.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
        valid = valid & (pts[..., 2] < robot_z[:, None] + self.ceiling_exclude)
        ix, iy, inside = self._cell_index(pts)
        valid = valid & inside
        sentinel = self.B * self._ncell
        flat = torch.arange(self.B, device=self.device)[:, None] * self._ncell + iy * self.nx + ix
        flat = torch.where(valid, flat, torch.full_like(flat, sentinel)).reshape(-1)
        z = (pts[..., 2] + self.foot_z_offset).reshape(-1)
        top = torch.full((self._grid.numel(),), -torch.inf, device=self.device).scatter_reduce_(0, flat, z, reduce="amax", include_self=True)
        hit = torch.isfinite(top)
        top = torch.where(hit, top, torch.zeros_like(top))
        hit[-1] = False
        close = self._seen & ((top - self._grid).abs() < self.ema_gate)
        blend = torch.where(close, self.ema_alpha * top + (1.0 - self.ema_alpha) * self._grid, top)
        self._grid = torch.where(hit, blend, self._grid)
        self._seen = self._seen | hit
        self._views()

    def update_upper(self, origin_w, pts, valid):
        """A ray that hits a point flew over every cell on the way, so the surface there is below the ray. Keep the lowest ray height per cell
        for the last upper_reach metres of every ray."""
        back = torch.arange(self.upper_step, self.upper_reach + 1e-6, self.upper_step, device=self.device)
        vec = pts - origin_w[:, None, :]
        length = vec.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        t = (1.0 - back[None, None, :] / length).clamp(min=0.0)
        q = origin_w[:, None, None, :] + vec[:, :, None, :] * t[..., None]
        ok = valid[..., None] & (back[None, None, :] < length) & torch.isfinite(q).all(-1)
        ix, iy, inside = self._cell_index_nd(q)
        ok = ok & inside
        flat = torch.arange(self.B, device=self.device)[:, None, None] * self._ncell + iy * self.nx + ix
        flat = torch.where(ok, flat, torch.full_like(flat, self.B * self._ncell)).reshape(-1)
        self._upper.scatter_reduce_(0, flat, (q[..., 2] + self.foot_z_offset).reshape(-1), reduce="amin", include_self=True)

    def _cell_index_nd(self, q):
        o = self.origin[:, None, None, :]
        ix = torch.floor((q[..., 0] - o[..., 0]) / self.res + 0.5).long()
        iy = torch.floor((q[..., 1] - o[..., 1]) / self.res + 0.5).long()
        return ix, iy, (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

    def _index(self, x, y):
        o = self.origin.reshape((self.B,) + (1,) * (x.ndim - 1) + (2,))
        ix = torch.floor((x - o[..., 0]) / self.res + 0.5).long().clamp(0, self.nx - 1)
        iy = torch.floor((y - o[..., 1]) / self.res + 0.5).long().clamp(0, self.ny - 1)
        return (iy * self.nx + ix).reshape(self.B, -1)

    def height(self, x, y):
        x, y = torch.broadcast_tensors(x, y)
        idx = self._index(x, y)
        h = torch.gather(self.grid.reshape(self.B, -1), 1, idx)
        seen = torch.gather(self.seen.reshape(self.B, -1), 1, idx)
        unseen_h = self.fallback[:, None].expand_as(h)
        ceiling = None
        if self.upper_fill:
            ceiling = torch.gather(self.upper.reshape(self.B, -1), 1, idx)
            unseen_h = torch.minimum(unseen_h, ceiling)
        if self.stair_prior and self._sp is not None:
            unseen_h = self._prior_height(x.reshape(self.B, -1), y.reshape(self.B, -1), unseen_h, ceiling)
        return torch.where(seen, h, unseen_h).reshape(x.shape)

    def seen_at(self, x, y):
        x, y = torch.broadcast_tensors(x, y)
        return torch.gather(self.seen.reshape(self.B, -1), 1, self._index(x, y)).reshape(x.shape)

    def update_stair_prior(self, xy, direction):
        """Fit the staircase the robot is walking down and predict the treads below the lip, which no sensor can see.

        Sample the map along the direction of travel: the contiguous seen run ends at the lip, and the last two descending risers give
        the rise (< 0) and the run. Unseen cells just beyond the lip then read h_last + rise * (1 + floor(distance / run))."""
        S = int(round((self.prior_reach + 1.5) / self.res)) + 1
        s = torch.arange(S, device=self.device) * self.res - 1.5
        px = xy[:, 0:1] + direction[:, 0:1] * s[None]
        py = xy[:, 1:2] + direction[:, 1:2] * s[None]
        idx = self._index(px, py)
        h = torch.gather(self.grid.reshape(self.B, -1), 1, idx)
        seen = torch.gather(self.seen.reshape(self.B, -1), 1, idx)
        i0 = int(round(1.5 / self.res))
        run_seen = torch.cumprod(seen[:, i0:].to(torch.int8), dim=1).bool()
        n_ok = run_seen.sum(1)
        last = (i0 + n_ok - 1).clamp(min=i0)
        h_last = torch.gather(h, 1, last[:, None])[:, 0]
        k = 2
        jump = torch.zeros_like(h)
        jump[:, k:-k] = h[:, 2 * k:] - h[:, :-2 * k]
        both_seen = torch.zeros_like(seen)
        both_seen[:, k:-k] = seen[:, 2 * k:] & seen[:, :-2 * k]
        local_min = jump <= torch.minimum(torch.roll(jump, 3, 1), torch.roll(jump, -3, 1))
        edge = (jump < -0.07) & both_seen & local_min & (torch.arange(S, device=self.device)[None] <= last[:, None])
        pos = torch.where(edge, s[None].expand_as(h), torch.full_like(h, -torch.inf))
        e1, i1 = pos.max(1)
        e2 = torch.where(pos < (e1[:, None] - 0.12), pos, torch.full_like(pos, -torch.inf)).amax(1)
        valid = torch.isfinite(e1) & torch.isfinite(e2) & (n_ok > 3)
        rise = torch.gather(jump, 1, i1[:, None])[:, 0].clamp(-0.30, -0.07)
        run = (e1 - e2).clamp(0.22, 0.40)
        self._sp = dict(xy=xy, dir=direction, s_last=s[last], h_last=h_last, rise=rise, run=run, valid=valid)

    def _prior_height(self, x, y, fallback, ceiling):
        sp = self._sp
        rel_x, rel_y = x - sp["xy"][:, 0:1], y - sp["xy"][:, 1:2]
        along = rel_x * sp["dir"][:, 0:1] + rel_y * sp["dir"][:, 1:2]
        lateral = (-rel_x * sp["dir"][:, 1:2] + rel_y * sp["dir"][:, 0:1]).abs()
        beyond = along - sp["s_last"][:, None]
        use = sp["valid"][:, None] & (beyond > 0.0) & (beyond < self.prior_reach) & (lateral < 0.7)
        pred = sp["h_last"][:, None] + sp["rise"][:, None] * (1.0 + torch.floor(beyond / sp["run"][:, None]))
        if ceiling is not None:
            pred = torch.minimum(pred, ceiling)
        return torch.where(use, pred, fallback)

    def project(self, xy, margin=0.08):
        return safe_foothold(self.height, xy, margin, self.reach, self.flat_tol)
