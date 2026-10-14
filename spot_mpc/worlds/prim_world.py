import math

import torch

from ..common.prims import build_ray_mesh, spawn_colliders
from ..control.footholds import safe_foothold
from .warehouse_static.layout import GROUND_PRIM, RAY_MESH_PRIM, WAREHOUSE_PRIM


class PrimWorld:
    """A building made of boxes and cylinders. The same primitive list gives the PhysX colliders, the ray-cast mesh and the surface()
    used for spawning and fall detection. There is no planner terrain here because the building has several floors: plan on the sensed map."""

    RAY_MESH_PRIM = RAY_MESH_PRIM
    collision_roots = [GROUND_PRIM, WAREHOUSE_PRIM]
    skip_in_surface = ()

    def __init__(self, cfg, device="cpu"):
        self.cfg, self.device = cfg, device
        self.prims = self.make_prims(cfg)
        boxes = [p for p in self.prims if p.kind == "box" and p.name not in self.skip_in_surface]
        cyls = [p for p in self.prims if p.kind == "cylinder"]
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        self._bc, self._bh = t([p.pos for p in boxes]), t([p.size for p in boxes]) / 2
        self._byaw = t([2 * math.atan2(p.quat[3], p.quat[0]) for p in boxes])
        self._cc, self._cr, self._ch = t([p.pos for p in cyls]), t([p.size[0] for p in cyls]), t([p.size[1] for p in cyls])

    def make_prims(self, cfg):
        raise NotImplementedError

    def to(self, device):
        for n in ("_bc", "_bh", "_byaw", "_cc", "_cr", "_ch"):
            setattr(self, n, getattr(self, n).to(device))
        self.device = device

    def surface(self, x, y, z_ref=None, margin=0.35):
        """Highest solid top at (x, y) that is below z_ref + margin (the floor the robot stands on), or 0 for the ground."""
        shape = x.shape
        xf, yf = x.reshape(-1), y.reshape(-1)
        zr = torch.full_like(xf, float("inf")) if z_ref is None else z_ref.expand(shape).reshape(-1)
        dx, dy = xf[:, None] - self._bc[None, :, 0], yf[:, None] - self._bc[None, :, 1]
        c, s = torch.cos(self._byaw), torch.sin(self._byaw)
        lx, ly = dx * c + dy * s, -dx * s + dy * c
        top = (self._bc[:, 2] + self._bh[:, 2])[None]
        ok = (lx.abs() <= self._bh[None, :, 0]) & (ly.abs() <= self._bh[None, :, 1]) & (top <= zr[:, None] + margin)
        h = torch.where(ok, top, torch.zeros_like(top)).amax(1)
        if len(self._cr):
            r2 = (xf[:, None] - self._cc[None, :, 0]) ** 2 + (yf[:, None] - self._cc[None, :, 1]) ** 2
            ctop = (self._cc[:, 2] + self._ch / 2)[None]
            okc = (r2 <= self._cr[None] ** 2) & (ctop <= zr[:, None] + margin)
            h = torch.maximum(h, torch.where(okc, ctop, torch.zeros_like(ctop)).amax(1))
        return self.extra_surface(xf, yf, zr, h, margin).reshape(shape)

    def extra_surface(self, xf, yf, zr, h, margin):
        return h

    def height(self, x, y):
        raise NotImplementedError("several floors: plan on the sensed map (terrain_source='sensed')")

    def project(self, xy, margin=0.08):
        return safe_foothold(self.height, xy, margin)

    def add_lights(self):
        pass

    def spawn(self):
        spawn_colliders(self.prims, WAREHOUSE_PRIM, self.cfg.friction)
        self.add_lights()

    def spawn_ray_mesh(self):
        build_ray_mesh(self.prims, self.RAY_MESH_PRIM)
