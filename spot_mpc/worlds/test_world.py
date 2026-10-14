import math
from dataclasses import dataclass

import torch

from ..common.prims import Prim, build_ray_mesh, spawn_colliders
from ..control.footholds import safe_foothold


@dataclass
class WorldCfg:
    n_steps: int = 16
    rise: float = 0.1875
    run: float = 0.30
    stair_width: float = 2.0
    stair_x: float = 6.0
    stair_y: float = 0.0
    landing: float = 4.0
    ramp_landing: float = 4.0
    ramp_deg: float = 18.4349
    ramp_len: float = 9.4868
    ramp_y: float = 6.0
    ramp_width: float = 1.8
    rubble_y: float = -6.0
    rubble_max: float = 0.06
    foot_z_offset: float = 0.035
    friction: float = 1.0


class World:
    """Flat floor with three lanes along +x: a 16-step staircase up and down (y = 0), a ramp up and down (y = 6) and a rubble field (y = -6)."""

    RAY_MESH_PRIM = "/World/RayMesh"
    collision_roots = ["/World/ground", "/World/terrain"]

    def __init__(self, cfg, device, seed=0):
        self.cfg, self.device = cfg, device
        g = torch.Generator().manual_seed(seed)
        self.rubble_cell = 0.25
        self.rubble = (torch.rand(24, 8, generator=g) * cfg.rubble_max).to(device)
        self.stairs_top = cfg.n_steps * cfg.rise
        cos = math.cos(math.radians(cfg.ramp_deg))
        self.ramp_down_x = cfg.stair_x + cfg.ramp_len * cos + cfg.ramp_landing
        self.ramp_bottom_x = self.ramp_down_x + cfg.ramp_len * cos
        self.descent_x = cfg.stair_x + cfg.n_steps * cfg.run + cfg.landing
        self.bottom_x = self.descent_x + cfg.n_steps * cfg.run
        self.ramp_top = math.sin(math.radians(cfg.ramp_deg)) * cfg.ramp_len

    def surface(self, x, y, z_ref=None):
        c = self.cfg
        in_y = (y - c.stair_y).abs() <= c.stair_width / 2
        k = torch.floor((x - c.stair_x) / c.run).clamp(min=-1, max=c.n_steps - 1) + 1.0
        kd = torch.floor((x - self.descent_x) / c.run).clamp(min=-1, max=c.n_steps - 1) + 1.0
        stairs = torch.where(in_y & (x >= c.stair_x), torch.minimum(k, c.n_steps - kd) * c.rise, torch.zeros_like(x))
        ca, sa = math.cos(math.radians(c.ramp_deg)), math.sin(math.radians(c.ramp_deg))
        s = ((x - c.stair_x) / ca).clamp(min=0.0, max=c.ramp_len)
        sd = ((x - self.ramp_down_x) / ca).clamp(min=0.0, max=c.ramp_len)
        ramp = torch.where((y - c.ramp_y).abs() <= c.ramp_width / 2, torch.minimum(s * sa, self.ramp_top - sd * sa), torch.zeros_like(x))
        ix = torch.floor((x - c.stair_x) / self.rubble_cell).long()
        iy = torch.floor((y - c.rubble_y) / self.rubble_cell).long() + 4
        ok = (ix >= 0) & (ix < 24) & (iy >= 0) & (iy < 8)
        rubble = torch.where(ok, self.rubble[ix.clamp(0, 23), iy.clamp(0, 7)], torch.zeros_like(x))
        return stairs + ramp + rubble

    def height(self, x, y):
        return self.surface(x, y) + self.cfg.foot_z_offset

    def project(self, xy, margin=0.08):
        return safe_foothold(self.height, xy, margin)

    def to(self, device):
        self.device = device
        self.rubble = self.rubble.to(device)

    def primitives(self):
        c = self.cfg
        out = []

        def add(name, size, pos, quat=(1.0, 0.0, 0.0, 0.0), color=(0.55, 0.55, 0.6)):
            out.append(Prim(name, "box", pos, size, quat, color))

        for i in range(c.n_steps):
            h = (i + 1) * c.rise
            add(f"step_{i}", (c.run, c.stair_width, h), (c.stair_x + (i + 0.5) * c.run, c.stair_y, h / 2), color=(0.7, 0.55, 0.4))
        add("stair_landing", (c.landing, c.stair_width, self.stairs_top),
            (c.stair_x + c.n_steps * c.run + c.landing / 2, c.stair_y, self.stairs_top / 2), color=(0.7, 0.55, 0.4))
        for j in range(c.n_steps - 1):
            h = (c.n_steps - 1 - j) * c.rise
            add(f"down_{j}", (c.run, c.stair_width, h), (self.descent_x + (j + 0.5) * c.run, c.stair_y, h / 2), color=(0.6, 0.5, 0.65))
        a = math.radians(c.ramp_deg)
        th, L = 0.3, c.ramp_len
        cx = c.stair_x + (L / 2) * math.cos(a) + (th / 2) * math.sin(a)
        cz = (L / 2) * math.sin(a) - (th / 2) * math.cos(a)
        add("ramp", (L, c.ramp_width, th), (cx, c.ramp_y, cz), (math.cos(-a / 2), 0.0, math.sin(-a / 2), 0.0), color=(0.45, 0.6, 0.5))
        top_x = c.stair_x + L * math.cos(a)
        add("ramp_landing", (c.ramp_landing, c.ramp_width, self.ramp_top), (top_x + c.ramp_landing / 2, c.ramp_y, self.ramp_top / 2), color=(0.45, 0.6, 0.5))
        xd = self.ramp_down_x
        cxd = xd + (L / 2) * math.cos(a) - (th / 2) * math.sin(a)
        czd = self.ramp_top - (L / 2) * math.sin(a) - (th / 2) * math.cos(a)
        add("ramp_down", (L, c.ramp_width, th), (cxd, c.ramp_y, czd), (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0), color=(0.45, 0.6, 0.5))
        rubble = self.rubble.cpu()
        for i in range(24):
            for j in range(8):
                h = float(rubble[i, j])
                if h > 1e-3:
                    add(f"rub_{i}_{j}", (self.rubble_cell, self.rubble_cell, h),
                        (c.stair_x + (i + 0.5) * self.rubble_cell, c.rubble_y + (j - 4 + 0.5) * self.rubble_cell, h / 2), color=(0.6, 0.5, 0.5))
        return out

    def spawn_ray_mesh(self):
        build_ray_mesh(self.primitives(), self.RAY_MESH_PRIM)

    def spawn(self):
        spawn_colliders(self.primitives(), "/World/terrain", self.cfg.friction)
