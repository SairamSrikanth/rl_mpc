import torch

from ..prim_world import PrimWorld
from .layout import WAREHOUSE_PRIM, light_rows, warehouse_prims
from .rails import rail_prims

LIGHT_XS = (-16.0, -8.0, 0.0, 8.0, 16.0)


class Warehouse(PrimWorld):
    """The two-floor warehouse: a 16-step staircase and an 18 degree ramp between the floors."""

    skip_in_surface = ("Ramp/Slab",)       # the ramp is an analytic plane in surface()

    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)
        self.stairs_top = cfg.num_steps * cfg.step_rise
        self.rails = rail_prims(cfg) if cfg.rails else []

    def make_prims(self, cfg):
        return warehouse_prims(cfg)

    def extra_surface(self, xf, yf, zr, h, margin):
        cfg = self.cfg
        p0, p1 = cfg.ramp_bottom_xyz, cfg.ramp_top_xyz
        t = ((yf - p0[1]) / (p1[1] - p0[1])).clamp(0.0, 1.0)
        rz = p0[2] + t * (p1[2] - p0[2])
        inside = ((xf - p0[0]).abs() <= cfg.ramp_width / 2) & (yf >= min(p0[1], p1[1])) & (yf <= max(p0[1], p1[1])) & (rz <= zr + margin)
        return torch.maximum(h, torch.where(inside, rz, torch.zeros_like(rz)))

    def add_lights(self):
        import isaaclab.sim as sim_utils
        cfg = self.cfg
        if not cfg.add_lighting:
            return
        light = sim_utils.SphereLightCfg(intensity=30000.0, radius=0.3, color=(1.0, 0.95, 0.85))
        slab_bottom, roof, k = cfg.upper_floor_top_z - cfg.upper_floor_thickness, cfg.wall_height - 0.5, 0
        for x in LIGHT_XS:
            for y in light_rows(cfg):
                atrium = x > cfg.atrium_x_min + 0.5 and y > cfg.atrium_y_min + 0.5
                light.func(f"{WAREHOUSE_PRIM}/Lights/Ground_{k:02d}", light, translation=(x, y, roof if atrium else slab_bottom - 0.25))
                if not atrium:
                    light.func(f"{WAREHOUSE_PRIM}/Lights/Upper_{k:02d}", light, translation=(x, y, roof))
                k += 1

    def dress(self):
        from .visual import dress
        return dress(self)
