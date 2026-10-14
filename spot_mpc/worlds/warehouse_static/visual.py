import math

import numpy as np

from ...common.usd import BARRELS, BOXES, FORKLIFT, KIT, ROOT, YELLOW, UsdBuilder
from .layout import GROUND_PRIM, WAREHOUSE_PRIM, EquipmentCfg, light_rows

DRESS = "/World/Dressing"
MAT = ROOT + "/NVIDIA/Materials/Base/"
LIGHT_XS = (-16.0, -8.0, 0.0, 8.0, 16.0)
BLACK, GREY = (0.015, 0.015, 0.017), (0.16, 0.16, 0.17)


class Dresser(UsdBuilder):
    """Puts NVIDIA props and materials on top of the collision proxies. Colliders, the ray mesh and surface() are untouched:
    the proxies are only hidden, and everything added here is visual."""

    def __init__(self, wh):
        super().__init__(DRESS)
        self.wh, self.cfg = wh, wh.cfg
        self.rng = np.random.default_rng(self.cfg.dense_seed + 100)

    def bind_path(self, path, color, **kw):
        self.bind(self.stage.GetPrimAtPath(path), color, **kw)

    def bind_mdl(self, path, mdl, name):
        import isaaclab.sim as sim_utils
        mat = f"/World/Looks/{name}"
        if not self.stage.GetPrimAtPath(mat).IsValid():
            sim_utils.MdlFileCfg(mdl_path=mdl).func(mat, sim_utils.MdlFileCfg(mdl_path=mdl))
        sim_utils.bind_visual_material(path, mat)

    def shell(self):
        cfg = self.cfg
        fx, fy, wh = cfg.footprint_x, cfg.footprint_y, cfg.wall_height
        yl, yh = -fy / 2, fy / 2 + cfg.extra_north
        g = self.group("Shell")
        floor, ceil = KIT + "SM_floor02.usd", KIT + "SM_CeilingA_6X6.usd"

        def tile(url, x0, x1, y0, y1, z, flip=False):
            nx, ny = max(1, round((x1 - x0) / 6)), max(1, round((y1 - y0) / 6))
            sx, sy = (x1 - x0) / (6 * nx), (y1 - y0) / (6 * ny)
            for i in range(nx):
                for j in range(ny):
                    c = (x0 + (i + 0.5) * (x1 - x0) / nx, y0 + (j + 0.5) * (y1 - y0) / ny, z)
                    p = self.ref(g, url, c, 0.0, (sx, sy, 1.0), "t")
                    if url == floor:
                        self.bind(p, GREY, rough=0.7)
                    if flip:
                        self.UsdGeom.Xformable(p).AddRotateXOp().Set(180.0)

        ax, ay = cfg.atrium_x_min, cfg.atrium_y_min
        top = cfg.upper_floor_top_z + 0.003
        under = cfg.upper_floor_top_z - cfg.upper_floor_thickness - 0.003
        tile(floor, -fx / 2, fx / 2, yl, yh, 0.003)
        tile(floor, -fx / 2, ax, yl, yh, top)
        tile(floor, ax, fx / 2, -fy / 2, ay, top)
        tile(ceil, -fx / 2, ax, yl, yh, under, flip=True)
        tile(ceil, ax, fx / 2, -fy / 2, ay, under, flip=True)
        tile(ceil, -fx / 2, fx / 2, yl, yh, wh - 0.003, flip=True)

        wall = KIT + "SM_WallA_6M.usd"
        for z in (0.0, 3.0):
            for s in (-1, 1):
                for i in range(math.ceil(fx / 6)):
                    self.ref(g, wall, (-fx / 2 + 3.0 + i * 6.0, yh if s > 0 else yl, z), math.pi / 2 * (-s), (1, 1, 1), "wn")
                for i in range(math.ceil((yh - yl) / 6)):
                    self.ref(g, wall, (s * (fx / 2), -fy / 2 + 3.0 + i * 6.0, z), (0.0 if s < 0 else math.pi), (1, 1, 1), "we")

        mdl = {"UpperFloor": "Masonry/Concrete_Smooth.mdl", "Columns": "Masonry/Concrete_Formed.mdl", "Ramp": "Metals/Steel_Carbon.mdl"}
        for p in self.wh.prims:
            group = p.name.split("/")[0]
            path = f"{WAREHOUSE_PRIM}/{p.name}"
            if group == "Stairs":
                self.bind_path(path, BLACK, rough=0.55)
            if group in mdl:
                self.bind_mdl(path, MAT + mdl[group], group)
            if group == "Shell":
                self.hide(path)
        self.hide(GROUND_PRIM)

        for x in np.arange(-fx / 2 + 3, fx / 2, 6.0):
            self.ref(g, KIT + "SM_BeamA_9M.usd", (x, -fy / 2 + 1.5, wh - 0.4), math.pi / 2, (1, 1.4, 1), "beam")
        lamp = KIT + "SM_LampCeilingA_04.usd"
        for x in LIGHT_XS:
            for y in light_rows(cfg):
                atrium = x > ax + 0.5 and y > ay + 0.5
                self.ref(g, lamp, (x, y, wh if atrium else under), 0.0, (1, 1, 1), "lamp")
                if not atrium:
                    self.ref(g, lamp, (x, y, wh), 0.0, (1, 1, 1), "lamp")

    def stair_and_ramp_lights(self):
        import isaaclab.sim as sim_utils
        cfg = self.cfg
        light = sim_utils.SphereLightCfg(intensity=9000.0, radius=0.15, color=(1.0, 0.96, 0.88))
        x0, y0 = cfg.stair_base_xy
        length = cfg.step_run * cfg.num_steps
        for i in range(5):
            x = x0 - (i + 0.5) / 5 * length
            z = cfg.step_rise + cfg.step_rise / cfg.step_run * (x0 - x) + 1.5
            light.func(f"{DRESS}/TraverseLights/Stair_{i}", light, translation=(x, y0, z))
        p0, p1 = cfg.ramp_bottom_xyz, cfg.ramp_top_xyz
        for i in range(5):
            f = (i + 0.5) / 5
            light.func(f"{DRESS}/TraverseLights/Ramp_{i}", light,
                       translation=(p0[0], p0[1] + f * (p1[1] - p0[1]), p0[2] + f * (p1[2] - p0[2]) + 1.5))

    def equipment(self):
        g = self.group("Equipment")
        by = {}
        for p in self.wh.prims:
            parts = p.name.split("/")
            if parts[0] in ("Equipment", "Fill") and len(parts) == 4:
                by.setdefault("/".join(parts[:3]), []).append(p)
                self.hide(f"{WAREHOUSE_PRIM}/{p.name}")
        for key, ps in by.items():
            kind = key.split("/")[2].split("_")[0]
            if kind == "Shelf":
                self.rack(g, ps)
            elif kind == "Pallet":
                self.pallet(g, ps)
            elif kind == "BoxPile":
                for p in ps:
                    name, size = BOXES[int(self.rng.integers(0, 4))]
                    yaw = 2 * math.atan2(p.quat[3], p.quat[0])
                    s = p.size[0] / max(size)
                    self.ref(g, KIT + name + ".usd", (p.pos[0], p.pos[1], p.pos[2] - p.size[2] / 2), yaw, (s, s, p.size[2] / size[2] if size[2] < 0.3 else s), "box")
            elif kind == "Barrels":
                for p in ps:
                    name = BARRELS[int(self.rng.integers(0, len(BARRELS)))]
                    self.ref(g, KIT + name + ".usd", (p.pos[0], p.pos[1], p.pos[2] - p.size[1] / 2), float(self.rng.uniform(0, 6.28)), (1, 1, p.size[1] / 0.9), "barrel")
            elif kind == "Forklift":
                body = next(q for q in ps if q.name.endswith("Body"))
                yaw = 2 * math.atan2(body.quat[3], body.quat[0])
                self.ref(g, FORKLIFT, (body.pos[0], body.pos[1], body.pos[2] - 0.5), yaw - math.pi / 2, (1, 1, 1), "forklift")

    def rack(self, g, ps):
        e = EquipmentCfg()
        decks = sorted([p for p in ps if "Deck" in p.name], key=lambda p: p.pos[2])
        posts = [p for p in ps if "Post" in p.name]
        cx, cy = decks[0].pos[0], decks[0].pos[1]
        z0 = min(p.pos[2] - p.size[2] / 2 for p in posts)
        h = posts[0].size[2]
        for sx in (-1.0, 1.0):
            self.ref(g, KIT + "SM_RackFrame_03.usd", (cx + sx * 1.0, cy, z0), 0.0, (1, 1, h / 3.0), "frame")
        for deck in decks:
            self.ref(g, KIT + "SM_RackShelf_01.usd", (cx, cy, deck.pos[2] + 0.03), 0.0, (0.5, 1, 1), "shelf")
            for sx in (-0.62, 0.0, 0.62):
                if self.rng.random() < 0.18 or deck.pos[2] - z0 > h - 0.2:
                    continue
                name, size = BOXES[int(self.rng.integers(0, len(BOXES)))]
                s = min(1.0, 0.58 / size[0], 0.46 / size[1])
                if size[2] * s > 0.8:
                    s = 0.8 / size[2]
                pos = (cx + sx + float(self.rng.normal(0, 0.02)), cy + float(self.rng.normal(0, 0.03)), deck.pos[2] + e.bay_deck_thickness / 2)
                self.ref(g, KIT + name + ".usd", pos, float(self.rng.normal(0, 0.05)), (s, s, s), "item")

    def pallet(self, g, ps):
        base = next(p for p in ps if p.name.endswith("Base"))
        yaw = 2 * math.atan2(base.quat[3], base.quat[0])
        z0 = base.pos[2] - base.size[2] / 2
        self.ref(g, KIT + "SM_PaletteA_01.usd", (base.pos[0], base.pos[1], z0), yaw, (1, 1, 1), "pallet")
        if not any(p.name.endswith("Box") for p in ps):
            return
        name, size = BOXES[int(self.rng.choice([0, 1, 7]))]
        c, s = math.cos(yaw), math.sin(yaw)
        for layer in range(int(self.rng.integers(2, 4))):
            for dx in (-0.5 * size[1], 0.5 * size[1]):
                pos = (base.pos[0] - dx * s, base.pos[1] + dx * c, z0 + base.size[2] + layer * size[2])
                self.ref(g, KIT + name + ".usd", pos, yaw, (1, 1, 1), "load")

    def clutter(self):
        g = self.group("Details")
        cone = KIT + "S_TrafficCone.usd"
        for x, y, z in [(9.0, 3.2, 0.0), (11.5, 5.0, 0.0), (6.5, 9.0, 0.0), (3.0, 3.0, 3.0)]:
            self.ref(g, cone, (x, y, z), 0.0, (1, 1, 1), "cone")
        self.edge_rails(g)
        self.side_rails(g)

    def edge_rails(self, g):
        """Guard rail along the drop from the upper floor into the atrium, with openings at the ramp top and the stair top."""
        cfg = self.cfg
        z, inset = cfg.upper_floor_top_z, 0.12
        ex, ey = cfg.atrium_x_min - inset, cfg.atrium_y_min - inset
        x_end, y_end = cfg.footprint_x / 2 - 0.12, cfg.footprint_y / 2 + cfg.extra_north - 0.12
        rx = cfg.ramp_bottom_xyz[0]
        ramp_gap = (rx - cfg.ramp_width / 2 - 0.25, rx + cfg.ramp_width / 2 + 0.25)
        sy = cfg.stair_base_xy[1]
        stair_gap = (sy - cfg.step_width / 2 - 0.25, sy + cfg.step_width / 2 + 0.25)
        self.guard_rail(g, (ex, ey), (ramp_gap[0], ey), z)
        self.guard_rail(g, (ramp_gap[1], ey), (x_end, ey), z)
        self.guard_rail(g, (ex, ey), (ex, stair_gap[0]), z)
        self.guard_rail(g, (ex, stair_gap[1]), (ex, y_end), z)

    def side_rails(self, g):
        """Short yellow rails just outside the walking width of the stair and the ramp, following the slope."""
        cfg = self.cfg
        height, off, post, rail = 0.5, 0.06, 0.05, 0.045
        look = dict(rough=0.38, metal=0.15)

        def slanted(center, size, quat):
            self.box(g, center, size, YELLOW, quat, "side", **look)

        x0, y0 = cfg.stair_base_xy
        run, rise, n = cfg.step_run, cfg.step_rise, cfg.num_steps
        xa, xb = x0, x0 - run * n
        za, zb = rise + height, rise + rise / run * (x0 - xb) + height
        length, pitch = math.hypot(xa - xb, zb - za), math.atan2(zb - za, xa - xb)
        quat = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
        for side in (-1, 1):
            y = y0 + side * (cfg.step_width / 2 + off)
            mid = ((xa + xb) / 2, y, (za + zb) / 2)
            slanted(mid, (length, rail, rail), quat)
            slanted((mid[0], mid[1], mid[2] - height / 2), (length, 0.02, 0.09), quat)
            for i in range(0, n + 1, 3):
                x = x0 - run * i
                z = rise + rise / run * (x0 - x)
                self.box(g, (x, y, z + height / 2 - 0.07), (post, post, height + 0.14), YELLOW, name="post", **look)

        (rx, ry0, rz0), (_, ry1, rz1) = cfg.ramp_bottom_xyz, cfg.ramp_top_xyz
        length = math.hypot(ry0 - ry1, rz1 - rz0)
        angle = math.atan2(rz1 - rz0, ry1 - ry0)
        quat = (math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0)
        for side in (-1, 1):
            x = rx + side * (cfg.ramp_width / 2 + off)
            c = (x, (ry0 + ry1) / 2, (rz0 + rz1) / 2 + height)
            slanted(c, (rail, length, rail), quat)
            slanted((c[0], c[1], c[2] - height / 2), (0.02, length, 0.09), quat)
            for t in np.linspace(0.0, 1.0, max(2, math.ceil(length / 1.5) + 1)):
                y, z = ry0 + t * (ry1 - ry0), rz0 + t * (rz1 - rz0)
                self.box(g, (x, y, z + height / 2 - 0.07), (post, post, height + 0.14), YELLOW, name="post", **look)


def dress(wh):
    d = Dresser(wh)
    d.shell()
    d.equipment()
    d.clutter()
    d.stair_and_ramp_lights()
    n = d.strip_physics(uninstance=False)
    print(f"[visual] {d.n} references under {DRESS}, {n} physics schemas removed")
    return d
