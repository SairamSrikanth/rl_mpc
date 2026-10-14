import math
from dataclasses import dataclass, field

import numpy as np

from ...common.prims import Prim, floats as _f, rot_xy as _rot_xy, yaw_quat as _yaw_quat

WAREHOUSE_PRIM = "/World/Warehouse"
GROUND_PRIM = "/World/ground"
RAY_MESH_PRIM = "/World/RayMesh"


@dataclass
class WarehouseCfg:
    footprint_x: float = 40.0
    footprint_y: float = 25.0
    extra_north: float = 8.0     # extra ground behind the ramp foot: the north wall sits at y = 12.5 + extra_north
    wall_height: float = 6.0
    wall_thickness: float = 0.3
    upper_floor_top_z: float = 3.0
    upper_floor_thickness: float = 0.25
    atrium_x_min: float = 5.0
    atrium_y_min: float = 2.5
    stair_base_xy: tuple = (9.8, 7.0)
    stair_ascend_dir: tuple = (-1.0, 0.0)
    num_steps: int = 16
    step_rise: float = 0.1875
    step_run: float = 0.30
    step_width: float = 2.0
    ramp_bottom_xyz: tuple = (15.0, 11.5, 0.0)
    ramp_top_xyz: tuple = (15.0, 2.5, 3.0)
    ramp_width: float = 1.8
    ramp_thickness: float = 0.05
    add_columns: bool = True
    column_size: float = 0.35
    column_grid_spacing: float = 8.0
    column_wall_inset: float = 1.0
    add_lighting: bool = True
    add_equipment: bool = True
    equipment_seed: int = 42
    friction: float = 1.0
    dense: bool = False          # fill both floors and the atrium with racks, pallets and clutter
    dense_seed: int = 7
    visual: bool = False         # dress the collision proxies with NVIDIA props (needs cameras)


@dataclass
class EquipmentCfg:
    ground_shelf_rows_y: list = field(default_factory=lambda: [-9.0, -2.5, 3.0, 9.0])
    ground_shelf_bays_x: list = field(default_factory=lambda: [-17.0, -13.0, -9.0, -5.0])
    upper_shelf_rows_y: list = field(default_factory=lambda: [-9.0, -5.0])
    upper_shelf_bays_x: list = field(default_factory=lambda: [8.0, 12.0, 16.0])
    ground_num_pallets: int = 12
    ground_num_box_piles: int = 5
    ground_num_barrel_clusters: int = 3
    ground_num_forklifts: int = 2
    upper_num_pallets: int = 5
    upper_num_box_piles: int = 2
    upper_num_barrel_clusters: int = 1
    upper_num_forklifts: int = 1
    bay_width: float = 2.0
    bay_depth: float = 1.0
    bay_height: float = 2.4
    bay_post_size: float = 0.08
    bay_deck_thickness: float = 0.05
    bay_shelf_heights: tuple = (0.5, 1.4, 2.3)
    pallet_size: tuple = (1.2, 1.0, 0.15)
    pallet_with_box_prob: float = 0.4
    pallet_box_size: float = 0.8
    barrel_radius: float = 0.30
    barrel_height: float = 0.9
    spot_spawn_xy: tuple = (-15.0, -8.0)
    spot_spawn_radius: float = 2.0
    stair_top_landing_xy: tuple = (5.5, 7.0)
    stair_top_landing_radius: float = 2.0
    ramp_top_landing_xy: tuple = (15.0, 3.5)
    ramp_top_landing_radius: float = 2.0
    stair_approach_xy: tuple = (12.0, 7.0)
    stair_approach_radius: float = 2.5


COLORS = dict(
    wall=(0.72, 0.70, 0.63), upper=(0.50, 0.52, 0.55), stair=(0.55, 0.55, 0.58), ramp=(0.55, 0.50, 0.42),
    column=(0.42, 0.42, 0.44), shelf_post=(0.30, 0.30, 0.35), shelf_deck=(0.60, 0.50, 0.35), pallet=(0.65, 0.50, 0.30),
    box=(0.72, 0.55, 0.35), barrel_blue=(0.15, 0.35, 0.65), barrel_red=(0.70, 0.15, 0.15),
    forklift_hi=(1.00, 0.75, 0.10), forklift_lo=(0.15, 0.15, 0.17),
)


def light_rows(cfg):
    ys, y = [-8.0, 0.0, 8.0], 16.0
    while y < cfg.footprint_y / 2 + cfg.extra_north - 2.0:
        ys.append(y)
        y += 8.0
    return tuple(ys)


def in_stair_footprint(x, y, cfg: WarehouseCfg, margin=0.5):
    x0, y0 = cfg.stair_base_xy
    dx, dy = cfg.stair_ascend_dir
    dn = math.hypot(dx, dy)
    dx, dy = dx / dn, dy / dn
    run = cfg.num_steps * cfg.step_run
    ex, ey = x0 + dx * run, y0 + dy * run
    hp = cfg.step_width / 2
    return (min(x0, ex) - hp * abs(dy) - margin <= x <= max(x0, ex) + hp * abs(dy) + margin
            and min(y0, ey) - hp * abs(dx) - margin <= y <= max(y0, ey) + hp * abs(dx) + margin)


def in_ramp_footprint(x, y, cfg: WarehouseCfg, margin=0.5):
    p0, p1 = cfg.ramp_bottom_xyz, cfg.ramp_top_xyz
    if abs(p1[0] - p0[0]) < 1e-6:
        return (p0[0] - cfg.ramp_width / 2 - margin <= x <= p0[0] + cfg.ramp_width / 2 + margin
                and min(p0[1], p1[1]) - margin <= y <= max(p0[1], p1[1]) + margin)
    return (min(p0[0], p1[0]) - margin <= x <= max(p0[0], p1[0]) + margin
            and p0[1] - cfg.ramp_width / 2 - margin <= y <= p0[1] + cfg.ramp_width / 2 + margin)


def structure_prims(cfg: WarehouseCfg) -> list[Prim]:
    fx, fy, wh, wt = cfg.footprint_x, cfg.footprint_y, cfg.wall_height, cfg.wall_thickness
    ex_ = cfg.extra_north                 # the north wall moves to y = fy/2 + ex_; the side walls and the floors grow with it
    fyl, yc = fy + ex_, ex_ / 2
    out = [
        Prim("Shell/WallNorth", "box", _f(0, fy / 2 + ex_ + wt / 2, wh / 2), _f(fx + 2 * wt, wt, wh), color=COLORS["wall"]),
        Prim("Shell/WallSouth", "box", _f(0, -fy / 2 - wt / 2, wh / 2), _f(fx + 2 * wt, wt, wh), color=COLORS["wall"]),
        Prim("Shell/WallEast", "box", _f(fx / 2 + wt / 2, yc, wh / 2), _f(wt, fyl, wh), color=COLORS["wall"]),
        Prim("Shell/WallWest", "box", _f(-fx / 2 - wt / 2, yc, wh / 2), _f(wt, fyl, wh), color=COLORS["wall"]),
    ]
    # upper floor: L-shape = two rectangles
    zc, th, ax0, ay0 = cfg.upper_floor_top_z - cfg.upper_floor_thickness / 2, cfg.upper_floor_thickness, cfg.atrium_x_min, cfg.atrium_y_min
    wa = ax0 + fx / 2
    out.append(Prim("UpperFloor/RectA", "box", _f(-fx / 2 + wa / 2, yc, zc), _f(wa, fyl, th), color=COLORS["upper"]))
    wb, hb = fx / 2 - ax0, ay0 + fy / 2
    out.append(Prim("UpperFloor/RectB", "box", _f(ax0 + wb / 2, -fy / 2 + hb / 2, zc), _f(wb, hb, th), color=COLORS["upper"]))
    # stairs: solid pillars (no overhangs), axis-aligned ascent
    dx, dy = cfg.stair_ascend_dir
    dn = math.hypot(dx, dy)
    dx, dy = dx / dn, dy / dn
    along_x = abs(abs(dx) - 1) < 1e-6
    if not (along_x or abs(abs(dy) - 1) < 1e-6):
        raise NotImplementedError("stair_ascend_dir must be axis-aligned")
    x0, y0 = cfg.stair_base_xy
    for i in range(cfg.num_steps):
        h = (i + 1) * cfg.step_rise
        a = cfg.step_run * (i + 0.5)
        size = (cfg.step_run, cfg.step_width, h) if along_x else (cfg.step_width, cfg.step_run, h)
        out.append(Prim(f"Stairs/Step_{i:02d}", "box", _f(x0 + dx * a, y0 + dy * a, h / 2), _f(*size), color=COLORS["stair"]))
    out.append(_ramp_prim(cfg))
    if cfg.add_columns:
        slab_bottom = cfg.upper_floor_top_z - cfg.upper_floor_thickness
        xs = np.arange(-fx / 2 + cfg.column_wall_inset, fx / 2 - cfg.column_wall_inset + 1e-6, cfg.column_grid_spacing)
        ys = np.arange(-fy / 2 + cfg.column_wall_inset, fy / 2 + ex_ - cfg.column_wall_inset + 1e-6, cfg.column_grid_spacing)
        k = 0
        for x in xs:
            for y in ys:
                if not (x <= ax0 or (x >= ax0 and y <= ay0)):
                    continue
                if in_stair_footprint(x, y, cfg) or in_ramp_footprint(x, y, cfg):
                    continue
                out.append(Prim(f"Columns/Col_{k:02d}", "box", _f(x, y, slab_bottom / 2), _f(cfg.column_size, cfg.column_size, slab_bottom),
                                color=COLORS["column"]))
                k += 1
    return out


def _ramp_prim(cfg: WarehouseCfg) -> Prim:
    """Inclined slab whose top surface passes through ramp_bottom_xyz and ramp_top_xyz (axis-aligned in xy)"""
    p0, p1 = np.array(cfg.ramp_bottom_xyz, float), np.array(cfg.ramp_top_xyz, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if abs(d[0]) > 1e-6 and abs(d[1]) > 1e-6:
        raise NotImplementedError("ramp must be axis-aligned in xy")
    if abs(d[1]) > 1e-6:  # long axis = y, rotate about +x
        pitch = math.atan2(d[2], abs(d[1])) * (1 if d[1] > 0 else -1)
        up = np.array([0.0, -math.sin(pitch), math.cos(pitch)])
        size, quat = (cfg.ramp_width, length, cfg.ramp_thickness), (math.cos(pitch / 2), math.sin(pitch / 2), 0.0, 0.0)
    else:                 # long axis = x, rotate about +y
        pitch = math.atan2(d[2], abs(d[0])) * (-1 if d[0] > 0 else 1)
        up = np.array([math.sin(pitch), 0.0, math.cos(pitch)])
        size, quat = (length, cfg.ramp_width, cfg.ramp_thickness), (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
    center = (p0 + p1) / 2 - 0.5 * cfg.ramp_thickness * up
    return Prim("Ramp/Slab", "box", _f(*center), _f(*size), _f(*quat), COLORS["ramp"])
def equipment_prims(wcfg: WarehouseCfg, ecfg: EquipmentCfg | None = None, seed: int | None = None) -> list[Prim]:
    """Shelving, pallets, box piles, barrels and forklifts on both floors; same draw order as the original scatterer"""
    ecfg = ecfg or EquipmentCfg()
    rng = np.random.default_rng(wcfg.equipment_seed if seed is None else seed)
    out: list[Prim] = []
    _populate(out, wcfg, ecfg, rng, upper=False)
    _populate(out, wcfg, ecfg, rng, upper=True)
    return out


def _populate(out, wcfg, ecfg, rng, upper):
    base = "Equipment/Upper" if upper else "Equipment/Ground"
    floor_z = wcfg.upper_floor_top_z if upper else 0.0
    fx, fy = wcfg.footprint_x, wcfg.footprint_y
    placed = []
    rows, bays = (ecfg.upper_shelf_rows_y, ecfg.upper_shelf_bays_x) if upper else (ecfg.ground_shelf_rows_y, ecfg.ground_shelf_bays_x)
    k = 0
    for ry in rows:
        for bx in bays:
            _shelving_bay(out, f"{base}/Shelf_{k:02d}", (bx, ry, floor_z), 0.0, ecfg)
            placed.append((bx, ry, 1.7))
            k += 1

    def valid(x, y, r):
        if upper:
            m = 1.0
            in_a = -fx / 2 + m <= x <= wcfg.atrium_x_min - m and -fy / 2 + m <= y <= fy / 2 - m
            in_b = wcfg.atrium_x_min + m <= x <= fx / 2 - m and -fy / 2 + m <= y <= wcfg.atrium_y_min - m
            if not (in_a or in_b):
                return False
            for (cx, cy), cr in ((ecfg.stair_top_landing_xy, ecfg.stair_top_landing_radius), (ecfg.ramp_top_landing_xy, ecfg.ramp_top_landing_radius)):
                if (x - cx) ** 2 + (y - cy) ** 2 < cr ** 2:
                    return False
        else:
            if not (-fx / 2 + 1.0 <= x <= fx / 2 - 1.0 and -fy / 2 + 1.0 <= y <= fy / 2 - 1.0):
                return False
            if in_stair_footprint(x, y, wcfg) or in_ramp_footprint(x, y, wcfg):
                return False
            for (cx, cy), cr in ((ecfg.spot_spawn_xy, ecfg.spot_spawn_radius), (ecfg.stair_approach_xy, ecfg.stair_approach_radius)):
                if (x - cx) ** 2 + (y - cy) ** 2 < cr ** 2:
                    return False
        return all((x - px) ** 2 + (y - py) ** 2 >= (pr + r) ** 2 for px, py, pr in placed)

    def sample(r):
        for _ in range(200):
            x, y = rng.uniform(-fx / 2 + 1.0, fx / 2 - 1.0), rng.uniform(-fy / 2 + 1.0, fy / 2 - 1.0)
            if valid(x, y, r):
                return x, y
        return None

    counts = ((ecfg.upper_num_pallets, ecfg.upper_num_box_piles, ecfg.upper_num_barrel_clusters, ecfg.upper_num_forklifts) if upper else
              (ecfg.ground_num_pallets, ecfg.ground_num_box_piles, ecfg.ground_num_barrel_clusters, ecfg.ground_num_forklifts))
    for i in range(counts[0]):
        if (p := sample(0.9)) is not None:
            yaw = float(rng.uniform(0, 2 * np.pi))
            _pallet(out, f"{base}/Pallet_{i:02d}", (*p, floor_z), yaw, ecfg, rng.random() < ecfg.pallet_with_box_prob)
            placed.append((*p, 0.9))
    for i in range(counts[1]):
        if (p := sample(0.8)) is not None:
            _box_pile(out, f"{base}/BoxPile_{i:02d}", p, floor_z, rng)
            placed.append((*p, 0.8))
    for i in range(counts[2]):
        if (p := sample(1.0)) is not None:
            _barrels(out, f"{base}/Barrels_{i:02d}", p, floor_z, rng, ecfg)
            placed.append((*p, 1.1))
    for i in range(counts[3]):
        if (p := sample(1.8)) is not None:
            _forklift(out, f"{base}/Forklift_{i:02d}", (*p, floor_z), float(rng.uniform(0, 2 * np.pi)))
            placed.append((*p, 1.8))


def _shelving_bay(out, name, pos, yaw, e: EquipmentCfg):
    cx, cy, cz = pos
    w, d, h, post = e.bay_width, e.bay_depth, e.bay_height, e.bay_post_size
    for i, (lx, ly) in enumerate([(-w / 2 + post / 2, -d / 2 + post / 2), (w / 2 - post / 2, -d / 2 + post / 2),
                                  (-w / 2 + post / 2, d / 2 - post / 2), (w / 2 - post / 2, d / 2 - post / 2)]):
        wx, wy = _rot_xy(lx, ly, yaw)
        out.append(Prim(f"{name}/Post_{i}", "box", _f(cx + wx, cy + wy, cz + h / 2), _f(post, post, h), color=COLORS["shelf_post"]))
    for j, sh in enumerate(e.bay_shelf_heights):
        out.append(Prim(f"{name}/Deck_{j}", "box", _f(cx, cy, cz + sh), _f(w, d, e.bay_deck_thickness), _yaw_quat(yaw), COLORS["shelf_deck"]))


def _pallet(out, name, pos, yaw, e: EquipmentCfg, with_box):
    (cx, cy, cz), (pw, pd, ph), q = pos, e.pallet_size, _yaw_quat(yaw)
    out.append(Prim(f"{name}/Base", "box", _f(cx, cy, cz + ph / 2), _f(pw, pd, ph), q, COLORS["pallet"]))
    if with_box:
        bs = e.pallet_box_size
        out.append(Prim(f"{name}/Box", "box", _f(cx, cy, cz + ph + bs / 2), _f(bs, bs, bs), q, COLORS["box"]))


def _box_pile(out, name, center, floor_z, rng):
    cx, cy = center
    placed = []
    for b in range(int(rng.integers(3, 7))):
        size = float(rng.uniform(0.35, 0.6))
        stack = int(rng.integers(0, len(placed))) if placed and rng.random() < 0.35 else None
        if stack is not None:
            sx, sy, top = placed[stack]
            bx, by, bz = sx + float(rng.uniform(-0.08, 0.08)), sy + float(rng.uniform(-0.08, 0.08)), top + size / 2
        else:
            ang, dist = float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0.0, 0.4))
            bx, by, bz = cx + dist * math.cos(ang), cy + dist * math.sin(ang), floor_z + size / 2
        yaw = float(rng.uniform(0, 2 * np.pi))
        color = tuple(np.clip(np.array(COLORS["box"]) + rng.uniform(-0.05, 0.05, size=3), 0, 1))
        out.append(Prim(f"{name}/Box_{b}", "box", _f(bx, by, bz), _f(size, size, size), _yaw_quat(yaw), _f(*color)))
        placed.append((bx, by, bz + size / 2))


def _barrels(out, name, center, floor_z, rng, e: EquipmentCfg):
    cx, cy = center
    r, h = e.barrel_radius, e.barrel_height
    for b in range(int(rng.integers(2, 4))):
        dx, dy = [(0.0, 0.0), (2.1 * r, 0.0), (1.05 * r, 1.85 * r)][b]
        color = COLORS["barrel_blue"] if rng.random() < 0.5 else COLORS["barrel_red"]
        out.append(Prim(f"{name}/Barrel_{b}", "cylinder", _f(cx + dx, cy + dy, floor_z + h / 2), _f(r, h, 0.0), color=color))


def _forklift(out, name, pos, yaw):
    cx, cy, cz = pos
    q = _yaw_quat(yaw)
    out.append(Prim(f"{name}/Body", "box", _f(cx, cy, cz + 0.5), _f(2.0, 1.2, 1.0), q, COLORS["forklift_hi"]))
    wx, wy = _rot_xy(-0.35, 0.0, yaw)
    out.append(Prim(f"{name}/Cab", "box", _f(cx + wx, cy + wy, cz + 1.0 + 0.45), _f(0.7, 1.0, 0.9), q, COLORS["forklift_lo"]))
    wx, wy = _rot_xy(1.0, 0.0, yaw)
    out.append(Prim(f"{name}/Mast", "box", _f(cx + wx, cy + wy, cz + 1.0), _f(0.15, 0.8, 2.0), q, COLORS["forklift_lo"]))
    for k, fyl in enumerate((-0.30, 0.30)):
        wx, wy = _rot_xy(1.0 + 0.55, fyl, yaw)
        out.append(Prim(f"{name}/Fork_{k}", "box", _f(cx + wx, cy + wy, cz + 0.12), _f(1.1, 0.13, 0.06), q, COLORS["forklift_lo"]))


def warehouse_prims(cfg: WarehouseCfg) -> list[Prim]:
    prims = structure_prims(cfg)
    if cfg.add_equipment:
        prims += equipment_prims(cfg)
        if cfg.dense:
            from .fill import fill_prims
            prims += fill_prims(cfg, prims)
    return prims


# Routes. All turns on the way to the stairs happen at x = 12, 2.2 m before the first riser.
STAIR_WAYPOINTS = [(12.0, 7.0), (9.5, 7.0), (8.6, 7.0), (7.4, 7.0), (6.2, 7.0), (4.5, 7.0), (0.0, 7.0)]
_cross_out = [(2.0, 1.3), (12.0, 1.3), (15.0, 1.3), (15.0, 2.5)]              # upper floor to the ramp top, turning on the floor and not on the ramp
_ground = [(15.0, 11.5), (11.0, 11.5), (15.0, 11.5)]                          # ramp foot, turn on the ground floor, back
_cross_back = [(15.0, 2.5), (15.0, 1.3), (12.0, 1.3), (2.0, 1.3), (0.0, 7.0)]
# Spot descends backward: walk to x = 3.5, turn around there (the footprint is +-0.45 m, so not at the edge) and reverse down.
_descend_back = [(3.5, 7.0), (6.2, 7.0), (7.4, 7.0), (8.6, 7.0), (9.5, 7.0), (12.0, 7.0)]
LOOP_BACK_WAYPOINTS = STAIR_WAYPOINTS + _cross_out + _ground + _cross_back + _descend_back
LOOP_BACK_MASK = [False] * (len(LOOP_BACK_WAYPOINTS) - len(_descend_back) + 1) + [True] * (len(_descend_back) - 1)     # segment into waypoint i is backward

SPAWN_POSES = [
    (12.0, 7.0, 0.0, math.pi), (12.5, 6.2, 0.0, math.pi), (12.5, 7.8, 0.0, math.pi), (11.5, 5.0, 0.0, math.pi / 2),
    (10.0, 5.0, 0.0, math.pi), (13.0, 9.5, 0.0, -math.pi / 2), (17.5, 6.0, 0.0, math.pi),
    (-1.0, 0.0, 0.0, 0.0), (-15.0, -8.0, 0.0, 0.0), (1.0, -6.0, 0.0, math.pi / 2),
    (3.0, 7.0, 3.0, math.pi), (-2.0, 7.0, 3.0, 0.0),
]
