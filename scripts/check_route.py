import argparse, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spot_mpc.worlds.warehouse_static import layout as L
from spot_mpc.worlds.warehouse_static.world import Warehouse

ap = argparse.ArgumentParser(description="Geometry-only check of the static warehouse loop: how close does it come to any prop at the robot's height?")
ap.add_argument("--dense", action="store_true")
ap.add_argument("--need", type=float, default=0.6, help="required clearance from the path centre [m]")
ap.add_argument("--extra_north", type=float, default=8.0)
args = ap.parse_args()

cfg = L.WarehouseCfg()
cfg.dense, cfg.extra_north = args.dense, args.extra_north
world = Warehouse(cfg)
route = L.LOOP_BACK_WAYPOINTS
structure = ("Stairs", "Ramp", "UpperFloor", "Shell")
solid = [p for p in world.prims if p.name.split("/")[0] not in structure]


def distance(p, x, y):
    if p.kind == "cylinder":
        return max(math.hypot(x - p.pos[0], y - p.pos[1]) - p.size[0], 0.0)
    yaw = 2 * math.atan2(p.quat[3], p.quat[0])
    dx, dy = x - p.pos[0], y - p.pos[1]
    lx, ly = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
    return math.hypot(max(abs(lx) - p.size[0] / 2, 0), max(abs(ly) - p.size[1] / 2, 0))


def z_span(p):
    half = p.size[2] / 2 if p.kind == "box" else p.size[1] / 2
    return p.pos[2] - half, p.pos[2] + half


level = 0.0


def walking_height(x, y):
    """Height of the floor under the route: the stair or ramp profile where on them, else the floor level it last left them on."""
    global level
    sx, sy = cfg.stair_base_xy
    z = None
    if cfg.atrium_x_min <= x <= sx and abs(y - sy) <= cfg.step_width / 2:
        z = min(cfg.num_steps, math.ceil((sx - x) / cfg.step_run - 1e-9)) * cfg.step_rise
    elif abs(x - cfg.ramp_bottom_xyz[0]) <= cfg.ramp_width / 2 and cfg.ramp_top_xyz[1] <= y <= cfg.ramp_bottom_xyz[1]:
        z = (cfg.ramp_bottom_xyz[1] - y) / (cfg.ramp_bottom_xyz[1] - cfg.ramp_top_xyz[1]) * cfg.ramp_top_xyz[2]
    if z is None:
        return level
    if z >= cfg.upper_floor_top_z - 0.01:
        level = cfg.upper_floor_top_z
    elif z <= 0.01:
        level = 0.0
    return z


points = [route[0]]
for (x0, y0), (x1, y1) in zip(route[:-1], route[1:]):
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 0.05))
    points += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(1, n + 1)]

closest, too_close = (9.9, None), {}
for x, y in points:
    z = walking_height(x, y)
    for p in solid:
        z0, z1 = z_span(p)
        if z1 < z + 0.10 or z0 > z + 0.60:
            continue
        d = distance(p, x, y)
        if d < closest[0]:
            closest = (d, (x, y, z, p.name))
        if d < args.need:
            too_close.setdefault(p.name, (round(x, 2), round(y, 2), round(z, 2), round(d, 2)))
print(f"[route] dense={args.dense}: {len(points)} samples, {len(solid)} solid props, closest approach {closest[0]:.2f} m at {closest[1]}")
print(f"[route] props closer than {args.need} m: {len(too_close)}")
for name, hit in list(too_close.items())[:15]:
    print("   ", name, hit)
