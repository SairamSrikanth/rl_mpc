import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spot_mpc.worlds.warehouse_static import layout as L
from spot_mpc.worlds.warehouse_static.rails import LANDING, rail_prims


def _inside(p, x, y):
    """Plan-view point-in-box test for a yawed box (the slanted rails count by their footprint)."""
    yaw = 2 * math.atan2(p.quat[3], p.quat[0])
    dx, dy = x - p.pos[0], y - p.pos[1]
    lx, ly = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
    return abs(lx) <= p.size[0] / 2 and abs(ly) <= p.size[1] / 2


def test_rails_exist_and_are_in_the_world():
    cfg = L.WarehouseCfg()
    rails = rail_prims(cfg)
    assert len(rails) > 40 and all(p.name.startswith("Rails/") for p in rails)
    assert len({p.name for p in rails}) == len(rails)             # one collider prim per name


def test_rails_are_not_in_the_ray_mesh_or_surface_list():
    from spot_mpc.worlds.warehouse_static.layout import warehouse_prims
    cfg = L.WarehouseCfg()
    assert not any(p.name.startswith("Rails/") for p in warehouse_prims(cfg))


def test_the_routes_clear_every_rail():
    cfg = L.WarehouseCfg()
    rails = [p for p in rail_prims(cfg) if p.size[0] < 3.0 or p.size[1] < 3.0]
    pts = []
    route = L.LOOP_BACK_WAYPOINTS
    for (x0, y0), (x1, y1) in zip(route[:-1], route[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 0.1))
        pts += [(x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n) for t in range(n + 1)]
    for x, y in pts:
        for p in rails:
            # grow the box by the robot's half width and see whether the path point is inside
            q = type(p)(p.name, p.kind, p.pos, (p.size[0] + 0.6, p.size[1] + 0.6, p.size[2]), p.quat)
            assert not _inside(q, x, y), f"route point ({x:.2f}, {y:.2f}) is within 0.3 m of {p.name}"


def test_the_ramp_foot_is_left_open_for_the_route_that_turns_onto_it():
    cfg = L.WarehouseCfg()
    ry0, ry1 = cfg.ramp_bottom_xyz[1], cfg.ramp_top_xyz[1]
    side = [p for p in rail_prims(cfg) if abs(abs(p.pos[0] - cfg.ramp_bottom_xyz[0]) - (cfg.ramp_width / 2 + 0.06)) < 1e-6]
    assert side
    angle = math.atan2(cfg.ramp_top_xyz[2] - cfg.ramp_bottom_xyz[2], ry0 - ry1)
    for p in side:
        nearest = p.pos[1] + (p.size[1] / 2) * math.cos(angle) if p.size[1] > 1.0 else p.pos[1] + p.size[1] / 2        # the foot end of a bar or a post
        assert ry0 - nearest >= 0.9 * LANDING * math.cos(angle) - 1e-6, (p.name, nearest)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as ex:
            fails += 1
            print(f"  [FAIL] {name}: {ex}")
    sys.exit(1 if fails else 0)
