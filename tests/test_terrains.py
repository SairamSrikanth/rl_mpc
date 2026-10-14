"""Simulator-free tests of the procedural terrain generator (trimesh only)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import trimesh

from spot_mpc.worlds.generated import geometry as T

SIZE = (T.SX, T.SY)
FAMS = list(T.FAMILIES)


def tile(family, d, seed):
    np.random.seed(seed)
    return T.generate_tile(family, d, SIZE)


def top_z(meshes, x, y):
    """Highest top surface at (x, y): ray from above through every mesh (trimesh's pure-python ray tracer)."""
    best = -1e9
    for m in meshes:
        hits, _, _ = m.ray.intersects_location([[x, y, 50.0]], [[0.0, 0.0, -1.0]])
        if len(hits):
            best = max(best, float(hits[:, 2].max()))
    return best


def test_every_family_stays_in_bounds_and_is_nonempty():
    for fam in FAMS:
        for d in (0.0, 0.5, 1.0):
            for seed in range(5):
                ms, origin = tile(fam, d, seed)
                assert len(ms) >= 1 and all(isinstance(m, trimesh.Trimesh) for m in ms)
                allm = trimesh.util.concatenate(ms)
                lo, hi = allm.bounds
                assert lo[0] >= -1e-6 and hi[0] <= T.SX + 1e-6, (fam, d, seed, lo, hi)
                assert lo[1] >= -1e-6 and hi[1] <= T.SY + 1e-6, (fam, d, seed, lo, hi)
                assert origin.shape == (3,) and abs(origin[0] - T.X_SPAWN) < 1e-9 and abs(origin[1] - T.SY / 2) < 1e-9


def test_spawn_point_stands_on_the_ground_the_origin_says():
    for fam in FAMS:
        for seed in range(4):
            ms, origin = tile(fam, 0.8, seed)
            z = top_z(ms, origin[0], origin[1])
            assert abs(z - origin[2]) < 0.02, (fam, seed, z, origin[2])


def test_descents_start_on_a_raised_platform_and_ascents_on_the_floor():
    for seed in range(5):
        assert tile("stairs_down", 0.7, seed)[1][2] > 0.2
        assert tile("ramp_down", 0.7, seed)[1][2] > 0.2
        assert tile("stairs_up", 0.7, seed)[1][2] == 0.0
        assert tile("ramp_up", 0.7, seed)[1][2] == 0.0


def test_the_goal_line_has_ground_to_stand_on():
    for fam in FAMS:
        for seed in range(4):
            ms, origin = tile(fam, 1.0, seed)
            z = top_z(ms, T.SX - T.GOAL_MARGIN, origin[1])
            assert z > -1e-3, (fam, seed)                                          # something is there: floor or platform


def test_stairs_geometry_matches_the_parameter_ranges():
    for seed in range(8):
        ms, origin = tile("stairs_up", 1.0, seed)
        x, step_tops = T.X_FEAT + 0.05, []
        while x < T.SX - 2.5:
            step_tops.append(round(top_z(ms, x, origin[1]), 4))
            x += 0.02
        levels = sorted(set(step_tops))
        rises = np.diff([0.0] + levels)
        assert 3 <= len(levels) <= 18, len(levels)
        assert rises.min() > 0.05 and rises.max() < 0.23, rises                      # risers within the generator's range
    # difficulty scales the height: the mean total rise at d = 1 beats d = 0
    tot = lambda d: np.mean([tile("stairs_up", d, s)[0] and max(top_z(tile("stairs_up", d, s)[0], T.SX - 2.0, T.SY / 2), 0) for s in range(6)])  # noqa: E731
    assert tot(1.0) > tot(0.0) + 0.3


def test_ramp_slope_is_inside_the_configured_range_and_tiles_are_random():
    for d, (lo, hi) in ((0.0, (3.0, 11.0)), (1.0, (9.0, 26.0))):
        for seed in range(6):
            ms, origin = tile("ramp_up", d, seed)
            xs = np.arange(T.X_FEAT + 0.3, T.X_FEAT + 1.0, 0.1)
            z = np.array([top_z(ms, x, origin[1]) for x in xs])
            ang = math.degrees(math.atan(np.polyfit(xs, z, 1)[0]))
            assert lo <= ang <= hi, (d, seed, ang)
    a, b = tile("stairs_up", 0.6, 1)[0], tile("stairs_up", 0.6, 2)[0]
    assert trimesh.util.concatenate(a).bounds.tolist() != trimesh.util.concatenate(b).bounds.tolist()      # different seeds, different tiles


def test_column_families_follow_the_proportions():
    c = T.GeneratedWorldCfg(num_cols=100)
    fam = T.column_families(c)
    assert len(fam) == 100 and set(fam) == set(T.DEFAULT_PROPORTIONS)
    for name, p in T.DEFAULT_PROPORTIONS.items():
        assert abs(fam.count(name) / 100 - p) <= 0.02, name


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as ex:
            fails += 1
            print(f"  [FAIL] {name}: {type(ex).__name__} {ex}")
    sys.exit(1 if fails else 0)
