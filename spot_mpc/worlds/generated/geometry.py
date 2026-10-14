# Procedural terrain: ten families of randomised tiles on one difficulty curriculum (numpy + trimesh, no simulator needed).
# A tile is 18 x 6 m; the robot spawns at local x = 1, the feature starts at x = 3 and the goal line is at sx - 1.5.
import math
from dataclasses import dataclass

import numpy as np
import trimesh

SX, SY = 18.0, 6.0          # default tile
X_SPAWN, X_FEAT = 1.0, 3.0  # local x of the spawn point and of the start of the feature
GOAL_MARGIN = 1.5           # the goal line is sx - GOAL_MARGIN
U = np.random.uniform


def _lerp(a, b, d):
    return a + (b - a) * d


def _box(size, center, rot_y=0.0):
    T = np.eye(4)
    c, s = math.cos(rot_y), math.sin(rot_y)
    T[:3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    T[:3, 3] = center
    return trimesh.creation.box(extents=tuple(float(v) for v in size), transform=T)


def _column(x0, x1, y_c, width, top):
    return _box((x1 - x0, width, top), ((x0 + x1) / 2, y_c, top / 2))


def _ground(sx, sy):
    return _box((sx, sy, 0.2), (sx / 2, sy / 2, -0.1))                    # top face at z = 0


def _staircase(x0, rises, runs, width, y_c, z0, up):
    ms, x, z = [], x0, z0
    for r, ru in zip(rises, runs):
        z = z + r if up else max(z - r, 0.0)
        if z > 1e-3:
            ms.append(_column(x, x + ru, y_c, width, z))
        x += ru
    return ms, x, z


def _ramp(x0, Lh, angle, width, y_c, z0, up, th=0.3):
    Ls, sgn = Lh / math.cos(angle), (1.0 if up else -1.0)
    t = np.array([math.cos(angle), 0.0, sgn * math.sin(angle)])
    n = np.array([-sgn * math.sin(angle), 0.0, math.cos(angle)])              # upward normal
    center = np.array([x0, y_c, z0]) + t * (Ls / 2) - n * (th / 2)
    return _box((Ls, width, th), center, rot_y=-sgn * angle), max(z0 + sgn * Lh * math.tan(angle), 0.0)


def _stair_params(d, sx, extra_len=0.0):
    rise_hi = _lerp(0.10, 0.21, d)
    rise = U(_lerp(0.07, 0.15, d), rise_hi)
    run = U(0.26, 0.34)
    n_max = int(_lerp(4, 16, d))
    n = int(np.random.randint(max(3, n_max - 4), n_max + 1))
    n = min(n, int((sx - X_FEAT - 2.0 - extra_len) / run))
    return rise, run, max(n, 2), U(1.3, 3.0)


def stairs_up(d, sx, sy):
    rise, run, n, w = _stair_params(d, sx)
    ms = [_ground(sx, sy)]
    st, x, z = _staircase(X_FEAT, [rise] * n, [run] * n, w, sy / 2, 0.0, True)
    ms += st + [_column(x, sx, sy / 2, w, z)]
    return ms, 0.0


def stairs_down(d, sx, sy):
    rise, run, n, w = _stair_params(d, sx)
    H = n * rise
    ms = [_ground(sx, sy), _column(0.0, X_FEAT, sy / 2, w, H)]
    st, _, _ = _staircase(X_FEAT, [rise] * n, [run] * n, w, sy / 2, H, False)
    return ms + st, H


def stairs_updown(d, sx, sy):
    landing = U(1.5, 4.0)
    rise, run, n, w = _stair_params(d, sx, extra_len=landing)
    ms = [_ground(sx, sy)]
    n = min(n, int((sx - X_FEAT - landing - 2.0) / (2 * run)))
    n = max(n, 2)
    up, x, z = _staircase(X_FEAT, [rise] * n, [run] * n, w, sy / 2, 0.0, True)
    ms += up + [_column(x, x + landing, sy / 2, w, z)]
    dn, _, _ = _staircase(x + landing, [rise] * n, [run] * n, w, sy / 2, z, False)
    return ms + dn, 0.0


def irregular_stairs(d, sx, sy):
    rise, run, n, w = _stair_params(d, sx)
    rises = [rise * U(0.75, 1.25) for _ in range(n)]
    runs = [run * U(0.85, 1.15) for _ in range(n)]
    up = np.random.rand() < 0.5
    H = sum(rises)
    ms = [_ground(sx, sy)]
    if up:
        st, x, z = _staircase(X_FEAT, rises, runs, w, sy / 2, 0.0, True)
        return ms + st + [_column(x, sx, sy / 2, w, z)], 0.0
    ms.append(_column(0.0, X_FEAT, sy / 2, w, H))
    st, _, _ = _staircase(X_FEAT, rises, runs, w, sy / 2, H, False)
    return ms + st, H


def _ramp_params(d, sx, extra_len=0.0):
    angle = math.radians(U(_lerp(4.0, 10.0, d), _lerp(10.0, 25.0, d)))
    Lh_max = min(8.0, sx - X_FEAT - 2.0 - extra_len)
    H = min(U(0.4, _lerp(1.0, 2.8, d)), Lh_max * math.tan(angle))
    return angle, H / math.tan(angle), H, U(1.2, 3.0)


def ramp_up(d, sx, sy):
    a, Lh, H, w = _ramp_params(d, sx)
    r, z = _ramp(X_FEAT, Lh, a, w, sy / 2, 0.0, True)
    return [_ground(sx, sy), r, _column(X_FEAT + Lh, sx, sy / 2, w, z)], 0.0


def ramp_down(d, sx, sy):
    a, Lh, H, w = _ramp_params(d, sx)
    r, _ = _ramp(X_FEAT, Lh, a, w, sy / 2, H, False)
    return [_ground(sx, sy), _column(0.0, X_FEAT, sy / 2, w, H), r], H


def ramp_updown(d, sx, sy):
    landing = U(1.5, 4.0)
    a = math.radians(U(_lerp(4.0, 10.0, d), _lerp(10.0, 25.0, d)))
    Lh_max = (sx - X_FEAT - landing - 2.0) / 2
    H = min(U(0.4, _lerp(1.0, 2.8, d)), Lh_max * math.tan(a))
    Lh, w = H / math.tan(a), U(1.2, 3.0)
    up, z = _ramp(X_FEAT, Lh, a, w, sy / 2, 0.0, True)
    top = _column(X_FEAT + Lh, X_FEAT + Lh + landing, sy / 2, w, z)
    dn, _ = _ramp(X_FEAT + Lh + landing, Lh, a, w, sy / 2, z, False)
    return [_ground(sx, sy), up, top, dn], 0.0


def rubble(d, sx, sy):
    cell, hmax = U(0.2, 0.35), _lerp(0.02, 0.10, d)
    ms = [_ground(sx, sy)]
    xs = np.arange(X_FEAT, sx - 2.0, cell)
    ys = np.arange(sy / 2 - 1.2, sy / 2 + 1.2, cell)
    for x in xs:
        for y in ys:
            h = U(0.0, hmax) if np.random.rand() < 0.8 else 0.0
            if h > 0.005:
                ms.append(_box((cell, cell, h), (x + cell / 2, y + cell / 2, h / 2)))
    return ms, 0.0


def curbs(d, sx, sy):
    ms = [_ground(sx, sy)]
    k = int(np.random.randint(2, 5))
    xs = np.sort(U(X_FEAT, sx - 3.0, k))
    last = -9.0
    for x in xs:
        if x - last < 1.6:
            continue
        h, depth = U(0.04, _lerp(0.10, 0.24, d)), U(0.15, 0.6)
        ms.append(_box((depth, sy, h), (x + depth / 2, sy / 2, h / 2)))
        last = x
    return ms, 0.0


def flat(d, sx, sy):
    ms = [_ground(sx, sy)]
    for _ in range(int(np.random.randint(0, 4))):
        h = U(0.02, 0.08)
        s = U(0.2, 0.6)
        ms.append(_box((s, s, h), (U(X_FEAT, sx - 2.5), sy / 2 + U(-1.0, 1.0), h / 2)))
    return ms, 0.0


FAMILIES = {"stairs_up": stairs_up, "stairs_down": stairs_down, "stairs_updown": stairs_updown, "irregular_stairs": irregular_stairs,
            "ramp_up": ramp_up, "ramp_down": ramp_down, "ramp_updown": ramp_updown, "rubble": rubble, "curbs": curbs, "flat": flat}
DEFAULT_PROPORTIONS = {"stairs_up": 0.13, "stairs_down": 0.15, "stairs_updown": 0.10, "irregular_stairs": 0.10, "ramp_up": 0.08,
                       "ramp_down": 0.10, "ramp_updown": 0.06, "rubble": 0.10, "curbs": 0.10, "flat": 0.08}


def generate_tile(family, difficulty, size):
    # Isaac Lab sub-terrain contract: meshes spanning [0, sx] x [0, sy] and the spawn origin
    sx, sy = size
    meshes, z0 = FAMILIES[family](float(difficulty), sx, sy)
    return meshes, np.array([X_SPAWN, sy / 2, z0])


@dataclass
class GeneratedWorldCfg:
    tile: tuple = (SX, SY)
    num_rows: int = 10              # difficulty levels
    num_cols: int = 30              # tiles per level (families by proportion, each tile drawn independently)
    seed: int = 0
    init_level: int = 2             # envs start on rows 0..init_level
    friction: float = 1.0
    border: float = 3.0
    proportions: dict = None


def column_families(c: GeneratedWorldCfg):
    props = c.proportions or DEFAULT_PROPORTIONS
    names, p = list(props), np.array(list(props.values()), dtype=float)
    cum = np.cumsum(p / p.sum())
    return [names[int(np.searchsorted(cum, col / c.num_cols + 1e-9))] for col in range(c.num_cols)]
