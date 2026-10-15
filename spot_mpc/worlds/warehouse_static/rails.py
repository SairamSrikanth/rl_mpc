import math

import numpy as np

from ...common.prims import Prim, floats as _f

YELLOW = (1.0, 0.78, 0.0)
LANDING = 1.6          # metres of the ramp's foot left open (slope length)


def _guard(out, p0, p1, z):
    """Guard rail from p0 to p1 (axis-aligned) on a floor at height z: posts every 1.5 m, a top and a middle rail, a toe board."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    length = float(np.linalg.norm(p1 - p0))
    if length < 0.2:
        return
    d = (p1 - p0) / length
    n = max(1, math.ceil(length / 1.5))
    for i in range(n + 1):
        p = p0 + d * length * i / n
        out.append(Prim(f"Rails/Post_{len(out):03d}", "box", _f(p[0], p[1], z + 0.55), _f(0.06, 0.06, 1.1), color=YELLOW))
    c = (p0 + p1) / 2
    for h, thick, height in ((1.05, 0.05, 0.05), (0.55, 0.05, 0.05), (0.05, 0.02, 0.10)):
        size = (length, thick, height) if abs(d[0]) > 0.5 else (thick, length, height)
        out.append(Prim(f"Rails/Bar_{len(out):03d}", "box", _f(c[0], c[1], z + h), _f(*size), color=YELLOW))


def rail_prims(cfg):
    """The guard rail along the drop from the upper floor into the atrium (open at the stair top and the ramp top) and the short rails on both sides of the stair and the ramp.
    Colliders for the robot only: they are not in the ray mesh or in surface(), so the sensed map and the measured climbs are the same with or without them."""
    out = []
    z, inset = cfg.upper_floor_top_z, 0.12
    ex, ey = cfg.atrium_x_min - inset, cfg.atrium_y_min - inset
    x_end, y_end = cfg.footprint_x / 2 - 0.12, cfg.footprint_y / 2 + cfg.extra_north - 0.12
    rx = cfg.ramp_bottom_xyz[0]
    ramp_gap = (rx - cfg.ramp_width / 2 - 0.25, rx + cfg.ramp_width / 2 + 0.25)
    sy = cfg.stair_base_xy[1]
    stair_gap = (sy - cfg.step_width / 2 - 0.25, sy + cfg.step_width / 2 + 0.25)
    _guard(out, (ex, ey), (ramp_gap[0], ey), z)
    _guard(out, (ramp_gap[1], ey), (x_end, ey), z)
    _guard(out, (ex, ey), (ex, stair_gap[0]), z)
    _guard(out, (ex, stair_gap[1]), (ex, y_end), z)

    height, off, post, rail = 0.5, 0.06, 0.05, 0.045
    x0, y0 = cfg.stair_base_xy
    run, rise, n = cfg.step_run, cfg.step_rise, cfg.num_steps
    xa, xb = x0, x0 - run * n
    za, zb = rise + height, rise + rise / run * (x0 - xb) + height
    length, pitch = math.hypot(xa - xb, zb - za), math.atan2(zb - za, xa - xb)
    quat = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
    for side in (-1, 1):
        y = y0 + side * (cfg.step_width / 2 + off)
        mid = ((xa + xb) / 2, y, (za + zb) / 2)
        out.append(Prim(f"Rails/Side_{len(out):03d}", "box", _f(*mid), _f(length, rail, rail), _f(*quat), YELLOW))
        out.append(Prim(f"Rails/Side_{len(out):03d}", "box", _f(mid[0], mid[1], mid[2] - height / 2), _f(length, 0.02, 0.09), _f(*quat), YELLOW))
        for i in range(0, n + 1, 3):
            x = x0 - run * i
            zz = rise + rise / run * (x0 - x)
            out.append(Prim(f"Rails/Post_{len(out):03d}", "box", _f(x, y, zz + height / 2 - 0.07), _f(post, post, height + 0.14), color=YELLOW))

    (rx, ry0, rz0), (_, ry1, rz1) = cfg.ramp_bottom_xyz, cfg.ramp_top_xyz
    total = math.hypot(ry0 - ry1, rz1 - rz0)
    f0 = LANDING / total                                   # the rails start a little way up: the routes turn onto the ramp at its foot, from the side
    angle = math.atan2(rz1 - rz0, ry1 - ry0)
    quat = (math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0)
    length = total * (1.0 - f0)
    mid = (1.0 + f0) / 2
    for side in (-1, 1):
        x = rx + side * (cfg.ramp_width / 2 + off)
        c = (x, ry0 + mid * (ry1 - ry0), rz0 + mid * (rz1 - rz0) + height)
        out.append(Prim(f"Rails/Side_{len(out):03d}", "box", _f(*c), _f(rail, length, rail), _f(*quat), YELLOW))
        out.append(Prim(f"Rails/Side_{len(out):03d}", "box", _f(c[0], c[1], c[2] - height / 2), _f(0.02, length, 0.09), _f(*quat), YELLOW))
        for t in np.linspace(f0, 1.0, max(2, math.ceil(length / 1.5) + 1)):
            y, zz = ry0 + t * (ry1 - ry0), rz0 + t * (rz1 - rz0)
            out.append(Prim(f"Rails/Post_{len(out):03d}", "box", _f(x, y, zz + height / 2 - 0.07), _f(post, post, height + 0.14), color=YELLOW))
    return out
