import dataclasses
import math

import numpy as np

from . import layout as W

RES = 0.25
TALL_SHELVES = (0.5, 1.4, 2.3, 3.2, 4.1, 5.0)


def _polys():
    """Route legs per floor (ground, upper), xy polylines"""
    ground = [[(12.0, 7.0), (9.8, 7.0)], [(15.0, 2.5), (15.0, 11.5), (11.0, 11.5)], [(9.8, 7.0), (15.0, 7.0)]]
    upper = [W.STAIR_WAYPOINTS[1:], W._cross_out, W._cross_back, W._descend_back, [(0.0, 7.0), (2.0, 1.3)], [(-20.0, 7.0), (0.0, 7.0)]]
    return ground, upper


def _seg_dist(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


class Floor:
    def __init__(self, cfg, upper, clear, existing):
        self.cfg, self.upper = cfg, upper
        fx, fy = cfg.footprint_x, cfg.footprint_y
        yh = fy / 2 + cfg.extra_north
        self.x0, self.y0 = -fx / 2, -fy / 2
        self.nx, self.ny = int(fx / RES), int((yh + fy / 2) / RES)
        gx = self.x0 + (np.arange(self.nx) + 0.5) * RES
        gy = self.y0 + (np.arange(self.ny) + 0.5) * RES
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        self.X, self.Y = X, Y
        wall = 0.7
        free = (X > -fx / 2 + wall) & (X < fx / 2 - wall) & (Y > -fy / 2 + wall) & (Y < yh - wall)
        if upper:                                           # L-shaped slab: remove the atrium void (+ edge margin)
            free &= ~((X > cfg.atrium_x_min - 0.8) & (Y > cfg.atrium_y_min - 0.8))
            free &= ~((X > cfg.stair_base_xy[0] - 6.0) & (X < cfg.stair_base_xy[0] + 1) & (np.abs(Y - cfg.stair_base_xy[1]) < 2.0))   # stair top landing
        else:
            free &= ~np.vectorize(lambda x, y: W.in_stair_footprint(x, y, cfg, 1.0) or W.in_ramp_footprint(x, y, cfg, 1.0))(X, Y)
        ground, up = _polys()
        for poly in (up if upper else ground):
            for a, b in zip(poly[:-1], poly[1:]):
                free &= _seg_dist(X, Y, a, b) > clear
        for (x, y, fz, _yaw) in W.SPAWN_POSES:
            if (fz > 1.0) == upper:
                free &= np.hypot(X - x, Y - y) > clear
        e = W.EquipmentCfg()
        for (cx, cy) in ((e.spot_spawn_xy,) if not upper else ()):
            free &= np.hypot(X - cx, Y - cy) > 2.5
        self.free = free
        for p in existing:
            if p.name.startswith("Equipment/Upper") != upper and p.name.startswith("Equipment"):
                continue
            if not (p.name.startswith("Equipment") or p.name.startswith("Columns")):
                continue
            self.claim_prim(p, 0.35)

    def claim_prim(self, p, pad):
        if p.kind == "box":
            yaw = 2 * math.atan2(p.quat[3], p.quat[0])
            self._claim(p.pos[0], p.pos[1], p.size[0] / 2 + pad, p.size[1] / 2 + pad, yaw)
        else:
            self._claim(p.pos[0], p.pos[1], p.size[0] + pad, p.size[0] + pad, 0.0)

    def _cells(self, cx, cy, hx, hy, yaw):
        r = math.hypot(hx, hy)
        i0, i1 = max(0, int((cx - r - self.x0) / RES)), min(self.nx, int((cx + r - self.x0) / RES) + 1)
        j0, j1 = max(0, int((cy - r - self.y0) / RES)), min(self.ny, int((cy + r - self.y0) / RES) + 1)
        X, Y = self.X[i0:i1, j0:j1] - cx, self.Y[i0:i1, j0:j1] - cy
        c, s = math.cos(yaw), math.sin(yaw)
        m = (np.abs(X * c + Y * s) <= hx) & (np.abs(-X * s + Y * c) <= hy)
        return i0, i1, j0, j1, m

    def _claim(self, cx, cy, hx, hy, yaw):
        i0, i1, j0, j1, m = self._cells(cx, cy, hx, hy, yaw)
        self.free[i0:i1, j0:j1] &= ~m

    def fits(self, cx, cy, hx, hy, yaw, pad=0.0):
        i0, i1, j0, j1, m = self._cells(cx, cy, hx + pad, hy + pad, yaw)
        return bool(m.any()) and bool(self.free[i0:i1, j0:j1][m].all())

    def put(self, cx, cy, hx, hy, yaw, pad=0.0):
        """Claim with `pad` of free aisle around it so nothing gets walled in"""
        if not self.fits(cx, cy, hx, hy, yaw, 0.0):
            return False
        self._claim(cx, cy, hx + pad, hy + pad, yaw)
        return True


def fill_prims(cfg, existing, seed=None):
    rng = np.random.default_rng(cfg.dense_seed if seed is None else seed)
    out = []
    e = W.EquipmentCfg()
    tall = dataclasses.replace(e, bay_height=5.4, bay_shelf_heights=TALL_SHELVES)
    k = {"Shelf": 0, "Pallet": 0, "BoxPile": 0, "Barrels": 0, "Forklift": 0}
    for upper in (False, True):
        fl = Floor(cfg, upper, clear=1.5, existing=existing)
        base = "Fill/Upper" if upper else "Fill/Ground"
        z0 = cfg.upper_floor_top_z if upper else 0.0

        # rack rows; the ones in the ground-level atrium are tall
        for yrow in np.arange(-cfg.footprint_y / 2 + 1.5, cfg.footprint_y / 2 + cfg.extra_north, 3.0):
            for bx in np.arange(-cfg.footprint_x / 2 + 1.5, cfg.footprint_x / 2, 2.05):
                atrium = (not upper) and bx > cfg.atrium_x_min and yrow > cfg.atrium_y_min
                ecfg = tall if atrium else e
                if fl.put(bx, yrow, 1.0, 0.5, 0.0, pad=0.0):
                    W._shelving_bay(out, f"{base}/Shelf_{k['Shelf']:03d}", (bx, yrow, z0), 0.0, ecfg)
                    k["Shelf"] += 1
        # leave an aisle around every rack
        for p in out:
            if p.name.startswith(base + "/Shelf") and p.name.endswith("Deck_0"):
                fl.claim_prim(p, 0.9)

        # pallets, box piles, barrels and forklifts in the space that is left
        want = dict(Forklift=3 if not upper else 1, Pallet=26 if not upper else 14, BoxPile=12 if not upper else 6, Barrels=8 if not upper else 3)
        for kind, n in want.items():
            r = dict(Pallet=(0.62, 0.52), BoxPile=(0.7, 0.7), Barrels=(0.65, 0.65), Forklift=(1.1, 1.75))[kind]
            made = 0
            for _ in range(600):
                if made >= n:
                    break
                x = rng.uniform(-cfg.footprint_x / 2 + 1, cfg.footprint_x / 2 - 1)
                y = rng.uniform(-cfg.footprint_y / 2 + 1, cfg.footprint_y / 2 + cfg.extra_north - 1)
                yaw = float(rng.choice([0.0, math.pi / 2])) + float(rng.normal(0, 0.06))
                if not fl.put(x, y, r[0], r[1], yaw, pad=0.4):
                    continue
                name = f"{base}/{kind}_{k[kind]:03d}"
                if kind == "Pallet":
                    W._pallet(out, name, (x, y, z0), yaw, e, rng.random() < 0.7)
                elif kind == "BoxPile":
                    W._box_pile(out, name, (x, y), z0, rng)
                elif kind == "Barrels":
                    W._barrels(out, name, (x, y), z0, rng, e)
                else:
                    W._forklift(out, name, (x, y, z0), yaw)
                k[kind] += 1
                made += 1
    return out
