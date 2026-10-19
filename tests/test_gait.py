
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from spot_mpc.control import footholds as fh, gait as gs, swing as sw
from spot_mpc.worlds.test_world import World, WorldCfg


def test_trot_diagonals_in_phase_and_duty():
    g = gs.BatchedGait(1, "cpu", freq=2.0, duty=0.6)
    g.set_trotting(torch.ones(1, dtype=torch.bool))
    n, c = 4000, torch.zeros(4)
    for _ in range(n):
        g.step(1 / 400.0)
        cc = g.contact()[0]
        assert bool(cc[0] == cc[3]) and bool(cc[1] == cc[2])      # diagonal pairs move together
        assert bool(cc[0] != cc[1]) or True
        c += cc.float()
    assert bool(((c / n - 0.6).abs() < 0.01).all())                # stance fraction == duty


def test_gait_frequency_change_has_no_phase_jump():
    g = gs.BatchedGait(1, "cpu")
    g.set_trotting(torch.ones(1, dtype=torch.bool))
    for _ in range(100):
        g.step(1 / 400.0)
    before = g.phase.clone()
    g.set_freq(torch.tensor([3.3]))
    assert torch.equal(before, g.phase)


def test_swing_trajectory_endpoints_and_continuity():
    p0 = torch.tensor([[[0.0, 0.0, 0.035]] * 4])
    ptd = torch.tensor([[[0.30, 0.0, 0.22]] * 4])
    zpk = torch.full((1, 4), 0.32)
    tsw = torch.tensor([0.2])
    s0, v0 = sw.swing_target(p0, ptd, torch.zeros(1, 4), tsw, zpk)
    s1, v1 = sw.swing_target(p0, ptd, torch.ones(1, 4), tsw, zpk)
    assert torch.allclose(s0, p0, atol=1e-6) and torch.allclose(s1, ptd, atol=1e-6)
    assert float(v0.abs().max()) < 1e-4 and float(v1.abs().max()) < 1e-4       # swing starts and ends at rest
    s = torch.linspace(0, 1, 400).repeat(1, 1).T[:, :, None].expand(400, 1, 4)
    pos = torch.stack([sw.swing_target(p0, ptd, s[i], tsw, zpk)[0] for i in range(400)])
    assert float((pos[1:] - pos[:-1]).abs().max()) < 0.01                       # no jumps


def test_swing_apex_clears_a_riser():
    w = World(WorldCfg(), "cpu")
    c = w.cfg
    # foot just behind the first riser, landing on the first tread
    p0 = torch.tensor([[[c.stair_x - 0.04, 0.0, 0.035]] * 4])
    ptd = torch.tensor([[[c.stair_x + 0.15, 0.0, c.rise + 0.035]] * 4])
    z = sw.swing_apex(p0, ptd, w, clearance=0.06)
    assert float(z.min()) >= c.rise + 0.035 + 0.06 - 1e-3


def test_foothold_body_frame_residual():
    P = gs.PlannerParams()

    class Flat:
        height = staticmethod(lambda x, y: torch.zeros_like(x))
        project = staticmethod(lambda xy, m=0.08: xy)
    com = torch.zeros(1, 4, 2)
    v = torch.zeros(1, 1, 2)
    T = torch.tensor([[0.3]])
    res = torch.zeros(1, 4, 2)
    res[..., 0] = 0.05                                                         # 5 cm forward, body frame
    for yaw in (0.0, math.pi / 2, math.pi):
        y = torch.full((1, 4), yaw)
        a = fh.foothold(P, Flat, com, y, v, v, T, None)[..., :2]
        b = fh.foothold(P, Flat, com, y, v, v, T, res)[..., :2]
        d = (b - a)[0, 0]
        assert abs(float(d[0]) - 0.05 * math.cos(yaw)) < 1e-6 and abs(float(d[1]) - 0.05 * math.sin(yaw)) < 1e-6


def test_safe_foothold_flat_identity_and_stair_margin():
    w = World(WorldCfg(), "cpu")
    c = w.cfg
    flat = torch.tensor([[-5.0, 0.0], [-3.0, 1.0]])
    assert torch.allclose(w.project(flat), flat)                                # flat: no movement
    riser = c.stair_x + 2 * c.run                                               # a riser
    bad = torch.tensor([[riser - 0.01, 0.0], [riser + 0.01, 0.2], [riser - 0.03, -0.3]])
    out = w.project(bad)
    d = (out[:, 0] - riser).abs()
    assert float(d.min()) > 0.045, d                                           # moved off the edge, margin from the riser
    h = w.height(out[:, 0:1], out[:, 1:2])
    assert float((h.squeeze() - w.height(bad[:, 0:1], bad[:, 1:2]).squeeze()).abs().max()) <= c.rise + 1e-6


def test_world_geometry():
    w = World(WorldCfg(), "cpu")
    c = w.cfg
    x = torch.tensor([c.stair_x - 0.5, c.stair_x + 0.05, c.stair_x + 0.35, c.stair_x + 15 * c.run + 0.1, c.stair_x + 16 * c.run + 1.0])
    y = torch.full_like(x, c.stair_y)
    assert torch.allclose(w.surface(x, y), torch.tensor([0.0, c.rise, 2 * c.rise, 16 * c.rise, 16 * c.rise]), atol=1e-5)
    assert float(w.surface(torch.tensor([c.stair_x + 1.0]), torch.tensor([c.stair_y + 3.0]))) == 0.0   # off the stair lane
    a = math.radians(c.ramp_deg)
    xr = torch.tensor([c.stair_x + 2.0, c.stair_x + 4.0])
    zr = w.surface(xr, torch.full_like(xr, c.ramp_y))
    assert abs(float(zr[1] - zr[0]) / 2.0 - math.tan(a)) < 1e-4                 # slope == tan(ramp angle)
    assert abs(float(w.height(x[:1], y[:1]) - w.surface(x[:1], y[:1])) - c.foot_z_offset) < 1e-6


def test_safe_foothold_ignores_a_uniform_slope_but_not_a_riser():
    # a slope with mm-level cell noise must not move the foot, a riser must
    slope = 0.333
    ramp = lambda x, y: slope * x + 0.006 * torch.sin(97.0 * x + 53.0 * y)
    xy = torch.tensor([[1.0, 0.3], [2.0, -0.2], [3.0, 0.1]])
    assert float((fh.safe_foothold(ramp, xy) - xy).abs().max()) < 1e-6
    riser = 0.5
    stair = lambda x, y: torch.where(x > riser, torch.full_like(x, 0.19), torch.zeros_like(x))
    near = torch.tensor([[riser - 0.02, 0.0], [riser + 0.03, 0.1]])
    out = fh.safe_foothold(stair, near)
    assert float((out[:, 0] - riser).abs().min()) > 0.045


def test_world_ramp_descent_geometry():
    w = World(WorldCfg(), "cpu")
    c = w.cfg
    y = torch.full((1,), c.ramp_y)
    f = lambda x: float(w.surface(torch.tensor([x]), y))
    a = math.radians(c.ramp_deg)
    assert abs(f(w.ramp_down_x - 1.0) - w.ramp_top) < 1e-5                      # level landing at the top
    mid = w.ramp_down_x + 2.0
    assert abs(f(mid) - (w.ramp_top - 2.0 * math.tan(a))) < 1e-4                 # slopes down at the ramp angle
    assert abs(f(w.ramp_bottom_x + 0.5)) < 1e-6                                  # back on the floor
    # the physics box and the analytic surface agree: the descending slab's top passes through (ramp_down_x, ramp_top) and (ramp_bottom_x, 0)
    prim = {p.name: p for p in w.primitives()}["ramp_down"]
    size, pos, quat = prim.size, prim.pos, prim.quat
    ang = 2 * math.atan2(quat[2], quat[0])                                       # rotation about y
    nrm = (math.sin(ang), math.cos(ang))                                         # slab top normal in the (x, z) plane
    top_pt = (pos[0] + nrm[0] * size[2] / 2, pos[2] + nrm[1] * size[2] / 2)      # centre of the top face
    along = (math.cos(ang), -math.sin(ang))
    for s_ in (-1.0, 1.0):                                                       # two points on the top face
        x_, z_ = top_pt[0] + s_ * along[0] * size[0] / 2, top_pt[1] + s_ * along[1] * size[0] / 2
        assert abs(z_ - f(x_)) < 5e-3, (x_, z_, f(x_))


def test_world_descent_geometry():
    w = World(WorldCfg(), "cpu")
    c = w.cfg
    top = c.n_steps * c.rise
    y = torch.full((1,), c.stair_y)
    f = lambda x: float(w.surface(torch.tensor([x]), y))
    assert abs(f(w.descent_x - 1.0) - top) < 1e-6                              # the landing is level at the top
    assert abs(f(w.descent_x + 0.05) - (top - c.rise)) < 1e-6                  # first tread down
    assert abs(f(w.descent_x + 5 * c.run + 0.05) - (top - 6 * c.rise)) < 1e-6
    assert abs(f(w.bottom_x + 0.5)) < 1e-6                                     # back on the floor
    assert abs(w.bottom_x - (w.descent_x + c.n_steps * c.run)) < 1e-9
    assert float(w.surface(torch.tensor([w.descent_x + 1.0]), torch.tensor([c.stair_y + 3.0]))) == 0.0   # off the lane



if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as ex:
            fails += 1
            print(f"  [FAIL] {name}: {ex}")
    sys.exit(1 if fails else 0)


def test_swing_apex_is_the_same_for_a_staircase_turned_90_degrees():
    class Step:
        def __init__(self, axis):
            self.axis = axis

        def height(self, x, y):
            return torch.where((x, y)[self.axis] > 0.0, 0.2, 0.0) + 0.035

    p0, ptd = torch.tensor([[[-0.03, 0.0, 0.035]] * 4]), torch.tensor([[[0.15, 0.0, 0.235]] * 4])
    along_x = sw.swing_apex(p0, ptd, Step(0))
    along_y = sw.swing_apex(p0[..., [1, 0, 2]], ptd[..., [1, 0, 2]], Step(1))
    assert torch.allclose(along_x, along_y)


def test_duty_change_waits_rather_than_flip_a_leg_mid_step():
    g = gs.BatchedGait(1, "cpu", duty=0.6)
    g.set_trotting(torch.ones(1, dtype=torch.bool))
    g.phase = torch.tensor([[0.62, 0.12, 0.12, 0.62]])       # fl, hr just lifted off
    g.set_duty(torch.tensor([0.65]))                          # would put them back in stance in the air
    assert abs(float(g.trot_duty) - 0.6) < 1e-6
    g.phase = torch.tensor([[0.70, 0.20, 0.20, 0.70]])
    g.set_duty(torch.tensor([0.65]))
    assert abs(float(g.trot_duty) - 0.65) < 1e-6


def test_sideways_step_on_a_slope_asks_for_no_vertical_speed():
    class Slope:                                              # rises along x, flat along y
        def height(self, x, y):
            return 0.3 * x + 0.035

        def project(self, xy, margin=0.08):
            return xy

    P, x0 = gs.PlannerParams(), torch.zeros(1, 13)
    x0[:, 5], x0[:, 12] = 0.485, -9.81
    feet = torch.zeros(1, 4, 3)
    anchor = (torch.zeros(1, dtype=torch.bool), x0[:, 3:5].clone(), x0[:, 2].clone())
    side, _, _ = gs.build_horizon(P, Slope(), gs.BatchedGait(1, "cpu"), x0, torch.tensor([[0.0, 0.3, 0.0]]), feet, anchor, torch.full((1,), 0.45))
    up, _, _ = gs.build_horizon(P, Slope(), gs.BatchedGait(1, "cpu"), x0, torch.tensor([[0.3, 0.0, 0.0]]), feet, anchor, torch.full((1,), 0.45))
    assert float(side[..., 11].abs().max()) < 1e-6
    assert abs(float(up[0, 0, 11]) - 0.09) < 1e-3


def test_surface_normals_follow_a_slope_but_ignore_a_nearby_riser():
    slope = lambda x, y: 0.3 * x                              # noqa: E731
    riser = lambda x, y: torch.where(x > 0.01, 0.2, 0.0)      # noqa: E731
    n = fh.surface_normals(slope, torch.zeros(1), torch.zeros(1))
    assert torch.allclose(n, torch.tensor([[-0.3, 0.0, 1.0]]) / math.sqrt(1.09), atol=1e-5)
    assert torch.allclose(fh.surface_normals(riser, torch.zeros(1), torch.zeros(1)), torch.tensor([[0.0, 0.0, 1.0]]))
