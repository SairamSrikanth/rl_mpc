import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from spot_mpc.control import gait as gs
from spot_mpc.control.mpc import BatchedSrbdMpc, MpcParams, project_first_step

MASS, G = 31.6, 9.81
INERTIA = np.array([[0.583, 0.0, -0.0781], [0.0, 2.2076, 0.0], [-0.0781, 0.0, 2.3596]])
TAU = np.array([45.0, 45.0, 90.0])


class Flat:
    """Flat ground at foot-origin height 0.035."""
    def height(self, x, y):
        return torch.full_like(x, 0.035)

    def project(self, xy, margin=0.08):
        return xy


def make(B=1, **kw):
    p = MpcParams(dt=0.025, mass=MASS, inertia_b=INERTIA, horizon=16)
    return p, BatchedSrbdMpc(p, iters=30, rho=0.003, dtype=torch.float32, device="cpu")


def stand_state(B, yaw=0.0, h=0.45):
    x = torch.zeros(B, 13)
    x[:, 2] = yaw
    x[:, 5] = h + 0.035
    x[:, 12] = -G
    return x


def feet_under(x0):
    P = gs.PlannerParams()
    nom = torch.as_tensor(P.nominal)
    fx, fy = gs.rot2(x0[:, 2:3], nom[:, 0], nom[:, 1])
    return torch.stack([x0[:, 3:4] + fx, x0[:, 4:5] + fy, torch.full_like(fx, 0.035)], -1)


def solve(mpc, x0, cmd, gait, feet, accel_resid=None):
    P = gs.PlannerParams()
    anchor = (torch.zeros(x0.shape[0], dtype=torch.bool), x0[:, 3:5].clone(), x0[:, 2].clone())
    x_ref, r, contact = gs.build_horizon(P, Flat(), gait, x0, cmd, feet, anchor, torch.full((x0.shape[0],), 0.45))
    tm = torch.zeros(x0.shape[0], 4, 3, 3) + torch.eye(3)       # identity torque map: |F| <= tau_lim bound only
    f, _ = mpc.solve(x0, x_ref, r, contact, u_prev=None, torque_map=tm, tau_lim=TAU, accel_resid=accel_resid)
    return f


def test_stand_supports_weight():
    p, mpc = make()
    x0 = stand_state(1)
    gait = gs.BatchedGait(1, "cpu")                    # standing: duty 1.01, every foot in stance
    f = solve(mpc, x0, torch.zeros(1, 3), gait, feet_under(x0))
    assert abs(float(f[0, :, 2].sum()) - MASS * G) < 0.05 * MASS * G
    assert float(f[..., 2].min()) >= 0.0


def test_swing_feet_carry_no_force():
    p, mpc = make()
    x0 = stand_state(1)
    gait = gs.BatchedGait(1, "cpu", duty=0.6)
    gait.set_trotting(torch.ones(1, dtype=torch.bool))
    gait.phase = torch.tensor([[0.1, 0.8, 0.8, 0.1]])  # fr, hl in swing (phase >= duty)
    f = solve(mpc, x0, torch.zeros(1, 3), gait, feet_under(x0))
    assert float(f[0, 1].abs().max()) == 0.0 and float(f[0, 2].abs().max()) == 0.0
    assert float(f[0, 0, 2]) > 0 and float(f[0, 3, 2]) > 0


def test_forward_command_pushes_forward():
    p, mpc = make()
    x0 = stand_state(1)
    gait = gs.BatchedGait(1, "cpu")
    f = solve(mpc, x0, torch.tensor([[0.6, 0.0, 0.0]]), gait, feet_under(x0))
    assert float(f[0, :, 0].sum()) > 1.0


def test_first_step_force_is_exactly_feasible():
    torch.manual_seed(0)
    u = torch.randn(64, 12) * 300.0
    contact = (torch.rand(64, 4) > 0.4).float()
    tm = torch.randn(64, 4, 3, 3)
    f = project_first_step(u, contact, 0.6, 220.0, tm, TAU)
    assert float(f[..., 2].min()) >= 0 and float(f[..., 2].max()) <= 220.0 + 1e-4
    assert bool((f[..., :2].abs() <= 0.6 * f[..., 2:3] + 1e-4).all())
    assert bool(((f * (1 - contact)[..., None]).abs() < 1e-9).all())
    assert bool((torch.einsum("bfjk,bfk->bfj", tm, f).abs() <= torch.tensor(TAU) + 1e-3).all())


def test_pipeline_is_rotation_equivariant():
    """Forces at heading `yaw` with a body-frame command AND body-frame accel residual equal the heading-0 forces
    rotated by yaw -- including headings at and across the +-pi wrap and an unwrapped yaw (yaw + 2 pi)."""
    res = torch.tensor([[0.8, -0.5, 0.0, 0.3, -0.2, 0.4]])
    cmd = torch.tensor([[0.5, 0.2, 0.3]])
    outs = {}
    for yaw in (0.0, 0.7, 1.9, math.pi - 1e-3, math.pi, -math.pi + 1e-3, 0.7 + 2 * math.pi):
        p, mpc = make()
        x0 = stand_state(1, yaw)
        gait = gs.BatchedGait(1, "cpu")
        gait.set_trotting(torch.ones(1, dtype=torch.bool))
        gait.phase = torch.tensor([[0.2, 0.7, 0.7, 0.2]])
        outs[yaw] = solve(mpc, x0, cmd, gait, feet_under(x0), accel_resid=res)[0]
    base = outs[0.0]
    for yaw, f in outs.items():
        c, s = math.cos(yaw), math.sin(yaw)
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        err = float((f - base @ rot.T).abs().max())
        assert err < 0.6, (yaw, err)          # N; ADMM runs a fixed 30 iterations, so exact only to solver tolerance


def test_yaw_wrap_invariant_state():
    """+-pi are the same heading: the planner/horizon must not care which representation it gets."""
    P = gs.PlannerParams()
    gait = gs.BatchedGait(2, "cpu")
    x0 = stand_state(2)
    x0[0, 2], x0[1, 2] = math.pi - 1e-4, -math.pi - 1e-4 + 2e-4 + 0.0  # same physical heading up to 1e-4
    x0[1, 2] = -math.pi + 1e-4
    cmd = torch.tensor([[0.4, 0.1, 0.0]] * 2)
    anchor = (torch.zeros(2, dtype=torch.bool), x0[:, 3:5].clone(), x0[:, 2].clone())
    xr, r, c = gs.build_horizon(P, Flat(), gait, x0, cmd, feet_under(x0), anchor, torch.full((2,), 0.45))
    assert float((xr[0, :, 9:11] - xr[1, :, 9:11]).abs().max()) < 1e-3          # world-frame velocity reference
    assert float((r[0] - r[1]).abs().max()) < 1e-3                               # foot-minus-CoM vectors


def test_reset_clears_a_non_finite_warm_start():
    """NaN * 0 is NaN: a reset that multiplies the warm start can never clear a poisoned one. Env 0 is poisoned, env 1 is not."""
    p, mpc = make()
    x0 = stand_state(2)
    gait = gs.BatchedGait(2, "cpu")
    cmd = torch.zeros(2, 3)
    f = solve(mpc, x0, cmd, gait, feet_under(x0))
    assert bool(torch.isfinite(f).all())
    x, z, y = mpc.state
    x = x.clone()
    x[0] = float("nan")
    mpc.state = (x, z, y)
    mpc.reset(torch.tensor([True, False]))                                     # reset env 0 only
    assert bool(torch.isfinite(mpc.state[0][0]).all()) and bool(torch.isfinite(mpc.state[0][1]).all())
    f = solve(mpc, x0, cmd, gait, feet_under(x0))
    assert bool(torch.isfinite(f).all())
    # and a poisoned warm start that is NOT reset is dropped by the solver itself
    x, z, y = mpc.state
    x = x.clone()
    x[1] = float("inf")
    mpc.state = (x, z, y)
    f = solve(mpc, x0, cmd, gait, feet_under(x0))
    assert bool(torch.isfinite(f).all())



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
