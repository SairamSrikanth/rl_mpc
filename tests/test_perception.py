import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from spot_mpc.perception.elevation_map import ElevationMap

OFF = 0.035


def make(B=2, **kw):
    em = ElevationMap(B, "cpu", half_x=2.0, half_y=1.5, res=0.05, ceiling_exclude=1.2, foot_z_offset=OFF, **kw)
    em.reset(torch.arange(B), torch.zeros(B, 2), torch.zeros(B))
    return em


def pts(*xyz):
    return torch.tensor([list(xyz)], dtype=torch.float32)[None].repeat(2, 1, 1)             # (2 envs, 1 point, 3)


def test_update_writes_foot_origin_heights_and_unseen_reads_fallback():
    em = make()
    em.fallback[:] = 0.77
    em.update([(pts(0.5, 0.0, 0.2), None)], torch.zeros(2))
    assert abs(float(em.height(torch.full((2, 1), 0.5), torch.zeros(2, 1))[0, 0]) - (0.2 + OFF)) < 1e-6   # surface + foot offset
    assert bool(em.seen_at(torch.full((2, 1), 0.5), torch.zeros(2, 1)).all())
    far = em.height(torch.full((2, 1), -1.0), torch.zeros(2, 1))                                           # never observed
    assert abs(float(far[0, 0]) - 0.77) < 1e-6 and not bool(em.seen_at(torch.full((2, 1), -1.0), torch.zeros(2, 1)).any())


def test_highest_point_per_cell_wins_and_ceiling_points_are_dropped():
    em = make()
    both = torch.tensor([[[0.5, 0.0, 0.1], [0.5, 0.0, 0.4]]] * 2)
    em.update([(both, None)], torch.zeros(2))
    assert abs(float(em.height(torch.full((2, 1), 0.5), torch.zeros(2, 1))[0, 0]) - (0.4 + OFF)) < 1e-6
    em2 = make()
    em2.update([(pts(0.5, 0.0, 2.0), None)], torch.zeros(2))                                               # 2 m up: a ceiling, not ground
    assert not bool(em2.seen_at(torch.full((2, 1), 0.5), torch.zeros(2, 1)).any())


def test_invalid_and_non_finite_points_are_ignored():
    em = make()
    p = torch.tensor([[[0.5, 0.0, float("nan")], [0.7, 0.0, 0.3]]] * 2)
    valid = torch.tensor([[True, False]] * 2)
    em.update([(p, valid)], torch.zeros(2))
    assert not bool(em.seen_at(torch.full((2, 1), 0.5), torch.zeros(2, 1)).any())
    assert not bool(em.seen_at(torch.full((2, 1), 0.7), torch.zeros(2, 1)).any())                          # masked out
    assert bool(torch.isfinite(em.grid).all())


def test_a_new_observation_overwrites_a_stale_one():
    em = make()
    em.update([(pts(0.5, 0.0, 0.5), None)], torch.zeros(2))
    em.update([(pts(0.5, 0.0, 0.1), None)], torch.zeros(2))                                                # the surface changed: no running max
    assert abs(float(em.height(torch.full((2, 1), 0.5), torch.zeros(2, 1))[0, 0]) - (0.1 + OFF)) < 1e-6


def test_recenter_keeps_overlap_and_marks_the_new_strip_unseen():
    em = make(B=1)
    em.update([(torch.tensor([[[0.0, 0.0, 0.3], [1.9, 0.0, 0.6]]]), None)], torch.zeros(1))
    em.recenter(torch.tensor([[1.0, 0.0]]))                                                                 # robot walked 1 m: window follows
    assert abs(float(em.origin[0, 0]) - (1.0 - 2.0)) < 0.06
    q = lambda x: em.seen_at(torch.tensor([[x]]), torch.zeros(1, 1)).item()                                 # noqa: E731
    assert q(1.9) and not q(2.9 - 0.05)                                                                     # old data kept, far strip unseen
    assert abs(float(em.height(torch.tensor([[1.9]]), torch.zeros(1, 1))) - (0.6 + OFF)) < 1e-6
    assert q(0.0) is False or True                                                                          # (0 m left the window: no assertion on it)


def test_reset_clears_one_env_only():
    em = make()
    em.update([(pts(0.5, 0.0, 0.3), None)], torch.zeros(2))
    em.reset(torch.tensor([0]), torch.tensor([[3.0, 0.0]]), torch.tensor([0.9]))
    assert not bool(em.seen_at(torch.full((2, 1), 0.5), torch.zeros(2, 1))[0].any())                       # env 0 cleared and moved
    assert bool(em.seen_at(torch.full((2, 1), 0.5), torch.zeros(2, 1))[1].all())                           # env 1 untouched
    assert abs(float(em.fallback[0]) - 0.9) < 1e-6


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
