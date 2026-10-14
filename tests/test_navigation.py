import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from spot_mpc.navigation import PathFollower


def follower(route, **kw):
    return PathFollower(1, torch.tensor(route, dtype=torch.float32), "cpu", **kw)


def test_walks_straight_to_a_waypoint_on_its_heading():
    nav = follower([(5.0, 0.0)])
    cmd = nav.command(torch.zeros(1, 2), torch.zeros(1))
    assert float(cmd[0, 0]) > 0.4 and abs(float(cmd[0, 2])) < 1e-6 and float(cmd[0, 1]) == 0.0


def test_turns_on_the_spot_when_facing_away():
    nav = follower([(5.0, 0.0)])
    cmd = nav.command(torch.zeros(1, 2), torch.full((1,), math.pi))
    assert float(cmd[0, 0]) == 0.0 and abs(float(cmd[0, 2])) > 0.1


def test_advances_at_the_arrival_radius_and_stops_when_complete():
    nav = follower([(1.0, 0.0), (2.0, 0.0)], arrive=0.5)
    nav.command(torch.tensor([[0.6, 0.0]]), torch.zeros(1))
    assert int(nav.idx[0]) == 1
    cmd = nav.command(torch.tensor([[1.8, 0.0]]), torch.zeros(1))
    assert bool(nav.complete[0]) and float(cmd.abs().max()) == 0.0


def test_backward_segments_reverse_the_robot():
    nav = follower([(-5.0, 0.0)], back=[True])
    cmd = nav.command(torch.zeros(1, 2), torch.zeros(1))                     # facing +x, walking toward -x backward
    assert float(cmd[0, 0]) < -0.4 and abs(float(cmd[0, 2])) < 1e-5


def test_per_env_routes_are_independent():
    routes = torch.tensor([[[5.0, 0.0]], [[0.0, 5.0]]])
    nav = PathFollower(2, routes, "cpu")
    cmd = nav.command(torch.zeros(2, 2), torch.zeros(2))
    assert float(cmd[0, 2]) == 0.0 and float(cmd[1, 2]) > 0.1                # env 1 has to turn left toward +y


def test_cross_track_error_bends_the_heading_back_to_the_line():
    nav = follower([(5.0, 0.0)])
    nav.seg0 = torch.zeros(1, 2)
    cmd = nav.command(torch.tensor([[1.0, 0.5]]), torch.zeros(1))            # left of the line, facing along it
    assert float(cmd[0, 2]) < 0.0                                             # turn right, back toward the line


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
