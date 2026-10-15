import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spot_mpc.teleop import KeyCommand, key_name


def test_no_key_is_a_stop():
    assert KeyCommand().command() == (0.0, 0.0, 0.0)


def test_each_key_drives_its_own_axis():
    k = KeyCommand(speed=0.5, strafe=0.2, yaw_rate=0.7)
    for key, want in (("W", (0.5, 0, 0)), ("S", (-0.5, 0, 0)), ("A", (0, 0.2, 0)), ("D", (0, -0.2, 0)), ("Q", (0, 0, 0.7)), ("E", (0, 0, -0.7))):
        k.clear()
        k.press(key)
        assert k.command() == want, key


def test_arrow_keys_match_wasd():
    k = KeyCommand()
    for arrow, letter in (("UP", "W"), ("DOWN", "S"), ("LEFT", "A"), ("RIGHT", "D")):
        k.clear(); k.press(arrow); a = k.command()
        k.clear(); k.press(letter); assert a == k.command(), arrow


def test_opposite_keys_cancel_and_a_pair_counts_once():
    k = KeyCommand(speed=0.45)
    k.press("W"); k.press("S")
    assert k.command()[0] == 0.0
    k.clear(); k.press("W"); k.press("UP")
    assert k.command()[0] == 0.45


def test_keys_combine_and_release():
    k = KeyCommand(speed=0.45, strafe=0.25, yaw_rate=0.6)
    k.press("W"); k.press("E")
    assert k.command() == (0.45, 0.0, -0.6)
    k.release("W")
    assert k.command() == (0.0, 0.0, -0.6)
    k.release("E"); k.release("E")                       # releasing a key that is not down is harmless
    assert k.command() == (0.0, 0.0, 0.0)


def test_a_key_name_comes_out_the_same_whatever_the_input_system_hands_over():
    class Enum:
        name = "UP"

    assert key_name(Enum()) == "UP" and key_name("up") == "UP" and key_name("KeyboardInput.W") == "W" and key_name("W") == "W"


def test_speed_steps_up_and_down_and_stops_at_the_ends():
    k = KeyCommand(speed=0.5, strafe=0.2, yaw_rate=0.6)
    k.press("W")
    base = k.command()[0]
    k.press("EQUAL")
    assert k.command()[0] > base
    for _ in range(10):
        k.press("NUMPAD_ADD")
    top = k.command()[0]
    k.press("EQUAL")
    assert k.command()[0] == top                                 # no further than the last step
    for _ in range(10):
        k.press("MINUS")
    low = k.command()[0]
    assert 0.0 < low < base
    k.press("NUMPAD_SUBTRACT")
    assert k.command()[0] == low


def test_speed_scales_the_translation_and_not_the_turn_and_is_not_a_held_key():
    k = KeyCommand(speed=0.5, strafe=0.2, yaw_rate=0.6)
    k.press("MINUS")
    k.press("W"); k.press("A"); k.press("Q")
    vx, vy, wz = k.command()
    assert vx < 0.5 and vy < 0.2 and wz == 0.6
    assert "MINUS" not in k.down


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
