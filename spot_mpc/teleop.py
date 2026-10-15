"""Keyboard to body-frame velocity command, without any simulator import (the script feeds it key names)."""

FASTER, SLOWER = ("EQUAL", "PLUS", "NUMPAD_ADD"), ("MINUS", "NUMPAD_SUBTRACT")
SPEED_LEVELS = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5)                  # multipliers of the base speed: 0.18 ... 0.68 m/s for 0.45 m/s
FORWARD, BACK = ("W", "UP"), ("S", "DOWN")
LEFT, RIGHT = ("A", "LEFT"), ("D", "RIGHT")
TURN_LEFT, TURN_RIGHT = ("Q",), ("E",)


def key_name(raw):
    """The key's name from whatever the input system hands over: an enum with .name, a string, or a string like 'KeyboardInput.W'."""
    name = getattr(raw, "name", raw)
    return str(name).split(".")[-1].strip().upper()


class KeyCommand:
    """Which keys are down -> (vx, vy, wz) in the body frame.

    The + / = key makes the translation speed one step faster and the - key one step slower (the turn rate does not change).

    W / up arrow walk forward, S / down arrow backward, A / left arrow step left, D / right arrow step right, Q turns left (counter-clockwise), E turns right.
    Opposite keys cancel, and either of a pair (say W and the up arrow) counts once.
    """

    def __init__(self, speed=0.45, strafe=0.25, yaw_rate=0.6):
        self.speed, self.strafe, self.yaw_rate = speed, strafe, yaw_rate
        self.down = set()
        self.level = SPEED_LEVELS.index(1.0)

    @property
    def scale(self):
        return SPEED_LEVELS[self.level]

    def faster(self):
        self.level = min(self.level + 1, len(SPEED_LEVELS) - 1)

    def slower(self):
        self.level = max(self.level - 1, 0)

    def press(self, key):
        if key in FASTER:
            self.faster()
        elif key in SLOWER:
            self.slower()
        else:
            self.down.add(key)

    def release(self, key):
        self.down.discard(key)

    def clear(self):
        self.down.clear()

    def _axis(self, plus, minus):
        return float(any(k in self.down for k in plus)) - float(any(k in self.down for k in minus))

    def command(self):
        return (self.scale * self.speed * self._axis(FORWARD, BACK), self.scale * self.strafe * self._axis(LEFT, RIGHT), self.yaw_rate * self._axis(TURN_LEFT, TURN_RIGHT))
