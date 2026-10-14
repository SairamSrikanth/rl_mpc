import math

import numpy as np


def look_at_from_euler(pos, euler_deg):
    """Eye and a look-at point 10 m ahead for a camera pose given as a position and the viewer's Euler readout (z applied first).
    The camera looks down its own -Z axis."""
    def rot(axis, deg):
        t = math.radians(deg)
        c, s = math.cos(t), math.sin(t)
        return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]

    rx, ry, rz = euler_deg
    R = rot("x", rx) @ rot("y", ry) @ rot("z", rz)
    eye = np.array(pos, float)
    return tuple(eye), tuple(eye + 10.0 * (R @ np.array([0.0, 0.0, -1.0])))


def set_lens(env, focal, width, height, aperture=20.955):
    """Focal length and aperture of the env's viewer camera (the camera that env.render() reads)."""
    import omni.usd
    from pxr import Gf
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(env.cfg.viewer.cam_prim_path)
    prim.GetAttribute("focalLength").Set(focal)
    prim.GetAttribute("horizontalAperture").Set(aperture)
    prim.GetAttribute("verticalAperture").Set(aperture * height / width)
    prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.05, 200.0))
