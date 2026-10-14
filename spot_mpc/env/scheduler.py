import torch


def previewed_slopes(pitch_ref, travel_sign, steep_pitch):
    """Slope along the direction of travel over the MPC horizon.

    pitch_ref (B,H) is the body pitch reference (negative = nose up) and travel_sign (B,1) is +1 walking forward, -1 backward.
    Returns the steepest previewed ascent and descent [rad] and the same values scaled to 0..1 by steep_pitch.
    """
    along = pitch_ref * travel_sign
    ascent = (-along).clamp(min=0.0).amax(1)
    descent = along.clamp(min=0.0).amax(1)
    return ascent, descent, (ascent / steep_pitch).clamp(0.0, 1.0), (descent / steep_pitch).clamp(0.0, 1.0)


def blend(base, w_up, up, w_back, back):
    """Interpolate a parameter between its base value and the ascent / backward-descent values."""
    return base + w_up * (up - base) + w_back * (back - base)
