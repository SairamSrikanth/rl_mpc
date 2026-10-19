from dataclasses import dataclass, field

import torch

GRAV = 9.81


def spot_footprint(hip_x=0.2978, lateral=0.20, com_x=-0.0259):
    """Nominal stance foot positions relative to the CoM in the yaw frame, order fl, fr, hl, hr."""
    fx, hx = hip_x - com_x, -hip_x - com_x
    return ((fx, lateral), (fx, -lateral), (hx, lateral), (hx, -lateral))


@dataclass
class PlannerParams:
    dt: float = 0.025
    H: int = 16
    nominal: tuple = field(default_factory=spot_footprint)
    kd: float = 0.15               # Raibert velocity-error gain
    max_dev: float = 0.15          # cap on the Raibert correction
    margin: float = 0.08           # keep footholds this far from a height discontinuity
    clearance: float = 0.06
    base_apex: float = 0.08
    edge_tol: float = 0.02
    max_apex: float = 0.35
    swing_delay: float = 0.2       # fraction of the swing spent lifting before moving forward


def rot2(yaw, x, y):
    c, s = torch.cos(yaw), torch.sin(yaw)
    return c * x - s * y, s * x + c * y


class BatchedGait:
    """Per-env gait clock. Leg i is in stance while its phase is below the duty factor."""

    def __init__(self, num_envs, device, freq=2.5, duty=0.6, offsets=(0.0, 0.5, 0.5, 0.0), stand_duty=1.01,
                 freq_range=(1.5, 3.5), dtype=torch.float32):
        self.B, self.device = num_envs, device
        self.freq0, self.freq_range, self.stand_duty = freq, freq_range, stand_duty
        self.offsets = torch.tensor(offsets, device=device, dtype=dtype)
        self.phase = self.offsets.repeat(num_envs, 1)
        self.freq = torch.full((num_envs,), freq, device=device, dtype=dtype)
        self.trot_duty = torch.full((num_envs,), duty, device=device, dtype=dtype)
        self.trotting = torch.zeros(num_envs, device=device, dtype=torch.bool)

    @property
    def duty(self):
        return torch.where(self.trotting, self.trot_duty, torch.full_like(self.trot_duty, self.stand_duty))

    def step(self, dt):
        self.phase = torch.remainder(self.phase + self.freq[:, None] * dt, 1.0)

    def set_duty(self, duty):
        """Change the trot duty factor unless that would flip a leg between stance and swing mid-step (that env retries next tick)."""
        lo, hi = torch.minimum(duty, self.trot_duty)[:, None], torch.maximum(duty, self.trot_duty)[:, None]
        flips = ((self.phase >= lo) & (self.phase < hi)).any(1) & self.trotting
        self.trot_duty = torch.where(flips, self.trot_duty, duty)

    def set_freq(self, freq):
        self.freq = freq.clamp(*self.freq_range)

    def set_trotting(self, mask):
        start = mask & ~self.trotting
        self.phase = torch.where(start[:, None], self.offsets.expand_as(self.phase), self.phase)
        self.trotting = mask.clone()

    def reset(self, env_ids):
        self.phase[env_ids] = self.offsets
        self.freq[env_ids] = self.freq0
        self.trotting[env_ids] = False

    def phase_ahead(self, tau):
        """Phase after tau seconds, tau of shape (H,) -> (B,H,4)."""
        return torch.remainder(self.phase[:, None, :] + self.freq[:, None, None] * tau[None, :, None], 1.0)

    def _per_env(self, x, ph):
        return x.reshape((-1,) + (1,) * (ph.ndim - 1))

    def contact(self, ph=None):
        ph = self.phase if ph is None else ph
        return ph < self._per_env(self.duty, ph)

    def swing_progress(self, ph=None):
        ph = self.phase if ph is None else ph
        d = self._per_env(self.duty, ph)
        return torch.where(ph >= d, (ph - d) / (1.0 - d).clamp(min=1e-6), torch.zeros_like(ph))

    def time_to_touchdown(self, ph=None):
        ph = self.phase if ph is None else ph
        return (1.0 - ph) / self._per_env(self.freq, ph)

    def stance_elapsed(self, ph=None):
        ph = self.phase if ph is None else ph
        return ph / self._per_env(self.freq, ph)

    @property
    def stance_duration(self):
        return self.duty / self.freq

    @property
    def swing_duration(self):
        return (1.0 - self.trot_duty) / self.freq


class ReferenceGovernor:
    """Latches a position/yaw anchor when the command is zero and the robot has settled, so a stand holds its place."""

    def __init__(self, num_envs, device, v_eps=1e-3, w_eps=1e-3, v_settle=0.05, w_settle=0.1, dtype=torch.float32):
        self.active = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.xy = torch.zeros(num_envs, 2, device=device, dtype=dtype)
        self.yaw = torch.zeros(num_envs, device=device, dtype=dtype)
        self.v_eps, self.w_eps, self.v_settle, self.w_settle = v_eps, w_eps, v_settle, w_settle

    def update(self, x0, cmd):
        zero = (cmd[:, :2].norm(dim=-1) < self.v_eps) & (cmd[:, 2].abs() < self.w_eps)
        slow = (x0[:, 9:11].norm(dim=-1) < self.v_settle) & (x0[:, 8].abs() < self.w_settle)
        latch = zero & ~self.active & slow
        self.active = zero & (self.active | latch)
        self.xy = torch.where(latch[:, None], x0[:, 3:5], self.xy)
        self.yaw = torch.where(latch, x0[:, 2], self.yaw)
        return self.active, self.xy, self.yaw

    def reset(self, env_ids):
        self.active[env_ids] = False


def command_world(x0, cmd, anchor_active):
    keep = (~anchor_active).to(x0.dtype)
    vx, vy = rot2(x0[:, 2], cmd[:, 0], cmd[:, 1])
    return torch.stack([vx, vy], -1) * keep[:, None], cmd[:, 2] * keep


def body_references(P, terrain, com_xy, yaw, h):
    """CoM height and pitch that follow the terrain under the four nominal feet (negative pitch = nose up)."""
    nom = torch.as_tensor(P.nominal, device=com_xy.device, dtype=com_xy.dtype)
    rx, ry = rot2(yaw[..., None], nom[:, 0], nom[:, 1])
    hz = getattr(terrain, "body_height", terrain.height)(com_xy[..., 0:1] + rx, com_xy[..., 1:2] + ry)
    span = float(sum(n[0] for n in P.nominal[:2]) / 2 - sum(n[0] for n in P.nominal[2:]) / 2)
    pitch = -torch.atan((hz[..., :2].mean(-1) - hz[..., 2:].mean(-1)) / span)
    return h[:, None] + hz.mean(-1), pitch


def build_horizon(P, terrain, gait, x0, cmd, feet_now, anchor, h, foot_res=None):
    """Reference trajectory (B,H,13), foot-minus-CoM vectors (B,H,4,3) and contact flags (B,H,4) for the QP."""
    from .footholds import foothold

    B, H, dt, dev, dty = x0.shape[0], P.H, P.dt, x0.device, x0.dtype
    active, a_xy, a_yaw = anchor
    yaw0 = x0[:, 2]
    v_w, wz = command_world(x0, cmd, active)
    keep = (~active).to(dty)
    k = torch.arange(H, device=dev, dtype=dty)
    tau = (k + 1.0) * dt
    pos_ref = (1 - keep)[:, None, None] * a_xy[:, None, :] + keep[:, None, None] * (x0[:, None, 3:5] + v_w[:, None, :] * tau[None, :, None])
    yaw_ref = (1 - keep)[:, None] * a_yaw[:, None] + keep[:, None] * (yaw0[:, None] + wz[:, None] * tau[None, :])
    zref, pref = body_references(P, terrain, pos_ref, yaw_ref, h)

    # vertical speed follows the slope analytically; differencing zref would give spikes at every riser
    # pitch is the slope along the body x axis, so only the body-x command climbs it (a sideways step follows the contour)
    vz = (cmd[:, 0] * keep)[:, None] * (-torch.tan(pref.clamp(-1.4, 1.4)))
    z0 = torch.zeros(B, H, device=dev, dtype=dty)
    x_ref = torch.stack([z0, pref, yaw_ref, pos_ref[..., 0], pos_ref[..., 1], zref, z0, z0, wz[:, None] + z0,
                         v_w[:, None, 0] + z0, v_w[:, None, 1] + z0, vz, z0 - GRAV], -1)

    ph = gait.phase_ahead(k * dt)
    contact = gait.contact(ph)
    stance_start = (k * dt)[None, :, None] - gait.stance_elapsed(ph)
    planted = (stance_start <= 1e-6) & gait.contact()[:, None, :]
    td = stance_start.clamp(min=0.0)
    com_td = x0[:, None, None, 3:5] + v_w[:, None, None, :] * td[..., None]
    yaw_td = yaw0[:, None, None] + wz[:, None, None] * td
    res = None if foot_res is None else foot_res[:, None]
    landing = foothold(P, terrain, com_td, yaw_td, x0[:, None, None, 9:11], v_w[:, None, None, :], gait.stance_duration[:, None, None], res)

    yaw_k = yaw0[:, None] + wz[:, None] * k[None, :] * dt
    com_xy = x0[:, None, 3:5] + v_w[:, None, :] * (k * dt)[None, :, None]
    com_k = torch.cat([com_xy, torch.cat([x0[:, 5:6], zref[:, :-1]], 1)[..., None]], -1)
    nom = torch.as_tensor(P.nominal, device=dev, dtype=dty)
    sx, sy = rot2(yaw_k[..., None], nom[:, 0], nom[:, 1])
    r_swing = torch.stack([sx, sy, torch.zeros_like(sx) - h[:, None, None]], -1)
    r_planted = feet_now[:, None] - com_k[:, :, None]
    r_land = landing - com_k[:, :, None]
    r = torch.where(contact[..., None], torch.where(planted[..., None], r_planted, r_land), r_swing)
    return x_ref, r, contact.to(dty)
