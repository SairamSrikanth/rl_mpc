from dataclasses import dataclass, field

import numpy as np
import torch

NX, NU = 13, 12
BIG = 1e9


@dataclass
class MpcParams:
    dt: float = 0.025
    horizon: int = 16
    mass: float = 31.6
    inertia_b: np.ndarray = field(default_factory=lambda: np.array([[0.583, 0.0, -0.0781], [0.0, 2.2076, 0.0], [-0.0781, 0.0, 2.3596]]))
    mu: float = 0.6
    fmax: float = 220.0
    # state weights: rpy, pos, omega, vel, gravity slot
    q_diag: np.ndarray = field(default_factory=lambda: np.array([40, 40, 10, 100, 100, 400, 0.1, 0.1, 0.3, 2, 2, 2, 0.0], dtype=float))
    r_alpha: float = 1e-4
    r_du: float = 5e-3


def _eye(n, ref):
    return torch.eye(n, dtype=ref.dtype, device=ref.device)


def _const(arr, ref):
    return torch.as_tensor(np.asarray(arr), dtype=ref.dtype, device=ref.device)


def linear_models(yaw, r, mass, ib_inv, dt):
    """Discrete SRB model about the reference yaw (B,H) with foot-minus-CoM vectors r (B,H,4,3)."""
    B, H = yaw.shape
    c, s = torch.cos(yaw), torch.sin(yaw)
    o, z = torch.ones_like(c), torch.zeros_like(c)
    rz = torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)
    rzt = rz.transpose(-1, -2)
    iinv = torch.einsum("bhij,jk,bhlk->bhil", rz, ib_inv, rz)
    x, y, zz = r[..., 0], r[..., 1], r[..., 2]
    zero = torch.zeros_like(x)
    skew = torch.stack([torch.stack([zero, -zz, y], -1), torch.stack([zz, zero, -x], -1), torch.stack([-y, x, zero], -1)], -2)
    bw = torch.einsum("bhij,bhfjk->bhfik", iinv, skew)
    e3 = _eye(3, yaw)

    ad = yaw.new_zeros((B, H, NX, NX)) + _eye(NX, yaw)
    ad[..., 0:3, 6:9] = dt * rzt
    ad[..., 3:6, 9:12] = dt * e3
    ad[..., 11, 12] = dt
    ad[..., 5, 12] = 0.5 * dt * dt
    bd = yaw.new_zeros((B, H, NX, NU))
    rzt_bw = torch.einsum("bhij,bhfjk->bhfik", rzt, bw)
    for f in range(4):
        cols = slice(3 * f, 3 * f + 3)
        bd[..., 6:9, cols] = dt * bw[:, :, f]
        bd[..., 9:12, cols] = (dt / mass) * e3
        bd[..., 0:3, cols] = 0.5 * dt * dt * rzt_bw[:, :, f]
        bd[..., 3:6, cols] = (0.5 * dt * dt / mass) * e3
    return ad, bd, rzt


def condense(x0, ad, bd, accel_resid=None, rzt=None, dt=None):
    """Stack the horizon: X = free + Bq U. Returns free (B,H*13) and Bq (B,H*13,H*12)."""
    B, H = ad.shape[:2]
    bq = x0.new_zeros((B, H, NX, H, NU))
    aprev = x0.new_zeros((B, NX, NX)) + _eye(NX, x0)
    free = []
    for k in range(H):
        aprev = ad[:, k] @ aprev
        free.append((aprev @ x0[..., None])[..., 0])
        if k > 0:
            bq[:, k, :, :k, :] = torch.einsum("bij,bjmn->bimn", ad[:, k], bq[:, k - 1, :, :k, :])
        bq[:, k, :, k, :] = bd[:, k]
    free = torch.stack(free, 1)
    if accel_resid is not None:
        # constant [linear acc, angular acc] residual added at every step
        acc = x0.new_zeros((B, NX))
        for k in range(H):
            d = x0.new_zeros((B, NX))
            d[:, 9:12] = dt * accel_resid[:, 0:3]
            d[:, 6:9] = dt * accel_resid[:, 3:6]
            d[:, 3:6] = 0.5 * dt * dt * accel_resid[:, 0:3]
            d[:, 0:3] = 0.5 * dt * dt * torch.einsum("bij,bj->bi", rzt[:, k], accel_resid[:, 3:6])
            acc = torch.einsum("bij,bj->bi", ad[:, k], acc) + d
            free[:, k] = free[:, k] + acc
    return free.reshape(B, H * NX), bq.reshape(B, H * NX, H * NU)


def rate_matrix(H):
    n = H * NU
    d = np.eye(n) - np.eye(n, k=-NU)
    return d.T @ d


def assemble(free, bq, x_ref, q_diag, r_alpha, r_du, u_prev, H):
    """QP matrices P (B,n,n) and q (B,n) for min 1/2 u'Pu + q'u."""
    B = free.shape[0]
    n = H * NU
    lq = torch.stack([q_diag] * H, 1).reshape(B, H * NX)
    err = free - x_ref.reshape(B, H * NX)
    bqt = bq.transpose(1, 2)
    p = 2.0 * bqt @ (bq * lq[:, :, None]) + 2.0 * r_alpha * _eye(n, free)
    q = 2.0 * (bqt @ (lq * err)[..., None])[..., 0]
    if r_du > 0.0 and u_prev is not None:
        p = p + 2.0 * r_du * _const(rate_matrix(H), free)
        q = torch.cat([q[:, :NU] - 2.0 * r_du * u_prev, q[:, NU:]], 1)
    return p, q


def contact_frames(normals):
    """(B,4,3,3) rows t1, t2, n of each foot's contact frame from unit surface normals (B,4,3); n = z gives the world axes."""
    ey = torch.zeros_like(normals)
    ey[..., 1] = 1.0
    t1 = torch.linalg.cross(ey, normals)
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.stack([t1, torch.linalg.cross(normals, t1), normals], -2)


def constraint_matrix(torque_map, mu, ref, B, normals=None):
    """(B,4,7,3): four friction-pyramid rows (about the surface normal if given, else world z), then three joint-torque rows per foot."""
    g = ref.new_zeros((B, 4, 7, 3))
    if normals is None:
        pyramid = [[1, 0, -mu], [-1, 0, -mu], [0, 1, -mu], [0, -1, -mu]]
        g[:, :, 0:4, :] = _const(pyramid, ref)
    else:
        t1, t2, n = contact_frames(normals).unbind(-2)
        g[:, :, 0:4, :] = torch.stack([t1 - mu * n, -t1 - mu * n, t2 - mu * n, -t2 - mu * n], -2)
    if torque_map is not None:
        g[:, :, 4:7, :] = torque_map
    return g


def bounds(contact, fmax, tau_lim, use_torque):
    """Lower/upper bounds (B,H,4,10): 3 force rows, 4 friction rows, 3 torque rows. Swing feet are pinned to zero force."""
    B, H = contact.shape[:2]
    lo = contact.new_zeros((B, H, 4, 10))
    hi = contact.new_zeros((B, H, 4, 10))
    c = contact
    lo[..., 0], hi[..., 0] = -BIG * c, BIG * c
    lo[..., 1], hi[..., 1] = -BIG * c, BIG * c
    hi[..., 2] = fmax * c
    lo[..., 3:7] = -BIG
    t = tau_lim if use_torque else np.full(3, BIG)
    lo[..., 7:10] = -_const(t, contact)
    hi[..., 7:10] = _const(t, contact)
    return lo, hi


def admm(p, q, g, lo, hi, iters, rho, sigma=1e-6, alpha=1.6, state=None):
    """OSQP-style ADMM with a fixed iteration count. Returns the iterate, the full (x, z, y) state and the primal residual."""
    B, n = q.shape
    H = lo.shape[1]
    m = H * 4
    gtg = torch.einsum("bfkj,bfkl->bfjl", g, g)
    blocks = torch.stack([_eye(3, q) + gtg] * H, 1).reshape(B, m, 3, 3)
    blockdiag = torch.einsum("bmij,mn->bminj", blocks, _eye(m, q)).reshape(B, n, n)
    kinv = torch.linalg.inv(p + sigma * _eye(n, q) + rho * blockdiag)
    if state is None:
        x = q.new_zeros((B, n))
        z = q.new_zeros((B, H, 4, 10))
        y = q.new_zeros((B, H, 4, 10))
    else:
        x, z, y = state
    for _ in range(iters):
        v = rho * z - y
        atv = v[..., 0:3] + torch.einsum("bfkj,bhfk->bhfj", g, v[..., 3:10])
        xt = (kinv @ (sigma * x - q + atv.reshape(B, n))[..., None])[..., 0]
        ft = xt.reshape(B, H, 4, 3)
        zt = torch.cat([ft, torch.einsum("bfkj,bhfj->bhfk", g, ft)], -1)
        zr = alpha * zt + (1.0 - alpha) * z
        x = alpha * xt + (1.0 - alpha) * x
        zn = torch.minimum(torch.maximum(zr + y / rho, lo), hi)
        y = y + rho * (zr - zn)
        z = zn
    f = x.reshape(B, H, 4, 3)
    ax = torch.cat([f, torch.einsum("bfkj,bhfj->bhfk", g, f)], -1)
    resid = (ax - z).abs().reshape(B, -1).amax(1)
    return x, (x, z, y), resid


def shift_state(state):
    """Warm start for the next tick: everything moves one step earlier, the last step is repeated."""
    x, z, y = state
    B, H = x.shape[0], z.shape[1]
    xs = x.reshape(B, H, NU)
    xs = torch.cat([xs[:, 1:], xs[:, -1:]], 1).reshape(B, H * NU)
    return xs, torch.cat([z[:, 1:], z[:, -1:]], 1), torch.cat([y[:, 1:], y[:, -1:]], 1)


def project_first_step(u0, contact0, mu, fmax, torque_map, tau_lim, normals=None):
    """Make the first-step force exactly feasible: swing feet zero, 0 <= Fz <= fmax, friction pyramid,
    then scale a foot's force down if a joint torque would exceed its limit. With normals, Fz and the pyramid are in the contact frame."""
    B = u0.shape[0]
    f = u0.reshape(B, 4, 3) * contact0[..., None]
    frames = None if normals is None else contact_frames(normals)
    if frames is not None:
        f = torch.einsum("bfij,bfj->bfi", frames, f)
    fz = torch.minimum(torch.clamp(f[..., 2], min=0.0), fmax * contact0)
    lim = mu * fz
    fx = torch.minimum(torch.maximum(f[..., 0], -lim), lim)
    fy = torch.minimum(torch.maximum(f[..., 1], -lim), lim)
    f = torch.stack([fx, fy, fz], -1)
    if frames is not None:
        f = torch.einsum("bfji,bfj->bfi", frames, f)
    if torque_map is not None and tau_lim is not None:
        tq = torch.einsum("bfjk,bfk->bfj", torque_map, f).abs()
        worst = (tq / _const(tau_lim, f)).amax(-1)
        scale = torch.clamp(1.0 / worst.clamp(min=1e-12), max=1.0)
        f = f * scale[..., None]
    return f


class BatchedSrbdMpc:
    """Solves one MPC problem per environment at once and keeps the ADMM warm start between calls."""

    def __init__(self, params, iters=40, rho=0.1, sigma=1e-6, alpha=1.6, dtype=torch.float32, device="cpu"):
        self.p = params
        self.iters, self.rho, self.sigma, self.alpha = iters, rho, sigma, alpha
        self.H = params.horizon
        self.dtype, self.device = dtype, device
        self.state = None

    def _t(self, arr):
        return torch.as_tensor(arr, dtype=self.dtype, device=self.device)

    def reset(self, mask=None):
        if self.state is None or mask is None:
            self.state = None
            return
        drop = self._t(mask) > 0.5
        x, z, y = self.state
        # assign zeros instead of multiplying: NaN * 0 would keep a poisoned warm start alive
        self.state = (torch.where(drop[:, None], torch.zeros_like(x), x),
                      torch.where(drop[:, None, None, None], torch.zeros_like(z), z),
                      torch.where(drop[:, None, None, None], torch.zeros_like(y), y))

    def solve(self, x0, x_ref, r, contact, u_prev=None, torque_map=None, tau_lim=None, q_diag=None, accel_resid=None, shift=True, normals=None):
        """x0 (B,13), x_ref (B,H,13), r (B,H,4,3), contact (B,H,4), torque_map (B,4,3,3), accel_resid (B,6) in the body frame,
        normals (B,4,3) unit surface normals under the feet (None: world-z friction pyramid).
        Returns the feasible first-step forces (B,4,3) and a dict with the raw iterate and residual."""
        p = self.p
        x0, x_ref, r = self._t(x0), self._t(x_ref), self._t(r)
        B, H = x0.shape[0], self.H
        c = self._t(contact)
        ib_inv = _const(np.linalg.inv(p.inertia_b), x0)
        ad, bd, rzt = linear_models(x_ref[..., 2], r, p.mass, ib_inv, p.dt)
        ar = None
        if accel_resid is not None:
            a = self._t(accel_resid)
            cy, sy = torch.cos(x0[:, 2]), torch.sin(x0[:, 2])
            ar = torch.stack([cy * a[:, 0] - sy * a[:, 1], sy * a[:, 0] + cy * a[:, 1], a[:, 2],
                              cy * a[:, 3] - sy * a[:, 4], sy * a[:, 3] + cy * a[:, 4], a[:, 5]], 1)
        free, bq = condense(x0, ad, bd, ar, rzt, p.dt)
        qd = _const(p.q_diag, x0)[None, :] + x0.new_zeros((B, NX)) if q_diag is None else self._t(q_diag)
        up = None if u_prev is None else self._t(u_prev)
        pm, q = assemble(free, bq, x_ref, qd, p.r_alpha, p.r_du, up, H)
        tm = None if torque_map is None else self._t(torque_map)
        use_t = tm is not None and tau_lim is not None
        nrm = None if normals is None else self._t(normals)
        g = constraint_matrix(tm if use_t else None, p.mu, x0, B, nrm)
        lo, hi = bounds(c, p.fmax, tau_lim if use_t else None, use_t)
        st = self.state if (self.state is not None and self.state[0].shape[0] == B) else None
        if st is not None:
            # a non-finite warm start is dropped for that problem only
            bad = ~(torch.isfinite(st[0]).all(-1) & torch.isfinite(st[1]).reshape(B, -1).all(-1) & torch.isfinite(st[2]).reshape(B, -1).all(-1))
            st = (torch.where(bad[:, None], torch.zeros_like(st[0]), st[0]),
                  torch.where(bad[:, None, None, None], torch.zeros_like(st[1]), st[1]),
                  torch.where(bad[:, None, None, None], torch.zeros_like(st[2]), st[2]))
        x, state, resid = admm(pm, q, g, lo, hi, self.iters, self.rho, self.sigma, self.alpha, st)
        self.state = shift_state(state) if shift else state
        f0 = project_first_step(x[:, :NU], c[:, 0], p.mu, p.fmax, tm if use_t else None, tau_lim if use_t else None, nrm)
        return f0, dict(x=x, resid=resid, P=pm, q=q, state=state)
