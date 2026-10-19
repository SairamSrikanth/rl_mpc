import torch

from .gait import rot2


def safe_foothold(height, xy, margin=0.08, reach=0.08, flat_tol=0.02):
    """Move a foothold to the nearest safe spot among a 5x5 grid of shifts within +-reach.

    A candidate is penalised when the terrain around it (a 5-point stencil of half-width margin, and a wider one at 1.75x)
    deviates from a local plane, so uniform slopes cost nothing but risers repel the foot.
    """
    dev, dt = xy.device, xy.dtype
    sh = torch.linspace(-reach, reach, 5, device=dev, dtype=dt)
    offs = torch.stack(torch.meshgrid(sh, sh, indexing="xy"), -1).reshape(-1, 2)
    unit = torch.tensor([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1.75, 0], [-1.75, 0], [0, 1.75], [0, -1.75]], device=dev, dtype=dt)
    m = margin if isinstance(margin, (int, float)) else margin.reshape((-1,) + (1,) * (xy.ndim - 2))[..., None, None, None]
    cand = xy[..., None, :] + offs
    pts = cand[..., :, None, :] + m * unit
    hs = height(pts[..., 0], pts[..., 1])
    curv = lambda a, b: (hs[..., a] + hs[..., b] - 2 * hs[..., 0]).abs()
    near = torch.maximum(curv(1, 2), curv(3, 4))
    wide = torch.maximum(near, torch.maximum(curv(5, 6), curv(7, 8)))
    cost = (offs ** 2).sum(-1) + torch.where(near <= flat_tol, 0.0, 1e3 + 1e3 * near) + torch.where(wide <= flat_tol, 0.0, 1.0 + wide)
    best = cost.argmin(-1)
    return torch.gather(cand, -2, best[..., None, None].expand(*best.shape, 1, 2)).squeeze(-2)


def foothold(P, terrain, com_xy, yaw, v_xy, v_des, stance_T, residual=None):
    """Raibert foothold plus an optional body-frame residual, moved off edges and placed at the terrain height."""
    nom = torch.as_tensor(P.nominal, device=com_xy.device, dtype=com_xy.dtype)
    nom_x, nom_y = rot2(yaw, nom[:, 0], nom[:, 1])
    corr = 0.5 * stance_T[..., None] * v_xy + P.kd * (v_xy - v_des)
    n = corr.norm(dim=-1, keepdim=True)
    corr = corr * torch.where(n > P.max_dev, P.max_dev / n.clamp(min=1e-12), torch.ones_like(n))
    fxy = com_xy + torch.stack([nom_x, nom_y], -1) + corr
    if residual is not None:
        rx, ry = rot2(yaw, residual[..., 0], residual[..., 1])
        fxy = fxy + torch.stack([rx, ry], -1)
    fxy = terrain.project(fxy, P.margin)
    return torch.cat([fxy, terrain.height(fxy[..., 0], fxy[..., 1])[..., None]], -1)


def surface_normals(height, x, y, h=0.03):
    """Unit surface normals (...,3) from one-sided height differences. Per axis the flatter side wins, so a riser next to the
    foot is ignored while a uniform slope (equal on both sides) is kept."""
    def slope(dx, dy):
        z = height(x, y)
        fwd, bwd = (height(x + dx, y + dy) - z) / h, (z - height(x - dx, y - dy)) / h
        return torch.where(fwd.abs() < bwd.abs(), fwd, bwd)
    gx, gy = slope(h, 0.0), slope(0.0, h)
    n = torch.stack([-gx, -gy, torch.ones_like(gx)], -1)
    return n / n.norm(dim=-1, keepdim=True)
