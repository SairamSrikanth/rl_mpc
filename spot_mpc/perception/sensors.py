import torch

import isaaclab.utils.math as math_utils


def lidar_points(lidar):
    """World-frame LiDAR hits (B,K,3) and a validity mask (missed rays are inf or nan)."""
    hits = lidar.data.ray_hits_w
    return hits, torch.isfinite(hits).all(-1)


def depth_points(camera, data_type="distance_to_image_plane"):
    """Unproject a depth image to world points (B,H*W,3), row-major with pixel-centre offsets like the ray pattern, plus a validity mask."""
    d = camera.data.output[data_type][..., 0]
    B, H, W = d.shape
    K = camera.data.intrinsic_matrices
    v, u = torch.meshgrid(torch.arange(H, device=d.device, dtype=d.dtype), torch.arange(W, device=d.device, dtype=d.dtype), indexing="ij")
    x = (u.reshape(1, -1) + 0.5 - K[:, 0, 2:3]) / K[:, 0, 0:1]
    y = (v.reshape(1, -1) + 0.5 - K[:, 1, 2:3]) / K[:, 1, 1:2]
    dd = d.reshape(B, -1)
    pts_c = torch.stack([x * dd, y * dd, dd], -1)
    valid = (dd > 1e-3) & (dd < 0.999 * camera.cfg.max_distance) & torch.isfinite(dd)
    return math_utils.transform_points(pts_c, camera.data.pos_w, camera.data.quat_w_ros), valid
