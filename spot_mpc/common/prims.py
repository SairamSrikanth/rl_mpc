import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Prim:
    """A static box (size = lx, ly, lz) or cylinder (size = radius, height, 0), world frame, quaternion w-x-y-z."""
    name: str
    kind: str
    pos: tuple
    size: tuple
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    color: tuple = (0.5, 0.5, 0.5)


def yaw_quat(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def rot_xy(lx, ly, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return lx * c - ly * s, lx * s + ly * c


def floats(*v):
    return tuple(float(x) for x in v)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


BOX_CORNERS = np.array([(-.5, -.5, -.5), (.5, -.5, -.5), (.5, .5, -.5), (-.5, .5, -.5), (-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)])
BOX_TRIS = np.array([(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)])


def mesh_arrays(prims, n_sides=16):
    """Vertices and triangles of every primitive plus a 200 x 200 m ground slab whose top is at z = 0 (cylinders become prisms)."""
    verts, tris = [], []

    def emit(v, t):
        tris.append(t + sum(len(a) for a in verts))
        verts.append(v)

    for p in prims:
        R = quat_to_mat(p.quat)
        if p.kind == "box":
            emit((BOX_CORNERS * np.array(p.size)) @ R.T + np.array(p.pos), BOX_TRIS)
            continue
        r, h = p.size[0], p.size[1]
        a = 2 * np.pi * np.arange(n_sides) / n_sides
        ring = np.stack([r * np.cos(a), r * np.sin(a)], -1)
        local = np.concatenate([np.c_[ring, np.full(n_sides, -h / 2)], np.c_[ring, np.full(n_sides, h / 2)], [[0, 0, -h / 2], [0, 0, h / 2]]])
        i = np.arange(n_sides)
        j = (i + 1) % n_sides
        t = np.concatenate([np.stack([i, j, n_sides + j], -1), np.stack([i, n_sides + j, n_sides + i], -1),
                            np.stack([np.full(n_sides, 2 * n_sides), j, i], -1), np.stack([np.full(n_sides, 2 * n_sides + 1), n_sides + i, n_sides + j], -1)])
        emit(local @ R.T + np.array(p.pos), t)
    emit(BOX_CORNERS * np.array([200.0, 200.0, 0.1]) + np.array([0.0, 0.0, -0.05]), BOX_TRIS)
    return np.concatenate(verts).astype(np.float32), np.concatenate(tris).astype(np.int32)


def build_ray_mesh(prims, path):
    """One invisible, collision-free mesh of the whole world: the single static mesh the ray casters trace."""
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, Vt
    v, t = mesh_arrays(prims)
    mesh = UsdGeom.Mesh.Define(omni.usd.get_context().get_stage(), Sdf.Path(path))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in v]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(t.reshape(-1).tolist()))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(t)))
    mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)


def spawn_colliders(prims, root, friction, extra=()):
    """Ground plane plus one PhysX collider per primitive under `root`; `extra` are colliders that are not in the ray mesh or in surface() (the rails)."""
    import isaaclab.sim as sim_utils
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg(physics_material=mat))
    col = sim_utils.CollisionPropertiesCfg()
    for p in [*prims, *extra]:
        vis = sim_utils.PreviewSurfaceCfg(diffuse_color=p.color)
        if p.kind == "box":
            cfg = sim_utils.CuboidCfg(size=p.size, collision_props=col, physics_material=mat, visual_material=vis)
        else:
            cfg = sim_utils.CylinderCfg(radius=p.size[0], height=p.size[1], axis="Z", collision_props=col, physics_material=mat, visual_material=vis)
        cfg.func(f"{root}/{p.name}", cfg, translation=p.pos, orientation=p.quat)
