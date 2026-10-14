import numpy as np

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from .geometry import DEFAULT_PROPORTIONS, GOAL_MARGIN, X_SPAWN, GeneratedWorldCfg, column_families, generate_tile


def _tile_function(difficulty, cfg):
    return generate_tile(cfg.family, difficulty, cfg.size)


@configclass
class GenTileCfg(SubTerrainBaseCfg):
    function = _tile_function
    family: str = "flat"


def make_generator_cfg(c: GeneratedWorldCfg) -> TerrainGeneratorCfg:
    props = c.proportions or DEFAULT_PROPORTIONS
    return TerrainGeneratorCfg(
        size=c.tile, border_width=c.border, num_rows=c.num_rows, num_cols=c.num_cols, curriculum=True, difficulty_range=(0.0, 1.0),
        horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75, use_cache=False, color_scheme="none", seed=c.seed,
        sub_terrains={name: GenTileCfg(proportion=p, family=name) for name, p in props.items()})


class GeneratedWorld:
    """The generated terrain as an env world: the terrain prim is both the collider and the ray-cast mesh."""

    RAY_MESH_PRIM = "/World/ground"
    collision_roots = ["/World/ground"]

    def __init__(self, cfg: GeneratedWorldCfg, device, num_envs, env_spacing=1.0):
        self.cfg, self.device, self.num_envs, self.env_spacing = cfg, device, num_envs, env_spacing
        self.families = column_families(cfg)
        self.importer = None

    def spawn(self):
        import isaaclab.sim as sim_utils
        from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
        c = self.cfg
        np.random.seed(c.seed)       # the tile geometry draws from numpy's global state, Isaac Lab only seeds its own generator
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=c.friction, dynamic_friction=c.friction, restitution=0.0)
        icfg = TerrainImporterCfg(prim_path="/World/ground", terrain_type="generator", terrain_generator=make_generator_cfg(c),
                                  num_envs=self.num_envs, env_spacing=self.env_spacing, max_init_terrain_level=c.init_level,
                                  collision_group=-1, physics_material=mat)
        self.importer = TerrainImporter(icfg)

    def spawn_ray_mesh(self):
        pass

    def to(self, device):
        self.device = device

    def surface(self, x, y, z_ref=None):
        raise NotImplementedError("no analytic surface on generated terrain, use env.floor_height()")

    @property
    def goal_dx(self):
        return self.cfg.tile[0] - GOAL_MARGIN - X_SPAWN
