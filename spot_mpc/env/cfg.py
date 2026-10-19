import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ..control.torque_interface import GRAVITY, make_spot_torque_cfg
from ..worlds.generated.geometry import GeneratedWorldCfg
from ..worlds.test_world import World, WorldCfg
from ..worlds.warehouse_static.layout import WarehouseCfg

PHYSICS_DT, DECIMATION = 1.0 / 400.0, 10
ACT_FREQ, ACT_ACCEL, ACT_FOOT, ACT_APEX, ACT_META = slice(0, 1), slice(1, 7), slice(7, 15), slice(15, 19), slice(19, 23)
NUM_ACTIONS = 23     # 19 residuals plus 4 tuning outputs: speed scale, velocity weight, duty offset, pitch weight

@configclass
class SpotMpcEnvCfg(DirectRLEnvCfg):
    decimation = DECIMATION
    episode_length_s = 120.0
    action_space = NUM_ACTIONS
    observation_space = 1
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=PHYSICS_DT, render_interval=DECIMATION, gravity=(0.0, 0.0, -GRAVITY),
                                       physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=True)
    robot = make_spot_torque_cfg().replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=3)

    # world: "test", "warehouse" or "generated"
    world_type: str = "test"
    world: WorldCfg = WorldCfg()
    generated: GeneratedWorldCfg = GeneratedWorldCfg()
    warehouse: WarehouseCfg = WarehouseCfg()

    # perception: "exact" plans on the analytic world, "sensed" on the elevation map
    terrain_source: str = "exact"
    map_half_x: float = 4.0
    map_half_y: float = 3.0
    map_res: float = 0.03
    map_flat_tol: float = 0.02
    map_ema_alpha: float = 0.10
    edge_tol: float = 0.02
    upper_fill: bool = True
    upper_reach: float = 1.0
    stair_prior: bool = True
    ceiling_exclude: float = 1.2
    sensor_noise_std: float = 0.0
    load_threshold: float = 15.0     # foot force [N] above which a foot counts as loaded
    depth_pitch_deg: float = 25.0    # the depth camera looks down so it can see the treads below a descending edge
    rear_depth: bool = False         # second depth camera at the back, for walking backward
    rear_depth_pitches: tuple = (25.0,)
    lidar: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body", update_period=PHYSICS_DT * DECIMATION,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)), ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(channels=32, vertical_fov_range=(-15.0, 15.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=2.0),
        max_distance=15.0, mesh_prim_paths=[World.RAY_MESH_PRIM])
    depth_camera: RayCasterCameraCfg = RayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body", update_period=PHYSICS_DT * DECIMATION,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.40, 0.0, 0.08), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        data_types=["distance_to_image_plane"],
        pattern_cfg=patterns.PinholeCameraPatternCfg(focal_length=5.0, horizontal_aperture=10.0, height=64, width=96),
        max_distance=8.0, mesh_prim_paths=[World.RAY_MESH_PRIM])

    # MPC
    mpc_horizon: int = 16
    mpc_iters: int = 30
    mpc_rho: float = 0.003
    mu: float = 0.6
    fmax: float = 220.0
    q_pos_xy: float = 100.0
    q_vel_xy: float = 200.0          # velocity weight on flat ground
    q_pitch_down: float = 40.0
    terrain_cone: bool = False       # friction pyramid about the sensed surface normal instead of world z (not yet benchmarked)
    contact_aware: bool = False      # a foot the clock puts in stance before it is loaded gets no MPC force and keeps reaching down (not yet benchmarked)
    touchdown_probe: float = 0.03    # how far below the planned touchdown point an unloaded stance foot reaches [m]

    # gait and swing (the flat-ground operating point)
    gait_freq: float = 2.0
    gait_offsets: tuple = (0.0, 0.5, 0.5, 0.0)
    gait_duty: float = 0.6
    freq_range: float = 1.0
    raibert_kd: float = 0.15
    foot_margin: float = 0.08
    swing_clearance: float = 0.06
    swing_base_apex: float = 0.08
    swing_delay: float = 0.2
    swing_kp: float = 600.0
    swing_kd: float = 30.0
    com_height: float = 0.45
    root_height: float = 0.52
    foot_z_offset: float = 0.035
    pd_kp: float = 250.0
    pd_kd: float = 5.0
    t_blend: float = 0.5             # joint PD hands over to the MPC over this long
    t_start: float = 1.0             # then the robot stands this long before it trots
    cmd_accel: float = 0.5

    # ascent operating point, blended in by the previewed ascent (found by CMA-ES)
    stair_speed: float = 0.4971
    q_vel_xy_steep: float = 3.407
    slope_crouch: float = 0.09903
    slope_crouch_pitch: float = 0.30
    gait_freq_up: float = 2.01
    gait_duty_up: float = 0.5626
    swing_clearance_up: float = 0.06454
    swing_base_apex_up: float = 0.1058
    foot_margin_up: float = 0.0562
    swing_kp_up: float = 679.8
    swing_kd_up: float = 38.71

    # backward-descent operating point, blended in by the previewed backward descent (found by CMA-ES)
    stair_speed_back: float = 0.3579
    q_vel_xy_back: float = 80.0
    gait_duty_back: float = 0.60
    gait_freq_back: float = 1.978
    swing_clearance_back: float = 0.03572
    swing_base_apex_back: float = 0.07012
    foot_margin_back: float = 0.05003
    q_pitch_back: float = 23.92
    descent_crouch_back: float = 0.01499
    swing_kp_back: float = 405.8
    swing_kd_back: float = 39.55

    # forward descent (only used when the robot walks down forwards)
    stair_speed_down: float = 0.40
    q_vel_xy_down: float = 16.0
    gait_duty_down: float = 0.65
    descent_crouch: float = 0.0
    steep_pitch: float = 0.15        # previewed pitch at which the operating point is fully 'steep'
    duty_tau: float = 0.0125

    # lane keeping, used when env.heading_target / env.lane_y are set
    k_yaw: float = 1.0
    k_lane: float = 0.5
    wz_max: float = 0.4
    vy_max: float = 0.15

    # residual policy
    accel_lin_scale: float = 1.0
    accel_ang_scale: float = 2.0
    accel_rate: float = 0.5
    foot_scale: float = 0.05
    foot_rate: float = 0.01
    apex_scale: float = 0.05
    apex_rate: float = 0.01
    meta_speed: float = 0.35         # speed scale = 1 + meta_speed * a
    meta_q: float = 1.0              # weights are multiplied by exp(meta_q * a)
    meta_duty: float = 0.05
    shield: bool = False             # fade the residual and slow down as the body nears a tip-over
    shield_tilt: tuple = (0.30, 0.60)
    shield_over: tuple = (0.20, 0.50)
    shield_slow: float = 0.6

    fall_tilt: float = 1.0
