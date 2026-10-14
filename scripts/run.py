import argparse, functools, math, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description="Run the MPC (zero residual unless --checkpoint) on a test-world scenario and print per-robot results.")
ap.add_argument("--scenario", default="flat", choices=["flat", "stairs", "descent", "ramp_down", "traverse", "ramp", "rubble"])
ap.add_argument("--num_envs", type=int, default=8)
ap.add_argument("--t", type=float, default=None, help="seconds to run (default depends on the scenario)")
ap.add_argument("--speed", type=float, default=0.45)
ap.add_argument("--checkpoint", default=None, help="PPO policy that drives the residuals")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--backward", action="store_true", help="walk backward (needs --set rear_depth=True with the sensed map)")
ap.add_argument("--lane_keep", action="store_true", help="hold heading and lane line (env.heading_target / lane_y)")
ap.add_argument("--pose_y", type=float, default=0.2, help="lateral start offset range (+-m)")
ap.add_argument("--pose_yaw", type=float, default=0.1, help="start yaw range (+-rad)")
ap.add_argument("--video", default=None)
ap.add_argument("--set", nargs="*", action="extend", default=[], help="config overrides KEY=VALUE")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
DEFAULT_T = {"flat": 15.0, "stairs": 60.0, "descent": 50.0, "ramp_down": 40.0, "traverse": 90.0, "ramp": 45.0, "rubble": 45.0}
args.t = args.t or DEFAULT_T[args.scenario]
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app
sys.path.insert(0, str(ROOT))
import torch
from spot_mpc.cli import apply_overrides, load_policy
from spot_mpc.tasks.task import SpotTask, SpotTaskCfg


def start_poses(scenario, n, c, world):
    g = torch.Generator().manual_seed(args.seed)
    noise = lambda: (torch.rand(n, generator=g) - 0.5) * 2
    if scenario == "flat":
        yaw = torch.linspace(-math.pi, math.pi, n + 1)[:-1]
        return torch.stack([torch.full((n,), -8.0), torch.zeros(n), yaw], 1)
    lane_y = {"stairs": c.stair_y, "traverse": c.stair_y, "descent": c.stair_y, "ramp_down": c.ramp_y, "ramp": c.ramp_y, "rubble": c.rubble_y}[scenario]
    run_up = 2.0
    if scenario == "descent":
        x0 = c.stair_x + c.n_steps * c.run + c.landing - run_up
    elif scenario == "ramp_down":
        x0 = world.ramp_down_x - run_up
    else:
        x0 = c.stair_x - run_up
    yaw = args.pose_yaw * noise() + (math.pi if args.backward else 0.0)
    return torch.stack([torch.full((n,), x0), lane_y + args.pose_y * noise(), yaw], 1)


def reached_end(env, x, z0):
    """Per robot: has it finished the scenario (top of the stairs, bottom of a descent)?"""
    w = env.world
    if args.scenario in ("descent", "traverse"):
        return x[:, 3] > w.bottom_x + 0.2
    if args.scenario == "ramp_down":
        return x[:, 3] > w.ramp_bottom_x + 0.3
    if args.scenario == "stairs":
        return x[:, 5] - z0 > 0.95 * w.stairs_top
    return None


def main():
    B = args.num_envs
    cfg = SpotTaskCfg()
    cfg.scene.num_envs = B
    cfg.episode_length_s = args.t + 30
    apply_overrides(cfg, args.set)
    if args.video:
        from isaaclab.envs.common import ViewerCfg
        cfg.viewer = ViewerCfg(eye=(-2.4, -2.4, 1.5), lookat=(0.8, 0.0, 0.1), origin_type="asset_root", asset_name="robot", env_index=0, resolution=(960, 540))
    env = SpotTask(cfg, render_mode="rgb_array" if args.video else None)
    dev = env.device
    env.fixed_spawn = True
    policy = load_policy(env, args.checkpoint) if args.checkpoint else None
    env.spawn_pose = start_poses(args.scenario, B, cfg.world, env.world).to(dev)
    with torch.inference_mode():
        obs, _ = env.reset()
    env.cmd_target[:, 0] = -args.speed if args.backward else args.speed
    if args.lane_keep:
        c = cfg.world
        lane = {"stairs": c.stair_y, "traverse": c.stair_y, "descent": c.stair_y, "ramp_down": c.ramp_y, "ramp": c.ramp_y, "rubble": c.rubble_y, "flat": 0.0}[args.scenario]
        env.heading_target = torch.full((B,), math.pi if args.backward else 0.0, device=dev)
        env.lane_y = torch.full((B,), lane, device=dev)
        if args.scenario == "flat":
            env.heading_target, env.lane_y = env.spawn_pose[:, 2].clone(), None

    writer = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.video, fps=40, codec="libx264", quality=8, macro_block_size=None)
        for _ in range(3):
            env.render()

    alive = torch.ones(B, dtype=torch.bool, device=dev)
    t_fall = torch.full((B,), float("nan"), device=dev)
    t_end = torch.full((B,), float("nan"), device=dev)
    best_dz = torch.zeros(B, device=dev)
    zero = torch.zeros(B, cfg.action_space, device=dev)
    start = z0 = last_xy = None
    for k in range(int(args.t / env.step_dt)):
        with torch.inference_mode():
            obs, _, term, _, _ = env.step(zero if policy is None else policy(obs))
        x = env.x0
        if start is None:
            start, z0, last_xy = x[:, 3:5].clone(), x[:, 5].clone(), x[:, 3:5].clone()
        t = k * env.step_dt
        t_fall[term & alive] = t
        alive &= ~term
        if not bool(alive.any()) and not writer:
            break
        best_dz = torch.where(alive, torch.maximum(best_dz, x[:, 5] - z0), best_dz)
        end = reached_end(env, x, z0)
        if end is not None:
            t_end[alive & end & torch.isnan(t_end)] = t
        last_xy = torch.where(alive[:, None], x[:, 3:5], last_xy)
        if writer:
            writer.append_data(env.render())
            if bool(term[0]) or (not torch.isnan(t_end[0]) and t > float(t_end[0]) + 4.0):
                break
        if k % int(5 / env.step_dt) == 0:
            print(f"[run] t={t:5.1f}s alive {int(alive.sum())}/{B} median dz {float((x[:, 5] - z0).median()):+.2f}")

    dist = (last_xy - start).norm(dim=-1)
    print(f"[result] scenario={args.scenario} speed={args.speed} t={args.t}s: survived {int(alive.sum())}/{B}")
    for i in range(B):
        print(f"[result]   env {i}: {'ok  ' if bool(alive[i]) else 'FELL'} t_fall={float(t_fall[i]):5.1f}s dist={float(dist[i]):5.2f}m "
              f"max_dz={float(best_dz[i]):+.2f}m t_end={float(t_end[i]):5.1f}s")
    if writer:
        writer.close()
        print(f"[video] {args.video}")
    os._exit(0)


if __name__ == "__main__":
    main()
