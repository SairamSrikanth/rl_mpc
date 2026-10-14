import argparse, functools, math, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description="Run the sensed MPC through the two-floor warehouse, one route segment (or the whole loop) at a time.")
ap.add_argument("--segment", default="stairs_up", help="stairs_up, upper_cross, ramp_down, ground_turn, ramp_up, upper_return, stairs_down_back or loop_back")
ap.add_argument("--num_envs", type=int, default=64)
ap.add_argument("--t", type=float, default=None, help="seconds to run (default depends on the segment)")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--speed", type=float, default=0.45)
ap.add_argument("--turn_rate", type=float, default=0.8)
ap.add_argument("--jitter_xy", type=float, default=0.3, help="spawn position jitter (+-m)")
ap.add_argument("--jitter_yaw", type=float, default=0.1, help="spawn yaw jitter (+-rad)")
ap.add_argument("--checkpoint", default=None, help="PPO residual policy (default: zero residual, the pure MPC)")
ap.add_argument("--dense", action="store_true", help="fill both floors with racks, pallets and clutter")
ap.add_argument("--extra_north", type=float, default=8.0)
ap.add_argument("--visual", action="store_true", help="dress the scene with NVIDIA props (needs a viewer or --video)")
ap.add_argument("--no_equipment", action="store_true")
ap.add_argument("--video", default=None, help="chase-camera video of env 0")
ap.add_argument("--video_every", type=int, default=1, help="save every n-th tick (the frame rate drops to 40 / n)")
ap.add_argument("--dump", default=None, help="save per-robot trajectories to this .npy")
ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE", help="config overrides")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
if args.video or args.visual:
    args.enable_cameras = True
app = AppLauncher(args).app
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from spot_mpc.cli import apply_overrides, load_policy
from spot_mpc.navigation import PathFollower
from spot_mpc.tasks.task import SpotTask, SpotTaskCfg
from spot_mpc.worlds.warehouse_static import layout as L

# spawn x, y, floor z, yaw, route, time budget [s]
SEGMENTS = {
    "stairs_up":        (12.0, 7.0, 0.0, math.pi, L.STAIR_WAYPOINTS, 60.0),
    "upper_cross":      (0.5, 7.0, 3.0, math.atan2(1.3 - 7.0, 2.0 - 0.5), [(2.0, 1.3), (12.0, 1.3)], 50.0),
    "ramp_down":        (12.0, 1.3, 3.0, math.atan2(1.2, 3.0), [(15.0, 2.5), (15.0, 11.5)], 60.0),
    "ground_turn":      (17.5, 11.5, 0.0, math.pi, [(11.0, 11.5), (15.0, 11.5)], 50.0),
    "ramp_up":          (11.0, 11.5, 0.0, 0.0, [(15.0, 11.5), (15.0, 2.5)], 60.0),
    "upper_return":     (14.0, 1.0, 3.0, math.pi, [(12.0, 1.3), (2.0, 1.3), (0.0, 7.0)], 60.0),
    "stairs_down_back": (0.5, 7.0, 3.0, 0.0, L._descend_back, 70.0),
    "loop_back":        (12.0, 7.0, 0.0, math.pi, L.LOOP_BACK_WAYPOINTS, 330.0),
}


def main():
    sx, sy, sz, syaw, route, tmax = SEGMENTS[args.segment]
    tmax = args.t or tmax
    B = args.num_envs
    cfg = SpotTaskCfg()
    cfg.scene.num_envs = B
    cfg.world_type, cfg.terrain_source = "warehouse", "sensed"
    cfg.rear_depth = args.segment.endswith("_back")                       # the backward descents need a camera that looks behind
    cfg.episode_length_s = tmax + 30
    cfg.map_half_x, cfg.map_half_y = 4.0, 3.0
    cfg.warehouse.dense, cfg.warehouse.visual, cfg.warehouse.extra_north = args.dense, args.visual, args.extra_north
    cfg.warehouse.add_equipment = not args.no_equipment
    apply_overrides(cfg, args.set)
    if args.video:
        from isaaclab.envs.common import ViewerCfg
        cfg.viewer = ViewerCfg(eye=(-2.6, -2.6, 1.6), lookat=(0.8, 0.0, 0.1), origin_type="asset_root", asset_name="robot", env_index=0, resolution=(960, 540))
    env = SpotTask(cfg, render_mode="rgb_array" if args.video else None)
    env.fixed_spawn = True
    dev = env.device
    policy = load_policy(env, args.checkpoint) if args.checkpoint else None

    gen = torch.Generator().manual_seed(args.seed)
    jitter = lambda s: ((torch.rand(B, generator=gen) - 0.5) * 2 * s).to(dev)
    env.spawn_pose = torch.stack([sx + jitter(args.jitter_xy), sy + jitter(args.jitter_xy), syaw + jitter(args.jitter_yaw)], 1)
    env.spawn_floor_z = torch.full((B,), sz, device=dev)
    back = {"stairs_down_back": [False] + [True] * (len(route) - 1), "loop_back": L.LOOP_BACK_MASK}.get(args.segment)
    env.nav = PathFollower(B, route, dev, back=back, v_max=args.speed, w_max=args.turn_rate)
    with torch.inference_mode():
        obs, _ = env.reset()
    env.nav.reset(torch.arange(B, device=dev), env.spawn_pose[:, :2], 0)

    writer = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.video, fps=40 // args.video_every, codec="libx264", quality=8, macro_block_size=None)
        for _ in range(3):
            env.render()

    zero = torch.zeros(B, cfg.action_space, device=dev)
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    t_done = torch.full((B,), float("nan"), device=dev)
    t_fall = torch.full((B,), float("nan"), device=dev)
    wp_at_fall = torch.zeros(B, dtype=torch.long, device=dev)
    fall_state = torch.zeros(B, 4, device=dev)
    traj = []
    for k in range(int(tmax / env.step_dt)):
        with torch.inference_mode():
            x_prev = env.x0.clone()          # the task re-reads the state after a reset, keep the one before it
            obs, _, term, _, _ = env.step(zero if policy is None else policy(obs))
        t = k * env.step_dt
        finished = alive & ~done & env.nav.complete
        t_done[finished] = t
        done |= finished
        fell_now = term & alive & ~done
        t_fall[fell_now] = t
        wp_at_fall[fell_now] = env.nav.idx[fell_now]
        fall_state[fell_now] = torch.cat([x_prev[fell_now][:, 3:6], x_prev[fell_now][:, 1:2]], 1)
        if args.dump and k % 10 == 0:
            traj.append(torch.cat([x_prev[:, 3:6], x_prev[:, 2:3], x_prev[:, 1:2], x_prev[:, 0:1], alive[:, None].float(), done[:, None].float(), env.nav.idx[:, None].float()], 1).cpu().numpy())
        alive &= ~term
        if writer:
            frame = env.render()
            if k % args.video_every == 0:
                writer.append_data(frame)
            if bool(term[0]) or bool(done[0]):
                break
        if not bool((alive & ~done).any()):
            break
        if k % int(10 / env.step_dt) == 0:
            print(f"[wh] t={t:5.1f}s done {int(done.sum())}/{B} fell {int((~alive & ~done).sum())} median waypoint {int(env.nav.idx.median())}/{len(route)}")

    if args.dump:
        np.save(args.dump, np.stack(traj))
    fell = ~alive & ~done
    median = float(t_done[done].median()) if bool(done.any()) else float("nan")
    print(f"[result] {args.segment}: reached the end {int(done.sum())}/{B} ({100 * float(done.float().mean()):.0f}%), fell {int(fell.sum())}, "
          f"timed out {int((~done & ~fell).sum())}; median time {median:.1f}s")
    if bool(fell.any()):
        print("[result]   falls by the waypoint index being approached:", torch.bincount(wp_at_fall[fell], minlength=len(route) + 1).tolist())
        for i in torch.nonzero(fell).flatten()[:6].tolist():
            x, y, z, pitch = fall_state[i].tolist()
            print(f"[result]   env {i}: ({x:.1f}, {y:.1f}, {z:.2f}) pitch {pitch:+.2f} at t={t_fall[i]:.1f}s")
    if writer:
        writer.close()
        print(f"[video] {args.video}")
    os._exit(0)


if __name__ == "__main__":
    main()
