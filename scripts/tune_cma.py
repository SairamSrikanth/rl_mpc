import argparse, functools, json, math, os, sys, time
from pathlib import Path
from isaaclab.app import AppLauncher

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--pop", type=int, default=8)
ap.add_argument("--gens", type=int, default=14)
ap.add_argument("--num_envs", type=int, default=96)
ap.add_argument("--t", type=float, default=None)
ap.add_argument("--sigma", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mode", default="down", choices=["down", "up"])
ap.add_argument("--name", default="descent_back")
ap.add_argument("--set", nargs="*", action="extend", default=[])
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, str(ROOT))
import cma, numpy as np, torch
from spot_mpc.cli import apply_overrides
from spot_mpc.navigation import PathFollower
from spot_mpc.tasks.task import SpotTask, SpotTaskCfg

# name, low, high, log-scale, default  (all are cfg fields; the env blends *_back / *_up by the previewed backward descent / ascent)
# defaults are the hand-tuned starting point; the foot_margin, swing_clearance, swing_kp and swing_kd bounds were widened after the v1 searches ended on them
PARAMS_DOWN = [
    ("stair_speed_back", 0.25, 0.45, False, 0.35), ("q_vel_xy_back", 10.0, 300.0, True, 80.0), ("gait_duty_back", 0.50, 0.70, False, 0.60),
    ("gait_freq_back", 1.6, 2.4, False, 2.0), ("swing_clearance_back", 0.02, 0.10, False, 0.06), ("swing_base_apex_back", 0.04, 0.12, False, 0.08),
    ("foot_margin_back", 0.02, 0.12, False, 0.08), ("q_pitch_back", 10.0, 100.0, True, 40.0), ("descent_crouch_back", -0.03, 0.06, False, 0.0),
    ("swing_kp_back", 250.0, 900.0, False, 600.0), ("swing_kd_back", 20.0, 60.0, False, 30.0)]
PARAMS_UP = [
    ("stair_speed", 0.30, 0.60, False, 0.45), ("q_vel_xy_steep", 0.5, 30.0, True, 2.0), ("slope_crouch", 0.0, 0.12, False, 0.07),
    ("gait_duty_up", 0.50, 0.70, False, 0.60), ("gait_freq_up", 1.6, 2.6, False, 2.0), ("swing_clearance_up", 0.03, 0.10, False, 0.06),
    ("swing_base_apex_up", 0.04, 0.12, False, 0.08), ("foot_margin_up", 0.02, 0.12, False, 0.08), ("swing_kp_up", 400.0, 900.0, False, 600.0),
    ("swing_kd_up", 20.0, 40.0, False, 30.0)]
PARAMS = PARAMS_UP if args.mode == "up" else PARAMS_DOWN


def to_unit(i, v):
    n, lo, hi, lg, _ = PARAMS[i]
    return (math.log(v / lo) / math.log(hi / lo)) if lg else (v - lo) / (hi - lo)


def from_unit(i, u):
    n, lo, hi, lg, _ = PARAMS[i]
    u = min(1.0, max(0.0, u))
    return lo * (hi / lo) ** u if lg else lo + u * (hi - lo)


def main():
    args.t = args.t or (34.0 if args.mode == "up" else 26.0)
    cfg = SpotTaskCfg()
    B = args.num_envs
    cfg.scene.num_envs = B
    cfg.world_type, cfg.terrain_source, cfg.rear_depth = "warehouse", "sensed", True
    cfg.episode_length_s = args.t + 30
    cfg.map_half_x, cfg.map_half_y = 4.0, 3.0
    cfg.gate_on_steps = False
    apply_overrides(cfg, args.set)
    e = SpotTask(cfg)
    e.fixed_spawn = True
    dev = e.device
    UP = args.mode == "up"
    route = [(9.5, 7.0), (8.6, 7.0), (7.4, 7.0), (6.2, 7.0), (4.5, 7.0)] if UP else [(6.2, 7.0), (7.4, 7.0), (8.6, 7.0), (9.5, 7.0), (12.0, 7.0)]
    e.nav = PathFollower(B, route, dev, v_max=0.45, w_max=0.8)
    e.nav.back_env[:] = not UP
    zero = torch.zeros(B, cfg.action_space, device=dev)
    allid = torch.arange(B, device=dev)

    def apply(x):
        for i, p in enumerate(PARAMS):
            setattr(e.cfg, p[0], float(from_unit(i, x[i])))

    @torch.inference_mode()
    def evaluate(x, gen_seed):
        apply(x)
        g = torch.Generator().manual_seed(gen_seed)
        jr = lambda s: ((torch.rand(B, generator=g) - 0.5) * 2 * s).to(dev)  # noqa: E731
        e.spawn_pose = torch.stack([(11.0 if UP else 3.6) + jr(0.3), 7.0 + jr(0.3), math.pi + jr(0.1)], 1)
        e.spawn_floor_z = torch.full((B,), 0.0 if UP else 3.0, device=dev)
        with torch.inference_mode():
            e.reset()
        e.nav.reset(allid, e.spawn_pose[:, :2], 0)
        e.nav.back_env[:] = not UP
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        t_done = torch.full((B,), args.t, device=dev)
        for k in range(int(args.t / e.step_dt)):
            with torch.inference_mode():
                x_prev = e.x0.clone()
                _, _, term, _, _ = e.step(zero)
            t = k * e.step_dt
            fin = alive & ~done & ((x_prev[:, 3] < 5.2) if UP else (x_prev[:, 3] > 10.0))
            t_done[fin] = t
            done |= fin
            alive &= ~(term & ~done)
            if not bool((alive & ~done).any()):
                break
        frac = float(done.float().mean())
        return frac - 0.002 * float(t_done[done].mean() if bool(done.any()) else args.t), frac

    x0 = [to_unit(i, p[4]) for i, p in enumerate(PARAMS)]
    es = cma.CMAEvolutionStrategy(x0, args.sigma, {"popsize": args.pop, "bounds": [0, 1], "seed": args.seed + 1, "verbose": -9})
    out = ROOT / "results" / "cma"
    os.makedirs(out, exist_ok=True)
    hist = []
    t0 = time.time()
    base_f, base_frac = evaluate(x0, 1000)
    print(f"[cma] default parameters: success {base_frac:.3f}  ({time.time() - t0:.0f}s)")
    best = (base_f, list(x0), base_frac)
    for gen in range(args.gens):
        xs = es.ask()
        res = [evaluate(x, 2000 + gen) for x in xs]
        es.tell(xs, [-r[0] for r in res])
        i = int(np.argmax([r[0] for r in res]))
        if res[i][0] > best[0]:
            best = (res[i][0], list(xs[i]), res[i][1])
        hist.append(dict(gen=gen, scores=[r[1] for r in res], best_vec={p[0]: from_unit(j, xs[i][j]) for j, p in enumerate(PARAMS)}))
        print(f"[cma] gen {gen:2d}: success mean {np.mean([r[1] for r in res]):.3f} max {max(r[1] for r in res):.3f} | best so far {best[2]:.3f} | {time.time() - t0:.0f}s")
        json.dump(dict(params=[p[0] for p in PARAMS], default={p[0]: p[4] for p in PARAMS}, best={p[0]: from_unit(j, best[1][j]) for j, p in enumerate(PARAMS)},
                       best_success=best[2], mean_vec={p[0]: from_unit(j, es.mean[j]) for j, p in enumerate(PARAMS)}, hist=hist), open(out / f"{args.name}.json", "w"), indent=1)
    mean = {p[0]: from_unit(j, es.mean[j]) for j, p in enumerate(PARAMS)}
    # the best candidate won a max over many noisy scores on changing seeds: re-score it, the mean and the default on one unseen seed
    holdout = {k: evaluate(x, 9000 + args.seed)[1] for k, x in (("default", x0), ("mean", list(es.mean)), ("best", best[1]))}
    print("[cma] held-out success (seed %d): " % (9000 + args.seed) + "  ".join(f"{k} {v:.3f}" for k, v in holdout.items()))
    log = json.load(open(out / f"{args.name}.json")) if args.gens else {}
    json.dump(dict(log, holdout=holdout), open(out / f"{args.name}.json", "w"), indent=1)
    print("[cma] distribution mean as overrides:  " + " ".join(f"{k}={v:.4g}" for k, v in mean.items()))
    print("[cma] best candidate as overrides:     " + " ".join(f"{p[0]}={from_unit(j, best[1][j]):.4g}" for j, p in enumerate(PARAMS)))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
