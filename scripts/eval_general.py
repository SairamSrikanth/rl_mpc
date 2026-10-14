import argparse, functools, os, re, sys
from pathlib import Path
from isaaclab.app import AppLauncher

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--seed", type=int, default=12345, help="terrain seed (held out from training)")
ap.add_argument("--bands", default="0,1,2;4,5;7,8,9", help="difficulty rows per band, bands separated by ';'")
ap.add_argument("--num_envs", type=int, default=256)
ap.add_argument("--dr", type=int, default=1, help="1 = friction / mass / motor / sensor-noise randomisation on, 0 = nominal")
ap.add_argument("--t", type=float, default=60.0)
ap.add_argument("--speed", type=float, default=0.45, help="commanded forward speed [m/s]")
ap.add_argument("--per_env", action="store_true", help="also print one outcome letter per robot (G goal, F fell, T timed out) for paired comparisons")
ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, str(ROOT))
import collections, torch
from spot_mpc.cli import apply_overrides, load_policy
from spot_mpc.tasks.general_task import GeneralTask, GeneralTaskCfg


def main():
    torch.manual_seed(args.seed)
    cfg = GeneralTaskCfg()
    B = args.num_envs
    cfg.scene.num_envs = B
    cfg.curriculum = False
    cfg.episode_length_s = args.t + 1.0
    cfg.generated.seed = args.seed
    cfg.domain_rand = bool(args.dr)
    cfg.push_speed = 0.35 if args.dr else 0.0
    cfg.sensor_noise_std = 0.01 if args.dr else 0.0
    apply_overrides(cfg, args.set)
    e = GeneralTask(cfg)
    dev = e.device
    policy = load_policy(e, args.checkpoint) if args.checkpoint else None
    bands = [[int(v) for v in b.split(",")] for b in args.bands.split(";")]
    fams = e.world.families
    ncols = e.importer.terrain_origins.shape[1]
    levels, types, band_of = [], [], []
    for i in range(B):
        b = i % len(bands)
        levels.append(bands[b][(i // len(bands)) % len(bands[b])])
        types.append((i // len(bands)) % ncols)
        band_of.append(b)
    levels, types = torch.tensor(levels), torch.tensor(types)
    e.assign_tiles(levels, types)
    e.fixed_command = True                          # one fixed command (start heading 0, 0.45 m/s); randomisation stays as configured by --dr
    e.eval_speed = args.speed
    zero = torch.zeros(B, cfg.action_space, device=dev)
    with torch.inference_mode():
        obs, _ = e.reset()
    seen = torch.zeros(B, dtype=torch.bool, device=dev)
    outcome = ["T"] * B
    res = collections.defaultdict(lambda: [0, 0, 0])      # (family, band) -> [episodes, reached, fell]
    for k in range(int(args.t / e.step_dt)):
        with torch.inference_mode():
            act = zero if policy is None else policy(obs)
            obs, _, term, trunc, _ = e.step(act)
        t = k * e.step_dt
        ended = (term | trunc) & ~seen
        for i in torch.nonzero(ended).flatten().tolist():
            key = (fams[int(types[i])], band_of[i])
            r = res[key]
            r[0] += 1
            r[2] += int(term[i])
            r[1] += int(bool(trunc[i]) and not bool(term[i]) and t < args.t - 0.5)       # an early time-out is the success exit
            outcome[i] = "F" if bool(term[i]) else ("G" if bool(trunc[i]) and t < args.t - 0.5 else "T")
            seen[i] = True
    name = "policy " + Path(args.checkpoint).parent.name + "/" + Path(args.checkpoint).name if args.checkpoint else "zero residual (pure MPC)"
    print(f"[gen] {name}; terrain seed {args.seed}; randomisation {'ON' if args.dr else 'off'}; {B} robots, one episode each (still walking at {args.t:.0f} s: {int((~seen).sum())})")
    blabels = [f"rows {','.join(map(str, b))}" for b in bands]
    print(f"[gen] {'family':18s} " + "  ".join(f"{b:>18s}" for b in blabels))
    print(f"[gen] {'':18s} " + "  ".join(f"{'goal / fell / n':>18s}" for _ in bands))
    tot = [[0, 0, 0] for _ in bands]
    for f in sorted({f for f, _ in res}):
        cells = []
        for b in range(len(bands)):
            n, s, fl = res.get((f, b), [0, 0, 0])
            tot[b][0] += n; tot[b][1] += s; tot[b][2] += fl
            cells.append(f"{s:3d} / {fl:3d} / {n:3d}".rjust(18))
        print(f"[gen] {f:18s} " + "  ".join(cells))
    print(f"[gen] {'ALL':18s} " + "  ".join(f"{tot[b][1]:3d} / {tot[b][2]:3d} / {tot[b][0]:3d}".rjust(18) for b in range(len(bands))))
    print(f"[gen] goal rate by band: " + "  ".join(f"{b}: {100 * tot[i][1] / max(1, tot[i][0]):.0f}%" for i, b in enumerate(blabels)))
    if args.per_env:
        print("[gen] per-robot:", "".join(outcome))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
