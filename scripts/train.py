import argparse, functools, os, sys, threading, time
from pathlib import Path
from isaaclab.app import AppLauncher

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=256)
ap.add_argument("--task", default="lanes", choices=["lanes", "general"], help="lanes = the hand-made test-world lanes (task.py); general = procedural terrain (general_task.py)")
ap.add_argument("--max_iterations", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume", default=None)
ap.add_argument("--run_name", default="")
ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, str(ROOT))
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from spot_mpc.cli import apply_overrides
from spot_mpc.tasks.general_task import GeneralTask, GeneralTaskCfg
from spot_mpc.tasks.ppo import PpoCfg
from spot_mpc.tasks.task import SpotTask, SpotTaskCfg


def main():
    cfg = GeneralTaskCfg() if args.task == "general" else SpotTaskCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    apply_overrides(cfg, args.set)
    agent = PpoCfg()
    agent.seed = args.seed
    if args.max_iterations:
        agent.max_iterations = args.max_iterations
    torch.manual_seed(args.seed)
    vec = RslRlVecEnvWrapper((GeneralTask if args.task == "general" else SpotTask)(cfg))
    log_dir = ROOT / "logs" / agent.experiment_name / (time.strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args.run_name}" if args.run_name else ""))
    os.makedirs(log_dir, exist_ok=True)
    (log_dir / "overrides.txt").write_text("\n".join(args.set) + "\n")
    print(f"[train] {vec.num_envs} envs | actions {vec.num_actions} | log {log_dir}")
    runner = OnPolicyRunner(vec, agent.to_dict(), log_dir=str(log_dir), device=agent.device)
    if args.resume:
        runner.load(args.resume)
    else:  # start AT the MPC baseline: shrink the actor's output layer so the initial mean action is ~0
        last = [m for m in runner.alg.policy.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.data *= 0.01
        last.bias.data.zero_()
    runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=False)
    vec.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    threading.Timer(20.0, os._exit, args=(0,)).start()
    app.close()
    os._exit(0)
