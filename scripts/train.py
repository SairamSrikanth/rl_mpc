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
ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="environment config overrides")
ap.add_argument("--agent_set", action="append", default=[], metavar="KEY=VALUE", help="PPO config overrides, e.g. algorithm.schedule='adaptive'")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, str(ROOT))
import json, subprocess
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.dict import class_to_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from spot_mpc.cli import apply_overrides
from spot_mpc.tasks.general_task import GeneralTask, GeneralTaskCfg
from spot_mpc.tasks.ppo import PpoCfg
from spot_mpc.tasks.task import SpotTask, SpotTaskCfg


def instrument(alg, n_sample=8192):
    """Add the policy's KL step and its mean action size to what PPO logs each iteration (rsl-rl only computes KL in adaptive mode)."""
    update = alg.update

    def wrapped():
        st = alg.storage
        n = st.mu.shape[0] * st.mu.shape[1]
        idx = torch.randperm(n, device=st.mu.device)[:n_sample]
        obs = st.observations.flatten(0, 1)[idx]
        mu0, sd0 = st.mu.flatten(0, 1)[idx].clone(), st.sigma.flatten(0, 1)[idx].clone()
        out = update()
        with torch.no_grad():
            alg.policy.act(obs)
            mu, sd = alg.policy.action_mean, alg.policy.action_std
            kl = (torch.log(sd / sd0) + (sd0 ** 2 + (mu0 - mu) ** 2) / (2 * sd ** 2) - 0.5).sum(-1).mean()
        out.update(kl=float(kl), action_abs=float(mu.abs().mean()), action_abs_p95=float(mu.abs().flatten().quantile(0.95)), noise_std=float(sd.mean()))
        return out

    alg.update = wrapped


def record(cfg, agent):
    """Everything a checkpoint was trained against: env and PPO config (incl. the CMA-ES operating points) and the code version."""
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(["git", "-C", str(ROOT), "diff", "HEAD", "--", "spot_mpc", "scripts"], capture_output=True, text=True).stdout
    except OSError:
        commit, diff = "unknown", ""
    return {"env_cfg": json.dumps(class_to_dict(cfg), default=str), "agent_cfg": json.dumps(agent.to_dict(), default=str), "git_commit": commit, "git_diff": diff,
            "task": args.task, "env_overrides": list(args.set), "agent_overrides": list(args.agent_set)}


def main():
    cfg = GeneralTaskCfg() if args.task == "general" else SpotTaskCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    apply_overrides(cfg, args.set)
    agent = PpoCfg()
    agent.seed = args.seed
    if args.max_iterations:
        agent.max_iterations = args.max_iterations
    apply_overrides(agent, args.agent_set)
    torch.manual_seed(args.seed)
    vec = RslRlVecEnvWrapper((GeneralTask if args.task == "general" else SpotTask)(cfg))
    log_dir = ROOT / "logs" / agent.experiment_name / (time.strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args.run_name}" if args.run_name else ""))
    os.makedirs(log_dir, exist_ok=True)
    (log_dir / "overrides.txt").write_text("\n".join(args.set + args.agent_set) + "\n")
    meta = record(cfg, agent)
    (log_dir / "code.diff").write_text(meta["git_diff"])               # uncommitted changes on top of git_commit
    (log_dir / "run_config.json").write_text(json.dumps({k: (json.loads(v) if k.endswith("_cfg") else v) for k, v in meta.items() if k != "git_diff"}, indent=1, default=str))
    print(f"[train] {vec.num_envs} envs | actions {vec.num_actions} | log {log_dir}")
    runner = OnPolicyRunner(vec, agent.to_dict(), log_dir=str(log_dir), device=agent.device)
    instrument(runner.alg)
    save = runner.save
    runner.save = lambda path, infos=None: save(path, infos={**(infos or {}), **meta})          # every checkpoint carries its config
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
