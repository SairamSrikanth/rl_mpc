import ast


def apply_overrides(cfg, items):
    """Apply KEY=VALUE overrides to a config. Dotted keys reach into nested configs, e.g. generated.proportions={...}."""
    for item in items:
        key, value = item.split("=", 1)
        target = cfg
        *path, name = key.split(".")
        for part in path:
            target = getattr(target, part)
        assert hasattr(target, name), f"unknown config field {key}"
        setattr(target, name, ast.literal_eval(value))
    print("[cfg] overrides:", items or "none")


def load_policy(env, checkpoint):
    """Inference policy from a PPO checkpoint (rsl-rl), for the residual actions."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict
    from .tasks.ppo import PpoCfg
    runner = OnPolicyRunner(RslRlVecEnvWrapper(env), PpoCfg().to_dict(), log_dir=None, device=str(env.device))
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=str(env.device))
    return lambda obs: policy(TensorDict(obs, batch_size=[env.num_envs]))
