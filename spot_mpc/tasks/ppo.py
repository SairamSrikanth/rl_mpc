from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PpoCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "spot_mpc_residual"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    # gentle exploration: stair success is knife-edge in the operating point, and a noise std of 0.3 halved it
    policy = RslRlPpoActorCriticCfg(init_noise_std=0.08, noise_std_type="log", actor_obs_normalization=True,
                                    critic_obs_normalization=True, actor_hidden_dims=[512, 256, 128],
                                    critic_hidden_dims=[512, 256, 128], activation="elu")
    # a fixed small learning rate and a tight clip: an adaptive schedule let the rate drift and success peaked early
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.1, entropy_coef=0.0,
                                     num_learning_epochs=4, num_mini_batches=8, learning_rate=3.0e-5, schedule="fixed",
                                     gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=0.5)
