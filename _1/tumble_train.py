import torch
import numpy as np
from vec_env import VecEnv
from config import TrainingConfig, RewardConfig, SpawnConfig
from stable_baselines3 import PPO
from example_train import SB3Wrapper, MetricsCallback, Points


recovery_reward = RewardConfig(
    shaping_scale=0.0,              # potential shaping can be confusing for this taks
    angular_velocity_scale=0.5,    # heavy penalty for spinning
    action_smoothness_scale=0.01,
    step_penalty=0.05,
    success_bonus=500.0,
    proximity_reward_scale=2.0,     # reward for being upright
    proximity_falloff=1.0,           # sharp falloff for alignment
    success_steps_in_tolerance_traj=128
)

# aggressive tumbling starts
recovery_spawn = SpawnConfig(
    spawn_radius_mean=0.0,          # centered
    initial_velocity_std=3.0,       # high initial drift
    initial_orientation_range=np.pi # fully randomized rotation
)

cfg = TrainingConfig(
    max_episode_steps=256,
    device="cuda" if torch.cuda.is_available() else "cpu"
)


device = cfg.device

# custom Zero-G reset logic
class ZeroGEnv(VecEnv):
    def _spawn_positions(self, env_ids, target_starts, num_resets):
        super()._spawn_positions(env_ids, target_starts, num_resets)
        # gravity to zero for these environments
        self.physics.gravity[env_ids] = 0.0
        # high angular velocity for recovery training
        self.omega[env_ids] = torch.randn((num_resets, 3), device=self.device) * 5.0

if __name__ == "__main__":
    print(f"Starting Zero-G Recovery Training on {device}...")
    
    num_envs = 2048
    max_steps = cfg.max_episode_steps
    n_steps = max_steps // 2
    batch_size = num_envs * n_steps // 64 # 64 updates each internal PPO update
    # manager
    recovery_manager = Points(num_envs=num_envs, device=device)
    
    # initialize the vectorized environment
    custom_env = ZeroGEnv(
        stage_names="recovery", 
        trajectory_manager=recovery_manager, 
        num_envs=num_envs, 
        config=cfg, 
        spawn_config=recovery_spawn, 
        reward_config=recovery_reward
    )
    
    wrapped_env = SB3Wrapper(custom_env)

    # PPO optimized for high throughput
    model = PPO(
        "MlpPolicy", 
        wrapped_env, 
        verbose=1, 
        device=device, 
        n_steps=n_steps, 
        batch_size=batch_size, 
        n_epochs=4,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=25_000_000, callback=MetricsCallback())
