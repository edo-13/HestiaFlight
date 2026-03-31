import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv as SB3VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from config import TrainingConfig, DomainRandomizationConfig, RewardConfig, SpawnConfig
from vec_env import VecEnv
import torch
import numpy as np

class Points:
    """Wrapper for the required TrajectoryManager/WaypointsManager required in vec_env.py"""
    def __init__(self, num_envs, device="cuda"):
        self.count = num_envs
        self.device = device

    def get_training_batch(self, stage, count, device):
        """Generates the points for hovering"""

        pos = torch.randn((count,1 , 3), device=device) * 20.0
        pos[:, :, 2] = torch.abs(pos[:, :, 2]) + 10.0 # ensuring the height is always higher than 10m

        vel = torch.zeros_like(pos)

        return {
            'positions': pos,
            'velocities': vel,
        }

class SB3Wrapper(SB3VecEnv):
    """Bridge between your high-speed VecEnv and Stable Baselines3."""
    def __init__(self, custom_env):
        self.env = custom_env
        super().__init__(
            num_envs=custom_env.num_envs,
            observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(custom_env.OBS_DIM,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(custom_env.ACTION_DIM,))
        )

    def reset(self):
        obs = self.env.reset()
        return obs.cpu().numpy()

    def step_async(self, actions):
        self.actions = torch.from_numpy(actions).to(self.env.device)

    def step_wait(self):
        obs, rews, terms, truncs, infos = self.env.step(self.actions)
        # SB3 expects 'dones' (bool array)
        dones = (terms | truncs).cpu().numpy()
        return obs.cpu().numpy(), rews.cpu().numpy(), dones, infos

    def close(self):
        """Clean up resources."""
        pass

    def get_attr(self, attr_name, indices=None):
        """Return attribute from the env(s)."""
        return getattr(self.env, attr_name)

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute in the env(s)."""
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call a method in the env(s)."""
        method = getattr(self.env, method_name)
        return method(*method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if env is wrapped."""
        return [False] * self.num_envs
    
class MetricsCallback(BaseCallback):
    def __init__(self, print_freq=1):
        super().__init__()
        self.print_freq = print_freq

    def _on_step(self):
        return True
        
    def _on_rollout_end(self):
        if self.n_calls % self.print_freq == 0:
            env = self.training_env.env
            rate = env.get_epoch_success_rate()
            eps = env.epoch_episodes
            print(f"Episodes: {eps:>4} | Success: {rate:>5.1%}")
            env.reset_epoch_stats()

if __name__ == "__main__":

    num_envs = 1024
    n_steps = 128
    
    spawn_cfg=SpawnConfig(
        spawn_radius_mean=15.0, # changing mean and std for this example, \
        spawn_radius_std=5.0    # if training other tasks, spawning close to the first waypoint is fine
        )
    
    dmr_cfg = DomainRandomizationConfig(
        enabled=False       # turning off ADR for this example
        )
    
    rwrd_cfg = RewardConfig(
        shaping_scale=2.0,              # stronger approach incentive
        step_penalty=0.1,              # 2x smaller than in basic config
        action_smoothness_scale=0.05,   # gentler for corrections
        angular_velocity_scale=0.02,    # reduced 
        success_bonus=200.0,
        success_steps_in_tolerance_traj=int(n_steps*0.6), #60%% of a episode within tolerance
        proximity_reward_scale=0.5, # turning it on for hovering, otherwise set at 0.0
        )
    
    cfg = TrainingConfig(
        max_episode_steps=n_steps*2, # 2x the rollout length
        domain_randomization=dmr_cfg,
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    
    batch_size = (num_envs * n_steps) // 64 # 64 updates per each PPO internal epoch - yes, it's a magic number
    
    hov_manager = Points(num_envs=num_envs)
    custom_vec_env = VecEnv(stage_names="hovering", trajectory_manager=hov_manager, num_envs=num_envs, config=cfg, spawn_config=spawn_cfg, reward_config=rwrd_cfg)
    wrapped_env = SB3Wrapper(custom_vec_env)
    
    model = PPO("MlpPolicy", wrapped_env, verbose=1, device=device, n_steps=n_steps, batch_size=batch_size, n_epochs=2) # couple of optimizations
    model.learn(total_timesteps=20_000_000, callback=MetricsCallback())