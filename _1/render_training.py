import torch
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from vec_env import VecEnv
from config import TrainingConfig, RewardConfig, SpawnConfig
from example_train import SB3Wrapper, Points
from render import QuadrotorRenderer


class VisualTrainingCallback(BaseCallback):
    """Callback that renders during training."""
    
    def __init__(
        self, 
        env: VecEnv, 
        renderer: QuadrotorRenderer,
        render_freq: int = 2,  # render every n steps
        print_freq: int = 10,
    ):
        super().__init__()
        self.env = env
        self.renderer = renderer
        self.render_freq = render_freq
        self.print_freq = print_freq
        self.paused = False
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    if self.renderer._recording:
                        self.renderer.stop_recording()
                    else:
                        self.renderer.start_recording("training.gif")
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("PAUSED" if self.paused else "RESUMED")
        
        # pause loop
        while self.paused:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.paused = False
                    print("RESUMED")
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return False
            pygame.time.wait(100)
        
        # render
        if self.step_count % self.render_freq == 0:
            cfg = self.env.config
            running = self.renderer.render(
                self.env.pos,
                self.env.quat,
                self.env.env_waypoints if cfg.use_waypoint_mode else self.env.env_traj_pos,
                self.env.current_waypoint_idx if cfg.use_waypoint_mode else self.env.target_idx
            )
            if not running:
                return False
        
        return True
    
    def _on_rollout_end(self):
        if self.n_calls % self.print_freq == 0:
            rate = self.env.get_epoch_success_rate()
            eps = self.env.epoch_episodes
            print(f"Rollout {self.n_calls:>4} | Episodes: {eps:>4} | Success: {rate:>5.1%}")
            self.env.reset_epoch_stats()


def train_with_visualization(
    num_envs: int = 1024,
    total_timesteps: int = 10_000_000,
    n_steps: int = 128,
    render_every: int = 4,
    record_on_start: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device} with {num_envs} envs")
    
    # configs
    spawn_cfg = SpawnConfig(spawn_radius_mean=15.0, spawn_radius_std=5.0)
    reward_cfg = RewardConfig(
        shaping_scale=2.0,
        step_penalty=0.1,
        action_smoothness_scale=0.05,
        angular_velocity_scale=0.02,
        success_bonus=200.0,
        success_steps_in_tolerance_traj=int(n_steps * 0.6),
        proximity_reward_scale=0.5,
        camera_alignment_scale=0.5
    )
    cfg = TrainingConfig(
        max_episode_steps=n_steps * 2,
        device=device,
    )
    cfg.domain_randomization.enabled = False
    
    # environment
    manager = Points(num_envs=num_envs, device=device)
    env = VecEnv(
        stage_names="hover",
        num_envs=num_envs,
        trajectory_manager=manager,
        config=cfg,
        spawn_config=spawn_cfg,
        reward_config=reward_cfg,
    )
    wrapped = SB3Wrapper(env)
    
    # renderer
    renderer = QuadrotorRenderer(
        width=1280,
        height=720,
        max_rendered_quads=8,
        show_trails=True,
    )
    
    if record_on_start:
        renderer.start_recording("training.gif")
    
    # callback
    callback = VisualTrainingCallback(
        env=env,
        renderer=renderer,
        render_freq=render_every,
    )
    
    # model optimized for good throughput while training
    batch_size = (num_envs * n_steps) // 64
    model = PPO(
        "MlpPolicy",
        wrapped,
        verbose=0,
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=2,
    )
    
    print("Controls: SPACE=pause, R=record, ESC=quit")
    print("-" * 50)
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    # cleanup
    if renderer._recording:
        renderer.stop_recording()
    renderer.close()
    
    return model

"""
HestiaFlight - Training with Real-time Visualization
Press R to record, ESC to quit, Space to pause training.
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--timesteps", type=int, default=15_000_000)
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    
    model = train_with_visualization(
        num_envs=args.envs,
        total_timesteps=args.timesteps,
        render_every=args.render_every,
        record_on_start=args.record,
    )
    
    # save
    model.save("hestiaflight_hover")
    print("Model saved to hestiaflight_hover.zip")