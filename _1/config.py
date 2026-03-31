from dataclasses import field, dataclass
from collections import deque
import torch
import numpy as np
from typing import List, Dict, Tuple, Any

@dataclass
class DomainRandomizationConfig:
    enabled: bool = True
    randomize_mass: bool = True
    mass_range: Tuple[float, float] = (0.8, 5)
    mass_nominal: float = 1.5

    randomize_inertia: bool = True
    inertia_range: Tuple[float, float] = (0.5, 2.5)
    inertia_nominal: float = 1.0

    randomize_thrust_to_weight: bool = True
    thrust_to_weight_range: Tuple[float, float] = (1.1, 3.2)
    thrust_to_weight_nominal: float = 2.2

    randomize_drag: bool = True
    drag_coef_range: Tuple[float, float] = (0.01, 0.08)
    drag_coef_nominal: float = 0.03

    randomize_wind: bool = True
    wind_speed_range: Tuple[float, float] = (0.0, 20.0)
    wind_speed_nominal: float = 0.0

    randomize_vertical_wind: bool = True
    wind_elevation_range: Tuple[float, float] = (-np.pi/6, np.pi/6)

    randomize_air_density: bool = True
    air_density_range: Tuple[float, float] = (0.85, 1.25)
    air_density_nominal: float = 1.0

    randomize_sensor_noise: bool = False
    position_noise_range: Tuple[float, float] = (0.0, 0.5)
    position_noise_nominal: float = 0.0
    velocity_noise_range: Tuple[float, float] = (0.0, 0.3)
    velocity_noise_nominal: float = 0.0
    attitude_noise_range: Tuple[float, float] = (0.0, 0.05)
    attitude_noise_nominal: float = 0.0

    randomize_control_delay: bool = True
    control_delay_range: Tuple[float, float] = (0.0, 0.05)
    control_delay_nominal: float = 0.0

    randomize_interval: bool = True
    time_interval_range: Tuple[float, float] = (0.005, 0.05)
    time_interval_nominal: float = 0.01
    

@dataclass
class TrainingConfig:
    domain_randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_parallel_envs: int = 16
    
    trajectory_stages: List[str] = field(default_factory=lambda: [
        'hovering', 'altitude_control', 'straight_lines', 'curved_paths'
    ])

    use_waypoint_mode: bool = False
    waypoint_reached_bonus: float = 5.0 
    
    # agent config reference for dimension consistency
    obs_dim: int = 27 
    action_dim: int = 4
    privileged_dim: int = 12
    
    base_epochs: int = 200
    base_learning_rate: float = 3e-4
    base_entropy_coef: float = 0.01
    base_clip_ratio: float = 0.2
    value_coef: float = 0.6
    max_grad_norm: float = 0.5
    
    ppo_epochs: int = 3
    mini_batch_size: int = 2048 * 2 * 3
    steps_per_update: int = 1024 + 2
    max_episode_steps: int = 1024 * 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    normalize_observations: bool = True
    normalize_rewards: bool = False
    obs_clip: float = 100.0
    
    # entropy scheduling
    entropy_decay_rate: float = 0.995
    entropy_floor: float = 1e-4
    entropy_decay_start_epoch: int = 50
    
    output_dir: str = "./training_outputs_ppo"
    checkpoint_dir: str = "./checkpoints_ppo"
    log_dir: str = "./logs_ppo"

@dataclass
class RewardConfig:
    """
    Configurable reward and penalty values for the quadrotor environment.
    
    Rewards are additive per step. Episode length can span up to few thousands steps.
    
    Shaping rewards:
        - shaping_scale: Multiplier for potential-based shaping (distance reduction)
        
    Waypoint rewards:
        - waypoint_reached_bonus: One-time bonus when reaching a waypoint
        - success_bonus: Final bonus when completing all waypoints
        
    Penalties (should be negative or zero):
        - step_penalty: Per-step cost to encourage efficiency
        - action_smoothness_scale: Penalty for jerky control inputs
        - angular_velocity_scale: Penalty for high body rates
        
    Optional rewards:
        - camera_alignment_scale: Reward for pointing toward target (set 0 to disable)
        - camera_alignment_proximity_min: Minimum proximity factor for alignment reward

    """
    # shaping
    shaping_scale: float = 1.0
    
    # waypoint / success bonuses
    waypoint_reached_bonus: float = 70.0
    success_bonus: float = 150.0
    
    # per-step penalty
    step_penalty: float = 0.2
    
    # smoothness penalties
    action_smoothness_scale: float = 0.2
    angular_velocity_scale: float = 0.1
    
    # camera alignment (yaw toward target)
    camera_alignment_scale: float = 0.0  # disabled by default
    camera_alignment_proximity_min: float = 0.3
    camera_alignment_proximity_range: float = 30.0

    proximity_reward_scale: float = 0.0   # reward per step near target, turned off to avoid reward hacking when not hovering
    proximity_falloff: float = 10.0 
    
    # termination conditions
    success_steps_in_tolerance: int = 1  # waypoint mode
    success_steps_in_tolerance_traj: int = 64  # trajectory mode
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.shaping_scale >= 0, "shaping_scale must be non-negative"
        assert self.waypoint_reached_bonus >= 0, "waypoint_reached_bonus must be non-negative"
        assert self.success_bonus >= 0, "success_bonus must be non-negative"
        assert self.step_penalty >= 0, "step_penalty should be positive (it's subtracted each step)"
        assert self.action_smoothness_scale >= 0, "action_smoothness_scale must be non-negative"
        assert self.angular_velocity_scale >= 0, "angular_velocity_scale must be non-negative"


@dataclass 
class SpawnConfig:
    """Configuration for environment spawn/reset behavior."""
    # spawn distance from first waypoint
    spawn_radius_mean: float = 6.0
    spawn_radius_std: float = 2.0
    spawn_radius_max: float = 30.0
    
    # initial velocity randomization
    initial_velocity_std: float = 2.0
    initial_velocity_clamp: float = 1.0
    
    # initial orientation randomization (radians)
    initial_orientation_range: float = np.pi / 3