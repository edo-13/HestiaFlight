from dataclasses import field, dataclass
from collections import deque
import torch
import numpy as np
from typing import List, Dict, Tuple, Any

# Physics constants and domain randoimzation parameters

@dataclass
class PhysicsConstants:
    mass: float = 1.5
    arm_length: float = 0.25
    thrust_to_weight: float = 2.5
    ixx: float = 0.0347563
    iyy: float = 0.0458929
    izz: float = 0.0977
    gravity: float = 9.81

    drag_xy: float = 0.04
    drag_z: float = 0.02

    kp_roll: float = 5.0
    kp_pitch: float = 5.0
    kp_yaw: float = 5.0
    max_body_rate: float = 5.0
    damping: float = 0.1

    yaw_torque_coef: float = 0.01

    motor_tau: float = 0.015
    motor_zeta: float = 0.85

    @property
    def motor_omega_n(self) -> float:
        return 2.0 / self.motor_tau

    rotor_radius: float = 0.065
    ground_effect_ceiling: float = 0.5

    integrator_euler: int = 0
    integrator_semi_implicit: int = 1
    integrator_rk4: int = 2

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

    randomize_sensor_noise: bool = False          # master switch

    randomize_position_noise: bool = True         # cascading sub-toggle
    position_noise_range: Tuple[float, float] = (0.0, 0.5)
    position_noise_nominal: float = 0.001         # near-zero for now

    randomize_velocity_noise: bool = True
    velocity_noise_range: Tuple[float, float] = (0.0, 0.3)
    velocity_noise_nominal: float = 0.001

    randomize_attitude_noise: bool = True
    attitude_noise_range: Tuple[float, float] = (0.0, 0.05)
    attitude_noise_nominal: float = 0.0

    randomize_gyro_noise: bool = True
    gyro_noise_range: Tuple[float, float] = (0.0, 0.3)
    gyro_noise_nominal: float = 0.0

    randomize_accel_noise: bool = True
    accel_noise_range: Tuple[float, float] = (0.0, 0.5)
    accel_noise_nominal: float = 0.0

    randomize_control_delay: bool = True
    control_delay_range: Tuple[float, float] = (0.0, 0.05)
    control_delay_nominal: float = 0.0

    randomize_interval: bool = True
    time_interval_range: Tuple[float, float] = (0.005, 0.05)
    time_interval_nominal: float = 0.01


# Training configuration
    

@dataclass
class TrainingConfig:
    domain_randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_parallel_envs: int = 16
    
    trajectory_stages: List[str] = field(default_factory=lambda: [
        'hovering', 'altitude_control', 'straight_lines', 'curved_paths'
    ])

    use_waypoint_mode: bool = True
    waypoint_reached_bonus: float = 5.0 
    
    # agent config reference for dimension consistency
    obs_dim: int = 27 
    action_dim: int = 4
    privileged_dim: int = 12
    
    base_epochs: int = 200
    base_learning_rate: float = 3e-4
    base_entropy_coef: float = 0.02
    base_clip_ratio: float = 0.2
    value_coef: float = 0.6
    max_grad_norm: float = 0.5
    
    ppo_epochs: int = 3
    mini_batch_size: int = 1024
    seq_len: int = 8 
    bptt_env_chunk: int = 8
    steps_per_update: int = 2048
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

    Rewards are additive per step. Episode length can span up to a few thousand steps.

    Shaping rewards:
        - shaping_scale: Multiplier applied to potential-based progress signal.
          Default 2.0 matches the `shaping * 2.0` weight in the tuned reward.

    Kinetic efficiency:
        - efficiency_bonus_scale: Scales `progress_rate * sqrt(speed)`.
          Rewards speed ONLY when it converts to actual path progress.
        - ke_reward_scale: Small unconditional speed bonus (keeps the drone
          from hovering even when momentarily off-path).

    Waypoint rewards:
        - waypoint_reached_bonus: One-time bonus when reaching an intermediate waypoint.
        - success_bonus: Added on final success AFTER reward_divisor is applied.

    Penalties (positive values; sign is applied internally):
        - step_penalty: Per-step time cost → incentivizes finishing fast.
        - action_smoothness_scale: Penalty for actuator jerk.
        - angular_velocity_scale: Penalty for high body rates (set 0 to disable).

    Normalization:
        - reward_divisor: Divides the total step reward before returning.
          Success bonus is added after division (consistent with training code).

    Optional rewards:
        - camera_alignment_scale: Reward for yaw toward target (0 = disabled).
        - proximity_reward_scale: Dense proximity reward (0 = disabled, avoids
          reward hacking when not hovering).
    """
    # shaping
    shaping_scale: float = 2.0              # was 1.0; *2.0 weight absorbed here

    # waypoint and success rewards
    waypoint_reached_bonus: float = 70.0
    success_bonus: float = 10.0             # added post-division in step()

    # per-step penalties, positive values wit minus sign applied internally
    step_penalty: float = 0.15              # was 0.03; larger value speeds up policy

    # kinetic efficiency
    efficiency_bonus_scale: float = 0.1    # progress_rate * sqrt(speed)
    ke_reward_scale: float = 0.02          # unconditional sqrt(speed) bonus

    # action smoothness, anti-jerk
    action_smoothness_scale: float = 0.1
    angular_velocity_scale: float = 0.0    # disabled by default to match new reward

    # reward scaling
    reward_divisor: float = 3.0            # total step reward divided by this

    # camera alignment, yaw toward target
    camera_alignment_scale: float = 0.0
    camera_alignment_proximity_min: float = 0.3
    camera_alignment_proximity_range: float = 30.0

    # proximity
    proximity_reward_scale: float = 0.0
    proximity_falloff: float = 10.0

    # termination conditions
    success_steps_in_tolerance: int = 1    # waypoint mode
    success_steps_in_tolerance_traj: int = 64  # trajectory mode

    def __post_init__(self):
        assert self.shaping_scale >= 0,           "shaping_scale must be non-negative"
        assert self.waypoint_reached_bonus >= 0,  "waypoint_reached_bonus must be non-negative"
        assert self.success_bonus >= 0,           "success_bonus must be non-negative"
        assert self.step_penalty >= 0,            "step_penalty should be positive (subtracted each step)"
        assert self.action_smoothness_scale >= 0, "action_smoothness_scale must be non-negative"
        assert self.angular_velocity_scale >= 0,  "angular_velocity_scale must be non-negative"
        assert self.efficiency_bonus_scale >= 0,  "efficiency_bonus_scale must be non-negative"
        assert self.ke_reward_scale >= 0,         "ke_reward_scale must be non-negative"
        assert self.reward_divisor > 0,           "reward_divisor must be positive"


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
    initial_orientation_range: float = 2 * np.pi