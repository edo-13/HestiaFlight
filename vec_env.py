import torch
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np

from phy import BatchedPhysicsEnv, quaternion_to_rotation_matrix
from config import TrainingConfig, RewardConfig, SpawnConfig

class RunningMeanStd:
    """Track running mean and std for online normalization"""
    
    def __init__(self, shape, device='cpu', epsilon=1e-4):
        self.device = torch.device(device)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = epsilon
        
    def update(self, x: torch.Tensor):
        """Update statistics with batch of observations (expects tensor on same device)"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)
            
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from pre-computed moments (Welford's algorithm)"""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        """Normalize tensor with running statistics"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -clip, clip)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Undo normalization"""
        return x * torch.sqrt(self.var + 1e-8) + self.mean
    
    def save(self, path):
        """Save statistics to file"""
        torch.save({
            'mean': self.mean.cpu(),
            'var': self.var.cpu(),
            'count': self.count
        }, path)
    
    def load(self, path):
        """Load statistics from file"""
        data = torch.load(path, map_location=self.device)
        self.mean = data['mean'].to(self.device)
        self.var = data['var'].to(self.device)
        self.count = data['count']
    
    def to(self, device):
        """Move statistics to device"""
        self.device = torch.device(device)
        self.mean = self.mean.to(self.device)
        self.var = self.var.to(self.device)
        return self


class VecEnv:
    """
    Fully vectorized quadrotor environment - all operations on GPU tensors.
    
    Observation space (27 dimensions):
        [0:3]   - World position (x, y, z)
        [3:6]   - World velocity (vx, vy, vz)
        [6:9]   - Gravity vector in body frame
        [9:12]  - Angular velocity in body frame (p, q, r)
        [12:15] - Linear acceleration (from physics)
        [15:18] - Relative position to current waypoint (body frame)
        [18:21] - Relative velocity to target (body frame)
        [21]    - Distance to current waypoint
        [22]    - Closure rate (negative = approaching)
        [23:26] - Next waypoint position (body frame, for look-ahead)
        [26]    - Path curvature indicator [0, 1]
    
    Action space (4 dimensions):
        [0]     - Collective thrust [-1, 1] -> [0, 1] normalized
        [1:4]   - Body rate commands (roll, pitch, yaw) [-1, 1] -> [-MAX_RATE, MAX_RATE]
    """
    
    # observation dimension for external reference
    OBS_DIM: int = 27
    ACTION_DIM: int = 4
    
    def __init__(
        self, 
        stage_names, 
        num_envs: int, 
        trajectory_manager, 
        config: Optional[TrainingConfig], 
        adr=None, 
        agent=None, 
        waypoint_manager=None,
        reward_config: Optional[RewardConfig] = None,
        spawn_config: Optional[SpawnConfig] = None,
    ):
        self.num_envs = num_envs
        self.device = torch.device(config.device)
        self.config = config
        self.agent = agent
        self.adr = adr
        self.trajectory_manager = trajectory_manager
        self.waypoint_manager = waypoint_manager
        
        # reward and spawn configuration
        self.reward_cfg = reward_config or RewardConfig()
        self.spawn_cfg = spawn_config or SpawnConfig()
        
        # state tensors
        self.pos = torch.zeros((num_envs, 3), device=self.device)
        self.vel = torch.zeros((num_envs, 3), device=self.device)
        self.quat = torch.zeros((num_envs, 4), device=self.device)
        self.quat[:, 0] = 1.0  # identity quaternion (w, x, y, z)
        self.omega = torch.zeros((num_envs, 3), device=self.device)
        
        # task state buffers
        self.max_waypoints = 100 
        self.max_traj_len = 1000
        
        self.env_waypoints = torch.zeros((num_envs, self.max_waypoints, 3), device=self.device)
        self.env_traj_pos = torch.zeros((num_envs, self.max_traj_len, 3), device=self.device)
        self.env_traj_vel = torch.zeros((num_envs, self.max_traj_len, 3), device=self.device)
        self.env_path_len = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        self.target_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.current_waypoint_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        # episode tracking
        self.episode_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.steps_within_tolerance = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.prev_potential = torch.zeros(num_envs, device=self.device)
        self.prev_action = torch.zeros((num_envs, 4), device=self.device)
        self.prev_vel = torch.zeros((num_envs, 3), device=self.device)
        
        # stage assignment
        self.stage_names = self._assign_stages(stage_names, num_envs)
        
        # physics engine
        self.physics = BatchedPhysicsEnv(num_envs, dt=0.01, device=config.device)
        self.tolerance = 5.0
        
        # epoch statistics
        self.epoch_successes = 0
        self.epoch_episodes = 0
        self.epoch_waypoints_reached = []
        
        # preallocated buffers for reward computation
        self._reward_buffer = torch.zeros(num_envs, device=self.device)
        self.rms = RunningMeanStd(shape=(), device=self.device)
        
        self.reset_all()

    def _assign_stages(self, stage_names, num_envs: int) -> List[str]:
        """Distribute environments across training stages."""
        if isinstance(stage_names, str):
            return [stage_names] * num_envs
        
        per_stage = num_envs // len(stage_names)
        stages = []
        for i, s in enumerate(stage_names):
            count = per_stage + (1 if i < num_envs % len(stage_names) else 0)
            stages.extend([s] * count)
        return stages

    def set_reward_config(self, reward_config: RewardConfig) -> None:
        """Update reward configuration at runtime."""
        self.reward_cfg = reward_config
    
    def get_reward_config(self) -> RewardConfig:
        """Get current reward configuration."""
        return self.reward_cfg

    def reset(self) -> torch.Tensor:
        """Returns observation tensor on GPU."""
        return self._get_obs()

    def reset_all(self) -> None:
        """Reset all environments."""
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_envs(all_ids)

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset specified environments."""
        if len(env_ids) == 0:
            return
        
        # apply domain randomization if enabled and existing
        if self.adr and self.config.domain_randomization.enabled:
            dr_params = self.adr.randomize_batch(env_ids, self.device)
            self.physics.apply_randomization_batch(env_ids, dr_params)
        else:
            self.physics.apply_randomization_batch(env_ids, None)
        self.physics.reset(env_ids)

        num_resets = len(env_ids)
        target_starts = torch.zeros((num_resets, 3), device=self.device)
        
        # group resets by stage
        reset_indices_list = env_ids.tolist()
        stages_to_process = {}

        for i, global_idx in enumerate(reset_indices_list):
            stage = self.stage_names[global_idx]
            if stage not in stages_to_process:
                stages_to_process[stage] = []
            stages_to_process[stage].append(i)

        # fetch trajectories/waypoints per stage
        for stage, batch_indices in stages_to_process.items():
            count = len(batch_indices)
            batch_idx_tensor = torch.tensor(batch_indices, device=self.device, dtype=torch.long)
            global_ids = env_ids[batch_idx_tensor]
            
            if self.config.use_waypoint_mode and self.waypoint_manager:
                batch_data = self.waypoint_manager.get_training_batch(stage, count, self.device)
                wps = batch_data['waypoints']
                len_wps = wps.shape[1]
                
                self.env_waypoints[global_ids] = 0.0
                max_len = min(len_wps, self.max_waypoints)
                self.env_waypoints[global_ids, :max_len] = wps[:, :max_len]
                
                self.env_path_len[global_ids] = max_len
                self.current_waypoint_idx[global_ids] = 0
                target_starts[batch_idx_tensor] = wps[:, 0]
                
            elif self.trajectory_manager:
                batch_data = self.trajectory_manager.get_training_batch(stage, count, self.device)
                pos = batch_data['positions']
                vel = batch_data['velocities']
                len_traj = min(pos.shape[1], self.max_traj_len)
                
                self.env_traj_pos[global_ids, :len_traj] = pos[:, :len_traj]
                self.env_traj_vel[global_ids, :len_traj] = vel[:, :len_traj]
                self.env_path_len[global_ids] = len_traj
                self.target_idx[global_ids] = 0
                target_starts[batch_idx_tensor] = pos[:, 0]

        # spawn positions at random offset from first waypoint
        self._spawn_positions(env_ids, target_starts, num_resets)
        
        # reset episode state
        self.episode_steps[env_ids] = 0
        self.steps_within_tolerance[env_ids] = 0
        self.prev_potential[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0
        self.prev_vel[env_ids] = self.vel[env_ids]

        if hasattr(self, 'agent') and self.agent is not None:
            self.agent.reset_action_state(env_ids.tolist())

    def _spawn_positions(
        self, 
        env_ids: torch.Tensor, 
        target_starts: torch.Tensor, 
        num_resets: int
    ) -> None:
        """Initialize spawn positions and orientations."""
        cfg = self.spawn_cfg
        
        # random distance from target
        radius = torch.randn(num_resets, device=self.device) * cfg.spawn_radius_std + cfg.spawn_radius_mean
        radius = torch.clamp(radius, 0.0, cfg.spawn_radius_max)
        
        # uniform spherical distribution
        theta = torch.rand(num_resets, device=self.device) * 2 * np.pi
        phi = torch.rand(num_resets, device=self.device) * np.pi
        
        offset = torch.stack([
            radius * torch.sin(phi) * torch.cos(theta),
            radius * torch.sin(phi) * torch.sin(theta),
            radius * torch.cos(phi)
        ], dim=1)

        self.pos[env_ids] = target_starts + offset
        
        # random initial velocity
        self.vel[env_ids] = torch.clamp(
            torch.randn((num_resets, 3), device=self.device) * cfg.initial_velocity_std,
            -cfg.initial_velocity_clamp,
            cfg.initial_velocity_clamp
        )
        self.omega[env_ids] = 0.0
        
        # random orientation (axis-angle -> quaternion)
        axis = torch.randn((num_resets, 3), device=self.device)
        axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)
        angle = (torch.rand(num_resets, device=self.device) - 0.5) * cfg.initial_orientation_range
        half = angle / 2
        
        self.quat[env_ids] = torch.stack([
            torch.cos(half),
            torch.sin(half) * axis[:, 0],
            torch.sin(half) * axis[:, 1],
            torch.sin(half) * axis[:, 2]
        ], dim=1)
        self.quat[env_ids] /= torch.norm(self.quat[env_ids], dim=1, keepdim=True)

    def step(
        self, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Step environment forward.
        
        Args:
            actions: (num_envs, 4) tensor of actions
            
        Returns:
            obs: (num_envs, obs_dim) observation tensor
            rewards: (num_envs,) reward tensor
            terminated: (num_envs,) bool tensor - true termination (goal reached)
            truncated: (num_envs,) bool tensor - timeout without success
            infos: list of dicts (only populated for done envs)
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
        
        # physics step
        state = {
            'position': self.pos, 
            'velocity': self.vel, 
            'quaternion': self.quat, 
            'angular_velocity': self.omega
        }
        res = self.physics.step(state, actions)
        
        self.pos = res['position']
        self.vel = res['velocity']
        self.quat = res['quaternion']
        self.omega = res['angular_velocity']
        self.episode_steps += 1
        
        # get current targets and compute target advancement
        target_pos, target_vel, wp_bonus = self._update_targets()
        
        # compute rewards
        rewards, dist_target = self._compute_rewards(actions, target_pos, target_vel, wp_bonus)
        
        # check termination conditions
        terminated, truncated, success = self._check_termination(dist_target)
        done = terminated | truncated
        
        # add success bonus
        rewards[success] += self.reward_cfg.success_bonus
        
        # handle episode resets
        infos = self._handle_dones(done, success, truncated, dist_target)
        
        # update state for next step
        self.prev_action = actions.clone()
        self.prev_vel = self.vel.clone()

        self.rms.update(rewards)
        rewards = self.rms.normalize(rewards) # normalizee rewards for better Critic convergence
        
        return self._get_obs(), rewards, terminated, truncated, infos

    def _update_targets(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update target indices and return current targets.
        
        Returns:
            target_pos: (num_envs, 3) current target position
            target_vel: (num_envs, 3) current target velocity
            wp_bonus: (num_envs,) waypoint reached bonus
        """
        batch_indices = torch.arange(self.num_envs, device=self.device)
        wp_bonus = torch.zeros(self.num_envs, device=self.device)
        
        if self.config.use_waypoint_mode:
            current_wps = self.env_waypoints[batch_indices, self.current_waypoint_idx]
            dist_to_wp = torch.norm(current_wps - self.pos, dim=1)
            
            # check waypoint advancement
            reached = dist_to_wp < self.tolerance
            can_advance = self.current_waypoint_idx < (self.env_path_len - 1)
            advance_mask = reached & can_advance
            
            self.current_waypoint_idx[advance_mask] += 1
            wp_bonus[advance_mask] = self.reward_cfg.waypoint_reached_bonus
            
            target_pos = self.env_waypoints[batch_indices, self.current_waypoint_idx]
            target_vel = torch.zeros_like(target_pos)
        else:
            # continuous trajectory tracking
            self.target_idx = torch.minimum(self.target_idx + 1, self.env_path_len - 1)
            target_pos = self.env_traj_pos[batch_indices, self.target_idx]
            target_vel = self.env_traj_vel[batch_indices, self.target_idx]
            
        return target_pos, target_vel, wp_bonus

    def _compute_rewards(
        self,
        actions: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        wp_bonus: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards for all environments.
        
        Args:
            actions: Current actions
            target_pos: Target positions
            target_vel: Target velocities  
            wp_bonus: Waypoint advancement bonuses
            
        Returns:
            rewards: (num_envs,) total reward
            dist_target: (num_envs,) distance to target (for termination check)
        """
        cfg = self.reward_cfg
        
        # distance-based potential shaping
        dist_target = torch.norm(target_pos - self.pos, dim=1)
        potential = -dist_target
        
        # initialize potential on first step
        first_step = self.episode_steps == 1
        self.prev_potential[first_step] = potential[first_step]
        
        shaping = (potential - self.prev_potential) * cfg.shaping_scale
        self.prev_potential = potential
        
        # action smoothness penalty
        action_diff = torch.norm(actions - self.prev_action, dim=1)
        smoothness_penalty = -cfg.action_smoothness_scale * action_diff
        
        # angular velocity penalty
        ang_vel_penalty = -cfg.angular_velocity_scale * torch.norm(self.omega, dim=1)
        
        # camera alignment reward
        camera_reward = torch.zeros(self.num_envs, device=self.device)
        if cfg.camera_alignment_scale > 0:
            camera_reward = self._compute_camera_alignment(target_pos, dist_target)

        proximity_reward = cfg.proximity_reward_scale * torch.exp(-dist_target / cfg.proximity_falloff)
        
        rewards = (
            shaping 
            + wp_bonus
            + proximity_reward 
            + smoothness_penalty 
            + ang_vel_penalty 
            + camera_reward 
            - cfg.step_penalty
        )
        
        return rewards, dist_target

    def _compute_camera_alignment(
        self, 
        target_pos: torch.Tensor, 
        dist_target: torch.Tensor
    ) -> torch.Tensor:
        """Compute reward for pointing camera toward target."""
        cfg = self.reward_cfg
        
        R = quaternion_to_rotation_matrix(self.quat)
        body_forward = R[:, :, 0]  # x-axis = camera forward
        
        # direction to target (horizontal only for yaw alignment)
        dir_to_target = target_pos - self.pos
        dir_to_target_norm = dir_to_target / (dist_target.unsqueeze(-1) + 1e-8)
        
        # project to horizontal plane
        body_fwd_h = body_forward.clone()
        body_fwd_h[:, 2] = 0
        body_fwd_h = body_fwd_h / (torch.norm(body_fwd_h, dim=1, keepdim=True) + 1e-8)
        
        target_dir_h = dir_to_target_norm.clone()
        target_dir_h[:, 2] = 0
        target_dir_h = target_dir_h / (torch.norm(target_dir_h, dim=1, keepdim=True) + 1e-8)
        
        alignment = torch.sum(body_fwd_h * target_dir_h, dim=1)
        
        # stronger alignment reward when close
        proximity_factor = torch.clamp(
            1.0 - dist_target / cfg.camera_alignment_proximity_range,
            cfg.camera_alignment_proximity_min,
            1.0
        )
        
        return cfg.camera_alignment_scale * alignment * proximity_factor

    def _check_termination(
        self, 
        dist_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Check termination conditions.
        
        Returns:
            terminated: True termination (goal reached)
            truncated: Timeout without success
            success: Success mask (subset of terminated)
        """
        cfg = self.reward_cfg
        
        # track time within tolerance
        in_tol = dist_target <= self.tolerance
        self.steps_within_tolerance[in_tol] += 1
        self.steps_within_tolerance[~in_tol] = 0
        
        # success conditions differ by mode
        if self.config.use_waypoint_mode:
            at_last = self.current_waypoint_idx == (self.env_path_len - 1)
            success = at_last & (self.steps_within_tolerance >= cfg.success_steps_in_tolerance)
        else:
            success = self.steps_within_tolerance >= cfg.success_steps_in_tolerance_traj
            
        # timeout
        timeout = self.episode_steps >= self.config.max_episode_steps
        
        terminated = success
        truncated = timeout & ~success
        
        return terminated, truncated, success

    def _handle_dones(
        self,
        done: torch.Tensor,
        success: torch.Tensor,
        truncated: torch.Tensor,
        dist_target: torch.Tensor,
    ) -> List[Dict]:
        """Handle episode terminations and return info dicts."""
        done_idxs = torch.nonzero(done).squeeze(-1)
        infos = [{} for _ in range(self.num_envs)]
        
        if len(done_idxs) > 0:
            for idx in done_idxs:
                i = idx.item()
                infos[i] = {
                    'episode_success': success[idx].item(),
                    'distance': dist_target[idx].item(),
                    'waypoint_idx': self.current_waypoint_idx[idx].item() + 1,
                    'truncated': truncated[idx].item(),
                    'episode_length': self.episode_steps[idx].item(),
                }
                
                # update epoch statistics
                self.epoch_episodes += 1
                if success[idx].item():
                    self.epoch_successes += 1
                if self.config.use_waypoint_mode:
                    self.epoch_waypoints_reached.append(infos[i]['waypoint_idx'])

                # report to ADR if enabled
                if hasattr(self, 'adr') and self.adr is not None:
                    episode_score = 1.0 if success[idx].item() else 0.0
                    self.adr.record_episode(i, episode_score)

            self._reset_envs(done_idxs)
        
        return infos

    def _get_obs(self) -> torch.Tensor:
        """
        Construct observation tensor.
        
        Returns:
            obs: (num_envs, 27) observation tensor
        """
        batch_indices = torch.arange(self.num_envs, device=self.device)
        
        # current and next waypoints
        if self.config.use_waypoint_mode:
            current_wp = self.env_waypoints[batch_indices, self.current_waypoint_idx]
            
            next_wp_idx = torch.minimum(
                self.current_waypoint_idx + 1, 
                self.env_path_len - 1
            )
            next_wp = self.env_waypoints[batch_indices, next_wp_idx]
            
            # path curvature estimation
            curvature = self._compute_path_curvature(current_wp, next_wp_idx)
            target_vel = torch.zeros_like(self.pos)
        else:
            current_wp = self.env_traj_pos[batch_indices, self.target_idx]
            next_wp = current_wp
            target_vel = self.env_traj_vel[batch_indices, self.target_idx]
            curvature = torch.zeros(self.num_envs, 1, device=self.device)
        
        # transform to body frame
        R = quaternion_to_rotation_matrix(self.quat)
        R_T = R.transpose(1, 2)
        
        rel_pos_world = current_wp - self.pos
        rel_vel_world = target_vel - self.vel
        
        rel_pos_body = torch.bmm(R_T, rel_pos_world.unsqueeze(-1)).squeeze(-1)
        rel_vel_body = torch.bmm(R_T, rel_vel_world.unsqueeze(-1)).squeeze(-1)
        
        next_wp_world = next_wp - self.pos
        next_wp_body = torch.bmm(R_T, next_wp_world.unsqueeze(-1)).squeeze(-1)
        
        # gravity in body frame
        gravity_world = torch.tensor([0., 0., -1.], device=self.device).expand(self.num_envs, 3)
        gravity_body = torch.bmm(R_T, gravity_world.unsqueeze(-1)).squeeze(-1)
        
        # distance metrics
        distance = torch.norm(rel_pos_world, dim=1, keepdim=True)
        closure_rate = torch.sum(rel_vel_world * rel_pos_world, dim=1, keepdim=True) / (distance + 1e-8)
        
        obs = torch.cat([
            self.pos,                 # [0:3]   world position
            self.vel,                 # [3:6]   world velocity
            gravity_body,             # [6:9]   gravity in body frame
            self.omega,               # [9:12]  angular velocity
            self.physics.last_accel,  # [12:15] acceleration
            rel_pos_body,             # [15:18] relative pos to waypoint (body)
            rel_vel_body,             # [18:21] relative velocity (body)
            distance,                 # [21]    distance to waypoint
            closure_rate,             # [22]    closure rate
            next_wp_body,             # [23:26] next waypoint (body)
            curvature,                # [26]    path curvature
        ], dim=1)
        
        return obs

    def _compute_path_curvature(
        self, 
        current_wp: torch.Tensor, 
        next_wp_idx: torch.Tensor
    ) -> torch.Tensor:
        """Compute path curvature for look-ahead."""
        batch_indices = torch.arange(self.num_envs, device=self.device)
        
        next_wp = self.env_waypoints[batch_indices, next_wp_idx]
        next_next_idx = torch.minimum(next_wp_idx + 1, self.env_path_len - 1)
        next_next_wp = self.env_waypoints[batch_indices, next_next_idx]
        
        vec1 = next_wp - current_wp
        vec2 = next_next_wp - next_wp
        
        norm1 = torch.norm(vec1, dim=1)
        norm2 = torch.norm(vec2, dim=1)
        
        curvature = torch.zeros(self.num_envs, device=self.device)
        valid_mask = (norm1 > 1e-3) & (norm2 > 1e-3)
        
        if valid_mask.any():
            vec1_norm = vec1[valid_mask] / norm1[valid_mask].unsqueeze(-1)
            vec2_norm = vec2[valid_mask] / norm2[valid_mask].unsqueeze(-1)
            cos_angle = torch.sum(vec1_norm * vec2_norm, dim=1)
            curvature[valid_mask] = (1.0 - cos_angle) * 0.5
        
        return curvature.unsqueeze(-1)

    def get_privileged_states(self) -> torch.Tensor:
        """Returns privileged states for asymmetric training."""
        batch_indices = torch.arange(self.num_envs, device=self.device)
        
        if self.config.use_waypoint_mode:
            target_pos = self.env_waypoints[batch_indices, self.current_waypoint_idx]
            target_vel = torch.zeros_like(self.pos)
        else:
            target_pos = self.env_traj_pos[batch_indices, self.target_idx]
            target_vel = self.env_traj_vel[batch_indices, self.target_idx]

        # physics parameters
        p_mass = self.physics.mass
        p_drag = self.physics.drag_coef
        p_thrust = self.physics.thrust_to_weight
        p_wind = self.physics.wind_velocity
        p_air = self.physics.air_density
        dt = self.physics.dt
        
        # turbulence estimation
        actual_accel = (self.vel - self.prev_vel) / dt.unsqueeze(-1)
        turb_est = (actual_accel - self.physics.last_accel) * 0.5
        
        # acceleration estimate
        rel_pos = target_pos - self.pos
        dist = torch.norm(rel_pos, dim=1)
        time_to_target = dist / (torch.norm(target_vel, dim=1) + 1e-3)
        req_accel_vec = rel_pos / (time_to_target.unsqueeze(1)**2 + 1e-3)
        req_accel = torch.clamp(torch.norm(req_accel_vec, dim=1) / 20.0, 0, 1)
        
        # path curvature
        curvature = torch.zeros(self.num_envs, device=self.device)
        if self.config.use_waypoint_mode:
            next_idxs = torch.minimum(self.current_waypoint_idx + 1, self.env_path_len - 1)
            next_wps = self.env_waypoints[batch_indices, next_idxs]
            
            vec1 = target_pos - self.pos
            vec2 = next_wps - target_pos
            
            norm1 = torch.norm(vec1, dim=1)
            norm2 = torch.norm(vec2, dim=1)
            
            valid = (norm1 > 1e-3) & (norm2 > 1e-3) & (next_idxs > self.current_waypoint_idx)
            if valid.any():
                cos = torch.sum(
                    (vec1[valid] / norm1[valid].unsqueeze(1)) * 
                    (vec2[valid] / norm2[valid].unsqueeze(1)), 
                    dim=1
                )
                curvature[valid] = (1.0 - cos) * 0.5

        return torch.stack([
            p_mass, p_drag, p_thrust,
            p_wind[:, 0], p_wind[:, 1], p_wind[:, 2],
            p_air,
            turb_est[:, 0], turb_est[:, 1], turb_est[:, 2],
            req_accel, curvature
        ], dim=1)

    def set_tolerance(self, tol: float) -> None:
        """Set waypoint tolerance radius."""
        self.tolerance = tol
    
    def get_epoch_success_rate(self) -> float:
        """Get success rate for current epoch."""
        return self.epoch_successes / max(self.epoch_episodes, 1)
        
    def reset_epoch_stats(self) -> None:
        """Reset epoch statistics."""
        self.epoch_successes = 0
        self.epoch_episodes = 0
        self.epoch_waypoints_reached = []