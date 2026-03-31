import math
import torch
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field

from phy_2 import BatchedPhysicsEnv, quaternion_to_rotation_matrix
from config import TrainingConfig, RewardConfig, SpawnConfig


class RunningMeanStd:
    """Track running mean and std for online normalization."""

    def __init__(self, shape, device="cpu", epsilon=1e-4):
        self.device = torch.device(device)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = epsilon

    def update(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
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
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)
        return torch.clamp(
            (x - self.mean) / torch.sqrt(self.var + 1e-8), -clip, clip
        )
    
    def normalize_variance_only(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)
        return torch.clamp(x / torch.sqrt(self.var + 1e-8), -clip, clip)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(self.var + 1e-8) + self.mean

    def save(self, path):
        torch.save(
            {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count},
            path,
        )

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.mean = data["mean"].to(self.device)
        self.var = data["var"].to(self.device)
        self.count = data["count"]

    def to(self, device):
        self.device = torch.device(device)
        self.mean = self.mean.to(self.device)
        self.var = self.var.to(self.device)
        return self


class VecEnv:
    """
    Fully vectorized quadrotor environment — all operations on GPU tensors.

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

        self.reward_cfg = reward_config or RewardConfig()
        self.spawn_cfg = spawn_config or SpawnConfig()

        # state tensors
        self.pos = torch.zeros((num_envs, 3), device=self.device)
        self.vel = torch.zeros((num_envs, 3), device=self.device)
        self.quat = torch.zeros((num_envs, 4), device=self.device)
        self.quat[:, 0] = 1.0
        self.omega = torch.zeros((num_envs, 3), device=self.device)

        # task state buffers
        self.max_waypoints = 100
        self.max_traj_len = 1000

        self.env_waypoints = torch.zeros(
            (num_envs, self.max_waypoints, 3), device=self.device
        )
        self.env_traj_pos = torch.zeros(
            (num_envs, self.max_traj_len, 3), device=self.device
        )
        self.env_traj_vel = torch.zeros(
            (num_envs, self.max_traj_len, 3), device=self.device
        )
        self.env_path_len = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )

        self.target_idx = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self.current_waypoint_idx = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )

        # episode tracking
        self.episode_steps = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self.steps_within_tolerance = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self.prev_potential = torch.zeros(num_envs, device=self.device)
        self.prev_action = torch.zeros((num_envs, 4), device=self.device)
        self.prev_vel = torch.zeros((num_envs, 3), device=self.device)

        # ----------------------------------------------------------------
        # Stage assignment — GPU integer tensor, no Python list
        # ----------------------------------------------------------------
        unique_stages, self.stage_ids = self._assign_stages_gpu(
            stage_names, num_envs
        )
        self.unique_stages = unique_stages  # list[str] for name lookup only
        # stage_ids: (num_envs,) long tensor on device — each value is an index
        # into unique_stages

        # physics engine
        self.physics = BatchedPhysicsEnv(num_envs, dt=0.01, device=config.device)
        self.tolerance = 5.0

        # epoch statistics (GPU counters, sync only at epoch boundary)
        self._epoch_successes = torch.zeros(1, dtype=torch.long, device=self.device)
        self._epoch_episodes = torch.zeros(1, dtype=torch.long, device=self.device)
        self._epoch_wp_reached_sum = torch.zeros(
            1, dtype=torch.long, device=self.device
        )
        self._epoch_wp_reached_count = torch.zeros(
            1, dtype=torch.long, device=self.device
        )

        # preallocated buffers
        self._reward_buffer = torch.zeros(num_envs, device=self.device)
        self.rms = RunningMeanStd(shape=(), device=self.device)

        self.reset_all()

    # ----------------------------------------------------------------
    # Stage helpers (GPU-resident)
    # ----------------------------------------------------------------

    @staticmethod
    def _assign_stages_gpu(
        stage_names, num_envs: int
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Map stage names to integer IDs on GPU.

        Returns:
            unique_stages: deduplicated list of stage name strings
            stage_ids: (num_envs,) long tensor on CPU (moved to device by caller)
        """
        if isinstance(stage_names, str):
            return [stage_names], torch.zeros(num_envs, dtype=torch.long)

        unique = list(dict.fromkeys(stage_names))  # preserves order
        name_to_id = {n: i for i, n in enumerate(unique)}

        per_stage = num_envs // len(stage_names)
        ids = []
        for i, s in enumerate(stage_names):
            count = per_stage + (1 if i < num_envs % len(stage_names) else 0)
            ids.extend([name_to_id[s]] * count)

        return unique, torch.tensor(ids, dtype=torch.long)

    def set_reward_config(self, reward_config: RewardConfig) -> None:
        self.reward_cfg = reward_config

    def get_reward_config(self) -> RewardConfig:
        return self.reward_cfg

    def reset(self) -> torch.Tensor:
        return self._get_obs()

    def reset_all(self) -> None:
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_envs(all_ids)

    # ----------------------------------------------------------------
    # Reset — fully GPU, no .tolist() / .item()
    # ----------------------------------------------------------------

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset specified environments (GPU-only path)."""
        if len(env_ids) == 0:
            return

        # domain randomization
        if self.adr and self.config.domain_randomization.enabled:
            dr_params = self.adr.randomize_batch(env_ids, self.device)
            self.physics.apply_randomization_batch(env_ids, dr_params)
        else:
            self.physics.apply_randomization_batch(env_ids, None)
        self.physics.reset(env_ids)

        num_resets = len(env_ids)
        target_starts = torch.zeros((num_resets, 3), device=self.device)

        # group by stage using GPU masking — no Python loop over envs
        env_stage_ids = self.stage_ids.to(self.device)[env_ids]

        for stage_int_id, stage_name in enumerate(self.unique_stages):
            mask = env_stage_ids == stage_int_id
            if not mask.any():
                continue

            local_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            global_ids = env_ids[local_indices]
            count = local_indices.shape[0]

            if self.config.use_waypoint_mode and self.waypoint_manager:
                batch_data = self.waypoint_manager.get_training_batch(
                    stage_name, count, self.device
                )
                wps = batch_data["waypoints"]
                max_len = min(wps.shape[1], self.max_waypoints)

                self.env_waypoints[global_ids] = 0.0
                self.env_waypoints[global_ids, :max_len] = wps[:, :max_len]
                self.env_path_len[global_ids] = max_len
                self.current_waypoint_idx[global_ids] = 0
                target_starts[local_indices] = wps[:, 0]

            elif self.trajectory_manager:
                batch_data = self.trajectory_manager.get_training_batch(
                    stage_name, count, self.device
                )
                pos = batch_data["positions"]
                vel = batch_data["velocities"]
                len_traj = min(pos.shape[1], self.max_traj_len)

                self.env_traj_pos[global_ids, :len_traj] = pos[:, :len_traj]
                self.env_traj_vel[global_ids, :len_traj] = vel[:, :len_traj]
                self.env_path_len[global_ids] = len_traj
                self.target_idx[global_ids] = 0
                target_starts[local_indices] = pos[:, 0]

        self._spawn_positions(env_ids, target_starts, num_resets)

        # reset episode state (all tensor ops)
        self.episode_steps[env_ids] = 0
        self.steps_within_tolerance[env_ids] = 0
        self.prev_potential[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0
        self.prev_vel[env_ids] = self.vel[env_ids]

        if self.agent is not None:
            # This is the one remaining CPU touch — agent RNN state reset.
            # If your agent stores hidden states as a GPU tensor indexed by
            # env id, replace this with a pure-tensor op.
            self.agent.reset_state(env_ids)

    def _spawn_positions(
        self,
        env_ids: torch.Tensor,
        target_starts: torch.Tensor,
        num_resets: int,
    ) -> None:
        """Initialize spawn positions and orientations (GPU-only)."""
        cfg = self.spawn_cfg

        radius = (
            torch.randn(num_resets, device=self.device) * cfg.spawn_radius_std
            + cfg.spawn_radius_mean
        )
        radius = torch.clamp(radius, 0.0, cfg.spawn_radius_max)

        theta = torch.rand(num_resets, device=self.device) * (2.0 * math.pi)
        phi = torch.rand(num_resets, device=self.device) * math.pi

        offset = torch.stack(
            [
                radius * torch.sin(phi) * torch.cos(theta),
                radius * torch.sin(phi) * torch.sin(theta),
                radius * torch.cos(phi),
            ],
            dim=1,
        )

        self.pos[env_ids] = target_starts + offset

        self.vel[env_ids] = torch.clamp(
            torch.randn((num_resets, 3), device=self.device) * cfg.initial_velocity_std,
            -cfg.initial_velocity_clamp,
            cfg.initial_velocity_clamp,
        )
        self.omega[env_ids] = 0.0

        axis = torch.randn((num_resets, 3), device=self.device)
        axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)
        angle = (
            (torch.rand(num_resets, device=self.device) - 0.5)
            * cfg.initial_orientation_range
        )
        half = angle / 2

        self.quat[env_ids] = torch.stack(
            [
                torch.cos(half),
                torch.sin(half) * axis[:, 0],
                torch.sin(half) * axis[:, 1],
                torch.sin(half) * axis[:, 2],
            ],
            dim=1,
        )
        self.quat[env_ids] /= torch.norm(
            self.quat[env_ids], dim=1, keepdim=True
        )

    # ----------------------------------------------------------------
    # Step
    # ----------------------------------------------------------------

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step environment forward.

        Returns:
            obs, rewards, terminated, truncated, info_dict
            info_dict is a *tensorized* dict (GPU-resident), not a list of
            per-env Python dicts.
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.device, dtype=torch.float32)

        state = {
            "position": self.pos,
            "velocity": self.vel,
            "quaternion": self.quat,
            "angular_velocity": self.omega,
        }
        res = self.physics.step(state, actions)

        self.pos = res["position"]
        self.vel = res["velocity"]
        self.quat = res["quaternion"]
        self.omega = res["angular_velocity"]
        self.episode_steps += 1

        target_pos, target_vel, wp_bonus = self._update_targets()
        rewards, dist_target = self._compute_rewards(
            actions, target_pos, target_vel, wp_bonus
        )

        terminated, truncated, success = self._check_termination(dist_target)
        done = terminated | truncated

        rewards = rewards + success.float() * self.reward_cfg.success_bonus

        # tensorized info — stays on GPU
        info = self._handle_dones_tensorized(
            done, success, truncated, dist_target
        )

        self.prev_action = actions.clone()
        self.prev_vel = self.vel.clone()

        self.rms.update(rewards)
        rewards = self.rms.normalize_variance_only(rewards)

        return self._get_obs(), rewards, terminated, truncated, info

    # ----------------------------------------------------------------
    # Target update
    # ----------------------------------------------------------------

    def _update_targets(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_indices = torch.arange(self.num_envs, device=self.device)
        wp_bonus = torch.zeros(self.num_envs, device=self.device)

        if self.config.use_waypoint_mode:
            current_wps = self.env_waypoints[
                batch_indices, self.current_waypoint_idx
            ]
            dist_to_wp = torch.norm(current_wps - self.pos, dim=1)

            reached = dist_to_wp < self.tolerance
            can_advance = self.current_waypoint_idx < (self.env_path_len - 1)
            advance_mask = reached & can_advance

            self.current_waypoint_idx[advance_mask] += 1
            wp_bonus[advance_mask] = self.reward_cfg.waypoint_reached_bonus

            target_pos = self.env_waypoints[
                batch_indices, self.current_waypoint_idx
            ]
            target_vel = torch.zeros_like(target_pos)
        else:
            self.target_idx = torch.minimum(
                self.target_idx + 1, self.env_path_len - 1
            )
            target_pos = self.env_traj_pos[batch_indices, self.target_idx]
            target_vel = self.env_traj_vel[batch_indices, self.target_idx]

        return target_pos, target_vel, wp_bonus

    # ----------------------------------------------------------------
    # Rewards
    # ----------------------------------------------------------------

    def _compute_rewards(
        self,
        actions: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        wp_bonus: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.reward_cfg

        dist_target = torch.norm(target_pos - self.pos, dim=1)
        potential = -dist_target

        # === 1. SHAPING — primary learning signal (progress along path) ===
        # Initialise potential on the very first step so shaping is zero
        # rather than a large spurious spike.
        first_step = self.episode_steps == 1
        self.prev_potential[first_step] = potential[first_step]

        raw_shaping = potential - self.prev_potential          # unscaled delta
        self.prev_potential = potential.clone()

        shaping = raw_shaping * cfg.shaping_scale              # scaled version

        # === 2. KINETIC EFFICIENCY BONUS ===
        # Rewards speed that actually converts to path progress; zero bonus
        # for wasted speed (e.g. wobbling perpendicular to the path).
        speed = torch.norm(self.vel, dim=1)
        progress_rate = raw_shaping.clamp(min=0.0)
        efficiency_bonus = cfg.efficiency_bonus_scale * progress_rate * torch.sqrt(speed + 1e-6)

        # Small unconditional speed reward to discourage hovering when
        # momentarily off-path (kept very small to avoid reward hacking).
        ke_reward = cfg.ke_reward_scale * torch.sqrt(speed + 1e-6)

        # === 3. SMOOTHNESS PENALTY (actuator jerk) ===
        action_diff = torch.norm(actions - self.prev_action, dim=1)
        smoothness_penalty = -cfg.action_smoothness_scale * action_diff

        # === 4. ANGULAR VELOCITY PENALTY (optional — zero by default) ===
        ang_vel_penalty = -cfg.angular_velocity_scale * torch.norm(self.omega, dim=1)

        # === 5. CAMERA ALIGNMENT (optional — disabled by default) ===
        camera_reward = torch.zeros(self.num_envs, device=self.device)
        if cfg.camera_alignment_scale > 0:
            camera_reward = self._compute_camera_alignment(target_pos, dist_target)

        # === 6. PROXIMITY REWARD (optional — disabled by default) ===
        proximity_reward = cfg.proximity_reward_scale * torch.exp(
            -dist_target / cfg.proximity_falloff
        )

        # === TOTAL REWARD ===
        rewards = (
            wp_bonus            # one-time milestone bonus per waypoint
            + shaping           # primary: progress along path (scaled)
            - cfg.step_penalty  # constant time cost → finish fast
            + efficiency_bonus  # speed that produces progress
            + ke_reward         # small unconditional speed nudge
            + smoothness_penalty
            + ang_vel_penalty
            + camera_reward
            + proximity_reward
        )

        # Normalise magnitude so PPO value targets stay in a reasonable range.
        # Success bonus is added by the caller (step()) after this division,
        # matching the behaviour of the reference reward computation.
        rewards = rewards / cfg.reward_divisor

        return rewards, dist_target

    def _compute_camera_alignment(
        self, target_pos: torch.Tensor, dist_target: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.reward_cfg

        R = quaternion_to_rotation_matrix(self.quat)
        body_forward = R[:, :, 0]

        dir_to_target = target_pos - self.pos
        dir_to_target_norm = dir_to_target / (
            dist_target.unsqueeze(-1) + 1e-8
        )

        body_fwd_h = body_forward.clone()
        body_fwd_h[:, 2] = 0
        body_fwd_h = body_fwd_h / (
            torch.norm(body_fwd_h, dim=1, keepdim=True) + 1e-8
        )

        target_dir_h = dir_to_target_norm.clone()
        target_dir_h[:, 2] = 0
        target_dir_h = target_dir_h / (
            torch.norm(target_dir_h, dim=1, keepdim=True) + 1e-8
        )

        alignment = torch.sum(body_fwd_h * target_dir_h, dim=1)

        proximity_factor = torch.clamp(
            1.0 - dist_target / cfg.camera_alignment_proximity_range,
            cfg.camera_alignment_proximity_min,
            1.0,
        )

        return cfg.camera_alignment_scale * alignment * proximity_factor

    # ----------------------------------------------------------------
    # Termination
    # ----------------------------------------------------------------

    def _check_termination(
        self, dist_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.reward_cfg

        in_tol = dist_target <= self.tolerance
        self.steps_within_tolerance[in_tol] += 1
        self.steps_within_tolerance[~in_tol] = 0

        if self.config.use_waypoint_mode:
            at_last = self.current_waypoint_idx == (self.env_path_len - 1)
            success = at_last & (
                self.steps_within_tolerance >= cfg.success_steps_in_tolerance
            )
        else:
            success = (
                self.steps_within_tolerance
                >= cfg.success_steps_in_tolerance_traj
            )

        timeout = self.episode_steps >= self.config.max_episode_steps

        terminated = success
        truncated = timeout & ~success

        return terminated, truncated, success

    # ----------------------------------------------------------------
    # Done handling — fully tensorized, no Python loop
    # ----------------------------------------------------------------

    def _handle_dones_tensorized(
        self,
        done: torch.Tensor,
        success: torch.Tensor,
        truncated: torch.Tensor,
        dist_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Handle episode resets and return tensorized info.

        Returns a dict of tensors (all num_envs-sized). Consumers can mask
        with `done` to extract finished-episode data.
        """
        # build info tensors (cheap — just views / existing data)
        info = {
            "done": done,
            "episode_success": success,
            "distance": dist_target,
            "waypoint_idx": self.current_waypoint_idx + 1,
            "truncated": truncated,
            "episode_length": self.episode_steps,
        }

        done_idxs = torch.nonzero(done, as_tuple=False).squeeze(-1)

        if done_idxs.numel() > 0:
            # update epoch counters (all on GPU, no sync)
            num_done = done_idxs.shape[0]
            num_success = success[done_idxs].sum()

            self._epoch_episodes += num_done
            self._epoch_successes += num_success

            if self.config.use_waypoint_mode:
                self._epoch_wp_reached_sum += (
                    self.current_waypoint_idx[done_idxs] + 1
                ).sum()
                self._epoch_wp_reached_count += num_done

            # report to ADR — batched
            if self.adr is not None:
                scores = success[done_idxs].float()
                self.adr.record_episodes_batch(done_idxs, scores)

            self._reset_envs(done_idxs)

        return info

    # ----------------------------------------------------------------
    # Observations
    # ----------------------------------------------------------------

    def _get_obs(self) -> torch.Tensor:
        batch_indices = torch.arange(self.num_envs, device=self.device)

        if self.config.use_waypoint_mode:
            current_wp = self.env_waypoints[
                batch_indices, self.current_waypoint_idx
            ]

            next_wp_idx = torch.minimum(
                self.current_waypoint_idx + 1, self.env_path_len - 1
            )
            next_wp = self.env_waypoints[batch_indices, next_wp_idx]

            curvature = self._compute_path_curvature(current_wp, next_wp_idx)
            target_vel = torch.zeros_like(self.pos)
        else:
            current_wp = self.env_traj_pos[batch_indices, self.target_idx]
            next_wp = current_wp
            target_vel = self.env_traj_vel[batch_indices, self.target_idx]
            curvature = torch.zeros(
                self.num_envs, 1, device=self.device
            )

        R = quaternion_to_rotation_matrix(self.quat)
        R_T = R.transpose(1, 2)

        rel_pos_world = current_wp - self.pos
        rel_vel_world = target_vel - self.vel

        rel_pos_body = torch.bmm(R_T, rel_pos_world.unsqueeze(-1)).squeeze(-1)
        rel_vel_body = torch.bmm(R_T, rel_vel_world.unsqueeze(-1)).squeeze(-1)

        next_wp_world = next_wp - self.pos
        next_wp_body = torch.bmm(R_T, next_wp_world.unsqueeze(-1)).squeeze(-1)

        gravity_world = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device
        ).expand(self.num_envs, 3)
        gravity_body = torch.bmm(R_T, gravity_world.unsqueeze(-1)).squeeze(-1)

        distance = torch.norm(rel_pos_world, dim=1, keepdim=True)
        closure_rate = (
            torch.sum(rel_vel_world * rel_pos_world, dim=1, keepdim=True)
            / (distance + 1e-8)
        )

        return torch.cat(
            [
                self.pos,
                self.vel,
                gravity_body,
                self.omega,
                self.physics.last_accel,
                rel_pos_body,
                rel_vel_body,
                distance,
                closure_rate,
                next_wp_body,
                curvature,
            ],
            dim=1,
        )

    def _compute_path_curvature(
        self, current_wp: torch.Tensor, next_wp_idx: torch.Tensor
    ) -> torch.Tensor:
        batch_indices = torch.arange(self.num_envs, device=self.device)

        next_wp = self.env_waypoints[batch_indices, next_wp_idx]
        next_next_idx = torch.minimum(
            next_wp_idx + 1, self.env_path_len - 1
        )
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

    # ----------------------------------------------------------------
    # Privileged states
    # ----------------------------------------------------------------

    def get_privileged_states(self) -> torch.Tensor:
        batch_indices = torch.arange(self.num_envs, device=self.device)

        if self.config.use_waypoint_mode:
            target_pos = self.env_waypoints[
                batch_indices, self.current_waypoint_idx
            ]
            target_vel = torch.zeros_like(self.pos)
        else:
            target_pos = self.env_traj_pos[batch_indices, self.target_idx]
            target_vel = self.env_traj_vel[batch_indices, self.target_idx]

        p_mass = self.physics.mass
        p_drag = self.physics.drag_coef  # legacy property -> drag_coef_xy
        p_thrust = self.physics.thrust_to_weight
        p_wind = self.physics.wind_velocity
        p_air = self.physics.air_density
        dt = self.physics.dt

        actual_accel = (self.vel - self.prev_vel) / dt.unsqueeze(-1)
        turb_est = (actual_accel - self.physics.last_accel) * 0.5

        rel_pos = target_pos - self.pos
        dist = torch.norm(rel_pos, dim=1)
        time_to_target = dist / (torch.norm(target_vel, dim=1) + 1e-3)
        req_accel_vec = rel_pos / (time_to_target.unsqueeze(1) ** 2 + 1e-3)
        req_accel = torch.clamp(
            torch.norm(req_accel_vec, dim=1) / 20.0, 0, 1
        )

        curvature = torch.zeros(self.num_envs, device=self.device)
        if self.config.use_waypoint_mode:
            next_idxs = torch.minimum(
                self.current_waypoint_idx + 1, self.env_path_len - 1
            )
            next_wps = self.env_waypoints[batch_indices, next_idxs]

            vec1 = target_pos - self.pos
            vec2 = next_wps - target_pos

            norm1 = torch.norm(vec1, dim=1)
            norm2 = torch.norm(vec2, dim=1)

            valid = (
                (norm1 > 1e-3)
                & (norm2 > 1e-3)
                & (next_idxs > self.current_waypoint_idx)
            )
            if valid.any():
                cos = torch.sum(
                    (vec1[valid] / norm1[valid].unsqueeze(1))
                    * (vec2[valid] / norm2[valid].unsqueeze(1)),
                    dim=1,
                )
                curvature[valid] = (1.0 - cos) * 0.5

        return torch.stack(
            [
                p_mass, p_drag, p_thrust,
                p_wind[:, 0], p_wind[:, 1], p_wind[:, 2],
                p_air,
                turb_est[:, 0], turb_est[:, 1], turb_est[:, 2],
                req_accel, curvature,
            ],
            dim=1,
        )

    # ----------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------

    def set_tolerance(self, tol: float) -> None:
        self.tolerance = tol

    def get_epoch_success_rate(self) -> float:
        """Sync GPU counters to CPU (call once per epoch, not per step)."""
        episodes = self._epoch_episodes.item()
        successes = self._epoch_successes.item()
        return successes / max(episodes, 1)

    def get_epoch_mean_waypoints(self) -> float:
        """Average waypoints reached per episode this epoch."""
        count = self._epoch_wp_reached_count.item()
        if count == 0:
            return 0.0
        return self._epoch_wp_reached_sum.item() / count

    def reset_epoch_stats(self) -> None:
        self._epoch_episodes.zero_()
        self._epoch_successes.zero_()
        self._epoch_wp_reached_sum.zero_()
        self._epoch_wp_reached_count.zero_()

    # ----------------------------------------------------------------
    # Backward-compatibility shims
    # ----------------------------------------------------------------

    @property
    def epoch_successes(self) -> int:
        """Legacy accessor — syncs to CPU."""
        return self._epoch_successes.item()

    @epoch_successes.setter
    def epoch_successes(self, v: int) -> None:
        self._epoch_successes.fill_(v)

    @property
    def epoch_episodes(self) -> int:
        """Legacy accessor — syncs to CPU."""
        return self._epoch_episodes.item()

    @epoch_episodes.setter
    def epoch_episodes(self, v: int) -> None:
        self._epoch_episodes.fill_(v)

    @property
    def epoch_waypoints_reached(self) -> List[int]:
        """Legacy accessor — returns a Python list (triggers CPU sync)."""
        count = self._epoch_wp_reached_count.item()
        if count == 0:
            return []
        mean_wp = self._epoch_wp_reached_sum.item() / count
        # Best-effort reconstruction: the original stored one int per
        # episode.  We only track sum+count now, so return [mean]*count
        # which preserves len() and np.mean() behavior.
        return [mean_wp] * count

    @epoch_waypoints_reached.setter
    def epoch_waypoints_reached(self, v) -> None:
        if isinstance(v, list) and len(v) == 0:
            self._epoch_wp_reached_sum.zero_()
            self._epoch_wp_reached_count.zero_()
        else:
            self._epoch_wp_reached_sum.fill_(int(sum(v)))
            self._epoch_wp_reached_count.fill_(len(v))

    @property
    def stage_names(self) -> List[str]:
        """Legacy accessor — returns per-env list of stage name strings."""
        ids = self.stage_ids.tolist()
        return [self.unique_stages[i] for i in ids]

    @stage_names.setter
    def stage_names(self, v) -> None:
        # Accepts assignment from _assign_stages_gpu or a plain list
        if isinstance(v, list):
            unique, ids = self._assign_stages_gpu(v, len(v))
            self.unique_stages = unique
            self.stage_ids = ids.to(self.device)

    def step_compat(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Backward-compatible step() that returns List[Dict] infos.

        Triggers one CPU sync for done environments. Prefer step() for
        pure-GPU training loops.
        """
        obs, rewards, terminated, truncated, info_tensors = self.step(actions)
        infos = self._tensorized_info_to_list(info_tensors)
        return obs, rewards, terminated, truncated, infos

    @staticmethod
    def _tensorized_info_to_list(info: Dict[str, torch.Tensor]) -> List[Dict]:
        """Convert tensorized info dict back to per-env List[Dict]."""
        done = info["done"]
        num_envs = done.shape[0]
        done_idxs = torch.nonzero(done, as_tuple=False).squeeze(-1)

        infos: List[Dict] = [{} for _ in range(num_envs)]

        if done_idxs.numel() == 0:
            return infos

        # Single CPU sync: pull all done-env data at once
        idxs_cpu = done_idxs.cpu().tolist()
        success_cpu = info["episode_success"][done_idxs].cpu().tolist()
        dist_cpu = info["distance"][done_idxs].cpu().tolist()
        wp_cpu = info["waypoint_idx"][done_idxs].cpu().tolist()
        trunc_cpu = info["truncated"][done_idxs].cpu().tolist()
        ep_len_cpu = info["episode_length"][done_idxs].cpu().tolist()

        for j, i in enumerate(idxs_cpu):
            infos[i] = {
                "episode_success": bool(success_cpu[j]),
                "distance": dist_cpu[j],
                "waypoint_idx": int(wp_cpu[j]),
                "truncated": bool(trunc_cpu[j]),
                "episode_length": int(ep_len_cpu[j]),
            }

        return infos

    def _handle_dones(
        self,
        done: torch.Tensor,
        success: torch.Tensor,
        truncated: torch.Tensor,
        dist_target: torch.Tensor,
    ) -> List[Dict]:
        """Legacy done handler — returns List[Dict]. Prefer _handle_dones_tensorized."""
        info = self._handle_dones_tensorized(done, success, truncated, dist_target)
        return self._tensorized_info_to_list(info)