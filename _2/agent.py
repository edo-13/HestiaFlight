#!/usr/bin/env python3
"""
6-DOF Interceptor Agent - Improved
- Fixed numerical stability in tanh log-prob
- Fixed prev_actions index handling
- Improved std clamping range
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class AgentConfig:
    """Configuration for 6-DOF interceptor agent"""
    
    # Input dimensions
    obs_dim: int = 27  # pos(3) + vel(3) + gravity(3) + omega(3) + accel(3) + 
                       # rel_pos(3) + rel_vel(3) + dist(1) + close_rate(1)
    privileged_dim: int = 12
    action_dim: int = 4  # thrust, roll_rate, pitch_rate, yaw_rate
    privileged_head_dim: int = 64
    
    # Encoder
    encoder_hidden_dim: int = 256
    encoder_layers: int = 3
    
    # Actor network
    actor_hidden_dim: int = 128
    actor_layers: int = 4
    
    # Critic network
    critic_hidden_dim: int = 128
    critic_layers: int = 4
    
    # Action distribution
    use_state_dependent_std: bool = False

    # Std clamping bounds (log space)
    log_std_min: float = -3.0   # std ≈ 0.05
    log_std_max: float = 0.0    # std ≈ 1.0 (more exploration room)
    
    # Action smoothing
    use_action_smoothing: bool = False
    action_smoothing_alpha: float = 0.3
    
    # Include previous action in observation
    use_prev_action: bool = True
    
    # Numerical stability
    tanh_epsilon: float = 1e-6
    action_bound: float = 0.999  # For atanh stability
    
    dropout: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class StateEncoder(nn.Module):
    """Encodes observations for actor"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ELU())
            if i < num_layers - 1 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        layers.append(nn.LayerNorm(hidden_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorNetwork(nn.Module):
    """Policy network - outputs action distribution"""
    
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        input_dim = config.encoder_hidden_dim
        
        layers = []
        current_dim = input_dim
        
        for _ in range(config.actor_layers):
            layers.append(nn.Linear(current_dim, config.actor_hidden_dim))
            layers.append(nn.ELU())
            current_dim = config.actor_hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(current_dim, config.action_dim)
        
        # State-dependent or fixed log std
        if config.use_state_dependent_std:
            self.log_std_head = nn.Linear(current_dim, config.action_dim)
        else:
            self.log_std = nn.Parameter(torch.ones(config.action_dim) * -1.0)
        
        self._init_weights()
        
    def _init_weights(self):
        with torch.no_grad():
            nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
            nn.init.zeros_(self.mean_head.bias)
            
            if self.config.use_state_dependent_std:
                nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
                # Initialize to std ≈ 0.37 (log(0.37) ≈ -1.0)
                nn.init.constant_(self.log_std_head.bias, -1.0)
    
    def forward(self, encoded_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(encoded_obs)
        mean = self.mean_head(features)
        
        if self.config.use_state_dependent_std:
            log_std = self.log_std_head(features)
            log_std = log_std.clamp(self.config.log_std_min, self.config.log_std_max)
            std = torch.exp(log_std)
        else:
            log_std = self.log_std.clamp(self.config.log_std_min, self.config.log_std_max)
            std = torch.exp(log_std).expand_as(mean)
        
        return mean, std
    
    def sample(self, encoded_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(encoded_obs)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        
        # Log prob with tanh correction (numerically stable)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        # Use log1p for better numerical stability: log(1 - x^2) = log((1-x)(1+x))
        # When x is close to 1, 1-x^2 is tiny, but log(1-x) + log(1+x) is more stable
        log_prob = log_prob - torch.log1p(-action.pow(2).clamp(max=1.0 - self.config.tanh_epsilon)).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate(self, encoded_obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(encoded_obs)
        
        # Inverse tanh with clamping for stability
        action_clamped = action.clamp(-self.config.action_bound, self.config.action_bound)
        raw_action = torch.atanh(action_clamped)
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        # Numerically stable tanh correction
        log_prob = log_prob - torch.log1p(-action_clamped.pow(2).clamp(max=1.0 - self.config.tanh_epsilon)).sum(dim=-1)
        
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """Value network - asymmetric (sees privileged state)"""
    
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        input_dim = config.encoder_hidden_dim + config.privileged_head_dim
        
        layers = []
        current_dim = input_dim
        
        for _ in range(config.critic_layers):
            layers.append(nn.Linear(current_dim, config.critic_hidden_dim))
            layers.append(nn.LeakyReLU())
            current_dim = config.critic_hidden_dim
        
        layers.append(nn.LayerNorm(current_dim))
        layers.append(nn.Linear(current_dim, 1))
        
        self.net = nn.Sequential(*layers)
        self.privileged_net = nn.Sequential(
            nn.Linear(config.privileged_dim, 64),
            nn.ELU(),
            nn.Linear(64, config.privileged_head_dim),
            nn.ELU(),
            nn.LayerNorm(config.privileged_head_dim),
        )
    
    def forward(self, encoded_obs: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
        if privileged.numel() == 0 or privileged.shape[-1] == 0:
            # If no privileged info, pad with zeros
            batch_size = encoded_obs.shape[0]
            priv_features = torch.zeros(
                batch_size, self.config.privileged_head_dim, 
                device=encoded_obs.device
            )
        else:
            priv_features = self.privileged_net(privileged)
        
        x = torch.cat([encoded_obs, priv_features], dim=-1)
        return self.net(x).squeeze(-1)


class InterceptorAgent(nn.Module):
    """Complete 6-DOF agent with separate action smoothing state"""
    
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        
        # Automatically adjust encoder input size
        obs_input_dim = config.obs_dim
        if config.use_prev_action:
            obs_input_dim += config.action_dim
        
        self.actor_encoder = StateEncoder(
            obs_input_dim, 
            config.encoder_hidden_dim,
            config.encoder_layers,
            config.dropout
        )

        self.critic_encoder = StateEncoder(
            obs_input_dim, 
            config.encoder_hidden_dim,
            config.encoder_layers,
            config.dropout
        )
        
        self.actor = ActorNetwork(config)
        self.critic = CriticNetwork(config)
        
        # Action smoothing state - will be properly initialized
        self._prev_actions: Optional[torch.Tensor] = None
        self._num_envs: int = 0
        
    @property
    def prev_actions(self) -> Optional[torch.Tensor]:
        return self._prev_actions
    
    @prev_actions.setter
    def prev_actions(self, value: Optional[torch.Tensor]):
        self._prev_actions = value
        
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    @num_envs.setter
    def num_envs(self, value: int):
        self._num_envs = value
        
    def initialize_action_state(self, num_envs: int, device: torch.device):
        """Initialize action state for the given number of environments."""
        self._num_envs = num_envs
        self._prev_actions = torch.zeros(num_envs, self.config.action_dim, device=device)
        # Small random initialization for thrust
        self._prev_actions[:, 0] = torch.randn(num_envs, device=device) * 0.1
    
    def reset_action_state(self, env_indices):
        """Reset action state for specific environment indices."""
        if self._prev_actions is None:
            return
        if isinstance(env_indices, int):
            env_indices = [env_indices]
        if len(env_indices) == 0:
            return

        # Convert to tensor for efficient batch indexing
        if not isinstance(env_indices, torch.Tensor):
            env_indices = torch.tensor(
                env_indices, 
                device=self._prev_actions.device, 
                dtype=torch.long
            )

        # Reset to hover-biased initial action
        self._prev_actions[env_indices] = 0.0
        self._prev_actions[env_indices, 0] = 0.4  # Slight thrust bias
    
    def _prepare_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Augment observation with previous action.
        
        FIXED: Now properly handles batch size mismatches without corrupting
        the environment-to-action mapping.
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        if not self.config.use_prev_action:
            # Return non-augmented obs if prev_action is not flagged True
            return obs
        
        if self._prev_actions is None or self._num_envs == 0:
            # Not initialized - append zeros
            null = torch.zeros(batch_size, self.config.action_dim, device=device)
            return torch.cat([obs, null], dim=-1)
        
        # FIXED: Handle batch size mismatch properly
        if batch_size == self._num_envs:
            # Normal case - direct use
            prev_act = self._prev_actions
        elif batch_size < self._num_envs:
            # Evaluation/inference with smaller batch
            # IMPORTANT: Only safe if caller passes first N environments
            # For arbitrary subsets, caller should handle prev_actions externally
            prev_act = self._prev_actions[:batch_size]
        else:
            # Batch larger than initialized - pad with zeros
            # This shouldn't happen in normal training
            pad_size = batch_size - self._num_envs
            padding = torch.zeros(pad_size, self.config.action_dim, device=device)
            prev_act = torch.cat([self._prev_actions, padding], dim=0)
        
        return torch.cat([obs, prev_act], dim=-1)
    
    def _update_prev_actions(self, action: torch.Tensor):
        """Update previous action state after taking an action."""
        if self._prev_actions is not None and action.shape[0] == self._num_envs:
            self._prev_actions = action.detach().clone()
    
    def _smooth_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Apply action smoothing if enabled."""
        enabled = getattr(self, 'smoothing_enabled', self.config.use_action_smoothing)
        
        if not enabled or self._prev_actions is None:
            # Update prev_actions for next step even without smoothing
            self._update_prev_actions(raw_action)
            return raw_action
        
        alpha = getattr(self, 'smoothing_alpha', self.config.action_smoothing_alpha)
        smoothed = alpha * raw_action + (1 - alpha) * self._prev_actions
        self._update_prev_actions(smoothed)
        return smoothed
    
    def forward(self, obs: torch.Tensor, privileged: torch.Tensor, 
                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Rollout forward pass.
        
        Args:
            obs: Raw observations (obs_dim,) - should NOT include prev_action
            privileged: Privileged state information
            deterministic: If True, use mean action without sampling
            
        Returns:
            Dictionary containing:
            - action: The action to execute
            - log_prob: Log probability of the action
            - value: Value estimate
            - augmented_obs: Observation with prev_action appended (for buffer storage)
        """
        # Verify obs doesn't already contain prev_action
        expected_dim = self.config.obs_dim
        if obs.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim}-dim observation, got {obs.shape[-1]}. "
                f"Observations should NOT include prev_action when passed to forward()."
            )
        
        # Augment observation with previous action
        obs_with_prev = self._prepare_obs(obs)

        # Encode
        encoded_actor = self.actor_encoder(obs_with_prev)
        encoded_critic = self.critic_encoder(obs_with_prev)
        
        # Get action
        if deterministic:
            mean, _ = self.actor(encoded_actor)
            action = torch.tanh(mean)
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            action, log_prob = self.actor.sample(encoded_actor)
        
        # Apply smoothing and update prev_actions
        action_smoothed = self._smooth_action(action)
        
        # Get value estimate
        value = self.critic(encoded_critic, privileged)
        
        return {
            'action': action_smoothed,
            'log_prob': log_prob,
            'value': value,
            'augmented_obs': obs_with_prev  # Store this in buffer for PPO update
        }
    
    def get_value(self, obs: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
        """Get value estimate for given observation."""
        obs_with_prev = self._prepare_obs(obs)
        encoded_critic = self.critic_encoder(obs_with_prev)
        return self.critic(encoded_critic, privileged)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                        privileged: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO update pass - evaluate log probabilities and values for stored transitions.
        
        Args:
            obs: Augmented observations from buffer (already includes prev_action)
            actions: Actions taken
            privileged: Privileged state information
            
        Returns:
            values: Value estimates
            log_probs: Log probabilities of actions
            entropy: Policy entropy
        """
        # Handle 3D input (seq, batch, dim)
        if obs.dim() == 3:
            seq_len, batch_size, _ = obs.shape
            # Use reshape instead of view to handle non-contiguous tensors
            obs_flat = obs.reshape(seq_len * batch_size, -1)
            actions_flat = actions.reshape(seq_len * batch_size, -1)
            priv_flat = privileged.reshape(seq_len * batch_size, -1)
            
            encoded_actor = self.actor_encoder(obs_flat)
            log_probs, entropy = self.actor.evaluate(encoded_actor, actions_flat)
            
            encoded_critic = self.critic_encoder(obs_flat)
            values = self.critic(encoded_critic, priv_flat)
            
            return (
                values.reshape(seq_len, batch_size),
                log_probs.reshape(seq_len, batch_size),
                entropy.reshape(seq_len, batch_size)
            )
        else:
            # 2D input (batch, dim)
            encoded_actor = self.actor_encoder(obs)
            log_probs, entropy = self.actor.evaluate(encoded_actor, actions)
            
            encoded_critic = self.critic_encoder(obs)
            values = self.critic(encoded_critic, privileged)
            
            return values, log_probs, entropy
    
    def get_action_deterministic(self, obs: torch.Tensor, privileged: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get deterministic action for evaluation/deployment.
        Convenience method that only returns the action.
        """
        if privileged is None:
            privileged = torch.zeros(obs.shape[0], self.config.privileged_dim, device=obs.device)
        
        with torch.no_grad():
            result = self.forward(obs, privileged, deterministic=True)
        return result['action']


def create_interceptor_agent(config: Optional[AgentConfig] = None) -> InterceptorAgent:
    """Factory function to create an InterceptorAgent."""
    if config is None:
        config = AgentConfig()
    return InterceptorAgent(config)


if __name__ == "__main__":
    # Test the agent
    config = AgentConfig()
    agent = create_interceptor_agent(config)
    
    device = torch.device(config.device)
    agent = agent.to(device)
    
    num_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 16
    agent.initialize_action_state(batch_size, device)
    
    obs = torch.randn(batch_size, config.obs_dim, device=device)
    priv = torch.randn(batch_size, config.privileged_dim, device=device)
    
    output = agent(obs, priv)
    print(f"Action shape: {output['action'].shape}")
    print(f"Value shape: {output['value'].shape}")
    print(f"Augmented obs shape: {output['augmented_obs'].shape}")
    
    # Test evaluate_actions
    aug_obs = output['augmented_obs']
    actions = output['action']
    values, log_probs, entropy = agent.evaluate_actions(aug_obs, actions, priv)
    print(f"Values shape: {values.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Entropy shape: {entropy.shape}")
    
    # Test with 3D input (seq, batch, dim)
    seq_len = 8
    obs_3d = aug_obs.unsqueeze(0).expand(seq_len, -1, -1)
    actions_3d = actions.unsqueeze(0).expand(seq_len, -1, -1)
    priv_3d = priv.unsqueeze(0).expand(seq_len, -1, -1)
    
    values_3d, log_probs_3d, entropy_3d = agent.evaluate_actions(obs_3d, actions_3d, priv_3d)
    print(f"3D Values shape: {values_3d.shape}")
    print(f"3D Log probs shape: {log_probs_3d.shape}")