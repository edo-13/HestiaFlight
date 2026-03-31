from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import pickle

from config import DomainRandomizationConfig
    

@dataclass
class ADRParameter:
    """Single DR parameter with regression-based tracking"""
    name: str
    nominal: float
    min_bound: float
    max_bound: float
    current_min: float = None
    current_max: float = None
    threshold_expand: float = 0.7
    threshold_contract: float = 0.25
    expand_rate: float = 0.05
    contract_rate: float = 0.04
    momentum: float = 0.0
    momentum_decay: float = 0.9
    history: deque = field(default_factory=lambda: deque(maxlen=200))
    
    def __post_init__(self):
        if self.current_min is None:
            range_width = (self.max_bound - self.min_bound) * 0.1
            self.current_min = max(self.min_bound, self.nominal - range_width / 2)
            self.current_max = min(self.max_bound, self.nominal + range_width / 2)
        self.history = deque(maxlen=200)
    
    def sample(self, rng=None) -> Tuple[float, int]:
        if rng is None:
            rng = np.random
        value = rng.uniform(self.current_min, self.current_max)
        sample_id = len(self.history)
        return value, sample_id
    
    def record_outcome(self, sampled_value: float, score: float):
        self.history.append((sampled_value, score))
    
    def compute_boundary_performance(self) -> Tuple[float, float]:
        if len(self.history) < 30:
            return 0.5, 0.5
        
        values = np.array([h[0] for h in self.history])
        scores = np.array([h[1] for h in self.history])
        
        range_size = self.current_max - self.current_min + 1e-8
        normalized = (values - self.current_min) / range_size
        
        weights_min = np.exp(np.clip(-3 * normalized, -20, 0))
        weights_max = np.exp(np.clip(-3 * (1 - normalized), -20, 0))

        score_at_min = np.average(scores, weights=weights_min + 1e-8)
        score_at_max = np.average(scores, weights=weights_max + 1e-8)
        
        return score_at_min, score_at_max
    
    def update_boundaries(self) -> Dict[str, Any]:
        if len(self.history) < 50:
            return {'updated': False, 'reason': 'insufficient_samples'}
        
        score_min, score_max = self.compute_boundary_performance()
        total_range = self.max_bound - self.min_bound
        
        updates = {'updated': False, 'min_action': None, 'max_action': None}
        
        if score_min > self.threshold_expand and self.current_min > self.min_bound:
            delta = total_range * self.expand_rate
            self.current_min = max(self.min_bound, self.current_min - delta)
            updates['min_action'] = 'expand'
            updates['updated'] = True
        elif score_min < self.threshold_contract and self.current_min < self.nominal:
            delta = total_range * self.contract_rate
            self.current_min = min(self.nominal, self.current_min + delta)
            updates['min_action'] = 'contract'
            updates['updated'] = True
        
        if score_max > self.threshold_expand and self.current_max < self.max_bound:
            delta = total_range * self.expand_rate
            self.current_max = min(self.max_bound, self.current_max + delta)
            updates['max_action'] = 'expand'
            updates['updated'] = True
        elif score_max < self.threshold_contract and self.current_max > self.nominal:
            delta = total_range * self.contract_rate
            self.current_max = max(self.nominal, self.current_max - delta)
            updates['max_action'] = 'contract'
            updates['updated'] = True
        
        updates['score_min'] = score_min
        updates['score_max'] = score_max
        
        return updates
    
    def get_coverage(self) -> float:
        total_range = self.max_bound - self.min_bound
        current_range = self.current_max - self.current_min
        return current_range / (total_range + 1e-8)


class AutomaticDomainRandomization:
    """Improved ADR with per-parameter tracking"""
    
    def __init__(self, config, num_envs: int = 1, halve: bool =False):
        self.config = config
        self.rng = np.random.default_rng(42)
        self.parameters: Dict[str, ADRParameter] = {}
        self._current_samples: Dict[str, float] = {}

        self.num_envs = num_envs
        self._env_samples = {}
        
        if config.randomize_mass:
            self.parameters['mass'] = ADRParameter(
                'mass', config.mass_nominal,
                config.mass_range[0], config.mass_range[1]
            )
        if config.randomize_inertia:
            self.parameters['inertia_scale'] = ADRParameter(
                'inertia_scale', config.inertia_nominal,
                config.inertia_range[0], config.inertia_range[1]
            )
        if config.randomize_thrust_to_weight:
            self.parameters['thrust_to_weight'] = ADRParameter(
                'thrust_to_weight', config.thrust_to_weight_nominal,
                config.thrust_to_weight_range[0], config.thrust_to_weight_range[1]
            )
        if config.randomize_drag:
            self.parameters['drag_coefficient'] = ADRParameter(
                'drag_coefficient', config.drag_coef_nominal,
                config.drag_coef_range[0], config.drag_coef_range[1]
            )
        if config.randomize_wind:
            self.parameters['wind_speed'] = ADRParameter(
                'wind_speed', config.wind_speed_nominal,
                config.wind_speed_range[0], config.wind_speed_range[1]
            )
        if config.randomize_air_density:
            self.parameters['air_density_ratio'] = ADRParameter(
                'air_density_ratio', config.air_density_nominal,
                config.air_density_range[0], config.air_density_range[1]
            )
        if config.randomize_sensor_noise:
            self.parameters['position_noise_std'] = ADRParameter(
                'position_noise_std', config.position_noise_nominal,
                config.position_noise_range[0], config.position_noise_range[1]
            )
            self.parameters['velocity_noise_std'] = ADRParameter(
                'velocity_noise_std', config.velocity_noise_nominal,
                config.velocity_noise_range[0], config.velocity_noise_range[1]
            )

        if config.randomize_interval:
            self.parameters['dt'] = ADRParameter(
                'dt', config.time_interval_nominal,
                config.time_interval_range[0], config.time_interval_range[1]
            )
        
        self.halve=halve
    
    def randomize_for_env(self, env_id: int) -> Dict[str, Any]:
        self._env_samples[env_id] = {}
        
        def sample_param(name: str, default: float) -> float:
            if name in self.parameters:
                val, _ = self.parameters[name].sample(self.rng)
                self._env_samples[env_id][name] = val
                return val
            return default
        
        physics = {
            'mass': sample_param('mass', 1.5),
            'inertia_scale': sample_param('inertia_scale', 1.0),
            'thrust_to_weight': sample_param('thrust_to_weight', 2.5),
            'drag_coefficient': sample_param('drag_coefficient', 0.03),
        }
        
        wind_speed = sample_param('wind_speed', 0.0)
        wind_dir = self.rng.uniform(0, 2 * np.pi)
        wind_elev = self.rng.uniform(-np.pi/6, np.pi/6)
        horiz = wind_speed * np.cos(wind_elev)
        
        environment = {
            'wind_velocity': np.array([
                horiz * np.cos(wind_dir),
                horiz * np.sin(wind_dir),
                wind_speed * np.sin(wind_elev)
            ]),
            'wind_turbulence': 0.0,
            'air_density_ratio': sample_param('air_density_ratio', 1.0),
            'gravity': 9.81,
        }
        
        sensors = {
            'position_noise_std': sample_param('position_noise_std', 0.0),
            'velocity_noise_std': sample_param('velocity_noise_std', 0.0),
            'attitude_noise_std': 0.0,
            'control_delay': 0.0,
        }

        time = {
            'dt': sample_param('dt', 0.0),
        }
        
        return {'physics': physics, 'environment': environment, 'sensors': sensors, 'time': time,}
    
    def record_episode(self, env_id: int, score: float):
        """Record outcome for specific environment"""
        if env_id not in self._env_samples:
            return
        
        samples = self._env_samples[env_id]
        for name, value in samples.items():
            if name in self.parameters:
                self.parameters[name].record_outcome(value, score)
        
        # clean up to avoid memory leak
        del self._env_samples[env_id]
    
    def update_boundaries(self) -> Dict[str, Any]:
        results = {}
        for name, param in self.parameters.items():
            result = param.update_boundaries()
            if result['updated']:
                results[name] = result
        return results
    
    def get_stats(self) -> Dict:
        stats = {}
        for name, param in self.parameters.items():
            score_min, score_max = param.compute_boundary_performance()
            stats[name] = {
                'range': (param.current_min, param.current_max),
                'coverage': param.get_coverage(),
                'score_at_min': score_min,
                'score_at_max': score_max,
                'samples': len(param.history)
            }
        
        coverages = [p.get_coverage() for p in self.parameters.values()]
        stats['_summary'] = {
            'avg_coverage': np.mean(coverages),
            'min_coverage': np.min(coverages),
            'max_coverage': np.max(coverages),
        }
        return stats
    
    def save(self, path):
        state = {}
        for name, p in self.parameters.items():
            state[name] = {
                'current_min': p.current_min,
                'current_max': p.current_max,
                'history': list(p.history)
            }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path):
         #halving ADR ranges when loading to allow gentler fine-tuning
        with open(path, 'rb') as f:
            state = pickle.load(f)
        for name, data in state.items():
            if name in self.parameters:
                self.parameters[name].current_min = data['current_min'] / 2.0 if self.halve else data['current_min'] #halving ADR ranges when loading to allow gentler fine-tuning
                self.parameters[name].current_max = data['current_max'] / 2.0 if self.halve else data['current_max']
                self.parameters[name].history = deque(data['history'], maxlen=200)

    def randomize_batch(self, env_ids: torch.Tensor, device: torch.device) -> Dict[str, Any]:
        """
        Batch-sample all DR parameters for multiple environments at once.

        Args:
            env_ids: Tensor of environment indices to randomize
            device: Target device for output tensors

        Returns:
            Dictionary with batched physics/environment/sensor parameters as tensors
        """
        n = len(env_ids)
        if n == 0:
            return None

        env_ids_list = env_ids.tolist()

        # pre-allocate numpy arrays for sampling (faster than individual calls)
        def batch_sample(param_name: str, default: float) -> Tuple[np.ndarray, np.ndarray]:
            """Sample n values, return (values, sample_ids)"""
            if param_name in self.parameters:
                param = self.parameters[param_name]
                values = self.rng.uniform(param.current_min, param.current_max, size=n)
                return values
            return np.full(n, default)

        # batch sample all parameters
        mass_vals = batch_sample('mass', 1.5)
        inertia_vals = batch_sample('inertia_scale', 1.0)
        thrust_vals = batch_sample('thrust_to_weight', 2.5)
        drag_vals = batch_sample('drag_coefficient', 0.03)
        wind_speed_vals = batch_sample('wind_speed', 0.0)
        air_density_vals = batch_sample('air_density_ratio', 1.0)
        pos_noise_vals = batch_sample('position_noise_std', 0.0)
        vel_noise_vals = batch_sample('velocity_noise_std', 0.0)

        time_vals = batch_sample('dt', 0.01)

        # wind direction (always randomized, not ADR-controlled)
        wind_dir = self.rng.uniform(0, 2 * np.pi, size=n)
        wind_elev = self.rng.uniform(-np.pi/6, np.pi/6, size=n)
        horiz = wind_speed_vals * np.cos(wind_elev)

        wind_velocity = np.stack([
            horiz * np.cos(wind_dir),
            horiz * np.sin(wind_dir),
            wind_speed_vals * np.sin(wind_elev)
        ], axis=1)

        # store samples for later outcome recording
        for i, env_id in enumerate(env_ids_list):
            self._env_samples[env_id] = {}
            for param_name, values in [
                ('mass', mass_vals),
                ('inertia_scale', inertia_vals),
                ('thrust_to_weight', thrust_vals),
                ('drag_coefficient', drag_vals),
                ('wind_speed', wind_speed_vals),
                ('air_density_ratio', air_density_vals),
                ('position_noise_std', pos_noise_vals),
                ('velocity_noise_std', vel_noise_vals),
                ('dt', time_vals)
            ]:
                if param_name in self.parameters:
                    self._env_samples[env_id][param_name] = values[i]

        # convert to tensors on target device
        return {
            'physics': {
                'mass': torch.tensor(mass_vals, dtype=torch.float32, device=device),
                'inertia_scale': torch.tensor(inertia_vals, dtype=torch.float32, device=device),
                'thrust_to_weight': torch.tensor(thrust_vals, dtype=torch.float32, device=device),
                'drag_coefficient': torch.tensor(drag_vals, dtype=torch.float32, device=device),
            },
            'environment': {
                'wind_velocity': torch.tensor(wind_velocity, dtype=torch.float32, device=device),
                'wind_turbulence': torch.zeros(n, dtype=torch.float32, device=device),
                'air_density_ratio': torch.tensor(air_density_vals, dtype=torch.float32, device=device),
            },
            'sensors': {
                'position_noise_std': torch.tensor(pos_noise_vals, dtype=torch.float32, device=device),
                'velocity_noise_std': torch.tensor(vel_noise_vals, dtype=torch.float32, device=device),
                'control_delay': torch.zeros(n, dtype=torch.float32, device=device),
            },
            'time': {
                'dt': torch.tensor(time_vals, dtype=torch.float32, device=device),
            }
        }


    def record_episodes_batch(self, env_ids: torch.Tensor, scores: torch.Tensor):
        """
        Record outcomes for multiple environments at once.

        Args:
            env_ids: Tensor of environment indices
            scores: Tensor of scores (0.0 for failure, 1.0 for success)
        """
        env_ids_list = env_ids.tolist()
        scores_list = scores.tolist()

        for env_id, score in zip(env_ids_list, scores_list):
            if env_id not in self._env_samples:
                continue
            
            samples = self._env_samples[env_id]
            for name, value in samples.items():
                if name in self.parameters:
                    self.parameters[name].record_outcome(value, score)

            del self._env_samples[env_id]