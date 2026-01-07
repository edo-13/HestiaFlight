# HestiaFlight

**High-performance GPU-vectorized quadrotor simulation for reinforcement learning**

HestiaFlight runs **10M+ environment steps/second** on a single GPU, enabling rapid policy training for drone control, recovery, and racing tasks.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-Hestia%20License-green.svg)

## **!!Open Alpha!!** — This is an early release from a small team at [Hestia]([hestiatech.squarespace.com](https://hestiatech.squarespace.com/)) working on multiple projects in parallel. Expect rough edges, missing docs, and breaking changes. We're sharing it early because it's already useful, but it's not polished. Feedback and contributions welcome!

---
## Demo GIF

![eval_recording](https://github.com/user-attachments/assets/04784520-a3e5-42c9-b679-0439667a5b13)

- This GIF was realized with the provided physiscs engine, Env, ADR.

## Features

- **Fully Vectorized**: All physics, rewards, and observations computed on GPU tensors

- **Second-Order Motor Dynamics**: Realistic ESC + rotor inertia modeling

- **Automatic Domain Randomization (ADR)**: Boundary regression for sim-to-real transfer
- **Stable-Baselines3 Integration**: Drop-in SB3 wrapper included
- **Configurable Rewards**: Dataclass-based reward shaping
- **Real-time Visualization**: Optional OpenGL renderer

## Benchmarks

### Full Environment (physics + obs + rewards)

| Environments | Env-steps/sec | ms/step | VRAM | KB/env |
|--------------|---------------|---------|------|--------|
| 4,096        | 0.91M         | 4.5     | 109 MB | 26.5 |
| 16,384       | 3.08M         | 5.3     | 411 MB | 25.1 |
| 65,536       | 9.10M         | 7.2     | 1.6 GB | 24.7 |
| 131,072      | **11.47M**    | 11.4    | 3.2 GB | 24.6 |

### Physics Engine Only

| Environments | Env-steps/sec | VRAM |
|--------------|---------------|------|
| 32,768       | 22.4M         | 19 MB |
| 65,536       | 43.4M         | 30 MB |
| 131,072      | **81.2M**     | 55 MB |

*RTX 5060 Laptop (8GB), PyTorch 2.11, CUDA 12.8*

---

## Installation

```bash
git clone https://github.com/edo-13/HestiaFlight.git
cd hestiaflight
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0 (with CUDA for GPU acceleration)
- NumPy
- stable-baselines3 (optional, for training examples)
- pygame + PyOpenGL (optional, for visualization)

---

## Quick Start

### Train a Hover Policy (5 minutes to convergence)

```python
from stable_baselines3 import PPO
from vec_env import VecEnv
from config import TrainingConfig, RewardConfig, SpawnConfig
from example_train import SB3Wrapper, Points

# Setup
num_envs = 2048
cfg = TrainingConfig(max_episode_steps=256)
manager = Points(num_envs)

env = VecEnv(
    stage_names="hover",
    num_envs=num_envs,
    trajectory_manager=manager,
    config=cfg
)

# Train
model = PPO("MlpPolicy", SB3Wrapper(env), device="cuda", n_steps=128)
model.learn(total_timesteps=15_000_000)
```

### Visualize

```python
from render import QuadrotorRenderer

renderer = QuadrotorRenderer()
while renderer.render(env.pos, env.quat, env.env_waypoints, env.current_waypoint_idx):
    obs, *_ = env.step(model.predict(obs)[0])
```

---

## Architecture

```
hestiaflight/
├── phy.py          # physics engine (QuadrotorDynamics, BatchedPhysicsEnv)
├── vec_env.py      # vectorized Gym-style environment
├── config.py       # dataclass configs (Training, Reward, Spawn, ADR)
├── adr.py          # Automatic Domain Randomization
├── render.py       # OpenGL visualization
├── benchmark.py    # Performance testing
├──render_training  # Render some quadrotors while training
└── example_train.py # SB3 integration example
```

### Observation Space (27D)
| Index | Description |
|-------|-------------|
| 0:3   | World position |
| 3:6   | World velocity |
| 6:9   | Gravity in body frame |
| 9:12  | Angular velocity (body) |
| 12:15 | Linear acceleration |
| 15:18 | Relative pos to waypoint (body) |
| 18:21 | Relative velocity (body) |
| 21    | Distance to waypoint |
| 22    | Closure rate |
| 23:26 | Next waypoint (body) |
| 26    | Path curvature |

### Action Space (4D)
| Index | Description |
|-------|-------------|
| 0     | Collective thrust [-1,1] → [0,1] |
| 1:3   | Body rate commands (roll, pitch, yaw) |

---

## Domain Randomization

HestiaFlight includes ADR with per-parameter boundary regression:

```python
from config import DomainRandomizationConfig

adr_cfg = DomainRandomizationConfig(
    enabled=True,
    mass_range=(0.8, 5.0),
    thrust_to_weight_range=(1.1, 3.2),
    wind_speed_range=(0.0, 20.0),
    # ... see config.py for all options
)
```

ADR automatically expands and contracts randomization ranges based on policy performance at boundaries.

---

## Examples

| Script | Description |
|--------|-------------|
| `example_train.py` | basic hover training |
| `tumble_train.py` | zero-G recovery from tumbles |
| `benchmark.py` | performance measurement |

---

## Citation

```bibtex
@software{hestiaflight2025,
  title = {HestiaFlight: GPU-Vectorized Quadrotor Simulation},
  author = {Edoardo @ Hestia Technologies},
  year = {2025},
  url = {https://github.com/hestia-ai/hestiaflight}
}
```

---

## License

Free for individuals and organizations under **$1M annual revenue**. See [LICENSE.md](LICENSE.md).

Contact: edoardo@hestiatech.net

---

Built with love by Edo
