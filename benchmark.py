"""
Benchmark script

Measures:
- Steps per second at various environment counts
- VRAM usage scaling
- Physics-only vs full environment throughput
- Latency distribution

Usage:
    python benchmark.py                    # Run all benchmarks
    python benchmark.py --num-envs 16384   # Test specific env count
    python benchmark.py --quick            # Quick sanity check
"""

import torch
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import gc

try:
    from phy import BatchedPhysicsEnv, QuadrotorDynamics, DEFAULT_GRAVITY
    HAS_PHY = True
except ImportError:
    HAS_PHY = False
    print("Warning: Could not import phy module. Physics-only benchmarks disabled.")

try:
    from vec_env import VecEnv, RewardConfig
    HAS_ENV = True
except ImportError:
    HAS_ENV = False
    print("Warning: Could not import vec_env module. Full env benchmarks disabled.")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    num_envs: int
    steps_per_second: float
    total_env_steps_per_second: float  # num_envs * steps_per_second
    mean_step_ms: float
    std_step_ms: float
    min_step_ms: float
    max_step_ms: float
    vram_mb: float
    warmup_steps: int
    benchmark_steps: int


def get_vram_usage_mb() -> float:
    """Get current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_vram_reserved_mb() -> float:
    """Get reserved VRAM in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 ** 2)
    return 0.0


def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class MockConfig:
    """Minimal config for benchmarking without full config system."""
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.use_waypoint_mode = True
        self.max_episode_steps = 1000
        self.domain_randomization = type('obj', (object,), {'enabled': False})()


class MockWaypointManager:
    """Minimal waypoint manager for benchmarking."""
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
    
    def get_training_batch(self, stage: str, count: int, device) -> dict:
        # generate random waypoints
        num_waypoints = 10
        waypoints = torch.randn(count, num_waypoints, 3, device=device) * 50
        return {'waypoints': waypoints}


class MockTrajectoryManager:
    """Minimal trajectory manager for benchmarking."""
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
    
    def get_training_batch(self, stage: str, count: int, device) -> dict:
        traj_len = 200
        positions = torch.randn(count, traj_len, 3, device=device) * 50
        velocities = torch.randn(count, traj_len, 3, device=device) * 5
        return {'positions': positions, 'velocities': velocities}


def benchmark_physics_only(
    num_envs: int,
    warmup_steps: int = 100,
    benchmark_steps: int = 1000,
    device: str = "cuda"
) -> BenchmarkResult:
    """Benchmark raw physics engine throughput."""
    
    clear_cuda_cache()
    
    # initialize physics
    physics = BatchedPhysicsEnv(num_envs, dt=0.01, device=device)
    
    # initialize state
    state = {
        'position': torch.zeros((num_envs, 3), device=device),
        'velocity': torch.zeros((num_envs, 3), device=device),
        'quaternion': torch.zeros((num_envs, 4), device=device),
        'angular_velocity': torch.zeros((num_envs, 3), device=device),
    }
    state['quaternion'][:, 0] = 1.0  # identity quaternion
    
    # random actions
    actions = torch.randn((num_envs, 4), device=device).clamp(-1, 1)
    
    # warmup
    for _ in range(warmup_steps):
        state = physics.step(state, actions)
        actions = torch.randn((num_envs, 4), device=device).clamp(-1, 1)
    
    torch.cuda.synchronize()
    
    # benchmarking
    step_times = []
    for _ in range(benchmark_steps):
        start = time.perf_counter()
        state = physics.step(state, actions)
        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - start) * 1000)  # ms
        actions = torch.randn((num_envs, 4), device=device).clamp(-1, 1)
    
    step_times = torch.tensor(step_times)
    mean_ms = step_times.mean().item()
    
    return BenchmarkResult(
        num_envs=num_envs,
        steps_per_second=1000.0 / mean_ms,
        total_env_steps_per_second=num_envs * 1000.0 / mean_ms,
        mean_step_ms=mean_ms,
        std_step_ms=step_times.std().item(),
        min_step_ms=step_times.min().item(),
        max_step_ms=step_times.max().item(),
        vram_mb=get_vram_usage_mb(),
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
    )


def benchmark_full_env(
    num_envs: int,
    warmup_steps: int = 100,
    benchmark_steps: int = 1000,
    device: str = "cuda"
) -> BenchmarkResult:
    """Benchmark full environment including observations and rewards."""
    
    clear_cuda_cache()
    
    config = MockConfig(device)
    waypoint_manager = MockWaypointManager(device)
    trajectory_manager = MockTrajectoryManager(device)
    
    env = VecEnv(
        stage_names="benchmark",
        num_envs=num_envs,
        trajectory_manager=trajectory_manager,
        config=config,
        waypoint_manager=waypoint_manager,
    )
    
    # random actions
    actions = torch.randn((num_envs, 4), device=device).clamp(-1, 1)
    
    # warmup
    for _ in range(warmup_steps):
        obs, rewards, terminated, truncated, infos = env.step(actions)
        actions = torch.randn((num_envs, 4), device=device).clamp(-1, 1)
    
    torch.cuda.synchronize()
    
    # benchmarking
    step_times = []
    for _ in range(benchmark_steps):
        start = time.perf_counter()
        obs, rewards, terminated, truncated, infos = env.step(actions)
        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - start) * 1000)
        actions = torch.randn((num_envs, 4), device=device).clamp(-1, 1)
    
    step_times = torch.tensor(step_times)
    mean_ms = step_times.mean().item()
    
    return BenchmarkResult(
        num_envs=num_envs,
        steps_per_second=1000.0 / mean_ms,
        total_env_steps_per_second=num_envs * 1000.0 / mean_ms,
        mean_step_ms=mean_ms,
        std_step_ms=step_times.std().item(),
        min_step_ms=step_times.min().item(),
        max_step_ms=step_times.max().item(),
        vram_mb=get_vram_usage_mb(),
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
    )


def find_max_envs(
    device: str = "cuda",
    start: int = 1024,
    max_attempts: int = 20
) -> Tuple[int, float]:
    """Find maximum number of environments that fit in VRAM."""
    
    if not torch.cuda.is_available():
        return start, 0.0
    
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    
    num_envs = start
    max_successful = start
    max_vram = 0.0
    
    for _ in range(max_attempts):
        try:
            clear_cuda_cache()
            
            config = MockConfig(device)
            waypoint_manager = MockWaypointManager(device)
            trajectory_manager = MockTrajectoryManager(device)
            
            env = VecEnv(
                stage_names="test",
                num_envs=num_envs,
                trajectory_manager=trajectory_manager,
                config=config,
                waypoint_manager=waypoint_manager,
            )
            
            # try a few steps to ensure everything is allocated
            actions = torch.randn((num_envs, 4), device=device).clamp(-1, 1)
            for _ in range(10):
                env.step(actions)
            
            torch.cuda.synchronize()
            vram = get_vram_usage_mb()
            
            max_successful = num_envs
            max_vram = vram
            
            vram_per_env = vram / num_envs
            remaining = total_vram * 0.9 - vram 
            
            if remaining < vram_per_env * 1000:
                break
                
            num_envs = int(num_envs * 1.5)
            
            del env
            clear_cuda_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                num_envs = int(num_envs * 0.75)
                clear_cuda_cache()
            else:
                raise
    
    return max_successful, max_vram


def print_result(result: BenchmarkResult, label: str = ""):
    """Pretty print benchmark result."""
    print(f"\n{'-' * 60}")
    if label:
        print(f"  {label}")
        print(f"{'-' * 60}")
    print(f"  Environments:        {result.num_envs:,}")
    print(f"  Steps/sec:           {result.steps_per_second:,.1f}")
    print(f"  Env-steps/sec:       {result.total_env_steps_per_second:,.0f}")
    print(f"  Mean step time:      {result.mean_step_ms:.3f} ms")
    print(f"  Std step time:       {result.std_step_ms:.3f} ms")
    print(f"  Min/Max step time:   {result.min_step_ms:.3f} / {result.max_step_ms:.3f} ms")
    print(f"  VRAM used:           {result.vram_mb:.1f} MB")
    print(f"  VRAM/env:            {result.vram_mb / result.num_envs * 1000:.2f} KB")


def print_scaling_table(results: List[BenchmarkResult], title: str):
    """Print scaling results as a table."""
    print(f"\n{'-' * 80}")
    print(f"  {title}")
    print(f"  {'-' * 10}-+-{'-' * 10}-+-{'-' * 14}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}")
    print(f"  {'Envs':>10} | {'Steps/s':>10} | {'M Env-steps/s':>14} | {'ms/step':>10} | {'VRAM MB':>10} | {'KB/env':>8}")
    print(f"  {'-' * 10}-+-{'-' * 10}-+-{'-' * 14}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}")
    
    for r in results:
        print(f"  {r.num_envs:>10,} | {r.steps_per_second:>10,.1f} | "
              f"{r.total_env_steps_per_second / 1e6:>14.2f} | {r.mean_step_ms:>10.3f} | "
              f"{r.vram_mb:>10.1f} | {r.vram_mb / r.num_envs * 1000:>8.2f}")


def run_scaling_benchmark(
    env_counts: List[int],
    benchmark_fn,
    warmup_steps: int = 50,
    benchmark_steps: int = 500,
    device: str = "cuda"
) -> List[BenchmarkResult]:
    """Run benchmark across multiple environment counts."""
    results = []
    
    for num_envs in env_counts:
        try:
            clear_cuda_cache()
            result = benchmark_fn(
                num_envs=num_envs,
                warmup_steps=warmup_steps,
                benchmark_steps=benchmark_steps,
                device=device
            )
            results.append(result)
            print(f"  ✓ {num_envs:,} envs: {result.total_env_steps_per_second/1e6:.2f}M env-steps/s, "
                  f"{result.vram_mb:.0f} MB VRAM")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ {num_envs:,} envs: Out of memory")
                break
            raise
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark vectorized quadrotor environment")
    parser.add_argument("--num-envs", type=int, default=None, help="Test specific environment count")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark with fewer steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to benchmark on")
    parser.add_argument("--find-max", action="store_true", help="Find maximum environment count")
    parser.add_argument("--physics-only", action="store_true", help="Benchmark physics engine only")
    args = parser.parse_args()
    
    # check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    d=0
    # print system info
    print("\n  SYSTEM INFORMATION")
    print(f"  PyTorch version:     {torch.__version__}")
    print(f"  Device:              {args.device}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:                 {props.name}")
        print(f"  VRAM:                {props.total_memory / (1024**3):.1f} GB")
        print(f"  CUDA version:        {torch.version.cuda}")
        print(f"  cuDNN version:       {torch.backends.cudnn.version()}")
    warmup_steps = 20 if args.quick else 100
    benchmark_steps = 100 if args.quick else 500

    # single environment count test
    if args.num_envs is not None:
        print(f"\n  Running single benchmark with {args.num_envs:,} environments...")
        if args.physics_only and HAS_PHY:
            result = benchmark_physics_only(
                args.num_envs, warmup_steps, benchmark_steps, args.device
            )
            print_result(result, "Physics Only")
        elif HAS_ENV:
            result = benchmark_full_env(
                args.num_envs, warmup_steps, benchmark_steps, args.device
            )
            print_result(result, "Full Environment")
        return

    # maximum environments
    if args.find_max and HAS_ENV:
        print("\n  Finding maximum environment count...")
        max_envs, max_vram = find_max_envs(args.device)
        print(f"\n  Maximum environments: {max_envs:,}")
        print(f"  VRAM at max:          {max_vram:.1f} MB")

    # scaling benchmarks
    env_counts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 65536*2]

    if args.quick:
        env_counts = [512, 2048, 8192, 16384]

    # physics-only scaling
    if HAS_PHY:
        print("\n  Running physics-only scaling benchmark")
        physics_results = run_scaling_benchmark(
            env_counts, benchmark_physics_only, warmup_steps, benchmark_steps, args.device
        )
        if physics_results:
            print_scaling_table(physics_results, "PHYSICS ENGINE SCALING")

    # full environment scaling
    if HAS_ENV:
        print("\n  Running full environment scaling benchmark")
        env_results = run_scaling_benchmark(
            env_counts, benchmark_full_env, warmup_steps, benchmark_steps, args.device
        )
        if env_results:
            print_scaling_table(env_results, "FULL ENVIRONMENT SCALING")

    # summary
    if HAS_ENV and env_results:
        best = max(env_results, key=lambda r: r.total_env_steps_per_second)

        print("\n  SUMMARY")
        print(f"  Peak throughput:     {best.total_env_steps_per_second / 1e6:.2f} M env-steps/sec")
        print(f"  At environment count: {best.num_envs:,}")
        print(f"  VRAM efficiency:     {best.vram_mb / best.num_envs * 1000:.2f} KB/env")

        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            estimated_max = int(total_vram * 1024 / (best.vram_mb / best.num_envs * 1000) * 0.85)
            print(f"  Estimated max envs:  ~{estimated_max:,}K (with 15% margin)")


if __name__ == "__main__":
    main()