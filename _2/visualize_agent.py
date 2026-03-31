#!/usr/bin/env python3
"""
Visualize a trained 6-DOF quadrotor agent using render.py.
Controls: WASD = pan, QE = up/down, Mouse drag = orbit, Scroll = zoom, ESC = quit, R = record
"""

import torch
import numpy as np
import os
import argparse
import pygame
from pygame.locals import KEYDOWN, K_r, K_ESCAPE

from render import QuadrotorRenderer
from vision_agent import VisionAgent as InterceptorAgent, VisionAgentConfig, create_vision_agent
from trajectories6 import WaypointManager, WaypointConfig
from config import TrainingConfig, DomainRandomizationConfig
from vecenv_2 import VecEnv, RunningMeanStd


# helpers

def load_normalization(trainer_like, path: str):
    """Load RunningMeanStd stats saved alongside a checkpoint."""
    obs_path = f"{path}_obs_rms.pt"
    priv_path = f"{path}_priv_rms.pt"
    if trainer_like.obs_rms is not None and os.path.exists(obs_path):
        trainer_like.obs_rms.load(obs_path)
        print(f"  obs  RMS loaded  ← {obs_path}")
    if trainer_like.priv_rms is not None and os.path.exists(priv_path):
        trainer_like.priv_rms.load(priv_path)
        print(f"  priv RMS loaded  ← {priv_path}")


def normalize_obs(obs: torch.Tensor, obs_rms, obs_dim: int, action_dim: int, clip: float = 10.0):
    if obs_rms is None:
        return obs
    if obs.shape[-1] == obs_dim:
        return obs_rms.normalize(obs, clip=clip)
    elif obs.shape[-1] == obs_dim + action_dim:
        return torch.cat([obs_rms.normalize(obs[..., :obs_dim], clip=clip),
                          obs[..., obs_dim:]], dim=-1)
    return obs


# main

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # config
    num_envs = args.num_envs
    config = TrainingConfig(
        num_parallel_envs=num_envs,
        normalize_observations=True,
        use_waypoint_mode=True,
    )
    # Disable domain randomisation for clean eval
    config.domain_randomization.enabled = False

    # agent
    agent_cfg = VisionAgentConfig(
        obs_dim=config.obs_dim,
        privileged_dim=config.privileged_dim,
        action_dim=config.action_dim,
        use_action_smoothing=False,
        use_prev_action=True,
    )
    agent = create_vision_agent(agent_cfg).to(device)
    agent.initialize_state(num_envs, device)

    # normalization holders
    class _NormHolder:
        pass
    norm = _NormHolder()
    norm.obs_rms  = RunningMeanStd(shape=(config.obs_dim,),        device=str(device)) if config.normalize_observations else None
    norm.priv_rms = RunningMeanStd(shape=(config.privileged_dim,), device=str(device)) if config.normalize_observations else None

    # checkpoint
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Support both compiled and plain state-dicts
    state = ckpt.get("agent_state", ckpt)
    try:
        agent.load_state_dict(state)
    except RuntimeError:
        # Compiled model keys start with "_orig_mod."
        stripped = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        agent.load_state_dict(stripped)

    load_normalization(norm, ckpt_path)
    agent.eval()
    print(f"Agent loaded ({sum(p.numel() for p in agent.parameters()):,} params)")

    # env
    waypoint_cfg     = WaypointConfig()
    waypoint_manager = WaypointManager(waypoint_cfg)

    stages = ['straight_lines', 'curved_paths', 'altitude_control']
    env = VecEnv(
        stages, num_envs,
        None, config, None, agent, waypoint_manager
    )

    # renderer
    renderer = QuadrotorRenderer(
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        camera_distance=args.cam_dist,
        show_trails=True,
        trail_length=120,
        max_rendered_quads=min(num_envs, args.max_render),
    )
    print("Controls: WASD=pan  QE=alt  Mouse-drag=orbit  Scroll=zoom  R=record  ESC=quit")

    # eval
    obs        = env.reset()
    running    = True
    step_count = 0
    zero_img   = torch.zeros(num_envs,
                             agent_cfg.image_channels,
                             agent_cfg.image_height,
                             agent_cfg.image_width,
                             device=device)

    while running:
        # toggle recording with R
        for event in pygame.event.get(KEYDOWN):
            if event.key == K_r:
                if renderer._recording:
                    renderer.stop_recording()
                else:
                    renderer.start_recording(args.record_path)

        # policy step
        with torch.no_grad():
            obs_norm = normalize_obs(obs, norm.obs_rms,
                                     config.obs_dim, config.action_dim,
                                     clip=config.obs_clip)
            priv = env.get_privileged_states()
            if norm.priv_rms is not None:
                priv = norm.priv_rms.normalize(priv)

            out = agent(
                None,        # no image
                obs_norm,
                priv,
                deterministic=args.deterministic,
            )

        obs, rewards, dones, truncated, infos = env.step(out["action"])
        step_count += 1

        # render
        running = renderer.render(
            positions   = env.pos,
            quaternions = env.quat,
            waypoints   = env.env_waypoints if config.use_waypoint_mode else env.env_traj_pos,
            current_wp_idx = env.current_waypoint_idx if config.use_waypoint_mode else env.target_idx,
        )

        # optional: print stats
        if step_count % 200 == 0:
            sr   = env.get_epoch_success_rate()
            ravg = rewards.mean().item()
            print(f"step {step_count:6d} | success={sr:.3f} | mean_rew={ravg:.3f}")

        # auto-stop recording
        if renderer._recording and len(renderer._frames) >= args.max_record_frames:
            renderer.stop_recording()

    if renderer._recording:
        renderer.stop_recording()
    renderer.close()
    print("Done.")


# CLI entry point

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualize trained quadrotor agent")

    p.add_argument("--checkpoint",        type=str,   default="./checkpoints_ppo/ckpt_ep300.pt",
                   help="Path to .pt checkpoint file")
    p.add_argument("--num_envs",          type=int,   default=8,
                   help="Number of parallel envs to run (keep low for smooth viz)")
    p.add_argument("--max_render",        type=int,   default=8,
                   help="Max quads to render (capped by num_envs)")
    p.add_argument("--deterministic",     action="store_true", default=True,
                   help="Use deterministic (mean) actions")
    p.add_argument("--fps",               type=int,   default=60)
    p.add_argument("--width",             type=int,   default=1280)
    p.add_argument("--height",            type=int,   default=720)
    p.add_argument("--cam_dist",          type=float, default=50.0)
    p.add_argument("--record_path",       type=str,   default="eval_recording.gif",
                   help="Output path for recording (press R in-window)")
    p.add_argument("--max_record_frames", type=int,   default=3000,
                   help="Auto-stop recording after this many frames")

    args = p.parse_args()
    main(args)