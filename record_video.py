# record_video.py
# Robust manual video recorder for SB3 + Mario in WSL.
# - Infers model's expected obs shape (CHW vs HWC, 84 vs 240x256)
# - Builds a single-env DummyVecEnv accordingly
# - Records frames via env.render("rgb_array") to an MP4

import argparse
import os
import time
from typing import Tuple

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage

from mario_env import make_env

def infer_requirements(obs_shape: Tuple[int, ...]) -> dict:
    """
    Infer resize and channels_order from model.observation_space.shape.
    Returns dict with keys: resize (int), channels_order ('first'|'last').
    """
    if len(obs_shape) != 3:
        raise ValueError(f"Unexpected obs shape {obs_shape}, expected 3D.")

    c0, c1, c2 = obs_shape
    # Heuristics:
    # - Native NES: HxW ~ 240x256
    # - Common downsample: 84x84
    # - frame_stack=4 -> either CHW: (4,H,W) or HWC: (H,W,4)

    if (c0 in (4, 1, 3)) and (c1 in (84, 240)) and (c2 in (84, 256)):
        # CHW
        resize = 84 if (c1 == 84 and c2 == 84) else 0
        return dict(resize=resize, channels_order="first")
    elif (c0 in (84, 240)) and (c1 in (84, 256)) and (c2 in (4, 1, 3)):
        # HWC
        resize = 84 if (c0 == 84 and c1 == 84) else 0
        return dict(resize=resize, channels_order="last")
    else:
        # Fallback: assume 84x84 HWC
        return dict(resize=84, channels_order="last")

def main(args):
    # Load model first to read its expected obs space
    model = PPO.load(args.model, device=args.device)
    obs_shape = tuple(model.observation_space.shape)
    cfg = infer_requirements(obs_shape)

    # Build single env (DummyVecEnv path) with matched params
    env = make_env(
        num_envs=1,
        grayscale=True,
        use_resize=bool(cfg["resize"] and cfg["resize"] > 0),
        resize=cfg["resize"],
        frame_stack=args.frame_stack,
        level_id=args.level_id,
        use_macros=args.use_macros,
        use_shaping=args.use_shaping,
        use_stuck=args.use_stuck,
        action_set=args.action_set,
        channels_order=cfg["channels_order"],
        # episode controls
        stuck_patience=args.stuck_patience,
        max_episode_steps=args.max_episode_steps,
    )

    # If shapes mismatched only by transpose, fix with VecTransposeImage
    env_shape = tuple(env.observation_space.shape)
    if env_shape != obs_shape:
        # try to resolve by transposing
        # CHW <-> HWC swap
        if sorted(env_shape) == sorted(obs_shape):
            env = VecTransposeImage(env)
            # refresh shape
            env_shape = tuple(env.observation_space.shape)

    if env_shape != obs_shape:
        raise RuntimeError(
            f"Env observation shape {env_shape} != model expected {obs_shape}. "
            f"Try adjusting --frame_stack / --resize / toggles to match training."
        )

    # Prepare writer
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = args.prefix or f"mario_{int(time.time())}"
    out_path = os.path.join(args.out_dir, f"{prefix}.mp4")
    writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264")

    # Rollout and record
    obs = env.reset()
    steps = 0
    while steps < args.video_length:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Grab frame (DummyVecEnv returns a single RGB array)
        frame = env.render(mode="rgb_array")
        if isinstance(frame, list):   # very defensive: some vec envs return list
            frame = frame[0]
        if frame is None or not isinstance(frame, np.ndarray):
            raise RuntimeError("render('rgb_array') returned no frame.")

        writer.append_data(frame)

        steps += 1
        if done.any() if hasattr(done, "any") else bool(done):
            obs = env.reset()

    writer.close()
    env.close()
    print(f"\nSaved video: {os.path.abspath(out_path)}")
    print(f"Steps: {steps} | FPS: {args.fps} | Obs shape matched: {env_shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .zip (e.g., train/ppo_s42/final_model.zip)")
    ap.add_argument("--device", type=str, default="auto")

    # Video params
    ap.add_argument("--out_dir", type=str, default="videos")
    ap.add_argument("--prefix", type=str, default="")
    ap.add_argument("--video_length", type=int, default=6000)
    ap.add_argument("--fps", type=int, default=30)

    # Env options — keep consistent with training (macros/shaping/stuck/action_set)
    ap.add_argument("--level_id", type=str, default="SuperMarioBros-1-1-v0")
    ap.add_argument("--frame_stack", type=int, default=4)
    ap.add_argument("--use_macros", action="store_true")
    ap.add_argument("--use_shaping", action="store_true")
    ap.add_argument("--use_stuck", action="store_true")
    ap.add_argument("--action_set", type=str, default="pipe", choices=["simple", "right_only", "pipe"])
    ap.add_argument("--stuck_patience", type=int, default=200)
    ap.add_argument("--max_episode_steps", type=int, default=3000)

    args = ap.parse_args()
    main(args)

