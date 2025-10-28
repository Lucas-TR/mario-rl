# eval.py
# Load a trained PPO model and (optionally) render gameplay.

import argparse
from stable_baselines3 import PPO
from mario_env import make_env

def _get_base_env(env):
    # If vectorized, return the underlying base env for rendering
    return getattr(env, "envs", [env])[0]

def main(model_path: str, render: bool, deterministic: bool, seconds: int, fps: int):
    env = make_env(num_envs=1)
    base_env = _get_base_env(env)
    model = PPO.load(model_path)

    steps = seconds * fps if seconds > 0 else None
    obs = env.reset()
    t = 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        if render:
            base_env.render()
        if hasattr(done, "any") and done.any():
            obs = env.reset()
        t += 1
        if steps is not None and t >= steps:
            break

    env.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .zip (e.g., ./train/ppo_s42/final_model.zip)")
    ap.add_argument("--render", action="store_true", help="Show game window if your system supports it")
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic policy at inference")
    ap.add_argument("--seconds", type=int, default=60, help="Limit playback length (0 = infinite)")
    ap.add_argument("--fps", type=int, default=30, help="Playback FPS for --seconds limit")
    args = ap.parse_args()
    main(args.model, args.render, args.deterministic, args.seconds, args.fps)
