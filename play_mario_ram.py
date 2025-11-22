# play_mario_ram.py
import time
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from env_utils import create_env  # your env: Mario + wrappers


# ============================================================
# CONFIG: SINGLE PRECONFIGURED MODEL (OPTION 1)
# ============================================================
MODEL_CONFIGS = {
    "1": {
        "description": "Main run (e.g., blue curve on TensorBoard)",
        "run_dir": Path("models/ppo_ram_notebook/run_20251120_092432"),
        "model_name": "ppo_ram_notebook_final",  # or model_3000000, etc.
    },
}
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play Super Mario Bros with a trained PPO model."
    )

    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="1",
        help="Preconfigured model to use (only option: 1).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to play (default: 1).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (deterministic=False).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = MODEL_CONFIGS[args.model]

    run_dir = cfg["run_dir"]
    model_name = cfg["model_name"]

    model_path = run_dir / f"{model_name}.zip"
    vecnorm_path = run_dir / "vecnormalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Using config [{args.model}]: {cfg['description']}")
    print(f"  run_dir   = {run_dir}")
    print(f"  model     = {model_name}.zip")

    # 1) Create base env (1 env for playing)
    env = create_env(num_envs=1)

    # 2) Load normalization stats if they exist
    if vecnorm_path.exists():
        print(f"Loading VecNormalize from: {vecnorm_path}")
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False      # do not keep updating stats
        env.norm_reward = False   # show raw rewards
    else:
        print("WARNING: vecnormalize.pkl not found; "
              "the model will see unnormalized observations (performance may drop).")

    # 3) Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(str(model_path), env=env)

    # 4) Run episodes and render
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=not args.stochastic
            )
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(0.02)  # slow down a bit to watch

        print(f"[Episode {ep}] Total reward: {episode_reward}")

    env.close()


if __name__ == "__main__":
    main()