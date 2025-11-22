# train_mario_ram.py
import os
import time
from typing import Callable

from stable_baselines3 import PPO

from config import (
    MODEL_DIR,
    LOG_DIR,
    MODEL_NAME,
    TOTAL_TIMESTEPS,
    LEARNING_RATE,
    CHECK_FREQ,
    SUCCESS_WINDOW,
    N_ENVS,
)
from env_utils import create_env
from callbacks import MarioTrainingCallback


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Same idea as in the notebook:
    lr(progress) = initial_value * progress_remaining
    (progress_remaining goes from 1 to 0 during training).
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Vectorized environment (Mario + JoypadSpace + SMBRamWrapper + Monitor)
    env = create_env(num_envs=N_ENVS)

    # Callback
    callback = MarioTrainingCallback(
        check_freq=CHECK_FREQ,
        save_path=str(MODEL_DIR),
        starting_steps=0,
        success_window=SUCCESS_WINDOW,
        verbose=1,
    )

    # PPO model (same as notebook: MlpPolicy + linear_schedule(3e-4))
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=linear_schedule(LEARNING_RATE),
        tensorboard_log=str(LOG_DIR),
        # you can add more hyperparameters here if you want to tune
    )

    print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
    t_start = time.time()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
    )

    t_elapsed = time.time() - t_start
    print(f"Wall time: {round(t_elapsed, 2)} s")

    # Save final model
    final_path = MODEL_DIR / f"{MODEL_NAME}_final"
    model.save(str(final_path))
    print(f"Final model saved to: {final_path}")

    env.close()


if __name__ == "__main__":
    main()