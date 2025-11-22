# env_utils.py
from typing import Callable, List

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_utils import SMBRamWrapper   # Your wrapper from the repo

from config import (
    ENV_ID,
    X0, X1,
    Y0, Y1,
    N_STACK,
    N_SKIP,
)


def make_single_env() -> Callable:
    """
    Build a single Mario environment with:
      - JoypadSpace(SIMPLE_MOVEMENT)
      - SMBRamWrapper (crop, stack, skip)
      - Monitor for logging
    """
    def _init():
        env = gym_super_mario_bros.make(ENV_ID)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        env = SMBRamWrapper(
            env,
            crop_dim=[X0, X1, Y0, Y1],
            n_stack=N_STACK,
            n_skip=N_SKIP,
        )

        env = Monitor(env)  # to log rewards/episodes
        return env

    return _init


def create_env(num_envs: int = 1):
    """
    Create a DummyVecEnv with num_envs copies of the processed environment.
    """
    env_fns: List[Callable] = [make_single_env() for _ in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)
    return vec_env