# callbacks.py
import os
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MarioTrainingCallback(BaseCallback):
    """
    Callback that:
    - Saves models every `check_freq` steps.
    - Logs episode metrics to TensorBoard:
        * episode/reward
        * episode/length
        * episode/max_x
        * episode/flag (0/1)
        * episode/flag_success_rate (moving average)
    """

    def __init__(
        self,
        check_freq: int,
        save_path: str,
        starting_steps: int = 0,
        success_window: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.check_freq = int(check_freq)
        self.save_path = save_path
        self.starting_steps = int(starting_steps)
        self.success_window = int(success_window)

        self.flag_success_buffer = deque(maxlen=self.success_window)

        self.ep_rewards = None
        self.ep_lengths = None
        self.ep_max_x = None

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

        n_envs = self.training_env.num_envs
        self.ep_rewards = np.zeros(n_envs, dtype=np.float32)
        self.ep_lengths = np.zeros(n_envs, dtype=np.int32)
        self.ep_max_x = np.zeros(n_envs, dtype=np.float32)

    def _on_step(self) -> bool:
        # 1) Periodic checkpoint saving
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path,
                f"model_{self.n_calls + self.starting_steps}",
            )
            self.model.save(model_path)
            if self.verbose:
                print(f"[Callback] Saved checkpoint to {model_path}")

        # 2) Episode statistics
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards")

        if infos is None or dones is None or rewards is None:
            return True

        dones = np.array(dones)
        rewards = np.array(rewards)

        for env_idx, info in enumerate(infos):
            r = float(rewards[env_idx])
            done = bool(dones[env_idx])

            self.ep_rewards[env_idx] += r
            self.ep_lengths[env_idx] += 1

            # Horizontal position
            x_pos = info.get("x_pos", 0.0)
            self.ep_max_x[env_idx] = max(self.ep_max_x[env_idx], float(x_pos))

            if done:
                flag = int(info.get("flag_get", False))
                self.flag_success_buffer.append(flag)

                self.logger.record("episode/reward", self.ep_rewards[env_idx])
                self.logger.record("episode/length", self.ep_lengths[env_idx])
                self.logger.record("episode/max_x", self.ep_max_x[env_idx])
                self.logger.record("episode/flag", flag)

                if len(self.flag_success_buffer) > 0:
                    success_rate = float(np.mean(self.flag_success_buffer))
                    self.logger.record(
                        "episode/flag_success_rate",
                        success_rate,
                    )

                # Reset stats for this env
                self.ep_rewards[env_idx] = 0.0
                self.ep_lengths[env_idx] = 0
                self.ep_max_x[env_idx] = 0.0

        return True