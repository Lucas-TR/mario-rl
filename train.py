# train.py
# Minimal PPO training script with newline progress + rolling metrics (no progress bar).
# Prints: steps/total, %, sps, elapsed, ETA, ep_r100, ep_len100, x_max100, flag@100.

import warnings
warnings.filterwarnings("ignore", message="overflow encountered", category=RuntimeWarning)

import argparse
import os
import sys
import time
from datetime import timedelta, datetime
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from mario_env import make_env
from utils import default_paths, seed_everything


# -------------------- Callbacks --------------------

class TrainAndLoggingCallback(BaseCallback):
    """Save periodic checkpoints at 'check_freq' steps."""
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = int(check_freq)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)
        return True


class ProgressAndMetricsCallback(BaseCallback):
    """
    Line-based progress + rolling metrics (works with TTY or pipes).
    - Prints every `print_every_steps` or ~`print_every_sec`, whichever comes first.
    - Metrics are rolling means over the last `window` episodes.
    """
    def __init__(self, total_timesteps: int, print_every_steps: int = 10_000,
                 print_every_sec: float = 2.0, window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.total = int(total_timesteps)
        self.print_every_steps = int(print_every_steps)
        self.print_every_sec = float(print_every_sec)
        self.window = int(window)

        # rolling buffers
        self.ep_rewards = deque(maxlen=self.window)
        self.ep_lens    = deque(maxlen=self.window)
        self.ep_xmax    = deque(maxlen=self.window)
        self.ep_flag    = deque(maxlen=self.window)

        # per-env trackers for x_pos max during an episode
        self.max_x = None

        # progress state
        self.last_print_step = 0
        self.last_print_time = 0.0
        self.start_time = 0.0

    def _ensure_max_x(self):
        # number of parallel envs
        try:
            n = getattr(self.training_env, "num_envs", 1)
        except Exception:
            n = 1
        if self.max_x is None or len(self.max_x) != n:
            self.max_x = [0 for _ in range(n)]

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.last_print_time = self.start_time
        self._ensure_max_x()

    def _maybe_print(self):
        n = self.model.num_timesteps
        now = time.time()
        if (n - self.last_print_step >= self.print_every_steps) or (now - self.last_print_time >= self.print_every_sec):
            elapsed = now - self.start_time
            sps = n / max(elapsed, 1e-9)
            remaining = max(self.total - n, 0)
            eta_sec = remaining / max(sps, 1e-9)
            pct = 100.0 * n / max(self.total, 1)

            def _mean(q):
                return float(sum(q)) / len(q) if len(q) > 0 else float("nan")

            r100 = _mean(self.ep_rewards)
            l100 = _mean(self.ep_lens)
            x100 = _mean(self.ep_xmax)
            f100 = (_mean(self.ep_flag) * 100.0) if len(self.ep_flag) > 0 else float("nan")

            elapsed_td = str(timedelta(seconds=int(elapsed)))
            eta_td = str(timedelta(seconds=int(eta_sec)))
            eta_abs_str = (datetime.now() + timedelta(seconds=int(eta_sec))).strftime("%H:%M:%S")

            print(
                f"{pct:5.1f}% | {n:,}/{self.total:,} steps | {int(sps):4d} sps | "
                f"elapsed {elapsed_td} | ETA {eta_td} (~{eta_abs_str}) | "
                f"ep_r100 {r100:7.2f} | ep_len100 {l100:6.1f} | x_max100 {x100:6.1f} | "
                f"flag@100 {f100:5.1f}%",
                flush=True,
            )
            self.last_print_step = n
            self.last_print_time = now


    def _on_step(self) -> bool:
        # collect infos (VecEnv: list of dicts)
        infos = self.locals.get("infos", None)
        self._ensure_max_x()
        if infos is not None:
            for i, info in enumerate(infos):
                # track per-episode max x
                x = info.get("x_pos", None)
                if x is not None and i < len(self.max_x):
                    if x > self.max_x[i]:
                        self.max_x[i] = x
                # when an episode ends, Monitor inserts 'episode'
                ep = info.get("episode")
                if ep:
                    self.ep_rewards.append(float(ep.get("r", 0.0)))
                    self.ep_lens.append(float(ep.get("l", 0.0)))
                    self.ep_xmax.append(float(self.max_x[i] if i < len(self.max_x) else 0.0))
                    self.ep_flag.append(1.0 if info.get("flag_get") else 0.0)
                    if i < len(self.max_x):
                        self.max_x[i] = 0  # reset for next episode

        self._maybe_print()
        return True

    def _on_training_end(self) -> None:
        print("Training finished step loop.", flush=True)


# -------------------- Main training --------------------

def main(
    total_steps: int,
    lr: float,
    n_steps: int,
    batch_size: int,
    ent_coef: float,
    gamma: float,
    check_freq: int,
    seed: int,
    num_envs: int,
    pre_run_long: int,
    pre_run_short: int,
    jump_hold: int,
    k_jump: int,
    k_other: int,
    resize: int,
    frame_stack: int,
    level_id: str,
    no_tb: bool,
    device: str,
):
    seed_everything(seed)
    ckpt_dir, log_dir = default_paths(".", tag=f"ppo_s{seed}")

    env = make_env(
        num_envs=num_envs,
        grayscale=True,
        resize=resize,
        frame_stack=frame_stack,
        level_id=level_id,
        pre_run_long=pre_run_long,
        pre_run_short=pre_run_short,
        jump_hold=jump_hold,
        k_jump=k_jump,
        k_other=k_other,
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,                           # keep console clean; our callback handles logging
        tensorboard_log=None if no_tb else log_dir,
        learning_rate=lr,
        n_steps=n_steps,                     # per env
        batch_size=batch_size,
        ent_coef=ent_coef,
        gamma=gamma,
        n_epochs=4,
        clip_range=0.1,
        device=device,                       # "auto" | "cuda:0" | "cuda:1" | "cpu"
        seed=seed,
    )

    cb_ckpt  = TrainAndLoggingCallback(check_freq=check_freq, save_path=ckpt_dir)
    cb_prog  = ProgressAndMetricsCallback(total_timesteps=total_steps, print_every_steps=10_000, print_every_sec=2.0)

    model.learn(total_timesteps=total_steps, callback=[cb_ckpt, cb_prog])

    model.save(os.path.join(ckpt_dir, "final_model"))
    print(f"\nTraining finished. Checkpoints in: {ckpt_dir}")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # PPO budget / algo
    ap.add_argument("--total_steps", type=int, default=8_000_000)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--n_steps", type=int, default=256)        # per env
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--ent_coef", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--check_freq", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_envs", type=int, default=8)

    ap.add_argument("--device", type=str, default="auto", help='e.g., "cuda:0", "cuda:1", "cpu", or "auto"')

    # Macro/repeat tuning for pipe-to-pipe
    ap.add_argument("--pre_run_long", type=int, default=10, help="run-up frames (Right+B) from ground")
    ap.add_argument("--pre_run_short", type=int, default=5, help="short run-up when on the first pipe")
    ap.add_argument("--jump_hold", type=int, default=16, help="hold Right+A+B frames for long jump")
    ap.add_argument("--k_jump", type=int, default=10, help="extra repeats when jumping")
    ap.add_argument("--k_other", type=int, default=2, help="short taps for fine alignment")

    # Visual config
    ap.add_argument("--resize", type=int, default=84)
    ap.add_argument("--frame_stack", type=int, default=4)
    ap.add_argument("--level_id", type=str, default="SuperMarioBros-1-1-v0")

    # Logging
    ap.add_argument("--no_tb", action="store_true", help="disable TensorBoard logging")

    args = ap.parse_args()
    main(**vars(args))
