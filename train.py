# train.py
# PPO training with clean bucket logs + per-episode CSV logs.
# Files saved under: ./train/<tag>/  ->  {best_model_*.zip, final_model.zip, episodes.csv, metrics_bucket.csv, config.json}

import warnings
warnings.filterwarnings("ignore", message="overflow encountered", category=RuntimeWarning)

from typing import Optional
import argparse
import os
import json
import time
from datetime import timedelta, datetime
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from mario_env import make_env
from utils import default_paths, seed_everything


# ---------- Callbacks ----------

class TrainAndLoggingCallback(BaseCallback):
    """Save periodic checkpoints at 'check_freq' steps."""
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = int(check_freq)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)
        return True


class EpisodeCSVLogger(BaseCallback):
    """
    Write one row per *completed* episode:
    columns = [global_step, episode_idx, reward, length, x_max, flag]
    """
    def __init__(self, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.fpath = os.path.join(self.save_dir, "episodes.csv")
        self.fh = None
        self.header_written = False
        self.episode_idx = 0
        self.max_x = None  # per-env running x max

    def _ensure_max_x(self):
        try:
            n = getattr(self.training_env, "num_envs", 1)
        except Exception:
            n = 1
        if self.max_x is None or len(self.max_x) != n:
            self.max_x = [0 for _ in range(n)]

    def _on_training_start(self) -> None:
        self._ensure_max_x()
        self.fh = open(self.fpath, "a", buffering=1)  # line-buffered
        if not self.header_written and (self.fh.tell() == 0):
            self.fh.write("global_step,episode_idx,reward,length,x_max,flag\n")
            self.header_written = True

    def _on_step(self) -> bool:
        self._ensure_max_x()
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        for i, info in enumerate(infos):
            x = info.get("x_pos", None)
            if x is not None and i < len(self.max_x) and x > self.max_x[i]:
                self.max_x[i] = x

            ep = info.get("episode")
            if ep:
                self.episode_idx += 1
                r = float(ep.get("r", 0.0))
                l = float(ep.get("l", 0.0))
                x_max = float(self.max_x[i] if i < len(self.max_x) else 0.0)
                flag = 1 if info.get("flag_get") else 0
                gstep = int(self.model.num_timesteps)
                self.fh.write(f"{gstep},{self.episode_idx},{r},{l},{x_max},{flag}\n")
                if i < len(self.max_x):
                    self.max_x[i] = 0  # reset for next episode

        return True

    def _on_training_end(self) -> None:
        if self.fh is not None:
            self.fh.flush()
            self.fh.close()
            self.fh = None


class BucketCSVLogger(BaseCallback):
    """
    Log once per fixed step bucket (e.g., every 50,000 steps) and write to metrics_bucket.csv.
    Rolling metrics are computed over the last 'window' completed episodes.
    """
    def __init__(self, save_dir: str, total_timesteps: int, bucket: int = 50_000, window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.total = int(total_timesteps)
        self.bucket = max(1, int(bucket))
        self.window = int(window)

        self.ep_rewards = deque(maxlen=self.window)
        self.ep_lens    = deque(maxlen=self.window)
        self.ep_xmax    = deque(maxlen=self.window)
        self.ep_flag    = deque(maxlen=self.window)

        self.max_x = None  # per-env running max x during current episode
        self.start_time = 0.0
        self.last_bucket_idx = -1
        self.completed_eps = 0

        self.fpath = os.path.join(self.save_dir, "metrics_bucket.csv")
        self.fh = None
        self.header_written = False

    def _ensure_max_x(self):
        try:
            n = getattr(self.training_env, "num_envs", 1)
        except Exception:
            n = 1
        if self.max_x is None or len(self.max_x) != n:
            self.max_x = [0 for _ in range(n)]

    def _fmt(self, x):
        # Pretty print with '-' if NaN / empty
        if x is None:
            return "-"
        try:
            if isinstance(x, float) and (x != x):  # NaN check
                return "-"
            return f"{x:.2f}"
        except Exception:
            return str(x)

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self._ensure_max_x()
        self.fh = open(self.fpath, "a", buffering=1)
        if not self.header_written and (self.fh.tell() == 0):
            self.fh.write("global_step,pct,sps,elapsed_s,eta_s,episodes,ep_r100,ep_len100,x_max100,flag@100\n")
            self.header_written = True

    def _maybe_log(self):
        n = self.model.num_timesteps
        cur_bucket = n // self.bucket
        if cur_bucket <= self.last_bucket_idx:
            return  # not yet crossed a bucket boundary

        elapsed = time.time() - self.start_time
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

        # Console line
        elapsed_td = str(timedelta(seconds=int(elapsed)))
        eta_td = str(timedelta(seconds=int(eta_sec)))
        eta_abs = (datetime.now() + timedelta(seconds=int(eta_sec))).strftime("%H:%M:%S")
        print(
            f"{pct:5.1f}% | {n:,}/{self.total:,} steps | {int(sps):4d} sps | "
            f"elapsed {elapsed_td} | ETA {eta_td} (~{eta_abs}) | "
            f"eps {self.completed_eps:,} | "
            f"ep_r100 {self._fmt(r100)} | ep_len100 {self._fmt(l100)} | "
            f"x_max100 {self._fmt(x100)} | flag@100 {self._fmt(f100)}%",
            flush=True,
        )

        # CSV line
        self.fh.write(f"{n},{pct:.2f},{sps:.2f},{elapsed:.1f},{eta_sec:.1f},{self.completed_eps},{self._fmt(r100)},{self._fmt(l100)},{self._fmt(x100)},{self._fmt(f100)}\n")

        self.last_bucket_idx = cur_bucket

    def _on_step(self) -> bool:
        self._ensure_max_x()
        infos = self.locals.get("infos", None)
        if infos is not None:
            for i, info in enumerate(infos):
                x = info.get("x_pos", None)
                if x is not None and i < len(self.max_x) and x > self.max_x[i]:
                    self.max_x[i] = x

                ep = info.get("episode")
                if ep:
                    self.completed_eps += 1
                    self.ep_rewards.append(float(ep.get("r", 0.0)))
                    self.ep_lens.append(float(ep.get("l", 0.0)))
                    self.ep_xmax.append(float(self.max_x[i] if i < len(self.max_x) else 0.0))
                    self.ep_flag.append(1.0 if info.get("flag_get") else 0.0)
                    if i < len(self.max_x):
                        self.max_x[i] = 0  # reset for next episode

        self._maybe_log()
        return True

    def _on_training_end(self) -> None:
        if self.fh is not None:
            self.fh.flush()
            self.fh.close()
            self.fh = None


# ---------- Main training ----------

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
    # env visuals / level
    resize: int,
    frame_stack: int,
    level_id: str,
    # env toggles
    use_macros: bool,
    use_shaping: bool,
    use_stuck: bool,
    use_resize: bool,
    action_set: str,
    stuck_patience: int,
    max_episode_steps: int,
    # PPO / logging
    device: str,
    no_tb: bool,
    target_kl: Optional[float],
    log_bucket: int,
):
    seed_everything(seed)
    ckpt_dir, log_dir = default_paths(".", tag=f"ppo_s{seed}")

    # Save config snapshot for reproducibility
    run_cfg = dict(
        total_steps=total_steps, lr=lr, n_steps=n_steps, batch_size=batch_size,
        ent_coef=ent_coef, gamma=gamma, check_freq=check_freq, seed=seed,
        num_envs=num_envs, resize=resize, frame_stack=frame_stack, level_id=level_id,
        use_macros=use_macros, use_shaping=use_shaping, use_stuck=use_stuck, use_resize=use_resize,
        action_set=action_set, stuck_patience=stuck_patience, max_episode_steps=max_episode_steps,
        device=device, no_tb=no_tb, target_kl=target_kl, log_bucket=log_bucket
    )
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    env = make_env(
        num_envs=num_envs,
        grayscale=True,
        use_resize=use_resize,
        resize=resize,
        frame_stack=frame_stack,
        level_id=level_id,
        use_macros=use_macros,
        use_shaping=use_shaping,
        use_stuck=use_stuck,
        action_set=action_set,
        pre_run_long=10, pre_run_short=3, jump_hold=14,  # safe defaults
        k_jump=10, k_other=2,
        stuck_patience=stuck_patience,
        max_episode_steps=max_episode_steps,
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,  # logs handled by our callbacks
        tensorboard_log=None if no_tb else log_dir,
        learning_rate=lr,
        n_steps=n_steps,               # per env
        batch_size=batch_size,
        ent_coef=ent_coef,
        gamma=gamma,
        n_epochs=4,
        clip_range=0.1,
        device=device,                 # "auto" | "cuda:0" | "cuda:1" | "cpu"
        seed=seed,
        target_kl=target_kl,           # None to disable KL early stop
    )

    cb_ckpt   = TrainAndLoggingCallback(check_freq=check_freq, save_path=ckpt_dir)
    cb_epcsv  = EpisodeCSVLogger(save_dir=ckpt_dir)
    cb_bucket = BucketCSVLogger(save_dir=ckpt_dir, total_timesteps=total_steps, bucket=log_bucket, window=100)

    model.learn(total_timesteps=total_steps, callback=[cb_ckpt, cb_epcsv, cb_bucket])

    model.save(os.path.join(ckpt_dir, "final_model"))
    print(f"\nTraining finished. Run artifacts in: {ckpt_dir}")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # PPO budget / algo
    ap.add_argument("--total_steps", type=int, default=5_000_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_steps", type=int, default=128)           # per env
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--check_freq", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto", help='e.g., "cuda:0", "cuda:1", "cpu", or "auto"')

    # Visual / level
    ap.add_argument("--resize", type=int, default=84)
    ap.add_argument("--frame_stack", type=int, default=4)
    ap.add_argument("--level_id", type=str, default="SuperMarioBros-1-1-v0")

    # Env toggles
    ap.add_argument("--use_macros", action="store_true", help="Enable MacroLongJump + VariableActionRepeat")
    ap.add_argument("--use_shaping", action="store_true", help="Enable RewardShaping")
    ap.add_argument("--use_stuck", action="store_true", help="Enable anti-stall episode resets")
    ap.add_argument("--use_resize", action="store_true", help="Enable ResizeObservation")
    ap.add_argument("--action_set", type=str, default="right_only", choices=["simple", "right_only", "pipe"])

    # Episode control
    ap.add_argument("--stuck_patience", type=int, default=200)
    ap.add_argument("--max_episode_steps", type=int, default=3000)

    # PPO extra
    ap.add_argument("--target_kl", type=float, default=None, help="KL early stopping (e.g., 0.01)")

    # Logging
    ap.add_argument("--no_tb", action="store_true", help="Disable TensorBoard logging")
    ap.add_argument("--log_bucket", type=int, default=50_000, help="Print & CSV once per this many steps")

    args = ap.parse_args()
    main(**vars(args))
