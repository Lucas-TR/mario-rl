# Mario-RL (SB3 + PPO)

Minimal, reproducible PPO training for **Super Mario Bros** using **Stable-Baselines3**, with clean CSV logging and periodic checkpoints. Includes optional macro actions, reward shaping, and anti-stall resets.

## Repo Layout

- **`mario_env.py`** — Environment factory and wrappers  
  - `VariableActionRepeat` (longer repeats when jumping)  
  - `MacroLongJump` (run-up + hold jump macro)  
  - `RewardShaping` (+progress, +flag, −death)  
  - `StuckReset` (early terminate if no x-progress)  
  - `make_env(...)` builds a vectorized env with `VecFrameStack` and `VecMonitor`
- **`train.py`** — PPO training loop  
  - Saves: `best_model_*.zip` (every N steps), `final_model.zip`, `episodes.csv`, `metrics_bucket.csv`, `config.json`
  - Two CSV loggers: per-episode and per-bucket (rolling window over last 100 episodes)
- **`utils.py`** — Small helpers (`default_paths`, `seed_everything`)
- *(Optional)* **`eval.py`** — Run a trained model for live playback  
- *(Optional)* **`record_video.py`** — Save gameplay to `.mp4` (uses `imageio` + `imageio-ffmpeg`)

## Dependencies

- Python ≥ 3.9  
- `torch`, `stable-baselines3`  
- `gym==0.21.0`, `gym-super-mario-bros==7.3.0`, `nes-py`  
- Optional for video: `imageio`, `imageio-ffmpeg`

```bash
pip install torch stable-baselines3==2.* gym==0.21.0 gym-super-mario-bros==7.3.0 nes-py
# Optional:
pip install imageio imageio-ffmpeg
```

> Note: `gym-super-mario-bros` uses the NES emulator backend (`nes-py`); no MuJoCo is required.

## Environment Details (defaults)

- **Observation pipeline:** grayscale (`keep_dim=True`) → optional `ResizeObservation` to `(84,84)` → `VecFrameStack`  
- **Stacking:** by default **channels_order="last"** → shape `(H, W, C * frame_stack)` = `(84, 84, 4)`  
  - Keep **train/eval exactly consistent** (frame_stack, resize, channels_order).  
- **Action sets:**  
  - `"right_only"` (NES right, right+jump)  
  - `"simple"` (predefined by gym-smb)  
  - `"pipe"` (compact set optimized for forward progress & long jumps)
- **Macros / repeats:**  
  - `MacroLongJump`: optional run-up (`pre_run_short` or `pre_run_long`) then hold `right+A+B` for `jump_hold` frames  
  - `VariableActionRepeat`: if action includes `A` (jump), repeat `k_jump` frames; else `k_other`
- **Reward shaping (optional):**  
  - `reward += right_coef * Δx`, `+flag_reward` if `flag_get`, `+death_penalty` if done without flag  
- **Anti-stall:** `StuckReset(patience)` ends eps if no forward progress for `patience` frames  
- **Time limit:** `TimeLimit(max_episode_steps)` ensures episodes end

### Key `make_env(...)` arguments
```
num_envs, grayscale, use_resize, resize, frame_stack, level_id,
use_macros, use_shaping, use_stuck, action_set,
stuck_patience, max_episode_steps, channels_order,
pre_run_long, pre_run_short, jump_hold, k_jump, k_other
```

## What gets logged

**Folder:** `./train/<tag>/` where `<tag>` is `ppo_s{seed}` by default.

- **`config.json`** — snapshot of the exact run config  
- **`best_model_*.zip`** — checkpoints every `--check_freq` steps  
- **`final_model.zip`** — final weights  
- **`episodes.csv`** (one row per completed episode):  
  `global_step,episode_idx,reward,length,x_max,flag`
  - `x_max`: max horizontal progress for that episode  
  - `flag`: `1` if `info["flag_get"]` was seen in the episode
- **`metrics_bucket.csv`** (one row per bucket):  
  `global_step,pct,sps,elapsed_s,eta_s,episodes,ep_r100,ep_len100,x_max100,flag@100`
  - Rolling stats over last 100 completed episodes

These CSVs are ready for plotting **reward vs steps**, **flag rate vs steps**, etc.

## Training

Example (recommended baseline with macros + shaping + anti-stall):

```bash
python train.py \
  --device cuda:0 \
  --total_steps 20000000 \
  --num_envs 16 \
  --n_steps 512 \
  --batch_size 4096 \
  --lr 2.5e-4 \
  --ent_coef 0.01 \
  --gamma 0.999 \
  --target_kl 0.01 \
  --check_freq 5000000 \
  --frame_stack 4 \
  --resize 84 \
  --level_id SuperMarioBros-1-1-v0 \
  --action_set pipe \
  --use_macros --use_shaping --use_stuck --use_resize \
  --stuck_patience 350 \
  --log_bucket 100000
```

**Notes**
- `batch_size` must divide `n_steps * num_envs` (here: `512 * 16 = 8192`, and `4096` is valid).  
- More `num_envs` increases sample throughput if your GPU/CPU can handle it.  
- Keep your **eval/record settings identical** to training (especially `frame_stack`, `resize`, `channels_order`, `action_set`, and macro toggles).

## Evaluate / Watch (optional)

```bash
python eval.py \
  --model ./train/ppo_s42/final_model.zip \
  --render \
  --deterministic \
  --seconds 60 --fps 30
```

> If you change the observation pipeline (e.g., resize/stacking), ensure the eval env matches training.

## Record Video (optional)

```bash
# Requires: imageio, imageio-ffmpeg
python record_video.py \
  --model ./train/ppo_s42/final_model.zip \
  --device cuda:0 \
  --level_id SuperMarioBros-1-1-v0 \
  --frame_stack 4 \
  --use_macros --use_shaping --use_stuck --use_resize \
  --action_set pipe \
  --video_length 6000 --fps 30 \
  --out_dir videos --prefix run_final
```

## Reproducibility

- We fix seeds via `seed_everything(seed)` and store the full CLI/config in `config.json`.  
- To compare runs, align: `level_id`, `action_set`, `resize`, `frame_stack`, `use_*` toggles, and PPO hyperparameters.

## Troubleshooting

- **Shape mismatch at inference:** ensure eval env uses the same `frame_stack`, `resize`, and **channels_order** as training.  
- **Unstable returns / rare flag gets:** reduce `ent_coef`, increase `gamma`, or raise `stuck_patience`. Long-horizon tasks can show noisy reward; track `flag@100` and `x_max100` in `metrics_bucket.csv`.
