# record_video.py
# Render a trained PPO Mario agent and save an .mp4.
# Auto-loads run config (config.json) and VecNormalize stats from the model's folder.

import argparse
import os
import json
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from mario_env import make_env

def unwrap_base_env(env):
    # Walk through vec wrappers to find the underlying base env for rgb_array rendering
    v = env
    seen = set()
    while True:
        if hasattr(v, "envs"):  # VecEnv -> take first
            try:
                v = v.envs[0]
                continue
            except Exception:
                pass
        if hasattr(v, "venv"):
            v = v.venv
            # guard against weird wrapper cycles
            if id(v) in seen:
                break
            seen.add(id(v))
            continue
        # reached a non-vec, non-wrapped leaf (should support render)
        return v

def load_run_config(run_dir):
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, "r") as f:
        return json.load(f)

def default_if_none(dct, key, default):
    v = dct.get(key, None)
    return v if v is not None else default

def main(args):
    # Infer run dir (parent of model file)
    run_dir = os.path.dirname(os.path.abspath(args.model))
    if os.path.basename(run_dir) == "":
        run_dir = os.path.dirname(run_dir)

    cfg = load_run_config(run_dir)

    # Build an env that mirrors training
    env = make_env(
        num_envs=1,
        grayscale=True,
        use_resize=cfg.get("use_resize", True),
        resize=cfg.get("resize", 84),
        frame_stack=cfg.get("frame_stack", 4),
        level_id=cfg.get("level_id", "SuperMarioBros-1-1-v0"),
        use_macros=cfg.get("use_macros", True),
        use_shaping=cfg.get("use_shaping", True),
        use_stuck=cfg.get("use_stuck", True),
        action_set=cfg.get("action_set", "pipe"),
        stuck_patience=cfg.get("stuck_patience", 350),
        max_episode_steps=cfg.get("max_episode_steps", 3000),
        channels_order="last",
    )

    # Attach VecNormalize stats if available
    vecnorm_path = args.vecnorm if args.vecnorm else None
    if vecnorm_path is None:
        # try final first, else look for a milestone
        cand = os.path.join(run_dir, "vecnorm_final.pkl")
        if os.path.isfile(cand):
            vecnorm_path = cand
        else:
            # find latest milestone vecnorm
            picks = [p for p in os.listdir(run_dir) if p.startswith("vecnorm_step_") and p.endswith(".pkl")]
            if picks:
                # choose the max step
                picks.sort(key=lambda s: int(s.replace("vecnorm_step_", "").replace(".pkl","")))
                vecnorm_path = os.path.join(run_dir, picks[-1])

    if vecnorm_path and os.path.isfile(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    # Load model
    model = PPO.load(args.model, device=args.device)

    # Prepare video writer
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.output if args.output.endswith(".mp4") else args.output + ".mp4")
    writer = imageio.get_writer(out_path, fps=args.fps)

    # Rollout and record
    base_env = unwrap_base_env(env)
    steps_limit = args.steps if args.steps > 0 else args.seconds * args.fps
    steps = 0
    obs = env.reset()

    try:
        while steps_limit == 0 or steps < steps_limit:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(action)

            frame = base_env.render(mode="rgb_array")
            writer.append_data(frame)

            if args.render:
                # If your setup supports a live window, NES-Py will already handle it
                pass

            if hasattr(done, "any") and done.any():
                obs = env.reset()
            elif isinstance(done, bool) and done:
                obs = env.reset()

            steps += 1
    finally:
        writer.close()
        env.close()

    print(f"Saved video to: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model zip (e.g., ./train/ppo_s42/final_model.zip or model_step_5000000.zip)")
    ap.add_argument("--vecnorm", type=str, default=None, help="Optional path to VecNormalize stats .pkl (auto-detected if omitted)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--render", action="store_true", help="Show a live window if supported")
    # recording length: choose either --seconds or --steps
    ap.add_argument("--seconds", type=int, default=60, help="Video length in seconds (ignored if --steps > 0)")
    ap.add_argument("--steps", type=int, default=0, help="Number of env steps to record (overrides --seconds if > 0)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="videos")
    ap.add_argument("--output", type=str, default="mario_run")
    args = ap.parse_args()
    main(args)
