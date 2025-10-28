# record_video.py
import argparse, os, time
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage

from mario_env import make_env

def main(args):
    model = PPO.load(args.model, device=args.device)
    obs_shape = tuple(model.observation_space.shape)  # modelo dice qué espera

    # ¿espera CHW (4,H,W) o HWC (H,W,4)?
    expects_chw = (len(obs_shape) == 3 and obs_shape[0] in (1,3,4))

    # Forzamos construir el env en HWC (channels_order="last"); luego transponemos si hace falta
    use_resize = False
    if obs_shape[0:2] == (84, 84) or obs_shape[-2:] == (84, 84):
        use_resize = True  # si entrenaste con 84x84

    env = make_env(
        num_envs=1,
        grayscale=True,
        use_resize=use_resize,
        resize=84,
        frame_stack=args.frame_stack,
        level_id=args.level_id,
        use_macros=args.use_macros,
        use_shaping=args.use_shaping,
        use_stuck=args.use_stuck,
        action_set=args.action_set,
        channels_order="last",   # SIEMPRE HWC aquí
        stuck_patience=args.stuck_patience,
        max_episode_steps=args.max_episode_steps,
    )

    if expects_chw:
        env = VecTransposeImage(env)  # HWC -> CHW para hacer match con el modelo
        env_shape = tuple(env.observation_space.shape)
        if env_shape != obs_shape:
            # si todavía no matchea, intentamos quitar resize (nativo 240x256)
            env.close()
            env = make_env(
                num_envs=1, grayscale=True, use_resize=False, resize=84,
                frame_stack=args.frame_stack, level_id=args.level_id,
                use_macros=args.use_macros, use_shaping=args.use_shaping, use_stuck=args.use_stuck,
                action_set=args.action_set, channels_order="last",
                stuck_patience=args.stuck_patience, max_episode_steps=args.max_episode_steps,
            )
            env = VecTransposeImage(env)
            env_shape = tuple(env.observation_space.shape)
            if env_shape != obs_shape:
                raise RuntimeError(f"Env {env_shape} != model {obs_shape}. Revisa si ese checkpoint fue entrenado con resize.")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.prefix or 'mario'}_{int(time.time())}.mp4")
    writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264")

    obs = env.reset()
    steps = 0
    while steps < args.video_length:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frame = env.render(mode="rgb_array")
        if isinstance(frame, list):
            frame = frame[0]
        if frame is None:
            raise RuntimeError("render('rgb_array') returned None.")
        writer.append_data(frame)
        steps += 1
        if done.any() if hasattr(done, "any") else bool(done):
            obs = env.reset()

    writer.close(); env.close()
    print(f"Saved video: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out_dir", type=str, default="videos")
    ap.add_argument("--prefix", type=str, default="run")
    ap.add_argument("--video_length", type=int, default=6000)
    ap.add_argument("--fps", type=int, default=30)

    # Env flags (coherentes con entrenamiento)
    ap.add_argument("--level_id", type=str, default="SuperMarioBros-1-1-v0")
    ap.add_argument("--frame_stack", type=int, default=4)
    ap.add_argument("--use_macros", action="store_true")
    ap.add_argument("--use_shaping", action="store_true")
    ap.add_argument("--use_stuck", action="store_true")
    ap.add_argument("--action_set", type=str, default="pipe", choices=["simple","right_only","pipe"])
    ap.add_argument("--stuck_patience", type=int, default=200)
    ap.add_argument("--max_episode_steps", type=int, default=3000)

    args = ap.parse_args()
    main(args)
