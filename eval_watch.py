# eval_watch.py
import argparse
from stable_baselines3 import PPO
from mario_env import make_env

def get_base_env(env):
    # Peel wrappers until we can render the raw NES window
    v = env
    while hasattr(v, "venv"):
        v = v.venv
    try:
        return v.envs[0]
    except Exception:
        return None

def main(args):
    env = make_env(
        num_envs=1,
        grayscale=True,
        resize=args.resize,
        frame_stack=args.frame_stack,
        level_id=args.level_id,
        pre_run_long=args.pre_run_long,
        pre_run_short=args.pre_run_short,
        jump_hold=args.jump_hold,
        k_jump=args.k_jump,
        k_other=args.k_other,
        use_macros=args.use_macros,
        use_shaping=args.use_shaping,
        use_stuck=args.use_stuck,
        use_resize=args.use_resize,
        action_set=args.action_set,
        stuck_patience=args.stuck_patience,
        max_episode_steps=args.max_episode_steps,
        channels_order=args.channels_order,  # keep consistent with training
    )
    base_env = get_base_env(env)
    model = PPO.load(args.model, device=args.device)

    obs = env.reset()
    episodes = 0
    ep_ret = 0.0
    while episodes < args.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_ret += float(reward if not hasattr(reward, "__len__") else reward[0])

        if args.render and base_env is not None:
            base_env.render()

        finished = done.any() if hasattr(done, "any") else bool(done)
        if finished:
            print(f"[episode {episodes+1}] return={ep_ret:.1f}")
            episodes += 1
            ep_ret = 0.0
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--render", action="store_true")

    # Env shape/compat
    ap.add_argument("--channels_order", type=str, default="first", choices=["first", "last"])
    ap.add_argument("--level_id", type=str, default="SuperMarioBros-1-1-v0")
    ap.add_argument("--frame_stack", type=int, default=4)
    ap.add_argument("--resize", type=int, default=84)

    # Feature toggles
    ap.add_argument("--use_macros", action="store_true")
    ap.add_argument("--use_shaping", action="store_true")
    ap.add_argument("--use_stuck", action="store_true")
    ap.add_argument("--use_resize", action="store_true")
    ap.add_argument("--action_set", type=str, default="pipe", choices=["simple","right_only","pipe"])

    # Macro params
    ap.add_argument("--pre_run_long", type=int, default=10)
    ap.add_argument("--pre_run_short", type=int, default=5)
    ap.add_argument("--jump_hold", type=int, default=16)
    ap.add_argument("--k_jump", type=int, default=10)
    ap.add_argument("--k_other", type=int, default=2)

    # Episode control
    ap.add_argument("--stuck_patience", type=int, default=250)
    ap.add_argument("--max_episode_steps", type=int, default=3000)

    args = ap.parse_args()
    main(args)
