# test_random.py
import argparse
import numpy as np
from mario_env import make_env

def main(render: bool, steps: int):
    # VecEnv con 1 entorno
    env = make_env(num_envs=1)
    base_env = getattr(env, "envs", [env])[0]  # para render()

    obs = env.reset()
    for t in range(steps):
        # acción como lista de largo = num_envs
        action = [env.action_space.sample()]
        obs, reward, done, info = env.step(action)

        if render:
            base_env.render()

        if hasattr(done, "any") and done.any():
            obs = env.reset()

        if t % 500 == 0:
            r = float(np.asarray(reward).mean())
            x = info[0].get("x_pos", 0) if isinstance(info, (list, tuple)) else info.get("x_pos", 0)
            print(f"[{t}] reward={r:.3f} x={x}")

    env.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="Muestra la ventana si WSLg/X está disponible")
    ap.add_argument("--steps", type=int, default=5000)
    args = ap.parse_args()
    main(args.render, args.steps)
