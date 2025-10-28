# mario_env.py
# Minimal Mario training env with action shortcuts tuned for pipe-to-pipe jumps.
# Uses a compact custom action set + two wrappers:
#  - VariableActionRepeat: hold "A" longer when jumping
#  - MacroLongJump: short run-up + long jump when the agent requests a jump
#
# Works with Gym 0.21 + gym-super-mario-bros 7.3.0 + nes-py, frame-stacked VecEnv.

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

# Compact action set designed for fine alignment and long jumps over two pipes
PIPE_ACTIONS = [
    ["NOOP"],            # 0
    ["right"],           # 1 - tiny forward tap (micro-adjust)
    ["right", "B"],      # 2 - run
    ["left"],            # 3 - tiny backward tap
    ["A"],               # 4 - jump in place (useful on top of a pipe)
    ["right", "A"],      # 5 - forward jump (no run)
    ["right", "A", "B"], # 6 - long jump (running)
    ["left", "A"],       # 7 - backward jump (rarely used but handy to recover)
]

# ---------- Wrappers ----------

class VariableActionRepeat(gym.Wrapper):
    """Repeat the action longer when it contains 'A' (hold jump). Short taps otherwise."""
    def __init__(self, env, actions_list, k_jump=10, k_other=2):
        super().__init__(env)
        self.actions_list = actions_list
        self.k_jump = int(k_jump)
        self.k_other = int(k_other)

    def step(self, action):
        act_spec = self.actions_list[action]
        k = self.k_jump if ("A" in act_spec) else self.k_other
        total_reward, done, info = 0.0, False, {}
        obs = None
        for _ in range(k):
            obs, r, done, info = self.env.step(action)
            total_reward += r
            if done:
                break
        return obs, total_reward, done, info


class MacroLongJump(gym.Wrapper):
    """
    If the agent requests a jump (any action containing 'A'):
      - If the previous action was a short forward tap ('right'), use a short run-up.
      - Otherwise, use a longer run-up.
    Then hold 'Right+A+B' for 'jump_hold' frames.
    """
    def __init__(self, env, actions_list, pre_run_long=10, pre_run_short=3, jump_hold=14):
        super().__init__(env)
        self.actions_list = actions_list
        self.pre_run_long = int(pre_run_long)
        self.pre_run_short = int(pre_run_short)
        self.jump_hold = int(jump_hold)

        self.idx_right    = self._find_index({"right"})
        self.idx_right_b  = self._find_index({"right", "B"})
        self.idx_right_ab = self._find_index({"right", "A", "B"})

        self._queue = []
        self._prev_action = None

    def _find_index(self, keys_set):
        ks = {k.lower() for k in keys_set}
        for i, combo in enumerate(self.actions_list):
            combo_set = {c.lower() for c in combo if c != "NOOP"}
            if ks.issubset(combo_set):
                return i
        return None

    def reset(self, **kwargs):
        self._queue.clear()
        self._prev_action = None
        return self.env.reset(**kwargs)

    def _enqueue_long_jump(self, use_short: bool):
        pre = self.pre_run_short if use_short else self.pre_run_long
        if self.idx_right_b is not None and pre > 0:
            self._queue.extend([self.idx_right_b] * pre)        # run-up
        if self.idx_right_ab is not None and self.jump_hold > 0:
            self._queue.extend([self.idx_right_ab] * self.jump_hold)  # long jump

    def step(self, action):
        # Continue any pending macro sequence
        if self._queue:
            a = self._queue.pop(0)
            obs, r, done, info = self.env.step(a)
            if not self._queue:
                self._prev_action = a
            return obs, r, done, info

        # Intercept jumps and convert them to a long-jump macro
        act_spec = self.actions_list[action]
        wants_jump = ("A" in act_spec)
        if wants_jump and self.idx_right_ab is not None:
            short = (self._prev_action == self.idx_right)  # short run-up if we just tapped right
            self._enqueue_long_jump(use_short=short)
            a = self._queue.pop(0)
            obs, r, done, info = self.env.step(a)
            return obs, r, done, info

        # Normal step
        obs, r, done, info = self.env.step(action)
        self._prev_action = action
        return obs, r, done, info


class RewardShaping(gym.Wrapper):
    """Mild reward shaping: +progress in x, +flag, -death."""
    def __init__(self, env, right_coef=0.02, death_penalty=-5.0, flag_reward=50.0):
        super().__init__(env)
        self.right_coef = float(right_coef)
        self.death_penalty = float(death_penalty)
        self.flag_reward = float(flag_reward)
        self._prev_x = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_x = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        x = info.get("x_pos", 0)
        dx = max(0, x - self._prev_x)
        shaped = reward + self.right_coef * dx
        if info.get("flag_get"):
            shaped += self.flag_reward
        if done and not info.get("flag_get"):
            shaped += self.death_penalty
        self._prev_x = x
        return obs, shaped, done, info


class StuckReset(gym.Wrapper):
    """Terminate episode when no forward progress for 'patience' steps (prevents long stalls)."""
    def __init__(self, env, patience=250):
        super().__init__(env)
        self.patience = int(patience)
        self._last_x = 0
        self._no_progress = 0

    def reset(self, **kwargs):
        self._last_x = 0
        self._no_progress = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        x = info.get("x_pos", 0)
        if x <= self._last_x + 1:
            self._no_progress += 1
        else:
            self._no_progress = 0
        self._last_x = x
        if self._no_progress >= self.patience:
            done = True
        return obs, reward, done, info


# ---------- Builders ----------

def _make_base_env(
    grayscale=True,
    resize=84,
    seed=None,
    level_id="SuperMarioBros-1-1-v0",
    pre_run_long=10,
    pre_run_short=3,
    jump_hold=14,
    k_jump=10,
    k_other=2,
):
    env = gym_super_mario_bros.make(level_id)
    actions = PIPE_ACTIONS
    env = JoypadSpace(env, actions)

    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    if resize:
        env = ResizeObservation(env, (int(resize), int(resize)))

    env = MacroLongJump(env, actions_list=actions,
                        pre_run_long=pre_run_long, pre_run_short=pre_run_short,
                        jump_hold=jump_hold)
    env = VariableActionRepeat(env, actions_list=actions,
                               k_jump=k_jump, k_other=k_other)

    env = RewardShaping(env, right_coef=0.02, death_penalty=-5.0, flag_reward=50.0)
    env = StuckReset(env, patience=250)
    env = Monitor(env)

    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_env(
    num_envs: int = 4,
    grayscale: bool = True,
    resize: int = 84,
    frame_stack: int = 4,
    level_id: str = "SuperMarioBros-1-1-v0",
    pre_run_long: int = 10,
    pre_run_short: int = 3,
    jump_hold: int = 14,
    k_jump: int = 10,
    k_other: int = 2,
):
    """Create a (vectorized) Mario environment ready for SB3 PPO."""
    def thunk(rank: int):
        return lambda: _make_base_env(
            grayscale=grayscale, resize=resize, seed=rank, level_id=level_id,
            pre_run_long=pre_run_long, pre_run_short=pre_run_short,
            jump_hold=jump_hold, k_jump=k_jump, k_other=k_other
        )

    if num_envs == 1:
        venv = DummyVecEnv([thunk(0)])
    else:
        venv = SubprocVecEnv([thunk(i) for i in range(num_envs)])

    venv = VecFrameStack(venv, frame_stack, channels_order="last")
    return venv
