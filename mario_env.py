# mario_env.py
# Mario + SB3: vectorized env with TimeLimit + VecMonitor + (optional) anti-stall.
# Requires: gym==0.21.0, gym-super-mario-bros==7.3.0, stable-baselines3

from typing import Optional
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation, TimeLimit
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecTransposeImage
)
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT

PIPE_ACTIONS = [
    ["NOOP"], ["right"], ["right", "B"], ["left"],
    ["A"], ["right", "A"], ["right", "A", "B"], ["left", "A"],
]

ACTION_SETS = {
    "right_only": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "pipe": PIPE_ACTIONS,
}

class VariableActionRepeat(gym.Wrapper):
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
            self._queue.extend([self.idx_right_b] * pre)
        if self.idx_right_ab is not None and self.jump_hold > 0:
            self._queue.extend([self.idx_right_ab] * self.jump_hold)
    def step(self, action):
        if self._queue:
            a = self._queue.pop(0)
            obs, r, done, info = self.env.step(a)
            if not self._queue:
                self._prev_action = a
            return obs, r, done, info
        act_spec = self.actions_list[action]
        wants_jump = ("A" in act_spec)
        if wants_jump and self.idx_right_ab is not None:
            short = (self._prev_action == self.idx_right)
            self._enqueue_long_jump(use_short=short)
            a = self._queue.pop(0)
            obs, r, done, info = self.env.step(a)
            return obs, r, done, info
        obs, r, done, info = self.env.step(action)
        self._prev_action = action
        return obs, r, done, info

class RewardShaping(gym.Wrapper):
    def __init__(self, env, right_coef=0.02, death_penalty=-5.0, flag_reward=50.0):
        super().__init__(env)
        self.right_coef = float(right_coef)
        self.death_penalty = float(death_penalty)
        self.flag_reward = float(flag_reward)
        self._prev_x = 0
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs); self._prev_x = 0; return obs
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
    def __init__(self, env, patience=250):
        super().__init__(env)
        self.patience = int(patience); self._last_x = 0; self._no_progress = 0
    def reset(self, **kwargs):
        self._last_x = 0; self._no_progress = 0; return self.env.reset(**kwargs)
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        x = info.get("x_pos", 0)
        if x <= self._last_x + 1: self._no_progress += 1
        else: self._no_progress = 0
        self._last_x = x
        if self._no_progress >= self.patience: done = True
        return obs, reward, done, info

def _make_base_env(
    grayscale: bool = True,
    use_resize: bool = True,
    resize: int = 84,
    seed: Optional[int] = None,
    level_id: str = "SuperMarioBros-1-1-v0",
    use_macros: bool = False,
    use_shaping: bool = False,
    use_stuck: bool = True,
    action_set: str = "right_only",
    pre_run_long: int = 10,
    pre_run_short: int = 3,
    jump_hold: int = 14,
    k_jump: int = 10,
    k_other: int = 2,
    stuck_patience: int = 250,
    max_episode_steps: int = 3000,
):
    env = gym_super_mario_bros.make(level_id)
    actions = ACTION_SETS.get(action_set, RIGHT_ONLY)
    env = JoypadSpace(env, actions)
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)  # (H,W,1)
    if use_resize and resize:
        env = ResizeObservation(env, (int(resize), int(resize)))  # (84,84,1)
    if use_macros:
        env = MacroLongJump(env, actions_list=actions,
                            pre_run_long=pre_run_long, pre_run_short=pre_run_short,
                            jump_hold=jump_hold)
        env = VariableActionRepeat(env, actions_list=actions, k_jump=k_jump, k_other=k_other)
    if use_shaping:
        env = RewardShaping(env, right_coef=0.02, death_penalty=-5.0, flag_reward=50.0)
    if use_stuck:
        env = StuckReset(env, patience=stuck_patience)
    env = TimeLimit(env, max_episode_steps=int(max_episode_steps))
    if seed is not None:
        env.seed(seed); env.action_space.seed(seed); env.observation_space.seed(seed)
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
    use_macros: bool = False,
    use_shaping: bool = False,
    use_stuck: bool = False,
    use_resize: bool = False,
    action_set: str = "pipe",
    stuck_patience: int = 250,
    max_episode_steps: Optional[int] = 3000,
    # stack LAST then transpose to CHW for SB3
    channels_order: str = "last",
    transpose_to_chw: bool = True,
):
    def thunk(rank: int):
        return lambda: _make_base_env(
            grayscale=grayscale, use_resize=use_resize, resize=resize, seed=rank, level_id=level_id,
            use_macros=use_macros, use_shaping=use_shaping, use_stuck=use_stuck, action_set=action_set,
            pre_run_long=pre_run_long, pre_run_short=pre_run_short, jump_hold=jump_hold,
            k_jump=k_jump, k_other=k_other, stuck_patience=stuck_patience,
            max_episode_steps=(max_episode_steps if max_episode_steps is not None else 3000),
        )

    venv = DummyVecEnv([thunk(0)]) if num_envs == 1 else SubprocVecEnv([thunk(i) for i in range(num_envs)])
    venv = VecFrameStack(venv, frame_stack, channels_order=channels_order)  # (H,W,4)
    if transpose_to_chw:
        venv = VecTransposeImage(venv)  # -> (4,H,W)
    venv = VecMonitor(venv)
    return venv
