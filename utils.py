import gym
import numpy as np
import random
from collections import deque
from gym.wrappers import AtariPreprocessing, FrameStack
from wrappers import EpisodicLifeEnv
from pathlib import Path

ENV_LIST = {
    "Atari": [
        "VideoPinball-v5",
        "Breakout-v5",
        "Pong-v5",
        "Boxing-v5",
    ],
    "MuJoCo": [
        "Hopper-v4",
        "Humanoid-v4",
        "HalfCheetah-v4",
        "Ant-v4",
    ]
}

ALGO_LIST = {
    "Value-Based": ["DQN"],
    "Policy-Based": ["PPO", "DDPG"]
}


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (
            np.array(state, dtype=np.float32),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def size(self):
        return len(self.buffer)

def check_env_algo(env_name, algo_name):
    if env_name in ENV_LIST["Atari"] and algo_name in ALGO_LIST["Value-Based"]:
        return True
    elif env_name in ENV_LIST["MuJoCo"] and algo_name in ALGO_LIST["Policy-Based"]:
        return True
    else:
        return False

def make_env(env_name, eval=False):
    if env_name in ENV_LIST["Atari"]:
        if env_name == "VideoPinball-v5":
            env = gym.make("ALE/VideoPinball-v5", obs_type="ram", render_mode="rgb_array" if eval else None)
        else:
            env = gym.make(f"ALE/{env_name}", render_mode="rgb_array" if eval else None, frameskip=1)
            env = AtariPreprocessing(env, grayscale_obs=True, screen_size=84, noop_max=30, frame_skip=4)
            if not eval:
                env = EpisodicLifeEnv(env)
            env = FrameStack(env, num_stack=4)
    elif env_name in ENV_LIST["MuJoCo"]:
        env = gym.make(env_name, render_mode="rgb_array" if eval else None)
    return env

def setup_visualization(args, config, TIMESTAMP):
    vis_interval = config.get("vis_interval", 500)
    video_dir = Path(f"gif/train/{args.env_name}/{args.model_type}_{TIMESTAMP}")
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir, vis_interval