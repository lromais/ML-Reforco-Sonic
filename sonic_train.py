import stable_retro as retro
import gymnasium as gym
import numpy as np
import cv2
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from gymnasium.wrappers import ResizeObservation
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation

# ============================================================
# FRAME SKIP
# ============================================================

class SkipFrame(gym.Wrapper):

    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):

        total_reward = 0

        for _ in range(self.skip):

            obs, reward, terminated, truncated, info = self.env.step(action)

            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info

# ============================================================
# CUSTOM REWARD
# ============================================================

class SonicReward(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.max_x = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.max_x = 0
        return obs, info

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        custom_reward = reward

        x = info.get("x", 0)

        if x > self.max_x:
            custom_reward += 1.0
            self.max_x = x
        else:
            custom_reward -= 0.01

        custom_reward += 0.001

        return obs, custom_reward, terminated, truncated, info

# ============================================================
# FIX OBSERVATION SHAPE
# ============================================================

class TransposeObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(obs_shape[0], obs_shape[1], obs_shape[2]),
            dtype=np.uint8
        )

    def observation(self, obs):
        return np.array(obs).squeeze(-1) if np.array(obs).ndim == 4 else np.array(obs)

# ============================================================
# CRIAR AMBIENTE (sem render)
# ============================================================

def make_env():

    env = retro.make(
        game='SonicTheHedgehog-Genesis-v0',
        state='GreenHillZone.Act1',
        render_mode=None
    )

    env = SkipFrame(env, skip=4)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = SonicReward(env)
    env = FrameStackObservation(env, 4)
    env = TransposeObservation(env)
    env = Monitor(env)

    return env

# ============================================================
# VECTOR ENV
# ============================================================

env = DummyVecEnv([make_env])

# ============================================================
# PPO
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando device: {device}")

model = PPO(
    "CnnPolicy",
    env,

    learning_rate=0.00025,

    n_steps=2048,

    batch_size=64,

    n_epochs=10,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.01,

    verbose=1,

    tensorboard_log="./logs/",

    device=device
)

# ============================================================
# TREINO
# ============================================================

model.learn(
    total_timesteps=1_000_000,
    progress_bar=True
)

# ============================================================
# SALVAR
# ============================================================

model.save("sonic_ppo")

print("Modelo salvo!")
