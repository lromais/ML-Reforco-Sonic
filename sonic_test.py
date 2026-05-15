import stable_retro as retro
import gymnasium as gym
import numpy as np
import cv2
import torch

from stable_baselines3 import PPO
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
        obs = np.array(obs)
        return obs.squeeze(-1) if obs.ndim == 4 else obs

# ============================================================
# CRIAR AMBIENTE
# ============================================================

env = retro.make(
    game='SonicTheHedgehog-Genesis-v0',
    state=retro.State.NONE,
    render_mode='rgb_array'
)
env = SkipFrame(env, skip=4)
env = GrayscaleObservation(env)
env = ResizeObservation(env, (84, 84))
env = SonicReward(env)
env = FrameStackObservation(env, 4)
env = TransposeObservation(env)

# ============================================================
# CARREGAR MODELO
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando device: {device}")

model = PPO.load("sonic_ppo", device=device)

print("Modelo carregado! Iniciando teste... Pressione Q para sair.")

# ============================================================
# TESTE COM VISUALIZAÇÃO
# ============================================================

obs, _ = env.reset()

# pula a tela de intro
for _ in range(120):
    obs, _, _, _, _ = env.step(env.action_space.sample())

while True:

    action, _ = model.predict(obs[np.newaxis, ...])
    action = action[0]

    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()

    if frame is not None:
        print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")

    if frame is not None:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (640, 480))
        cv2.imshow('Sonic RL', frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if terminated or truncated:
        obs, _ = env.reset()

cv2.destroyAllWindows()
env.close()
