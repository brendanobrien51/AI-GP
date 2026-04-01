# run_policy.py
# This script loads a previously trained PPO model and runs it in the
# AirSim Drone Racing Lab environment defined in adrl_env.py.
# It continuously resets the environment, asks the trained policy for the
# next action, applies that action, and restarts whenever an episode ends.
# This is used to test or demonstrate a saved RL policy after training.

from stable_baselines3 import PPO
from adrl_env import ADRLRacingEnv

env = ADRLRacingEnv()
model = PPO.load("adrl_ppo_racer")

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()


