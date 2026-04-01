# This script is a quick sanity test for the custom AirSim Drone Racing Lab
# reinforcement learning environment in adrl_env.py.
# It creates the environment, resets it, then steps through it using random
# actions from the action space while printing rewards and info values.
# If an episode ends, it automatically resets and keeps going.
# This is useful for checking that the environment loads correctly, the drone
# moves, rewards are being produced, and the reset/step loop works before
# starting full RL training.

from adrl_env import ADRLRacingEnv

env = ADRLRacingEnv()
obs, _ = env.reset()

for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"TEST STEP {i}: reward={reward:.3f}, info={info}")
    if terminated or truncated:
        obs, _ = env.reset()

env.close()


