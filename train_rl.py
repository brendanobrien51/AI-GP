# This script trains a PPO reinforcement learning policy for the
# AirSim Drone Racing Lab environment defined in adrl_env.py.
# Scaled to 1M timesteps with EvalCallback + CheckpointCallback.
# Final model saved as adrl_ppo_racer_v2.zip

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

from adrl_env import ADRLRacingEnv

TOTAL_TIMESTEPS = 1_000_000

print("[TRAIN] Creating environment")
env = ADRLRacingEnv()
env = Monitor(env)

print("[TRAIN] Running random warmup so you can verify motion")
obs, _ = env.reset()

for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"[WARMUP {i}] reward={reward:.3f} info={info}")
    if terminated or truncated:
        obs, _ = env.reset()

print("[TRAIN] Building PPO model")
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log="./tb_logs/",
)

checkpoint_cb = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="adrl_ppo_v2",
)

eval_cb = EvalCallback(
    env,
    best_model_save_path="./best_model/",
    log_path="./eval_logs/",
    eval_freq=50_000,
    n_eval_episodes=3,
    deterministic=True,
    render=False,
)

callbacks = CallbackList([checkpoint_cb, eval_cb])

print(f"[TRAIN] Starting learning ({TOTAL_TIMESTEPS:,} timesteps)")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

print("[TRAIN] Saving final model as adrl_ppo_racer_v2")
model.save("adrl_ppo_racer_v2")

env.close()
print("[TRAIN] Done")
