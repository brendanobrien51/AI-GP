"""
Teacher Policy Training
=======================
Stage 1 of teacher-student bootstrap RL.

Trains a PPO policy on the privileged 22D observation (full state, no vision noise).
The teacher learns quickly because it has perfect information.
Its policy is later distilled into the vision-based student (distill_student.py).

Run: python train_teacher.py
Output: adrl_teacher.zip
"""

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from adrl_teacher_env import ADRLTeacherEnv

TOTAL_TIMESTEPS = 500_000

print("[TEACHER] Creating privileged environment")
env = ADRLTeacherEnv()
env = Monitor(env)

print("[TEACHER] Building PPO model")
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
    device="cpu",   # MLP policy trains faster on CPU
)

checkpoint_cb = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="adrl_teacher",
)

eval_cb = EvalCallback(
    env,
    best_model_save_path="./best_model/teacher/",
    log_path="./eval_logs/teacher/",
    eval_freq=50_000,
    n_eval_episodes=3,
    deterministic=True,
    render=False,
)

callbacks = CallbackList([checkpoint_cb, eval_cb])

print(f"[TEACHER] Starting training ({TOTAL_TIMESTEPS:,} timesteps)")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

print("[TEACHER] Saving teacher policy as adrl_teacher")
model.save("adrl_teacher")

env.close()
print("[TEACHER] Done — run distill_student.py next")
