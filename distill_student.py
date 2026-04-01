"""
Teacher → Student Distillation
================================
Stage 2+3 of teacher-student bootstrap RL.

Step 1 — Behavior Cloning:
  Run teacher policy in AirSim, record (student_obs, teacher_action) pairs.
  Train student MLP via MSE loss to mimic teacher actions.
  Saves: adrl_student_bc.zip

Step 2 — RL Fine-tuning:
  Initialize student PPO from behavior-cloned weights.
  Fine-tune with RL for 500k more steps.
  Saves: adrl_ppo_racer_v2.zip

Based on:
  "Bootstrapping RL with Imitation for Vision-Based Agile Flight" (Xing et al.)
  "Student-Informed Teacher Training" (ICLR 2025)

Run: python distill_student.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from adrl_env import ADRLRacingEnv
from adrl_teacher_env import ADRLTeacherEnv

COLLECT_TRANSITIONS = 10_000
BC_EPOCHS           = 50
BC_BATCH_SIZE       = 256
BC_LR               = 1e-3
FINETUNE_STEPS      = 500_000


# ---------------------------------------------------------------------------
# Step 1 — Collect teacher rollouts
# ---------------------------------------------------------------------------

def collect_teacher_rollouts(n_transitions: int):
    """
    Run teacher in AirSim, record (student_obs, teacher_action) pairs.
    Teacher uses the 22D privileged env, but we record the 18D student obs
    by calling the student env's _get_obs() in parallel.
    """
    print(f"\n[DISTILL] Collecting {n_transitions:,} teacher transitions...")

    teacher_env = ADRLTeacherEnv()
    student_env = ADRLRacingEnv()   # same AirSim client — shares sim state

    # Both envs share the same AirSim client by connecting independently.
    # We drive teacher_env and extract obs from student_env._get_obs()
    # at each step (both read from the same simulation state).

    teacher_model = PPO.load("adrl_teacher", env=teacher_env)

    student_obs_list = []
    teacher_action_list = []

    teacher_obs, _ = teacher_env.reset()
    # Sync student env state
    student_env.next_gate_idx = teacher_env.next_gate_idx
    student_env.prev_gate_dist = teacher_env.prev_gate_dist

    collected = 0
    while collected < n_transitions:
        teacher_action, _ = teacher_model.predict(teacher_obs, deterministic=True)

        # Record student's view of current state
        s_obs = student_env._get_obs()
        student_obs_list.append(s_obs)
        teacher_action_list.append(teacher_action.copy())

        teacher_obs, reward, terminated, truncated, info = teacher_env.step(teacher_action)
        # Keep student env in sync
        student_env.next_gate_idx = teacher_env.next_gate_idx
        student_env.prev_gate_dist = teacher_env.prev_gate_dist

        collected += 1
        if collected % 1000 == 0:
            print(f"  Collected {collected}/{n_transitions}")

        if terminated or truncated:
            teacher_obs, _ = teacher_env.reset()
            student_env.next_gate_idx = teacher_env.next_gate_idx
            student_env.prev_gate_dist = teacher_env.prev_gate_dist

    teacher_env.close()
    student_env.close()

    student_obs = np.array(student_obs_list, dtype=np.float32)
    teacher_acts = np.array(teacher_action_list, dtype=np.float32)
    print(f"[DISTILL] Collected {len(student_obs)} transitions")
    return student_obs, teacher_acts


# ---------------------------------------------------------------------------
# Step 2 — Behavior cloning
# ---------------------------------------------------------------------------

def behavior_clone(student_obs: np.ndarray, teacher_actions: np.ndarray,
                   student_model: PPO) -> PPO:
    """Train student policy network to mimic teacher actions via MSE."""
    print(f"\n[DISTILL] Behavior cloning for {BC_EPOCHS} epochs...")

    policy_net = student_model.policy
    policy_net.train()

    optimizer = optim.Adam(policy_net.parameters(), lr=BC_LR)
    loss_fn   = nn.MSELoss()

    obs_t  = torch.tensor(student_obs,    dtype=torch.float32)
    acts_t = torch.tensor(teacher_actions, dtype=torch.float32)

    n = len(obs_t)
    for epoch in range(BC_EPOCHS):
        # Shuffle
        perm = torch.randperm(n)
        obs_t  = obs_t[perm]
        acts_t = acts_t[perm]

        total_loss = 0.0
        batches = 0
        for i in range(0, n, BC_BATCH_SIZE):
            obs_batch  = obs_t[i : i + BC_BATCH_SIZE]
            acts_batch = acts_t[i : i + BC_BATCH_SIZE]

            # Forward through policy: get action distribution mean
            with torch.no_grad():
                features = policy_net.extract_features(obs_batch,
                                                       policy_net.features_extractor)
            latent_pi, _ = policy_net.mlp_extractor(features)
            pred_actions  = policy_net.action_net(latent_pi)

            loss = loss_fn(pred_actions, acts_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches    += 1

        avg_loss = total_loss / max(batches, 1)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3}/{BC_EPOCHS}  loss={avg_loss:.5f}")

    print("[DISTILL] Behavior cloning complete")
    return student_model


# ---------------------------------------------------------------------------
# Step 3 — RL fine-tuning
# ---------------------------------------------------------------------------

def finetune_student(student_model: PPO):
    print(f"\n[DISTILL] RL fine-tuning for {FINETUNE_STEPS:,} timesteps...")

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/",
        name_prefix="adrl_student_ft",
    )

    eval_env = ADRLRacingEnv()
    eval_cb  = EvalCallback(
        Monitor(eval_env),
        best_model_save_path="./best_model/student/",
        log_path="./eval_logs/student/",
        eval_freq=50_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb])
    student_model.learn(
        total_timesteps=FINETUNE_STEPS,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    student_model.save("adrl_ppo_racer_v2")
    print("[DISTILL] Saved fine-tuned student as adrl_ppo_racer_v2")
    eval_env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Stage 1: collect teacher rollouts
    student_obs, teacher_acts = collect_teacher_rollouts(COLLECT_TRANSITIONS)

    # Stage 2: init student env + behavior clone
    student_env   = Monitor(ADRLRacingEnv())
    student_model = PPO(
        "MlpPolicy",
        student_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,     # lower entropy for fine-tuning
        tensorboard_log="./tb_logs/",
        device="cpu",
    )

    student_model = behavior_clone(student_obs, teacher_acts, student_model)
    student_model.save("adrl_student_bc")
    print("[DISTILL] Saved behavior-cloned student as adrl_student_bc")

    # Stage 3: RL fine-tune
    finetune_student(student_model)

    student_env.close()
    print("\n[DISTILL] All stages complete!")
    print("  adrl_student_bc.zip  — behavior cloned weights")
    print("  adrl_ppo_racer_v2.zip — final fine-tuned policy")
