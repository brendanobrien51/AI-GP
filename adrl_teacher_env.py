"""
Teacher Environment — Privileged Full-State Observation
=========================================================
Subclass of ADRLRacingEnv with a richer 22D observation that includes
exact drone position, velocity, and relative vectors to the next 3 gates.

No visual features, no VIO noise — the teacher sees everything perfectly.
This makes learning much easier, and the trained teacher policy is later
distilled into the student (18D YOLO-based) policy via behavior cloning.

Based on:
  "Bootstrapping RL with Imitation for Vision-Based Agile Flight" (Xing et al.)
  "Student-Informed Teacher Training" (ICLR 2025)

Usage:
    env = ADRLTeacherEnv()
    model = PPO("MlpPolicy", env, ...)
    model.learn(total_timesteps=500_000)
    model.save("adrl_teacher")
"""

import math
import numpy as np
from gymnasium import spaces

from adrl_env import ADRLRacingEnv, norm, vec3


class ADRLTeacherEnv(ADRLRacingEnv):
    """
    Privileged teacher environment.

    Observation (22D):
        pos(3)          — exact world position (NED, m)
        vel(3)          — exact velocity (m/s)
        yaw sin/cos(2)  — heading
        rel_gate1(3)    — vector to next gate
        rel_gate2(3)    — vector to gate+1
        rel_gate3(3)    — vector to gate+2
        dist1/2/3(3)    — distances to next 3 gates
        gate_found(1)   — YOLO detection (0/1)
        speed(1)        — current speed magnitude

    Action space: unchanged (4D roll/pitch/yaw_rate/z_offset)

    Reward: adds time penalty (-0.002/step) to encourage fast lap completion.
    """

    def __init__(self):
        super().__init__()

        # Override observation space to 22D
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),
            dtype=np.float32,
        )
        print("[TEACHER] Using privileged 22D observation (full state)")

    def _get_obs(self):
        pos, vel, yaw = self._get_state()

        idx = self.next_gate_idx
        n   = len(self.gates)

        g1 = self.gates[min(idx,     n - 1)]
        g2 = self.gates[min(idx + 1, n - 1)]
        g3 = self.gates[min(idx + 2, n - 1)]

        rel1 = g1 - pos
        rel2 = g2 - pos
        rel3 = g3 - pos

        # YOLO gate detection (binary: 1 if detected)
        cv = self._extract_cv_features()
        gate_found = float(cv[0])

        speed = float(np.linalg.norm(vel))

        obs = np.concatenate([
            pos.astype(np.float32),
            vel.astype(np.float32),
            np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32),
            rel1.astype(np.float32),
            rel2.astype(np.float32),
            rel3.astype(np.float32),
            np.array([norm(rel1), norm(rel2), norm(rel3)], dtype=np.float32),
            np.array([gate_found, speed], dtype=np.float32),
        ]).astype(np.float32)

        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Teacher-specific: time penalty (time-optimal objective)
        reward -= 0.002

        # Re-build 22D obs (parent returned 18D from _get_obs after step)
        obs = self._get_obs()

        return obs, reward, terminated, truncated, info
