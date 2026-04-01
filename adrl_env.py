# This file defines the custom Gymnasium reinforcement learning environment
# used for training autonomous racing policies in AirSim Drone Racing Lab.
# It connects to the simulator, loads gate positions, builds a consistent
# starting pose near the first gate, and exposes a PPO-friendly action space
# and observation space for learning.
# Observations combine drone state, relative gate positions, and simple
# computer-vision features from the forward camera, while rewards encourage
# progress toward the correct gate, forward motion, visual gate tracking,
# and completing the course without collisions.
# This is the core environment file used by train_rl.py and run_policy.py.

import math
import random
import re
import time

import airsimdroneracinglab as airsim
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gate_detector import GateDetector

_HARD_BUF_MAX = 100
_HARD_BUF_PROB = 0.30


TAKEOFF_HEIGHT = -1.5
EPISODE_DT = 0.10
MAX_STEPS = 1500

GATE_HIT_RADIUS = 1.5
START_OFFSET = 6.0

CANNY_LOW = 60
CANNY_HIGH = 140
MIN_CONTOUR_AREA = 300


def vec3(x, y, z):
    return np.array([float(x), float(y), float(z)], dtype=np.float32)


def norm(v):
    return float(np.linalg.norm(v))


def unit(v):
    n = norm(v)
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return math.atan2(siny_cosp, cosy_cosp)


class ADRLRacingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.client = airsim.MultirotorClient()
        self.client.race_tier = None
        self.client.level_name = ""
        self.client.confirmConnection()

        self.dt = EPISODE_DT
        self.max_steps = MAX_STEPS
        self.base_z = TAKEOFF_HEIGHT
        self.step_count = 0
        self.episode_idx = 0

        self.gates = self._load_gate_positions()
        self.start_pos, self.start_yaw = self._build_start_pose()

        self.next_gate_idx = 0
        self.prev_gate_dist = None
        self._gates_base = None   # original gate positions before per-episode jitter

        self._detector = GateDetector("gate_detector.pt")
        self._hard_state_buffer = []  # contrastive initial state buffer

        # Normalized PPO-friendly action space in [-1, 1]
        # [roll, pitch, yaw_rate, z_offset]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # vel(3), yaw sin/cos(2), rel next gate(3), rel gate after(3),
        # dist next/dist after(2), cv features(5)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32,
        )

        print(f"[INIT] Loaded {len(self.gates)} gates")
        print(
            f"[INIT] Start pose: ({self.start_pos[0]:.2f}, {self.start_pos[1]:.2f}, {self.start_pos[2]:.2f}) "
            f"yaw={math.degrees(self.start_yaw):.2f}"
        )

    def _get_object_pose_safe(self, name):
        if not hasattr(self.client, "race_tier"):
            self.client.race_tier = None
        if not hasattr(self.client, "level_name"):
            self.client.level_name = ""

        try:
            return self.client.simGetObjectPose(name)
        except Exception:
            internal = getattr(self.client, "_VehicleClient__internalGetObjectPose", None)
            if internal is not None:
                return internal(name)
            raise

    def _load_gate_positions(self):
        names = self.client.simListSceneObjects(".*[Gg]ate.*")
        names = [n for n in names if isinstance(n, str) and n.strip()]

        def gate_sort_key(name):
            nums = re.findall(r"\d+", name)
            return [int(n) for n in nums] if nums else [10**9]

        names.sort(key=gate_sort_key)

        gates = []
        for name in names:
            pose = self._get_object_pose_safe(name)
            if pose is None:
                print(f"[INIT] Gate {name}: SKIPPED (no pose)")
                continue
            p = pose.position
            x, y, z = p.x_val, p.y_val, p.z_val
            # Skip gates with NaN positions
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                print(f"[INIT] Gate {name}: SKIPPED (NaN position)")
                continue
            gates.append(vec3(x, y, z))
            print(f"[INIT] Gate {name}: ({x:.2f}, {y:.2f}, {z:.2f})")
        self._gates_base = [g.copy() for g in gates]
        return gates

    def _build_start_pose(self):
        gate0 = self.gates[0]
        if len(self.gates) > 1:
            course_dir = unit(self.gates[1] - self.gates[0])
        else:
            course_dir = vec3(1.0, 0.0, 0.0)

        start_pos = gate0 - course_dir * START_OFFSET
        start_pos[2] = gate0[2]
        start_yaw = math.atan2(course_dir[1], course_dir[0])
        return start_pos, start_yaw

    def _get_state(self):
        state = self.client.getMultirotorState()
        kin = state.kinematics_estimated

        pos = vec3(
            kin.position.x_val,
            kin.position.y_val,
            kin.position.z_val,
        )
        vel = vec3(
            kin.linear_velocity.x_val,
            kin.linear_velocity.y_val,
            kin.linear_velocity.z_val,
        )
        yaw = quat_to_yaw(kin.orientation)
        return pos, vel, yaw

    def _extract_cv_features(self):
        responses = self.client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
        )

        if not responses or responses[0].height == 0:
            return np.zeros(5, dtype=np.float32)

        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        frame = img1d.reshape(responses[0].height, responses[0].width, 3).copy()
        h, w = frame.shape[:2]

        # --- YOLO detection (fast path) ---
        if self._detector.available:
            det = self._detector.detect(frame)
            if det is not None:
                cx, cy = det["centroid_px"]
                return np.array([
                    1.0,
                    (cx - w / 2.0) / max(w / 2.0, 1.0),
                    (cy - h / 2.0) / max(h / 2.0, 1.0),
                    float(det["area_frac"]),
                    float(det["score"]),
                ], dtype=np.float32)
            return np.zeros(5, dtype=np.float32)

        # --- Contour fallback (when YOLO model not available) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score = -1.0
        best = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue

            rect = cv2.minAreaRect(contour)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area < 1.0:
                continue

            rectangularity = area / rect_area
            x, y, bw, bh = cv2.boundingRect(contour)
            cx = x + bw / 2.0
            cy = y + bh / 2.0

            score = rectangularity * area
            if score > best_score:
                best_score = score
                best = (cx, cy, area, rectangularity)

        if best is None:
            return np.zeros(5, dtype=np.float32)

        cx, cy, area, rectangularity = best
        return np.array(
            [
                1.0,
                (cx - w / 2.0) / max(w / 2.0, 1.0),
                (cy - h / 2.0) / max(h / 2.0, 1.0),
                area / float(w * h),
                rectangularity,
            ],
            dtype=np.float32,
        )

    def _get_obs(self):
        pos, vel, yaw = self._get_state()

        next_gate = self.gates[min(self.next_gate_idx, len(self.gates) - 1)]
        next2_gate = self.gates[min(self.next_gate_idx + 1, len(self.gates) - 1)]

        rel1 = next_gate - pos
        rel2 = next2_gate - pos
        cv_feats = self._extract_cv_features()

        obs = np.concatenate(
            [
                vel,
                np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32),
                rel1,
                rel2,
                np.array([norm(rel1), norm(rel2)], dtype=np.float32),
                cv_feats,
            ]
        ).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_idx += 1
        print(f"\n[RESET] Episode {self.episode_idx} starting")

        self.client.reset()
        time.sleep(0.5)

        self.client.enableApiControl()
        self.client.arm()
        print("[RESET] Armed")
        time.sleep(0.2)

        # --- Domain randomization: gate position jitter ±0.3m (from base, not accumulated) ---
        if self._gates_base is not None:
            self.gates = [
                g + np.random.uniform(-0.3, 0.3, size=3).astype(np.float32) * np.array([1, 1, 0])
                for g in self._gates_base
            ]

        # --- Domain randomization: wind disturbance ±2 m/s ---
        try:
            wx = float(np.random.uniform(-2.0, 2.0))
            wy = float(np.random.uniform(-2.0, 2.0))
            self.client.simSetWind(airsim.Vector3r(wx, wy, 0.0))
        except Exception:
            pass  # simSetWind not available in all AirSim versions

        # --- Domain randomization: start position jitter ±1m in x/y, ±0.2m altitude ---
        jitter = np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
        jitter[2] = float(np.random.uniform(-0.2, 0.2))
        spawn_pos = self.start_pos + jitter

        # --- Contrastive Initial State Buffer: 30% chance to start from a hard state ---
        use_hard = (
            len(self._hard_state_buffer) > 10
            and np.random.rand() < _HARD_BUF_PROB
        )
        if use_hard:
            hard = random.choice(self._hard_state_buffer)
            spawn_pos = hard["pos"].copy()
            hard_gate_idx = hard["gate_idx"]
        else:
            hard_gate_idx = 0

        try:
            self.client.moveToPositionAsync(
                float(spawn_pos[0]),
                float(spawn_pos[1]),
                float(spawn_pos[2]),
                2.0,
            ).join()
            print(
                f"[RESET] Moved to {'hard' if use_hard else 'start'} pose "
                f"({spawn_pos[0]:.2f}, {spawn_pos[1]:.2f}, {spawn_pos[2]:.2f})"
            )
        except Exception as exc:
            print(f"[RESET] moveToPositionAsync exception: {exc}")
            raise

        time.sleep(0.5)

        self.step_count = 0
        self.next_gate_idx = hard_gate_idx

        pos, _, _ = self._get_state()
        gate_for_dist = self.gates[min(self.next_gate_idx, len(self.gates) - 1)]
        self.prev_gate_dist = norm(gate_for_dist - pos)
        print(f"[RESET] Distance to gate {self.next_gate_idx}: {self.prev_gate_dist:.2f}")

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        roll_n, pitch_n, yaw_n, z_n = action.astype(np.float32)

        roll_cmd = float(np.clip(roll_n * 0.10, -0.20, 0.20))
        pitch_cmd = float(np.clip(0.18 + pitch_n * 0.10, -0.20, 0.25))
        yaw_rate_deg = float(np.clip(yaw_n * 20.0, -30.0, 30.0))
        z_offset = float(np.clip(z_n * 0.30, -0.40, 0.40))

        pos, vel, yaw = self._get_state()
        target_z = float(pos[2] + z_offset)
        yaw_rate_rad = float(math.radians(yaw_rate_deg))

        try:
            self.client.moveByRollPitchYawrateZAsync(
                float(roll_cmd),
                float(pitch_cmd),
                float(yaw_rate_rad),
                float(target_z),
                float(self.dt),
            ).join()
        except Exception as exc:
            print(f"[STEP {self.step_count}] control exception: {exc}")
            obs = self._get_obs()
            info = {
                "next_gate_idx": self.next_gate_idx,
                "gate_dist": float("inf"),
                "gate_found": 0.0,
                "closest_gate_idx": -1,
                "control_error": str(exc),
            }
            return obs, -25.0, True, False, info

        pos, vel, yaw = self._get_state()
        obs = self._get_obs()

        gate_pos = self.gates[min(self.next_gate_idx, len(self.gates) - 1)]
        gate_dist = norm(gate_pos - pos)

        reward = 0.0
        terminated = False
        truncated = False

        # Time penalty: every step costs a small amount (time-optimal objective)
        reward -= 0.002

        progress = self.prev_gate_dist - gate_dist
        reward += 2.0 * progress   # reduced from 5.0: less myopic guidance
        self.prev_gate_dist = gate_dist

        # Speed bonus: encourage fast flight
        reward += 0.01 * float(np.linalg.norm(vel))
        reward -= 0.01 * abs(yaw_rate_deg)
        reward -= 0.01 * (abs(roll_cmd) + abs(pitch_cmd))

        # Update contrastive buffer if moving away from gate (hard state)
        if progress < -0.2 * self.dt:
            entry = {"pos": pos.copy(), "vel": vel.copy(), "gate_idx": self.next_gate_idx}
            self._hard_state_buffer.append(entry)
            if len(self._hard_state_buffer) > _HARD_BUF_MAX:
                self._hard_state_buffer.pop(0)

        gate_found = float(obs[-5])
        gate_area = float(obs[-2])
        reward += 0.2 * gate_found
        reward += 0.5 * gate_area

        all_gate_dists = [norm(g - pos) for g in self.gates]
        closest_gate_idx = int(np.argmin(all_gate_dists))
        if closest_gate_idx != self.next_gate_idx:
            reward -= 0.5

        if gate_dist < GATE_HIT_RADIUS:
            print(f"[STEP {self.step_count}] Passed gate {self.next_gate_idx} at dist {gate_dist:.2f}")
            reward += 75.0
            self.next_gate_idx += 1
            if self.next_gate_idx >= len(self.gates):
                reward += 250.0
                terminated = True
                print(f"[STEP {self.step_count}] Course complete")
            else:
                self.prev_gate_dist = norm(self.gates[self.next_gate_idx] - pos)

        collision = self.client.simGetCollisionInfo()
        if collision.has_collided:
            reward -= 100.0
            terminated = True
            print(f"[STEP {self.step_count}] Collision detected")

        if self.step_count >= self.max_steps:
            truncated = True
            print(f"[STEP {self.step_count}] Episode truncated at max steps")

        if self.step_count % 10 == 0:
            print(
                f"[STEP {self.step_count}] "
                f"active_gate={self.next_gate_idx} "
                f"closest_gate={closest_gate_idx} "
                f"dist={gate_dist:.2f} "
                f"reward={reward:.3f} "
                f"vel=({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}) "
                f"cmd=(r={roll_cmd:.2f},p={pitch_cmd:.2f},yaw_rate={yaw_rate_deg:.2f},z={target_z:.2f}) "
                f"gate_found={gate_found:.0f} "
                f"gate_area={gate_area:.4f}"
            )

        info = {
            "next_gate_idx": self.next_gate_idx,
            "gate_dist": gate_dist,
            "gate_found": gate_found,
            "closest_gate_idx": closest_gate_idx,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        print("[CLOSE] Shutting down env")
        try:
            self.client.hoverAsync().join()
        except Exception:
            pass
        try:
            self.client.landAsync().join()
        except Exception:
            pass
        try:
            self.client.disarm()
        except Exception:
            pass
        try:
            self.client.disableApiControl()
        except Exception:
            pass
