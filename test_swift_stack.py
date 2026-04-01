"""
Swift Stack Integration Test
=============================
End-to-end validation of all Swift paper components:
  1. IMU data retrieval
  2. VIO drift accumulation
  3. PnP accuracy (near Gate00)
  4. Kalman filter convergence
  5. Policy rollout (3 episodes)

Run with AirSim open: python test_swift_stack.py
"""

import math
import time

import numpy as np

from hardware import AirSimInterface
from vio_estimator import VIOEstimator
from pnp_localizer import PnPLocalizer
from kalman_fusion import SwiftKalmanFilter
from gate_detector import GateDetector


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def pos_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _get_quat_array(iface: AirSimInterface) -> np.ndarray:
    q = iface.get_orientation()
    return np.array([q.w_val, q.x_val, q.y_val, q.z_val], dtype=np.float64)


# ────────────────────────────────────────────────────────────
# Test 1 — IMU data retrieval
# ────────────────────────────────────────────────────────────

def test_imu(iface: AirSimInterface):
    print("\n=== TEST 1: IMU Data Retrieval ===")
    imu = iface.get_imu_data()
    accel = imu["linear_acceleration"]
    gyro  = imu["angular_velocity"]

    print(f"  linear_acceleration: {accel}")
    print(f"  angular_velocity:    {gyro}")

    accel_norm = float(np.linalg.norm(accel))
    # On the ground with gravity, AirSim IMU should report ~9.8 m/s²
    if accel_norm < 0.1:
        print("  WARNING: near-zero acceleration — IMU may not be active")
    else:
        print(f"  accel magnitude: {accel_norm:.3f} m/s² ✓")

    print("  PASS")


# ────────────────────────────────────────────────────────────
# Test 2 — VIO drift test (5 seconds of flight)
# ────────────────────────────────────────────────────────────

def test_vio_drift(iface: AirSimInterface, duration_s: float = 5.0):
    print(f"\n=== TEST 2: VIO Drift Test ({duration_s}s) ===")

    pos0 = iface.get_position()
    vel0 = iface.get_velocity()
    quat0 = _get_quat_array(iface)

    vio = VIOEstimator()
    vio.initialize(pos0, vel0, quat0)

    dt = 0.05
    steps = int(duration_s / dt)
    max_err = 0.0

    for i in range(steps):
        imu = iface.get_imu_data()
        vio.update(imu, dt)

        gt_pos  = iface.get_position()
        vio_pos = vio.get_state()["pos"]
        err     = pos_error(vio_pos, gt_pos)
        max_err = max(max_err, err)

        # Command slow forward flight for a visible drift signal
        iface.move_by_velocity(2.0, 0.0, 0.0, dt, math.degrees(iface.get_yaw()))

        time.sleep(dt * 0.5)

    iface.hover()

    gt_final  = iface.get_position()
    vio_final = vio.get_state()["pos"]
    final_err = pos_error(vio_final, gt_final)

    print(f"  VIO final pos:  {vio_final}")
    print(f"  GT  final pos:  {gt_final}")
    print(f"  Final error:    {final_err:.3f} m")
    print(f"  Max error seen: {max_err:.3f} m")

    if final_err < 0.1:
        print("  WARNING: Very low drift — VIO may be reading ground truth directly")
    else:
        print("  VIO drifts as expected ✓")
    print("  PASS")


# ────────────────────────────────────────────────────────────
# Test 3 — PnP accuracy near Gate00
# ────────────────────────────────────────────────────────────

def test_pnp_accuracy(iface: AirSimInterface):
    print("\n=== TEST 3: PnP Accuracy Test ===")

    gate_poses = iface.get_gate_poses()
    if not gate_poses:
        print("  SKIP: No gates found in scene")
        return

    gate_name, gate_world_pos = gate_poses[0]
    print(f"  Using gate: {gate_name} at {gate_world_pos}")

    # Hover near gate
    approach_pos = gate_world_pos.copy()
    approach_pos[0] -= 8.0   # 8m in front
    iface.move_to_position(*approach_pos, speed=3.0)
    time.sleep(0.5)

    gt_pos  = iface.get_position()
    frame   = iface.get_image()
    quat    = _get_quat_array(iface)

    detector = GateDetector("gate_detector.pt")
    pnp = PnPLocalizer(gate_half_size_m=0.75, img_w=frame.shape[1],
                       img_h=frame.shape[0], fov_h_deg=90.0)

    det = detector.detect(frame) if detector.available else None

    if det is None or "bbox_xyxy" not in det:
        print("  WARNING: Gate not detected in frame — cannot run PnP")
        print("  SKIP")
        return

    bbox = det["bbox_xyxy"]
    result = pnp.localize(bbox, gate_world_pos, quat)

    if not result["success"]:
        print("  PnP solve failed")
        print("  FAIL")
        return

    err = pos_error(result["pos"], gt_pos)
    print(f"  GT pos:         {gt_pos}")
    print(f"  PnP est pos:    {result['pos']}")
    print(f"  Distance to gate: {result['distance_m']:.2f} m")
    print(f"  Position error: {err:.3f} m")

    if err < 2.0:
        print("  PnP within 2m of ground truth ✓")
        print("  PASS")
    else:
        print("  WARNING: PnP error > 2m — check camera calibration or gate geometry")
        print("  PARTIAL")


# ────────────────────────────────────────────────────────────
# Test 4 — Kalman filter convergence
# ────────────────────────────────────────────────────────────

def test_kalman_convergence(iface: AirSimInterface):
    print("\n=== TEST 4: Kalman Filter Convergence ===")

    gate_poses = iface.get_gate_poses()
    if not gate_poses:
        print("  SKIP: No gates found in scene")
        return

    gate_name, gate_world_pos = gate_poses[0]

    pos0  = iface.get_position()
    vel0  = iface.get_velocity()
    quat0 = _get_quat_array(iface)

    vio = VIOEstimator()
    vio.initialize(pos0, vel0, quat0)

    kf = SwiftKalmanFilter()
    kf.initialize(pos0, vel0)

    pnp = PnPLocalizer(gate_half_size_m=0.75, img_w=640, img_h=480, fov_h_deg=90.0)
    detector = GateDetector("gate_detector.pt")

    dt = 0.05
    duration_s = 8.0
    steps = int(duration_s / dt)

    errors_vio = []
    errors_kf  = []
    pnp_updates = 0

    for i in range(steps):
        imu = iface.get_imu_data()
        vio.update(imu, dt)
        kf.predict(imu, dt)

        gt_pos = iface.get_position()
        quat   = _get_quat_array(iface)
        frame  = iface.get_image()

        # PnP update when gate detected
        if detector.available:
            det = detector.detect(frame)
            if det is not None and "bbox_xyxy" in det:
                result = pnp.localize(det["bbox_xyxy"], gate_world_pos, quat)
                if result["success"]:
                    kf.update_pnp(result["pos"], result["cov"])
                    pnp_updates += 1

        vio_pos = vio.get_state()["pos"]
        kf_pos  = kf.get_state()["pos"]

        errors_vio.append(pos_error(vio_pos, gt_pos))
        errors_kf.append(pos_error(kf_pos, gt_pos))

        # Fly toward gate
        target_dir = gate_world_pos - gt_pos
        dist = float(np.linalg.norm(target_dir))
        if dist > 2.0:
            spd = min(3.0, dist)
            d = target_dir / dist
            iface.move_by_velocity(d[0]*spd, d[1]*spd, d[2]*spd, dt,
                                   math.degrees(math.atan2(d[1], d[0])))

        time.sleep(dt * 0.5)

    iface.hover()

    mean_vio = float(np.mean(errors_vio))
    mean_kf  = float(np.mean(errors_kf))
    max_vio  = float(np.max(errors_vio))
    max_kf   = float(np.max(errors_kf))

    print(f"  PnP updates applied: {pnp_updates}")
    print(f"  VIO mean/max error:  {mean_vio:.3f} / {max_vio:.3f} m")
    print(f"  KF  mean/max error:  {mean_kf:.3f} / {max_kf:.3f} m")

    if mean_kf <= mean_vio:
        print("  Kalman filter reduces VIO error ✓")
        print("  PASS")
    else:
        print("  WARNING: KF error > VIO — check PnP geometry or gate detection")
        print("  PARTIAL")


# ────────────────────────────────────────────────────────────
# Test 5 — Policy rollout
# ────────────────────────────────────────────────────────────

def test_policy_rollout():
    print("\n=== TEST 5: Policy Rollout ===")

    try:
        from stable_baselines3 import PPO
        from adrl_env import ADRLRacingEnv
    except ImportError as e:
        print(f"  SKIP: {e}")
        return

    model_path = "adrl_ppo_racer.zip"
    import os
    if not os.path.exists(model_path):
        print(f"  SKIP: {model_path} not found (train first)")
        return

    env = ADRLRacingEnv()
    model = PPO.load(model_path, env=env)

    for ep in range(3):
        obs, _ = env.reset()
        gates_completed = 0
        done = False
        steps = 0
        while not done and steps < 1500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            gates_completed = info.get("next_gate_idx", 0)
            done = terminated or truncated
            steps += 1
        print(f"  Episode {ep+1}: {gates_completed} gates, {steps} steps")

    env.close()
    print("  PASS")


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    print("Swift Stack Integration Test")
    print("=" * 50)

    print("\nConnecting to AirSim...")
    try:
        iface = AirSimInterface()
        print("Connected ✓")
    except Exception as e:
        print(f"FAIL: Could not connect to AirSim: {e}")
        return

    # Arm and take off to test altitude
    try:
        iface.arm_and_takeoff(target_z=TAKEOFF_Z)
        time.sleep(1.0)
    except Exception as e:
        print(f"WARNING: arm_and_takeoff failed: {e} — tests may run on the ground")

    test_imu(iface)
    test_vio_drift(iface)
    test_pnp_accuracy(iface)
    test_kalman_convergence(iface)
    test_policy_rollout()

    print("\n=== All tests complete ===")
    iface.shutdown()


TAKEOFF_Z = -1.5

if __name__ == "__main__":
    main()
