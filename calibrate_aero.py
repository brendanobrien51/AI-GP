"""
Aerodynamic Residual Calibration
==================================
Flies drone at several speeds in a straight line, compares VIO position
against AirSim ground truth, and fits the AeroResidual drag coefficient.

After running this, vio_estimator.py will automatically load the fitted
coefficient and subtract the drag correction on every update step.

Run with AirSim open: python calibrate_aero.py
Output: aero_residual.npy
"""

import time
import numpy as np

from hardware import AirSimInterface
from vio_estimator import VIOEstimator, AeroResidual

TEST_SPEEDS   = [3.0, 6.0, 9.0, 12.0]  # m/s
RUN_DURATION  = 4.0                      # seconds per speed
DT            = 0.05                     # control timestep


def run_at_speed(iface: AirSimInterface, speed: float,
                 vio: VIOEstimator) -> tuple[float, float]:
    """
    Fly straight at `speed` for RUN_DURATION seconds.
    Returns (mean_speed, mean_vio_error_m).
    """
    print(f"  Flying at {speed:.1f} m/s for {RUN_DURATION:.1f}s...")

    yaw = iface.get_yaw()
    vx  = speed * np.cos(yaw)
    vy  = speed * np.sin(yaw)

    pos0 = iface.get_position()
    vel0 = iface.get_velocity()
    q0   = iface.get_orientation()
    quat0 = np.array([q0.w_val, q0.x_val, q0.y_val, q0.z_val])

    vio.initialize(pos0, vel0, quat0)

    errors = []
    steps  = int(RUN_DURATION / DT)

    for _ in range(steps):
        imu = iface.get_imu_data()
        vio.update(imu, DT)

        gt_pos  = iface.get_position()
        vio_pos = vio.get_state()["pos"]
        err     = float(np.linalg.norm(vio_pos - gt_pos))
        errors.append(err)

        import math
        iface.move_by_velocity(float(vx), float(vy), 0.0, DT,
                               math.degrees(yaw))
        time.sleep(DT * 0.8)

    iface.hover()
    time.sleep(0.5)

    mean_err = float(np.mean(errors))
    print(f"    Mean VIO error: {mean_err:.4f} m")
    return speed, mean_err


def main():
    print("Aerodynamic Residual Calibration")
    print("=" * 40)

    iface = AirSimInterface()
    print("Connected to AirSim")

    # Take off
    iface.arm_and_takeoff(target_z=-2.0)
    time.sleep(1.0)

    vio    = VIOEstimator()
    speeds = []
    errors = []

    for spd in TEST_SPEEDS:
        s, e = run_at_speed(iface, spd, vio)
        speeds.append(s)
        errors.append(e)
        time.sleep(1.0)

    iface.hover()

    speeds_arr = np.array(speeds)
    errors_arr = np.array(errors)

    print("\nCalibration data:")
    for s, e in zip(speeds_arr, errors_arr):
        print(f"  {s:.1f} m/s → {e:.4f} m error")

    aero = AeroResidual()
    aero.fit(speeds_arr, errors_arr)
    print(f"\nCalibration complete. Drag coefficient k={aero._k:.5f}")
    print("aero_residual.npy saved — VIO will use this correction automatically.")

    iface.shutdown()


if __name__ == "__main__":
    main()
