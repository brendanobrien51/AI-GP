"""
VIO Estimator — Swift-style Visual-Inertial Odometry Simulation
===============================================================
Simulates what a real VIO pipeline (e.g., VINS-Mono, MSCKF) would produce
when running on the drone's IMU + camera. Uses AirSim's IMU data, integrates
with realistic noise and bias drift so the estimate diverges over time the
same way real VIO does at high speeds.

PnP corrections (via kalman_fusion.py) are needed to keep the estimate bounded.

Usage:
    vio = VIOEstimator()
    vio.initialize(pos, vel, quat)          # seed from AirSim ground truth once
    for each control loop step:
        imu = iface.get_imu_data()
        vio.update(imu, dt=0.05)
        state = vio.get_state()             # pos, vel, quat, cov
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Default noise parameters (tuned to mimic real drone-racing VIO)
# ---------------------------------------------------------------------------
_DEFAULT = {
    # Accelerometer white noise (m/s² per sqrt(Hz))
    "accel_noise_std":    0.05,
    # Gyroscope white noise (rad/s per sqrt(Hz))
    "gyro_noise_std":     0.01,
    # Accelerometer bias random-walk (m/s³ per sqrt(Hz))
    "accel_bias_std":     0.002,
    # Gyroscope bias random-walk (rad/s² per sqrt(Hz))
    "gyro_bias_std":      0.0005,
    # Additional position noise injected per step (captures feature-track
    # failures and other visual odometry errors at high speed)
    "pos_noise_std":      0.003,
}


class AeroResidual:
    """
    HDVIO-style aerodynamic residual model (Cioffi & Bauersfeld, 2023).

    Maps flight speed → estimated drag correction vector in world frame.
    Default weights are zero (no-op) until calibrated via calibrate_aero.py.

    Simple quadratic model: drag ∝ speed² in the velocity direction.
    Learns scalar coefficient k from flight data.
    """

    _WEIGHTS_FILE = "aero_residual.npy"

    def __init__(self):
        # k: drag coefficient (m/s² per (m/s)²)
        # Defaults to 0 → no correction until calibrated
        self._k = 0.0
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._WEIGHTS_FILE):
            try:
                self._k = float(np.load(self._WEIGHTS_FILE))
                print(f"[AeroResidual] Loaded k={self._k:.5f} from {self._WEIGHTS_FILE}")
            except Exception:
                pass

    def predict(self, vel_world: np.ndarray) -> np.ndarray:
        """
        Estimate aerodynamic drag deceleration vector.

        Args:
            vel_world: [vx, vy, vz] in world frame (m/s)

        Returns:
            drag_correction: [dx, dy, dz] in m/s² — subtract from accel_world
        """
        spd = float(np.linalg.norm(vel_world))
        if spd < 0.1 or self._k == 0.0:
            return np.zeros(3)
        # Drag opposes velocity: direction = -vel/speed
        return (vel_world / spd) * self._k * spd**2

    def fit(self, speeds: np.ndarray, errors: np.ndarray) -> None:
        """
        Fit drag coefficient from calibration data.

        Args:
            speeds: 1D array of flight speeds (m/s)
            errors: 1D array of VIO-vs-GT error magnitudes (m) at those speeds
        """
        # Linear least squares: error ≈ k * speed²
        A = (speeds**2).reshape(-1, 1)
        b = errors.reshape(-1, 1)
        k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        self._k = float(k[0])
        np.save(self._WEIGHTS_FILE, np.array(self._k))
        print(f"[AeroResidual] Fitted k={self._k:.5f}, saved to {self._WEIGHTS_FILE}")


class VIOEstimator:
    """
    Simulated VIO pipeline.

    Integrates IMU measurements (with noise + bias drift) to propagate
    a running position/velocity/orientation estimate.  The estimate will
    drift at roughly the same rate as a real state-of-the-art VIO system
    operating at 10–15 m/s.
    """

    def __init__(self, noise: dict | None = None):
        cfg = {**_DEFAULT, **(noise or {})}
        self._an_std  = cfg["accel_noise_std"]
        self._gn_std  = cfg["gyro_noise_std"]
        self._ab_std  = cfg["accel_bias_std"]
        self._gb_std  = cfg["gyro_bias_std"]
        self._pn_std  = cfg["pos_noise_std"]

        # State
        self._pos  = np.zeros(3)
        self._vel  = np.zeros(3)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)

        # Bias terms (random-walk)
        self._accel_bias = np.zeros(3)
        self._gyro_bias  = np.zeros(3)

        # 9×9 covariance for [pos(3), vel(3), att(3)]
        self._cov = np.eye(9) * 0.01

        # HDVIO aerodynamic residual (zero by default until calibrated)
        self._aero = AeroResidual()

        self._initialized = False

    # ------------------------------------------------------------------
    def initialize(self, pos: np.ndarray, vel: np.ndarray,
                   quat_wxyz: np.ndarray) -> None:
        """Seed the estimator from ground-truth AirSim state."""
        self._pos  = np.array(pos,  dtype=np.float64)
        self._vel  = np.array(vel,  dtype=np.float64)
        q = np.array(quat_wxyz, dtype=np.float64)
        self._quat = q / np.linalg.norm(q)
        self._accel_bias = np.random.randn(3) * self._ab_std * 10  # initial bias
        self._gyro_bias  = np.random.randn(3) * self._gb_std * 10
        self._cov = np.eye(9) * 0.01
        self._initialized = True

    # ------------------------------------------------------------------
    def update(self, imu_data: dict, dt: float) -> None:
        """
        Propagate estimate using one IMU measurement.

        Args:
            imu_data: dict with keys
                "linear_acceleration": [ax, ay, az] in body frame (m/s²)
                "angular_velocity":    [gx, gy, gz] in body frame (rad/s)
            dt: timestep in seconds
        """
        if not self._initialized:
            return

        # ---- Add noise to raw IMU readings ----
        accel_raw = np.array(imu_data["linear_acceleration"], dtype=np.float64)
        gyro_raw  = np.array(imu_data["angular_velocity"],    dtype=np.float64)

        accel_noisy = (accel_raw
                       + self._accel_bias
                       + np.random.randn(3) * self._an_std / np.sqrt(dt))
        gyro_noisy  = (gyro_raw
                       + self._gyro_bias
                       + np.random.randn(3) * self._gn_std / np.sqrt(dt))

        # ---- Rotate acceleration from body to world frame ----
        R_body2world = self._rotation_matrix()
        accel_world  = R_body2world @ accel_noisy

        # Remove gravity (AirSim IMU includes gravity in Z-down direction)
        gravity_ned = np.array([0.0, 0.0, 9.81])  # NED: +Z is down
        accel_world -= gravity_ned

        # HDVIO: subtract aerodynamic drag residual (zero until calibrated)
        accel_world -= self._aero.predict(self._vel)

        # ---- Integrate: vel += a*dt, pos += v*dt ----
        self._vel += accel_world * dt
        self._pos += (self._vel * dt
                      + np.random.randn(3) * self._pn_std)  # visual noise

        # ---- Integrate quaternion from gyro ----
        omega = gyro_noisy
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-8:
            axis  = omega / omega_norm
            angle = omega_norm * dt
            dq    = _axis_angle_to_quat(axis, angle)
            self._quat = _quat_multiply(self._quat, dq)
            self._quat /= np.linalg.norm(self._quat)

        # ---- Bias random walk ----
        self._accel_bias += np.random.randn(3) * self._ab_std * np.sqrt(dt)
        self._gyro_bias  += np.random.randn(3) * self._gb_std * np.sqrt(dt)

        # ---- Propagate covariance (simplified linearised) ----
        Q = np.diag([
            self._pn_std**2,  self._pn_std**2,  self._pn_std**2,    # pos
            (self._an_std*dt)**2, (self._an_std*dt)**2, (self._an_std*dt)**2,  # vel
            (self._gn_std*dt)**2, (self._gn_std*dt)**2, (self._gn_std*dt)**2,  # att
        ])
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt   # pos += vel*dt
        self._cov = F @ self._cov @ F.T + Q

    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """
        Return current VIO state estimate.

        Returns:
            pos:  np.ndarray [x, y, z] (m, NED)
            vel:  np.ndarray [vx, vy, vz] (m/s, NED)
            quat: np.ndarray [w, x, y, z] (unit quaternion)
            cov:  np.ndarray 9×9 covariance matrix [pos, vel, att]
        """
        return {
            "pos":  self._pos.copy(),
            "vel":  self._vel.copy(),
            "quat": self._quat.copy(),
            "cov":  self._cov.copy(),
        }

    # ------------------------------------------------------------------
    def set_position(self, pos: np.ndarray) -> None:
        """External correction (e.g., from Kalman filter)."""
        self._pos = np.array(pos, dtype=np.float64)

    def set_velocity(self, vel: np.ndarray) -> None:
        self._vel = np.array(vel, dtype=np.float64)

    # ------------------------------------------------------------------
    def _rotation_matrix(self) -> np.ndarray:
        """Current body→world rotation matrix from stored quaternion."""
        w, x, y, z = self._quat
        return np.array([
            [1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),    1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),    2*(y*z+w*x),   1-2*(x*x+y*y)],
        ], dtype=np.float64)

    @property
    def initialized(self) -> bool:
        return self._initialized


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return unit quaternion [w, x, y, z] for rotation of `angle` about `axis`."""
    half = angle / 2.0
    return np.array([
        np.cos(half),
        axis[0] * np.sin(half),
        axis[1] * np.sin(half),
        axis[2] * np.sin(half),
    ])


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
