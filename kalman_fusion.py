"""
Kalman Fusion — Swift-style VIO + PnP State Estimator
======================================================
Extended Kalman Filter (EKF) that fuses:
  - High-frequency VIO predictions (IMU integration, ~20 Hz)
  - Occasional PnP absolute position corrections (when gate visible)

State vector: [x, y, z, vx, vy, vz]  (6D, NED metres/m·s⁻¹)
Attitude (quaternion) is tracked separately inside VIOEstimator.

As noted in the Swift paper, the Kalman filter posterior variance
decreases quadratically as the drone approaches a detected gate —
exactly because PnP measurement noise ∝ distance².

Usage:
    kf = SwiftKalmanFilter()
    kf.initialize(pos, vel)

    # each control loop step:
    kf.predict(imu_data, dt)

    # whenever YOLO detects a gate:
    kf.update_pnp(pnp_result["pos"], pnp_result["cov"])

    state = kf.get_state()   # pos, vel, uncertainty
"""

import numpy as np
from filterpy.kalman import KalmanFilter


class SwiftKalmanFilter:
    """
    6-state linear Kalman filter (constant-velocity process model).

    Predict: uses VIO-corrected velocity to propagate position.
    Update:  absorbs PnP position fixes with distance-scaled covariance.
    """

    def __init__(
        self,
        pos_process_noise: float = 0.01,
        vel_process_noise: float = 0.1,
    ):
        # filterpy KalmanFilter: dim_x=6 states, dim_z=3 (position obs)
        self._kf = KalmanFilter(dim_x=6, dim_z=3)

        # State transition: constant velocity
        # x_{k+1} = F * x_k   (dt filled in at each predict step)
        self._kf.F = np.eye(6)

        # Observation: we measure position only
        self._kf.H = np.zeros((3, 6))
        self._kf.H[0, 0] = 1.0
        self._kf.H[1, 1] = 1.0
        self._kf.H[2, 2] = 1.0

        # Process noise Q (will be rebuilt at each predict with dt)
        self._pos_q = pos_process_noise
        self._vel_q = vel_process_noise

        # Measurement noise R (overridden per-update from PnP covariance)
        self._kf.R = np.eye(3) * 0.1

        # Initial covariance
        self._kf.P = np.eye(6) * 1.0

        self._initialized = False

    # ------------------------------------------------------------------
    def initialize(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """Seed filter from ground-truth or VIO initial state."""
        self._kf.x = np.concatenate([
            np.array(pos, dtype=np.float64),
            np.array(vel, dtype=np.float64),
        ])
        self._kf.P = np.eye(6) * 0.1
        self._initialized = True

    # ------------------------------------------------------------------
    def predict(self, imu_data: dict, dt: float) -> None:
        """
        Predict step driven by IMU acceleration (from VIO pipeline).

        Args:
            imu_data: dict with "linear_acceleration" [ax, ay, az] in world NED
                      (already rotated out of body frame by VIOEstimator)
            dt: timestep seconds
        """
        if not self._initialized:
            return

        # Rebuild F with current dt
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self._kf.F = F

        # Control input: Δv = a * dt  (add to velocity states)
        accel = np.array(imu_data.get("linear_acceleration", [0, 0, 0]),
                         dtype=np.float64)
        # Apply acceleration as additive control (u = [0,0,0, ax,ay,az]*dt)
        self._kf.x[3:6] += accel * dt

        # Process noise Q
        Q = np.diag([
            self._pos_q * dt**2,
            self._pos_q * dt**2,
            self._pos_q * dt**2,
            self._vel_q * dt,
            self._vel_q * dt,
            self._vel_q * dt,
        ])
        self._kf.Q = Q

        self._kf.predict()

    # ------------------------------------------------------------------
    def update_pnp(
        self,
        pnp_pos: np.ndarray,
        measurement_cov: np.ndarray | None = None,
    ) -> None:
        """
        Measurement update from PnP gate localisation.

        Args:
            pnp_pos: [x, y, z] drone position from PnP (world NED, metres).
            measurement_cov: 3×3 covariance from pnp_localizer (optional;
                             if None uses default R).
        """
        if not self._initialized:
            return

        if measurement_cov is not None:
            self._kf.R = measurement_cov.astype(np.float64)

        z = np.array(pnp_pos, dtype=np.float64)
        self._kf.update(z)

    # ------------------------------------------------------------------
    def update_vio(
        self,
        vio_pos: np.ndarray,
        vio_cov: np.ndarray | None = None,
    ) -> None:
        """
        Soft update from VIO position estimate (higher noise than PnP).
        Use sparingly — mainly to prevent unbounded drift between PnP fixes.
        """
        if not self._initialized:
            return

        if vio_cov is not None:
            self._kf.R = vio_cov[:3, :3].astype(np.float64)
        else:
            self._kf.R = np.eye(3) * 2.0  # VIO position is noisy

        z = np.array(vio_pos, dtype=np.float64)
        self._kf.update(z)

    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """
        Return current fused state estimate.

        Returns:
            pos:         np.ndarray [x, y, z] (m, NED)
            vel:         np.ndarray [vx, vy, vz] (m/s, NED)
            uncertainty: float — trace of position covariance (m²)
                         Low = confident, high = uncertain.
            cov:         np.ndarray 6×6 full state covariance
        """
        x = self._kf.x
        P = self._kf.P
        return {
            "pos":         x[0:3].copy(),
            "vel":         x[3:6].copy(),
            "uncertainty": float(np.trace(P[0:3, 0:3])),
            "cov":         P.copy(),
        }

    # ------------------------------------------------------------------
    def set_pos(self, pos: np.ndarray) -> None:
        """Hard-set position (e.g., after VIO re-initialisation)."""
        self._kf.x[0:3] = np.array(pos, dtype=np.float64)

    @property
    def initialized(self) -> bool:
        return self._initialized
