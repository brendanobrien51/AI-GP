"""
MPCC Controller — Model Predictive Contouring Control
======================================================
Based on: "Model Predictive Contouring Control for High-Speed Autonomous
           Drone Racing" (Romero et al.) and "MPCC++" (Krinner, Romero et al.)

Instead of tracking a single lookahead point (pure pursuit), MPCC solves
a short-horizon optimization that simultaneously:
  - Maximises forward progress along the racing line (Δθ)
  - Minimises lateral deviation from the racing line (e_lat)
  - Tracks a reference speed

The optimization runs in receding-horizon fashion: solve N steps ahead,
apply only the first command, repeat next cycle.

Usage:
    mpcc = MPCCController(horizon=10, dt=0.05)
    mpcc.set_path(list_of_xyz_waypoints)

    # Each control loop step:
    vx, vy, vz, yaw = mpcc.compute(pos, vel, speed_limit)
    # Returns world-frame velocity command + desired heading
"""

import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


class MPCCController:
    """
    Receding-horizon contouring controller for drone racing.

    The racing line is arc-length parameterized via cubic splines.
    At each step, SLSQP solves for N velocity commands minimizing:
        J = -w_prog * sum(Δθ) + w_lat * sum(e_lat²) + w_speed * sum((v - v_ref)²)
    subject to ||v_cmd|| <= speed_limit.
    """

    def __init__(
        self,
        horizon: int = 10,
        dt: float = 0.05,
        w_progress: float = 2.0,
        w_lateral: float = 5.0,
        w_speed: float = 0.5,
    ):
        self._H   = horizon
        self._dt  = dt
        self._wp  = w_progress
        self._wl  = w_lateral
        self._ws  = w_speed

        self._spline_x: CubicSpline | None = None
        self._spline_y: CubicSpline | None = None
        self._spline_z: CubicSpline | None = None
        self._arc_total: float = 0.0
        self._waypoints: np.ndarray | None = None
        self._arc_s: np.ndarray | None = None    # arc-length at each waypoint
        self._theta: float = 0.0                  # current progress (arc-length)

    # ------------------------------------------------------------------
    def set_path(self, waypoints: list) -> None:
        """
        Build arc-length parameterized spline from a list of [x, y, z] waypoints.

        Args:
            waypoints: List of np.ndarray or list [x, y, z], at least 4 points.
        """
        pts = np.array(waypoints, dtype=np.float64)
        if len(pts) < 4:
            # Need at least 4 points for cubic spline — pad if needed
            pts = np.vstack([pts, pts[-1:], pts[-1:], pts[-1:]])[:max(4, len(pts))]

        # Compute cumulative arc-length
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        arc_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        # Remove duplicate arc-length values (degenerate segments)
        _, unique_idx = np.unique(arc_s, return_index=True)
        arc_s = arc_s[unique_idx]
        pts   = pts[unique_idx]

        self._arc_s     = arc_s
        self._arc_total = float(arc_s[-1])
        self._waypoints = pts

        self._spline_x = CubicSpline(arc_s, pts[:, 0])
        self._spline_y = CubicSpline(arc_s, pts[:, 1])
        self._spline_z = CubicSpline(arc_s, pts[:, 2])

        self._theta = 0.0

    # ------------------------------------------------------------------
    def compute(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        speed_limit: float = 12.0,
        v_ref: float | None = None,
    ) -> tuple[float, float, float, float]:
        """
        Solve MPCC and return the first-step velocity command.

        Args:
            pos:         [x, y, z] drone position (world NED, m)
            vel:         [vx, vy, vz] current velocity (m/s)
            speed_limit: maximum allowed speed magnitude (m/s)
            v_ref:       reference speed (defaults to speed_limit)

        Returns:
            (vx, vy, vz, yaw_deg) — world-frame velocity command + desired heading
        """
        if self._spline_x is None:
            return (0.0, 0.0, 0.0, 0.0)

        if v_ref is None:
            v_ref = speed_limit

        # Find closest point on path to current position → update θ
        self._theta = self._project_to_path(pos)

        # Initial guess: extend current velocity for N steps
        cur_speed = float(np.linalg.norm(vel))
        if cur_speed < 0.1:
            v0 = np.array([v_ref, 0.0, 0.0])
        else:
            v0 = vel.astype(np.float64) / cur_speed * min(cur_speed, v_ref)

        x0 = np.tile(v0, self._H)   # shape (3*H,)

        # Bounds: each velocity component in [-speed_limit, speed_limit]
        bounds = [(-speed_limit, speed_limit)] * (3 * self._H)

        result = minimize(
            fun=self._cost,
            x0=x0,
            args=(pos, self._theta, v_ref, speed_limit),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-4},
        )

        v_cmd = result.x[:3].astype(np.float64)

        # Clamp to speed limit
        spd = float(np.linalg.norm(v_cmd))
        if spd > speed_limit:
            v_cmd = v_cmd / spd * speed_limit

        # Desired heading: tangent direction at current progress
        yaw = self._path_yaw(self._theta)

        return float(v_cmd[0]), float(v_cmd[1]), float(v_cmd[2]), yaw

    # ------------------------------------------------------------------
    def get_progress_frac(self) -> float:
        """Return completion fraction 0→1 along the racing line."""
        if self._arc_total < 1e-6:
            return 0.0
        return float(np.clip(self._theta / self._arc_total, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _cost(
        self,
        v_flat: np.ndarray,
        pos0: np.ndarray,
        theta0: float,
        v_ref: float,
        speed_limit: float,
    ) -> float:
        """MPCC cost function over the N-step horizon."""
        H  = self._H
        dt = self._dt
        pos    = pos0.copy()
        theta  = theta0
        J      = 0.0

        for k in range(H):
            vk = v_flat[3*k : 3*k+3]
            pos_next = pos + vk * dt

            # New theta: project pos_next onto path
            theta_next = self._project_to_path(pos_next)

            # Progress gained (handle wrap-around if path is a loop)
            dtheta = theta_next - theta
            if dtheta < 0:
                dtheta = 0.0

            # Lateral error: distance from path at theta_next
            path_pt  = self._path_point(theta_next)
            path_tan = self._path_tangent(theta_next)
            diff     = pos_next - path_pt
            # Lateral = component of diff perpendicular to tangent
            e_along  = float(np.dot(diff, path_tan))
            e_lat_vec = diff - e_along * path_tan
            e_lat    = float(np.linalg.norm(e_lat_vec))

            # Speed deviation
            spd   = float(np.linalg.norm(vk))
            e_spd = spd - v_ref

            J += -self._wp * dtheta
            J +=  self._wl * (e_lat ** 2)
            J +=  self._ws * (e_spd ** 2)

            pos   = pos_next
            theta = theta_next

        return J

    # ------------------------------------------------------------------
    def _project_to_path(self, pos: np.ndarray) -> float:
        """Find arc-length parameter θ closest to pos."""
        pts = self._waypoints
        dists = np.linalg.norm(pts - pos, axis=1)
        idx   = int(np.argmin(dists))

        # Refine within [s[idx-1], s[idx+1]] using scalar minimization
        s_lo = self._arc_s[max(0, idx - 1)]
        s_hi = self._arc_s[min(len(self._arc_s) - 1, idx + 1)]

        def dist_sq(s):
            p = self._path_point(float(s))
            return float(np.sum((p - pos) ** 2))

        from scipy.optimize import minimize_scalar
        res = minimize_scalar(dist_sq, bounds=(s_lo, s_hi), method="bounded")
        return float(np.clip(res.x, 0.0, self._arc_total))

    def _path_point(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self._arc_total))
        return np.array([
            float(self._spline_x(s)),
            float(self._spline_y(s)),
            float(self._spline_z(s)),
        ])

    def _path_tangent(self, s: float) -> np.ndarray:
        """Unit tangent vector at arc-length s."""
        s = float(np.clip(s, 0.0, self._arc_total))
        t = np.array([
            float(self._spline_x(s, 1)),
            float(self._spline_y(s, 1)),
            float(self._spline_z(s, 1)),
        ])
        n = float(np.linalg.norm(t))
        if n < 1e-8:
            return np.array([1.0, 0.0, 0.0])
        return t / n

    def _path_yaw(self, s: float) -> float:
        """Desired heading (degrees) from path tangent at arc-length s."""
        tan = self._path_tangent(s)
        return math.degrees(math.atan2(float(tan[1]), float(tan[0])))
