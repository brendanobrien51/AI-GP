"""
PnP Localizer — Swift-style Gate-Based Position Fix
====================================================
Given:
  - A gate bounding box detected by YOLOv8 (x1, y1, x2, y2)
  - The gate's known 3D world position (from AirSim pre-race)
  - The drone's current orientation estimate

Uses OpenCV's solvePnP to recover the drone's world position.
The uncertainty of the estimate decreases quadratically with distance
(as described in the Swift paper): closer gate = more reliable fix.

Usage:
    pnp = PnPLocalizer(gate_half_size_m=0.75, img_w=640, img_h=480,
                       fov_h_deg=90.0)
    result = pnp.localize(
        bbox_xyxy      = (x1, y1, x2, y2),
        gate_world_pos = np.array([gx, gy, gz]),
        drone_quat_wxyz= np.array([w, x, y, z]),
    )
    if result["success"]:
        drone_pos = result["pos"]     # [x, y, z] world NED
        dist      = result["distance_m"]
        cov       = result["cov"]     # 3×3 measurement covariance
"""

import math
import numpy as np
import cv2


class PnPLocalizer:
    """
    Perspective-n-Point drone localizer using detected gate bounding boxes.

    The gate is modelled as a square planar target (4 corners), which gives
    4 point correspondences to solvePnP.  This is a planar configuration
    so we use SOLVEPNP_IPPE (or ITERATIVE as fallback) for stability.
    """

    def __init__(
        self,
        gate_half_size_m: float = 0.75,
        img_w: int = 640,
        img_h: int = 480,
        fov_h_deg: float = 90.0,
    ):
        """
        Args:
            gate_half_size_m: Half the gate opening width/height in metres.
                Soccer Field gates are ~1.5 m square → half = 0.75 m.
            img_w, img_h: Camera image dimensions in pixels.
            fov_h_deg: Horizontal field-of-view in degrees (from config.yaml).
        """
        self._half = float(gate_half_size_m)

        # ---- Camera intrinsics (pinhole, no distortion) ----
        # fx = (img_w/2) / tan(fov_h/2)
        fov_h_rad = math.radians(fov_h_deg)
        fx = (img_w / 2.0) / math.tan(fov_h_rad / 2.0)
        fy = fx  # square pixels
        cx = img_w / 2.0
        cy = img_h / 2.0

        self._K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)
        self._dist = np.zeros(4, dtype=np.float64)

        # ---- 3D gate corners in gate-local frame (X-right, Y-down, Z-fwd) ----
        # Order: top-left, top-right, bottom-right, bottom-left
        h = self._half
        self._obj_pts = np.array([
            [-h, -h, 0],
            [ h, -h, 0],
            [ h,  h, 0],
            [-h,  h, 0],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    def localize(
        self,
        bbox_xyxy: tuple,
        gate_world_pos: np.ndarray,
        drone_quat_wxyz: np.ndarray,
    ) -> dict:
        """
        Estimate drone world position from a single gate detection.

        Args:
            bbox_xyxy: (x1, y1, x2, y2) bounding box from YOLO in pixels.
            gate_world_pos: [gx, gy, gz] gate centre in world NED (m).
            drone_quat_wxyz: [w, x, y, z] drone orientation estimate.

        Returns:
            {
                "success":    bool,
                "pos":        np.ndarray [x, y, z] drone world NED (m),
                "distance_m": float  — camera-to-gate distance,
                "cov":        np.ndarray 3×3 position measurement covariance,
                "tvec":       np.ndarray [tx, ty, tz] camera-to-gate vector,
            }
        """
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]

        # 2D image corners corresponding to 3D gate corners
        img_pts = np.array([
            [x1, y1],   # top-left
            [x2, y1],   # top-right
            [x2, y2],   # bottom-right
            [x1, y2],   # bottom-left
        ], dtype=np.float64)

        # Guard against degenerate boxes
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            return {"success": False, "pos": None, "distance_m": float("inf"),
                    "cov": None, "tvec": None}

        # ---- Solve PnP ----
        try:
            ok, rvec, tvec = cv2.solvePnP(
                self._obj_pts, img_pts,
                self._K, self._dist,
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error:
            ok = False

        if not ok:
            return {"success": False, "pos": None, "distance_m": float("inf"),
                    "cov": None, "tvec": None}

        tvec = tvec.flatten()            # camera-to-gate in camera frame
        dist = float(np.linalg.norm(tvec))

        # ---- Convert camera-frame tvec → world-frame drone position ----
        # drone_pos_world = gate_world - R_body2world @ R_cam2body @ tvec
        # Camera is aligned with body forward (X), right (Y), down (Z).
        # AirSim camera frame: X-right, Y-down, Z-forward
        # We need to rotate from camera frame to body NED frame.
        R_cam2body = np.array([
            [0, 0, 1],   # body X (forward) ← camera Z (forward)
            [1, 0, 0],   # body Y (right)   ← camera X (right)
            [0, 1, 0],   # body Z (down)    ← camera Y (down)
        ], dtype=np.float64)

        R_body2world = _quat_to_rot(drone_quat_wxyz)
        # gate_world = drone_world + R_body2world @ R_cam2body @ tvec
        # → drone_world = gate_world - R_body2world @ R_cam2body @ tvec
        tvec_body  = R_cam2body  @ tvec
        tvec_world = R_body2world @ tvec_body
        drone_pos  = np.array(gate_world_pos, dtype=np.float64) - tvec_world

        # ---- Uncertainty: scales as distance² (Swift paper Eq. 1) ----
        # σ² = k * d²  where k is a tunable constant
        sigma2 = (dist ** 2) * 0.005
        cov = np.eye(3) * sigma2

        return {
            "success":    True,
            "pos":        drone_pos,
            "distance_m": dist,
            "cov":        cov,
            "tvec":       tvec,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3×3 rotation matrix (body → world)."""
    w, x, y, z = q_wxyz / np.linalg.norm(q_wxyz)
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),   1-2*(x*x+y*y)],
    ], dtype=np.float64)
