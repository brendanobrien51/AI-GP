"""
AirSim Drone Racing Lab - Advanced Autonomous Gate Racer v16
===========================================================
V16 CHANGES vs V14:
  1. YAML config loading - all parameters loaded from config.yaml
  2. Hardware abstraction layer - DroneInterface replaces direct airsim calls
  3. Predictive speed profiling - decel_start_dist computed at path-build time
  4. Dynamic CV search window - sky/floor masks and min-area scale with distance
  5. Telemetry integration - LapTelemetry feeds optimizer feedback
  6. --headless flag - suppress OpenCV windows for optimizer runs
  7. --telemetry_out - write JSON result for Optuna optimizer
"""
import argparse
import json
import math
import re
import time
import cv2
import numpy as np
import yaml

from hardware import AirSimInterface, HardwareInterface, DroneInterface
from telemetry import LapTelemetry
from gate_detector import GateDetector
from mpcc_controller import MPCCController
from vio_estimator import VIOEstimator
from pnp_localizer import PnPLocalizer
from kalman_fusion import SwiftKalmanFilter

try:
    from stable_baselines3 import PPO as _SB3_PPO
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONFIGURATION LOADING
# ---------------------------------------------------------------------------
class Cfg:
    """Configuration object - loads all parameters from YAML at startup."""
    def __init__(self, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        for k, v in d.items():
            setattr(self, k, v)

def parse_args():
    p = argparse.ArgumentParser(
        description="AirSim Drone Racing Lab - V16 Autonomous Gate Racer"
    )
    p.add_argument("--config", default="config.yaml",
                   help="YAML config file with all tuning parameters")
    p.add_argument("--hardware", action="store_true",
                   help="Use HardwareInterface instead of AirSimInterface")
    p.add_argument("--headless", action="store_true",
                   help="Suppress OpenCV dashboard (for optimizer runs)")
    p.add_argument("--telemetry_out", default=None,
                   help="Write JSON lap summary to this path and exit")
    p.add_argument("--no-yolo", action="store_true",
                   help="Disable YOLOv8 detector, use contour detection only")
    return p.parse_args()

# Dashboard colors (shared across all modes)
C_WHITE   = (255, 255, 255)
C_BLACK   = (  0,   0,   0)
C_GREEN   = (  0, 220,  80)
C_YELLOW  = (  0, 220, 220)
C_ORANGE  = (  0, 165, 255)
C_RED     = ( 50,  50, 240)
C_CYAN    = (230, 220,   0)
C_GRAY    = (120, 120, 120)
C_DGRAY   = ( 40,  40,  40)
C_BLUE    = (230, 100,   0)
C_LIME    = ( 50, 255, 100)
C_MAGENTA = (200,  50, 200)
PHASE_COLORS = {"APPROACH": C_YELLOW, "CENTER": C_GREEN, "EXIT": C_ORANGE}

# Dashboard layout (constant across runs)
DASH_W, DASH_H = 1280, 720
CAM_W          = 854
RIGHT_W        = DASH_W - CAM_W
MAP_H          = 380
BRAIN_H        = DASH_H - MAP_H

# HSV color ranges for gate detection (fixed, not tunable)
HSV_RANGES = [
    (np.array([  5,  80,  80], dtype=np.uint8), np.array([ 25, 255, 255], dtype=np.uint8)),
    (np.array([ 18, 100, 100], dtype=np.uint8), np.array([ 38, 255, 255], dtype=np.uint8)),
    (np.array([  0,  80,  80], dtype=np.uint8), np.array([  8, 255, 255], dtype=np.uint8)),
    (np.array([172,  80,  80], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
]

# ---------------------------------------------------------------------------
# MATH HELPERS
# ---------------------------------------------------------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def vec3(x, y, z):
    return np.array([float(x), float(y), float(z)], dtype=np.float64)

def norm(v):
    return float(np.linalg.norm(v))

def unit(v):
    n = norm(v)
    return v / n if n > 1e-7 else np.zeros(3, dtype=np.float64)

def lerp(a, b, t):
    t = clamp(t, 0.0, 1.0)
    return a * (1.0 - t) + b * t

def angle_lerp(a, b, t):
    t = clamp(t, 0.0, 1.0)
    diff = math.atan2(math.sin(b - a), math.cos(b - a))
    return a + diff * t

def quat_to_yaw(q):
    siny = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return math.atan2(siny, cosy)

def quat_to_rot_matrix(q):
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ], dtype=np.float64)

class SmoothQuat:
    """Temporal low-pass filter on quaternion orientation.
    Uses component-wise lerp + renormalize (good approximation of SLERP
    for small frame-to-frame changes).  This keeps the guideline locked
    to the track without the jitter from raw drone attitude."""
    def __init__(self, alpha=0.12):
        self._w = 1.0
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._alpha = alpha
        self._init = False

    def update(self, q):
        nw, nx, ny, nz = q.w_val, q.x_val, q.y_val, q.z_val
        if not self._init:
            self._w, self._x, self._y, self._z = nw, nx, ny, nz
            self._init = True
            return
        # Ensure shortest path (avoid sign flip)
        dot = self._w*nw + self._x*nx + self._y*ny + self._z*nz
        if dot < 0:
            nw, nx, ny, nz = -nw, -nx, -ny, -nz
        a = self._alpha
        self._w = self._w*(1-a) + nw*a
        self._x = self._x*(1-a) + nx*a
        self._y = self._y*(1-a) + ny*a
        self._z = self._z*(1-a) + nz*a
        n = math.sqrt(self._w**2 + self._x**2 + self._y**2 + self._z**2)
        if n > 1e-8:
            self._w /= n; self._x /= n; self._y /= n; self._z /= n

    @property
    def w_val(self): return self._w
    @property
    def x_val(self): return self._x
    @property
    def y_val(self): return self._y
    @property
    def z_val(self): return self._z

def pose_to_np(pose):
    p = pose.position
    return vec3(p.x_val, p.y_val, p.z_val)

def state_pos(state):
    p = state.kinematics_estimated.position
    return vec3(p.x_val, p.y_val, p.z_val)

def state_vel(state):
    v = state.kinematics_estimated.linear_velocity
    return vec3(v.x_val, v.y_val, v.z_val)

# ---------------------------------------------------------------------------
# GATE DISCOVERY & PATH PLANNING
# ---------------------------------------------------------------------------
def get_gate_names(iface):
    """Extract gate names from iface.get_gate_poses()."""
    poses = iface.get_gate_poses()
    return [name for name, _ in poses]

def get_object_pose_safe(iface, name):
    """Get pose of a single gate from iface."""
    poses = iface.get_gate_poses()
    for gname, pos in poses:
        if gname == name:
            return type('Pose', (), {
                'position': type('Pos', (), {
                    'x_val': float(pos[0]),
                    'y_val': float(pos[1]),
                    'z_val': float(pos[2])
                })()
            })()

def build_path(iface, gate_names, cfg):
    gates = []
    for name in gate_names:
        pose = get_object_pose_safe(iface, name)
        if pose is None:
            print(f"[WARNING] Gate {name} pose is None, skipping.")
            continue
        pos  = pose_to_np(pose)
        if np.isfinite(pos).all():
            gates.append((name, pos.copy()))
    path  = []
    total = len(gates)
    for i, (name, center) in enumerate(gates):
        if i == 0:
            fwd = unit(gates[1][1] - center) if total > 1 else vec3(1, 0, 0)
        elif i == total - 1:
            fwd = unit(center - gates[i - 1][1])
        else:
            fwd = unit(gates[i + 1][1] - gates[i - 1][1])
        if norm(fwd) < 1e-6:
            fwd = vec3(1, 0, 0)
        if i == 0:
            base_app = cfg.first_approach_dist
        else:
            gap      = norm(center - gates[i - 1][1])
            base_app = max(1.5, min(cfg.approach_dist, gap * cfg.dyn_approach_frac))
        path.append({
            "index":    i,
            "name":     name,
            "center":   center.copy(),
            "approach": center - fwd * base_app,
            "exit":     center + fwd * cfg.exit_dist,
            "forward":  fwd,
        })
    # Compute turn angle at each gate and derive speed multiplier
    for i, g in enumerate(path):
        if i == 0 or i == len(path) - 1:
            g["turn_angle"] = 0.0
            g["turn_speed_mult"] = cfg.turn_speed_straight
        else:
            v_in  = unit(g["center"] - path[i - 1]["center"])
            v_out = unit(path[i + 1]["center"] - g["center"])
            dot   = clamp(float(np.dot(v_in, v_out)), -1.0, 1.0)
            angle = math.degrees(math.acos(dot))  # 0 = straight, 180 = U-turn
            g["turn_angle"] = angle
            # Interpolate speed multiplier based on angle
            if angle <= 0.1:
                g["turn_speed_mult"] = cfg.turn_speed_straight
            elif angle <= 90.0:
                t = angle / 90.0
                g["turn_speed_mult"] = cfg.turn_speed_straight + t * (cfg.turn_speed_90deg - cfg.turn_speed_straight)
            else:
                t = (angle - 90.0) / 90.0
                g["turn_speed_mult"] = cfg.turn_speed_90deg + t * (cfg.turn_speed_180deg - cfg.turn_speed_90deg)
    # Print turn analysis
    for g in path:
        print(f"  Gate {g['index']+1:>2}: {g['name']:<14} "
              f"turn={g['turn_angle']:5.1f}deg  "
              f"speed_mult={g['turn_speed_mult']:.2f}")

    # Compute predictive speed profile
    compute_predictive_speed_profile(path, cfg)
    return path

# ---------------------------------------------------------------------------
# PREDICTIVE SPEED PROFILING
# ---------------------------------------------------------------------------
def compute_predictive_speed_profile(path, cfg):
    """Compute decel_start_dist for each gate based on kinematics."""
    for g in path:
        mult = g["turn_speed_mult"]
        g["required_speed"] = cfg.max_approach_speed * mult

    for i in range(len(path)):
        v_target = path[i]["required_speed"]
        v_prev = 0.0 if i == 0 else path[i - 1]["required_speed"]

        if v_target < v_prev:
            path[i]["decel_start_dist"] = (
                (v_prev ** 2 - v_target ** 2) / (2.0 * cfg.max_decel)
            )
        else:
            path[i]["decel_start_dist"] = 0.0

    for g in path:
        print(f"  Gate {g['index']+1:>2}: req_speed={g['required_speed']:.1f} "
              f"decel_start_dist={g['decel_start_dist']:.1f}m")

def compute_predictive_max_speed(drone_pos, gate_idx, path, cfg):
    """Scan upcoming gates for deceleration constraints."""
    predictive_cap = float("inf")
    for offset in range(1, cfg.predictive_lookahead_gates + 1):
        look_idx = gate_idx + offset
        if look_idx >= len(path):
            break
        g = path[look_idx]
        dist_to_gate = float(np.linalg.norm(g["center"] - drone_pos))
        if dist_to_gate <= g["decel_start_dist"]:
            predictive_cap = min(predictive_cap, g["required_speed"])
    return predictive_cap

# ---------------------------------------------------------------------------
# RACING LINE
# ---------------------------------------------------------------------------
def build_racing_line(path):
    line = []
    for g in path:
        line.append(g["approach"].copy())
        line.append(g["center"].copy())
        line.append(g["exit"].copy())
    return line

def find_closest_segment(line, pos, min_seg=0):
    best_dist = float("inf")
    best_idx  = min_seg
    best_t    = 0.0
    for i in range(min_seg, len(line) - 1):
        a  = line[i]
        ab = line[i + 1] - a
        ab_sq = float(np.dot(ab, ab))
        if ab_sq < 1e-8:
            t = 0.0
        else:
            t = clamp(float(np.dot(pos - a, ab)) / ab_sq, 0.0, 1.0)
        d = norm(pos - (a + ab * t))
        if d < best_dist:
            best_dist = d
            best_idx  = i
            best_t    = t
    return best_idx, best_t, best_dist

def get_lookahead_point(line, seg_idx, seg_t, lookahead_dist):
    if seg_idx >= len(line) - 1:
        return line[-1].copy()
    a, b    = line[seg_idx], line[seg_idx + 1]
    seg_len = norm(b - a)
    remain  = seg_len * (1.0 - seg_t)
    if lookahead_dist <= remain and seg_len > 1e-7:
        return lerp(a, b, seg_t + lookahead_dist / seg_len)
    dist_left = lookahead_dist - remain
    for i in range(seg_idx + 1, len(line) - 1):
        s = norm(line[i + 1] - line[i])
        if dist_left <= s and s > 1e-7:
            return lerp(line[i], line[i + 1], dist_left / s)
        dist_left -= s
    return line[-1].copy()

def sample_path_ahead(line, seg_idx, seg_t, total_dist, n_pts):
    pts  = []
    step = total_dist / max(n_pts, 1)
    for i in range(n_pts + 1):
        pts.append(get_lookahead_point(line, seg_idx, seg_t, step * i))
    return pts

# ---------------------------------------------------------------------------
# CAMERA PROJECTION
# ---------------------------------------------------------------------------
def project_points_to_camera(world_pts, drone_pos, drone_quat,
                              disp_w, disp_h, fov_h_deg=90.0):
    R  = quat_to_rot_matrix(drone_quat)
    Rt = R.T
    fx = disp_w / (2.0 * math.tan(math.radians(fov_h_deg / 2.0)))
    fy = fx
    cx = disp_w / 2.0
    cy = disp_h / 2.0
    result = []
    for pt in world_pts:
        local = Rt @ (pt - drone_pos)
        if local[0] < 0.3:
            result.append(None)
            continue
        px = int(cx + local[1] / local[0] * fx)
        py = int(cy + local[2] / local[0] * fy)
        result.append((px, py))
    return result

# ---------------------------------------------------------------------------
# DYNAMIC CV SEARCH WINDOW
# ---------------------------------------------------------------------------
def compute_cv_window(dist_to_gate, cfg):
    """Scale sky/floor masks and min_area by gate distance."""
    d_near = cfg.dist_near   # 8.0 m
    d_far  = cfg.dist_far    # 15.0 m
    d      = float(np.clip(dist_to_gate, 0.0, d_far * 2))

    if d <= d_near:
        t = 0.0
    elif d >= d_far:
        t = 1.0
    else:
        t = (d - d_near) / (d_far - d_near)

    sky   = lerp(cfg.sky_mask_near, cfg.sky_mask_far, t)
    floor = lerp(cfg.floor_mask_near, cfg.floor_mask_far, t)
    area  = int(lerp(cfg.contour_min_area, cfg.contour_min_area_far, t))
    return sky, floor, area

def apply_search_window(frame, sky_frac, floor_frac):
    """Zero out sky and floor regions."""
    h = frame.shape[0]
    masked = frame.copy()
    sky_rows = int(h * sky_frac)
    floor_rows = int(h * floor_frac)
    if sky_rows > 0:
        masked[:sky_rows, :] = 0
    if floor_rows > 0:
        masked[h - floor_rows:, :] = 0
    return masked

# ---------------------------------------------------------------------------
# COMPUTER VISION
# ---------------------------------------------------------------------------
def multi_hsv_mask(frame, cfg):
    hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))
    k_close  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k_open   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)
    return combined

def edge_mask(frame, cfg):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)
    return cv2.dilate(edges, None, iterations=1)

def score_contour(contour, frame_shape, min_area, cfg):
    area = cv2.contourArea(contour)
    if area < min_area:
        return None
    fh, fw = frame_shape[:2]
    frac   = area / (fh * fw)
    if frac < 0.001 or frac > 0.65:
        return None
    rect   = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    if min(rw, rh) < 5:
        return None
    rect_area      = rw * rh
    rectangularity = area / rect_area if rect_area > 1 else 0.0
    aspect         = max(rw, rh) / (min(rw, rh) + 1e-6)
    if aspect > 5.0:
        return None
    aspect_score   = 1.0 if aspect < 1.8 else max(0.25, 1.0 - (aspect - 1.8) * 0.18)
    hull           = cv2.convexHull(contour)
    hull_area      = cv2.contourArea(hull)
    solidity       = area / hull_area if hull_area > 1 else 0.0
    solidity_score = 0.5 + 0.5 * solidity
    score = rectangularity * aspect_score * solidity_score
    if score < cfg.cv_score_threshold:
        return None
    cx, cy = int(rect[0][0]), int(rect[0][1])
    return {
        "score": score, "rect_score": rectangularity,
        "aspect_score": aspect_score, "solidity": solidity,
        "area_frac": frac, "aspect": aspect,
        "centroid_px": (cx, cy), "contour": contour, "rect": rect,
    }

def find_best_gate_contour(hsv_mask, edge_mask_img, frame_shape, min_area, cfg):
    merged      = cv2.bitwise_or(hsv_mask, edge_mask_img)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        result = score_contour(c, frame_shape, min_area, cfg)
        if result is not None:
            if best is None or result["score"] > best["score"]:
                best = result
    return best

class GateTracker:
    def __init__(self, cfg):
        self._best = None
        self._t = 0.0
        self.frames_lost = 0
        self._cfg = cfg
    def update(self, detection):
        if detection is not None:
            self._best = detection
            self._t = time.time()
            self.frames_lost = 0
        else:
            self.frames_lost += 1
    @property
    def current(self):
        if self._best is None:
            return None
        if time.time() - self._t > self._cfg.tracker_stale_s:
            return None
        return self._best
    def reset(self):
        self._best = None
        self._t = 0.0
        self.frames_lost = 0

# ---------------------------------------------------------------------------
# CV STEERING
# ---------------------------------------------------------------------------
def cv_steer_correction(tracker, frame_shape, gate_forward, phase,
                        in_commit_zone, cfg, cur_speed=0.0):
    det = tracker.current
    if det is None:
        return np.zeros(3, dtype=np.float64), 0.0, 0.0
    fh, fw = frame_shape[:2]
    cx, cy = det["centroid_px"]
    nx = (cx - fw / 2.0) / (fw / 2.0)
    ny = (cy - fh / 2.0) / (fh / 2.0)
    if phase == "CENTER":
        gain = cfg.cv_steer_gain_center
    elif phase == "APPROACH":
        gain = cfg.cv_steer_gain_approach
    else:
        gain = cfg.cv_steer_gain_exit
    # Speed-proportional: more force at higher speed
    gain *= (1.0 + cur_speed / cfg.cv_steer_speed_scale)
    if in_commit_zone:
        gain *= cfg.commit_lat_damp
    down  = vec3(0, 0, 1)
    right = np.cross(gate_forward, down)
    r_n   = norm(right)
    if r_n < 1e-6:
        right = vec3(0, 1, 0)
    else:
        right = right / r_n
    correction = right * nx * gain + vec3(0, 0, ny * gain * 0.5)
    mag = norm(correction)
    if mag > cfg.cv_steer_max:
        correction = correction / mag * cfg.cv_steer_max
    return correction, nx, ny

# ---------------------------------------------------------------------------
# STUCK DETECTOR
# ---------------------------------------------------------------------------
class StuckDetector:
    def __init__(self, cfg):
        self._history = []
        self._cfg = cfg
    def update(self, progress):
        now = time.time()
        self._history.append((now, progress))
        cutoff = now - self._cfg.stuck_window
        self._history = [(t, p) for t, p in self._history if t >= cutoff]
    def is_stuck(self):
        if len(self._history) < 10:
            return False
        span = self._history[-1][0] - self._history[0][0]
        if span < self._cfg.stuck_window * 0.75:
            return False
        return (self._history[-1][1] - self._history[0][1]) < self._cfg.stuck_min_progress
    def reset(self):
        self._history = []

# ---------------------------------------------------------------------------
# FLIGHT CONTROL HELPERS
# ---------------------------------------------------------------------------
def lateral_correction_vec(drone_pos, gate_center, gate_forward, gain, max_mag):
    to_gate   = gate_center - drone_pos
    fwd_proj  = float(np.dot(to_gate, gate_forward))
    lat_error = to_gate - gate_forward * fwd_proj
    corr      = lat_error * gain
    mag       = norm(corr)
    if mag > max_mag:
        corr = corr / mag * max_mag
    return corr

def compute_speed_scale(yaw_err_rad, dist_to_target, phase, cfg):
    yaw_s = max(cfg.min_speed_fraction,
                math.cos(clamp(abs(yaw_err_rad), 0, math.pi / 2))
                ** cfg.yaw_scale_exponent)
    if phase == "CENTER":
        return yaw_s
    brake_s = (max(cfg.brake_min_fraction, dist_to_target / cfg.brake_dist)
               if dist_to_target < cfg.brake_dist else 1.0)
    return min(yaw_s, brake_s)

def signed_progress(drone_pos, gate_center, gate_forward):
    return float(np.dot(drone_pos - gate_center, gate_forward))

def altitude_cmd(drone_z, target_z, phase, cfg):
    z_err = target_z - drone_z
    kp    = cfg.alt_kp_center if phase == "CENTER" else cfg.alt_kp
    return clamp(z_err * kp, -cfg.alt_max_z_vel, cfg.alt_max_z_vel)

# ---------------------------------------------------------------------------
# DASHBOARD RENDERING
# ---------------------------------------------------------------------------
class TopDownMap:
    def __init__(self, gate_centers, pw, ph, margin=28):
        xs = [g[0] for g in gate_centers]
        ys = [g[1] for g in gate_centers]
        span_x = max(max(xs) - min(xs), 5.0)
        span_y = max(max(ys) - min(ys), 5.0)
        uw = pw - 2 * margin
        uh = ph - 2 * margin
        self.scale = min(uw / span_x, uh / span_y)
        self.ox = margin + (uw - span_x * self.scale) / 2 - min(xs) * self.scale
        self.oy = margin + (uh - span_y * self.scale) / 2 - min(ys) * self.scale
        self.pw, self.ph = pw, ph
    def to_px(self, wx, wy):
        return (clamp(int(wx * self.scale + self.ox), 0, self.pw - 1),
                clamp(int(wy * self.scale + self.oy), 0, self.ph - 1))

def draw_top_down_map(path, gate_idx, drone_pos, drone_yaw, target,
                      racing_line, rl_seg_idx):
    panel = np.full((MAP_H, RIGHT_W, 3), (18, 18, 28), dtype=np.uint8)
    if not path:
        return panel
    tdm = TopDownMap([g["center"] for g in path], RIGHT_W, MAP_H)
    # Racing line
    for i in range(max(0, rl_seg_idx), len(racing_line) - 1):
        p1 = tdm.to_px(racing_line[i][0], racing_line[i][1])
        p2 = tdm.to_px(racing_line[i + 1][0], racing_line[i + 1][1])
        cv2.line(panel, p1, p2, (40, 80, 40), 1, cv2.LINE_AA)
    # Gate markers
    for i, g in enumerate(path):
        px, py = tdm.to_px(g["center"][0], g["center"][1])
        is_cur = (i == gate_idx)
        col = C_CYAN if is_cur else ((60, 160, 60) if i < gate_idx else C_GRAY)
        fwd  = g["forward"]
        perp = np.array([-fwd[1], fwd[0]])
        half = max(4, int((10 if is_cur else 6) / tdm.scale))
        a = tdm.to_px(g["center"][0] + perp[0] * half / tdm.scale,
                       g["center"][1] + perp[1] * half / tdm.scale)
        b = tdm.to_px(g["center"][0] - perp[0] * half / tdm.scale,
                       g["center"][1] - perp[1] * half / tdm.scale)
        cv2.line(panel, a, b, col, 3 if is_cur else 2, cv2.LINE_AA)
        arw = tdm.to_px(g["center"][0] + fwd[0] * 2.0,
                         g["center"][1] + fwd[1] * 2.0)
        cv2.arrowedLine(panel, (px, py), arw, col, 1,
                         tipLength=0.4, line_type=cv2.LINE_AA)
        cv2.putText(panel, str(i + 1), (px + 5, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)
    # Target
    tx, ty = tdm.to_px(target[0], target[1])
    cv2.drawMarker(panel, (tx, ty), C_ORANGE, cv2.MARKER_DIAMOND, 10, 2)
    # Drone
    dx, dy  = tdm.to_px(drone_pos[0], drone_pos[1])
    tl = 11
    tip   = (int(dx + math.cos(drone_yaw) * tl), int(dy + math.sin(drone_yaw) * tl))
    left  = (int(dx + math.cos(drone_yaw + 2.4) * 6), int(dy + math.sin(drone_yaw + 2.4) * 6))
    right = (int(dx + math.cos(drone_yaw - 2.4) * 6), int(dy + math.sin(drone_yaw - 2.4) * 6))
    cv2.fillPoly(panel, [np.array([tip, left, right])], C_RED)
    cv2.polylines(panel, [np.array([tip, left, right])], True, C_WHITE, 1)
    cv2.putText(panel, "TOP-DOWN MAP", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)
    return panel

def _bar(panel, x, y, w, h, frac, fg, label="", val=""):
    frac = clamp(frac, 0.0, 1.0)
    cv2.rectangle(panel, (x, y), (x + w, y + h), C_DGRAY, -1)
    cv2.rectangle(panel, (x, y), (x + max(2, int(w * frac)), y + h), fg, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), C_GRAY, 1)
    if label:
        cv2.putText(panel, label, (x - 2, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_GRAY, 1)
    if val:
        cv2.putText(panel, val, (x + w + 4, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_WHITE, 1)

def _cbar(panel, x, y, w, h, frac, fg, label="", val=""):
    frac = clamp(frac, -1.0, 1.0)
    mid  = x + w // 2
    cv2.rectangle(panel, (x, y), (x + w, y + h), C_DGRAY, -1)
    fw = int(w / 2 * abs(frac))
    if frac >= 0:
        cv2.rectangle(panel, (mid, y), (mid + fw, y + h), fg, -1)
    else:
        cv2.rectangle(panel, (mid - fw, y), (mid, y + h), fg, -1)
    cv2.line(panel, (mid, y), (mid, y + h), C_GRAY, 1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), C_GRAY, 1)
    if label:
        cv2.putText(panel, label, (x - 2, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_GRAY, 1)
    if val:
        cv2.putText(panel, val, (x + w + 4, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_WHITE, 1)

def draw_brain_panel(phase, gate_idx, total, gate_name, cfg,
                     speed, max_speed, yaw_err_deg, lat_err_mag,
                     gate_progress, dist_gate, detection,
                     vel, elapsed_s, stuck_det, z_err_m,
                     cv_nx, cv_ny, in_commit, lookahead_d):
    panel = np.full((BRAIN_H, RIGHT_W, 3), (14, 14, 22), dtype=np.uint8)
    px    = 10
    bar_x = 92
    bar_w = RIGHT_W - bar_x - 54
    row   = 25
    y     = 24
    cv2.putText(panel, "BRAIN STATE", (px, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)
    pcol  = PHASE_COLORS.get(phase, C_WHITE)
    badge = f"  {phase}  "
    if in_commit:
        badge = " COMMIT "
        pcol  = C_MAGENTA
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
    bx = RIGHT_W - bw - px - 4
    cv2.rectangle(panel, (bx - 4, 2), (bx + bw + 2, 2 + bh + 6), pcol, -1)
    cv2.putText(panel, badge, (bx, 2 + bh + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, C_BLACK, 2)
    cv2.putText(panel, f"Gate {gate_idx+1}/{total}  {gate_name}",
                (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_WHITE, 1)
    y += 12
    cv2.line(panel, (px, y), (RIGHT_W - px, y), C_DGRAY, 1)
    y += 8
    _bar(panel, bar_x, y, bar_w, 13, speed / max(max_speed, 0.1),
         C_GREEN, "SPEED", f"{speed:.1f}/{max_speed:.0f}")
    y += row
    _cbar(panel, bar_x, y, bar_w, 13, yaw_err_deg / 90.0,
          C_ORANGE, "YAW ERR", f"{yaw_err_deg:+.1f}d")
    y += row
    lat_f = min(lat_err_mag / 2.5, 1.0)
    _bar(panel, bar_x, y, bar_w, 13, lat_f,
         C_RED if lat_f > 0.6 else C_YELLOW, "LAT ERR", f"{lat_err_mag:.2f}m")
    y += row
    _cbar(panel, bar_x, y, bar_w, 13, clamp(gate_progress / 4.0, -1, 1),
          C_GREEN if gate_progress > 0 else C_YELLOW,
          "PROGRESS", f"{gate_progress:+.2f}m")
    y += row
    _cbar(panel, bar_x, y, bar_w, 13, clamp(z_err_m / 3.0, -1, 1),
          C_BLUE, "ALT ERR", f"{z_err_m:+.2f}m")
    y += row
    _bar(panel, bar_x, y, bar_w, 13,
         clamp(1.0 - dist_gate / 25.0, 0, 1), C_BLUE,
         "DIST", f"{dist_gate:.1f}m")
    y += row
    _bar(panel, bar_x, y, bar_w, 13,
         clamp(lookahead_d / cfg.lookahead_max, 0, 1), C_LIME,
         "LOOK-D", f"{lookahead_d:.1f}m")
    y += row
    _cbar(panel, bar_x, y, bar_w, 13, clamp(cv_nx, -1, 1),
          C_CYAN, "CV STEER", f"x:{cv_nx:+.2f} y:{cv_ny:+.2f}")
    y += row
    if detection is not None:
        d = detection
        cv_col = C_GREEN if d["score"] > 0.65 else (
                 C_YELLOW if d["score"] > 0.45 else C_RED)
        _bar(panel, bar_x, y, bar_w, 13, d["score"], cv_col,
             "CV TOTAL", f"{d['score']:.2f}")
        y += 20
    else:
        cv2.putText(panel, "CV: NO DETECTION",
                    (bar_x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_RED, 1)
        y += 20
    cv2.line(panel, (px, y), (RIGHT_W - px, y), C_DGRAY, 1)
    y += 8
    stuck_col = C_RED if stuck_det.is_stuck() else C_DGRAY
    cv2.rectangle(panel, (px, y), (px + 80, y + 14), stuck_col, -1)
    cv2.putText(panel, "STUCK" if stuck_det.is_stuck() else "FLOWING",
                (px + 4, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1)
    spd3 = norm(vel)
    cv2.putText(panel,
                f"Vx:{vel[0]:+5.1f} Vy:{vel[1]:+5.1f} Vz:{vel[2]:+5.1f}  |V|={spd3:.1f}",
                (px, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_CYAN, 1)
    cv2.putText(panel, f"gate_t={elapsed_s:.1f}s",
                (px, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GRAY, 1)
    return panel

def draw_camera_panel(frame, tracker, phase, gate_name, cfg, drone_vel,
                       yaw_err_deg, lat_corr, in_commit,
                       path_px, gate_center_px_list, gate_labels,
                       target_px):
    cam    = cv2.resize(frame, (CAM_W, DASH_H), interpolation=cv2.INTER_LINEAR)
    h, w   = cam.shape[:2]
    cw, ch = w // 2, h // 2
    margin = 80

    # ---- PATH GUIDELINE with gradient ----
    n_pts = len(path_px)
    # Draw segments with color gradient green -> yellow
    for i in range(n_pts - 1):
        p1 = path_px[i]
        p2 = path_px[i + 1]
        if p1 is None or p2 is None:
            continue
        if (p1[0] < -margin or p1[0] > w + margin or
            p1[1] < -margin or p1[1] > h + margin):
            continue
        if (p2[0] < -margin or p2[0] > w + margin or
            p2[1] < -margin or p2[1] > h + margin):
            continue
        t   = i / max(n_pts - 1, 1)
        # green (0,220,80) -> yellow (0,220,220)
        col = (0, 220, int(80 + 140 * t))
        thickness = max(1, 3 - int(t * 2))
        cv2.line(cam, p1, p2, col, thickness, cv2.LINE_AA)

    # Dots along guideline (every 5th point)
    for i, pt in enumerate(path_px):
        if pt is not None and 0 <= pt[0] < w and 0 <= pt[1] < h:
            if i % 5 == 0:
                t   = i / max(n_pts - 1, 1)
                col = (0, 220, int(80 + 140 * t))
                cv2.circle(cam, pt, 2, col, -1, cv2.LINE_AA)

    # Target diamond (lookahead point)
    if target_px is not None and 0 <= target_px[0] < w and 0 <= target_px[1] < h:
        cv2.drawMarker(cam, target_px, C_CYAN, cv2.MARKER_DIAMOND, 14, 2)

    # Gate center markers with labels
    for i, gpt in enumerate(gate_center_px_list):
        if gpt is not None and -20 < gpt[0] < w + 20 and -20 < gpt[1] < h + 20:
            cv2.circle(cam, gpt, 10, C_CYAN, 2, cv2.LINE_AA)
            cv2.circle(cam, gpt, 3, C_CYAN, -1)
            if i < len(gate_labels):
                cv2.putText(cam, gate_labels[i],
                            (gpt[0] + 14, gpt[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_CYAN, 1)

    # ---- CV DETECTION ----
    det = tracker.current
    if det is not None:
        sx = CAM_W / frame.shape[1]
        sy = DASH_H / frame.shape[0]
        box = cv2.boxPoints(det["rect"])
        box[:, 0] *= sx
        box[:, 1] *= sy
        t    = clamp((det["score"] - cfg.cv_score_threshold) /
                     (1.0 - cfg.cv_score_threshold), 0, 1)
        bcol = (int(50 * t), int(200 * t), int(50 + 200 * (1 - t)))
        cv2.drawContours(cam, [np.int32(box)], 0, bcol, 2)
        px = int(det["centroid_px"][0] * sx)
        py = int(det["centroid_px"][1] * sy)
        cv2.circle(cam, (px, py), 7, bcol, -1)
        cv2.circle(cam, (px, py), 7, C_WHITE, 1)
        cv2.line(cam, (cw, ch), (px, py), (0, 200, 120), 1, cv2.LINE_AA)
        stale = tracker.frames_lost > 0
        label = f"CV {det['score']:.2f}" + (" [CACHED]" if stale else "")
        cv2.putText(cam, label, (px + 10, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, bcol, 1)
    else:
        cv2.putText(cam, "NO GATE DETECTED",
                    (cw - 110, ch - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_RED, 2)

    # Crosshair
    cv2.circle(cam, (cw, ch), 32, C_WHITE, 1, cv2.LINE_AA)
    cv2.line(cam, (cw - 24, ch), (cw + 24, ch), C_WHITE, 1)
    cv2.line(cam, (cw, ch - 24), (cw, ch + 24), C_WHITE, 1)

    # Velocity vector
    spd = norm(drone_vel)
    if spd > 0.3:
        sc = 5.5
        ax = int(cw + drone_vel[1] * sc)
        ay = int(ch - drone_vel[2] * sc)
        cv2.arrowedLine(cam, (cw, ch), (ax, ay), C_CYAN, 2,
                         tipLength=0.28, line_type=cv2.LINE_AA)
        cv2.putText(cam, f"{spd:.1f}m/s", (ax + 6, ay - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_CYAN, 1)

    # Lateral bar
    lat_mag = norm(lat_corr[:2])
    if lat_mag > 0.05:
        bar_len = int(clamp(lat_corr[1] * 10, -70, 70))
        col = C_RED if abs(bar_len) > 35 else C_ORANGE
        cv2.arrowedLine(cam, (cw, ch + 45), (cw + bar_len, ch + 45),
                         col, 2, tipLength=0.25, line_type=cv2.LINE_AA)

    # Yaw indicator
    yr = math.radians(yaw_err_deg)
    yaw_tip = (int(cw + math.sin(yr) * 55), int(ch - math.cos(yr) * 55))
    yaw_col = (C_GREEN if abs(yaw_err_deg) < 10 else
               (C_YELLOW if abs(yaw_err_deg) < 30 else C_RED))
    cv2.line(cam, (cw, ch), yaw_tip, yaw_col, 2, cv2.LINE_AA)
    cv2.circle(cam, yaw_tip, 3, yaw_col, -1)

    # Phase badge
    pcol  = PHASE_COLORS.get(phase, C_WHITE)
    badge = f" {phase} "
    if in_commit:
        badge = " COMMIT "
        pcol  = C_MAGENTA
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
    cv2.rectangle(cam, (8, 6), (8 + bw + 4, 6 + bh + 8), pcol, -1)
    cv2.putText(cam, badge, (10, 6 + bh + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, C_BLACK, 2)
    cv2.putText(cam, f"-> {gate_name}",
                (w - 210, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_WHITE, 1)
    if tracker.frames_lost > 0:
        cv2.putText(cam, f"LOST {tracker.frames_lost}f",
                    (w - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_RED, 1)
    return cam

def build_dashboard(frame, tracker, path, gate_idx, phase, gate, cfg,
                    drone_pos, drone_yaw, drone_vel, target,
                    speed, max_speed, speed_scale, yaw_err_deg,
                    lat_err_mag, lat_corr, gate_progress, dist_gate,
                    z_err_m, gate_elapsed, stuck_det, cv_nx, cv_ny,
                    in_commit, lookahead_d, racing_line, rl_seg_idx,
                    path_px, gate_center_px_list, gate_labels,
                    target_px):
    dash = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)
    cam_p = draw_camera_panel(frame, tracker, phase, gate["name"], cfg,
                               drone_vel, yaw_err_deg, lat_corr, in_commit,
                               path_px, gate_center_px_list, gate_labels,
                               target_px)
    dash[:, :CAM_W] = cam_p
    cv2.line(dash, (CAM_W, 0), (CAM_W, DASH_H), C_DGRAY, 1)
    map_p = draw_top_down_map(path, gate_idx, drone_pos, drone_yaw, target,
                              racing_line, rl_seg_idx)
    dash[:MAP_H, CAM_W:] = map_p
    cv2.line(dash, (CAM_W, MAP_H), (DASH_W, MAP_H), C_DGRAY, 1)
    brain_p = draw_brain_panel(
        phase, gate_idx, len(path), gate["name"], cfg,
        speed, max_speed, yaw_err_deg, lat_err_mag,
        gate_progress, dist_gate, tracker.current,
        drone_vel, gate_elapsed, stuck_det, z_err_m,
        cv_nx, cv_ny, in_commit, lookahead_d
    )
    dash[MAP_H:, CAM_W:] = brain_p
    return dash

# ---------------------------------------------------------------------------
# SHUTDOWN
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = Cfg(args.config)

    # Initialize YOLOv8 gate detector (loads gate_detector.pt if available)
    gate_det = GateDetector("gate_detector.pt") if not args.no_yolo else GateDetector.__new__(GateDetector)
    if args.no_yolo:
        gate_det._model = None
        print("[YOLO] Disabled via --no-yolo flag, using contour detection only")
    elif gate_det.available:
        print("[YOLO] Gate detector loaded and ready")
    else:
        print("[YOLO] Model not found, using contour detection only")

    print("Initializing drone interface...")
    if args.hardware:
        iface = HardwareInterface()
    else:
        iface = AirSimInterface()

    print("Arming and taking off...")
    iface.arm_and_takeoff(cfg.takeoff_height)

    print("Getting gate poses...")
    gate_poses = iface.get_gate_poses()
    gate_names = [name for name, _ in gate_poses]

    if not gate_names:
        iface.shutdown()
        raise RuntimeError("No gate objects found in scene.")
    print(f"Found {len(gate_names)} gates.")
    for i, n in enumerate(gate_names):
        print(f"  {i + 1:>2}. {n}")

    path = build_path(iface, gate_names, cfg)
    racing_line = build_racing_line(path)

    # MPCC controller (replaces pure pursuit when cfg.mpcc_enabled)
    mpcc = MPCCController(
        horizon=getattr(cfg, "mpcc_horizon", 10),
        dt=cfg.control_dt,
        w_progress=getattr(cfg, "mpcc_w_progress", 2.0),
        w_lateral=getattr(cfg, "mpcc_w_lateral", 5.0),
        w_speed=getattr(cfg, "mpcc_w_speed", 0.5),
    )
    mpcc.set_path(racing_line)
    use_mpcc = getattr(cfg, "mpcc_enabled", False)

    # PRE-RACE: align with first gate
    if path:
        g1_app = path[0]["approach"]
        g1_ctr = path[0]["center"]
        cur = iface.get_position()
        safe_z = cfg.takeoff_height  # use safe hovering altitude, not gate Z
        print(f"Pre-race: flying to gate 1 approach point at z={safe_z:.1f} ...")
        iface.move_to_position(float(g1_app[0]), float(g1_app[1]), float(safe_z), 3.0)
        iface.hover()
        time.sleep(2.0)
        print("Pre-race alignment complete. Starting race loop.")

    # ============================================================
    # STATE ESTIMATOR INIT — seeded from AirSim ground truth once
    # Implements Swift-style EKF: VIO (IMU propagation) +
    # PnP (gate-based absolute position fix).
    # ============================================================
    use_estimator = getattr(cfg, "use_state_estimator", False)
    vio = None
    kf  = None
    pnp = None
    if use_estimator:
        _seed_pos  = iface.get_position()
        _seed_vel  = iface.get_velocity()
        _q         = iface.get_orientation()
        _seed_quat = np.array([_q.w_val, _q.x_val, _q.y_val, _q.z_val], dtype=np.float64)
        _vio_noise = {
            "accel_noise_std": getattr(cfg, "vio_accel_noise", 0.01),
            "gyro_noise_std":  getattr(cfg, "vio_gyro_noise",  0.001),
        }
        vio = VIOEstimator(noise=_vio_noise)
        vio.initialize(_seed_pos, _seed_vel, _seed_quat)
        kf = SwiftKalmanFilter()
        kf.initialize(_seed_pos, _seed_vel)
        pnp = PnPLocalizer(
            gate_half_size_m=getattr(cfg, "gate_half_size", 0.75),
            img_w=640,
            img_h=480,
            fov_h_deg=getattr(cfg, "cam_fov",
                              getattr(cfg, "camera_fov_h_deg", 90.0)),
        )
        print("[StateEstimator] VIO + PnP + Kalman initialized.")

    # ============================================================
    # RL POLICY INIT (loads adrl_ppo_racer_v2.zip if enabled)
    # ============================================================
    use_rl_policy = getattr(cfg, "use_rl_policy", False)
    rl_model = None
    if use_rl_policy:
        if not _SB3_AVAILABLE:
            print("[RL] stable_baselines3 not installed — RL disabled.")
            use_rl_policy = False
        else:
            import os as _os
            _rl_path = "adrl_ppo_racer_v2.zip"
            if not _os.path.exists(_rl_path):
                print(f"[RL] Model file {_rl_path} not found — RL disabled. Run train_rl.py first.")
                use_rl_policy = False
            else:
                rl_model = _SB3_PPO.load(_rl_path)
                print(f"[RL] Loaded policy from {_rl_path}")

    if not args.headless:
        cv2.namedWindow("Drone Racing Dashboard", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drone Racing Dashboard", DASH_W, DASH_H)

    telem = LapTelemetry(total_gates=len(path), csv_path="cv_race_log.csv")
    telem.lap_start()

    gate_idx   = 0
    phase      = "APPROACH"
    gate_start = time.time()
    tracker    = GateTracker(cfg)
    stuck_det  = StuckDetector(cfg)
    cv_nx_out  = 0.0
    cv_ny_out  = 0.0
    rl_seg_idx = 0
    retries    = 0        # backup attempts on current gate
    collision_timer = 0.0 # seconds spent slow+close to gate
    smooth_quat = SmoothQuat(alpha=0.12)  # guideline orientation filter

    # Start telemetry for first gate
    if path:
        telem.gate_start(0, path[0]["name"])

    try:
        while True:
            t_loop = time.time()

            # Ground truth always fetched — used for fallback + diagnostics
            _gt_pos       = iface.get_position()
            _gt_vel       = iface.get_velocity()
            drone_yaw     = iface.get_yaw()
            drone_quat    = iface.get_orientation()
            gate_elapsed  = time.time() - gate_start

            # --- VIO + Kalman predict (50 Hz) ---
            if use_estimator and kf is not None and kf.initialized:
                try:
                    _imu_data     = iface.get_imu_data()
                    vio.update(_imu_data, cfg.control_dt)
                    kf.predict(_imu_data, cfg.control_dt)
                    _kf_state     = kf.get_state()
                    drone_pos     = _kf_state["pos"]
                    drone_vel_vec = _kf_state["vel"]
                except Exception as _e:
                    print(f"[StateEstimator] predict failed: {_e}; falling back to AirSim")
                    drone_pos     = _gt_pos
                    drone_vel_vec = _gt_vel
            else:
                drone_pos     = _gt_pos
                drone_vel_vec = _gt_vel

            cur_speed = norm(drone_vel_vec)

            # --- Camera ---
            frame = iface.get_image()

            if gate_idx >= len(path):
                if not args.headless:
                    done = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)
                    cv2.putText(done, "COURSE COMPLETE",
                                (DASH_W // 2 - 230, DASH_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.2, C_GREEN, 4)
                    cv2.imshow("Drone Racing Dashboard", done)
                    cv2.waitKey(2500)
                break

            gate = path[gate_idx]

            if gate_elapsed > cfg.max_gate_time:
                print(f"  [TIMEOUT] Skipping gate {gate_idx + 1}")
                gate_idx  += 1
                phase      = "APPROACH"
                gate_start = time.time()
                tracker.reset()
                stuck_det.reset()
                retries = 0
                collision_timer = 0.0
                continue

            gate_progress = signed_progress(drone_pos, gate["center"],
                                            gate["forward"])
            dist_gate     = norm(gate["center"] - drone_pos)
            z_err_m       = gate["center"][2] - drone_pos[2]

            # Dynamic CV window based on gate distance
            sky_frac, floor_frac, dynamic_min_area = compute_cv_window(dist_gate, cfg)
            windowed_frame = apply_search_window(frame, sky_frac, floor_frac)

            hsv_m   = multi_hsv_mask(windowed_frame, cfg)
            edge_m  = edge_mask(windowed_frame, cfg)
            # YOLOv8 primary detector — falls back to contour if unavailable/uncertain
            raw_det = (gate_det.detect(windowed_frame) or
                       find_best_gate_contour(hsv_m, edge_m, windowed_frame.shape,
                                              min_area=dynamic_min_area, cfg=cfg))
            tracker.update(raw_det)

            # --- PnP + Kalman update on gate detection ---
            if use_estimator and kf is not None and kf.initialized:
                _det = tracker.current
                if _det is not None:
                    # YOLO detections carry bbox_xyxy; derive from contour otherwise
                    _bbox = _det.get("bbox_xyxy", None)
                    if _bbox is None and _det.get("contour") is not None:
                        _rx, _ry, _rw, _rh = cv2.boundingRect(_det["contour"])
                        _bbox = (_rx, _ry, _rx + _rw, _ry + _rh)
                    if _bbox is not None and (_bbox[2] - _bbox[0]) > 5 and (_bbox[3] - _bbox[1]) > 5:
                        _quat_wxyz = np.array([
                            drone_quat.w_val, drone_quat.x_val,
                            drone_quat.y_val, drone_quat.z_val,
                        ], dtype=np.float64)
                        try:
                            _pnp_result = pnp.localize(_bbox, gate["center"], _quat_wxyz)
                            if _pnp_result["success"]:
                                kf.update_pnp(_pnp_result["pos"], _pnp_result["cov"])
                                _kf_c = kf.get_state()
                                vio.set_position(_kf_c["pos"])
                                vio.set_velocity(_kf_c["vel"])
                        except Exception as _pnp_e:
                            print(f"[PnP] update failed: {_pnp_e}")

            telem.update_frame(speed=cur_speed, detection_found=(tracker.current is not None))

            # Auto-skip
            if phase in ("APPROACH", "CENTER"):
                if gate_progress > cfg.auto_skip_progress:
                    print(f"  [AUTO-SKIP] Past gate {gate_idx + 1}")
                    gate_idx  += 1
                    phase      = "APPROACH"
                    gate_start = time.time()
                    tracker.reset()
                    stuck_det.reset()
                    retries = 0
                    collision_timer = 0.0
                    continue

            # =============================================================
            # COLLISION DETECTION & BACKUP (v11)
            # If the drone is slow and close to a gate for too long,
            # it has probably hit the gate frame. Back up and retry.
            # =============================================================
            if (cur_speed < cfg.collision_speed_thresh and
                dist_gate < cfg.collision_dist_thresh and
                phase in ("APPROACH", "CENTER") and
                gate_elapsed > 2.0):
                collision_timer += cfg.control_dt
            else:
                collision_timer = max(0.0, collision_timer - cfg.control_dt * 0.5)

            if collision_timer > cfg.collision_time_thresh:
                if retries < cfg.max_retries:
                    retries += 1
                    collision_timer = 0.0
                    stuck_det.reset()
                    print(f"  [BACKUP] gate {gate_idx + 1}, "
                          f"retry {retries}/{cfg.max_retries}")
                    iface.hover()
                    time.sleep(0.2)
                    backup = gate["center"] - gate["forward"] * cfg.backup_dist
                    backup[2] = gate["center"][2]
                    iface.move_to_position(
                        float(backup[0]), float(backup[1]),
                        float(backup[2]), cfg.backup_speed)
                    iface.hover()
                    time.sleep(0.3)
                    phase = "APPROACH"
                    gate_start = time.time()
                    tracker.reset()
                    continue
                else:
                    print(f"  [SKIP after {cfg.max_retries} retries] "
                          f"gate {gate_idx + 1}")
                    gate_idx  += 1
                    phase      = "APPROACH"
                    gate_start = time.time()
                    tracker.reset()
                    stuck_det.reset()
                    retries = 0
                    collision_timer = 0.0
                    continue

            # Commit zone
            in_commit = (phase == "CENTER" and
                         abs(gate_progress) < cfg.commit_dist and
                         abs(z_err_m) < cfg.commit_alt_tol)

            # =============================================================
            # PURE PURSUIT with adaptive speed and proximity scaling
            # =============================================================
            min_seg = gate_idx * 3
            rl_seg_idx, rl_seg_t, _ = find_closest_segment(
                racing_line, drone_pos, min_seg=min_seg)

            if phase == "APPROACH":
                max_speed = cfg.max_approach_speed
            elif phase == "CENTER":
                max_speed = cfg.max_center_speed
            else:
                max_speed = cfg.max_exit_speed

            # TURN-ANGLE ADAPTIVE SPEED: scale max_speed by the turn
            # difficulty at the current gate.  Approaching a sharp turn
            # gradually slows the drone; on straights it stays full speed.
            turn_mult = gate["turn_speed_mult"]
            if phase in ("APPROACH", "CENTER"):
                # Taper: full speed far away, turn_mult speed near gate
                taper = clamp(1.0 - dist_gate / cfg.turn_taper_dist, 0.0, 1.0)
                effective_mult = 1.0 + taper * (turn_mult - 1.0)
                max_speed *= effective_mult
            elif phase == "EXIT" and gate_idx + 1 < len(path):
                # Also slow for upcoming gate's turn angle
                next_mult = path[gate_idx + 1]["turn_speed_mult"]
                next_dist = norm(path[gate_idx + 1]["center"] - drone_pos)
                taper = clamp(1.0 - next_dist / cfg.turn_taper_dist, 0.0, 1.0)
                effective_mult = 1.0 + taper * (next_mult - 1.0)
                max_speed *= effective_mult

            # PREDICTIVE SPEED PROFILING: Look ahead at upcoming gates
            # and cap speed if we're approaching a deceleration zone
            pred_cap = compute_predictive_max_speed(drone_pos, gate_idx, path, cfg)
            max_speed = min(max_speed, pred_cap)

            # On a retry, slow down further
            if retries > 0 and phase in ("APPROACH", "CENTER"):
                max_speed *= cfg.retry_speed_mult

            # Lookahead with proximity scaling near gates (pure pursuit + MPCC dist ref)
            lookahead_d = clamp(cur_speed * cfg.lookahead_speed_scale,
                                cfg.lookahead_min, cfg.lookahead_max)
            if phase in ("APPROACH", "CENTER"):
                prox_scale = clamp(dist_gate / cfg.prox_scale_dist,
                                   cfg.prox_scale_min, 1.0)
                lookahead_d = max(lookahead_d * prox_scale,
                                  cfg.prox_lookahead_floor)

            target = get_lookahead_point(racing_line, rl_seg_idx,
                                         rl_seg_t, lookahead_d)

            to_target   = target - drone_pos
            dist_target = norm(to_target)

            # --- Stuck detection ---
            if phase == "CENTER":
                stuck_det.update(gate_progress)
                if stuck_det.is_stuck():
                    print(f"  [STUCK] Forcing EXIT on gate {gate_idx + 1}")
                    phase = "EXIT"
                    stuck_det.reset()
                    continue
            else:
                stuck_det.reset()

            # --- Phase transitions ---
            if phase == "APPROACH" and dist_gate < cfg.approach_tol:
                phase = "CENTER"
                stuck_det.reset()
                continue
            if phase == "CENTER" and gate_progress > cfg.gate_cross_eps:
                phase = "EXIT"
                stuck_det.reset()
                continue
            if phase == "EXIT":
                past_exit = gate_progress > cfg.exit_dist * 0.6
                close_to_next = (gate_idx + 1 < len(path) and
                    norm(path[gate_idx + 1]["approach"] - drone_pos) <
                    cfg.approach_tol * 1.5)
                if dist_target < cfg.exit_tol or past_exit or close_to_next:
                    telem.gate_exit()
                    gate_idx  += 1
                    phase      = "APPROACH"
                    gate_start = time.time()
                    tracker.reset()
                    stuck_det.reset()
                    retries = 0
                    collision_timer = 0.0
                    if gate_idx < len(path):
                        telem.gate_start(gate_idx, path[gate_idx]["name"])
                    continue

            # --- Direction & yaw ---
            direction    = unit(to_target)
            target_yaw   = math.atan2(direction[1], direction[0])
            gate_fwd_yaw = math.atan2(gate["forward"][1],
                                       gate["forward"][0])
            if dist_gate < cfg.yaw_align_start_dist:
                yaw_blend = clamp(
                    1.0 - (dist_gate - cfg.yaw_align_full_dist) /
                          (cfg.yaw_align_start_dist - cfg.yaw_align_full_dist),
                    0.0, 0.7)
                desired_yaw = angle_lerp(target_yaw, gate_fwd_yaw, yaw_blend)
            else:
                desired_yaw = target_yaw

            yaw_err     = math.atan2(math.sin(desired_yaw - drone_yaw),
                                      math.cos(desired_yaw - drone_yaw))
            yaw_err_deg = math.degrees(yaw_err)

            # --- Speed & velocity desired ---
            if use_mpcc:
                # MPCC: solve receding-horizon optimization for velocity command
                mx, my, mz, mpcc_yaw_deg = mpcc.compute(
                    drone_pos, drone_vel_vec, speed_limit=max_speed)
                vel_desired  = np.array([mx, my, mz], dtype=np.float64)
                desired_yaw  = math.radians(mpcc_yaw_deg)
                # Gate-aligned yaw blend still applies when close
                if dist_gate < cfg.yaw_align_start_dist:
                    desired_yaw = angle_lerp(desired_yaw, gate_fwd_yaw, yaw_blend)
                yaw_err      = math.atan2(math.sin(desired_yaw - drone_yaw),
                                          math.cos(desired_yaw - drone_yaw))
                yaw_err_deg  = math.degrees(yaw_err)
                speed        = float(np.linalg.norm(vel_desired))
                speed_scale  = speed / max_speed if max_speed > 0 else 0.0
            else:
                # Pure pursuit (fallback / default when mpcc_enabled: false)
                speed_scale = compute_speed_scale(yaw_err, dist_target, phase, cfg)
                speed       = max_speed * speed_scale
                if phase == "CENTER":
                    speed = max(speed, cfg.min_center_speed)
                vel_desired = direction * speed

            # --- Lateral corrections (speed-proportional) ---
            speed_boost = 1.0 + cur_speed / cfg.lateral_speed_scale
            lat_gain = (cfg.lateral_gain_center   if phase == "CENTER"   else
                        cfg.lateral_gain_approach  if phase == "APPROACH" else 0.0)
            lat_gain *= speed_boost
            lat_corr    = lateral_correction_vec(
                drone_pos, gate["center"], gate["forward"],
                lat_gain, cfg.lateral_max)
            lat_err_mag = norm(lat_corr) / max(lat_gain, 0.01)

            cv_corr, cv_nx_out, cv_ny_out = cv_steer_correction(
                tracker, windowed_frame.shape, gate["forward"], phase, in_commit,
                cfg, cur_speed)

            if tracker.current is not None:
                combined_lat = lerp(lat_corr, cv_corr, cfg.cv_steer_blend)
            else:
                combined_lat = lat_corr

            if in_commit:
                combined_lat = combined_lat * cfg.commit_lat_damp
                fwd_component = float(np.dot(vel_desired, gate["forward"]))
                if fwd_component < cfg.commit_min_fwd_speed:
                    vel_desired = vel_desired + gate["forward"] * (
                        cfg.commit_min_fwd_speed - fwd_component)

            vel_cmd     = vel_desired + combined_lat
            vel_cmd     = lerp(vel_cmd, drone_vel_vec, cfg.vel_ff_alpha)

            # =============================================================
            # GATE CUSHION: speed-proportional corrective force toward
            # gate center.  At 14 m/s the drone needs ~3x the lateral
            # force vs 5 m/s to bend its trajectory through center.
            # Also includes Z component for vertical gate misalignment.
            # =============================================================
            if dist_gate < cfg.cushion_dist and phase in ("APPROACH", "CENTER"):
                to_gate   = gate["center"] - drone_pos
                fwd_proj  = float(np.dot(to_gate, gate["forward"]))
                lat_off   = to_gate - gate["forward"] * fwd_proj
                lat_mag   = norm(lat_off)
                if lat_mag > cfg.cushion_threshold:
                    proximity  = clamp(1.0 - dist_gate / cfg.cushion_dist, 0.0, 1.0)
                    speed_mult = 1.0 + cur_speed / cfg.cushion_speed_scale
                    force_mag  = ((lat_mag - cfg.cushion_threshold) *
                                  cfg.cushion_gain * proximity * speed_mult)
                    force_mag  = min(force_mag, cfg.cushion_max * speed_mult)
                    cushion    = unit(lat_off) * force_mag
                    vel_cmd    = vel_cmd + cushion

            # =============================================================
            # ALTITUDE: use racing-line target Z for APPROACH/EXIT so the
            # drone starts climbing/descending early; gate center Z for
            # CENTER so it threads the gate precisely.
            # =============================================================
            if phase == "CENTER":
                alt_target_z = gate["center"][2]
            else:
                alt_target_z = target[2]
            vel_cmd[2] = altitude_cmd(drone_pos[2], alt_target_z, phase, cfg)

            if use_rl_policy and rl_model is not None:
                # Build 18D obs matching adrl_env.py obs space exactly
                _g1 = gate["center"]
                _g2 = path[min(gate_idx + 1, len(path) - 1)]["center"]
                _rel1 = (_g1 - drone_pos).astype(np.float32)
                _rel2 = (_g2 - drone_pos).astype(np.float32)
                _det_cur = tracker.current
                if _det_cur is not None:
                    _cp = _det_cur.get("centroid_px") or _det_cur.get("centroid", (320, 240))
                    _cx, _cy = float(_cp[0]), float(_cp[1])
                    _af = float(_det_cur.get("area_frac", _det_cur.get("area", 0) / (640 * 480)))
                    _sc = float(_det_cur.get("score", 0.0))
                    _cv_feats = np.array([1.0, (_cx - 320) / 320, (_cy - 240) / 240, _af, _sc], dtype=np.float32)
                else:
                    _cv_feats = np.zeros(5, dtype=np.float32)
                _rl_obs = np.concatenate([
                    drone_vel_vec.astype(np.float32),
                    np.array([math.sin(drone_yaw), math.cos(drone_yaw)], dtype=np.float32),
                    _rel1, _rel2,
                    np.array([norm(_rel1), norm(_rel2)], dtype=np.float32),
                    _cv_feats,
                ])
                _act, _ = rl_model.predict(_rl_obs, deterministic=True)
                _roll  = float(np.clip(_act[0] * 0.10, -0.20, 0.20))
                _pitch = float(np.clip(0.18 + _act[1] * 0.10, -0.20, 0.25))
                _yr    = float(math.radians(np.clip(_act[2] * 20.0, -30.0, 30.0)))
                _z_tgt = float(drone_pos[2] + np.clip(_act[3] * 0.30, -0.40, 0.40))
                iface.move_by_roll_pitch_yawrate_z(_roll, _pitch, _yr, _z_tgt, cfg.control_dt * 1.5)
            else:
                iface.move_by_velocity(
                    float(vel_cmd[0]),
                    float(vel_cmd[1]),
                    float(vel_cmd[2]),
                    cfg.control_dt * 1.5,
                    float(math.degrees(desired_yaw))
                )

            # =============================================================
            # PROJECT PATH FOR VISUALISATION
            # Smoothed quaternion filters out frame-to-frame jitter
            # while keeping correct perspective - guideline looks locked
            # to the track instead of bouncing with every attitude change.
            # =============================================================
            smooth_quat.update(drone_quat)
            vis_pts = sample_path_ahead(racing_line, rl_seg_idx,
                                        rl_seg_t, cfg.path_vis_dist,
                                        cfg.path_vis_samples)
            path_px = project_points_to_camera(vis_pts, drone_pos,
                                                smooth_quat, CAM_W, DASH_H)

            # Project target (lookahead) point
            target_px_list = project_points_to_camera(
                [target], drone_pos, smooth_quat, CAM_W, DASH_H)
            target_px = target_px_list[0] if target_px_list else None

            # Project upcoming gate centers with labels
            gate_centers_3d = []
            gate_labels     = []
            for gi in range(gate_idx, min(gate_idx + 4, len(path))):
                gate_centers_3d.append(path[gi]["center"])
                gate_labels.append(f"G{gi + 1}")
            gate_center_px_list = project_points_to_camera(
                gate_centers_3d, drone_pos, smooth_quat, CAM_W, DASH_H)

            # --- Dashboard ---
            dash = build_dashboard(
                frame, tracker, path, gate_idx, phase, gate, cfg,
                drone_pos, drone_yaw, drone_vel_vec, target,
                speed, max_speed, speed_scale, yaw_err_deg,
                lat_err_mag, combined_lat, gate_progress, dist_gate,
                z_err_m, gate_elapsed, stuck_det, cv_nx_out, cv_ny_out,
                in_commit, lookahead_d, racing_line, rl_seg_idx,
                path_px, gate_center_px_list, gate_labels, target_px
            )
            if not args.headless:
                cv2.imshow("Drone Racing Dashboard", dash)

            cv_label = ("%.2f" % tracker.current["score"]
                        if tracker.current else " -- ")
            commit_tag = " COMMIT" if in_commit else ""
            _ekf_unc_str = ""
            if use_estimator and kf is not None and kf.initialized:
                _ekf_unc_str = f" | ekf_unc={kf.get_state()['uncertainty']:.3f}"
            print(
                f"G{gate_idx+1:>2}/{len(path)} | {phase:<8}{commit_tag:<7} | "
                f"{gate['name']:<12} | "
                f"dst={dist_target:5.2f} prg={gate_progress:+5.2f} "
                f"z={z_err_m:+5.2f} | "
                f"lk={lookahead_d:4.1f} | "
                f"spd={speed:5.1f}/{max_speed:5.1f} | "
                f"trn={gate['turn_angle']:4.0f}d | "
                f"cv={cv_label} | r={retries}"
                + _ekf_unc_str
            )

            if not args.headless and cv2.waitKey(1) & 0xFF == ord("q"):
                break
            elapsed = time.time() - t_loop
            if cfg.control_dt - elapsed > 0:
                time.sleep(cfg.control_dt - elapsed)
    finally:
        telem.lap_end()
        telem.save_csv()
        if args.telemetry_out:
            telem.to_json(args.telemetry_out)
        iface.shutdown()

if __name__ == "__main__":
    main()
