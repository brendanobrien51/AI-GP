"""
AirSim Drone Racing Lab - Advanced Autonomous Gate Racer v4
===========================================================
FIX 1 - CV: Multi-range HSV + GateTracker
  - Four overlapping HSV ranges cover orange, yellow-orange, and red-orange
    gate frames regardless of lighting.  The masks are OR-ed together before
    morphological cleanup, giving a clean single blob per gate.
  - Contour scoring now checks rectangularity, aspect ratio (gates are ~square),
    convex-hull solidity, and frame-fraction size; each sub-score is shown in
    the brain panel so you can see why a contour was accepted or rejected.
  - GateTracker remembers the last good detection for up to TRACKER_STALE_S
    seconds so a single occluded frame does not blank the overlay.
FIX 2 - Stuck gates: StuckDetector
  - StuckDetector records gate_progress every frame with a timestamp.
  - If progress has not advanced by STUCK_MIN_PROGRESS metres over the last
    STUCK_WINDOW seconds while in CENTER phase it forces the phase to EXIT,
    breaking the oscillation loop.
  - On every gate transition the detector is reset.
FIX 3 - Altitude: always-on proportional Z controller
  - Replaced the threshold-only altitude_guard_z with a continuous proportional
    controller targeting gate["center"][2] every frame (gate altitude, not
    waypoint altitude which may be wrong for angled gates).
  - Gain is higher in CENTER (ALT_KP_CENTER) so the drone snaps to the gate
    plane height just before passing through.
  - Drone no longer drifts high/low between gates because it is always being
    softly pulled toward the upcoming gate's Z coordinate.
"""
import math
import re
import time
import airsimdroneracinglab as airsim
import cv2
import numpy as np
# ---------------------------------------------------------------------------
# TUNABLE PARAMETERS
# ---------------------------------------------------------------------------
TAKEOFF_HEIGHT        = -1.5   # NED z (negative = up)
MAX_APPROACH_SPEED    = 8.0
MAX_CENTER_SPEED      = 5.5    # deliberately slow for precision
MAX_EXIT_SPEED        = 10.0
MIN_SPEED_FRACTION    = 0.22
YAW_SCALE_EXPONENT    = 1.6
BRAKE_DIST            = 3.5
BRAKE_MIN_FRACTION    = 0.30
VEL_FF_ALPHA          = 0.15
APPROACH_TOL          = 3.8
EXIT_TOL              = 14.0
FIRST_APPROACH_DIST   = 5.5
APPROACH_DIST         = 2.8
EXIT_DIST             = 3.0
CENTER_THROUGH_DIST   = 1.0
GATE_CROSS_EPS        = 0.15
DYN_APPROACH_FRAC     = 0.45
LOOKAHEAD_BLEND_START = 0.4
# Lateral centering
LATERAL_GAIN_APPROACH = 1.8
LATERAL_GAIN_CENTER   = 3.5
LATERAL_MAX           = 5.0
# Altitude PID (proportional only)
ALT_KP                = 2.0    # gain outside CENTER phase
ALT_KP_CENTER         = 4.5    # stronger gain while threading gate
ALT_MAX_Z_VEL         = 4.0    # clamp on Z velocity command (m/s)
# Stuck detection
STUCK_WINDOW          = 3.0    # seconds of history to evaluate
STUCK_MIN_PROGRESS    = 0.25   # metres; less than this = stuck
# Gate tracker
TRACKER_STALE_S       = 0.5    # seconds before a cached detection expires
# Gate timeout
MAX_GATE_TIME         = 18.0
# Control loop
CONTROL_DT            = 0.08
# CV
CONTOUR_MIN_AREA      = 300
CV_SCORE_THRESHOLD    = 0.38   # lower = more permissive; raise if false positives
CANNY_LOW             = 50
CANNY_HIGH            = 130
# Multi-range HSV (orange, yellow-orange, red-orange low, red-orange high)
HSV_RANGES = [
    (np.array([  5,  80,  80], dtype=np.uint8), np.array([ 25, 255, 255], dtype=np.uint8)),
    (np.array([ 18, 100, 100], dtype=np.uint8), np.array([ 38, 255, 255], dtype=np.uint8)),
    (np.array([  0,  80,  80], dtype=np.uint8), np.array([  8, 255, 255], dtype=np.uint8)),
    (np.array([172,  80,  80], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
]
# Dashboard
DASH_W, DASH_H = 1280, 720
CAM_W          = 854
RIGHT_W        = DASH_W - CAM_W
MAP_H          = 380
BRAIN_H        = DASH_H - MAP_H
C_WHITE  = (255, 255, 255)
C_BLACK  = (  0,   0,   0)
C_GREEN  = (  0, 220,  80)
C_YELLOW = (  0, 220, 220)
C_ORANGE = (  0, 165, 255)
C_RED    = ( 50,  50, 240)
C_CYAN   = (230, 220,   0)
C_GRAY   = (120, 120, 120)
C_DGRAY  = ( 40,  40,  40)
C_BLUE   = (230, 100,   0)
C_LIME   = ( 50, 255, 100)
PHASE_COLORS = {"APPROACH": C_YELLOW, "CENTER": C_GREEN, "EXIT": C_ORANGE}
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
def quat_to_yaw(q):
    siny = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return math.atan2(siny, cosy)
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
def get_gate_names(client):
    names   = client.simListSceneObjects(".*[Gg]ate.*")
    cleaned = [n for n in names if isinstance(n, str) and n.strip()]
    def sort_key(name):
        nums = re.findall(r"\d+", name)
        return [int(n) for n in nums] if nums else [10**9]
    cleaned.sort(key=sort_key)
    return cleaned
def get_object_pose_safe(client, name):
    if not hasattr(client, "race_tier"):
        client.race_tier = None
    if not hasattr(client, "level_name"):
        client.level_name = ""
    try:
        return client.simGetObjectPose(name)
    except Exception:
        internal = getattr(client, "_VehicleClient__internalGetObjectPose", None)
        if internal is not None:
            return internal(name)
        raise
def build_path(client, gate_names):
    gates = []
    for name in gate_names:
        pose = get_object_pose_safe(client, name)
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
            base_app = FIRST_APPROACH_DIST
        else:
            gap      = norm(center - gates[i - 1][1])
            base_app = max(1.5, min(APPROACH_DIST, gap * DYN_APPROACH_FRAC))
        path.append({
            "index":    i,
            "name":     name,
            "center":   center.copy(),
            "approach": center - fwd * base_app,
            "exit":     center + fwd * EXIT_DIST,
            "forward":  fwd,
        })
    return path
# ---------------------------------------------------------------------------
# COMPUTER VISION
# ---------------------------------------------------------------------------
def multi_hsv_mask(frame):
    hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))
    k_close  = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k_open   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)
    return combined
def edge_mask(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    return cv2.dilate(edges, None, iterations=1)
def score_contour(contour, frame_shape):
    area = cv2.contourArea(contour)
    if area < CONTOUR_MIN_AREA:
        return None
    fh, fw   = frame_shape[:2]
    frac     = area / (fh * fw)
    if frac < 0.001 or frac > 0.65:
        return None
    rect     = cv2.minAreaRect(contour)
    rw, rh   = rect[1]
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
    if score < CV_SCORE_THRESHOLD:
        return None
    cx, cy = int(rect[0][0]), int(rect[0][1])
    return {
        "score": score, "rect_score": rectangularity,
        "aspect_score": aspect_score, "solidity": solidity,
        "area_frac": frac, "aspect": aspect,
        "centroid_px": (cx, cy), "contour": contour, "rect": rect,
    }
def find_best_gate_contour(hsv_mask, edge_mask_img, frame_shape):
    merged      = cv2.bitwise_or(hsv_mask, edge_mask_img)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        result = score_contour(c, frame_shape)
        if result is not None:
            if best is None or result["score"] > best["score"]:
                best = result
    return best
class GateTracker:
    def __init__(self):
        self._best = None
        self._t = 0.0
        self.frames_lost = 0
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
        if time.time() - self._t > TRACKER_STALE_S:
            return None
        return self._best
    def reset(self):
        self._best = None
        self._t = 0.0
        self.frames_lost = 0
# ---------------------------------------------------------------------------
# STUCK DETECTOR
# ---------------------------------------------------------------------------
class StuckDetector:
    def __init__(self):
        self._history = []
    def update(self, progress):
        now = time.time()
        self._history.append((now, progress))
        cutoff = now - STUCK_WINDOW
        self._history = [(t, p) for t, p in self._history if t >= cutoff]
    def is_stuck(self):
        if len(self._history) < 10:
            return False
        span = self._history[-1][0] - self._history[0][0]
        if span < STUCK_WINDOW * 0.75:
            return False
        delta = self._history[-1][1] - self._history[0][1]
        return delta < STUCK_MIN_PROGRESS
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
def compute_speed_scale(yaw_err_rad, dist_to_target):
    yaw_s   = max(MIN_SPEED_FRACTION,
                  math.cos(clamp(abs(yaw_err_rad), 0, math.pi / 2))
                  ** YAW_SCALE_EXPONENT)
    brake_s = (max(BRAKE_MIN_FRACTION, dist_to_target / BRAKE_DIST)
               if dist_to_target < BRAKE_DIST else 1.0)
    return min(yaw_s, brake_s)
def signed_progress(drone_pos, gate_center, gate_forward):
    return float(np.dot(drone_pos - gate_center, gate_forward))
def altitude_cmd(drone_z, gate_z, phase):
    z_err = gate_z - drone_z
    kp    = ALT_KP_CENTER if phase == "CENTER" else ALT_KP
    return clamp(z_err * kp, -ALT_MAX_Z_VEL, ALT_MAX_Z_VEL)
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
        self.ox    = margin + (uw - span_x * self.scale) / 2 - min(xs) * self.scale
        self.oy    = margin + (uh - span_y * self.scale) / 2 - min(ys) * self.scale
        self.pw, self.ph = pw, ph
    def to_px(self, wx, wy):
        return (clamp(int(wx * self.scale + self.ox), 0, self.pw - 1),
                clamp(int(wy * self.scale + self.oy), 0, self.ph - 1))
def draw_top_down_map(path, gate_idx, drone_pos, drone_yaw, target):
    panel = np.full((MAP_H, RIGHT_W, 3), (18, 18, 28), dtype=np.uint8)
    if not path:
        return panel
    tdm = TopDownMap([g["center"] for g in path], RIGHT_W, MAP_H)
    for i in range(len(path) - 1):
        p1 = tdm.to_px(path[i]["center"][0],   path[i]["center"][1])
        p2 = tdm.to_px(path[i+1]["center"][0], path[i+1]["center"][1])
        cv2.line(panel, p1, p2,
                 (60, 100, 60) if i < gate_idx else C_DGRAY, 1, cv2.LINE_AA)
    for i, g in enumerate(path):
        px, py  = tdm.to_px(g["center"][0], g["center"][1])
        is_cur  = (i == gate_idx)
        col     = C_CYAN if is_cur else ((60, 160, 60) if i < gate_idx else C_GRAY)
        fwd     = g["forward"]
        perp    = np.array([-fwd[1], fwd[0]])
        half    = max(4, int((10 if is_cur else 6) / tdm.scale))
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
    tx, ty = tdm.to_px(target[0], target[1])
    cv2.line(panel, (tx - 7, ty), (tx + 7, ty), C_ORANGE, 1)
    cv2.line(panel, (tx, ty - 7), (tx, ty + 7), C_ORANGE, 1)
    cv2.circle(panel, (tx, ty), 3, C_ORANGE, -1)
    dx, dy  = tdm.to_px(drone_pos[0], drone_pos[1])
    tip_len = 11
    tip   = (int(dx + math.cos(drone_yaw) * tip_len),
              int(dy + math.sin(drone_yaw) * tip_len))
    left  = (int(dx + math.cos(drone_yaw + 2.4) * 6),
              int(dy + math.sin(drone_yaw + 2.4) * 6))
    right = (int(dx + math.cos(drone_yaw - 2.4) * 6),
              int(dy + math.sin(drone_yaw - 2.4) * 6))
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
def draw_brain_panel(phase, gate_idx, total, gate_name,
                     speed, max_speed, yaw_err_deg, lat_err_mag,
                     gate_progress, dist_gate, detection,
                     vel, elapsed_s, stuck_det, z_err_m):
    panel = np.full((BRAIN_H, RIGHT_W, 3), (14, 14, 22), dtype=np.uint8)
    px    = 10
    bar_x = 92
    bar_w = RIGHT_W - bar_x - 54
    row   = 27
    y     = 24
    cv2.putText(panel, "BRAIN STATE", (px, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)
    pcol  = PHASE_COLORS.get(phase, C_WHITE)
    badge = f"  {phase}  "
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
    bx = RIGHT_W - bw - px - 4
    cv2.rectangle(panel, (bx - 4, 2), (bx + bw + 2, 2 + bh + 6), pcol, -1)
    cv2.putText(panel, badge, (bx, 2 + bh + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, C_BLACK, 2)
    cv2.putText(panel, f"Gate {gate_idx + 1}/{total}  {gate_name}",
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
    if detection is not None:
        d      = detection
        cv_col = C_GREEN if d["score"] > 0.65 else (C_YELLOW if d["score"] > 0.45 else C_RED)
        _bar(panel, bar_x, y, bar_w, 13, d["score"], cv_col,
             "CV TOTAL", f"{d['score']:.2f}")
        y += row
        sw = (bar_w - 4) // 3
        _bar(panel, bar_x,          y, sw, 10, d["rect_score"],   C_CYAN,   "RECT")
        _bar(panel, bar_x + sw + 2, y, sw, 10, d["aspect_score"], C_ORANGE, "ASPT")
        _bar(panel, bar_x + 2*sw+4, y, sw, 10, d["solidity"],     C_LIME,   "SOLD")
        y += 16
    else:
        cv2.putText(panel, "CV: NO DETECTION",
                    (bar_x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_RED, 1)
        y += row + 16
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
def draw_camera_panel(frame, tracker, phase, gate_name, drone_vel,
                       yaw_err_deg, lat_corr):
    cam    = cv2.resize(frame, (CAM_W, DASH_H), interpolation=cv2.INTER_LINEAR)
    h, w   = cam.shape[:2]
    cw, ch = w // 2, h // 2
    det    = tracker.current
    if det is not None:
        sx = CAM_W / frame.shape[1]
        sy = DASH_H / frame.shape[0]
        box = cv2.boxPoints(det["rect"])
        box[:, 0] *= sx
        box[:, 1] *= sy
        t    = clamp((det["score"] - CV_SCORE_THRESHOLD) / (1.0 - CV_SCORE_THRESHOLD), 0, 1)
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
    cv2.circle(cam, (cw, ch), 32, C_WHITE, 1, cv2.LINE_AA)
    cv2.line(cam, (cw - 24, ch), (cw + 24, ch), C_WHITE, 1)
    cv2.line(cam, (cw, ch - 24), (cw, ch + 24), C_WHITE, 1)
    spd = norm(drone_vel)
    if spd > 0.3:
        sc  = 5.5
        ax  = int(cw + drone_vel[1] * sc)
        ay  = int(ch - drone_vel[2] * sc)
        cv2.arrowedLine(cam, (cw, ch), (ax, ay), C_CYAN, 2,
                         tipLength=0.28, line_type=cv2.LINE_AA)
        cv2.putText(cam, f"{spd:.1f}m/s", (ax + 6, ay - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_CYAN, 1)
    lat_mag = norm(lat_corr[:2])
    if lat_mag > 0.05:
        bar_len = int(clamp(lat_corr[1] * 10, -70, 70))
        col     = C_RED if abs(bar_len) > 35 else C_ORANGE
        cv2.arrowedLine(cam, (cw, ch + 45), (cw + bar_len, ch + 45),
                         col, 2, tipLength=0.25, line_type=cv2.LINE_AA)
        cv2.putText(cam, f"lat {lat_mag:.2f}m",
                    (cw + bar_len + 6, ch + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)
    yr      = math.radians(yaw_err_deg)
    yaw_tip = (int(cw + math.sin(yr) * 55), int(ch - math.cos(yr) * 55))
    yaw_col = C_GREEN if abs(yaw_err_deg) < 10 else               (C_YELLOW if abs(yaw_err_deg) < 30 else C_RED)
    cv2.line(cam, (cw, ch), yaw_tip, yaw_col, 2, cv2.LINE_AA)
    cv2.circle(cam, yaw_tip, 3, yaw_col, -1)
    pcol  = PHASE_COLORS.get(phase, C_WHITE)
    badge = f" {phase} "
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
def build_dashboard(frame, tracker, path, gate_idx, phase, gate,
                    drone_pos, drone_yaw, drone_vel, target,
                    speed, max_speed, speed_scale, yaw_err_deg,
                    lat_err_mag, lat_corr, gate_progress, dist_gate,
                    z_err_m, gate_elapsed, stuck_det):
    dash = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)
    cam_p = draw_camera_panel(frame, tracker, phase, gate["name"],
                               drone_vel, yaw_err_deg, lat_corr)
    dash[:, :CAM_W] = cam_p
    cv2.line(dash, (CAM_W, 0), (CAM_W, DASH_H), C_DGRAY, 1)
    map_p = draw_top_down_map(path, gate_idx, drone_pos, drone_yaw, target)
    dash[:MAP_H, CAM_W:] = map_p
    cv2.line(dash, (CAM_W, MAP_H), (DASH_W, MAP_H), C_DGRAY, 1)
    brain_p = draw_brain_panel(
        phase, gate_idx, len(path), gate["name"],
        speed, max_speed, yaw_err_deg, lat_err_mag,
        gate_progress, dist_gate, tracker.current,
        drone_vel, gate_elapsed, stuck_det, z_err_m
    )
    dash[MAP_H:, CAM_W:] = brain_p
    return dash
# ---------------------------------------------------------------------------
# SHUTDOWN
# ---------------------------------------------------------------------------
def shutdown(client):
    print("\nShutting down...")
    for fn in [lambda: client.hoverAsync().join(),
               lambda: client.landAsync().join(),
               lambda: client.disarm(),
               lambda: client.disableApiControl(),
               cv2.destroyAllWindows]:
        try:
            fn()
        except Exception:
            pass
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Connecting to AirSim Drone Racing Lab...")
    client            = airsim.MultirotorClient()
    client.race_tier  = None
    client.level_name = ""
    client.confirmConnection()
    client.enableApiControl()
    client.arm()
    print("Taking off...")
    client.takeoffAsync().join()
    state0    = client.getMultirotorState()
    s0_pos    = state_pos(state0)
    climb_tgt = vec3(s0_pos[0], s0_pos[1], TAKEOFF_HEIGHT)
    client.moveToPositionAsync(
        float(climb_tgt[0]), float(climb_tgt[1]), float(climb_tgt[2]), 2.0).join()
    gate_names = get_gate_names(client)
    if not gate_names:
        shutdown(client)
        raise RuntimeError("No gate objects found in scene.")
    print(f"Found {len(gate_names)} gates.")
    for i, n in enumerate(gate_names):
        print(f"  {i + 1:>2}. {n}")
    path = build_path(client, gate_names)
    cv2.namedWindow("Drone Racing Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Racing Dashboard", DASH_W, DASH_H)
    gate_idx   = 0
    phase      = "APPROACH"
    gate_start = time.time()
    tracker    = GateTracker()
    stuck_det  = StuckDetector()
    try:
        while True:
            t_loop = time.time()
            state         = client.getMultirotorState()
            drone_pos     = state_pos(state)
            drone_yaw     = quat_to_yaw(state.kinematics_estimated.orientation)
            drone_vel_vec = state_vel(state)
            gate_elapsed  = time.time() - gate_start
            responses = client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
            )
            if responses and responses[0].height > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                frame = img1d.reshape(responses[0].height,
                                       responses[0].width, 3).copy()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            hsv_m   = multi_hsv_mask(frame)
            edge_m  = edge_mask(frame)
            raw_det = find_best_gate_contour(hsv_m, edge_m, frame.shape)
            tracker.update(raw_det)
            if gate_idx >= len(path):
                done = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)
                cv2.putText(done, "COURSE COMPLETE",
                            (DASH_W // 2 - 230, DASH_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, C_GREEN, 4)
                cv2.imshow("Drone Racing Dashboard", done)
                cv2.waitKey(2500)
                break
            gate = path[gate_idx]
            if gate_elapsed > MAX_GATE_TIME:
                print(f"  [TIMEOUT] Skipping gate {gate_idx + 1}: {gate['name']}")
                gate_idx  += 1
                phase      = "APPROACH"
                gate_start = time.time()
                tracker.reset()
                stuck_det.reset()
                continue
            if phase == "APPROACH":
                target    = gate["approach"]
                max_speed = MAX_APPROACH_SPEED
            elif phase == "CENTER":
                target    = gate["center"] + gate["forward"] * CENTER_THROUGH_DIST
                max_speed = MAX_CENTER_SPEED
            else:
                target    = gate["exit"]
                max_speed = MAX_EXIT_SPEED
                if gate_idx + 1 < len(path):
                    next_app = path[gate_idx + 1]["approach"]
                    blend_t  = clamp(
                        1.0 - norm(gate["exit"] - drone_pos) /
                              max(EXIT_DIST * LOOKAHEAD_BLEND_START, 0.01),
                        0.0, 0.85)
                    target = lerp(gate["exit"], next_app, blend_t)
            to_target     = target - drone_pos
            dist_target   = norm(to_target)
            dist_gate     = norm(gate["center"] - drone_pos)
            gate_progress = signed_progress(drone_pos, gate["center"], gate["forward"])
            z_err_m       = gate["center"][2] - drone_pos[2]
            if phase == "CENTER":
                stuck_det.update(gate_progress)
                if stuck_det.is_stuck():
                    print(f"  [STUCK] Forcing EXIT on gate {gate_idx + 1}")
                    phase = "EXIT"
                    stuck_det.reset()
                    continue
            else:
                stuck_det.reset()
            if phase == "APPROACH" and dist_target < APPROACH_TOL:
                phase = "CENTER"
                stuck_det.reset()
                continue
            if phase == "CENTER" and gate_progress > GATE_CROSS_EPS:
                phase = "EXIT"
                stuck_det.reset()
                continue
            if phase == "EXIT" and dist_target < EXIT_TOL:
                gate_idx  += 1
                phase      = "APPROACH"
                gate_start = time.time()
                tracker.reset()
                stuck_det.reset()
                continue
            direction   = unit(to_target)
            desired_yaw = math.atan2(direction[1], direction[0])
            yaw_err     = math.atan2(math.sin(desired_yaw - drone_yaw),
                                      math.cos(desired_yaw - drone_yaw))
            yaw_err_deg = math.degrees(yaw_err)
            speed_scale = compute_speed_scale(yaw_err, dist_target)
            speed       = max_speed * speed_scale
            vel_desired = direction * speed
            lat_gain = (LATERAL_GAIN_CENTER   if phase == "CENTER"   else
                        LATERAL_GAIN_APPROACH  if phase == "APPROACH" else 0.0)
            lat_corr    = lateral_correction_vec(
                drone_pos, gate["center"], gate["forward"], lat_gain, LATERAL_MAX)
            lat_err_mag = norm(lat_corr) / max(lat_gain, 0.01)
            vel_cmd     = vel_desired + lat_corr
            vel_cmd     = lerp(vel_cmd, drone_vel_vec, VEL_FF_ALPHA)
            vel_cmd[2]  = altitude_cmd(drone_pos[2], gate["center"][2], phase)
            client.moveByVelocityAsync(
                float(vel_cmd[0]),
                float(vel_cmd[1]),
                float(vel_cmd[2]),
                CONTROL_DT * 1.5,
                yaw_mode=airsim.YawMode(
                    is_rate=False,
                    yaw_or_rate=float(math.degrees(desired_yaw))),
            )
            dash = build_dashboard(
                frame, tracker, path, gate_idx, phase, gate,
                drone_pos, drone_yaw, drone_vel_vec, target,
                speed, max_speed, speed_scale, yaw_err_deg,
                lat_err_mag, lat_corr, gate_progress, dist_gate,
                z_err_m, gate_elapsed, stuck_det
            )
            cv2.imshow("Drone Racing Dashboard", dash)
            print(
                f"G{gate_idx + 1:>2}/{len(path)} | {phase:<8} | "
                f"{gate['name']:<12} | "
                f"dst={dist_target:5.2f} prg={gate_progress:+5.2f} "
                f"z_err={z_err_m:+5.2f} | "
                f"lat={lat_err_mag:4.2f} | "
                f"spd={speed:4.1f}({speed_scale*100:.0f}%) | "
                f"yaw={yaw_err_deg:+6.1f} | "
                f"cv={'%.2f' % tracker.current['score'] if tracker.current else ' -- '}"
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            elapsed = time.time() - t_loop
            if CONTROL_DT - elapsed > 0:
                time.sleep(CONTROL_DT - elapsed)
    finally:
        shutdown(client)
if __name__ == "__main__":
    main()