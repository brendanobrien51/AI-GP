"""
AirSim Drone Racing Lab - Advanced Autonomous Gate Racer v7
===========================================================
v7 CHANGES:

1. RACING LINE + PURE PURSUIT
   - A dense polyline (approach -> center -> exit for every gate) is
     pre-computed.  At each frame the closest point on the line is found,
     then a lookahead point speed*LOOKAHEAD_SCALE metres ahead is used as
     the target.  This naturally cuts corners and gives smooth arcs.

2. SPEED-PROPORTIONAL LOOKAHEAD
   - Lookahead distance = speed * 0.9, clamped between 5-14 m.
   - At high speed the drone "looks further ahead", starting its turn
     earlier.  At low speed (e.g. near a tight gate) it tightens the arc.

3. CAMERA PATH GUIDELINE
   - Upcoming ~35 m of the racing line is sampled and projected onto the
     camera image using a pinhole model.  The result is a green/yellow
     polyline overlaid on the camera feed so you can see the planned arc.
   - Gate center positions are drawn as larger circles.

4. All v6 fixes retained (commit zone, altitude guard, CV steering, etc.)
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
TAKEOFF_HEIGHT        = -1.5

MAX_APPROACH_SPEED    = 8.0
MAX_CENTER_SPEED      = 7.5
MAX_EXIT_SPEED        = 10.0
MIN_SPEED_FRACTION    = 0.22
MIN_CENTER_SPEED      = 4.0
YAW_SCALE_EXPONENT    = 1.6
BRAKE_DIST            = 3.5
BRAKE_MIN_FRACTION    = 0.30
VEL_FF_ALPHA          = 0.12

APPROACH_TOL          = 4.2
EXIT_TOL              = 14.0
FIRST_APPROACH_DIST   = 8.0
APPROACH_DIST         = 2.8
EXIT_DIST             = 3.0
CENTER_THROUGH_DIST   = 4.0
GATE_CROSS_EPS        = 0.15
DYN_APPROACH_FRAC     = 0.45

AUTO_SKIP_PROGRESS    = 3.5

# Commit zone
COMMIT_DIST           = 2.5
COMMIT_LAT_DAMP       = 0.20
COMMIT_MIN_FWD_SPEED  = 4.5
COMMIT_ALT_TOL        = 1.2

# Pure-pursuit lookahead
LOOKAHEAD_MIN         = 5.0
LOOKAHEAD_MAX         = 14.0
LOOKAHEAD_SPEED_SCALE = 0.9

# Lateral centering (waypoint-based)
LATERAL_GAIN_APPROACH = 1.8
LATERAL_GAIN_CENTER   = 3.5
LATERAL_MAX           = 5.0

# CV-guided lateral correction
CV_STEER_GAIN_APPROACH = 1.2
CV_STEER_GAIN_CENTER   = 3.0
CV_STEER_GAIN_EXIT     = 0.5
CV_STEER_MAX           = 3.5
CV_STEER_BLEND         = 0.6

# Yaw alignment
YAW_ALIGN_START_DIST   = 8.0
YAW_ALIGN_FULL_DIST    = 2.5

# Altitude
ALT_KP                = 3.5
ALT_KP_CENTER         = 4.5
ALT_MAX_Z_VEL         = 5.0

# Stuck detection
STUCK_WINDOW          = 3.0
STUCK_MIN_PROGRESS    = 0.25

# Gate tracker
TRACKER_STALE_S       = 0.8

# Gate timeout
MAX_GATE_TIME         = 18.0

# Control loop
CONTROL_DT            = 0.08

# CV
CONTOUR_MIN_AREA      = 250
CV_SCORE_THRESHOLD    = 0.30
CANNY_LOW             = 50
CANNY_HIGH            = 130

HSV_RANGES = [
    (np.array([  5,  80,  80], dtype=np.uint8), np.array([ 25, 255, 255], dtype=np.uint8)),
    (np.array([ 18, 100, 100], dtype=np.uint8), np.array([ 38, 255, 255], dtype=np.uint8)),
    (np.array([  0,  80,  80], dtype=np.uint8), np.array([  8, 255, 255], dtype=np.uint8)),
    (np.array([172,  80,  80], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
]

# Camera projection
CAMERA_FOV_H_DEG      = 90.0

# Path visualisation
PATH_VIS_DIST         = 35.0   # metres ahead to draw
PATH_VIS_SAMPLES      = 40

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
C_MAGENTA = (200, 50, 200)
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
# RACING LINE (pure-pursuit polyline through all gates)
# ---------------------------------------------------------------------------
def build_racing_line(path):
    """Build a dense waypoint polyline: approach->center->exit for each gate."""
    line = []
    for g in path:
        line.append(g["approach"].copy())
        line.append(g["center"].copy())
        line.append(g["exit"].copy())
    return line

def find_closest_segment(line, pos, min_seg=0):
    """Find the closest point on the polyline from min_seg onwards.
    Returns (segment_index, t_on_segment, distance)."""
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
        pt = a + ab * t
        d  = norm(pos - pt)
        if d < best_dist:
            best_dist = d
            best_idx  = i
            best_t    = t
    return best_idx, best_t, best_dist

def get_lookahead_point(line, seg_idx, seg_t, lookahead_dist):
    """Walk forward along the polyline by lookahead_dist metres."""
    if seg_idx >= len(line) - 1:
        return line[-1].copy()
    a, b       = line[seg_idx], line[seg_idx + 1]
    seg_len    = norm(b - a)
    remaining  = seg_len * (1.0 - seg_t)

    if lookahead_dist <= remaining and seg_len > 1e-7:
        t_new = seg_t + lookahead_dist / seg_len
        return lerp(a, b, t_new)

    dist_left = lookahead_dist - remaining
    for i in range(seg_idx + 1, len(line) - 1):
        s = norm(line[i + 1] - line[i])
        if dist_left <= s and s > 1e-7:
            return lerp(line[i], line[i + 1], dist_left / s)
        dist_left -= s

    return line[-1].copy()

def sample_path_ahead(line, seg_idx, seg_t, total_dist, n_pts):
    """Sample n_pts evenly spaced points along the path ahead."""
    pts = []
    step = total_dist / max(n_pts, 1)
    for i in range(n_pts + 1):
        pts.append(get_lookahead_point(line, seg_idx, seg_t, step * i))
    return pts

# ---------------------------------------------------------------------------
# CAMERA PROJECTION (world -> pixel)
# ---------------------------------------------------------------------------
def project_points_to_camera(world_pts, drone_pos, drone_quat,
                              disp_w, disp_h, fov_h_deg=CAMERA_FOV_H_DEG):
    """Project 3D world points to 2D display pixels.
    Returns list of (px,py)|None for each point."""
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
# COMPUTER VISION
# ---------------------------------------------------------------------------
def multi_hsv_mask(frame):
    hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))
    k_close  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
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
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
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
# CV STEERING
# ---------------------------------------------------------------------------
def cv_steer_correction(tracker, frame_shape, gate_forward, phase,
                        in_commit_zone):
    det = tracker.current
    if det is None:
        return np.zeros(3, dtype=np.float64), 0.0, 0.0

    fh, fw = frame_shape[:2]
    cx, cy = det["centroid_px"]
    nx = (cx - fw / 2.0) / (fw / 2.0)
    ny = (cy - fh / 2.0) / (fh / 2.0)

    if phase == "CENTER":
        gain = CV_STEER_GAIN_CENTER
    elif phase == "APPROACH":
        gain = CV_STEER_GAIN_APPROACH
    else:
        gain = CV_STEER_GAIN_EXIT

    if in_commit_zone:
        gain *= COMMIT_LAT_DAMP

    down  = vec3(0, 0, 1)
    right = np.cross(gate_forward, down)
    r_n   = norm(right)
    if r_n < 1e-6:
        right = vec3(0, 1, 0)
    else:
        right = right / r_n

    correction = right * nx * gain + vec3(0, 0, ny * gain * 0.5)
    mag = norm(correction)
    if mag > CV_STEER_MAX:
        correction = correction / mag * CV_STEER_MAX
    return correction, nx, ny

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
        return (self._history[-1][1] - self._history[0][1]) < STUCK_MIN_PROGRESS

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

def compute_speed_scale(yaw_err_rad, dist_to_target, phase):
    yaw_s = max(MIN_SPEED_FRACTION,
                math.cos(clamp(abs(yaw_err_rad), 0, math.pi / 2))
                ** YAW_SCALE_EXPONENT)
    if phase == "CENTER":
        return yaw_s
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

    # Draw racing line (upcoming portion)
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

    # Target crosshair
    tx, ty = tdm.to_px(target[0], target[1])
    cv2.line(panel, (tx - 7, ty), (tx + 7, ty), C_ORANGE, 1)
    cv2.line(panel, (tx, ty - 7), (tx, ty + 7), C_ORANGE, 1)
    cv2.circle(panel, (tx, ty), 3, C_ORANGE, -1)

    # Drone
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
         clamp(lookahead_d / LOOKAHEAD_MAX, 0, 1), C_LIME,
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

def draw_camera_panel(frame, tracker, phase, gate_name, drone_vel,
                       yaw_err_deg, lat_corr, in_commit,
                       path_px, gate_center_px_list):
    cam    = cv2.resize(frame, (CAM_W, DASH_H), interpolation=cv2.INTER_LINEAR)
    h, w   = cam.shape[:2]
    cw, ch = w // 2, h // 2
    margin = 60  # allow some off-screen bleed before clipping

    # ---- PATH GUIDELINE ----
    # Draw polyline of projected path points
    visible = []
    for pt in path_px:
        if pt is not None and -margin < pt[0] < w + margin and -margin < pt[1] < h + margin:
            visible.append(pt)
        else:
            # break on None to avoid connecting segments behind the camera
            if visible and len(visible) >= 2:
                arr = np.array(visible, dtype=np.int32)
                cv2.polylines(cam, [arr], False, (80, 255, 80), 2, cv2.LINE_AA)
            visible = []
    if len(visible) >= 2:
        arr = np.array(visible, dtype=np.int32)
        cv2.polylines(cam, [arr], False, (80, 255, 80), 2, cv2.LINE_AA)

    # Draw dots along the guideline
    for i, pt in enumerate(path_px):
        if pt is not None and 0 <= pt[0] < w and 0 <= pt[1] < h:
            if i % 4 == 0:
                cv2.circle(cam, pt, 3, (60, 200, 60), -1, cv2.LINE_AA)

    # Draw gate center markers
    for gpt in gate_center_px_list:
        if gpt is not None and 0 <= gpt[0] < w and 0 <= gpt[1] < h:
            cv2.circle(cam, gpt, 8, C_CYAN, 2, cv2.LINE_AA)
            cv2.circle(cam, gpt, 2, C_CYAN, -1)

    # ---- CV DETECTION ----
    det = tracker.current
    if det is not None:
        sx = CAM_W / frame.shape[1]
        sy = DASH_H / frame.shape[0]
        box = cv2.boxPoints(det["rect"])
        box[:, 0] *= sx
        box[:, 1] *= sy
        t    = clamp((det["score"] - CV_SCORE_THRESHOLD) /
                     (1.0 - CV_SCORE_THRESHOLD), 0, 1)
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

def build_dashboard(frame, tracker, path, gate_idx, phase, gate,
                    drone_pos, drone_yaw, drone_vel, target,
                    speed, max_speed, speed_scale, yaw_err_deg,
                    lat_err_mag, lat_corr, gate_progress, dist_gate,
                    z_err_m, gate_elapsed, stuck_det, cv_nx, cv_ny,
                    in_commit, lookahead_d, racing_line, rl_seg_idx,
                    path_px, gate_center_px_list):
    dash = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)
    cam_p = draw_camera_panel(frame, tracker, phase, gate["name"],
                               drone_vel, yaw_err_deg, lat_corr, in_commit,
                               path_px, gate_center_px_list)
    dash[:, :CAM_W] = cam_p
    cv2.line(dash, (CAM_W, 0), (CAM_W, DASH_H), C_DGRAY, 1)
    map_p = draw_top_down_map(path, gate_idx, drone_pos, drone_yaw, target,
                              racing_line, rl_seg_idx)
    dash[:MAP_H, CAM_W:] = map_p
    cv2.line(dash, (CAM_W, MAP_H), (DASH_W, MAP_H), C_DGRAY, 1)
    brain_p = draw_brain_panel(
        phase, gate_idx, len(path), gate["name"],
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
    state0  = client.getMultirotorState()
    s0_pos  = state_pos(state0)
    climb_tgt = vec3(s0_pos[0], s0_pos[1], TAKEOFF_HEIGHT)
    client.moveToPositionAsync(
        float(climb_tgt[0]), float(climb_tgt[1]), float(climb_tgt[2]), 2.0
    ).join()

    gate_names = get_gate_names(client)
    if not gate_names:
        shutdown(client)
        raise RuntimeError("No gate objects found in scene.")
    print(f"Found {len(gate_names)} gates.")
    for i, n in enumerate(gate_names):
        print(f"  {i + 1:>2}. {n}")

    path = build_path(client, gate_names)
    racing_line = build_racing_line(path)

    # PRE-RACE: align with first gate
    if path:
        g1_app = path[0]["approach"]
        g1_z   = path[0]["center"][2]
        cur = state_pos(client.getMultirotorState())
        print(f"Pre-race: climbing to gate altitude z={g1_z:.1f} ...")
        client.moveToPositionAsync(
            float(cur[0]), float(cur[1]), float(g1_z), 3.0
        ).join()
        print("Pre-race: flying to gate 1 approach point ...")
        client.moveToPositionAsync(
            float(g1_app[0]), float(g1_app[1]), float(g1_z), 3.0
        ).join()
        client.hoverAsync().join()
        time.sleep(0.5)
        print("Pre-race alignment complete. Starting race loop.")

    cv2.namedWindow("Drone Racing Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Racing Dashboard", DASH_W, DASH_H)

    gate_idx   = 0
    phase      = "APPROACH"
    gate_start = time.time()
    tracker    = GateTracker()
    stuck_det  = StuckDetector()
    cv_nx_out  = 0.0
    cv_ny_out  = 0.0
    rl_seg_idx = 0   # current segment on racing line

    try:
        while True:
            t_loop = time.time()

            state         = client.getMultirotorState()
            drone_pos     = state_pos(state)
            drone_yaw     = quat_to_yaw(state.kinematics_estimated.orientation)
            drone_quat    = state.kinematics_estimated.orientation
            drone_vel_vec = state_vel(state)
            gate_elapsed  = time.time() - gate_start
            cur_speed     = norm(drone_vel_vec)

            # --- Camera ---
            responses = client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
            )
            if responses and responses[0].height > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8,
                                     dtype=np.uint8)
                frame = img1d.reshape(responses[0].height,
                                       responses[0].width, 3).copy()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # --- CV pipeline ---
            hsv_m   = multi_hsv_mask(frame)
            edge_m  = edge_mask(frame)
            raw_det = find_best_gate_contour(hsv_m, edge_m, frame.shape)
            tracker.update(raw_det)

            # --- Completion check ---
            if gate_idx >= len(path):
                done = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)
                cv2.putText(done, "COURSE COMPLETE",
                            (DASH_W // 2 - 230, DASH_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, C_GREEN, 4)
                cv2.imshow("Drone Racing Dashboard", done)
                cv2.waitKey(2500)
                break

            gate = path[gate_idx]

            # --- Timeout ---
            if gate_elapsed > MAX_GATE_TIME:
                print(f"  [TIMEOUT] Skipping gate {gate_idx + 1}")
                gate_idx  += 1
                phase      = "APPROACH"
                gate_start = time.time()
                tracker.reset()
                stuck_det.reset()
                continue

            # --- Progress & distance ---
            gate_progress = signed_progress(drone_pos, gate["center"],
                                            gate["forward"])
            dist_gate     = norm(gate["center"] - drone_pos)
            z_err_m       = gate["center"][2] - drone_pos[2]

            # --- Auto-skip ---
            if phase in ("APPROACH", "CENTER"):
                if gate_progress > AUTO_SKIP_PROGRESS:
                    print(f"  [AUTO-SKIP] Past gate {gate_idx + 1} "
                          f"(prg={gate_progress:+.2f})")
                    gate_idx  += 1
                    phase      = "APPROACH"
                    gate_start = time.time()
                    tracker.reset()
                    stuck_det.reset()
                    continue

            # --- Commit zone ---
            in_commit = (phase == "CENTER" and
                         abs(gate_progress) < COMMIT_DIST and
                         abs(z_err_m) < COMMIT_ALT_TOL)

            # =============================================================
            # PURE PURSUIT on racing line
            # =============================================================
            # Minimum segment = current gate's first waypoint (3 per gate)
            min_seg = gate_idx * 3
            rl_seg_idx, rl_seg_t, _ = find_closest_segment(
                racing_line, drone_pos, min_seg=min_seg)

            # Phase-dependent max speed
            if phase == "APPROACH":
                max_speed = MAX_APPROACH_SPEED
            elif phase == "CENTER":
                max_speed = MAX_CENTER_SPEED
            else:
                max_speed = MAX_EXIT_SPEED

            # Speed-proportional lookahead
            lookahead_d = clamp(cur_speed * LOOKAHEAD_SPEED_SCALE,
                                LOOKAHEAD_MIN, LOOKAHEAD_MAX)
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
            if phase == "APPROACH" and dist_gate < APPROACH_TOL:
                phase = "CENTER"
                stuck_det.reset()
                continue
            if phase == "CENTER" and gate_progress > GATE_CROSS_EPS:
                phase = "EXIT"
                stuck_det.reset()
                continue
            if phase == "EXIT":
                past_exit = gate_progress > EXIT_DIST * 0.6
                close_to_next = (gate_idx + 1 < len(path) and
                    norm(path[gate_idx + 1]["approach"] - drone_pos) <
                    APPROACH_TOL * 1.5)
                if dist_target < EXIT_TOL or past_exit or close_to_next:
                    gate_idx  += 1
                    phase      = "APPROACH"
                    gate_start = time.time()
                    tracker.reset()
                    stuck_det.reset()
                    continue

            # --- Direction & yaw ---
            direction = unit(to_target)
            target_yaw   = math.atan2(direction[1], direction[0])
            gate_fwd_yaw = math.atan2(gate["forward"][1],
                                       gate["forward"][0])
            if dist_gate < YAW_ALIGN_START_DIST:
                yaw_blend = clamp(
                    1.0 - (dist_gate - YAW_ALIGN_FULL_DIST) /
                          (YAW_ALIGN_START_DIST - YAW_ALIGN_FULL_DIST),
                    0.0, 0.7)
                desired_yaw = angle_lerp(target_yaw, gate_fwd_yaw, yaw_blend)
            else:
                desired_yaw = target_yaw

            yaw_err     = math.atan2(math.sin(desired_yaw - drone_yaw),
                                      math.cos(desired_yaw - drone_yaw))
            yaw_err_deg = math.degrees(yaw_err)

            # --- Speed ---
            speed_scale = compute_speed_scale(yaw_err, dist_target, phase)
            speed       = max_speed * speed_scale
            if phase == "CENTER":
                speed = max(speed, MIN_CENTER_SPEED)
            vel_desired = direction * speed

            # --- Lateral corrections ---
            lat_gain = (LATERAL_GAIN_CENTER   if phase == "CENTER"   else
                        LATERAL_GAIN_APPROACH  if phase == "APPROACH" else 0.0)
            lat_corr    = lateral_correction_vec(
                drone_pos, gate["center"], gate["forward"],
                lat_gain, LATERAL_MAX)
            lat_err_mag = norm(lat_corr) / max(lat_gain, 0.01)

            cv_corr, cv_nx_out, cv_ny_out = cv_steer_correction(
                tracker, frame.shape, gate["forward"], phase, in_commit)

            if tracker.current is not None:
                combined_lat = lerp(lat_corr, cv_corr, CV_STEER_BLEND)
            else:
                combined_lat = lat_corr

            # --- Commit zone ---
            if in_commit:
                combined_lat = combined_lat * COMMIT_LAT_DAMP
                fwd_component = float(np.dot(vel_desired, gate["forward"]))
                if fwd_component < COMMIT_MIN_FWD_SPEED:
                    vel_desired = vel_desired + gate["forward"] * (
                        COMMIT_MIN_FWD_SPEED - fwd_component)

            vel_cmd     = vel_desired + combined_lat
            vel_cmd     = lerp(vel_cmd, drone_vel_vec, VEL_FF_ALPHA)
            vel_cmd[2]  = altitude_cmd(drone_pos[2], gate["center"][2],
                                       phase)

            client.moveByVelocityAsync(
                float(vel_cmd[0]),
                float(vel_cmd[1]),
                float(vel_cmd[2]),
                CONTROL_DT * 1.5,
                yaw_mode=airsim.YawMode(
                    is_rate=False,
                    yaw_or_rate=float(math.degrees(desired_yaw))),
            )

            # =============================================================
            # PROJECT PATH ONTO CAMERA for visualisation
            # =============================================================
            vis_pts = sample_path_ahead(racing_line, rl_seg_idx,
                                        rl_seg_t, PATH_VIS_DIST,
                                        PATH_VIS_SAMPLES)
            path_px = project_points_to_camera(vis_pts, drone_pos,
                                                drone_quat, CAM_W, DASH_H)

            # Project upcoming gate centers (current + next 2)
            gate_centers_3d = []
            for gi in range(gate_idx, min(gate_idx + 3, len(path))):
                gate_centers_3d.append(path[gi]["center"])
            gate_center_px_list = project_points_to_camera(
                gate_centers_3d, drone_pos, drone_quat, CAM_W, DASH_H)

            # --- Dashboard ---
            dash = build_dashboard(
                frame, tracker, path, gate_idx, phase, gate,
                drone_pos, drone_yaw, drone_vel_vec, target,
                speed, max_speed, speed_scale, yaw_err_deg,
                lat_err_mag, combined_lat, gate_progress, dist_gate,
                z_err_m, gate_elapsed, stuck_det, cv_nx_out, cv_ny_out,
                in_commit, lookahead_d, racing_line, rl_seg_idx,
                path_px, gate_center_px_list
            )
            cv2.imshow("Drone Racing Dashboard", dash)

            cv_label = ("%.2f" % tracker.current["score"]
                        if tracker.current else " -- ")
            commit_tag = " COMMIT" if in_commit else ""
            print(
                f"G{gate_idx+1:>2}/{len(path)} | {phase:<8}{commit_tag:<7} | "
                f"{gate['name']:<12} | "
                f"dst={dist_target:5.2f} prg={gate_progress:+5.2f} "
                f"z_err={z_err_m:+5.2f} | "
                f"look={lookahead_d:4.1f} | "
                f"spd={speed:4.1f}({speed_scale*100:.0f}%) | "
                f"yaw={yaw_err_deg:+6.1f} | "
                f"cv={cv_label}"
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