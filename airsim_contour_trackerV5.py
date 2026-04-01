"""
AirSim Drone Racing Lab - Advanced Autonomous Gate Racer v3
===========================================================

GATE ACCURACY FIXES
  - Lateral centering correction: during APPROACH and CENTER a perpendicular
    correction force steers the drone onto the gate center axis, not just
    "near" the gate.  This is the primary fix for clipping gate edges.
  - CV nudge disabled (alpha=0): the nudge was drifting gate positions into
    edges.  CV is used for display and confidence only.
  - CENTER target is the gate center itself (not forward-shifted) so the drone
    crosses the plane cleanly before the EXIT phase begins.

UNIFIED DASHBOARD (single 1280x720 window)
  - Left 2/3: annotated camera feed with gate detection overlay, velocity
    vector arrow, lateral error indicator, and phase badge.
  - Top-right: top-down minimap showing all gate positions, gate forward
    arrows, done/current/future colouring, drone position + heading triangle,
    and current target crosshair.
  - Bottom-right: brain-state panel with filled bars for speed throttle, yaw
    error, lateral offset, gate progress, and CV confidence.
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

TAKEOFF_HEIGHT       = -1.5    # NED z (negative = up)

MAX_APPROACH_SPEED   = 9.0
MAX_CENTER_SPEED     = 7.0     # slower through gate for accuracy
MAX_EXIT_SPEED       = 10.0

MIN_SPEED_FRACTION   = 0.22
YAW_SCALE_EXPONENT   = 1.6
BRAKE_DIST           = 3.5
BRAKE_MIN_FRACTION   = 0.30

VEL_FF_ALPHA         = 0.15    # velocity feed-forward blend

APPROACH_TOL         = 3.8
CENTER_TOL           = 3.5
EXIT_TOL             = 14.0

FIRST_APPROACH_DIST  = 5.5
APPROACH_DIST        = 2.8
EXIT_DIST            = 3.0
CENTER_THROUGH_DIST  = 1.2     # shorter: just past the gate plane

GATE_CROSS_EPS       = 0.20
DYN_APPROACH_FRAC    = 0.45
LOOKAHEAD_BLEND_START = 0.4

# Lateral centering
LATERAL_GAIN_APPROACH = 1.8    # proportional gain on lateral error in APPROACH
LATERAL_GAIN_CENTER   = 3.5    # stronger correction through the gate opening
LATERAL_MAX           = 5.0    # clamp on lateral correction magnitude (m/s)

# CV nudge - disabled (set to 0 to use known positions only)
CV_NUDGE_ALPHA       = 0.0
CV_SCORE_THRESHOLD   = 0.50

MAX_GATE_TIME        = 18.0
ALT_DRIFT_MAX        = 0.8
CONTROL_DT           = 0.08

CANNY_LOW            = 50
CANNY_HIGH           = 130
CONTOUR_MIN_AREA     = 250

HSV_ORANGE_LO = np.array([  5,  90,  80], dtype=np.uint8)
HSV_ORANGE_HI = np.array([ 30, 255, 255], dtype=np.uint8)

# Dashboard layout
DASH_W, DASH_H = 1280, 720
CAM_W          = 854
RIGHT_W        = DASH_W - CAM_W   # 426
MAP_H          = 380
BRAIN_H        = DASH_H - MAP_H   # 340

# Colours (BGR)
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

PHASE_COLORS = {
    "APPROACH": C_YELLOW,
    "CENTER":   C_GREEN,
    "EXIT":     C_ORANGE,
}

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
    names = client.simListSceneObjects(".*[Gg]ate.*")
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
            base_approach = FIRST_APPROACH_DIST
        else:
            gap           = norm(center - gates[i - 1][1])
            base_approach = max(1.5, min(APPROACH_DIST, gap * DYN_APPROACH_FRAC))

        path.append({
            "index":    i,
            "name":     name,
            "center":   center.copy(),
            "approach": center - fwd * base_approach,
            "exit":     center + fwd * EXIT_DIST,
            "forward":  fwd,
        })

    return path


# ---------------------------------------------------------------------------
# FLIGHT CONTROL HELPERS
# ---------------------------------------------------------------------------

def lateral_correction_vec(drone_pos, gate_center, gate_forward, gain, max_mag):
    """
    Returns an XYZ velocity correction that pulls the drone onto the gate
    center axis (the line through gate_center along gate_forward).
    """
    to_gate    = gate_center - drone_pos
    fwd_proj   = float(np.dot(to_gate, gate_forward))
    lat_error  = to_gate - gate_forward * fwd_proj
    correction = lat_error * gain
    mag        = norm(correction)
    if mag > max_mag:
        correction = correction / mag * max_mag
    return correction


def compute_speed_scale(yaw_err_rad, dist_to_target):
    yaw_scale   = max(MIN_SPEED_FRACTION,
                      math.cos(clamp(abs(yaw_err_rad), 0, math.pi / 2))
                      ** YAW_SCALE_EXPONENT)
    brake_scale = (max(BRAKE_MIN_FRACTION, dist_to_target / BRAKE_DIST)
                   if dist_to_target < BRAKE_DIST else 1.0)
    return min(yaw_scale, brake_scale)


def signed_progress(drone_pos, gate_center, gate_forward):
    return float(np.dot(drone_pos - gate_center, gate_forward))


def altitude_guard_z(target_z, drone_z):
    drift = drone_z - target_z
    if abs(drift) > ALT_DRIFT_MAX:
        return target_z + clamp(drift * 0.5, -1.5, 1.5)
    return target_z


# ---------------------------------------------------------------------------
# COMPUTER VISION
# ---------------------------------------------------------------------------

def hsv_gate_mask(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_ORANGE_LO, HSV_ORANGE_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return mask


def edge_mask(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    return cv2.dilate(edges, None, iterations=1)


def find_best_gate_contour(merged_mask):
    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < CONTOUR_MIN_AREA:
            continue
        rect   = cv2.minAreaRect(c)
        rw, rh = rect[1]
        ra     = rw * rh
        if ra < 1:
            continue
        score  = area / ra
        aspect = max(rw, rh) / (min(rw, rh) + 1e-6)
        ap     = 1.0 if aspect < 3.0 else max(0.4, 1.0 - (aspect - 3.0) * 0.1)
        score *= ap
        if score > CV_SCORE_THRESHOLD:
            if best is None or score > best["score"]:
                cx, cy = int(rect[0][0]), int(rect[0][1])
                best   = {"score": score, "centroid_px": (cx, cy),
                           "contour": c, "rect": rect}
    return best


# ---------------------------------------------------------------------------
# DASHBOARD RENDERING
# ---------------------------------------------------------------------------

class TopDownMap:
    def __init__(self, gate_centers, panel_w, panel_h, margin=28):
        xs      = [g[0] for g in gate_centers]
        ys      = [g[1] for g in gate_centers]
        span_x  = max(max(xs) - min(xs), 5.0)
        span_y  = max(max(ys) - min(ys), 5.0)
        uw      = panel_w - 2 * margin
        uh      = panel_h - 2 * margin
        self.scale = min(uw / span_x, uh / span_y)
        self.off_x = margin + (uw - span_x * self.scale) / 2 - min(xs) * self.scale
        self.off_y = margin + (uh - span_y * self.scale) / 2 - min(ys) * self.scale
        self.pw    = panel_w
        self.ph    = panel_h

    def to_px(self, wx, wy):
        px = int(wx * self.scale + self.off_x)
        py = int(wy * self.scale + self.off_y)
        return (clamp(px, 0, self.pw - 1), clamp(py, 0, self.ph - 1))


def draw_top_down_map(path, gate_idx, drone_pos, drone_yaw, target):
    panel      = np.full((MAP_H, RIGHT_W, 3), (18, 18, 28), dtype=np.uint8)
    if not path:
        return panel

    tdm = TopDownMap([g["center"] for g in path], RIGHT_W, MAP_H)

    # Path lines
    for i in range(len(path) - 1):
        p1  = tdm.to_px(path[i]["center"][0],   path[i]["center"][1])
        p2  = tdm.to_px(path[i+1]["center"][0], path[i+1]["center"][1])
        col = (60, 100, 60) if i < gate_idx else C_DGRAY
        cv2.line(panel, p1, p2, col, 1, cv2.LINE_AA)

    # Gates
    for i, g in enumerate(path):
        px, py   = tdm.to_px(g["center"][0], g["center"][1])
        fwd      = g["forward"]
        perp     = np.array([-fwd[1], fwd[0]])
        is_cur   = (i == gate_idx)
        col      = C_CYAN if is_cur else ((60, 160, 60) if i < gate_idx else C_GRAY)
        half_px  = int(10 / tdm.scale) if is_cur else int(6 / tdm.scale)
        half_px  = max(half_px, 4)
        a        = tdm.to_px(g["center"][0] + perp[0] * half_px / tdm.scale,
                              g["center"][1] + perp[1] * half_px / tdm.scale)
        b        = tdm.to_px(g["center"][0] - perp[0] * half_px / tdm.scale,
                              g["center"][1] - perp[1] * half_px / tdm.scale)
        cv2.line(panel, a, b, col, 3 if is_cur else 2, cv2.LINE_AA)

        # Forward arrow
        arw = tdm.to_px(g["center"][0] + fwd[0] * 2.0,
                          g["center"][1] + fwd[1] * 2.0)
        cv2.arrowedLine(panel, (px, py), arw, col, 1, tipLength=0.4,
                         line_type=cv2.LINE_AA)
        cv2.putText(panel, str(i + 1), (px + 5, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)

    # Target crosshair
    tx, ty = tdm.to_px(target[0], target[1])
    cv2.line(panel, (tx - 7, ty), (tx + 7, ty), C_ORANGE, 1)
    cv2.line(panel, (tx, ty - 7), (tx, ty + 7), C_ORANGE, 1)
    cv2.circle(panel, (tx, ty), 3, C_ORANGE, -1)

    # Drone triangle
    dx, dy  = tdm.to_px(drone_pos[0], drone_pos[1])
    tip_len = 11
    tip     = (int(dx + math.cos(drone_yaw) * tip_len),
                int(dy + math.sin(drone_yaw) * tip_len))
    left    = (int(dx + math.cos(drone_yaw + 2.4) * 6),
                int(dy + math.sin(drone_yaw + 2.4) * 6))
    right   = (int(dx + math.cos(drone_yaw - 2.4) * 6),
                int(dy + math.sin(drone_yaw - 2.4) * 6))
    cv2.fillPoly(panel, [np.array([tip, left, right])], C_RED)
    cv2.polylines(panel, [np.array([tip, left, right])], True, C_WHITE, 1)

    cv2.putText(panel, "TOP-DOWN MAP", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)
    return panel


def _bar(panel, x, y, w, h, frac, fg, label="", val_str=""):
    frac = clamp(frac, 0.0, 1.0)
    cv2.rectangle(panel, (x, y), (x + w, y + h), C_DGRAY, -1)
    fw = max(2, int(w * frac))
    cv2.rectangle(panel, (x, y), (x + fw, y + h), fg, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), C_GRAY, 1)
    if label:
        cv2.putText(panel, label,   (x - 2, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GRAY, 1)
    if val_str:
        cv2.putText(panel, val_str, (x + w + 4, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_WHITE, 1)


def _cbar(panel, x, y, w, h, frac, fg, label="", val_str=""):
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
        cv2.putText(panel, label,   (x - 2, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GRAY, 1)
    if val_str:
        cv2.putText(panel, val_str, (x + w + 4, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_WHITE, 1)


def draw_brain_panel(phase, gate_idx, total_gates, gate_name,
                     speed, max_speed, yaw_err_deg, lat_err_mag,
                     gate_progress, dist_gate, cv_score, vel, elapsed_s):
    panel      = np.full((BRAIN_H, RIGHT_W, 3), (14, 14, 22), dtype=np.uint8)
    pad_x      = 10
    bar_x      = 88
    bar_w      = RIGHT_W - bar_x - 52
    row_h      = 28
    y0         = 26

    # Title
    cv2.putText(panel, "BRAIN STATE", (pad_x, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_GRAY, 1)

    # Phase badge
    pcol = PHASE_COLORS.get(phase, C_WHITE)
    txt  = f"  {phase}  "
    (bw, bh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    bx = RIGHT_W - bw - pad_x - 4
    cv2.rectangle(panel, (bx - 4, 2), (bx + bw + 2, 2 + bh + 6), pcol, -1)
    cv2.putText(panel, txt, (bx, 2 + bh + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_BLACK, 2)

    # Gate info
    cv2.putText(panel, f"Gate {gate_idx + 1}/{total_gates}  {gate_name}",
                (pad_x, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_WHITE, 1)
    y0 += 14

    cv2.line(panel, (pad_x, y0), (RIGHT_W - pad_x, y0), C_DGRAY, 1)
    y0 += 10

    _bar(panel, bar_x, y0, bar_w, 14, speed / max(max_speed, 0.1),
         C_GREEN, "SPEED", f"{speed:.1f}/{max_speed:.0f}")
    y0 += row_h

    _cbar(panel, bar_x, y0, bar_w, 14, yaw_err_deg / 90.0,
          C_ORANGE, "YAW ERR", f"{yaw_err_deg:+.1f}")
    y0 += row_h

    lat_frac = min(lat_err_mag / 2.5, 1.0)
    _bar(panel, bar_x, y0, bar_w, 14, lat_frac,
         C_RED if lat_frac > 0.6 else C_YELLOW,
         "LAT ERR", f"{lat_err_mag:.2f}m")
    y0 += row_h

    _cbar(panel, bar_x, y0, bar_w, 14, clamp(gate_progress / 4.0, -1, 1),
          C_GREEN if gate_progress > 0 else C_YELLOW,
          "PROGRESS", f"{gate_progress:+.2f}m")
    y0 += row_h

    _bar(panel, bar_x, y0, bar_w, 14, clamp(1.0 - dist_gate / 25.0, 0, 1),
         C_BLUE, "DIST GATE", f"{dist_gate:.1f}m")
    y0 += row_h

    cv_col = C_GREEN if cv_score > 0.7 else (C_YELLOW if cv_score > 0.5 else C_RED)
    _bar(panel, bar_x, y0, bar_w, 14, cv_score, cv_col,
         "CV CONF", f"{cv_score:.2f}" if cv_score > 0 else "NONE")
    y0 += row_h

    cv2.line(panel, (pad_x, y0), (RIGHT_W - pad_x, y0), C_DGRAY, 1)
    y0 += 10

    spd_3d = norm(vel)
    cv2.putText(panel,
                f"Vx:{vel[0]:+5.1f} Vy:{vel[1]:+5.1f} Vz:{vel[2]:+5.1f}",
                (pad_x, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1)
    y0 += 20
    cv2.putText(panel,
                f"|V|={spd_3d:.2f} m/s   gate_t={elapsed_s:.1f}s",
                (pad_x, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)

    return panel


def draw_camera_panel(frame, best_cv, phase, gate_name,
                       drone_vel, yaw_err_deg, lat_corr):
    cam   = cv2.resize(frame, (CAM_W, DASH_H), interpolation=cv2.INTER_LINEAR)
    h, w  = cam.shape[:2]
    cw, ch = w // 2, h // 2

    # Gate detection box and centroid line
    if best_cv is not None:
        sx = CAM_W / frame.shape[1]
        sy = DASH_H / frame.shape[0]
        box = cv2.boxPoints(best_cv["rect"])
        box[:, 0] *= sx
        box[:, 1] *= sy
        cv2.drawContours(cam, [np.int32(box)], 0, C_GREEN, 2)
        px = int(best_cv["centroid_px"][0] * sx)
        py = int(best_cv["centroid_px"][1] * sy)
        cv2.circle(cam, (px, py), 8, C_GREEN, -1)
        cv2.circle(cam, (px, py), 8, C_WHITE,  1)
        cv2.line(cam, (cw, ch), (px, py), (0, 200, 120), 1, cv2.LINE_AA)
        cv2.putText(cam, f"CV {best_cv['score']:.2f}",
                    (px + 12, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_GREEN, 1)

    # Crosshair + aim circle
    cv2.circle(cam, (cw, ch), 32, C_WHITE, 1, cv2.LINE_AA)
    cv2.line(cam, (cw - 24, ch), (cw + 24, ch), C_WHITE, 1)
    cv2.line(cam, (cw, ch - 24), (cw, ch + 24), C_WHITE, 1)

    # Velocity arrow (projected onto camera plane: Y=right, Z=-up in NED)
    spd = norm(drone_vel)
    if spd > 0.3:
        sc   = 5.5
        ax   = int(cw + drone_vel[1] * sc)
        ay   = int(ch - drone_vel[2] * sc)
        cv2.arrowedLine(cam, (cw, ch), (ax, ay), C_CYAN, 2,
                         tipLength=0.28, line_type=cv2.LINE_AA)
        cv2.putText(cam, f"{spd:.1f}m/s",
                    (ax + 6, ay - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_CYAN, 1)

    # Lateral error arrow below crosshair
    lat_mag = norm(lat_corr[:2])
    if lat_mag > 0.05:
        # lat_corr[1] is Y axis (right in NED)
        bar_len = int(clamp(lat_corr[1] * 10, -70, 70))
        col     = C_RED if abs(bar_len) > 35 else C_ORANGE
        cv2.arrowedLine(cam, (cw, ch + 45), (cw + bar_len, ch + 45),
                         col, 2, tipLength=0.25, line_type=cv2.LINE_AA)
        cv2.putText(cam, f"lat {lat_mag:.2f}m",
                    (cw + bar_len + 6, ch + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    # Yaw error spoke from center
    yr      = math.radians(yaw_err_deg)
    yaw_tip = (int(cw + math.sin(yr) * 55), int(ch - math.cos(yr) * 55))
    yaw_col = C_GREEN if abs(yaw_err_deg) < 10 else \
              (C_YELLOW if abs(yaw_err_deg) < 30 else C_RED)
    cv2.line(cam, (cw, ch), yaw_tip, yaw_col, 2, cv2.LINE_AA)
    cv2.circle(cam, yaw_tip, 3, yaw_col, -1)

    # Phase badge (top-left)
    pcol  = PHASE_COLORS.get(phase, C_WHITE)
    badge = f" {phase} "
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
    cv2.rectangle(cam, (8, 6), (8 + bw + 4, 6 + bh + 8), pcol, -1)
    cv2.putText(cam, badge, (10, 6 + bh + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, C_BLACK, 2)

    # Gate name (top-right)
    cv2.putText(cam, f"-> {gate_name}",
                (w - 210, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_WHITE, 1)

    return cam


def build_dashboard(frame, best_cv, path, gate_idx, phase, gate,
                    drone_pos, drone_yaw, drone_vel, target,
                    speed, max_speed, speed_scale, yaw_err_deg,
                    lat_err_mag, lat_corr, gate_progress, dist_gate,
                    cv_score, gate_elapsed):
    dash = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)

    cam_panel = draw_camera_panel(frame, best_cv, phase, gate["name"],
                                   drone_vel, yaw_err_deg, lat_corr)
    dash[:, :CAM_W] = cam_panel

    cv2.line(dash, (CAM_W, 0), (CAM_W, DASH_H), C_DGRAY, 1)

    map_panel = draw_top_down_map(path, gate_idx, drone_pos, drone_yaw, target)
    dash[:MAP_H, CAM_W:] = map_panel

    cv2.line(dash, (CAM_W, MAP_H), (DASH_W, MAP_H), C_DGRAY, 1)

    brain_panel = draw_brain_panel(
        phase, gate_idx, len(path), gate["name"],
        speed, max_speed, yaw_err_deg, lat_err_mag,
        gate_progress, dist_gate, cv_score, drone_vel, gate_elapsed
    )
    dash[MAP_H:, CAM_W:] = brain_panel

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
    start_pos = state_pos(state0)
    climb_tgt = vec3(start_pos[0], start_pos[1], TAKEOFF_HEIGHT)
    client.moveToPositionAsync(float(climb_tgt[0]), float(climb_tgt[1]),
                                float(climb_tgt[2]), 2.0).join()

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

    try:
        while True:
            t_loop = time.time()

            # --- state ---
            state         = client.getMultirotorState()
            drone_pos     = state_pos(state)
            drone_yaw     = quat_to_yaw(state.kinematics_estimated.orientation)
            drone_vel_vec = state_vel(state)
            gate_elapsed  = time.time() - gate_start

            # --- camera ---
            responses = client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
            )
            if responses and responses[0].height > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                frame = img1d.reshape(
                    responses[0].height, responses[0].width, 3).copy()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # --- CV ---
            hsv_mask = hsv_gate_mask(frame)
            edges    = edge_mask(frame)
            merged   = cv2.bitwise_or(hsv_mask, edges)
            best_cv  = find_best_gate_contour(merged)
            cv_score = best_cv["score"] if best_cv is not None else 0.0

            # --- course complete ---
            if gate_idx >= len(path):
                done = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)
                cv2.putText(done, "COURSE COMPLETE",
                            (DASH_W // 2 - 230, DASH_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, C_GREEN, 4)
                cv2.imshow("Drone Racing Dashboard", done)
                cv2.waitKey(2500)
                break

            gate = path[gate_idx]

            # --- gate timeout ---
            if gate_elapsed > MAX_GATE_TIME:
                print(f"  [TIMEOUT] Skipping gate {gate_idx + 1}: {gate['name']}")
                gate_idx  += 1
                phase      = "APPROACH"
                gate_start = time.time()
                continue

            # --- target by phase ---
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
                        0.0, 0.85
                    )
                    target = lerp(gate["exit"], next_app, blend_t)

            to_target     = target - drone_pos
            dist_target   = norm(to_target)
            dist_gate     = norm(gate["center"] - drone_pos)
            gate_progress = signed_progress(
                drone_pos, gate["center"], gate["forward"])

            # --- phase transitions ---
            if phase == "APPROACH" and dist_target < APPROACH_TOL:
                phase = "CENTER"
                continue

            if phase == "CENTER" and gate_progress > GATE_CROSS_EPS:
                phase = "EXIT"
                continue

            if phase == "EXIT" and dist_target < EXIT_TOL:
                gate_idx  += 1
                phase      = "APPROACH"
                gate_start = time.time()
                continue

            # --- velocity command ---
            direction   = unit(to_target)
            desired_yaw = math.atan2(direction[1], direction[0])
            yaw_err     = math.atan2(math.sin(desired_yaw - drone_yaw),
                                      math.cos(desired_yaw - drone_yaw))
            yaw_err_deg = math.degrees(yaw_err)

            speed_scale = compute_speed_scale(yaw_err, dist_target)
            speed       = max_speed * speed_scale
            vel_desired = direction * speed

            # Lateral centering correction
            if phase == "APPROACH":
                lat_gain = LATERAL_GAIN_APPROACH
            elif phase == "CENTER":
                lat_gain = LATERAL_GAIN_CENTER
            else:
                lat_gain = 0.0

            lat_corr    = lateral_correction_vec(
                drone_pos, gate["center"], gate["forward"], lat_gain, LATERAL_MAX)
            lat_err_mag = norm(lat_corr) / max(lat_gain, 0.01)

            vel_cmd     = vel_desired + lat_corr
            vel_cmd     = lerp(vel_cmd, drone_vel_vec, VEL_FF_ALPHA)

            # Altitude guard (override Z)
            guarded_z  = altitude_guard_z(target[2], drone_pos[2])
            vel_cmd[2] = clamp((guarded_z - drone_pos[2]) * 1.5, -3.0, 3.0)

            client.moveByVelocityAsync(
                float(vel_cmd[0]),
                float(vel_cmd[1]),
                float(vel_cmd[2]),
                CONTROL_DT * 1.5,
                yaw_mode=airsim.YawMode(
                    is_rate=False,
                    yaw_or_rate=float(math.degrees(desired_yaw))),
            )

            # --- dashboard ---
            dash = build_dashboard(
                frame, best_cv, path, gate_idx, phase, gate,
                drone_pos, drone_yaw, drone_vel_vec, target,
                speed, max_speed, speed_scale, yaw_err_deg,
                lat_err_mag, lat_corr, gate_progress, dist_gate,
                cv_score, gate_elapsed
            )
            cv2.imshow("Drone Racing Dashboard", dash)

            # --- console ---
            print(
                f"G{gate_idx + 1:>2}/{len(path)} | {phase:<8} | "
                f"{gate['name']:<12} | "
                f"dst={dist_target:5.2f} gate={dist_gate:5.2f} "
                f"prog={gate_progress:+5.2f} | "
                f"lat={lat_err_mag:4.2f} | "
                f"spd={speed:4.1f}({speed_scale*100:.0f}%) | "
                f"yaw={yaw_err_deg:+6.1f} | cv={cv_score:.2f}"
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
