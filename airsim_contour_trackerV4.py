"""
AirSim Drone Racing Lab - Advanced Autonomous Gate Racer
=========================================================
Improvements over the baseline:

FLIGHT CONTROL
  - Velocity commands (moveByVelocityAsync) instead of position commands,
    giving smooth, momentum-preserving flight with no hard stops at waypoints.
  - Continuous speed scaling: cos(yaw_err)^exponent * proximity brake ramp,
    clamped to a minimum fraction so the drone never stalls.
  - Lookahead trajectory blending: during EXIT the target smoothly interpolates
    from the current exit point toward the NEXT gate's approach point, removing
    the dead pause between gates entirely.
  - Velocity feed-forward: the commanded velocity blends a small fraction of
    current velocity so abrupt direction reversals are damped.
  - Dynamic approach distances: scaled to half the inter-gate spacing so the
    drone never overshoots a tight gate cluster.

COMPUTER VISION
  - Dual-pipeline: HSV color mask (orange gate frames) + Canny edges run in
    parallel; detections from each are merged before scoring.
  - Rectangularity + aspect-ratio scoring selects the most gate-like contour.
  - Pixel centroid of the best contour is converted to a bearing offset and used
    to produce a soft CV refinement nudge on the known simulator gate position.
    The nudge is weighted low so CV noise cannot throw the drone off course but
    will correct a drifting gate position over several frames.
  - Separate debug windows: raw frame, HSV mask, edge contours, merged view.

ROBUSTNESS
  - Per-gate timeout: if the drone cannot clear a gate within MAX_GATE_TIME
    seconds it flags the gate as skipped and moves on, preventing infinite loops.
  - Altitude guard: if the drone drifts more than ALT_DRIFT_MAX metres above or
    below the target Z the controller blends in a corrective Z component.
  - Graceful shutdown with hover -> land -> disarm regardless of exception type.
  - All NumPy calls are guarded for degenerate zero-length vectors.
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

TAKEOFF_HEIGHT = -1.5          # NED z, negative = up

# Maximum commanded speeds per phase (m/s)
MAX_APPROACH_SPEED  = 9.0
MAX_CENTER_SPEED    = 9.0
MAX_EXIT_SPEED      = 10.0

# Speed scaling
MIN_SPEED_FRACTION  = 0.22     # never drop below this fraction of max speed
YAW_SCALE_EXPONENT  = 1.6      # higher => harder slowdown on sharp turns
BRAKE_DIST          = 3.5      # metres from target at which braking begins
BRAKE_MIN_FRACTION  = 0.30     # minimum brake scale (avoids micro-speed near wp)

# Velocity feed-forward blend (0 = pure desired, 1 = pure current vel)
VEL_FF_ALPHA        = 0.18

# Phase transition tolerances (metres)
APPROACH_TOL        = 3.8
CENTER_TOL          = 3.5
EXIT_TOL            = 14.0

# Gate path geometry
FIRST_APPROACH_DIST = 5.5      # standoff for gate 0 (farther for safety)
APPROACH_DIST       = 2.8      # standoff for subsequent gates
EXIT_DIST           = 3.0      # distance past gate center for exit waypoint
CENTER_THROUGH_DIST = 2.0      # target this far *through* the gate in CENTER phase

# Signed-progress threshold to confirm gate crossing
GATE_CROSS_EPS      = 0.20     # metres past gate plane

# Dynamic approach distance: cap approach at this fraction of inter-gate gap
DYN_APPROACH_FRAC   = 0.45

# Lookahead blending during EXIT phase
LOOKAHEAD_BLEND_START = 0.4    # start blending when exit dist drops below this
                                # fraction of EXIT_DIST

# CV refinement
CV_NUDGE_MAX        = 0.4      # max metres the CV hint can shift the gate center
CV_NUDGE_ALPHA      = 0.12     # low-pass weight for CV nudge (higher = faster adapt)
CV_SCORE_THRESHOLD  = 0.55     # min rectangularity score to trust a contour

# Timeout: skip gate if not cleared within this many seconds
MAX_GATE_TIME       = 18.0

# Altitude guard: correct Z if drift exceeds this (metres)
ALT_DRIFT_MAX       = 0.8

# Control loop
CONTROL_DT          = 0.08     # seconds between velocity commands

# CV pipeline thresholds
CANNY_LOW           = 50
CANNY_HIGH          = 130
CONTOUR_MIN_AREA    = 250

# HSV range for typical orange AirSim gate frames
# Tune these if your gates are a different colour
HSV_ORANGE_LO = np.array([  5,  90,  80], dtype=np.uint8)
HSV_ORANGE_HI = np.array([ 30, 255, 255], dtype=np.uint8)

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
    """
    Fetch gate world positions and compute approach / center / exit waypoints
    for every gate.  Dynamic approach distances are capped to DYN_APPROACH_FRAC
    of the inter-gate spacing so the drone never overshoot a tight cluster.
    """
    gates = []
    for name in gate_names:
        pose = get_object_pose_safe(client, name)
        pos  = pose_to_np(pose)
        if np.isfinite(pos).all():
            gates.append((name, pos.copy()))

    path = []
    total = len(gates)

    for i, (name, center) in enumerate(gates):
        # gate forward direction
        if i == 0:
            fwd = unit(gates[1][1] - center) if total > 1 else vec3(1, 0, 0)
        elif i == total - 1:
            fwd = unit(center - gates[i - 1][1])
        else:
            fwd = unit(gates[i + 1][1] - gates[i - 1][1])

        if norm(fwd) < 1e-6:
            fwd = vec3(1, 0, 0)

        # dynamic standoff
        if i == 0:
            base_approach = FIRST_APPROACH_DIST
        else:
            gap = norm(center - gates[i - 1][1])
            base_approach = min(APPROACH_DIST, gap * DYN_APPROACH_FRAC)
            base_approach = max(1.5, base_approach)  # floor for safety

        approach = center - fwd * base_approach
        exit_pt  = center + fwd * EXIT_DIST

        path.append({
            "index":    i,
            "name":     name,
            "center":   center.copy(),
            "center_refined": center.copy(),  # updated by CV nudge
            "approach": approach,
            "exit":     exit_pt,
            "forward":  fwd,
        })

    return path


# ---------------------------------------------------------------------------
# SPEED SCALING
# ---------------------------------------------------------------------------

def compute_speed_scale(yaw_err_rad, dist_to_target):
    """
    Returns a scalar in [MIN_SPEED_FRACTION, 1.0] combining:
      - yaw_scale:   cos(yaw_err)^exponent -- slows on sharp turns
      - brake_scale: linear ramp when close to target
    """
    yaw_scale   = max(MIN_SPEED_FRACTION,
                      math.cos(clamp(abs(yaw_err_rad), 0, math.pi / 2))
                      ** YAW_SCALE_EXPONENT)

    if dist_to_target < BRAKE_DIST:
        brake_scale = max(BRAKE_MIN_FRACTION, dist_to_target / BRAKE_DIST)
    else:
        brake_scale = 1.0

    return min(yaw_scale, brake_scale)


# ---------------------------------------------------------------------------
# COMPUTER VISION PIPELINE
# ---------------------------------------------------------------------------

def hsv_gate_mask(frame):
    """Return binary mask of pixels matching the gate colour (orange by default)."""
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
    edges   = cv2.dilate(edges, None, iterations=1)
    return edges


def score_contour(contour):
    """
    Return (score, rect_center_px, rect_wh) where score is a rectangularity
    metric in [0, 1].  Returns None if contour is too small.
    """
    area = cv2.contourArea(contour)
    if area < CONTOUR_MIN_AREA:
        return None

    rect      = cv2.minAreaRect(contour)
    rw, rh    = rect[1]
    rect_area = rw * rh
    if rect_area < 1:
        return None

    score  = area / rect_area
    aspect = max(rw, rh) / (min(rw, rh) + 1e-6)

    # Penalise very non-square shapes (gates are roughly square frames)
    aspect_pen = 1.0 if aspect < 3.0 else max(0.4, 1.0 - (aspect - 3.0) * 0.1)
    score      = score * aspect_pen

    cx, cy = int(rect[0][0]), int(rect[0][1])
    return score, (cx, cy), (int(rw), int(rh)), contour, rect


def find_best_gate_contour(merged_mask, frame_shape):
    """
    Run contour analysis on a binary mask and return the best gate candidate.
    Returns dict with keys: score, centroid_px, contour, rect
    or None if nothing passes the threshold.
    """
    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        result = score_contour(c)
        if result is None:
            continue
        score, center_px, wh, contour, rect = result
        if score > CV_SCORE_THRESHOLD:
            if best is None or score > best["score"]:
                best = {"score": score, "centroid_px": center_px,
                        "contour": contour, "rect": rect}
    return best


def pixel_bearing_offset(centroid_px, frame_shape, h_fov_deg=90.0):
    """
    Convert pixel centroid to a normalised horizontal & vertical bearing offset
    in [-1, 1].  1.0 = full half-FOV to one side.
    """
    h, w = frame_shape[:2]
    cx, cy = centroid_px
    dx = (cx - w / 2.0) / (w / 2.0)   # [-1, 1]
    dy = (cy - h / 2.0) / (h / 2.0)   # [-1, 1]
    return dx, dy


def cv_refine_gate(gate, centroid_px, frame_shape, drone_pos, drone_yaw):
    """
    Produce a soft nudge to gate['center_refined'] based on the detected pixel
    centroid.  The nudge is in the camera-right and camera-up directions,
    scaled by distance to the gate and clamped to CV_NUDGE_MAX.
    """
    dist = norm(gate["center_refined"] - drone_pos)
    if dist < 0.5:
        return  # too close to trust CV geometry

    dx_norm, dy_norm = pixel_bearing_offset(centroid_px, frame_shape)

    # Camera right = drone yaw + 90 deg in XY plane
    cam_right = vec3(math.cos(drone_yaw + math.pi / 2),
                     math.sin(drone_yaw + math.pi / 2), 0.0)
    cam_up    = vec3(0.0, 0.0, -1.0)  # NED: negative Z is up

    # Angular offset scaled by distance (small-angle approximation)
    nudge = cam_right * (dx_norm * dist * 0.15) \
          + cam_up    * (dy_norm * dist * 0.10)

    # Clamp magnitude
    n = norm(nudge)
    if n > CV_NUDGE_MAX:
        nudge = nudge / n * CV_NUDGE_MAX

    # Low-pass blend into refined center
    gate["center_refined"] = lerp(gate["center_refined"],
                                   gate["center_refined"] + nudge,
                                   CV_NUDGE_ALPHA)


def build_debug_frame(frame, best_contour, edges, hsv_mask,
                       gate_idx, total_gates, phase, gate,
                       dist_target, dist_gate, gate_progress,
                       speed, max_speed, speed_scale,
                       yaw_err_deg, drone_vel, contour_count):
    """Compose annotated debug overlay on frame copy."""
    debug = frame.copy()
    h, w  = debug.shape[:2]

    # Crosshair
    cv2.line(debug, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (255, 255, 255), 1)
    cv2.line(debug, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (255, 255, 255), 1)

    # Best contour highlight
    if best_contour is not None:
        box = np.int32(cv2.boxPoints(best_contour["rect"]))
        cv2.drawContours(debug, [box], 0, (0, 255, 0), 2)
        cx, cy = best_contour["centroid_px"]
        cv2.circle(debug, (cx, cy), 6, (0, 255, 0), -1)
        cv2.line(debug, (w // 2, h // 2), (cx, cy), (0, 200, 255), 1)
        cv2.putText(debug, f"CV:{best_contour['score']:.2f}",
                    (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Phase colour
    phase_col = {"APPROACH": (255, 200, 0),
                 "CENTER":   (0, 255, 180),
                 "EXIT":     (0, 140, 255)}.get(phase, (200, 200, 200))

    def txt(label, val, row, col=(200, 255, 200)):
        cv2.putText(debug, f"{label}: {val}", (20, 35 + row * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, col, 2)

    txt(f"GATE {gate_idx + 1}/{total_gates}", gate["name"],              0, (0, 255, 0))
    txt("PHASE",     phase,                                               1, phase_col)
    txt("DIST TGT",  f"{dist_target:>5.2f} m",                           2)
    txt("DIST GATE", f"{dist_gate:>5.2f} m",                             3)
    txt("PROGRESS",  f"{gate_progress:>+5.2f} m",                        4)
    txt("SPEED",     f"{speed:4.1f}/{max_speed:.1f} m/s ({speed_scale*100:.0f}%)", 5)
    txt("YAW ERR",   f"{yaw_err_deg:>+6.1f} deg",                        6, (0, 165, 255))
    txt("VEL",       f"({drone_vel[0]:.1f}, {drone_vel[1]:.1f}, {drone_vel[2]:.1f})", 7, (200, 200, 255))
    txt("CONTOURS",  contour_count,                                       8, (0, 165, 255))

    return debug


# ---------------------------------------------------------------------------
# FLIGHT LOOP HELPERS
# ---------------------------------------------------------------------------

def signed_progress(drone_pos, gate_center, gate_forward):
    return float(np.dot(drone_pos - gate_center, gate_forward))


def altitude_guard_z(target_z, drone_z):
    """Return a corrected z if the drone is drifting away from target altitude."""
    drift = drone_z - target_z  # positive = drone is below target in NED
    if abs(drift) > ALT_DRIFT_MAX:
        correction = clamp(drift * 0.5, -1.5, 1.5)
        return target_z + correction
    return target_z


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
    client = airsim.MultirotorClient()
    client.race_tier  = None   # required by simGetObjectPose internally
    client.level_name = ""     # required by simGetObjectPose internally
    client.confirmConnection()
    client.enableApiControl()
    client.arm()

    print("Taking off...")
    client.takeoffAsync().join()

    state0     = client.getMultirotorState()
    start_pos  = state_pos(state0)
    climb_tgt  = vec3(start_pos[0], start_pos[1], TAKEOFF_HEIGHT)
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

    for win in ["Forward Camera", "HSV Mask", "Edge Contours", "Merged Detection"]:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 640, 360)

    gate_idx   = 0
    phase      = "APPROACH"
    gate_start = time.time()

    try:
        while True:
            t_loop = time.time()

            # --- drone state ---
            state     = client.getMultirotorState()
            drone_pos = state_pos(state)
            drone_yaw = quat_to_yaw(state.kinematics_estimated.orientation)
            drone_vel_vec = state_vel(state)

            # --- camera ---
            responses = client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
            )
            if responses and responses[0].height > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                frame = img1d.reshape(responses[0].height, responses[0].width, 3).copy()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # --- CV pipeline ---
            hsv_mask   = hsv_gate_mask(frame)
            edges      = edge_mask(frame)
            merged     = cv2.bitwise_or(hsv_mask, edges)
            best_cv    = find_best_gate_contour(merged, frame.shape)

            # Count all valid contours for HUD
            all_contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            contour_count = sum(1 for c in all_contours
                                if cv2.contourArea(c) >= CONTOUR_MIN_AREA)

            # --- course complete ---
            if gate_idx >= len(path):
                cv2.putText(frame, "COURSE COMPLETE", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
                cv2.imshow("Forward Camera", frame)
                cv2.waitKey(1500)
                break

            gate = path[gate_idx]

            # CV refinement of gate center
            if best_cv is not None and phase != "EXIT":
                cv_refine_gate(gate, best_cv["centroid_px"], frame.shape,
                               drone_pos, drone_yaw)

            # --- gate timeout ---
            if time.time() - gate_start > MAX_GATE_TIME:
                print(f"  [TIMEOUT] Skipping gate {gate_idx + 1}: {gate['name']}")
                gate_idx   += 1
                phase       = "APPROACH"
                gate_start  = time.time()
                continue

            # --- target & max speed by phase ---
            if phase == "APPROACH":
                target    = gate["approach"]
                max_speed = MAX_APPROACH_SPEED
                tol       = APPROACH_TOL
            elif phase == "CENTER":
                # Aim slightly through the refined center
                target    = gate["center_refined"] + gate["forward"] * CENTER_THROUGH_DIST
                max_speed = MAX_CENTER_SPEED
                tol       = CENTER_TOL
            else:  # EXIT
                target    = gate["exit"]
                max_speed = MAX_EXIT_SPEED
                tol       = EXIT_TOL

                # Lookahead blend: start steering toward next approach early
                if gate_idx + 1 < len(path):
                    next_approach = path[gate_idx + 1]["approach"]
                    blend_t = clamp(
                        1.0 - (norm(gate["exit"] - drone_pos) /
                               max(EXIT_DIST * LOOKAHEAD_BLEND_START, 0.01)),
                        0.0, 0.85
                    )
                    target = lerp(gate["exit"], next_approach, blend_t)

            to_target     = target - drone_pos
            dist_target   = norm(to_target)
            dist_gate     = norm(gate["center_refined"] - drone_pos)
            gate_progress = signed_progress(drone_pos, gate["center_refined"],
                                            gate["forward"])

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
            direction     = unit(to_target)
            desired_yaw   = math.atan2(direction[1], direction[0])
            yaw_err       = math.atan2(math.sin(desired_yaw - drone_yaw),
                                        math.cos(desired_yaw - drone_yaw))
            yaw_err_deg   = math.degrees(yaw_err)

            speed_scale   = compute_speed_scale(yaw_err, dist_target)
            speed         = max_speed * speed_scale

            # Desired velocity vector
            vel_desired   = direction * speed

            # Velocity feed-forward (blend with current vel for smoother commands)
            vel_cmd = lerp(vel_desired, drone_vel_vec, VEL_FF_ALPHA)

            # Altitude guard: correct Z component if drifting
            target_z      = target[2]
            guarded_z     = altitude_guard_z(target_z, drone_pos[2])
            z_correction  = clamp((guarded_z - drone_pos[2]) * 1.5, -3.0, 3.0)
            vel_cmd[2]    = z_correction  # override z with altitude correction

            client.moveByVelocityAsync(
                float(vel_cmd[0]),
                float(vel_cmd[1]),
                float(vel_cmd[2]),
                CONTROL_DT * 1.5,         # slightly longer than loop time for smooth hand-off
                yaw_mode=airsim.YawMode(is_rate=False,
                                         yaw_or_rate=float(math.degrees(desired_yaw))),
            )

            # --- build & show debug frames ---
            debug_main = build_debug_frame(
                frame, best_cv, edges, hsv_mask,
                gate_idx, len(path), phase, gate,
                dist_target, dist_gate, gate_progress,
                speed, max_speed, speed_scale,
                yaw_err_deg, drone_vel_vec, contour_count
            )

            # HSV mask visualisation (convert to BGR for display)
            hsv_display = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)

            # Edge visualisation
            edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # Merged mask visualisation
            merged_display = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)
            if best_cv is not None:
                box = np.int32(cv2.boxPoints(best_cv["rect"]))
                cv2.drawContours(merged_display, [box], 0, (0, 255, 0), 2)

            cv2.imshow("Forward Camera",   debug_main)
            cv2.imshow("HSV Mask",         hsv_display)
            cv2.imshow("Edge Contours",    edge_display)
            cv2.imshow("Merged Detection", merged_display)

            # --- console log ---
            print(
                f"G{gate_idx + 1:>2}/{len(path)} | {phase:<8} | {gate['name']:<12} | "
                f"dst={dist_target:5.2f} gate={dist_gate:5.2f} prog={gate_progress:+5.2f} | "
                f"spd={speed:4.1f}/{max_speed:.0f}({speed_scale*100:.0f}%) | "
                f"yaw={yaw_err_deg:+6.1f} | "
                f"cv={'%.2f' % best_cv['score'] if best_cv else ' --- '}"
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Pace the loop to CONTROL_DT
            elapsed = time.time() - t_loop
            sleep_t = CONTROL_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        shutdown(client)


if __name__ == "__main__":
    main()
"""

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


