# This script is a more advanced classic AirSim gate-navigation controller
# built around computer vision, PID control, smoothing, and a finite-state machine.
# It detects rectangular gate-like contours from the forward camera, tracks the
# best candidate, and uses separate PID controllers for yaw, altitude, and forward
# speed to guide the drone through gates.
# The controller moves through several states such as searching, aligning,
# approaching, traversing, and recovering, which makes it more structured and
# more stable than simple contour-chasing scripts.
# It is a CV-based autonomy prototype and does not use reinforcement learning.

import airsim
import cv2
import numpy as np
import time
from collections import deque

# =============================================
#  PID CONTROLLER
# =============================================
class PID:
    def __init__(self, kp, ki, kd, limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()

    def update(self, error):
        now = time.time()
        dt = max(now - self._prev_time, 1e-4)
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._prev_error = error
        self._prev_time = now
        if self.limit:
            output = np.clip(output, -self.limit, self.limit)
        return output

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


# =============================================
#  EXPONENTIAL MOVING AVERAGE SMOOTHER
# =============================================
class EMA:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.value = None

    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value


# =============================================
#  GATE DETECTOR
#  Scores contours by how rectangular they are.
#  Gates are rectangles -- we exploit that.
# =============================================
class GateDetector:
    def __init__(self, min_area=3000, rectangularity_thresh=0.55):
        self.min_area = min_area
        self.rect_thresh = rectangularity_thresh

    def detect(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CLAHE -- handles variable indoor lighting
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Otsu threshold for adaptive edge detection
        high_t, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(blurred, int(0.5 * high_t), int(high_t))
        edges = cv2.dilate(edges, None, iterations=2)

        # Mask sky and floor -- gates won't be there
        edges[:int(h * 0.30), :] = 0
        edges[int(h * 0.88):, :] = 0

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        gates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            # Rectangularity score: how close is this to a rectangle?
            rect = cv2.minAreaRect(c)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area < 1:
                continue
            score = area / rect_area  # 1.0 = perfect rectangle

            if score > self.rect_thresh:
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cx = int(rect[0][0])
                cy = int(rect[0][1])
                rw = int(max(rect[1]))
                rh = int(min(rect[1]))
                gates.append({
                    "contour": c,
                    "box": box,
                    "cx": cx,
                    "cy": cy,
                    "width": rw,
                    "height": rh,
                    "area": area,
                    "score": score,
                })

        # Sort by area descending -- biggest gate is the target
        gates.sort(key=lambda g: g["area"], reverse=True)
        return gates, edges


# =============================================
#  HUD RENDERER
# =============================================
def draw_hud(frame, state, gate, screen_cx, screen_cy, v_x, v_z, yaw_rate):
    h, w = frame.shape[:2]

    # Crosshair
    cv2.line(frame, (screen_cx - 20, screen_cy), (screen_cx + 20, screen_cy), (200, 200, 200), 1)
    cv2.line(frame, (screen_cx, screen_cy - 20), (screen_cx, screen_cy + 20), (200, 200, 200), 1)

    # State label color
    colors = {
        "SEARCHING":  (0, 0, 255),
        "ALIGNING":   (0, 165, 255),
        "APPROACHING":(0, 255, 255),
        "TRAVERSING": (0, 255, 0),
        "RECOVERING": (255, 0, 255),
    }
    color = colors.get(state, (255, 255, 255))

    cv2.putText(frame, f"STATE: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"vx:{v_x:+.2f}  vz:{v_z:+.2f}  yaw:{yaw_rate:+.1f}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if gate:
        cx, cy = gate["cx"], gate["cy"]
        # Rotated bounding box
        cv2.drawContours(frame, [gate["box"]], 0, color, 2)
        # Center dot
        cv2.circle(frame, (cx, cy), 6, color, -1)
        # Line from crosshair to gate center
        cv2.line(frame, (screen_cx, screen_cy), (cx, cy), color, 1)
        # Gate info
        cv2.putText(frame, f"GATE  area:{gate['area']}  score:{gate['score']:.2f}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Border flash on TRAVERSING
    if state == "TRAVERSING":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 4)

    return frame


# =============================================
#  MAIN
# =============================================
def main():
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(1.5)
    print("Airborne -- Perception Stack Online.")

    detector = GateDetector(min_area=3000, rectangularity_thresh=0.55)

    # PID controllers
    pid_yaw = PID(kp=0.10, ki=0.001, kd=0.05, limit=40.0)
    pid_alt = PID(kp=0.018, ki=0.0005, kd=0.008, limit=2.0)
    pid_fwd = PID(kp=0.004, ki=0.0,   kd=0.002, limit=5.0)

    # Output smoothers
    smooth_yaw = EMA(alpha=0.35)
    smooth_vz  = EMA(alpha=0.35)
    smooth_vx  = EMA(alpha=0.5)

    state = "SEARCHING"
    traverse_timer = 0
    recover_timer  = 0
    no_gate_count  = 0
    gates_passed   = 0

    # Rolling history for gate confidence (avoids reacting to single-frame noise)
    gate_history = deque(maxlen=5)

    cv2.namedWindow("Guidance HUD",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("Edge Map",      cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Guidance HUD", 960, 540)
    cv2.resizeWindow("Edge Map",     480, 270)

    while True:
        # -- Grab frame ------------------------------------------
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        if not responses or responses[0].height == 0:
            continue

        img1d   = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        frame   = img1d.reshape(responses[0].height, responses[0].width, 3).copy()
        h, w    = frame.shape[:2]
        scx, scy = w // 2, h // 2

        # -- Detect gates ----------------------------------------
        gates, edge_map = detector.detect(frame)
        gate_history.append(len(gates) > 0)
        gate_visible = sum(gate_history) >= 2  # require 2 of last 5 frames

        primary = gates[0] if gates else None

        # -- Control outputs (defaults) --------------------------
        v_x, v_z, yaw_rate = 0.0, 0.0, 0.0

        # == FINITE STATE MACHINE ================================

        if state == "SEARCHING":
            # Slow yaw scan -- no forward motion
            yaw_rate = smooth_yaw.update(18.0)
            v_x = 0.0
            no_gate_count = 0

            if gate_visible and primary:
                err_x = primary["cx"] - scx
                if abs(err_x) < 220:       # gate roughly centered
                    state = "ALIGNING"
                    pid_yaw.reset()
                    pid_alt.reset()

        elif state == "ALIGNING":
            # Lock heading and altitude onto gate center before approaching
            if not gate_visible or not primary:
                no_gate_count += 1
                if no_gate_count > 10:
                    state = "SEARCHING"
            else:
                no_gate_count = 0
                err_x = primary["cx"] - scx
                err_y = primary["cy"] - scy

                yaw_rate = smooth_yaw.update(pid_yaw.update(err_x))
                v_z      = smooth_vz.update(pid_alt.update(err_y))

                # Move to APPROACHING only when well-aligned
                if abs(err_x) < 60 and abs(err_y) < 60:
                    state = "APPROACHING"
                    pid_fwd.reset()

        elif state == "APPROACHING":
            if not gate_visible or not primary:
                no_gate_count += 1
                if no_gate_count > 8:
                    state = "RECOVERING"
                    recover_timer = 25
            else:
                no_gate_count = 0
                err_x = primary["cx"] - scx
                err_y = primary["cy"] - scy
                area  = primary["area"]

                yaw_rate = smooth_yaw.update(pid_yaw.update(err_x))
                v_z      = smooth_vz.update(pid_alt.update(err_y))

                # Forward speed scales with distance (smaller area = farther away)
                # Target area ~300k px^2 = gate is close enough to traverse
                area_error = 300000 - area
                v_x = smooth_vx.update(pid_fwd.update(area_error))
                v_x = np.clip(v_x, 1.0, 6.0)   # always moving forward

                # Gate fill threshold -> switch to blind traversal
                if area > 280000:
                    state = "TRAVERSING"
                    traverse_timer = 22   # ~2.2s at 10Hz to punch through
                    gates_passed += 1
                    print(f"Gate {gates_passed} -- TRAVERSING")

        elif state == "TRAVERSING":
            # Fly straight through at speed -- don't react to perception noise
            v_x      = smooth_vx.update(5.5)
            yaw_rate = smooth_yaw.update(0.0)
            v_z      = smooth_vz.update(0.0)
            traverse_timer -= 1
            if traverse_timer <= 0:
                state = "SEARCHING"
                pid_yaw.reset()
                pid_alt.reset()
                pid_fwd.reset()

        elif state == "RECOVERING":
            # Lost the gate mid-approach -- back up slightly and re-scan
            v_x = smooth_vx.update(-1.5)
            yaw_rate = smooth_yaw.update(15.0)
            recover_timer -= 1
            if recover_timer <= 0:
                state = "SEARCHING"

        # -- Send commands ---------------------------------------
        client.moveByVelocityBodyFrameAsync(
            v_x, 0, v_z, 0.1,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        )

        # -- Render HUD ------------------------------------------
        hud = draw_hud(frame.copy(), state, primary, scx, scy, v_x, v_z, yaw_rate)
        cv2.imshow("Guidance HUD", hud)
        cv2.imshow("Edge Map", edge_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Manual abort.")
            break

    # -- Shutdown --------------------------------------------
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    cv2.destroyAllWindows()
    print(f"Session ended. Gates passed: {gates_passed}")


if __name__ == "__main__":
    main()
