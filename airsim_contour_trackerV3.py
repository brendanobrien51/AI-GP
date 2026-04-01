# This script is an improved AirSim Drone Racing Lab gate racer that fixes the
# common “hover in the middle of the gate” problem from simpler phase-based trackers.
# It still reads gate positions directly from the simulator and flies through an
# approach-center-exit path, but during the center phase it targets a point slightly
# through the gate instead of stopping at the gate center.
# The script also uses signed progress across the gate plane to decide when the drone
# has actually crossed the gate, which makes transitions from CENTER to EXIT much more
# natural and keeps forward momentum through the course.
# OpenCV contours are still shown for debug visibility, but flight control is based on
# known simulator gate positions rather than pure camera-based perception.

import math
import re

import airsimdroneracinglab as airsim
import cv2
import numpy as np


TAKEOFF_HEIGHT = -1.5

APPROACH_SPEED = 8
CENTER_SPEED = 8
EXIT_SPEED = 8

FIRST_APPROACH_DIST = 5.5
APPROACH_DIST = 3.0
EXIT_DIST = 3

APPROACH_TOL = 4
CENTER_TOL = 4
EXIT_TOL = 18

CONTROL_DT = 0.10

CENTER_THROUGH_DIST = 1.8
GATE_CROSS_EPS = 0.15

CONTOUR_MIN_AREA = 300
CANNY_LOW = 60
CANNY_HIGH = 140


def clamp(value, low, high):
    return max(low, min(high, value))


def vec3(x, y, z):
    return np.array([float(x), float(y), float(z)], dtype=np.float64)


def norm(v):
    return float(np.linalg.norm(v))


def unit(v):
    n = norm(v)
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return math.atan2(siny_cosp, cosy_cosp)


def pose_position_to_np(pose):
    p = pose.position
    return vec3(p.x_val, p.y_val, p.z_val)


def multirotor_position_to_np(state):
    p = state.kinematics_estimated.position
    return vec3(p.x_val, p.y_val, p.z_val)


def get_gate_names(client):
    names = client.simListSceneObjects(".*[Gg]ate.*")
    cleaned = [name for name in names if isinstance(name, str) and name.strip()]

    def gate_sort_key(name):
        nums = re.findall(r"\d+", name)
        if nums:
            return [int(n) for n in nums]
        return [10**9, name]

    cleaned.sort(key=gate_sort_key)
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


def get_gate_positions(client, gate_names):
    result = []
    for name in gate_names:
        pose = get_object_pose_safe(client, name)
        pos = pose_position_to_np(pose)
        if np.isfinite(pos).all():
            result.append((name, pos))
    return result


def make_gate_path(gates):
    path = []
    total = len(gates)

    for i, (name, center) in enumerate(gates):
        if i == 0:
            if total > 1:
                forward = unit(gates[i + 1][1] - center)
            else:
                forward = vec3(1.0, 0.0, 0.0)
        elif i == total - 1:
            forward = unit(center - gates[i - 1][1])
        else:
            forward = unit(gates[i + 1][1] - gates[i - 1][1])

        if norm(forward) < 1e-6:
            forward = vec3(1.0, 0.0, 0.0)

        approach_dist = FIRST_APPROACH_DIST if i == 0 else APPROACH_DIST
        approach = center - forward * approach_dist
        exit_pt = center + forward * EXIT_DIST

        path.append(
            {
                "index": i,
                "name": name,
                "center": center,
                "approach": approach,
                "exit": exit_pt,
                "forward": forward,
            }
        )

    return path


def find_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    edges = cv2.dilate(edges, None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return edges, contours


def draw_contours(frame, contours):
    debug = frame.copy()
    drawn = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < CONTOUR_MIN_AREA:
            continue

        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]
        score = area / rect_area if rect_area > 1 else 0.0
        box = np.int32(cv2.boxPoints(rect))
        cx, cy = int(rect[0][0]), int(rect[0][1])

        if score > 0.75:
            color = (0, 255, 0)
        elif score > 0.55:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        cv2.drawContours(debug, [box], 0, color, 2)
        cv2.putText(
            debug,
            f"a:{int(area)} s:{score:.2f}",
            (cx - 45, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
        drawn += 1

    return debug, drawn


def signed_gate_progress(drone_pos, gate_center, gate_forward):
    return float(np.dot(drone_pos - gate_center, gate_forward))


def shutdown(client):
    try:
        client.hoverAsync().join()
    except Exception:
        pass

    try:
        client.landAsync().join()
    except Exception:
        pass

    try:
        client.disarm()
    except Exception:
        pass

    try:
        client.disableApiControl()
    except Exception:
        pass

    cv2.destroyAllWindows()


print("Connecting to AirSim Drone Racing Lab...")
client = airsim.MultirotorClient()
client.race_tier = None
client.level_name = ""
client.confirmConnection()
client.enableApiControl()
client.arm()

print("Taking off...")
client.takeoffAsync().join()

state0 = client.getMultirotorState()
start_pos = multirotor_position_to_np(state0)
takeoff_target = vec3(start_pos[0], start_pos[1], TAKEOFF_HEIGHT)
client.moveToPositionAsync(
    float(takeoff_target[0]),
    float(takeoff_target[1]),
    float(takeoff_target[2]),
    1.5,
).join()

gate_names = get_gate_names(client)
if not gate_names:
    shutdown(client)
    raise RuntimeError("No gate objects found in scene.")

print(f"Found {len(gate_names)} gate objects.")
for i, name in enumerate(gate_names):
    print(f"  Gate {i + 1}: {name}")

gate_positions = get_gate_positions(client, gate_names)
path = make_gate_path(gate_positions)

cv2.namedWindow("Forward Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Edges Debug", cv2.WINDOW_NORMAL)
cv2.namedWindow("All Contours", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Forward Camera", 960, 540)
cv2.resizeWindow("Edges Debug", 960, 540)
cv2.resizeWindow("All Contours", 960, 540)

gate_idx = 0
phase = "APPROACH"

try:
    while True:
        state = client.getMultirotorState()
        drone_pos = multirotor_position_to_np(state)
        drone_yaw = quat_to_yaw(state.kinematics_estimated.orientation)
        vel = state.kinematics_estimated.linear_velocity
        drone_vel = vec3(vel.x_val, vel.y_val, vel.z_val)

        responses = client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
        )
        if responses and responses[0].height > 0:
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(responses[0].height, responses[0].width, 3).copy()
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        edges, contours = find_contours(frame)
        contour_debug, contour_count = draw_contours(frame, contours)

        if gate_idx >= len(path):
            cv2.putText(
                frame,
                "COURSE COMPLETE",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )
            cv2.imshow("Forward Camera", frame)
            cv2.imshow("Edges Debug", edges)
            cv2.imshow("All Contours", contour_debug)
            cv2.waitKey(500)
            break

        gate = path[gate_idx]

        if phase == "APPROACH":
            target = gate["approach"]
            speed = APPROACH_SPEED
            tol = APPROACH_TOL
        elif phase == "CENTER":
            target = gate["center"] + gate["forward"] * CENTER_THROUGH_DIST
            speed = CENTER_SPEED
            tol = CENTER_TOL
        else:
            target = gate["exit"]
            speed = EXIT_SPEED
            tol = EXIT_TOL

        to_target = target - drone_pos
        dist_to_target = norm(to_target)
        dist_to_gate = norm(gate["center"] - drone_pos)
        gate_progress = signed_gate_progress(drone_pos, gate["center"], gate["forward"])

        if phase == "APPROACH" and dist_to_target < tol:
            phase = "CENTER"
            continue

        if phase == "CENTER" and gate_progress > GATE_CROSS_EPS:
            phase = "EXIT"
            continue

        if phase == "EXIT" and dist_to_target < tol:
            gate_idx += 1
            phase = "APPROACH"
            continue

        direction = unit(to_target)
        desired_yaw = math.atan2(direction[1], direction[0])
        yaw_error_deg = math.degrees(
            math.atan2(math.sin(desired_yaw - drone_yaw), math.cos(desired_yaw - drone_yaw))
        )

        if abs(yaw_error_deg) > 30.0:
            speed *= 0.55
        elif abs(yaw_error_deg) > 15.0:
            speed *= 0.75

        client.moveToPositionAsync(
            float(target[0]),
            float(target[1]),
            float(target[2]),
            float(speed),
            yaw_mode=airsim.YawMode(
                is_rate=False,
                yaw_or_rate=float(math.degrees(desired_yaw)),
            ),
        )

        h, w = frame.shape[:2]
        cv2.circle(frame, (w // 2, h // 2), 5, (255, 255, 255), -1)

        cv2.putText(frame, f"GATE {gate_idx + 1}/{len(path)}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"PHASE: {phase}", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(frame, f"TARGET: {gate['name']}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"DIST TO TARGET: {dist_to_target:.2f} m", (30, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        cv2.putText(frame, f"DIST TO GATE: {dist_to_gate:.2f} m", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        cv2.putText(frame, f"GATE PROGRESS: {gate_progress:.2f}", (30, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        cv2.putText(frame, f"SPEED CMD: {speed:.2f} m/s", (30, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        cv2.putText(frame, f"YAW ERR: {yaw_error_deg:.1f} deg", (30, 285),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(frame, f"VEL: ({drone_vel[0]:.1f}, {drone_vel[1]:.1f}, {drone_vel[2]:.1f})", (30, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
        cv2.putText(frame, f"CONTOURS: {contour_count}", (30, 355),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        print(
            f"Gate {gate_idx + 1}/{len(path)} | "
            f"phase={phase:<8} | target={gate['name']:<10} | "
            f"dist_target={dist_to_target:>5.2f} | dist_gate={dist_to_gate:>5.2f} | "
            f"progress={gate_progress:>5.2f} | yaw_err={yaw_error_deg:>6.2f} | "
            f"speed={speed:>4.2f} | contours={contour_count}"
        )

        cv2.imshow("Forward Camera", frame)
        cv2.imshow("Edges Debug", edges)
        cv2.imshow("All Contours", contour_debug)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        try:
            cv2.waitKey(int(CONTROL_DT * 1000))
        except Exception:
            pass

finally:
    shutdown(client)

