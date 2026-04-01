"""
Training Data Collector for YOLOv8 Gate Detector
=================================================
Connects to AirSim, flies slowly past all gates, and auto-labels
every frame using AirSim's ground-truth 3D gate positions.

No manual annotation needed — AirSim gives perfect labels for free.

Run:
    python collect_training_data.py

Output:
    training_data/images/  — JPEG frames
    training_data/labels/  — YOLO-format .txt labels (class cx cy w h, normalized)

When done, run:
    python train_gate_detector.py
"""

import math
import os
import time
import numpy as np
import cv2
from pathlib import Path

# --- Config ---
OUTPUT_DIR    = Path("training_data")
MAX_IMAGES    = 4000          # Stop after this many total saved frames
SAVE_EVERY_N  = 3             # Save every Nth frame (skip near-duplicates)
PATROL_SPEED  = 3.5           # m/s — slow enough to capture clean frames
GATE_SIZE_M   = 1.8           # Physical gate opening size in meters
FOV_H_DEG     = 90.0          # Camera horizontal FOV (matches config.yaml)
MIN_DIST_M    = 2.0           # Don't label gates closer than this
MAX_DIST_M    = 22.0          # Don't label gates farther than this
CONF_MARGIN   = 0.05          # Margin to shrink bbox (avoids labeling gate frame only)
TAKEOFF_Z     = -1.5          # meters (AirSim NED)
PATROL_ALTS   = [-1.5, -2.5, -0.8]  # Vary altitude for diverse angles

(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)


def rotation_matrix_from_quaternion(q):
    """Convert AirSim quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def project_gate_to_image(gate_pos_world, drone_pos, drone_orient,
                           img_w, img_h, fov_h_deg):
    """
    Project a 3D gate center (world frame) to 2D image coordinates.

    Returns (px, py, dist) or None if gate is behind drone / out of frame.
    """
    # Drone rotation matrix (world → body)
    R_body = rotation_matrix_from_quaternion(drone_orient)

    # Gate position relative to drone in world frame
    delta = np.array([
        gate_pos_world.x_val - drone_pos.x_val,
        gate_pos_world.y_val - drone_pos.y_val,
        gate_pos_world.z_val - drone_pos.z_val,
    ])

    # Transform to drone body frame
    gate_body = R_body.T @ delta

    # AirSim body frame: X=forward, Y=right, Z=down
    # Camera points forward (X axis)
    x_fwd = gate_body[0]
    y_right = gate_body[1]
    z_down = gate_body[2]

    dist = float(np.linalg.norm(gate_body))

    # Gate must be in front of drone
    if x_fwd <= 0.1:
        return None

    # Pinhole projection
    fov_h_rad = math.radians(fov_h_deg)
    fx = (img_w / 2.0) / math.tan(fov_h_rad / 2.0)
    fy = fx  # Square pixels

    px = fx * (y_right / x_fwd) + (img_w / 2.0)
    py = fy * (z_down  / x_fwd) + (img_h / 2.0)

    # Must be within image
    if not (0 <= px < img_w and 0 <= py < img_h):
        return None

    return px, py, dist


def compute_bbox(px, py, dist, img_w, img_h, fov_h_deg, gate_size_m):
    """Compute YOLO-format normalized bounding box for a gate at given distance."""
    fov_h_rad = math.radians(fov_h_deg)
    fx = (img_w / 2.0) / math.tan(fov_h_rad / 2.0)

    # Angular size of gate in pixels
    half_box_px = fx * (gate_size_m / 2.0) / dist

    # Expand slightly for better bbox coverage
    half_box_px *= 1.2

    x1 = max(0, px - half_box_px)
    y1 = max(0, py - half_box_px)
    x2 = min(img_w, px + half_box_px)
    y2 = min(img_h, py + half_box_px)

    # YOLO normalized format: cx cy w h
    cx_n = ((x1 + x2) / 2.0) / img_w
    cy_n = ((y1 + y2) / 2.0) / img_h
    w_n  = (x2 - x1) / img_w
    h_n  = (y2 - y1) / img_h

    # Filter degenerate boxes
    if w_n < 0.01 or h_n < 0.01:
        return None

    return cx_n, cy_n, w_n, h_n


def main():
    try:
        import airsimdroneracinglab as airsim
    except ImportError:
        import airsim

    print("=" * 60)
    print("Gate Detector Training Data Collector")
    print("=" * 60)
    print("Connecting to AirSim...")

    client = airsim.MultirotorClient()
    client.race_tier = None   # required by airsimdroneracinglab before API calls
    client.confirmConnection()
    client.enableApiControl()
    client.arm()

    # Discover gates
    obj_list = client.simListSceneObjects("Gate.*")
    gate_names = sorted([n for n in obj_list if "Gate" in n],
                        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    if not gate_names:
        gate_names = sorted(client.simListSceneObjects(".*[Gg]ate.*"))

    print(f"Found {len(gate_names)} gates: {gate_names}")
    if not gate_names:
        print("ERROR: No gates found in scene. Is Soccer Field - Easy loaded?")
        return

    # Get gate positions
    gate_poses = {}
    for name in gate_names:
        pose = client.simGetObjectPose(name)
        gate_poses[name] = pose.position
        print(f"  {name}: ({pose.position.x_val:.1f}, "
              f"{pose.position.y_val:.1f}, {pose.position.z_val:.1f})")

    # Takeoff
    print("\nTaking off...")
    client.takeoffAsync().join()
    time.sleep(1.0)

    img_count = 0
    frame_idx = 0
    patrol_round = 0

    print(f"\nCollecting data... (target: {MAX_IMAGES} images)")
    print("Press Ctrl+C to stop early.\n")

    try:
        while img_count < MAX_IMAGES:
            alt = PATROL_ALTS[patrol_round % len(PATROL_ALTS)]
            patrol_round += 1

            # Fly through each gate slowly
            for gate_idx, gate_name in enumerate(gate_names):
                gate_pos = gate_poses[gate_name]

                # Approach point: 8m in front of gate
                next_gate_pos = gate_poses[gate_names[(gate_idx + 1) % len(gate_names)]]
                direction = np.array([
                    next_gate_pos.x_val - gate_pos.x_val,
                    next_gate_pos.y_val - gate_pos.y_val,
                    0,
                ], dtype=np.float64)
                dist = np.linalg.norm(direction)
                if dist > 0:
                    direction /= dist

                approach_x = gate_pos.x_val - direction[0] * 6.0
                approach_y = gate_pos.y_val - direction[1] * 6.0

                # Move to approach point
                client.moveToPositionAsync(
                    approach_x, approach_y, alt, PATROL_SPEED
                ).join()

                # Fly through gate, capturing frames along the way
                exit_x = gate_pos.x_val + direction[0] * 5.0
                exit_y = gate_pos.y_val + direction[1] * 5.0

                client.moveToPositionAsync(
                    exit_x, exit_y, alt, PATROL_SPEED
                )

                # Capture frames while flying
                capture_start = time.time()
                while time.time() - capture_start < 3.0:
                    frame_idx += 1
                    if frame_idx % SAVE_EVERY_N != 0:
                        time.sleep(0.05)
                        continue

                    # Capture image
                    responses = client.simGetImages([
                        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
                    ])
                    if not responses or responses[0].height == 0:
                        time.sleep(0.05)
                        continue

                    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                    frame = img1d.reshape(responses[0].height,
                                         responses[0].width, 3).copy()
                    img_h, img_w = frame.shape[:2]

                    # Get drone pose for projection
                    state = client.getMultirotorState()
                    drone_pos = state.kinematics_estimated.position
                    drone_orient = state.kinematics_estimated.orientation

                    # Build labels for all visible gates
                    labels = []
                    for g_name, g_pos in gate_poses.items():
                        proj = project_gate_to_image(
                            g_pos, drone_pos, drone_orient,
                            img_w, img_h, FOV_H_DEG
                        )
                        if proj is None:
                            continue
                        px, py, dist = proj

                        if dist < MIN_DIST_M or dist > MAX_DIST_M:
                            continue

                        bbox = compute_bbox(px, py, dist, img_w, img_h,
                                            FOV_H_DEG, GATE_SIZE_M)
                        if bbox is None:
                            continue

                        cx_n, cy_n, w_n, h_n = bbox
                        labels.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

                    # Only save if we have at least one label
                    if not labels:
                        time.sleep(0.05)
                        continue

                    # Save image and label
                    img_name = f"gate_{img_count:05d}.jpg"
                    lbl_name = f"gate_{img_count:05d}.txt"

                    cv2.imwrite(str(OUTPUT_DIR / "images" / img_name), frame)
                    with open(OUTPUT_DIR / "labels" / lbl_name, "w") as f:
                        f.write("\n".join(labels))

                    img_count += 1

                    if img_count % 100 == 0:
                        print(f"  Saved {img_count}/{MAX_IMAGES} images...")

                    if img_count >= MAX_IMAGES:
                        break

                    time.sleep(0.05)

                if img_count >= MAX_IMAGES:
                    break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    print(f"\nDone! Saved {img_count} labeled images to {OUTPUT_DIR}/")
    print("Next step: python train_gate_detector.py")

    client.landAsync().join()
    client.disarm()
    client.disableApiControl()


if __name__ == "__main__":
    main()
