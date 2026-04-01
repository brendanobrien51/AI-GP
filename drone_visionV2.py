# This script is a more advanced classic AirSim vision-tracking prototype.
# It uses the forward camera, improves image contrast with CLAHE, applies
# Otsu-based adaptive edge detection, removes sky and floor regions, and
# tracks the largest valid contour as the main target.
# A finite-state machine handles searching, tracking, and arrived behavior,
# while proportional control and simple obstacle avoidance generate body-frame
# velocity and yaw-rate commands.
# Compared with simpler versions, this script is more robust to lighting changes
# and is meant for stronger contour-based visual navigation experiments.

import airsim
import cv2
import numpy as np

print("Establishing telemetry link...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Executing takeoff sequence...")
client.takeoffAsync().join()
print("Airborne. HD-Perception Stack Online.")

# Initialize the HUD windows
cv2.namedWindow("Industrial HUD", cv2.WINDOW_NORMAL)
cv2.namedWindow("Otsu Edge Map", cv2.WINDOW_NORMAL)

state = "SEARCHING"
arrival_cooldown = 0

while True:
    # 1. Grab the 720p frame
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8) 
    img_bgr = img1d.reshape(responses[0].height, responses[0].width, 3).copy()
    h, w = img_bgr.shape[:2]

    # 2. Advanced Pre-Processing (Lighting Immune)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # HD CLAHE: Surgical contrast normalization across 256 zones (16x16)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    equalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    # 3. Otsu's Method: Mathematical Thresholding
    # Automatically finds the perfect 'cut-off' for edges based on the live histogram
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(blurred, int(low_thresh), int(high_thresh))
    edges = cv2.dilate(edges, None, iterations=2)

    # 4. Apply Optical Blinders to the Edge Map
    # Erase the horizon and floor edges so the drone doesn't 'hit' the sky
    edges[0:int(h * 0.35), :] = 0  # Sky/Horizon
    edges[int(h * 0.85):h, :] = 0  # Floor/Immediate Ground

    # 5. Geometry Analysis
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_objects = [c for c in contours if cv2.contourArea(c) > 4000]
    valid_objects = sorted(valid_objects, key=cv2.contourArea, reverse=True)

    screen_cx, screen_cy = w // 2, h // 2
    v_x, v_z, yaw_rate = 0.0, 0.0, 0.0

    # --- FINITE STATE MACHINE ---

    if state == "ARRIVED":
        cv2.putText(img_bgr, "EDGE REACHED: ROTATING", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
        yaw_rate = 50.0 # Aggressive spin
        arrival_cooldown -= 1
        if arrival_cooldown <= 0:
            state = "SEARCHING"

    elif state == "SEARCHING":
        cv2.putText(img_bgr, "SCANNING SECTOR...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        yaw_rate = 20.0 # Slow, precise scan
        if valid_objects:
            target = valid_objects[0]
            tx, ty, tw, th = cv2.boundingRect(target)
            if abs((tx + tw // 2) - screen_cx) < 200:
                state = "TRACKING"

    elif state == "TRACKING":
        if not valid_objects:
            state = "SEARCHING"
        else:
            target = valid_objects[0]
            tx, ty, tw, th = cv2.boundingRect(target)
            target_cx, target_cy = tx + (tw // 2), ty + (th // 2)
            area = tw * th
            
            # HUD Visuals
            cv2.rectangle(img_bgr, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 3)
            cv2.line(img_bgr, (screen_cx, screen_cy), (target_cx, target_cy), (0, 255, 0), 2)
            
            # Arrival condition (720p scale)
            if area > 350000: 
                state = "ARRIVED"
                arrival_cooldown = 20 
            else:
                cv2.putText(img_bgr, f"TRACKING | Area: {area}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
                # Proportional Control
                yaw_rate = (target_cx - screen_cx) * 0.12
                v_z = (target_cy - screen_cy) * 0.015
                v_x = 2.5 # Constant forward pressure

                # Obstacle Avoidance (Repulsive Force)
                for obs in valid_objects[1:]:
                    ox, oy, ow, oh = cv2.boundingRect(obs)
                    dist_x = (ox + ow // 2) - screen_cx
                    if abs(dist_x) < 300:
                        cv2.rectangle(img_bgr, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 2)
                        yaw_rate += -np.sign(dist_x) * (300 - abs(dist_x)) * 0.15

    # Execute Movement
    client.moveByVelocityBodyFrameAsync(v_x, 0, v_z, 0.1, 
        airsim.DrivetrainType.MaxDegreeOfFreedom, 
        airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate))

    cv2.imshow("Industrial HUD", img_bgr)
    cv2.imshow("Otsu Edge Map", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
cv2.destroyAllWindows()

