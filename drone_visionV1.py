# This script is a classic AirSim monocular-vision drone tracker.
# It uses the forward RGB camera, converts the image into an edge map,
# removes sky and floor regions, finds large contours, and treats the
# biggest valid contour as the main visual target.
# A small finite-state machine switches between searching, tracking,
# and arrived behaviors, while simple proportional control commands
# forward speed, vertical motion, and yaw rate to chase the target.
# It is useful as an early computer-vision navigation prototype and
# does not use reinforcement learning or known gate coordinates.

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
print("Airborne. Monocular Vision Stack Online.")

cv2.namedWindow("Monocular HUD", cv2.WINDOW_NORMAL)
cv2.namedWindow("AI Vision Mask", cv2.WINDOW_NORMAL)

state = "SEARCHING"
arrival_cooldown = 0

while True:
    # 1. Pull the standard 720p RGB camera feed
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8) 
    img_bgr = img1d.reshape(responses[0].height, responses[0].width, 3).copy()
    
    h, w = img_bgr.shape[:2]
    
   # 2. Geometric Mapping (Process the raw image first)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    
    # 3. THE FIX: Apply Optical Blinders to the Edge Map
    # Erase the edges found in the sky and on the floor
    edges[0:int(h * 0.35), :] = 0  # Black out the top 35%
    edges[int(h * 0.85):h, :] = 0  # Black out the bottom 15%
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_objects = [c for c in contours if cv2.contourArea(c) > 3000]
    valid_objects = sorted(valid_objects, key=cv2.contourArea, reverse=True)

    screen_cx = w // 2
    screen_cy = h // 2
    v_x, v_z, yaw_rate = 0.0, 0.0, 0.0

    # --- THE FINITE STATE MACHINE ---

    if state == "ARRIVED":
        cv2.putText(img_bgr, "TARGET REACHED! EVASIVE MANEUVER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
        yaw_rate = 45.0  
        arrival_cooldown -= 1
        if arrival_cooldown <= 0:
            state = "SEARCHING"
            
    elif state == "SEARCHING":
        cv2.putText(img_bgr, "SCANNING SECTOR...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        yaw_rate = 20.0  # Slowed down slightly to prevent camera tearing/blur
        
        if valid_objects:
            target = valid_objects[0]
            tx, ty, tw, th = cv2.boundingRect(target)
            target_cx = tx + (tw // 2)
            
            if abs(target_cx - screen_cx) < 250:
                state = "TRACKING"
                
    elif state == "TRACKING":
        if not valid_objects:
            state = "SEARCHING"
        else:
            target = valid_objects[0]
            tx, ty, tw, th = cv2.boundingRect(target)
            target_cx, target_cy = tx + (tw // 2), ty + (th // 2)
            area = tw * th
            
            cv2.rectangle(img_bgr, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 3)
            cv2.line(img_bgr, (screen_cx, screen_cy), (target_cx, target_cy), (0, 255, 0), 2)
            
            # Monocular Depth Estimation: If the box takes up >250,000 pixels, it's very close
            if area > 250000: 
                state = "ARRIVED"
                arrival_cooldown = 25 
            else:
                cv2.putText(img_bgr, f"TRACKING | Target Area: {area}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
                # Flight Vectors
                yaw_rate = (target_cx - screen_cx) * 0.1
                v_z = (target_cy - screen_cy) * 0.01
                
                # Fly faster when area is small (far away), slower when area is large (close)
                v_x = max(1.0, min((300000 - area) * 0.00002, 4.0)) 
                
                # Artificial Potential Field (Obstacle Avoidance)
                for obs in valid_objects[1:]:
                    ox, oy, ow, oh = cv2.boundingRect(obs)
                    obs_cx = ox + (ow // 2)
                    dist_x = obs_cx - screen_cx
                    
                    if abs(dist_x) < 300:
                        cv2.rectangle(img_bgr, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 2)
                        yaw_rate += -np.sign(dist_x) * (300 - abs(dist_x)) * 0.1

    # --- SEND KINEMATICS ---
    client.moveByVelocityBodyFrameAsync(v_x, 0, v_z, 0.1, 
        airsim.DrivetrainType.MaxDegreeOfFreedom, 
        airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate))

    cv2.imshow("Monocular HUD", img_bgr)
    
    # Show the "Edge AI" feed so you can see the black bars covering the sky/floor
    cv2.imshow("AI Vision Mask", edges) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
cv2.destroyAllWindows()

