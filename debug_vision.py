import airsim
import cv2
import numpy as np

print("Connecting...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
print("Airborne. Press Q to quit. Watch the console for detection info.")

cv2.namedWindow("Raw Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
cv2.namedWindow("All Contours", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Raw Frame", 960, 540)
cv2.resizeWindow("Edges", 960, 540)
cv2.resizeWindow("All Contours", 960, 540)

while True:
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    if not responses or responses[0].height == 0:
        continue

    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    frame = img1d.reshape(responses[0].height, responses[0].width, 3).copy()
    h, w = frame.shape[:2]

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    high_t, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred, int(0.5 * high_t), int(high_t))
    edges = cv2.dilate(edges, None, iterations=2)

    # NO masks -- show everything so we can see what exists
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_frame = frame.copy()
    print(f"\n--- Frame | {len(contours)} contours found ---")

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 500:  # only show meaningful contours
            continue

        rect = cv2.minAreaRect(c)
        rect_area = rect[1][0] * rect[1][1]
        score = area / rect_area if rect_area > 1 else 0

        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cx, cy = int(rect[0][0]), int(rect[0][1])

        # Color by rectangularity score
        if score > 0.75:
            color = (0, 255, 0)    # green = very rectangular
        elif score > 0.55:
            color = (0, 165, 255)  # orange = somewhat rectangular
        else:
            color = (0, 0, 255)    # red = not rectangular

        cv2.drawContours(debug_frame, [box], 0, color, 2)
        cv2.putText(debug_frame, f"a:{int(area)} s:{score:.2f}",
                    (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        print(f"  Contour {i}: area={int(area):>8}  rect_score={score:.3f}  center=({cx},{cy})")

    cv2.imshow("Raw Frame", frame)
    cv2.imshow("Edges", edges)
    cv2.imshow("All Contours", debug_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
cv2.destroyAllWindows()
