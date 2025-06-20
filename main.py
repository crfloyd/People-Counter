import cv2
import numpy as np
from utils.video import get_video_capture

cap = get_video_capture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

line_y = 240  # Horizontal counting line
count = 0
already_counted = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.erode(fg_mask, None, iterations=1)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_centroids = []

    # Merge overlapping contours into one blob using a mask
    merged_mask = np.zeros_like(fg_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) < 800:
            continue
        cv2.drawContours(merged_mask, [cnt], -1, 255, -1)  # Fill contour

    # Find merged blobs from mask
    merged_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in merged_contours:
        if cv2.contourArea(cnt) < 2000:  # Bigger threshold after merge
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        if cy < line_y + 5 and cy > line_y - 5 and (cx, cy) not in already_counted:
            count += 1
            already_counted.append((cx, cy))


    # Draw line and count
    cv2.line(frame, (0, line_y), (640, line_y), (0, 0, 255), 2)
    cv2.putText(frame, f"Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("People Detector", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

