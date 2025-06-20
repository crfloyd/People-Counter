import cv2
from utils.video import get_video_capture

cap = get_video_capture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistency (optional)
    frame = cv2.resize(frame, (640, 480))

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Clean the mask
    fg_mask = cv2.erode(fg_mask, None, iterations=1)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    # Find contours (moving objects)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 800:  # filter out small noise
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display
    cv2.imshow("People Detector", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
