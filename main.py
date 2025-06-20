import cv2

# Initialize HOG + SVM person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistent detection performance
    frame = cv2.resize(frame, (640, 480))

    # Detect people
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))

    # Draw detections
    for (x, y, w, h) in boxes:
        cx, cy = x + w // 2, y + h // 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Show result
    cv2.imshow("People Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
