import cv2

# Load the MobileNet SSD model
prototxt_path = "models/MobileNetSSD_deploy.prototxt"
model_path = "models/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Class index for 'person'
PERSON_CLASS_ID = 15

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Create input blob and run inference
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.007843, size=(300, 300),
                                 mean=(127.5, 127.5, 127.5), swapRB=True)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if class_id == PERSON_CLASS_ID and confidence > 0.5:
            box = detections[0, 0, i, 3:7] * \
                  [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (x1, y1, x2, y2) = box.astype("int")
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    cv2.imshow("People Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
