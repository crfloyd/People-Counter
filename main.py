import cv2
import math

# === Config ===
PROTOTXT_PATH = "models/MobileNetSSD_deploy.prototxt"
MODEL_PATH = "models/MobileNetSSD_deploy.caffemodel"
PERSON_CLASS_ID = 15

LINE_ORIENTATION = 'vertical'  # 'vertical' or 'horizontal'
LINE_POS = 320                 # x if vertical, y if horizontal

# === Tracking State ===
entry_count = 0
exit_count = 0
next_id = 0
tracked_objects = {}  # {id: [prev_pos, curr_pos]} — position is (x, y)

# === Load Detector ===
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
cap = cv2.VideoCapture(0)

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Run detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300),
                                 (127.5, 127.5, 127.5), swapRB=True)
    net.setInput(blob)
    detections = net.forward()

    current_centroids = []

    # Draw counting line
    if LINE_ORIENTATION == 'vertical':
        cv2.line(frame, (LINE_POS, 0), (LINE_POS, 480), (0, 0, 255), 2)
    else:
        cv2.line(frame, (0, LINE_POS), (640, LINE_POS), (0, 0, 255), 2)

    # Collect centroids from current detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if class_id == PERSON_CLASS_ID and confidence > 0.5:
            box = detections[0, 0, i, 3:7] * \
                  [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (x1, y1, x2, y2) = box.astype("int")
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            current_centroids.append((cx, cy))

            # Draw box and center
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Match detections to previous tracked objects
    updated_objects = {}
    for cx, cy in current_centroids:
        matched_id = None
        for obj_id, (_, prev_centroid) in tracked_objects.items():
            if euclidean((cx, cy), prev_centroid) < 50:
                matched_id = obj_id
                break

        if matched_id is None:
            matched_id = next_id
            next_id += 1

        prev_centroid = tracked_objects.get(matched_id, [None, None])[1]
        curr_centroid = (cx, cy)

        # Directional crossing logic
        if prev_centroid:
            prev_val = prev_centroid[0] if LINE_ORIENTATION == 'vertical' else prev_centroid[1]
            curr_val = curr_centroid[0] if LINE_ORIENTATION == 'vertical' else curr_centroid[1]

            if prev_val < LINE_POS and curr_val >= LINE_POS:
                entry_count += 1
            elif prev_val > LINE_POS and curr_val <= LINE_POS:
                exit_count += 1

        updated_objects[matched_id] = [prev_centroid, curr_centroid]

    tracked_objects = updated_objects

    # === Draw Count Overlays ===
    cv2.putText(frame, f"Entries: {entry_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Exits:   {exit_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Occupancy: {entry_count - exit_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Red

    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
