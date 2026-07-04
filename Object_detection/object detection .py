import cv2
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("/Users/apple/yolov8n.pt")

# Open Video
video_path = "/Users/apple/Desktop/Object_detection/video2.MP4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

print("Starting Car Detection... Press 'Q' to Quit.")

# Map Tracker IDs to Display IDs
car_id_map = {}
next_car_id = 1

# Process Video
while True:

    ret, frame = cap.read()

    if not ret:
        print("Video Finished.")
        break

    # Detect only Cars
    results = model.track(
        source=frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[2],    # COCO class 2 = Car
        conf=0.35
    )

    output = frame.copy()

    if len(results) > 0 and results[0].boxes is not None:

        for box in results[0].boxes:

            if box.id is None:
                continue

            tracker_id = int(box.id.item())

            # Assign user-friendly IDs
            if tracker_id not in car_id_map:
                car_id_map[tracker_id] = next_car_id
                next_car_id += 1

            display_id = car_id_map[tracker_id]

            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf.item())

            # Draw Bounding Box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"Car #{display_id}"

            # Label Background
            (w, h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            cv2.rectangle(
                output,
                (x1, y1 - h - 10),
                (x1 + w + 8, y1),
                (0, 0, 0),
                -1
            )

            # Label
            cv2.putText(
                output,
                label,
                (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Confidence
            cv2.putText(
                output,
                f"{confidence:.2f}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    cv2.imshow("Car Detection & Tracking", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()