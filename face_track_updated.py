import cv2
import numpy as np
import dlib
import pandas as pd
import datetime
from ultralytics import YOLO
from sort import Sort

# Initialize YOLO Model for Object Detection
yolo_model = YOLO("yolov8n.pt")

# Initialize SORT Tracker for Multi-Object Tracking
tracker = Sort()

# Initialize Face Detector (Dlib)
face_detector = dlib.get_frontal_face_detector()

# Open Webcam
cap = cv2.VideoCapture(0)

# Data Logging for Object Movement
tracking_data = []

def drawBox(img, bbox, label="Tracking"):
    """Draw Bounding Box with Label"""
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

while True:
    timer = cv2.getTickCount()
    success, img = cap.read()
    
    if not success:
        print("Failed to capture frame. Exiting...")
        break

    # Face Detection using Dlib
    faces = face_detector(img)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        drawBox(img, (x, y, w, h), "Face")

    # Object Detection using YOLO
    detections = []
    yolo_results = yolo_model(img)
    for result in yolo_results:
        for box in result.boxes:
            x, y, w, h = map(int, box.xywh[0])
            detections.append([x, y, x + w, y + h, 1.0])  # (x1, y1, x2, y2, score)

    # Track Objects using SORT Multi-Object Tracker
    tracked_objects = tracker.update(np.array(detections))

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        drawBox(img, (x1, y1, x2 - x1, y2 - y1), f"ID {obj_id}")

        # Save tracking data for analysis
        tracking_data.append([datetime.datetime.now(), obj_id, x1, y1, x2 - x1, y2 - y1])

    # Calculate FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Display FPS
    cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow("Tracking", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save Tracking Data to CSV
df = pd.DataFrame(tracking_data, columns=["Time", "Object_ID", "X", "Y", "Width", "Height"])
df.to_csv("tracking_data.csv", index=False)

cap.release()
cv2.destroyAllWindows()
