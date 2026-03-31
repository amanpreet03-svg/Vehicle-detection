import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("media/video1.avi")

ret, frame = cap.read()
if ret:
    print("Frame read successfully!")
    print("Frame shape:", frame.shape)
    results = model(frame, verbose=True)
    print("Detections:", results[0].boxes)
else:
    print("ERROR: Could not read frame from video!")

cap.release()
