# main.py
# Vehicle Counter that works with BOTH images and videos
# Just change the INPUT_PATH to your file

import cv2
import os
from ultralytics import YOLO
from tracker import VehicleTracker

# ─────────────────────────────────────────────
# CONFIGURATION — change INPUT_PATH to your file
# ─────────────────────────────────────────────
INPUT_PATH = "media/video1.avi"   # Change this to your file name
OUTPUT_FOLDER = "output"
CONFIDENCE_THRESHOLD = 0.25
COUNT_LINE_Y = 120

# Vehicle class IDs in YOLO (COCO dataset)
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Supported file types
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

# ─────────────────────────────────────────────
# LOAD YOLO MODEL
# ─────────────────────────────────────────────
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded!")

# ─────────────────────────────────────────────
# HELPER — Draw detections on a frame
# ─────────────────────────────────────────────
def process_frame(frame):
    """
    Takes a single frame/image
    Runs YOLO detection
    Draws boxes around vehicles
    Returns frame + list of center points
    """
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id in VEHICLE_CLASSES and confidence > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append((cx, cy))

            # Draw green box around vehicle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Write label above the box
            label = f"{results.names[class_id]} {confidence:.0%}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detections


# ─────────────────────────────────────────────
# PROCESS IMAGE
# ─────────────────────────────────────────────
def process_image(input_path):
    """
    Handles a single image file
    Detects vehicles and saves result
    """
    print(f"\nProcessing IMAGE: {input_path}")

    # Read the image
    frame = cv2.imread(input_path)

    if frame is None:
        print("ERROR: Could not read image. Check the file path.")
        return

    # Run detection
    frame, detections = process_frame(frame)
    total_count = len(detections)

    # Draw count on image
    cv2.putText(
        frame,
        f"Vehicles Detected: {total_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        3
    )

    # Save output image
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"{name}_result{ext}")
    cv2.imwrite(output_path, frame)

    # Show the image
    cv2.imshow("Vehicle Detection - Image", frame)
    print(f"Vehicles detected: {total_count}")
    print(f"Result saved to: {output_path}")
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# PROCESS VIDEO
# ─────────────────────────────────────────────
def process_video(input_path):
    """
    Handles a video file
    Detects and counts vehicles crossing a line
    Saves output video
    """
    print(f"\nProcessing VIDEO: {input_path}")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("ERROR: Could not open video. Check the file path.")
        return

    # Get video properties
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = int(cap.get(cv2.CAP_PROP_FPS))

    # Setup output video
    filename = os.path.basename(input_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"{name}_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize tracker
    tracker = VehicleTracker()
    counted_ids = set()
    total_count = 0
    frame_number = 0

    print("Processing... Press Q to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video finished.")
            break

        frame_number += 1

        # Run detection on this frame
        frame, detections = process_frame(frame)

        # Update tracker
        tracked = tracker.update(detections)

        # Check if vehicle crossed counting line
        for vehicle_id, positions in tracked.items():
            cx, cy = positions[-1]

            # Draw red dot at vehicle center
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Count if vehicle crosses the line
            if abs(cy - COUNT_LINE_Y) < 15 and vehicle_id not in counted_ids:
                counted_ids.add(vehicle_id)
                total_count += 1

        # Draw counting line
        cv2.line(frame,
                 (0, COUNT_LINE_Y),
                 (frame_width, COUNT_LINE_Y),
                 (255, 0, 0), 2)

        # Draw count on frame
        cv2.putText(
            frame,
            f"Vehicle Count: {total_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3
        )

        # Show frame
        cv2.imshow("Vehicle Counter - Video", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped early by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nTotal vehicles counted: {total_count}")
    print(f"Result saved to: {output_path}")


# ─────────────────────────────────────────────
# PROCESS FOLDER (mix of images and videos)
# ─────────────────────────────────────────────
def process_folder(folder_path):
    """
    Goes through an entire folder
    Automatically processes each image and video it finds
    """
    print(f"\nProcessing FOLDER: {folder_path}")

    files = os.listdir(folder_path)
    images = []
    videos = []

    # Sort files into images and videos
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(file)
        elif ext in VIDEO_EXTENSIONS:
            videos.append(file)

    print(f"Found {len(images)} images and {len(videos)} videos")

    # Process all images
    for image_file in images:
        full_path = os.path.join(folder_path, image_file)
        process_image(full_path)

    # Process all videos
    for video_file in videos:
        full_path = os.path.join(folder_path, video_file)
        process_video(full_path)


# ─────────────────────────────────────────────
# MAIN — Auto detect what INPUT_PATH is
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # Check if input exists
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Could not find: {INPUT_PATH}")
        print("Please check the INPUT_PATH in main.py")
        exit()

    # Get file extension
    ext = os.path.splitext(INPUT_PATH)[1].lower()

    # Auto detect and process
    if os.path.isdir(INPUT_PATH):
        # It's a folder — process everything inside
        process_folder(INPUT_PATH)

    elif ext in IMAGE_EXTENSIONS:
        # It's an image
        process_image(INPUT_PATH)

    elif ext in VIDEO_EXTENSIONS:
        # It's a video
        process_video(INPUT_PATH)

    else:
        print(f"ERROR: Unsupported file type: {ext}")
        print(f"Supported images: {IMAGE_EXTENSIONS}")
        print(f"Supported videos: {VIDEO_EXTENSIONS}")
        print("Processing... Press Q to stop early.")
        COUNT_LINE_Y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
print(f"Counting line set at Y = {COUNT_LINE_Y}")