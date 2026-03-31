# 🚗 Vehicle Counter for Traffic

A computer vision project that automatically detects and counts vehicles (cars, motorcycles, buses, and trucks) in traffic images and videos using **YOLOv8** and **OpenCV**. The system tracks each vehicle with a unique ID to avoid double-counting, overlays bounding boxes and labels in real time, and saves the processed output.

---
# preview of the code


https://github.com/user-attachments/assets/d7495fbe-0995-43f2-944a-994b8fd376fd


## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Output](#output)
- [Supported File Formats](#supported-file-formats)
- [Troubleshooting](#troubleshooting)
- [Technologies Used](#technologies-used)

---

## Project Overview

This project uses the **YOLOv8n** (nano) model — a lightweight, fast object detection model — to identify vehicles in each frame of a video or in a static image. A custom centroid-based **VehicleTracker** assigns each detected vehicle a unique ID and tracks it across frames. For videos, vehicles are counted when they cross a horizontal counting line drawn across the frame.

This is ideal for:
- Traffic monitoring and analysis
- Understanding vehicle flow at intersections
- Research projects involving computer vision and object detection

---

## Features

- Detects **4 vehicle types**: cars, motorcycles, buses, and trucks (using COCO class IDs)
- Works with **images** (JPG, PNG, BMP, WEBP) and **videos** (MP4, AVI, MOV, MKV)
- Can process an **entire folder** of mixed images and videos in one run
- Real-time **bounding boxes** drawn around each detected vehicle
- Unique **vehicle ID tracking** across frames to prevent double-counting
- A **counting line** in videos — vehicles are counted once when they cross it
- **Live preview window** while processing
- Processed results saved automatically to the `output/` folder
- Confidence threshold filtering to reduce false positives

---

## How It Works

```
Input (image / video / folder)
        │
        ▼
 YOLOv8n Model detects vehicles in each frame
        │
        ▼
 Bounding boxes drawn; center points extracted
        │
        ▼
 VehicleTracker assigns unique IDs via centroid proximity (< 50px)
        │
        ▼
 [Video only] Check if vehicle center crosses the counting line (Y = 120)
        │
        ▼
 Count incremented once per unique vehicle ID
        │
        ▼
 Annotated frame shown live + written to output file
```

The tracker in `tracker.py` uses a simple but effective centroid-distance algorithm: if a detected center point in the current frame is within 50 pixels of a previously tracked vehicle's last known position, it is treated as the same vehicle.

---

## Project Structure

```
Vehicle-Counter-for-Traffic/
│
├── main.py             ← Main entry point; handles image, video, and folder processing
├── tracker.py          ← Custom VehicleTracker class for multi-frame vehicle ID tracking
├── requirements.txt    ← Python package dependencies
├── test.py             ← Optional test/development script
├── .gitignore
│
├── media/              ← ⬅ Place your input files here (create this folder)
│   └── video1.avi      ← Default input path set in main.py
│
└── output/             ← ⬅ Processed results are saved here (create this folder)
```

> **Note:** The `media/` and `output/` folders are not included in the repository. You must create them manually before running the project (instructions below).

---
dataset link- https://www.kaggle.com/datasets/aryashah2k/highway-traffic-videos-dataset
## Prerequisites

Before you begin, make sure you have the following installed on your system:

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 or 3.11 | Other versions may work but are untested |
| pip | Latest | Comes bundled with Python |
| Git | Any | Required to clone the repository |

> **Windows users:** Make sure Python is added to your system PATH during installation. You can verify by running `python --version` in Command Prompt.

> **Mac/Linux users:** You may need to use `python3` and `pip3` instead of `python` and `pip` depending on your system.

---

## Environment Setup

### Step 1 — Clone the Repository

Open a terminal (Command Prompt, PowerShell, or bash) and run:

```bash
git clone https://github.com/RANVEER12082005/Vehicle-Counter-for-Traffic.git
cd Vehicle-Counter-for-Traffic
```

### Step 2 — Create a Virtual Environment

A virtual environment keeps this project's dependencies isolated from other Python projects on your machine.

```bash
# Create the virtual environment
python -m venv venv
```

Then activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

You should now see `(venv)` at the beginning of your terminal prompt, confirming the environment is active.

### Step 3 — Create Required Folders

The project expects `media/` and `output/` folders to exist. Create them now:

```bash
# Windows
mkdir media
mkdir output

# Mac / Linux
mkdir -p media output
```

---

## Installation

With the virtual environment active, install all required dependencies:

```bash
pip install -r requirements.txt
```

This installs the following packages:

| Package | Purpose |
|---|---|
| `opencv-python-headless` | Video/image reading, drawing, and display |
| `ultralytics` | YOLOv8 model loading and inference |
| `numpy` | Numerical operations for frame processing |
| `Pillow` | Image handling support |

> **First run note:** When you run the project for the first time, the YOLOv8n model weights file (`yolov8n.pt`, ~6 MB) will be **automatically downloaded** from the internet by the `ultralytics` library. This only happens once.

---

## Configuration

All configuration is done inside `main.py`. Open the file and look for the configuration section near the top:

```python
# ─────────────────────────────────────────────
# CONFIGURATION — change INPUT_PATH to your file
# ─────────────────────────────────────────────
INPUT_PATH = "media/video1.avi"   # ← Change this to your file path
OUTPUT_FOLDER = "output"
CONFIDENCE_THRESHOLD = 0.25
COUNT_LINE_Y = 120
```

| Setting | Default | Description |
|---|---|---|
| `INPUT_PATH` | `"media/video1.avi"` | Path to your input file or folder. Change this to match your file. |
| `OUTPUT_FOLDER` | `"output"` | Folder where results are saved. |
| `CONFIDENCE_THRESHOLD` | `0.25` | Minimum detection confidence (0.0–1.0). Increase to reduce false positives. |
| `COUNT_LINE_Y` | `120` | Vertical pixel position of the counting line in videos. Vehicles crossing this line are counted. |

### Setting the Input Path

Change `INPUT_PATH` to point to your file:

```python
# Single video
INPUT_PATH = "media/traffic.mp4"

# Single image
INPUT_PATH = "media/intersection.jpg"

# Entire folder (processes all images and videos inside)
INPUT_PATH = "media/"
```

---

## Running the Project

### Step 1 — Add Your Input File

Copy your traffic video or image into the `media/` folder you created earlier. For example:

```
media/
└── traffic.mp4
```

### Step 2 — Update the Input Path in main.py

Edit `main.py` and change `INPUT_PATH` to match your filename:

```python
INPUT_PATH = "media/traffic.mp4"
```

### Step 3 — Run the Script

Make sure your virtual environment is still active (`(venv)` appears in the terminal), then run:

```bash
python main.py
```

You will see output like:

```
Loading YOLO model...
Model loaded!

Processing VIDEO: media/traffic.mp4
Processing... Press Q to stop early.
Video finished.

Total vehicles counted: 47
Result saved to: output/traffic_result.mp4
```

A live preview window will open while the video processes. Press **Q** to stop early if needed.

---

## Output

| Input Type | Output |
|---|---|
| **Image** | Annotated image saved to `output/<filename>_result.<ext>`. Vehicle count displayed as text on the image. A preview window opens — press any key to close. |
| **Video** | Annotated video saved to `output/<filename>_result.mp4`. Vehicle count shown live on-screen. Total printed in terminal when finished. |
| **Folder** | Each file in the folder is processed individually; all results saved in `output/`. |

### What You'll See in the Output

- **Green bounding boxes** around each detected vehicle
- **Label text** above each box showing vehicle type and confidence percentage (e.g., `car 87%`)
- **Red dots** at the center of each tracked vehicle (video only)
- **Blue horizontal line** — the counting line (video only)
- **Yellow counter text** in the top-left corner showing the current vehicle count

---

## Supported File Formats

| Type | Supported Extensions |
|---|---|
| Images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` |
| Videos | `.mp4`, `.avi`, `.mov`, `.mkv` |

---

## Troubleshooting

**`ERROR: Could not find: media/video1.avi`**
→ You haven't placed a video file in `media/`, or the `INPUT_PATH` in `main.py` doesn't match the actual filename. Double-check the path and filename.

**`ERROR: Could not open video. Check the file path.`**
→ The file exists but OpenCV can't read it. Try converting your video to `.mp4` format using a tool like VLC or FFmpeg.

**Model not downloading / no internet**
→ The first run requires internet access to download `yolov8n.pt`. Once downloaded, it is cached and works offline.

**Window not appearing (headless server)**
→ If you're running on a server without a display, the live preview window will fail. Use `opencv-python` instead of `opencv-python-headless` in requirements.txt, or remove the `cv2.imshow(...)` lines from `main.py`.

**Low detection accuracy**
→ Try lowering `CONFIDENCE_THRESHOLD` to `0.15` in `main.py` to detect more vehicles, or make sure your input video is clear and not excessively blurry or dark.

**Vehicles counted multiple times**
→ Adjust `COUNT_LINE_Y` so the counting line is positioned in a part of the frame where vehicles pass through clearly. The default value of `120` is near the top of the frame — increase it to move the line further down.

---

## Technologies Used

| Technology | Role |
|---|---|
| **Python 3.10 / 3.11** | Core programming language |
| **YOLOv8n (Ultralytics)** | Real-time object detection model |
| **OpenCV** | Video/image I/O, rendering, display |
| **NumPy** | Array and matrix operations |
| **Pillow** | Image format support |

---

## License

This project is open source. Refer to the repository for license details.

---

*Built by [RANVEER12082005](https://github.com/RANVEER12082005)*
