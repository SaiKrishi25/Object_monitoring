# Object Monitoring System

This project uses YOLOv8 for real-time object detection and tracking in videos.

## Prerequisites

- Docker installed on your system
- Input video file

## Building the Docker Image

```bash
docker build -t object-monitor .
```

## Running the Container

1. Place your input video in the `input` directory
2. Run the container:

```bash
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output object-monitor
```

The processed video will be saved in the `output` directory.

## Notes

- The input video should be placed in the `input` directory
- The processed output will be saved in the `output` directory
- The container uses YOLOv8s model for object detection
- Press 'q' to quit the video processing

## Features

- Missing Object Detection: Identifies when previously present objects are no longer visible
- New Object Placement Detection: Detects when new objects appear in the scene
- Real-time tracking and visualization
- Performance metrics (FPS)
- Output video generation with annotations

## Project Structure

```
object-monitoring/
│
├── main.py                      # Core script with YOLO + tracking logic
├── input/                       # Folder for test videos
│   └── test_video1.mp4
├── output/                      # Store processed output videos
│
├── utils/                       # Utility scripts
│   └── visualizer.py            # Drawing and visualization utilities
│
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure the directory structure is correctly set up:
   ```
   mkdir -p input output
   ```

3. Place your test video in the `input` folder

## Usage

Run the main script:

```
python main.py
```

- Press 'q' to quit the application
- The processed video will be saved to `output/processed_output.mp4`

## Configuration

You can adjust tracking parameters in `main.py`:

- `missing_threshold`: Number of frames before declaring an object as missing
- Video input/output paths
- Model selection (change `yolov8s.pt` to other YOLOv8 models for different speed/accuracy tradeoffs)

## Results

The system will provide:
- Real-time visualization with bounding boxes
- Status panel showing active, new, and missing objects
- Terminal output with events and final statistics
- Processed output video with annotations