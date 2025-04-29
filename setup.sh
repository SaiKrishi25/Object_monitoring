#!/bin/bash

# Create necessary directories
mkdir -p models input output

# Check if YOLOv8 model exists, download if not
if [ ! -f "models/yolov8s.pt" ]; then
    echo "Downloading YOLOv8s model..."
    curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -o models/yolov8s.pt
fi

# Create requirements.txt file
cat > requirements.txt << EOL
ultralytics>=8.0.0
opencv-python-headless>=4.5.0
numpy>=1.20.0
tqdm>=4.64.0
PyYAML>=6.0
EOL

# Create necessary utils directory and visualizer.py file if it doesn't exist
mkdir -p utils
if [ ! -f "utils/visualizer.py" ]; then
    cat > utils/visualizer.py << EOL
import cv2
import numpy as np

def draw_box(frame, track_id, x, y, width, height, label="object", color=None):
    """Draw bounding box with label on the frame."""
    if color is None:
        color = (0, 255, 0)
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    
    # Draw label background
    text = f"{label} #{track_id}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (x, y - 25), (x + text_size[0] + 10, y), color, -1)
    
    # Draw label text
    cv2.putText(frame, text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def draw_fps(frame, fps):
    """Draw FPS counter on the frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def draw_status_info(frame, active_count, new_count, missing_count):
    """Draw status information on the frame."""
    cv2.putText(frame, f"Active: {active_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"New: {new_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Missing: {missing_count}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
EOL
fi

echo "Setup complete! You can build the Docker image with: docker-compose build"
echo "Run the container with: docker-compose up"