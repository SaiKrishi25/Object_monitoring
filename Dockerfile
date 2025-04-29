# Use Python 3.9 as base image with security updates
FROM python:3.9-slim-bullseye

# Set label metadata
LABEL maintainer="developer"
LABEL description="Real-time Object Monitoring with YOLOv8"
LABEL version="1.0"

# Install system dependencies with security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a requirements.txt file with the necessary dependencies
RUN echo "ultralytics>=8.0.0\nopencv-python>=4.5.0\nnumpy>=1.20.0\ntqdm>=4.64.0" > requirements.txt

# Install Python dependencies with security updates
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY realtime_monitor.py /app/
COPY utils/ /app/utils/

# Create input and output directories
RUN mkdir -p input output && \
    chmod -R 755 input output

# Create the models directory and download YOLOv8 model if not mounted
RUN mkdir -p models

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/yolov8s.pt \
    INPUT_PATH=/app/input \
    OUTPUT_PATH=/app/output

# Run as non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Use ENTRYPOINT for better handling of command-line arguments
ENTRYPOINT ["python", "realtime_monitor.py"]

# Default command if no arguments are provided
CMD ["--source", "0", "--model", "/app/models/yolov8s.pt", "--display", "--output", "/app/output/monitored_output.mp4"]