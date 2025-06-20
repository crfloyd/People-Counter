# Real-Time People Counting Using MobileNet SSD

This project implements a real-time people counting system using a fixed webcam and a deep learning-based object detector. The system detects people entering or exiting a monitored area and tracks overall occupancy using a virtual line-crossing mechanism.

## Features

- Live detection using MobileNet SSD (OpenCV DNN)
- Real-time occupancy tracking
- Horizontal or vertical line-crossing configuration
- Jitter-resistant centroid-based tracking
- Frame display with bounding boxes, IDs, and occupancy counters

## Requirements

- Python 3.8+
- OpenCV (4.x)
- NumPy

Install requirements:

```bash
pip install opencv-python numpy
```

## Running the System

Place the MobileNet SSD model files in a `models/` folder:

- `MobileNetSSD_deploy.prototxt.txt`
- `MobileNetSSD_deploy.caffemodel`

Then run:

```bash
python main.py
```

## Output

- Live video window showing:
  - Detected people with bounding boxes and centroids
  - Virtual line (red)
  - Entry/exit and current occupancy counts (top left corner)

## Notes

- Default input resolution is 640x480.
- By default, the virtual line is vertical and centered.
- You can adjust line orientation and position in the script config.

## Future Work

- Multi-person tracking improvements
- Multi-camera handoff support
- Re-identification for lost IDs

## Author

Corey Floyd
