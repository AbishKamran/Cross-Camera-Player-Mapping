# Cross-Camera Player Mapping

A computer vision system for tracking and mapping players across multiple camera angles in sports gameplay using YOLOv11 object detection.

## Project Overview

This project implements a cross-camera player tracking system that maintains consistent player identities across two different camera feeds (broadcast and tactical views) of the same gameplay. The system uses YOLOv11-based object detection to identify players and implements sophisticated mapping algorithms to ensure each player retains a consistent `player_id` across both camera views.

## Features

- **Multi-Camera Player Detection**: Simultaneous player detection in broadcast and tactical camera feeds
- **Cross-Camera ID Mapping**: Consistent player identity mapping between different camera angles
- **Temporal Tracking**: Frame-by-frame player tracking with ID persistence
- **Performance Analytics**: Comprehensive tracking quality metrics and statistics
- **Real-time Processing**: Efficient processing of video streams with detailed logging

## Requirements

### Prerequisites
- Python 3.8+
- OpenCV
- YOLOv11 (Ultralytics)
- NumPy
- JSON

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-repo/cross-camera-player-mapping.git
cd cross-camera-player-mapping
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the YOLOv11 model**
```bash
# The trained model should be placed in the models/ directory
# Model trained specifically for player and ball detection
```

## Usage

### Basic Usage

1. **Prepare your video files**
   - Place `broadcast.mp4` and `tacticam.mp4` in the `videos/` directory
   - Ensure both videos are synchronized and cover the same gameplay

2. **Run the mapping system**
```bash
python main.py
```

3. **View results**
   - Console output shows real-time tracking statistics
   - Detailed results saved to `enhanced_player_mapping_results.json`

### Command Line Options

```bash
python main.py --broadcast videos/broadcast.mp4 --tactical videos/tacticam.mp4 --output results/
```

## Output Format

The system generates detailed tracking information including:

### Console Output
```
Frame 46: Cam1=11, Cam2=22, Matches=11
  Matches: {1: 2, 2: 1, 8: 5, 4: 6, 10: 7, 14: 8, 11: 9, 7: 10, 13: 11, 6: 13, 5: 20}
Cross-camera mappings (local IDs): {1: 2, 2: 1, 8: 5, 4: 6, 10: 7, 14: 8, 11: 9, 7: 10, 13: 11, 6: 13, 5: 20}
Common global IDs: {1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 20}
```

##  Algorithm Details

### Detection Pipeline
1. **YOLOv11 Object Detection**: Identifies players and ball in each frame
2. **Confidence Filtering**: Removes low-confidence detections
3. **Multi-class Processing**: Handles player and ball detection separately

### Mapping Strategy
1. **Spatial Matching**: Uses bounding box positions for initial matching
2. **Temporal Consistency**: Maintains ID persistence across frames
3. **Cross-Camera Correlation**: Maps local IDs between camera views
4. **Global ID Assignment**: Ensures consistent global player identities

### Key Features
- **Robust Matching**: Handles occlusions and temporary disappearances
- **Multi-Feature Fusion**: Combines spatial, temporal, and visual features
- **Adaptive Thresholding**: Dynamic confidence adjustment based on scene complexity

##  Performance Metrics

The system provides comprehensive performance analytics:

- **Detection Rate**: Players detected per frame in each camera
- **Tracking Persistence**: Average duration of continuous player tracking
- **Cross-Camera Consistency**: Percentage of successful ID mappings between cameras
- **ID Stability**: Number of identity switches during tracking
- **Re-identification Success**: Ability to recover lost player identities

## Configuration

### Model Parameters
```python
# Detection confidence threshold
DETECTION_CONFIDENCE = 0.5

# Tracking parameters
MAX_DISAPPEARED_FRAMES = 10
DISTANCE_THRESHOLD = 100

# Cross-camera mapping
SIMILARITY_THRESHOLD = 0.7
```
---

**Made with ❤️ by Abish**
