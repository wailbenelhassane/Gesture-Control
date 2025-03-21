# Hand Gesture Volume Control

A computer vision project that allows you to control your system's volume using hand gestures captured through your webcam.

## Description

This project uses computer vision and hand tracking to create a touchless volume control system. By detecting specific hand gestures through your webcam, you can adjust your system's volume, mute/unmute, and switch between adjustment modes.

### Features

- **Real-time hand detection and tracking** using the MediaPipe library
- **Volume adjustment** by changing the distance between your thumb and index finger
- **Mute/unmute** with a single index finger gesture
- **Mode switching** between fixed and adjustable volume modes
- **Visual feedback** with an on-screen volume bar and percentage indicator
- **Efficient performance** using multi-threaded camera capture

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- PyCaw (for Windows audio control)
- Threading and Queue libraries (included in standard Python)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/hand-gesture-volume-control.git
   cd hand-gesture-volume-control
   ```

2. Install the required packages:
   ```
   pip install opencv-python mediapipe numpy pycaw
   ```

## Usage

Run the main script:
```
python volume_control.py
```

### Gestures

- **Adjust volume**: Pinch your thumb and index finger together or move them apart
- **Mute/Unmute**: Show only your index finger
- **Toggle adjustment mode**: Close your pinky finger while keeping other fingers open

### Controls

- Press 'q' to quit the application

## Project Structure

- `HandTrackingModule.py`: Contains the `HandDetector` class for hand tracking and landmark detection
- `volume_control.py`: Main script that uses the hand tracking module to control system volume

## How It Works

1. The application captures webcam feed in a separate thread for better performance
2. Hand detection and tracking are performed using the MediaPipe library
3. The distance between specific finger landmarks is measured to control volume
4. Specific finger gestures are recognized for additional controls (mute/mode switching)
5. Volume adjustments are mapped to the Windows audio system using PyCaw

## Customization

You can adjust several parameters in the code:
- Detection sensitivity
- Volume mapping ranges
- Gesture cooldown times
- Visual feedback appearance

## Limitations

- Currently only supports Windows operating systems for volume control
- Requires good lighting conditions for reliable hand detection
