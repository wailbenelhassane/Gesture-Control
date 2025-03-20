import cv2
import time
import numpy as np
import HandTrackingModule as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


def initialize_camera(width=640, height=480):
    """Initialize webcam with specified dimensions"""
    camera = cv2.VideoCapture(0)
    camera.set(3, width)
    camera.set(4, height)
    time.sleep(1)

    if not camera.isOpened():
        raise RuntimeError("Failed to open camera")

    return camera


def initialize_audio_controller():
    """Initialize Windows audio controller"""
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_controller = interface.QueryInterface(IAudioEndpointVolume)
    volume_range = volume_controller.GetVolumeRange()

    return volume_controller, volume_range


def calculate_hand_area(bbox):
    """Calculate hand area from bounding box"""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return (width * height) // 100


def map_length_to_volume(length, min_length=50, max_length=200):
    """Map hand gesture length to volume percentage and bar height"""
    volume_bar = np.interp(length, [min_length, max_length], [400, 150])
    volume_percentage = np.interp(length, [min_length, max_length], [0, 100])
    return volume_bar, volume_percentage


def smooth_volume(volume_percentage, smoothness=10):
    """Apply smoothing to volume changes"""
    return smoothness * round(volume_percentage / smoothness)


def draw_volume_bar(image, bar_height, volume_percentage):
    """Draw volume indicator bar and percentage"""
    cv2.rectangle(image, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(image, (50, int(bar_height)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, f'{int(volume_percentage)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)


def draw_fps(image, current_time, previous_time):
    """Calculate and display FPS"""
    fps = 1 / (current_time - previous_time)
    cv2.putText(image, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    return current_time


def draw_current_volume(image, volume_controller, color):
    """Display current system volume"""
    current_volume = int(volume_controller.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(image, f'Volume set: {current_volume}', (400, 50), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)


def main():
    camera = initialize_camera(640, 480)
    hand_detector = htm.HandDetector(detection_confidence=0.7, max_hands=1)
    volume_controller, volume_range = initialize_audio_controller()

    prev_time = 0
    volume_bar = 400
    volume_percentage = 0
    volume_color = (255, 0, 0)

    while True:
        success, image = camera.read()
        if not success:
            print("Failed to capture frame")
            break

        image = hand_detector.find_hands(image)
        landmarks, bounding_box = hand_detector.find_position(image, draw=False)

        if landmarks:
            hand_area = calculate_hand_area(bounding_box)

            if 250 < hand_area < 1000:
                length, image, line_info = hand_detector.find_distance(4, 8, image)

                volume_bar, volume_percentage = map_length_to_volume(length)
                volume_percentage = smooth_volume(volume_percentage)

                fingers = hand_detector.fingers_up()

                if not fingers[4]:
                    volume_controller.SetMasterVolumeLevelScalar(volume_percentage / 100, None)
                    cv2.circle(image, (line_info[4], line_info[5]), 7, (0, 255, 0), cv2.FILLED)
                    volume_color = (0, 255, 0)
                else:
                    volume_color = (255, 0, 0)

        draw_volume_bar(image, volume_bar, volume_percentage)
        draw_current_volume(image, volume_controller, volume_color)

        current_time = time.time()
        prev_time = draw_fps(image, current_time, prev_time)

        cv2.imshow("Volume Control", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()