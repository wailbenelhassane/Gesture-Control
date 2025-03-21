import cv2
import time
import numpy as np
import queue
import threading
import HandTrackingModule as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class CameraCapture:
    """Clase para manejar la captura de video en un hilo separado"""

    def __init__(self, width=640, height=480, queue_size=10):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, width)
        self.cap.set(4, height)
        time.sleep(1)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            if self.q.full():
                self.q.get()

            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                return

            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True
        self.cap.release()


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


def draw_volume_bar(image, bar_height, volume_percentage, is_muted=False):
    """Draw volume indicator bar and percentage with better font"""
    cv2.rectangle(image, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(image, (50, int(bar_height)), (85, 400), (255, 0, 0), cv2.FILLED)

    if is_muted:
        cv2.putText(image, 'MUTED', (40, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(image, f'{int(volume_percentage)} %', (40, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


def draw_fps(image, current_time, previous_time):
    """Calculate and display FPS with better font"""
    fps = 1 / (current_time - previous_time)

    cv2.putText(image, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return current_time


def draw_current_volume(image, volume_controller, is_muted=False):
    """Display current system volume with better font - siempre azul"""
    current_volume = int(volume_controller.GetMasterVolumeLevelScalar() * 100)

    if is_muted:
        cv2.putText(image, 'MUTED', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(image, f'Volume set: {current_volume}', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


def draw_mode_status(image, volume_mode, is_muted=False):
    """Display mode status with better font - no mostrar en mute"""
    if not is_muted:
        mode_text = "Adjusting" if volume_mode else "Fixed"
        mode_color = (255, 0, 0)

        cv2.putText(image, f"Mode: {mode_text}", (400, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)


def main():
    camera_stream = CameraCapture(width=640, height=480, queue_size=5).start()
    hand_detector = htm.HandDetector(detection_confidence=0.7, max_hands=1)
    volume_controller, volume_range = initialize_audio_controller()

    prev_time = 0
    volume_bar = 400
    volume_percentage = 0

    is_muted = False
    last_volume = volume_controller.GetMasterVolumeLevelScalar()

    volume_mode = False
    pinky_down_timer = 0
    last_index_gesture_time = 0

    gesture_cooldown = 0.7

    process_every_n_frames = 1
    detect_gesture_every_n_frames = 3
    frame_count = 0

    last_fingers = []
    last_hand_detected = False

    last_line_info = None

    last_hand_detected_time = 0
    line_display_timeout = 2.0
    show_line = False

    while True:
        try:
            image = camera_stream.read()
            current_time = time.time()
            frame_count += 1

            image = hand_detector.find_hands(image)
            landmarks, bounding_box = hand_detector.find_position(image, draw=False)

            hand_detected = landmarks and len(landmarks) > 0

            if hand_detected:
                last_hand_detected_time = current_time
                show_line = True

            elif current_time - last_hand_detected_time > line_display_timeout:
                show_line = False

            last_hand_detected = hand_detected

            if frame_count % detect_gesture_every_n_frames == 0 and hand_detected:
                last_fingers = hand_detector.fingers_up()

                hand_area = calculate_hand_area(bounding_box)

                if 250 < hand_area < 1000:
                    length, image, line_info = hand_detector.find_distance(4, 8, image)
                    last_line_info = line_info

                    if not last_fingers[4] and (
                            last_fingers[0] or last_fingers[1] or last_fingers[2] or last_fingers[3]):
                        if current_time - pinky_down_timer > gesture_cooldown:
                            volume_mode = not volume_mode
                            pinky_down_timer = current_time

                    if last_fingers[1] == 1 and sum(last_fingers) == 1:
                        if current_time - last_index_gesture_time > gesture_cooldown:
                            if not is_muted:
                                last_volume = volume_controller.GetMasterVolumeLevelScalar()
                                volume_controller.SetMute(True, None)
                                is_muted = True
                            else:
                                volume_controller.SetMute(False, None)
                                volume_controller.SetMasterVolumeLevelScalar(last_volume, None)
                                is_muted = False

                            last_index_gesture_time = current_time

                    if volume_mode and not is_muted:
                        volume_bar, volume_percentage = map_length_to_volume(length)
                        volume_percentage = smooth_volume(volume_percentage)

                        volume_controller.SetMasterVolumeLevelScalar(volume_percentage / 100, None)

                        last_volume = volume_percentage / 100

            elif hand_detected and last_line_info is not None and show_line:
                cv2.line(image, (last_line_info[0], last_line_info[1]),
                         (last_line_info[2], last_line_info[3]), (255, 0, 0), 2)  # Reducido a 2 de grosor
                cv2.circle(image, (last_line_info[0], last_line_info[1]), 7, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, (last_line_info[2], last_line_info[3]), 7, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, (last_line_info[4], last_line_info[5]), 7, (255, 0, 0), cv2.FILLED)

            if not hand_detected:
                if not is_muted:
                    current_volume = volume_controller.GetMasterVolumeLevelScalar() * 100
                    volume_bar = np.interp(current_volume, [0, 100], [400, 150])
                    volume_percentage = current_volume

            draw_volume_bar(image, volume_bar, volume_percentage, is_muted)
            draw_current_volume(image, volume_controller, is_muted)

            if not is_muted:
                draw_mode_status(image, volume_mode, is_muted)

            prev_time = draw_fps(image, current_time, prev_time)

            cv2.imshow("Volume Control", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error: {e}")
            break

    camera_stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()