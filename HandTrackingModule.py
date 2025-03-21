import cv2
import mediapipe as mp
import time
import math


class HandDetector:
    """Class for detecting and tracking hands using MediaPipe."""

    def __init__(self, static_image_mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """Initialize the hand detector with specified parameters.

        Args:
            static_image_mode: Whether to treat input images as static (not video).
            max_hands: Maximum number of hands to detect.
            detection_confidence: Minimum confidence for hand detection.
            tracking_confidence: Minimum confidence for hand tracking.
        """
        self.static_image_mode = static_image_mode
        self.max_hands = max_hands
        self.detection_confidence = float(detection_confidence)
        self.tracking_confidence = float(tracking_confidence)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.finger_tip_ids = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
        self.results = None
        self.landmark_list = []

    def find_hands(self, image, draw=True):
        """Detect hands in the image.

        Args:
            image: The image to process (BGR format).
            draw: Whether to draw hand landmarks on the image.

        Returns:
            The processed image with hand landmarks drawn if draw=True.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

        return image

    def find_position(self, image, hand_index=0, draw=True):
        """Find the positions of hand landmarks.

        Args:
            image: The image containing the hand.
            hand_index: Which hand to track if multiple are detected.
            draw: Whether to draw position markers and bounding box.

        Returns:
            Tuple of (landmark_list, bounding_box) where landmark_list contains
            [id, x, y] for each landmark and bounding_box contains
            (xmin, ymin, xmax, ymax).
        """
        x_coordinates = []
        y_coordinates = []
        bounding_box = []
        self.landmark_list = []

        if self.results.multi_hand_landmarks:
            if hand_index < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_index]

                height, width, _ = image.shape
                for landmark_id, landmark in enumerate(hand.landmark):

                    pixel_x, pixel_y = int(landmark.x * width), int(landmark.y * height)
                    x_coordinates.append(pixel_x)
                    y_coordinates.append(pixel_y)
                    self.landmark_list.append([landmark_id, pixel_x, pixel_y])

                    if draw:
                        cv2.circle(image, (pixel_x, pixel_y), 7, (0, 255, 0), cv2.FILLED)


                if x_coordinates and y_coordinates:
                    x_min, x_max = min(x_coordinates), max(x_coordinates)
                    y_min, y_max = min(y_coordinates), max(y_coordinates)
                    bounding_box = (x_min, y_min, x_max, y_max)

                    if draw:

                        padding = 20
                        cv2.rectangle(
                            image,
                            (bounding_box[0] - padding, bounding_box[1] - padding),
                            (bounding_box[2] + padding, bounding_box[3] + padding),
                            (0, 255, 0),
                            2
                        )

        return self.landmark_list, bounding_box

    def fingers_up(self):
        """Determine which fingers are raised.

        Returns:
            A list of 5 values (0 or 1) indicating which fingers are up.
            [thumb, index, middle, ring, pinky]
        """
        if not self.landmark_list or len(self.landmark_list) < 21:
            return [0, 0, 0, 0, 0]  # Return all fingers down if no hand detected

        fingers = []

        if self.landmark_list[self.finger_tip_ids[0]][1] > self.landmark_list[self.finger_tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)


        for finger_id in range(1, 5):
            if self.landmark_list[self.finger_tip_ids[finger_id]][2] < \
                    self.landmark_list[self.finger_tip_ids[finger_id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, point1_id, point2_id, image, draw=True):
        """Calculate the distance between two landmarks.

        Args:
            point1_id: ID of the first landmark point.
            point2_id: ID of the second landmark point.
            image: Image to draw on if draw=True.
            draw: Whether to visualize the distance.

        Returns:
            Tuple of (distance, image, line_info) where line_info contains
            [x1, y1, x2, y2, cx, cy] for drawing purposes.
        """
        if not self.landmark_list or max(point1_id, point2_id) >= len(self.landmark_list):
            return 0, image, [0, 0, 0, 0, 0, 0]


        x1, y1 = self.landmark_list[point1_id][1], self.landmark_list[point1_id][2]
        x2, y2 = self.landmark_list[point2_id][1], self.landmark_list[point2_id][2]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            # Usar siempre color azul (BGR) y reducir grosor a 2
            cv2.circle(image, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(image, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Línea más fina (grosor 2)
            cv2.circle(image, (center_x, center_y), 7, (255, 0, 0), cv2.FILLED)


        distance = math.hypot(x2 - x1, y2 - y1)

        return distance, image, [x1, y1, x2, y2, center_x, center_y]


def main():
    """Demo function to test the HandDetector."""
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot access camera")
        exit()

    hand_detector = HandDetector()
    previous_time = 0

    while True:
        success, image = camera.read()
        if not success:
            print("Failed to capture frame")
            break

        image = hand_detector.find_hands(image)
        landmark_list, bounding_box = hand_detector.find_position(image)

        if landmark_list:

            print(landmark_list[4])


        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(
            image,
            f"FPS: {int(fps)}",
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 0),
            3
        )

        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()