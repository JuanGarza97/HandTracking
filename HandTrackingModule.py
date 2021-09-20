"""
Hand Tracking Module
By: Juan Carlos Garza Sanchez
"""

import cv2
import mediapipe as mp
import time
import math
from enum import Enum
import numpy as np


class Fingers(Enum):
    Thumb = 4
    Index = 8
    Middle = 12
    Ring = 16
    Pinky = 20


class HandType(Enum):
    Right = "Right"
    Left = "Left"


class Hand:
    def __init__(self, type_of_hand, landmarks, bounding_box, center):
        self.type = type_of_hand
        self.landmarks = landmarks
        self.bounding_box = bounding_box
        self.center = center


class HandDetector:
    def __init__(self, static_image_mode: bool = False, max_num_hands: int = 2, min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5) -> None:
        self.mode = static_image_mode
        self.maxHands = max_num_hands
        self.detectionConfidence = min_detection_confidence
        self.trackConfidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = np.array([Fingers.Thumb, Fingers.Index, Fingers.Middle, Fingers.Ring, Fingers.Pinky])

    def find_hands(self, img: np.ndarray, is_mirrored: bool = True, draw: bool = True,
                   color: list[int, int, int] = (0, 255, 0)
                   , thickness: int = 2) -> [np.ndarray, list[Hand]]:

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        hands = []

        if self.results.multi_hand_landmarks:
            h, w, c = img.shape
            for handType, landmarks in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                lm_list = []
                x_list = np.array([], int)
                y_list = np.array([], int)
                for lm_id, lm in enumerate(landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_list = np.append(x_list, cx)
                    y_list = np.append(y_list, cy)
                    lm_list.append([lm_id, cx, cy])
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                bounding_box = x_min, y_min, x_max, y_max

                hand_type = ""
                if not is_mirrored:
                    if handType.classification[0].label == "Right":
                        hand_type = "Left"
                    else:
                        hand_type = "Right"
                else:
                    hand_type = handType.classification[0].label

                hands.append(Hand(hand_type, lm_list, bounding_box, (cx, cy)))

                if draw:
                    cv2.rectangle(img, (bounding_box[0] - 20, bounding_box[1] - 20),
                                  (bounding_box[2] + 20, bounding_box[3] + 20), color, thickness)
                    self.mpDraw.draw_landmarks(img, landmarks, self.mpHands.HAND_CONNECTIONS)
        return img, hands

    @staticmethod
    def get_hand(hands: list[Hand], type_of_hand: HandType = HandType.Right) -> Hand or None:
        hand_type = type_of_hand.value
        for hand in hands:
            if hand.type == hand_type:
                return hand
        return None

    def fingers_up(self, hand: Hand) -> np.ndarray:
        fingers = np.zeros(5, bool)
        hand_type = hand.type
        landmarks = hand.landmarks

        # Thumb
        if hand_type == "Right":
            if landmarks[self.tipIds[0].value][1] < landmarks[self.tipIds[0].value - 1][1]:
                fingers[0] = 1
        else:
            if landmarks[self.tipIds[0].value][1] > landmarks[self.tipIds[0].value - 1][1]:
                fingers[0] = 1

        # Other 4 Fingers
        for i in range(1, 5):
            if landmarks[self.tipIds[i].value][2] < landmarks[self.tipIds[i].value - 2][2]:
                fingers[i] = 1

        return fingers

    @staticmethod
    def pointing_direction(hand: Hand, finger: Fingers = Fingers.Index, degrees: bool = True) -> int:
        landmarks = hand.landmarks
        x1, y1 = landmarks[finger.value][1], landmarks[finger.value][2]
        x2, y2 = landmarks[finger.value - 3][1], landmarks[finger.value - 3][2]
        angle = math.atan2(y2 - y1, x1 - x2)
        if degrees:
            angle = math.degrees(angle)
            if angle < 0:
                angle = angle + 360
        return int(angle)

    def find_angle(self, hand: Hand, finger1: Fingers, finger2: Fingers) -> int:
        return abs(self.pointing_direction(hand, finger1) - self.pointing_direction(hand, finger2))

    @staticmethod
    def find_distance(img: np.ndarray, point1: int, point2: int, draw: bool = False, radius: int = 7,
                      color: tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> [np.ndarray, float, list[int]]:

        x1, y1 = point1
        x2, y2 = point2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), radius, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), radius, color, cv2.FILLED)
            cv2.circle(img, (cx, cy), radius, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        length = math.hypot(x2 - x1, y2 - y1)
        return img, length, [x1, y1, x2, y2, cx, cy]


def main():
    p_time = 0

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:

        success, img = cap.read()
        img = cv2.flip(img, 1)
        img, hands = detector.find_hands(img)
        # right_hand = detector.get_hand(hands, HandType.Left)
        for hand in hands:
            if hand.type == "Right":
                print(detector.pointing_direction(hand))
                # print(detector.fingers_up(right_hand))
                # print(detector.find_angle(right_hand, Fingers.Index, Fingers.Middle))
                # print(detector.pointing_direction(Fingers.Index))
                # print(lmList[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break


if __name__ == "__main__":
    main()
