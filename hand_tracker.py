# Hand tracking: MediaPipe HandLandmarker for landmarks and skeleton drawing.

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

from config import (
    MODEL_PATH,
    MODEL_URL,
    LANDMARK_NAMES,
    BONE_CONNECTIONS,
    FINGER_COLORS,
    MIN_DETECTION_CONFIDENCE,
    MIN_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)


class HandTracker:
    """Hand tracker: detect landmarks and draw skeleton via MediaPipe HandLandmarker.

    - Download/load MediaPipe model
    - Detect hand landmarks from image frames
    - Draw skeleton and landmarks on image
    - Format landmark data for output
    """

    def __init__(self):
        self._ensure_model()
        self._init_detector()

    def _ensure_model(self):
        """Ensure model file exists; download if missing."""
        if not os.path.exists(MODEL_PATH):
            print("Downloading hand model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Hand model ready")

    def _init_detector(self):
        """Initialize MediaPipe hand detector."""
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame):
        """Detect hand landmarks.

        Args:
            frame: BGR image frame

        Returns:
            HandLandmarkerResult: hand_landmarks (21 per hand), handedness
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect(img)

    @staticmethod
    def get_finger_color(landmark_index):
        """Return finger color for landmark index."""
        if landmark_index <= 4:
            return FINGER_COLORS[0]
        elif landmark_index <= 8:
            return FINGER_COLORS[1]
        elif landmark_index <= 12:
            return FINGER_COLORS[2]
        elif landmark_index <= 16:
            return FINGER_COLORS[3]
        return FINGER_COLORS[4]

    @classmethod
    def draw_hand(cls, frame, landmarks, w, h):
        """Draw hand skeleton on image. frame: image; landmarks: list; w,h: width, height."""
        # Draw bone lines
        for a, b in BONE_CONNECTIONS:
            p1 = (int(landmarks[a].x * w), int(landmarks[a].y * h))
            p2 = (int(landmarks[b].x * w), int(landmarks[b].y * h))
            cv2.line(frame, p1, p2, cls.get_finger_color(a), 2)

        # Draw landmarks (larger circles for fingertips)
        for i, lm in enumerate(landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            r = 8 if i in [4, 8, 12, 16, 20] else 5
            cv2.circle(frame, (x, y), r, cls.get_finger_color(i), -1)

    @staticmethod
    def format_landmarks(landmarks):
        """Format landmarks to list of dicts with id, name, x, y, z."""
        return [
            {
                "id": i,
                "name": LANDMARK_NAMES[i],
                "x": round(lm.x, 4),
                "y": round(lm.y, 4),
                "z": round(lm.z, 4),
            }
            for i, lm in enumerate(landmarks)
        ]
