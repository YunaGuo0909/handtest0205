# 手部追踪模块
# 使用 MediaPipe HandLandmarker 检测手部关键点并绘制骨架

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
    """手部追踪器 - 检测手部关键点并绘制骨架

    职责:
        - 自动下载/加载 MediaPipe 模型
        - 从图像帧中检测手部关键点
        - 在图像上绘制骨架和关键点
        - 格式化关键点数据供输出
    """

    def __init__(self):
        self._ensure_model()
        self._init_detector()

    def _ensure_model(self):
        """确保模型文件存在，不存在则下载"""
        if not os.path.exists(MODEL_PATH):
            print("下载手部模型中...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("模型下载完成")

    def _init_detector(self):
        """初始化 MediaPipe 手部检测器"""
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
        """检测手部关键点

        Args:
            frame: BGR格式的图像帧

        Returns:
            HandLandmarkerResult: MediaPipe检测结果，包含:
                - hand_landmarks: 各手的21个关键点
                - handedness: 各手的左右手信息
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect(img)

    @staticmethod
    def get_finger_color(landmark_index):
        """根据关键点索引获取对应手指颜色"""
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
        """在图像上绘制手部骨架

        Args:
            frame: 图像帧
            landmarks: 手部关键点列表
            w: 图像宽度
            h: 图像高度
        """
        # 画骨架连线
        for a, b in BONE_CONNECTIONS:
            p1 = (int(landmarks[a].x * w), int(landmarks[a].y * h))
            p2 = (int(landmarks[b].x * w), int(landmarks[b].y * h))
            cv2.line(frame, p1, p2, cls.get_finger_color(a), 2)

        # 画关键点（指尖用大圆，其他用小圆）
        for i, lm in enumerate(landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            r = 8 if i in [4, 8, 12, 16, 20] else 5
            cv2.circle(frame, (x, y), r, cls.get_finger_color(i), -1)

    @staticmethod
    def format_landmarks(landmarks):
        """将关键点格式化为字典列表

        Args:
            landmarks: 手部关键点

        Returns:
            list[dict]: 格式化后的关键点数据，每项包含 id, name, x, y, z
        """
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
