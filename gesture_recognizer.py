# 手势识别模块
# 基于手部关键点的几何关系，判断当前手势类型和手掌朝向

import math


class GestureRecognizer:
    """手势识别器 - 基于关键点几何关系判断手势和手掌朝向

    支持的手势:
        fist(握拳), open(张开), thumb_up(点赞), thumb_down(倒赞),
        peace(比耶), ok(OK), pointing(指向), rock(摇滚),
        call(打电话), three(比三), four(比四)

    手掌朝向:
        palm_facing=True  → 手心朝镜头(正面)
        palm_facing=False → 手背朝镜头(背面)
    """

    GESTURES = [
        "unknown",      # 未知
        "fist",         # 握拳
        "open",         # 张开
        "thumb_up",     # 点赞
        "thumb_down",   # 倒赞
        "peace",        # 比耶/剪刀
        "ok",           # OK手势
        "pointing",     # 指向(食指)
        "rock",         # 摇滚(食指+小指)
        "call",         # 打电话(拇指+小指)
        "three",        # 比三
        "four",         # 比四
    ]

    # 各手指的关键点索引
    FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    FINGER_PIPS = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
    FINGER_MCPS = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

    @staticmethod
    def _distance(p1, p2):
        """计算两点之间的距离"""
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    @classmethod
    def _is_finger_extended(cls, landmarks, finger):
        """判断手指是否伸直（支持任意手部朝向）"""
        tip = landmarks[cls.FINGER_TIPS[finger]]
        pip_joint = landmarks[cls.FINGER_PIPS[finger]]
        mcp = landmarks[cls.FINGER_MCPS[finger]]
        wrist = landmarks[0]

        if finger == "thumb":
            return abs(tip.x - wrist.x) > abs(mcp.x - wrist.x)

        # 方法1：经典 y 轴检测（手指朝上时有效）
        if tip.y < pip_joint.y < mcp.y:
            return True

        # 方法2：距离检测（任意朝向均有效）
        # 伸直时指尖到 MCP 距离 >> PIP 到 MCP 距离
        # 弯曲时指尖靠近 MCP，距离 ≤ PIP 到 MCP
        tip_mcp = math.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
        pip_mcp = math.sqrt((pip_joint.x - mcp.x)**2 + (pip_joint.y - mcp.y)**2)
        return tip_mcp > pip_mcp * 1.4

    @classmethod
    def _get_pointing_direction(cls, landmarks):
        """根据食指角度判断指向: 'right'/'left'/'up'/'down' 或 None"""
        index_mcp = landmarks[5]
        index_tip = landmarks[8]
        dx = index_tip.x - index_mcp.x
        dy = index_tip.y - index_mcp.y
        angle = math.degrees(math.atan2(dy, dx))
        if -45 < angle < 45:
            return "right"
        elif angle > 135 or angle < -135:
            return "left"
        elif -135 <= angle <= -45:
            return "up"
        elif 45 <= angle <= 135:
            return "down"
        return None

    @classmethod
    def _is_palm_facing_camera(cls, landmarks, hand_type):
        """判断手掌是否正面朝向摄像头（手心对着镜头）

        原理: 计算 手腕→食指根 和 手腕→小指根 两个向量的叉积,
              叉积的符号结合左右手信息即可判断手掌朝向。

        Args:
            landmarks: 21个手部关键点
            hand_type: "Right" 或 "Left"（MediaPipe 返回的手类型）

        Returns:
            bool: True=手心朝镜头(正面), False=手背朝镜头(背面)
        """
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        v1 = (index_mcp.x - wrist.x, index_mcp.y - wrist.y)
        v2 = (pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y)
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]

        if hand_type == "Right":
            return cross_z > 0
        else:
            return cross_z < 0

    @classmethod
    def recognize(cls, landmarks, hand_type="Right"):
        """识别手势和手掌朝向

        Args:
            landmarks: 21个手部关键点
            hand_type: "Right" 或 "Left"

        Returns:
            tuple: (手势名称, 置信度, 手心是否朝镜头)
        """
        if not landmarks or len(landmarks) < 21:
            return "unknown", 0.0, False

        palm_facing = cls._is_palm_facing_camera(landmarks, hand_type)

        # 检测每个手指的伸直状态
        thumb = cls._is_finger_extended(landmarks, "thumb")
        index = cls._is_finger_extended(landmarks, "index")
        middle = cls._is_finger_extended(landmarks, "middle")
        ring = cls._is_finger_extended(landmarks, "ring")
        pinky = cls._is_finger_extended(landmarks, "pinky")

        extended_count = sum([thumb, index, middle, ring, pinky])
        confidence = 0.8

        # ---- 手势判断规则 ----

        if extended_count == 0:
            return "fist", confidence, palm_facing

        if extended_count == 5:
            return "open", confidence, palm_facing

        if thumb and not index and not middle and not ring and not pinky:
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            if thumb_tip.y < thumb_mcp.y:
                return "thumb_up", confidence, palm_facing
            else:
                return "thumb_down", confidence, palm_facing

        if not thumb and index and middle and not ring and not pinky:
            return "peace", confidence, palm_facing

        if middle and ring and pinky:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            dist = cls._distance(thumb_tip, index_tip)
            if dist < 0.1:
                return "ok", confidence, palm_facing
            elif extended_count == 4 and not thumb:
                return "four", confidence, palm_facing

        if not thumb and index and not middle and not ring and not pinky:
            direction = cls._get_pointing_direction(landmarks)
            gesture_name = f"point_{direction}" if direction else "pointing"
            return gesture_name, confidence, palm_facing

        if not thumb and index and not middle and not ring and pinky:
            return "rock", confidence, palm_facing

        if thumb and not index and not middle and not ring and pinky:
            return "call", confidence, palm_facing

        if extended_count == 3:
            if (index and middle and ring) or (thumb and index and middle):
                return "three", confidence, palm_facing

        return "unknown", 0.0, palm_facing
