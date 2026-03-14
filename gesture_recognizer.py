# 手势识别模块
# 基于手部关键点的几何关系，判断当前手势类型和手掌朝向

import math


class GestureRecognizer:
    """手势识别器

    支持的手势:
        fist, open, thumb_up, thumb_down, peace, ok,
        point_right/left/up/down, pointing, rock, call,
        three, four, pinch

    手掌朝向:
        palm_facing=True  → 手心朝镜头
        palm_facing=False → 手背朝镜头
    """

    GESTURES = [
        "unknown", "fist", "open", "thumb_up", "thumb_down",
        "peace", "ok", "pointing", "rock", "call", "three", "four",
        "pinch", "pre_pinch",
    ]

    FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    FINGER_PIPS = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
    FINGER_MCPS = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    # ------------------------------------------------------------------
    # 通用手指伸直检测（3D PIP 角度 + y-axis 兜底）
    # ------------------------------------------------------------------

    @classmethod
    def _is_finger_extended(cls, landmarks, finger):
        """3D PIP 关节角度判断伸直（含深度方向弯曲检测）"""
        tip = landmarks[cls.FINGER_TIPS[finger]]
        pip_j = landmarks[cls.FINGER_PIPS[finger]]
        mcp = landmarks[cls.FINGER_MCPS[finger]]
        wrist = landmarks[0]

        if finger == "thumb":
            return abs(tip.x - wrist.x) > abs(mcp.x - wrist.x)

        # 3D PIP 角度（含 z 深度，能检测深度方向弯曲）
        v1 = (mcp.x - pip_j.x, mcp.y - pip_j.y, mcp.z - pip_j.z)
        v2 = (tip.x - pip_j.x, tip.y - pip_j.y, tip.z - pip_j.z)
        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        mag_sq = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) * (
            v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2
        )
        if mag_sq > 1e-12:
            cos_a = max(-1.0, min(1.0, dot / math.sqrt(mag_sq)))
            if cos_a < -0.5:
                return True

        # y-axis 兜底（手指明确朝上，且纵跨度足够大才触发）
        if tip.y < pip_j.y < mcp.y and (mcp.y - tip.y) > 0.06:
            return True

        return False

    # ------------------------------------------------------------------
    # 独立检测器：捏合（pinch）— 不依赖通用手指检测
    # ------------------------------------------------------------------

    @classmethod
    def _pinch_geometry(cls, landmarks, min_dist, max_dist, others_mult=1.1):
        """检查拇指+食指是否处于捏合/预捏合几何形态。

        条件：
        1. 拇指-食指距离在 [min_dist, max_dist) 范围内
        2. 捏合中点离手腕足够远（排除握拳）
        3. 其余手指 tip 不超过中点距离 × others_mult（排除 OK/张开）
        """
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        wrist = landmarks[0]

        tip_dist = math.sqrt(
            (thumb_tip.x - index_tip.x) ** 2
            + (thumb_tip.y - index_tip.y) ** 2
        )
        if tip_dist < min_dist or tip_dist >= max_dist:
            return False

        mx = (thumb_tip.x + index_tip.x) / 2
        my = (thumb_tip.y + index_tip.y) / 2
        reach = math.sqrt((mx - wrist.x) ** 2 + (my - wrist.y) ** 2)
        if reach < 0.1:
            return False

        for tip_idx in (12, 16, 20):
            t = landmarks[tip_idx]
            t_r = math.sqrt((t.x - wrist.x) ** 2 + (t.y - wrist.y) ** 2)
            if t_r > reach * others_mult:
                return False

        return True

    @classmethod
    def _detect_pinch(cls, landmarks):
        """拇指+食指指尖靠近（< 0.08）"""
        return cls._pinch_geometry(landmarks, 0.0, 0.08, others_mult=1.1)

    @classmethod
    def _detect_pre_pinch(cls, landmarks):
        """拇指正在接近食指（0.08-0.18），预捏合状态"""
        return cls._pinch_geometry(landmarks, 0.08, 0.18, others_mult=1.2)

    # ------------------------------------------------------------------
    # 独立检测器：指向（pointing）— 不依赖通用手指检测
    # ------------------------------------------------------------------

    @classmethod
    def _detect_pointing(cls, landmarks):
        """食指在 3D 空间中"唯一地"远离手腕（≥1.3× 其余手指）。

        用 3D 距离（含深度 z），即使弯曲发生在深度方向也能正确排除。
        当拇指正在靠近食指（预捏合状态）时抑制指向判定。
        """
        # 拇指靠近食指时抑制（预捏合 dead zone）
        thumb_index = math.sqrt(
            (landmarks[4].x - landmarks[8].x) ** 2
            + (landmarks[4].y - landmarks[8].y) ** 2
        )
        if thumb_index < 0.18:
            return None

        wrist = landmarks[0]

        def reach(tip_idx):
            t = landmarks[tip_idx]
            return math.sqrt(
                (t.x - wrist.x) ** 2
                + (t.y - wrist.y) ** 2
                + (t.z - wrist.z) ** 2
            )

        idx_r = reach(8)
        max_other = max(reach(12), reach(16), reach(20))

        if idx_r > 0.10 and idx_r > max_other * 1.3:
            return cls._get_pointing_direction(landmarks)
        return None

    # ------------------------------------------------------------------
    # 指向方向
    # ------------------------------------------------------------------

    @classmethod
    def _get_pointing_direction(cls, landmarks):
        """食指 tip→mcp 向量角度 → right/left/up/down"""
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

    # ------------------------------------------------------------------
    # 手掌朝向
    # ------------------------------------------------------------------

    @classmethod
    def _is_palm_facing_camera(cls, landmarks, hand_type):
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        v1 = (index_mcp.x - wrist.x, index_mcp.y - wrist.y)
        v2 = (pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y)
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]

        return cross_z > 0 if hand_type == "Right" else cross_z < 0

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    @classmethod
    def recognize(cls, landmarks, hand_type="Right"):
        if not landmarks or len(landmarks) < 21:
            return "unknown", 0.0, False

        palm_facing = cls._is_palm_facing_camera(landmarks, hand_type)

        # ========== 独立检测器（优先级最高，不走通用逻辑）==========

        if cls._detect_pinch(landmarks):
            return "pinch", 0.85, palm_facing

        if cls._detect_pre_pinch(landmarks):
            return "pre_pinch", 0.85, palm_facing

        direction = cls._detect_pointing(landmarks)
        if direction:
            return f"point_{direction}", 0.85, palm_facing

        # ========== 通用手指伸直检测 ==========

        thumb = cls._is_finger_extended(landmarks, "thumb")
        index = cls._is_finger_extended(landmarks, "index")
        middle = cls._is_finger_extended(landmarks, "middle")
        ring = cls._is_finger_extended(landmarks, "ring")
        pinky = cls._is_finger_extended(landmarks, "pinky")

        extended_count = sum([thumb, index, middle, ring, pinky])
        confidence = 0.8

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

        # 通用指向兜底（通过逐指检测走到这里）
        if not thumb and index and not middle and not ring and not pinky:
            dir2 = cls._get_pointing_direction(landmarks)
            gesture_name = f"point_{dir2}" if dir2 else "pointing"
            return gesture_name, confidence, palm_facing

        if not thumb and index and not middle and not ring and pinky:
            return "rock", confidence, palm_facing

        if thumb and not index and not middle and not ring and pinky:
            return "call", confidence, palm_facing

        if extended_count == 3:
            if (index and middle and ring) or (thumb and index and middle):
                return "three", confidence, palm_facing

        return "unknown", 0.0, palm_facing
