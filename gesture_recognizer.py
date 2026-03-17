# Gesture recognizer: rule-based classification from hand landmarks (geometry).
# Palm facing: palm_facing=True -> palm toward camera, False -> back of hand.

import math


class GestureRecognizer:
    """Rule-based gesture recognizer from hand landmarks.

    Gestures: fist, open, thumb_up, thumb_down, peace, ok,
    point_right/left/up/down, pointing, rock, call, three, four, pinch, pre_pinch.

    Palm: palm_facing=True -> palm toward camera, False -> back of hand.
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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    # ------------------------------------------------------------------
    # Finger extension (3D PIP angle + y-axis fallback)
    # ------------------------------------------------------------------

    @classmethod
    def _is_finger_extended(cls, landmarks, finger):
        """Extended = 3D PIP angle (includes depth-direction curl)."""
        tip = landmarks[cls.FINGER_TIPS[finger]]
        pip_j = landmarks[cls.FINGER_PIPS[finger]]
        mcp = landmarks[cls.FINGER_MCPS[finger]]
        wrist = landmarks[0]

        if finger == "thumb":
            return abs(tip.x - wrist.x) > abs(mcp.x - wrist.x)

        # 3D PIP angle (includes z for depth curl)
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

        # y-axis fallback: finger clearly up and span large enough
        if tip.y < pip_j.y < mcp.y and (mcp.y - tip.y) > 0.06:
            return True

        return False

    # ------------------------------------------------------------------
    # Pinch detector (independent of generic finger logic)
    # ------------------------------------------------------------------

    @classmethod
    def _pinch_geometry(cls, landmarks, min_dist, max_dist, others_mult=1.1):
        """Check thumb+index in pinch/pre-pinch geometry.

        1. Thumb-index distance in [min_dist, max_dist)
        2. Pinch midpoint far enough from wrist (exclude fist)
        3. Other finger tips not beyond midpoint * others_mult (exclude OK/open)
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
    def _detect_pinch(cls, landmarks, hand_type="Right"):
        """Pinch: left hand only; thumb+index tips close, index extended, middle curled."""
        if hand_type != "Left":
            return False
        if not cls._pinch_geometry(landmarks, 0.0, 0.08, others_mult=1.1):
            return False
        wrist = landmarks[0]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        index_middle_dist = math.sqrt(
            (index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2
        )
        if index_middle_dist < 0.12:
            return False
        index_reach = math.sqrt(
            (index_tip.x - wrist.x) ** 2 + (index_tip.y - wrist.y) ** 2
        )
        middle_reach = math.sqrt(
            (middle_tip.x - wrist.x) ** 2 + (middle_tip.y - wrist.y) ** 2
        )
        if index_reach < 0.15 or index_reach < middle_reach * 1.2:
            return False
        return True

    @classmethod
    def _detect_pre_pinch(cls, landmarks, hand_type="Right"):
        """Pre-pinch: left hand only; thumb+index extended, others curled; index and middle well apart."""
        if hand_type != "Left":
            return False
        if not cls._pinch_geometry(landmarks, 0.08, 0.18, others_mult=1.2):
            return False
        wrist = landmarks[0]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        index_middle_dist = math.sqrt(
            (index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2
        )
        if index_middle_dist < 0.12:
            return False
        index_reach = math.sqrt(
            (index_tip.x - wrist.x) ** 2 + (index_tip.y - wrist.y) ** 2
        )
        middle_reach = math.sqrt(
            (middle_tip.x - wrist.x) ** 2 + (middle_tip.y - wrist.y) ** 2
        )
        if index_reach < 0.15 or index_reach < middle_reach * 1.2:
            return False
        return True

    # ------------------------------------------------------------------
    # Pointing detector (independent of generic finger logic)
    # ------------------------------------------------------------------

    @classmethod
    def _detect_pointing(cls, landmarks):
        """Index uniquely far from wrist in 3D (>= 1.3x other fingers). Suppress when thumb near index (pre-pinch)."""
        # Dead zone when thumb near index (pre-pinch)
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
    # Pointing direction
    # ------------------------------------------------------------------

    @classmethod
    def _get_pointing_direction(cls, landmarks):
        """Index tip->mcp vector angle -> right/left/up/down."""
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
    # Palm facing
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
    # Main entry
    # ------------------------------------------------------------------

    @classmethod
    def recognize(cls, landmarks, hand_type="Right"):
        if not landmarks or len(landmarks) < 21:
            return "unknown", 0.0, False

        palm_facing = cls._is_palm_facing_camera(landmarks, hand_type)

        # ========== Dedicated detectors (highest priority) ==========

        if cls._detect_pinch(landmarks, hand_type):
            return "pinch", 0.85, palm_facing

        if cls._detect_pre_pinch(landmarks, hand_type):
            return "pre_pinch", 0.85, palm_facing

        direction = cls._detect_pointing(landmarks)
        if direction:
            return f"point_{direction}", 0.85, palm_facing

        # ========== Generic finger extension ==========

        thumb = cls._is_finger_extended(landmarks, "thumb")
        index = cls._is_finger_extended(landmarks, "index")
        middle = cls._is_finger_extended(landmarks, "middle")
        ring = cls._is_finger_extended(landmarks, "ring")
        pinky = cls._is_finger_extended(landmarks, "pinky")

        extended_count = sum([thumb, index, middle, ring, pinky])
        confidence = 0.8

        if extended_count == 0:
            return "fist", confidence, palm_facing
        # Single extended finger (not thumb/index) -> loose fist for W/S
        if extended_count == 1 and not thumb and not index:
            return "fist", 0.75, palm_facing

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

        # Generic pointing fallback
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
