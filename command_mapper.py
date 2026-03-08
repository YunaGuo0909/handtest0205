# 指令映射模块
# 左手连续移动 / 右手瞬发技能 / 双手组合技 → 键盘按键

import math
import time
from collections import deque, Counter

from config import (
    MOVE_STABLE_FRAMES, ACTION_COOLDOWN, COMBO_COOLDOWN,
    SWIPE_UP_THRESHOLD, SWIPE_HISTORY_FRAMES, HANDS_TOUCH_THRESHOLD,
    POINTING_X_THRESHOLD,
)


class CommandMapper:
    """手势 → 键盘指令映射器

    左手(连续): 食指指右→D, 食指指左→A, 其他→停止
    右手(瞬发): 手背向上挥→W(跳), 指向→Shift(加速)
    双手(瞬发): 双拳展开→E, 手腕碰撞→Q
    """

    # ===== 右手 · 瞬发键位 =====
    RIGHT_HAND_ACTION = {
        "swipe_up":  "w",           # 手背快速向上 → W (跳跃)
        "pointing":  "shift",       # 指向 → Shift (加速)
    }

    # ===== 双手 · 组合键位 =====
    COMBO_ACTION = {
        "fist_to_open": "e",        # 双拳展开为手掌 → E
        "wrist_clash":  "q",        # 手腕碰撞 → Q
    }

    # 双拳展开的过渡容忍帧数
    _FIST_TO_OPEN_WINDOW = 8

    def __init__(self):
        # 移动防抖
        self._move_history = deque(maxlen=MOVE_STABLE_FRAMES)
        self._current_move_key = None

        # 瞬发冷却
        self._last_action_time = 0
        self._last_combo_time = 0

        # 边缘检测
        self._prev_right_gesture = "unknown"
        self._both_fist_counter = 0

        # 右手挥动检测
        self._right_wrist_history = deque(maxlen=SWIPE_HISTORY_FRAMES)

        # 显示用
        self.last_move_cmd = "stop"
        self.last_action = ""
        self.last_combo = ""

    # --------------------------------------------------
    # 主入口
    # --------------------------------------------------
    def update(self, left_data, right_data, keyboard):
        """每帧调用，处理所有手势 → 键盘逻辑

        Args:
            left_data:  (gesture, palm_facing, landmarks) or None
            right_data: (gesture, palm_facing, landmarks) or None
            keyboard:   KeyboardController

        Returns:
            list[str]: 本帧触发的动作描述（用于日志/显示）
        """
        triggered = []
        now = time.time()

        # --- 1. 双手组合（最高优先级）---
        combo = self._check_combo(left_data, right_data, now)
        if combo:
            keyboard.tap_key(self.COMBO_ACTION[combo])
            self.last_combo = combo
            triggered.append(f"COMBO {combo} → {self.COMBO_ACTION[combo].upper()}")

        # --- 2. 左手移动（连续）---
        new_key = self._update_movement(left_data)
        if new_key != self._current_move_key:
            if self._current_move_key:
                keyboard.release_key(self._current_move_key)
            if new_key:
                keyboard.hold_key(new_key)
            self._current_move_key = new_key

        # --- 3. 右手技能（瞬发）---
        action = self._check_action(right_data, now)
        if action:
            keyboard.tap_key(self.RIGHT_HAND_ACTION[action])
            self.last_action = action
            triggered.append(f"ACTION {action} → {self.RIGHT_HAND_ACTION[action].upper()}")

        # --- 更新上一帧状态 ---
        self._update_prev_state(left_data, right_data)

        return triggered

    # --------------------------------------------------
    # 左手 · 连续移动（食指方向）
    # --------------------------------------------------
    @staticmethod
    def _get_pointing_direction(landmarks):
        """根据食指 tip→mcp 向量角度判断指向: 指右→'d', 指左→'a', 非水平→None

        用 atan2 计算角度（度数）:
          0° = 右, ±180° = 左, -90° = 上, 90° = 下
        只有水平方向（±45° 以内）才触发移动
        """
        index_mcp = landmarks[5]
        index_tip = landmarks[8]
        dx = index_tip.x - index_mcp.x
        dy = index_tip.y - index_mcp.y
        angle = math.degrees(math.atan2(dy, dx))

        if -45 < angle < 45:
            return "d"
        elif angle > 135 or angle < -135:
            return "a"
        return None

    def _update_movement(self, left_data):
        """防抖后返回应按住的键 (str) 或 None"""
        if left_data is None:
            raw_key = None
        else:
            gesture, _, landmarks = left_data
            if gesture == "point_right":
                raw_key = "d"
            elif gesture == "point_left":
                raw_key = "a"
            elif gesture == "pointing":
                raw_key = self._get_pointing_direction(landmarks)
            else:
                raw_key = None

        self._move_history.append(raw_key)

        if len(self._move_history) < self._move_history.maxlen:
            return self._current_move_key

        counts = Counter(self._move_history)
        most, count = counts.most_common(1)[0]

        if count >= self._move_history.maxlen:
            self.last_move_cmd = most or "stop"
            return most

        return self._current_move_key

    # --------------------------------------------------
    # 右手 · 瞬发技能
    # --------------------------------------------------
    def _check_action(self, right_data, now):
        if right_data is None:
            self._right_wrist_history.clear()
            return None

        if now - self._last_action_time < ACTION_COOLDOWN:
            return None

        gesture, palm_facing, landmarks = right_data

        # 手背快速向上挥动 (y 减小 = 向上)
        wrist_y = landmarks[0].y
        self._right_wrist_history.append(wrist_y)

        if len(self._right_wrist_history) >= 2:
            delta = self._right_wrist_history[0] - self._right_wrist_history[-1]
            if delta > SWIPE_UP_THRESHOLD:
                self._last_action_time = now
                self._right_wrist_history.clear()
                return "swipe_up"

        # 指向 → 边缘触发（从非 pointing 变为 pointing）
        is_pointing = gesture in ("pointing", "point_right", "point_left", "point_up", "point_down")
        was_pointing = self._prev_right_gesture in ("pointing", "point_right", "point_left", "point_up", "point_down")
        if is_pointing and not was_pointing:
            self._last_action_time = now
            return "pointing"

        return None

    # --------------------------------------------------
    # 双手 · 组合技
    # --------------------------------------------------
    def _check_combo(self, left_data, right_data, now):
        if left_data is None or right_data is None:
            return None

        if now - self._last_combo_time < COMBO_COOLDOWN:
            return None

        l_gesture, _, l_lms = left_data
        r_gesture, _, r_lms = right_data

        # 双拳展开为手掌 (E): 最近 N 帧内有双拳 → 当前双掌
        if (self._both_fist_counter > 0
                and l_gesture == "open" and r_gesture == "open"):
            self._last_combo_time = now
            self._both_fist_counter = 0
            return "fist_to_open"

        # 手腕碰撞 (Q): 双手腕距离 < 阈值
        dist = ((l_lms[0].x - r_lms[0].x) ** 2
                + (l_lms[0].y - r_lms[0].y) ** 2) ** 0.5
        if dist < HANDS_TOUCH_THRESHOLD:
            self._last_combo_time = now
            return "wrist_clash"

        return None

    # --------------------------------------------------
    # 状态更新
    # --------------------------------------------------
    def _update_prev_state(self, left_data, right_data):
        self._prev_right_gesture = right_data[0] if right_data else "unknown"

        both_fist = (left_data is not None and right_data is not None
                     and left_data[0] == "fist" and right_data[0] == "fist")
        if both_fist:
            self._both_fist_counter = self._FIST_TO_OPEN_WINDOW
        elif self._both_fist_counter > 0:
            self._both_fist_counter -= 1

    # --------------------------------------------------
    # 控制
    # --------------------------------------------------
    def stop(self, keyboard):
        """立即停止所有移动"""
        if self._current_move_key:
            keyboard.release_key(self._current_move_key)
            self._current_move_key = None
        self._move_history.clear()
        self.last_move_cmd = "stop"

    def reset(self):
        """重置所有状态"""
        self._move_history.clear()
        self._current_move_key = None
        self._prev_right_gesture = "unknown"
        self._both_fist_counter = 0
        self._right_wrist_history.clear()
        self.last_move_cmd = "stop"
        self.last_action = ""
        self.last_combo = ""
