# Command mapper: gesture → keyboard/mouse mapping for game control.
#
# Movement (continuous hold):
#   point_right → D, point_left → A
#   fist + back of hand → W, fist + palm facing → S
#
# Action (tap):
#   grab (open → fist) → Space
#   wrist swing (up/down) → Q
#
# Two-hand combo (tap):
#   虎口互击 (tiger mouth clash) → T
#   掌根互拍 (palm heel clap) → E
#
# Mouse:
#   pre_pinch → mouse move
#   pinch → mouse left click
#
# All other gestures → no mapping.

import math
import time
from collections import deque, Counter

from config import (
    MOVE_STABLE_FRAMES, ACTION_COOLDOWN, COMBO_COOLDOWN,
    SWIPE_UP_THRESHOLD, SWIPE_HISTORY_FRAMES, HANDS_TOUCH_THRESHOLD,
)

# Gestures that count as "pointing" for movement
_POINTING_GESTURES = ("pointing", "point_right", "point_left", "point_up", "point_down")

# Gestures that count as "open hand" for grab detection
_OPEN_GESTURES = ("open", "three", "four")


class CommandMapper:
    """Map gestures to keyboard/mouse."""

    def __init__(self):
        self._move_history = deque(maxlen=MOVE_STABLE_FRAMES)
        self._current_move_key = None
        self._last_action_time = 0
        self._last_combo_time = 0
        self._last_click_time = 0

        # Grab detection (open → fist transition)
        self._prev_gesture = {"Left": "unknown", "Right": "unknown"}

        # Wrist swing detection
        self._wrist_history_left = deque(maxlen=SWIPE_HISTORY_FRAMES)
        self._wrist_history_right = deque(maxlen=SWIPE_HISTORY_FRAMES)

        self.last_move_cmd = "stop"
        self.last_action = ""
        self.last_combo = ""

    # ------------------------------------------------------------------
    # Main update (called per frame)
    # ------------------------------------------------------------------

    def update(self, left_data, right_data, keyboard):
        """Process gestures → keyboard/mouse. Returns list of triggered strings."""
        triggered = []
        now = time.time()

        # --- 1. Two-hand combos (highest priority) ---
        combo = self._check_combo(left_data, right_data, now)
        if combo:
            key = {"tiger_mouth_clash": "t", "palm_heel_clap": "e"}[combo]
            keyboard.tap_key(key)
            self.last_combo = combo
            triggered.append(f"COMBO {combo} → {key.upper()}")

        # --- 2. Pinch / mouse (left hand only) ---
        pinch_msg = self._check_pinch(left_data, keyboard, now)
        if pinch_msg:
            triggered.append(pinch_msg)

        # --- 3. Movement (continuous hold) ---
        new_key = self._update_movement(left_data, right_data)
        if new_key != self._current_move_key:
            if self._current_move_key:
                keyboard.release_key(self._current_move_key)
            if new_key:
                keyboard.hold_key(new_key)
            self._current_move_key = new_key

        # --- 4. Tap actions (grab, wrist swing) ---
        action = self._check_action(left_data, right_data, now)
        if action:
            key = {"grab": "space", "wrist_swing": "q"}[action]
            keyboard.tap_key(key)
            self.last_action = action
            triggered.append(f"ACTION {action} → {key.upper()}")

        # --- Update prev state ---
        self._prev_gesture["Left"] = left_data[0] if left_data else "unknown"
        self._prev_gesture["Right"] = right_data[0] if right_data else "unknown"

        return triggered

    # ------------------------------------------------------------------
    # Movement: point → D/A, fist → W/S
    # ------------------------------------------------------------------

    def _update_movement(self, left_data, right_data):
        """Determine movement key from either hand (left priority)."""
        raw_key = None

        for data in (left_data, right_data):
            if data is None:
                continue
            gesture, palm_facing, landmarks = data

            if gesture == "point_right":
                raw_key = "d"
                break
            elif gesture == "point_left":
                raw_key = "a"
                break
            elif gesture == "pointing":
                raw_key = self._pointing_to_key(landmarks)
                if raw_key:
                    break
            elif gesture == "fist":
                raw_key = "s" if palm_facing else "w"
                break
            # All other gestures → no movement

        self._move_history.append(raw_key)

        if len(self._move_history) < self._move_history.maxlen:
            return self._current_move_key

        counts = Counter(self._move_history)
        most, count = counts.most_common(1)[0]

        if count >= self._move_history.maxlen:
            self.last_move_cmd = most or "stop"
            return most

        return self._current_move_key

    @staticmethod
    def _pointing_to_key(landmarks):
        """Convert generic pointing direction to D/A (horizontal only)."""
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

    # ------------------------------------------------------------------
    # Tap actions: grab (open→fist) → Space, wrist swing → Q
    # ------------------------------------------------------------------

    def _check_action(self, left_data, right_data, now):
        if now - self._last_action_time < ACTION_COOLDOWN:
            return None

        # Grab: open → fist transition on either hand
        for hand, data in [("Left", left_data), ("Right", right_data)]:
            if data is None:
                continue
            gesture = data[0]
            prev = self._prev_gesture[hand]
            if gesture == "fist" and prev in _OPEN_GESTURES:
                self._last_action_time = now
                return "grab"

        # Wrist swing: significant vertical wrist movement on either hand
        for data, history in [
            (left_data, self._wrist_history_left),
            (right_data, self._wrist_history_right),
        ]:
            if data is None:
                history.clear()
                continue
            wrist_y = data[2][0].y  # landmarks[0].y
            history.append(wrist_y)

            if len(history) >= 2:
                delta = abs(history[0] - history[-1])
                if delta > SWIPE_UP_THRESHOLD:
                    self._last_action_time = now
                    history.clear()
                    return "wrist_swing"

        return None

    # ------------------------------------------------------------------
    # Two-hand combos: 虎口互击 → T, 掌根互拍 → E
    # ------------------------------------------------------------------

    def _check_combo(self, left_data, right_data, now):
        if left_data is None or right_data is None:
            return None
        if now - self._last_combo_time < COMBO_COOLDOWN:
            return None

        l_lms = left_data[2]
        r_lms = right_data[2]

        # 虎口互击: midpoint of (thumb_tip + index_tip) on each hand close together
        l_tiger_x = (l_lms[4].x + l_lms[8].x) / 2
        l_tiger_y = (l_lms[4].y + l_lms[8].y) / 2
        r_tiger_x = (r_lms[4].x + r_lms[8].x) / 2
        r_tiger_y = (r_lms[4].y + r_lms[8].y) / 2
        tiger_dist = math.sqrt(
            (l_tiger_x - r_tiger_x) ** 2 + (l_tiger_y - r_tiger_y) ** 2
        )
        if tiger_dist < HANDS_TOUCH_THRESHOLD:
            self._last_combo_time = now
            return "tiger_mouth_clash"

        # 掌根互拍: wrist (landmark 0) of both hands close together
        wrist_dist = math.sqrt(
            (l_lms[0].x - r_lms[0].x) ** 2 + (l_lms[0].y - r_lms[0].y) ** 2
        )
        if wrist_dist < HANDS_TOUCH_THRESHOLD:
            self._last_combo_time = now
            return "palm_heel_clap"

        return None

    # ------------------------------------------------------------------
    # Pinch: pre_pinch → mouse move, pinch → mouse click
    # ------------------------------------------------------------------

    def _check_pinch(self, left_data, keyboard, now):
        if left_data is None:
            return None

        gesture, _, landmarks = left_data

        if gesture == "pinch":
            if now - self._last_click_time > ACTION_COOLDOWN:
                keyboard.mouse_click()
                self._last_click_time = now
                return "PINCH → Mouse Click"
        elif gesture == "pre_pinch":
            # Move mouse to index fingertip position (normalized 0~1)
            keyboard.mouse_move(landmarks[8].x, landmarks[8].y)
            return "PRE_PINCH → Mouse Move"

        return None

    # ------------------------------------------------------------------
    # Stop / reset
    # ------------------------------------------------------------------

    def stop(self, keyboard):
        """Release all movement keys."""
        if self._current_move_key:
            keyboard.release_key(self._current_move_key)
            self._current_move_key = None
        self._move_history.clear()
        self.last_move_cmd = "stop"

    def reset(self):
        """Reset all state."""
        self._move_history.clear()
        self._current_move_key = None
        self._prev_gesture = {"Left": "unknown", "Right": "unknown"}
        self._wrist_history_left.clear()
        self._wrist_history_right.clear()
        self.last_move_cmd = "stop"
        self.last_action = ""
        self.last_combo = ""
