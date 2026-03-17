# Command mapper: left-hand continuous move / right-hand tap / two-hand combo → keyboard keys.

import math
import time
from collections import deque, Counter

from config import (
    MOVE_STABLE_FRAMES, ACTION_COOLDOWN, COMBO_COOLDOWN,
    SWIPE_UP_THRESHOLD, SWIPE_HISTORY_FRAMES, HANDS_TOUCH_THRESHOLD,
    POINTING_X_THRESHOLD,
)


class CommandMapper:
    """Map gestures to keyboard: left continuous (D/A), right tap (W/Shift), combo (E/Q)."""

    # Right hand tap keys
    RIGHT_HAND_ACTION = {
        "swipe_up":  "w",
        "pointing":  "shift",
    }

    # Two-hand combo keys
    COMBO_ACTION = {
        "fist_to_open": "e",
        "wrist_clash":  "q",
    }

    _FIST_TO_OPEN_WINDOW = 8

    def __init__(self):
        self._move_history = deque(maxlen=MOVE_STABLE_FRAMES)
        self._current_move_key = None
        self._last_action_time = 0
        self._last_combo_time = 0
        self._prev_right_gesture = "unknown"
        self._both_fist_counter = 0
        self._right_wrist_history = deque(maxlen=SWIPE_HISTORY_FRAMES)
        self.last_move_cmd = "stop"
        self.last_action = ""
        self.last_combo = ""

    def update(self, left_data, right_data, keyboard):
        """Called per frame; process gestures → keyboard. Returns list of triggered action strings."""
        triggered = []
        now = time.time()

        # --- 1. Two-hand combo (highest priority) ---
        combo = self._check_combo(left_data, right_data, now)
        if combo:
            keyboard.tap_key(self.COMBO_ACTION[combo])
            self.last_combo = combo
            triggered.append(f"COMBO {combo} → {self.COMBO_ACTION[combo].upper()}")

        # --- 2. Left hand move (continuous) ---
        new_key = self._update_movement(left_data)
        if new_key != self._current_move_key:
            if self._current_move_key:
                keyboard.release_key(self._current_move_key)
            if new_key:
                keyboard.hold_key(new_key)
            self._current_move_key = new_key

        # --- 3. Right hand action (tap) ---
        action = self._check_action(right_data, now)
        if action:
            keyboard.tap_key(self.RIGHT_HAND_ACTION[action])
            self.last_action = action
            triggered.append(f"ACTION {action} → {self.RIGHT_HAND_ACTION[action].upper()}")

        # --- Update prev state ---
        self._update_prev_state(left_data, right_data)

        return triggered

    @staticmethod
    def _get_pointing_direction(landmarks):
        """Index tip->mcp angle: right→'d', left→'a', non-horizontal→None. Only ±45° triggers move."""
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
        """Return key to hold (str) or None after debounce."""
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

    def _check_action(self, right_data, now):
        if right_data is None:
            self._right_wrist_history.clear()
            return None

        if now - self._last_action_time < ACTION_COOLDOWN:
            return None

        gesture, palm_facing, landmarks = right_data

        # Swipe up (y decreasing = up)
        wrist_y = landmarks[0].y
        self._right_wrist_history.append(wrist_y)

        if len(self._right_wrist_history) >= 2:
            delta = self._right_wrist_history[0] - self._right_wrist_history[-1]
            if delta > SWIPE_UP_THRESHOLD:
                self._last_action_time = now
                self._right_wrist_history.clear()
                return "swipe_up"

        # Pointing: edge trigger (non-pointing -> pointing)
        is_pointing = gesture in ("pointing", "point_right", "point_left", "point_up", "point_down")
        was_pointing = self._prev_right_gesture in ("pointing", "point_right", "point_left", "point_up", "point_down")
        if is_pointing and not was_pointing:
            self._last_action_time = now
            return "pointing"

        return None

    def _check_combo(self, left_data, right_data, now):
        if left_data is None or right_data is None:
            return None

        if now - self._last_combo_time < COMBO_COOLDOWN:
            return None

        l_gesture, _, l_lms = left_data
        r_gesture, _, r_lms = right_data

        # Fists to open (E): had both fists in last N frames, now both open
        if (self._both_fist_counter > 0
                and l_gesture == "open" and r_gesture == "open"):
            self._last_combo_time = now
            self._both_fist_counter = 0
            return "fist_to_open"

        # Wrist clash (Q): wrist distance < threshold
        dist = ((l_lms[0].x - r_lms[0].x) ** 2
                + (l_lms[0].y - r_lms[0].y) ** 2) ** 0.5
        if dist < HANDS_TOUCH_THRESHOLD:
            self._last_combo_time = now
            return "wrist_clash"

        return None

    def _update_prev_state(self, left_data, right_data):
        self._prev_right_gesture = right_data[0] if right_data else "unknown"

        both_fist = (left_data is not None and right_data is not None
                     and left_data[0] == "fist" and right_data[0] == "fist")
        if both_fist:
            self._both_fist_counter = self._FIST_TO_OPEN_WINDOW
        elif self._both_fist_counter > 0:
            self._both_fist_counter -= 1

    def stop(self, keyboard):
        """Stop all movement immediately."""
        if self._current_move_key:
            keyboard.release_key(self._current_move_key)
            self._current_move_key = None
        self._move_history.clear()
        self.last_move_cmd = "stop"

    def reset(self):
        """Reset all state."""
        self._move_history.clear()
        self._current_move_key = None
        self._prev_right_gesture = "unknown"
        self._both_fist_counter = 0
        self._right_wrist_history.clear()
        self.last_move_cmd = "stop"
        self.last_action = ""
        self.last_combo = ""
