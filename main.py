# Gesture recognition + keyboard/mouse control
# Dual engine (simple + LSTM) with keyboard/mouse output:
#   - Simple: point right→D, point left→A, fist palm→W, fist back→S (hold)
#     pinch→mouse click, pre_pinch→mouse move (left hand only)
#   - LSTM: complex gestures 1-7 → keys (tap)
#   - Both hands: suppress simple gestures, use LSTM
# Usage: python main.py

import math
import numpy as np
import torch
import cv2
import time
import ctypes
import os
from collections import deque, Counter

from pynput.mouse import Controller as MouseController, Button as MouseButton

from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from keyboard_controller import KeyboardController
from config import CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, BASE_DIR
from train_gesture import GestureLSTM, normalize_coords, add_velocity_features

MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pt")

LABEL_DISPLAY = {
    "1": "1 leftright",
    "2": "2 shoubeihuipai",
    "3": "3 shouwan shangxia",
    "4": "4 shouwan zuoyou",
    "5": "5 zhua shouzhi",
    "6": "6 zhanggen huipai",
    "7": "7 hukou huji",
    "idle": "idle",
}

COLORS = [
    (0, 255, 0),
    (255, 165, 0),
    (0, 255, 255),
    (255, 0, 0),
    (255, 0, 255),
    (0, 165, 255),
    (128, 0, 255),
    (128, 128, 128),
]

# ===== LSTM params =====
LSTM_CONFIDENCE_THRESHOLD = 0.45
CONSISTENCY_REQUIRED = 3
CONSISTENCY_WINDOW = 6
SIMPLE_STABLE_FRAMES_WHEN_TWO_HANDS = 8

# Pinch anti-flicker
PINCH_STABLE_FRAMES = 4
LSTM_CONSISTENCY_EXTRA_FOR_1_4 = 2
LSTM_CONSISTENCY_EXTRA_SWITCH_1_4 = 2
LSTM_PINCH_SUPPRESS_THRESHOLD = 0.45

# ===== Keyboard mapping =====
MOVE_KEYS = {
    "point_right": "d",
    "point_left":  "a",
}

EXERCISE_KEYS = {
    "1": "f",        # leftright → F
    "5": "space",    # Grab fingers → Space
    "6": "e",        # Palm-heel clap → E
    "7": "t",        # Thumb-webbing tap → T
}

SWIPE_UP_KEY = "space"
SWIPE_UP_THRESHOLD = 0.07
SWIPE_DETECT_FRAMES = 8
ACTION_COOLDOWN = 0.8
LSTM_CONFIDENCE_FOR_5 = 0.45
LSTM_CONSISTENCY_FOR_5 = 3
LSTM_CONFIDENCE_FOR_6 = 0.35
LSTM_CONSISTENCY_FOR_6 = 2
MOVE_DEBOUNCE_FRAMES = 5

# ===== Open palm still → P =====
OPEN_PALM_STILL_DURATION = 3.0       # hold duration (seconds)
OPEN_PALM_STILL_MOVE_THRESH = 0.05   # max wrist movement per frame (normalized)
FINGERS_TOGETHER_THRESH = 0.12       # max distance between adjacent fingertips

# ===== Mouse (pinch control) =====
MOUSE_SMOOTH = 0.15
MOUSE_MAP_X = (0.15, 0.85)
MOUSE_MAP_Y = (0.15, 0.85)

try:
    _u32 = ctypes.windll.user32
    SCREEN_W = _u32.GetSystemMetrics(0)
    SCREEN_H = _u32.GetSystemMetrics(1)
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080


def load_model(model_path):
    checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")
    label_map = checkpoint["label_map"]
    idx_to_label = {v: k for k, v in label_map.items()}

    model = GestureLSTM(
        checkpoint["input_dim"],
        checkpoint["hidden_dim"],
        checkpoint["num_layers"],
        checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    use_velocity = checkpoint["input_dim"] > 126
    return model, device, idx_to_label, checkpoint["window_size"], use_velocity


def extract_frame_coords(left_lms, right_lms):
    def hand_to_list(lms):
        if lms is None:
            return [0.0] * 63
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]
        zs = [lm.z for lm in lms]
        return xs + ys + zs
    return hand_to_list(left_lms) + hand_to_list(right_lms)


def _fingers_together(lms):
    """Check four fingers (index, middle, ring, pinky) tips are close together."""
    tips = [lms[8], lms[12], lms[16], lms[20]]
    for i in range(len(tips) - 1):
        d = math.sqrt((tips[i].x - tips[i+1].x)**2 + (tips[i].y - tips[i+1].y)**2)
        if d > FINGERS_TOGETHER_THRESH:
            return False
    return True


def main():
    has_lstm = os.path.exists(MODEL_PATH)

    if has_lstm:
        print("Loading LSTM model...")
        model, device, idx_to_label, window_size, use_velocity = load_model(MODEL_PATH)
        print(f"LSTM ready | window: {window_size} frames | velocity: {use_velocity} | classes: {list(idx_to_label.values())}")
    else:
        print(f"[Note] {MODEL_PATH} not found, using simple gestures only")
        model, device, idx_to_label, window_size, use_velocity = None, None, {}, 30, False

    tracker = HandTracker()
    keyboard = KeyboardController()
    mouse = MouseController()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    WIN_NAME = "Gesture Control - ESC to quit"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 1)

    frame_buffer = []
    prediction_history = deque(maxlen=CONSISTENCY_WINDOW)

    # LSTM state
    lstm_prediction = ""
    lstm_confidence = 0.0
    lstm_label_idx = -1
    lstm_probs = None
    lstm_confirmed = ""
    lstm_confirmed_conf = 0.0
    lstm_confirmed_idx = -1

    # Output mode
    active_output = "SIMPLE"
    prev_active_output = "SIMPLE"

    # Keyboard state
    move_history = deque(maxlen=MOVE_DEBOUNCE_FRAMES)
    current_move_key = None
    right_wrist_history = deque(maxlen=SWIPE_DETECT_FRAMES)
    last_swipe_time = 0
    # Right hand grab (open↔fist repeated) → Space
    right_saw_open = False
    grab_timestamps = []       # timestamps of each open→fist transition
    last_grab_trigger = 0
    GRAB_WINDOW = 2.0          # time window (seconds)
    GRAB_COUNT_REQUIRED = 2    # open→fist count needed within window
    last_exercise_triggered = None
    triggered_display = ""
    triggered_time = 0

    # Mouse state
    was_pinching = False
    smooth_mouse_x = float(SCREEN_W) / 2
    smooth_mouse_y = float(SCREEN_H) / 2

    # When both hands: delay simple gesture output
    simple_stable_key = None
    simple_stable_count = 0
    prev_lstm_confidence = 0.0

    # Pinch anti-flicker
    pinch_last_raw = None
    pinch_same_count = 0
    pinch_smoothed_state = ""

    # Open palm still → P
    open_palm_start = 0.0          # timestamp when pose first detected
    open_palm_active = False       # currently in pose
    open_palm_triggered = False    # already fired P for this hold
    open_palm_prev_wrists = None   # (left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y)

    # Camera retry
    read_fail_count = 0
    MAX_READ_FAILS = 30

    print("=" * 50)
    print("Gesture Control | Press ESC to quit")
    print(f"  Left: point_right→D  point_left→A  fist palm→W  fist back→S")
    print(f"  Left: pre_pinch→mouse move  pinch→left click")
    print(f"  Right: palm up + swipe up→Space")
    print(f"  Exercises 1-7 → F/2/3/4/Space/E/T")
    print(f"  Screen: {SCREEN_W}x{SCREEN_H}")
    print("=" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                read_fail_count += 1
                if read_fail_count >= MAX_READ_FAILS:
                    print(f"Camera lost ({MAX_READ_FAILS} consecutive failures), exiting")
                    break
                time.sleep(0.03)
                continue
            read_fail_count = 0

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()

            results = tracker.detect(frame)

            left_lms = None
            right_lms = None

            if results.hand_landmarks:
                for lms, handed in zip(results.hand_landmarks, results.handedness):
                    # Flip label: mirrored frame swaps left/right
                    raw_type = handed[0].category_name
                    hand_type = "Right" if raw_type == "Left" else "Left"
                    if hand_type == "Left":
                        left_lms = lms
                    elif hand_type == "Right":
                        right_lms = lms
                    tracker.draw_hand(frame, lms, w, h)

            num_hands = (1 if left_lms else 0) + (1 if right_lms else 0)
            # Only count as "both hands" if both have a recognized gesture
            both_hands_raw = left_lms is not None and right_lms is not None
            both_hands = both_hands_raw

            # ===== Simple gestures =====
            simple_results = {}
            if results.hand_landmarks:
                for lms, handed in zip(results.hand_landmarks, results.handedness):
                    raw_type = handed[0].category_name
                    hand_type = "Right" if raw_type == "Left" else "Left"
                    gesture, conf, palm_facing = GestureRecognizer.recognize(
                        lms, hand_type
                    )
                    simple_results[hand_type] = (gesture, palm_facing)

            # If one hand is unknown, don't treat as both_hands → stay in SIMPLE
            if both_hands_raw:
                l_gest = simple_results.get("Left", ("unknown",))[0]
                r_gest = simple_results.get("Right", ("unknown",))[0]
                both_hands = (l_gest != "unknown" and r_gest != "unknown")

            # ===== LSTM =====
            lstm_is_exercise = False

            if num_hands > 0:
                coords = extract_frame_coords(left_lms, right_lms)
                frame_buffer.append(coords)
                if len(frame_buffer) > window_size * 2:
                    frame_buffer = frame_buffer[-window_size * 2:]

            if has_lstm and both_hands and len(frame_buffer) >= window_size:
                normed = normalize_coords(list(frame_buffer[-window_size:]))
                if use_velocity:
                    normed = add_velocity_features(normed)
                tensor = torch.FloatTensor(normed).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor)
                    lstm_probs = torch.softmax(output, dim=1)[0]
                    lstm_label_idx = lstm_probs.argmax().item()
                    lstm_confidence = lstm_probs[lstm_label_idx].item()
                    lstm_prediction = idx_to_label[lstm_label_idx]

                conf_ok = lstm_confidence >= LSTM_CONFIDENCE_THRESHOLD
                if lstm_prediction == "5":
                    conf_ok = conf_ok or lstm_confidence >= LSTM_CONFIDENCE_FOR_5
                elif lstm_prediction == "6":
                    conf_ok = conf_ok or lstm_confidence >= LSTM_CONFIDENCE_FOR_6
                LSTM_BLOCKED = ("idle", "4")  # 4=shouwan zuoyou disabled
                if conf_ok and lstm_prediction not in LSTM_BLOCKED:
                    prediction_history.append(lstm_prediction)
                else:
                    prediction_history.append(None)

                counts = Counter(prediction_history)
                top_pred, top_count = counts.most_common(1)[0]

                required = CONSISTENCY_REQUIRED
                if top_pred == "5":
                    required = min(required, LSTM_CONSISTENCY_FOR_5)
                elif top_pred == "6":
                    required = min(required, LSTM_CONSISTENCY_FOR_6)
                elif top_pred == "1":
                    required += LSTM_CONSISTENCY_EXTRA_FOR_1_4

                if top_pred is not None and top_count >= required:
                    lstm_confirmed = top_pred
                    lstm_confirmed_conf = lstm_confidence
                    lstm_confirmed_idx = lstm_label_idx
                    lstm_is_exercise = True
                else:
                    low_conf = lstm_confidence < LSTM_CONFIDENCE_THRESHOLD
                    if lstm_prediction == "5":
                        low_conf = lstm_confidence < LSTM_CONFIDENCE_FOR_5
                    if lstm_prediction == "idle" or low_conf:
                        lstm_confirmed = ""
                        lstm_is_exercise = False

            elif not both_hands:
                prediction_history.clear()
                lstm_confirmed = ""
                lstm_probs = None
                lstm_prediction = ""

            if num_hands == 0:
                frame_buffer.clear()

            # ===== Output decision =====
            # Both hands → always prefer LSTM; only fall back to SIMPLE
            # if LSTM buffer not ready yet AND no rising confidence
            if num_hands <= 1:
                active_output = "SIMPLE"
            elif lstm_is_exercise:
                active_output = "LSTM"
            elif both_hands and has_lstm and len(frame_buffer) >= window_size // 2:
                # Buffer half-full: suppress simple, wait for LSTM
                active_output = "WAIT_LSTM"
            else:
                active_output = "SIMPLE"

            # ===== Keyboard/mouse control =====

            if active_output != prev_active_output:
                if active_output != "LSTM":
                    last_exercise_triggered = None
                prev_active_output = active_output

            r_mouse_mode = ""  # "", "track", "click"

            # --- Left hand WASD: ALWAYS active ---
            raw_key = None
            if "Left" in simple_results:
                gesture, palm_facing = simple_results["Left"][0], simple_results["Left"][1]
                if gesture == "fist":
                    raw_key = "w" if palm_facing else "s"
                else:
                    raw_key = MOVE_KEYS.get(gesture)

            move_history.append(raw_key)
            new_key = current_move_key
            if len(move_history) >= move_history.maxlen:
                cnt = Counter(move_history)
                most_common, count = cnt.most_common(1)[0]
                if count >= move_history.maxlen:
                    new_key = most_common

            if new_key != current_move_key:
                if current_move_key:
                    keyboard.release_key(current_move_key)
                if new_key:
                    keyboard.hold_key(new_key)
                current_move_key = new_key

            # --- Right hand: repeated open↔fist → Space (always active) ---
            if right_lms and "Right" in simple_results:
                r_gesture = simple_results["Right"][0]
                if r_gesture in ("open", "three", "four"):
                    right_saw_open = True
                elif r_gesture == "fist" and right_saw_open:
                    right_saw_open = False
                    grab_timestamps.append(now)
                    grab_timestamps = [t for t in grab_timestamps
                                       if now - t <= GRAB_WINDOW]
                    if (len(grab_timestamps) >= GRAB_COUNT_REQUIRED
                            and now - last_grab_trigger > GRAB_WINDOW):
                        keyboard.tap_key("space")
                        last_grab_trigger = now
                        grab_timestamps.clear()
                        triggered_display = f"Grab x{GRAB_COUNT_REQUIRED} → Space"
                        triggered_time = now
                elif r_gesture != "fist":
                    pass

            # --- Left hand pinch/pre_pinch → mouse ---
            suppress_pinch_for_lstm = (
                both_hands and has_lstm
                and lstm_confidence >= LSTM_PINCH_SUPPRESS_THRESHOLD
            )
            if left_lms and "Left" in simple_results and not suppress_pinch_for_lstm:
                l_gesture_raw = simple_results["Left"][0]
                raw_pinch = l_gesture_raw if l_gesture_raw in ("pinch", "pre_pinch") else ""
                if raw_pinch != pinch_last_raw:
                    pinch_last_raw = raw_pinch
                    pinch_same_count = 1
                else:
                    pinch_same_count += 1
                if pinch_same_count >= PINCH_STABLE_FRAMES:
                    pinch_smoothed_state = raw_pinch
                l_gesture = pinch_smoothed_state if pinch_smoothed_state else l_gesture_raw

                if l_gesture in ("pinch", "pre_pinch"):
                    mid_x = (left_lms[4].x + left_lms[8].x) / 2
                    mid_y = (left_lms[4].y + left_lms[8].y) / 2
                    sx = (mid_x - MOUSE_MAP_X[0]) / (MOUSE_MAP_X[1] - MOUSE_MAP_X[0])
                    sy = (mid_y - MOUSE_MAP_Y[0]) / (MOUSE_MAP_Y[1] - MOUSE_MAP_Y[0])
                    sx = max(0.0, min(1.0, sx))
                    sy = max(0.0, min(1.0, sy))
                    target_x = sx * SCREEN_W
                    target_y = sy * SCREEN_H
                    smooth_mouse_x += MOUSE_SMOOTH * (target_x - smooth_mouse_x)
                    smooth_mouse_y += MOUSE_SMOOTH * (target_y - smooth_mouse_y)
                    mouse.position = (int(smooth_mouse_x), int(smooth_mouse_y))

                    if l_gesture == "pinch":
                        r_mouse_mode = "click"
                        if not was_pinching:
                            mouse.click(MouseButton.left)
                            triggered_display = "Mouse Click"
                            triggered_time = now
                    else:
                        r_mouse_mode = "track"
                else:
                    pinch_last_raw = None
                    pinch_same_count = 0
                    pinch_smoothed_state = ""
            else:
                pinch_last_raw = None
                pinch_same_count = 0
                pinch_smoothed_state = ""

            was_pinching = (r_mouse_mode == "click")

            # --- Both hands open + palm facing + fingers together + still → P ---
            # Must run BEFORE LSTM trigger so it can suppress LSTM
            open_palm_ok = False
            if both_hands_raw and left_lms and right_lms:
                l_g, l_pf = simple_results.get("Left", ("unknown", False))
                r_g, r_pf = simple_results.get("Right", ("unknown", False))
                _OPEN_LIKE = ("open", "four", "three")
                if (l_g in _OPEN_LIKE and r_g in _OPEN_LIKE and l_pf and r_pf
                        and _fingers_together(left_lms)
                        and _fingers_together(right_lms)):
                    cur_wrists = (left_lms[0].x, left_lms[0].y,
                                  right_lms[0].x, right_lms[0].y)
                    still = True
                    if open_palm_prev_wrists is not None:
                        for a, b in zip(cur_wrists, open_palm_prev_wrists):
                            if abs(a - b) > OPEN_PALM_STILL_MOVE_THRESH:
                                still = False
                                break
                    open_palm_prev_wrists = cur_wrists

                    if still:
                        if not open_palm_active:
                            open_palm_active = True
                            open_palm_start = now
                            open_palm_triggered = False
                        elif (not open_palm_triggered
                              and now - open_palm_start >= OPEN_PALM_STILL_DURATION):
                            keyboard.tap_key("p")
                            open_palm_triggered = True
                            triggered_display = "Open Palm Still 3s → P"
                            triggered_time = now
                        open_palm_ok = True
                    else:
                        open_palm_active = False
                        open_palm_triggered = False

            if not open_palm_ok:
                open_palm_active = False
                open_palm_triggered = False
                open_palm_prev_wrists = None

            # --- LSTM exercise trigger (suppressed when open-palm-still active) ---
            if active_output == "LSTM" and lstm_confirmed and not open_palm_ok:
                key = EXERCISE_KEYS.get(lstm_confirmed)
                if key and lstm_confirmed != last_exercise_triggered:
                    keyboard.tap_key(key)
                    last_exercise_triggered = lstm_confirmed
                    display = LABEL_DISPLAY.get(lstm_confirmed, lstm_confirmed)
                    triggered_display = f"{display} -> [{key}]"
                    triggered_time = now
                was_pinching = False

            if num_hands == 0 and current_move_key:
                keyboard.release_key(current_move_key)
                current_move_key = None
                move_history.clear()

            prev_lstm_confidence = lstm_confidence

            # ===== HUD =====
            y_offset = 25

            if active_output == "SIMPLE":
                out_color = (200, 200, 0)
            elif active_output == "WAIT_LSTM":
                out_color = (100, 100, 255)
            else:
                out_color = (0, 180, 255)
            cv2.putText(frame, f"Output: {active_output}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, out_color, 2)
            y_offset += 25

            cv2.putText(frame, f"Hands: {num_hands}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y_offset += 20

            # Move key
            if current_move_key:
                mk_text = f"Move: {current_move_key.upper()}"
                mk_color = (0, 255, 0)
            else:
                mk_text = "Move: ---"
                mk_color = (100, 100, 100)
            cv2.putText(frame, mk_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, mk_color, 2)
            y_offset += 22

            # Mouse state
            if r_mouse_mode == "click":
                cv2.putText(frame, "Mouse: CLICK", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                y_offset += 22
            elif r_mouse_mode == "track":
                cv2.putText(frame, "Mouse: TRACK", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                y_offset += 22

            # Open palm still countdown
            if open_palm_active and not open_palm_triggered:
                elapsed = now - open_palm_start
                remaining = max(0, OPEN_PALM_STILL_DURATION - elapsed)
                bar_w = 120
                pct = min(elapsed / OPEN_PALM_STILL_DURATION, 1.0)
                cv2.rectangle(frame, (10, y_offset),
                              (10 + bar_w, y_offset + 10), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, y_offset),
                              (10 + int(bar_w * pct), y_offset + 10),
                              (0, 200, 255), -1)
                cv2.putText(frame, f"P in {remaining:.1f}s",
                            (140, y_offset + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
                y_offset += 16

            # Tap hint
            if now - triggered_time < 1.0:
                cv2.putText(frame, triggered_display, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                y_offset += 24

            # Simple gesture area
            if active_output == "SIMPLE":
                cv2.putText(frame, "-- Simple --", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 0), 1)
                y_offset += 18

                for hand_type in ("Left", "Right"):
                    if hand_type in simple_results:
                        gesture, palm_facing = simple_results[hand_type]
                        if gesture == "point_right":
                            tag = "D"
                        elif gesture == "point_left":
                            tag = "A"
                        elif gesture == "fist":
                            tag = "W" if palm_facing else "S"
                        elif gesture in ("pinch", "pre_pinch"):
                            tag = "Mouse"
                        else:
                            tag = "—"
                        color = ((0, 255, 0) if gesture != "unknown"
                                 else (128, 128, 128))
                        cv2.putText(
                            frame,
                            f"  {hand_type}: {gesture} [{tag}]",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                        )
                        y_offset += 20

                        lms = left_lms if hand_type == "Left" else right_lms
                        if lms:
                            wrist = lms[0]
                            label = f"{gesture} [{tag}]"
                            cv2.putText(
                                frame,
                                label,
                                (int(wrist.x * w) - 40,
                                 int(wrist.y * h) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                            )

            # LSTM area
            if has_lstm and both_hands:
                y_offset += 5

                buf_pct = min(len(frame_buffer) / window_size, 1.0)
                bar_w = 180
                cv2.rectangle(frame, (10, y_offset),
                              (10 + bar_w, y_offset + 8), (50, 50, 50), -1)
                b_color = (0, 200, 0) if buf_pct >= 1.0 else (100, 100, 100)
                cv2.rectangle(frame, (10, y_offset),
                              (10 + int(bar_w * buf_pct), y_offset + 8),
                              b_color, -1)
                y_offset += 14

                if lstm_prediction:
                    sec_color = ((0, 180, 255) if active_output == "LSTM"
                                 else (100, 100, 100))
                    cv2.putText(
                        frame,
                        f"LSTM: {lstm_prediction} {lstm_confidence:.0%}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, sec_color, 1,
                    )
                    y_offset += 18

                if active_output == "LSTM" and lstm_confirmed:
                    color = COLORS[lstm_confirmed_idx % len(COLORS)]
                    display = LABEL_DISPLAY.get(lstm_confirmed, lstm_confirmed)
                    key = EXERCISE_KEYS.get(lstm_confirmed, "?")
                    cv2.putText(frame, f"{display} -> [{key}]",
                                (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    y_offset += 32
                    cv2.putText(frame, f"{lstm_confirmed_conf:.0%}",
                                (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    y_offset += 22

                if lstm_probs is not None:
                    for label_idx in sorted(idx_to_label.keys()):
                        label_name = idx_to_label[label_idx]
                        prob_val = lstm_probs[label_idx].item()
                        bar_color = COLORS[label_idx % len(COLORS)]
                        bar_len = int(prob_val * 180)
                        cv2.rectangle(frame, (10, y_offset),
                                      (10 + bar_len, y_offset + 11),
                                      bar_color, -1)
                        cv2.putText(frame, label_name,
                                    (200, y_offset + 9),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                                    (200, 200, 200), 1)
                        y_offset += 14

            cv2.imshow(WIN_NAME, frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        if current_move_key:
            keyboard.release_key(current_move_key)
        keyboard.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Exited")


if __name__ == "__main__":
    main()
