# 手势识别 + 键盘/鼠标控制（整合版 v6）
# 双引擎并行 + 键盘/鼠标输出：
#   - 简单手势: 左手食指指右→D, 指左→A (持续)
#               右手张开+向上挥→Space (瞬发)
#               右手捏合(pinch)→鼠标移动+点击
#   - LSTM 手操: 复杂动作 1-7 → 数字键 1-7 (瞬发)
#
# 用法: python demo_gesture.py

import numpy as np
import torch
import cv2
import time
import ctypes
from collections import deque, Counter

from pynput.mouse import Controller as MouseController, Button as MouseButton

from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from keyboard_controller import KeyboardController
from config import CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT
from train_gesture import GestureLSTM, normalize_coords, add_velocity_features

MODEL_PATH = "gesture_model.pt"

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

# ===== LSTM 检测参数 =====
LSTM_CONFIDENCE_THRESHOLD = 0.50
CONSISTENCY_REQUIRED = 3
CONSISTENCY_WINDOW = 5

# ===== 键盘映射 =====
MOVE_KEYS = {
    "point_right": "d",
    "point_left":  "a",
}

EXERCISE_KEYS = {
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
}

SWIPE_UP_KEY = "space"
SWIPE_UP_THRESHOLD = 0.10
SWIPE_DETECT_FRAMES = 6
ACTION_COOLDOWN = 0.8
MOVE_DEBOUNCE_FRAMES = 5

# ===== 鼠标（pinch 捏合控制）=====
MOUSE_SMOOTH = 0.35
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


def main():
    import os
    has_lstm = os.path.exists(MODEL_PATH)

    if has_lstm:
        print("加载 LSTM 模型...")
        model, device, idx_to_label, window_size, use_velocity = load_model(MODEL_PATH)
        print(f"LSTM 就绪 | 窗口: {window_size}帧 | 速度特征: {use_velocity} | 类别: {list(idx_to_label.values())}")
    else:
        print(f"[提示] 未找到 {MODEL_PATH}，仅使用简单手势识别")
        model, device, idx_to_label, window_size, use_velocity = None, None, {}, 30, False

    tracker = HandTracker()
    keyboard = KeyboardController()
    mouse = MouseController()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("打不开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    frame_buffer = []
    prediction_history = deque(maxlen=CONSISTENCY_WINDOW)

    # LSTM 状态
    lstm_prediction = ""
    lstm_confidence = 0.0
    lstm_label_idx = -1
    lstm_probs = None
    lstm_confirmed = ""
    lstm_confirmed_conf = 0.0
    lstm_confirmed_idx = -1

    # 输出模式
    active_output = "SIMPLE"
    prev_active_output = "SIMPLE"

    # 键盘状态
    move_history = deque(maxlen=MOVE_DEBOUNCE_FRAMES)
    current_move_key = None
    right_wrist_history = deque(maxlen=SWIPE_DETECT_FRAMES)
    last_swipe_time = 0
    last_exercise_triggered = None
    triggered_display = ""
    triggered_time = 0

    # 鼠标状态
    was_pinching = False
    smooth_mouse_x = float(SCREEN_W) / 2
    smooth_mouse_y = float(SCREEN_H) / 2

    print("=" * 50)
    print("手势键盘/鼠标控制 v6 | 按 Q 退出")
    print(f"  左手: 食指指右→D  食指指左→A")
    print(f"  右手: 张开+向上挥→Space")
    print(f"  右手: 捏合(pinch)→鼠标移动+点击")
    print(f"  手操 1-7 → 数字键 1-7")
    print(f"  屏幕: {SCREEN_W}x{SCREEN_H}")
    print("=" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()

            results = tracker.detect(frame)

            left_lms = None
            right_lms = None

            if results.hand_landmarks:
                for lms, handed in zip(results.hand_landmarks, results.handedness):
                    hand_type = handed[0].category_name
                    if hand_type == "Left":
                        left_lms = lms
                    elif hand_type == "Right":
                        right_lms = lms
                    tracker.draw_hand(frame, lms, w, h)

            num_hands = (1 if left_lms else 0) + (1 if right_lms else 0)
            both_hands = left_lms is not None and right_lms is not None

            # ===== 简单手势 =====
            simple_results = {}
            if results.hand_landmarks:
                for lms, handed in zip(results.hand_landmarks, results.handedness):
                    hand_type = handed[0].category_name
                    gesture, conf, palm_facing = GestureRecognizer.recognize(
                        lms, hand_type
                    )
                    simple_results[hand_type] = (gesture, palm_facing)

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

                if (lstm_confidence >= LSTM_CONFIDENCE_THRESHOLD
                        and lstm_prediction != "idle"):
                    prediction_history.append(lstm_prediction)
                else:
                    prediction_history.append(None)

                counts = Counter(prediction_history)
                top_pred, top_count = counts.most_common(1)[0]

                if top_pred is not None and top_count >= CONSISTENCY_REQUIRED:
                    lstm_confirmed = top_pred
                    lstm_confirmed_conf = lstm_confidence
                    lstm_confirmed_idx = lstm_label_idx
                    lstm_is_exercise = True
                elif (lstm_prediction == "idle"
                      or lstm_confidence < LSTM_CONFIDENCE_THRESHOLD):
                    lstm_confirmed = ""
                    lstm_is_exercise = False

            elif not both_hands:
                prediction_history.clear()
                lstm_confirmed = ""
                lstm_probs = None
                lstm_prediction = ""

            if num_hands == 0:
                frame_buffer.clear()

            # ===== 输出决策 =====
            if num_hands <= 1:
                active_output = "SIMPLE"
            elif lstm_is_exercise:
                active_output = "LSTM"
            else:
                active_output = "SIMPLE"

            # ===== 键盘/鼠标控制 =====

            if active_output != prev_active_output:
                if active_output == "LSTM":
                    if current_move_key:
                        keyboard.release_key(current_move_key)
                        current_move_key = None
                    move_history.clear()
                    right_wrist_history.clear()
                else:
                    last_exercise_triggered = None
                prev_active_output = active_output

            r_mouse_mode = ""  # "", "track", "click"

            if active_output == "SIMPLE":
                # --- 左手: 食指方向 → 持续按键 ---
                raw_key = None
                if "Left" in simple_results:
                    gesture = simple_results["Left"][0]
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

                # --- 右手动作 ---
                if right_lms and "Right" in simple_results:
                    r_gesture = simple_results["Right"][0]

                    # pinch / pre_pinch → 鼠标控制
                    if r_gesture in ("pinch", "pre_pinch"):
                        right_wrist_history.clear()

                        mid_x = (right_lms[4].x + right_lms[8].x) / 2
                        mid_y = (right_lms[4].y + right_lms[8].y) / 2

                        sx = (mid_x - MOUSE_MAP_X[0]) / (
                            MOUSE_MAP_X[1] - MOUSE_MAP_X[0]
                        )
                        sy = (mid_y - MOUSE_MAP_Y[0]) / (
                            MOUSE_MAP_Y[1] - MOUSE_MAP_Y[0]
                        )
                        sx = max(0.0, min(1.0, sx))
                        sy = max(0.0, min(1.0, sy))
                        target_x = sx * SCREEN_W
                        target_y = sy * SCREEN_H

                        smooth_mouse_x += MOUSE_SMOOTH * (
                            target_x - smooth_mouse_x
                        )
                        smooth_mouse_y += MOUSE_SMOOTH * (
                            target_y - smooth_mouse_y
                        )
                        mouse.position = (
                            int(smooth_mouse_x),
                            int(smooth_mouse_y),
                        )

                        if r_gesture == "pinch":
                            r_mouse_mode = "click"
                            if not was_pinching:
                                mouse.click(MouseButton.left)
                                triggered_display = "Mouse Click"
                                triggered_time = now
                        else:
                            r_mouse_mode = "track"

                    # 张开/四指 + 向上挥 → Space
                    elif r_gesture in ("open", "four"):
                        right_wrist_history.append(right_lms[0].y)
                        if len(right_wrist_history) >= 2:
                            delta = (right_wrist_history[0]
                                     - right_wrist_history[-1])
                            if (delta > SWIPE_UP_THRESHOLD
                                    and now - last_swipe_time > ACTION_COOLDOWN):
                                keyboard.tap_key(SWIPE_UP_KEY)
                                last_swipe_time = now
                                right_wrist_history.clear()
                                triggered_display = "Space ^"
                                triggered_time = now
                    else:
                        right_wrist_history.clear()
                else:
                    right_wrist_history.clear()

                was_pinching = (r_mouse_mode == "click")
                last_exercise_triggered = None

            elif active_output == "LSTM" and lstm_confirmed:
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

            # ===== HUD =====
            y_offset = 25

            out_color = ((200, 200, 0) if active_output == "SIMPLE"
                         else (0, 180, 255))
            cv2.putText(frame, f"Output: {active_output}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, out_color, 2)
            y_offset += 25

            cv2.putText(frame, f"Hands: {num_hands}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y_offset += 20

            # 移动键
            if current_move_key:
                mk_text = f"Move: {current_move_key.upper()}"
                mk_color = (0, 255, 0)
            else:
                mk_text = "Move: ---"
                mk_color = (100, 100, 100)
            cv2.putText(frame, mk_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, mk_color, 2)
            y_offset += 22

            # 鼠标状态
            if r_mouse_mode == "click":
                cv2.putText(frame, "Mouse: CLICK", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                y_offset += 22
            elif r_mouse_mode == "track":
                cv2.putText(frame, "Mouse: TRACK", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                y_offset += 22

            # 瞬发提示
            if now - triggered_time < 1.0:
                cv2.putText(frame, triggered_display, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                y_offset += 24

            # 简单手势区
            if active_output == "SIMPLE":
                cv2.putText(frame, "-- Simple --", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 0), 1)
                y_offset += 18

                for hand_type in ("Left", "Right"):
                    if hand_type in simple_results:
                        gesture, palm_facing = simple_results[hand_type]
                        tag = "F" if palm_facing else "B"
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
                            cv2.putText(
                                frame,
                                f"{gesture} [{tag}]",
                                (int(wrist.x * w) - 40,
                                 int(wrist.y * h) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                            )

            # LSTM 区
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

            cv2.imshow("Gesture Control - Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        keyboard.close()
        cap.release()
        cv2.destroyAllWindows()
        print("已退出")


if __name__ == "__main__":
    main()
