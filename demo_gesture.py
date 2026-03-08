# 手势识别实时演示（整合版 v3）
# 双引擎并行 + 竞争输出：
#   - 简单手势引擎：每帧即时识别，追踪稳定性
#   - LSTM 引擎：持续缓冲推理（需要双手）
#   - 输出决策：基于各自置信度和稳定性竞争
#
# 不再做"模式切换"，两个引擎始终运行，
# 根据置信度和条件自动决定显示哪个结果。
#
# 用法: python demo_gesture.py

import math
import numpy as np
import torch
import cv2
from collections import deque, Counter

from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from config import CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT
from train_gesture import GestureLSTM, normalize_coords, INPUT_DIM

MODEL_PATH = "gesture_model.pt"

LABEL_DISPLAY = {
    "1": "1 leftright",
    "2": "2 shoubeihuipai",
    "3": "3 shouwan shangxia",
    "4": "4 shouwan zuoyou",
    "5": "5 zhua shouzhi",
    "6": "6 zhanggen huipai",
    "7": "7 hukou huji",
}

COLORS = [
    (0, 255, 0),    # 1 green
    (255, 165, 0),  # 2 orange
    (0, 255, 255),  # 3 yellow
    (255, 0, 0),    # 4 blue
    (255, 0, 255),  # 5 purple
    (0, 165, 255),  # 6 dark orange
    (128, 0, 255),  # 7 violet
]

# ========== 简单手势稳定性 ==========
SIMPLE_STABLE_FRAMES = 4        # 连续 N 帧同手势 = "稳定"

# ========== LSTM 参数 ==========
LSTM_CONFIDENCE_THRESHOLD = 0.70
ENTROPY_THRESHOLD = 1.4
CONSISTENCY_REQUIRED = 3
CONSISTENCY_WINDOW = 5
MOTION_WINDOW = 15
WRIST_MOTION_THRESHOLD = 0.045


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

    return model, device, idx_to_label, checkpoint["window_size"]


def extract_frame_coords(left_lms, right_lms):
    def hand_to_list(lms):
        if lms is None:
            return [0.0] * 63
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]
        zs = [lm.z for lm in lms]
        return xs + ys + zs
    return hand_to_list(left_lms) + hand_to_list(right_lms)


def calc_wrist_motion(frame_buffer, n_recent=None):
    if n_recent is None:
        n_recent = MOTION_WINDOW
    if len(frame_buffer) < 3:
        return 0.0, 0.0

    recent = frame_buffer[-n_recent:]
    arr = np.array(recent, dtype=np.float32)

    def _wrist_range(x_idx, y_idx):
        xs = arr[:, x_idx]
        ys = arr[:, y_idx]
        xs_nz = xs[xs != 0]
        ys_nz = ys[ys != 0]
        if len(xs_nz) < 3 or len(ys_nz) < 3:
            return 0.0
        return max(float(xs_nz.max() - xs_nz.min()),
                   float(ys_nz.max() - ys_nz.min()))

    return _wrist_range(0, 21), _wrist_range(63, 84)


def calc_entropy(probs_tensor):
    probs = probs_tensor.cpu().numpy()
    probs = probs[probs > 1e-8]
    return float(-np.sum(probs * np.log(probs)))


class GestureStabilizer:
    """追踪每只手的手势稳定性"""

    def __init__(self, required_frames=SIMPLE_STABLE_FRAMES):
        self.required = required_frames
        self._history = {}  # hand_type -> deque of gesture names

    def update(self, hand_type, gesture):
        if hand_type not in self._history:
            self._history[hand_type] = deque(maxlen=self.required + 2)
        self._history[hand_type].append(gesture)

    def is_stable(self, hand_type):
        """该手是否有稳定的（非 unknown）手势"""
        hist = self._history.get(hand_type)
        if not hist or len(hist) < self.required:
            return False
        recent = list(hist)[-self.required:]
        if recent[0] == "unknown":
            return False
        return all(g == recent[0] for g in recent)

    def stable_gesture(self, hand_type):
        """返回稳定的手势名，不稳定则返回 None"""
        if not self.is_stable(hand_type):
            return None
        return self._history[hand_type][-1]

    def clear(self, hand_type=None):
        if hand_type:
            self._history.pop(hand_type, None)
        else:
            self._history.clear()


def main():
    import os
    has_lstm = os.path.exists(MODEL_PATH)

    if has_lstm:
        print("加载 LSTM 模型...")
        model, device, idx_to_label, window_size = load_model(MODEL_PATH)
        print(f"LSTM 模型就绪 | 窗口: {window_size}帧 | 类别: {list(idx_to_label.values())}")
    else:
        print(f"[提示] 未找到 {MODEL_PATH}，仅使用简单手势识别")
        model, device, idx_to_label, window_size = None, None, {}, 30

    tracker = HandTracker()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("打不开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    frame_buffer = []
    stabilizer = GestureStabilizer()

    # LSTM 状态
    lstm_prediction = ""
    lstm_confidence = 0.0
    lstm_label_idx = -1
    lstm_probs = None
    entropy_value = 0.0
    prediction_history = deque(maxlen=CONSISTENCY_WINDOW)
    lstm_confirmed = ""
    lstm_confirmed_conf = 0.0
    lstm_confirmed_idx = -1
    lstm_status = ""

    # 运动量
    left_motion = 0.0
    right_motion = 0.0

    # 最终输出
    active_output = "NONE"  # "SIMPLE", "LSTM", "NONE"

    print("=" * 50)
    print("整合手势识别 v3 | 按 Q 退出")
    print("  双引擎并行 + 竞争输出")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

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

        # ===== 运动量计算（先于两个引擎）=====
        if num_hands > 0:
            coords = extract_frame_coords(left_lms, right_lms)
            frame_buffer.append(coords)
            if len(frame_buffer) > window_size + MOTION_WINDOW:
                frame_buffer = frame_buffer[-(window_size + MOTION_WINDOW):]
            left_motion, right_motion = calc_wrist_motion(frame_buffer)
        else:
            frame_buffer.clear()
            left_motion = 0.0
            right_motion = 0.0

        # ===== 引擎 1: 简单手势（每帧运行）=====
        simple_results = {}
        any_stable = False

        if results.hand_landmarks:
            for lms, handed in zip(results.hand_landmarks, results.handedness):
                hand_type = handed[0].category_name
                gesture, conf, palm_facing = GestureRecognizer.recognize(lms, hand_type)
                simple_results[hand_type] = (gesture, palm_facing)
                stabilizer.update(hand_type, gesture)

                # 稳定手势只在手腕静止时才生效
                # 做操时手腕在动，即使手势形状暂时一样也不算"稳定简单手势"
                wrist_still = (
                    (hand_type == "Left" and left_motion < WRIST_MOTION_THRESHOLD) or
                    (hand_type == "Right" and right_motion < WRIST_MOTION_THRESHOLD)
                )
                if stabilizer.is_stable(hand_type) and gesture != "unknown" and wrist_still:
                    any_stable = True

        # 清理消失的手
        if not left_lms:
            stabilizer.clear("Left")
        if not right_lms:
            stabilizer.clear("Right")

        # ===== 引擎 2: LSTM（双手时运行）=====
        lstm_ready = False

        if has_lstm and both_hands and len(frame_buffer) >= window_size:
            normed = normalize_coords(list(frame_buffer[-window_size:]))
            tensor = torch.FloatTensor(normed).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                lstm_probs = torch.softmax(output, dim=1)[0]
                lstm_label_idx = lstm_probs.argmax().item()
                lstm_confidence = lstm_probs[lstm_label_idx].item()
                lstm_prediction = idx_to_label[lstm_label_idx]

            entropy_value = calc_entropy(lstm_probs)
            conf_pass = lstm_confidence >= LSTM_CONFIDENCE_THRESHOLD
            entropy_pass = entropy_value <= ENTROPY_THRESHOLD

            if not conf_pass:
                lstm_status = f"LOW CONF {lstm_confidence:.0%}"
                prediction_history.append(None)
            elif not entropy_pass:
                lstm_status = f"HIGH ENTROPY {entropy_value:.2f}"
                prediction_history.append(None)
            else:
                prediction_history.append(lstm_prediction)
                counts = Counter(prediction_history)
                top_pred, top_count = counts.most_common(1)[0]

                if top_pred is not None and top_count >= CONSISTENCY_REQUIRED:
                    lstm_confirmed = top_pred
                    lstm_confirmed_conf = lstm_confidence
                    lstm_confirmed_idx = lstm_label_idx
                    lstm_status = "CONFIRMED"
                    lstm_ready = True
                else:
                    lstm_status = f"WAIT {top_count}/{CONSISTENCY_REQUIRED}"
        elif not both_hands:
            prediction_history.clear()
            lstm_confirmed = ""
            lstm_probs = None
            lstm_status = ""

        # ===== 输出决策：竞争 =====
        #
        # 规则优先级：
        # 1. 只有一只手 → 简单手势
        # 2. 有稳定的简单手势（某只手连续N帧同手势）→ 简单手势
        #    （一只手在稳定做pointing，另一只手乱动 → 简单手势优先）
        # 3. LSTM 已确认 + 双手都在大幅运动 + 没有稳定简单手势 → LSTM
        # 4. 其他 → 简单手势（默认安全选项）

        both_moving = (left_motion >= WRIST_MOTION_THRESHOLD
                       and right_motion >= WRIST_MOTION_THRESHOLD)

        if num_hands <= 1:
            active_output = "SIMPLE"
        elif any_stable:
            active_output = "SIMPLE"
        elif lstm_ready and both_moving:
            active_output = "LSTM"
        else:
            active_output = "SIMPLE"

        # ===== HUD =====
        y_offset = 25

        # 当前输出来源
        if active_output == "SIMPLE":
            out_color = (200, 200, 0)
        elif active_output == "LSTM":
            out_color = (0, 180, 255)
        else:
            out_color = (128, 128, 128)
        cv2.putText(frame, f"Output: {active_output}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, out_color, 2)
        y_offset += 25

        cv2.putText(frame, f"Hands: {num_hands}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y_offset += 18

        # 左右手腕运动条
        full_scale = WRIST_MOTION_THRESHOLD * 5
        for val, label in [(left_motion, "L"), (right_motion, "R")]:
            bar_w = 120
            bar_fill = min(val / full_scale, 1.0)
            bar_color = (0, 200, 0) if val >= WRIST_MOTION_THRESHOLD else (80, 80, 200)
            cv2.rectangle(frame, (10, y_offset), (10 + bar_w, y_offset + 8), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, y_offset), (10 + int(bar_w * bar_fill), y_offset + 8), bar_color, -1)
            tx = 10 + int(bar_w * (WRIST_MOTION_THRESHOLD / full_scale))
            cv2.line(frame, (tx, y_offset - 1), (tx, y_offset + 9), (255, 255, 255), 1)
            cv2.putText(frame, f"{label}:{val:.3f}",
                        (10 + bar_w + 4, y_offset + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
            y_offset += 13

        y_offset += 4

        # --- 简单手势区 ---
        simple_section_color = (200, 200, 0) if active_output == "SIMPLE" else (100, 100, 100)
        cv2.putText(frame, "Simple:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, simple_section_color, 1)
        y_offset += 16

        for hand_type in ("Left", "Right"):
            if hand_type in simple_results:
                gesture, palm_facing = simple_results[hand_type]
                stable = stabilizer.is_stable(hand_type)
                tag = "F" if palm_facing else "B"
                stab_mark = " *" if stable else ""

                if active_output == "SIMPLE":
                    color = (0, 255, 0) if gesture != "unknown" else (128, 128, 128)
                    thickness = 2 if stable else 1
                else:
                    color = (100, 100, 100)
                    thickness = 1

                cv2.putText(frame, f"  {hand_type[0]}: {gesture} [{tag}]{stab_mark}",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness)
                y_offset += 18

                # 手腕旁标注（仅 SIMPLE 输出时大字显示）
                if active_output == "SIMPLE":
                    lms = left_lms if hand_type == "Left" else right_lms
                    if lms:
                        wrist = lms[0]
                        label_color = (0, 255, 0) if gesture != "unknown" else (128, 128, 128)
                        cv2.putText(
                            frame,
                            f"{gesture} [{tag}]",
                            (int(wrist.x * w) - 40, int(wrist.y * h) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2,
                        )

        y_offset += 4

        # --- LSTM 区 ---
        if has_lstm:
            lstm_section_color = (0, 180, 255) if active_output == "LSTM" else (100, 100, 100)
            status_text = f"LSTM: {lstm_status}" if lstm_status else "LSTM: --"
            cv2.putText(frame, status_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, lstm_section_color, 1)
            y_offset += 16

            # 缓冲区
            if both_hands:
                buf_pct = min(len(frame_buffer) / window_size, 1.0)
                bar_w = 160
                cv2.rectangle(frame, (10, y_offset), (10 + bar_w, y_offset + 8), (50, 50, 50), -1)
                b_color = (0, 200, 0) if buf_pct >= 1.0 else (100, 100, 100)
                cv2.rectangle(frame, (10, y_offset), (10 + int(bar_w * buf_pct), y_offset + 8), b_color, -1)
                cv2.putText(frame, f"{len(frame_buffer)}/{window_size}",
                            (10 + bar_w + 4, y_offset + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
                y_offset += 14

            if lstm_confirmed and active_output == "LSTM":
                color = COLORS[lstm_confirmed_idx % len(COLORS)]
                display = LABEL_DISPLAY.get(lstm_confirmed, lstm_confirmed)
                cv2.putText(frame, display, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                y_offset += 32
                cv2.putText(frame, f"{lstm_confirmed_conf:.0%}",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                y_offset += 22
            elif lstm_prediction and active_output == "LSTM":
                cv2.putText(frame, f"({lstm_prediction} {lstm_confidence:.0%})",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
                y_offset += 20

            if lstm_probs is not None and active_output == "LSTM":
                for label_idx in sorted(idx_to_label.keys()):
                    label_name = idx_to_label[label_idx]
                    prob_val = lstm_probs[label_idx].item()
                    bar_color = COLORS[label_idx % len(COLORS)]
                    bar_len = int(prob_val * 160)
                    cv2.rectangle(frame, (10, y_offset), (10 + bar_len, y_offset + 12), bar_color, -1)
                    cv2.putText(frame, label_name,
                                (180, y_offset + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)
                    y_offset += 15

        cv2.imshow("Gesture Demo - Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    main()
