# 标准版 手部追踪 + 手势识别 + 游戏指令
# 功能: 摄像头实时追踪 → 识别手势和朝向 → 映射游戏指令 → UDP发送到UE
# 运行: python main.py

import cv2
import json
import time

from config import (
    UDP_IP, UDP_PORT, CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT,
    SAVE_TO_FILE, OUTPUT_FILE, PRIMARY_HAND,
)
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from command_mapper import CommandMapper
from udp_utils import UDPSender

COMMAND_DISPLAY = {
    "forward": ">> FORWARD",
    "backward": "<< BACKWARD",
    "stop": "|| STOP",
    "none": "",
}

COMMAND_COLORS = {
    "forward": (0, 255, 0),
    "backward": (0, 0, 255),
    "stop": (0, 255, 255),
    "none": (128, 128, 128),
}


def main():
    tracker = HandTracker()
    sender = UDPSender(UDP_IP, UDP_PORT)
    mapper = CommandMapper()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("打不开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    print(f"已启动 | UDP: {UDP_IP}:{UDP_PORT} | 主控手: {PRIMARY_HAND} | 按Q退出")
    print(f"指令映射: 握拳正面→前进  握拳背面→后退  张开正面→停止")
    if SAVE_TO_FILE:
        print(f"数据将保存到: {OUTPUT_FILE}")

    fps_count = 0
    fps_time = time.time()
    fps = 0
    recorded_frames = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            results = tracker.detect(frame)

            hands = []
            primary_gesture = "unknown"
            primary_palm_facing = False

            if results.hand_landmarks:
                for idx, (lms, handed) in enumerate(
                    zip(results.hand_landmarks, results.handedness)
                ):
                    hand_type = handed[0].category_name
                    conf = handed[0].score

                    gesture, gesture_conf, palm_facing = GestureRecognizer.recognize(
                        lms, hand_type
                    )

                    tracker.draw_hand(frame, lms, w, h)

                    facing_tag = "F" if palm_facing else "B"
                    wrist = lms[0]
                    cv2.putText(
                        frame,
                        f"{hand_type}: {gesture} [{facing_tag}]",
                        (int(wrist.x * w) - 50, int(wrist.y * h) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                    if hand_type == PRIMARY_HAND:
                        primary_gesture = gesture
                        primary_palm_facing = palm_facing

                    hands.append({
                        "hand_index": idx,
                        "hand_type": hand_type,
                        "confidence": round(conf, 4),
                        "gesture": gesture,
                        "gesture_confidence": gesture_conf,
                        "palm_facing": palm_facing,
                        "landmarks": tracker.format_landmarks(lms),
                    })

            # 指令映射（防抖）
            command = mapper.update(primary_gesture, primary_palm_facing)
            raw_cmd = mapper.raw_command(primary_gesture, primary_palm_facing)

            # UDP 发送
            data = sender.send(hands, command)

            if SAVE_TO_FILE:
                recorded_frames.append(data)

            # FPS
            fps_count += 1
            if time.time() - fps_time >= 1:
                fps = fps_count
                fps_count = 0
                fps_time = time.time()

            # --- 画面信息 ---

            # 顶部状态栏
            info = f"FPS:{fps} Hands:{len(hands)}"
            if SAVE_TO_FILE:
                info += f" REC:{len(recorded_frames)}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 指令显示（大字居中）
            cmd_text = COMMAND_DISPLAY.get(command, "")
            cmd_color = COMMAND_COLORS.get(command, (128, 128, 128))
            if cmd_text:
                text_size = cv2.getTextSize(cmd_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(frame, cmd_text, (text_x, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cmd_color, 3)

            # 原始指令（防抖前）小字显示
            if raw_cmd != "none":
                cv2.putText(frame, f"raw: {raw_cmd}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow("Hand Tracking - Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sender.close()

        if SAVE_TO_FILE and recorded_frames:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(
                    {"total_frames": len(recorded_frames), "frames": recorded_frames},
                    f,
                    indent=2,
                )
            print(f"已保存 {len(recorded_frames)} 帧数据到 {OUTPUT_FILE}")

        print("已停止")


if __name__ == "__main__":
    main()
