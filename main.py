# 手部追踪 + 手势识别 + 游戏指令
# 摄像头实时追踪 → 识别手势和朝向 → 映射游戏指令 → OSC 发送到 UE
# 运行: python main.py

import cv2
import time

from config import OSC_IP, OSC_PORT, CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, PRIMARY_HAND
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from command_mapper import CommandMapper
from osc_utils import OSCSender

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
    sender = OSCSender(OSC_IP, OSC_PORT)
    mapper = CommandMapper()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("打不开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    print(f"已启动 | OSC: {OSC_IP}:{OSC_PORT} | 主控手: {PRIMARY_HAND} | 按Q退出")
    print(f"指令映射: 握拳正面→前进  握拳背面→后退  张开正面→停止")
    print(f"OSC 发送: /hand/move (float: 1.0=前进, -1.0=后退, 0.0=停止)")

    fps_count = 0
    fps_time = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            results = tracker.detect(frame)

            primary_gesture = "unknown"
            primary_palm_facing = False

            if results.hand_landmarks:
                for idx, (lms, handed) in enumerate(
                    zip(results.hand_landmarks, results.handedness)
                ):
                    hand_type = handed[0].category_name

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

            # 指令映射（防抖）
            command = mapper.update(primary_gesture, primary_palm_facing)
            raw_cmd = mapper.raw_command(primary_gesture, primary_palm_facing)

            # OSC 发送
            move_value = CommandMapper.COMMAND_MOVE_VALUES.get(command, 0.0)
            sender.send(move_value=move_value)

            # FPS
            fps_count += 1
            if time.time() - fps_time >= 1:
                fps = fps_count
                fps_count = 0
                fps_time = time.time()

            # --- 画面信息 ---

            num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
            cv2.putText(frame, f"FPS:{fps} Hands:{num_hands}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cmd_text = COMMAND_DISPLAY.get(command, "")
            cmd_color = COMMAND_COLORS.get(command, (128, 128, 128))
            if cmd_text:
                text_size = cv2.getTextSize(cmd_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(frame, cmd_text, (text_x, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, cmd_color, 3)

            if raw_cmd != "none":
                cv2.putText(frame, f"raw: {raw_cmd}", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow("Hand Tracking - Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sender.close()
        print("已停止")


if __name__ == "__main__":
    main()
