# YOLO GPU加速版 手部追踪 + 手势识别 + 游戏指令
# 功能: YOLO人体检测 + MediaPipe手部追踪 + 手势识别 + 指令映射 + UDP
# 运行: python main_yolo.py

import cv2
import time

from config import (
    UDP_IP, UDP_PORT, CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT,
    USE_GPU, PRIMARY_HAND,
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
    import torch

    use_gpu = torch.cuda.is_available() and USE_GPU
    device = "cuda:0" if use_gpu else "cpu"
    print(f"GPU: {'有 - ' + torch.cuda.get_device_name(0) if use_gpu else '无'}")

    from ultralytics import YOLO

    print("加载 YOLO...")
    yolo = YOLO("yolov8n.pt")
    yolo.to(device)
    print(f"YOLO 已加载到 {device}")

    tracker = HandTracker()
    sender = UDPSender(UDP_IP, UDP_PORT)
    mapper = CommandMapper()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("打不开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    print(f"已启动(GPU) | UDP: {UDP_IP}:{UDP_PORT} | 主控手: {PRIMARY_HAND} | 按Q退出")
    print(f"指令映射: 握拳正面→前进  握拳背面→后退  张开正面→停止")

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

            # YOLO 检测人体
            yolo_results = yolo(frame, verbose=False, classes=[0])
            for r in yolo_results:
                for box in r.boxes:
                    if box.conf[0] > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # MediaPipe 检测手部
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
                        (int(wrist.x * w) - 50, int(wrist.y * h) + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
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

            command = mapper.update(primary_gesture, primary_palm_facing)
            sender.send(hands, command)

            # FPS
            fps_count += 1
            if time.time() - fps_time >= 1:
                fps = fps_count
                fps_count = 0
                fps_time = time.time()

            # 顶部状态栏
            cv2.putText(
                frame,
                f"FPS:{fps} Hands:{len(hands)} [GPU]",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # 指令显示
            cmd_text = COMMAND_DISPLAY.get(command, "")
            cmd_color = COMMAND_COLORS.get(command, (128, 128, 128))
            if cmd_text:
                text_size = cv2.getTextSize(cmd_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(frame, cmd_text, (text_x, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cmd_color, 3)

            cv2.imshow("Hand Tracking GPU - Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sender.close()
        print("已停止")


if __name__ == "__main__":
    main()
