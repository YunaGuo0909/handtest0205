# Gesture keyboard control: track hands → recognize gestures → simulate keys for game control.
# Run: python main.py

import cv2
import time

from config import CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, MOVE_HAND, ACTION_HAND
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from command_mapper import CommandMapper
from keyboard_controller import KeyboardController

MOVE_DISPLAY = {"d": ">> D", "a": "<< A", None: "STOP"}
MOVE_COLORS = {"d": (0, 255, 0), "a": (0, 0, 255), None: (128, 128, 128)}


def main():
    tracker = HandTracker()
    keyboard = KeyboardController()
    mapper = CommandMapper()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    print("=" * 50)
    print("Gesture keyboard control | Press Q to quit")
    print(f"Left({MOVE_HAND}): point right→D  point left→A  none→stop")
    print(f"Right({ACTION_HAND}): swipe up→W(jump)  point→Shift(sprint)")
    print(f"Both: fists to open→E  wrist clash→Q")
    print("=" * 50)

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

            left_data = None
            right_data = None

            if results.hand_landmarks:
                for lms, handed in zip(results.hand_landmarks, results.handedness):
                    hand_type = handed[0].category_name
                    gesture, conf, palm_facing = GestureRecognizer.recognize(
                        lms, hand_type
                    )

                    tracker.draw_hand(frame, lms, w, h)

                    tag = "F" if palm_facing else "B"
                    wrist = lms[0]
                    cv2.putText(
                        frame,
                        f"{hand_type}: {gesture} [{tag}]",
                        (int(wrist.x * w) - 50, int(wrist.y * h) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    )

                    data = (gesture, palm_facing, lms)
                    if hand_type == MOVE_HAND:
                        left_data = data
                    elif hand_type == ACTION_HAND:
                        right_data = data

            triggered = mapper.update(left_data, right_data, keyboard)

            if not results.hand_landmarks:
                mapper.stop(keyboard)

            for t in triggered:
                print(f"  >> {t}")

            # FPS
            fps_count += 1
            if time.time() - fps_time >= 1:
                fps = fps_count
                fps_count = 0
                fps_time = time.time()

            # --- HUD ---
            num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
            cv2.putText(frame, f"FPS:{fps} Hands:{num_hands}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            mk = mapper._current_move_key
            mv_text = MOVE_DISPLAY.get(mk, "STOP")
            mv_color = MOVE_COLORS.get(mk, (128, 128, 128))
            cv2.putText(frame, f"Move: {mv_text}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mv_color, 2)

            if mapper.last_action:
                cv2.putText(frame, f"Skill: {mapper.last_action}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            if mapper.last_combo:
                cv2.putText(frame, f"Combo: {mapper.last_combo}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            cv2.imshow("Hand Gesture Keyboard - Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        mapper.stop(keyboard)
        keyboard.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped")


if __name__ == "__main__":
    main()
