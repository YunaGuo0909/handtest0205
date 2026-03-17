# Gesture data collection (hand exercises + generic gestures)
# Modes:
#   1) Batch from videos: put videos in folders by action name, extract landmarks
#   2) Realtime camera: do gestures in front of camera, press number keys to label and record
#
# Usage:
#   python collect_data.py --videos training_videos/
#   python collect_data.py --videos training_videos/ --no-flip   # no mirror for back-camera videos
#   python collect_data.py --camera
#   python collect_data.py --videos training_videos/ --preview   # preview detection when batching

import argparse
import csv
import os
import sys
import time

import cv2

from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from config import CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT

# ========== Config ==========

OUTPUT_CSV = "gesture_data.csv"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# Key -> label for camera realtime mode (hand exercises)
KEY_LABEL_MAP = {
    ord("1"): "leftright",
    ord("2"): "手背互拍",
    ord("3"): "手腕上下弯曲伸展",
    ord("4"): "手腕左右侧弯",
    ord("5"): "抓手指",
    ord("6"): "掌根互拍",
    ord("7"): "虎口互击",
    ord("0"): "idle",
}

# Both hands: merge left+right landmarks per frame into one row
# Row: left 63-d + right 63-d = 126-d (missing hand filled with 0)
CSV_HEADER = (
    ["label", "source", "frame_idx"]
    + [f"L_x{i}" for i in range(21)]
    + [f"L_y{i}" for i in range(21)]
    + [f"L_z{i}" for i in range(21)]
    + [f"R_x{i}" for i in range(21)]
    + [f"R_y{i}" for i in range(21)]
    + [f"R_z{i}" for i in range(21)]
)


def _extract_hand_coords(landmarks):
    """Extract 63-d coords for one hand (x0..x20, y0..y20, z0..z20)."""
    xs = [round(lm.x, 6) for lm in landmarks]
    ys = [round(lm.y, 6) for lm in landmarks]
    zs = [round(lm.z, 6) for lm in landmarks]
    return xs + ys + zs


ZERO_HAND = [0.0] * 63


def frame_to_row(label, source, frame_idx, left_lms, right_lms):
    """Merge one frame's both-hand data into one CSV row (left 63-d + right 63-d)."""
    left_coords = _extract_hand_coords(left_lms) if left_lms else ZERO_HAND
    right_coords = _extract_hand_coords(right_lms) if right_lms else ZERO_HAND
    return [label, source, frame_idx] + left_coords + right_coords


def _parse_hands(results):
    """Split MediaPipe results into left/right hand landmarks."""
    left_lms = None
    right_lms = None
    if results.hand_landmarks:
        for lms, handed in zip(results.hand_landmarks, results.handedness):
            hand_type = handed[0].category_name
            if hand_type == "Left":
                left_lms = lms
            elif hand_type == "Right":
                right_lms = lms
    return left_lms, right_lms


def process_video(tracker, video_path, label, writer, stats, flip=True, preview=False):
    """Process one video file, extract both-hand landmarks per frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [Skip] Cannot open: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    source_name = os.path.basename(video_path)
    frame_idx = 0
    saved = 0
    no_hand_frames = 0

    print(f"  Processing: {source_name}  ({total_frames} frames, {fps:.0f} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if flip:
            frame = cv2.flip(frame, 1)

        results = tracker.detect(frame)
        left_lms, right_lms = _parse_hands(results)

        if left_lms or right_lms:
            row = frame_to_row(label, source_name, frame_idx, left_lms, right_lms)
            writer.writerow(row)
            saved += 1
        else:
            no_hand_frames += 1

        if preview:
            h, w = frame.shape[:2]
            if left_lms:
                tracker.draw_hand(frame, left_lms, w, h)
            if right_lms:
                tracker.draw_hand(frame, right_lms, w, h)
            hands_n = (1 if left_lms else 0) + (1 if right_lms else 0)
            cv2.putText(frame, f"[{label}] frame:{frame_idx} hands:{hands_n}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Preview - Q to skip", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"    Progress: {frame_idx}/{total_frames} ({pct:.0f}%)")

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    stats[label] = stats.get(label, 0) + saved
    detect_rate = saved / max(frame_idx, 1) * 100
    print(f"    Done: {saved}/{frame_idx} frames with hands ({detect_rate:.0f}%), missed {no_hand_frames} frames")


def batch_from_videos(tracker, videos_dir, writer, stats, flip=True, preview=False):
    """Process all videos in folder; subfolder name = gesture label.

    Folder layout:
        training_videos/
        ├── 1/   or  leftright/
        │   ├── p1_001.mp4
        │   └── ...
        └── ...
    """
    if not os.path.isdir(videos_dir):
        print(f"Error: directory not found → {videos_dir}")
        sys.exit(1)

    gesture_dirs = sorted(
        d for d in os.listdir(videos_dir)
        if os.path.isdir(os.path.join(videos_dir, d))
    )

    if not gesture_dirs:
        print(f"Error: no subfolders under {videos_dir}")
        print("Organize videos like:")
        print("  training_videos/")
        print("  ├── 1/  (or leftright/)")
        print("  │   ├── p1_001.mp4")
        print("  │   └── ...")
        print("  └── ...")
        sys.exit(1)

    print(f"\nFound {len(gesture_dirs)} gesture(s): {', '.join(gesture_dirs)}")
    print("=" * 50)

    for gesture_name in gesture_dirs:
        gesture_path = os.path.join(videos_dir, gesture_name)
        videos = sorted(
            f for f in os.listdir(gesture_path)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        )

        if not videos:
            print(f"\n[{gesture_name}] No video files, skip")
            continue

        print(f"\n[{gesture_name}] {len(videos)} video(s)")

        for vfile in videos:
            vpath = os.path.join(gesture_path, vfile)
            process_video(tracker, vpath, gesture_name, writer, stats,
                          flip=flip, preview=preview)


def realtime_from_camera(tracker, writer, stats):
    """Realtime camera recording: press number keys to label, same key again to stop."""
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    current_label = None
    frame_idx = 0
    session_saved = 0

    print("\n" + "=" * 50)
    print("Realtime camera recording")
    print("Press number key to select label, same key again to stop")
    print("-" * 50)
    for key, label in sorted(KEY_LABEL_MAP.items()):
        print(f"  [{chr(key)}] {label}")
    print("  [Q] Quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        results = tracker.detect(frame)
        left_lms, right_lms = _parse_hands(results)

        if left_lms:
            tracker.draw_hand(frame, left_lms, w, h)
        if right_lms:
            tracker.draw_hand(frame, right_lms, w, h)

        if current_label and (left_lms or right_lms):
            row = frame_to_row(current_label, "camera", frame_idx,
                               left_lms, right_lms)
            writer.writerow(row)
            session_saved += 1
            stats[current_label] = stats.get(current_label, 0) + 1

        # HUD
        status_color = (0, 0, 255) if current_label else (128, 128, 128)
        status_text = f"REC: {current_label} ({session_saved})" if current_label else "IDLE - press 1~7, 0=idle"
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        hands_n = (1 if left_lms else 0) + (1 if right_lms else 0)
        cv2.putText(frame, f"Hands: {hands_n}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Data Collection - Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in KEY_LABEL_MAP:
            new_label = KEY_LABEL_MAP[key]
            if current_label == new_label:
                print(f"  Stop recording [{current_label}] (this session {session_saved} rows)")
                current_label = None
                session_saved = 0
            else:
                if current_label:
                    print(f"  Switch: [{current_label}] → [{new_label}]")
                else:
                    print(f"  Start recording [{new_label}]")
                current_label = new_label
                session_saved = 0

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Gesture data collection (hand exercises + generic)")
    parser.add_argument("--videos", type=str, help="Video folder (subfolders = gesture labels)")
    parser.add_argument("--camera", action="store_true", help="Realtime camera recording")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV, help=f"Output CSV (default: {OUTPUT_CSV})")
    parser.add_argument("--no-flip", action="store_true",
                        help="Do not mirror video (for back-camera recordings)")
    parser.add_argument("--preview", action="store_true",
                        help="Show preview window when batching")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing CSV and rebuild from all videos in folder")
    args = parser.parse_args()

    if not args.videos and not args.camera:
        parser.print_help()
        print("\nExamples:")
        print("  python collect_data.py --videos training_videos/")
        print("  python collect_data.py --videos training_videos/ --overwrite")
        print("  python collect_data.py --videos training_videos/ --no-flip")
        print("  python collect_data.py --videos training_videos/ --preview")
        print("  python collect_data.py --camera")
        print("  python collect_data.py --videos training_videos/ --camera")
        sys.exit(0)

    tracker = HandTracker()

    file_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    overwrite = getattr(args, "overwrite", False) and args.videos
    if overwrite and file_exists:
        csvfile = open(args.output, "w", newline="", encoding="utf-8")
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADER)
        print(f"Overwriting and rebuilding: {args.output} (all videos in folder)")
    else:
        csvfile = open(args.output, "a", newline="", encoding="utf-8")
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(CSV_HEADER)
            print(f"Creating new data file: {args.output}")
        else:
            print(f"Appending to existing file: {args.output}")

    stats = {}
    do_flip = not args.no_flip

    try:
        if args.videos:
            print("\n===== Batch from videos =====")
            print(f"Mirror flip: {'yes' if do_flip else 'no'}")
            batch_from_videos(tracker, args.videos, writer, stats,
                              flip=do_flip, preview=args.preview)

        if args.camera:
            realtime_from_camera(tracker, writer, stats)
    finally:
        csvfile.close()

    # Summary
    print("\n" + "=" * 50)
    print("Collection done. Stats:")
    print("-" * 50)
    total = 0
    for label, count in sorted(stats.items()):
        print(f"  {label:20s} → {count:6d} rows")
        total += count
    print("-" * 50)
    print(f"  {'Total':20s} → {total:6d} rows")
    print(f"  Saved to: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
