# 手势数据采集工具（手保健操 + 通用手势）
# 两种模式：
#   1) 视频批量模式：把不同人录的视频按动作名放进文件夹，批量提取关键点
#   2) 摄像头实时模式：对着摄像头做手势，按数字键标记类别实时录制
#
# 用法:
#   python collect_data.py --videos training_videos/                 # 视频批量
#   python collect_data.py --videos training_videos/ --no-flip       # 后置摄像头录的视频不镜像
#   python collect_data.py --camera                                  # 摄像头实时
#   python collect_data.py --videos training_videos/ --camera        # 先处理视频再实时补录
#   python collect_data.py --videos training_videos/ --preview       # 批量时弹窗预览检测效果

import argparse
import csv
import os
import sys
import time

import cv2

from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from config import CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT

# ========== 配置 ==========

OUTPUT_CSV = "gesture_data.csv"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# 摄像头实时模式下的按键 → 标签映射（手保健操动作）
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

# 双手动态手势需要把同一帧两只手的数据合并到一行
# 每行存: 左手 63维 + 右手 63维 = 126维（检测不到的手填 0）
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
    """提取单只手的 63 维坐标 (x0..x20, y0..y20, z0..z20)"""
    xs = [round(lm.x, 6) for lm in landmarks]
    ys = [round(lm.y, 6) for lm in landmarks]
    zs = [round(lm.z, 6) for lm in landmarks]
    return xs + ys + zs


ZERO_HAND = [0.0] * 63


def frame_to_row(label, source, frame_idx, left_lms, right_lms):
    """将一帧的双手数据合并成一行 CSV (左手63维 + 右手63维)"""
    left_coords = _extract_hand_coords(left_lms) if left_lms else ZERO_HAND
    right_coords = _extract_hand_coords(right_lms) if right_lms else ZERO_HAND
    return [label, source, frame_idx] + left_coords + right_coords


def _parse_hands(results):
    """从 MediaPipe 结果中分离左右手关键点"""
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
    """处理单个视频文件，提取每帧的双手关键点"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [跳过] 无法打开: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    source_name = os.path.basename(video_path)
    frame_idx = 0
    saved = 0
    no_hand_frames = 0

    print(f"  处理: {source_name}  ({total_frames}帧, {fps:.0f}fps)")

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
            print(f"    进度: {frame_idx}/{total_frames} ({pct:.0f}%)")

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    stats[label] = stats.get(label, 0) + saved
    detect_rate = saved / max(frame_idx, 1) * 100
    print(f"    完成: {saved}/{frame_idx} 帧检测到手 ({detect_rate:.0f}%), 丢失 {no_hand_frames} 帧")


def batch_from_videos(tracker, videos_dir, writer, stats, flip=True, preview=False):
    """批量处理视频文件夹

    文件夹结构:
        training_videos/
        ├── leftright/
        │   ├── person1.mp4
        │   └── person2.mp4
        ├── 手背互拍/
        │   └── person1.mp4
        └── 虎口互击/
            └── ...
    """
    if not os.path.isdir(videos_dir):
        print(f"错误: 目录不存在 → {videos_dir}")
        sys.exit(1)

    gesture_dirs = sorted(
        d for d in os.listdir(videos_dir)
        if os.path.isdir(os.path.join(videos_dir, d))
    )

    if not gesture_dirs:
        print(f"错误: {videos_dir} 下没有子文件夹")
        print("请按以下结构组织视频：")
        print("  training_videos/")
        print("  ├── leftright/")
        print("  │   ├── person1.mp4")
        print("  │   └── person2.mp4")
        print("  ├── 手背互拍/")
        print("  │   └── person1.mp4")
        print("  └── ...")
        sys.exit(1)

    print(f"\n发现 {len(gesture_dirs)} 种动作: {', '.join(gesture_dirs)}")
    print("=" * 50)

    for gesture_name in gesture_dirs:
        gesture_path = os.path.join(videos_dir, gesture_name)
        videos = sorted(
            f for f in os.listdir(gesture_path)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        )

        if not videos:
            print(f"\n[{gesture_name}] 没有视频文件，跳过")
            continue

        print(f"\n[{gesture_name}] {len(videos)} 个视频")

        for vfile in videos:
            vpath = os.path.join(gesture_path, vfile)
            process_video(tracker, vpath, gesture_name, writer, stats,
                          flip=flip, preview=preview)


def realtime_from_camera(tracker, writer, stats):
    """摄像头实时录制模式"""
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("错误: 打不开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    current_label = None
    frame_idx = 0
    session_saved = 0

    print("\n" + "=" * 50)
    print("摄像头实时录制模式")
    print("按数字键选择动作标签，再次按同一键停止录制")
    print("-" * 50)
    for key, label in sorted(KEY_LABEL_MAP.items()):
        print(f"  [{chr(key)}] {label}")
    print("  [Q] 退出")
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
                print(f"  停止录制 [{current_label}] (本次 {session_saved} 条)")
                current_label = None
                session_saved = 0
            else:
                if current_label:
                    print(f"  切换: [{current_label}] → [{new_label}]")
                else:
                    print(f"  开始录制 [{new_label}]")
                current_label = new_label
                session_saved = 0

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="手势数据采集工具（手保健操 + 通用手势）")
    parser.add_argument("--videos", type=str, help="视频文件夹路径（按动作名分子文件夹）")
    parser.add_argument("--camera", action="store_true", help="启用摄像头实时录制模式")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV, help=f"输出CSV路径 (默认: {OUTPUT_CSV})")
    parser.add_argument("--no-flip", action="store_true",
                        help="视频不做水平镜像（后置摄像头录的视频用这个）")
    parser.add_argument("--preview", action="store_true",
                        help="批量处理时弹窗预览检测效果")
    parser.add_argument("--overwrite", action="store_true",
                        help="视频模式下覆盖已有 CSV，用当前文件夹内全部视频重新生成（新增视频后重训时用）")
    args = parser.parse_args()

    if not args.videos and not args.camera:
        parser.print_help()
        print("\n示例:")
        print("  python collect_data.py --videos training_videos/")
        print("  python collect_data.py --videos training_videos/ --overwrite   # 新增视频后重训：覆盖 CSV 再采集")
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
        print(f"覆盖并重新生成: {args.output}（当前 training_videos 下全部视频）")
    else:
        csvfile = open(args.output, "a", newline="", encoding="utf-8")
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(CSV_HEADER)
            print(f"创建新数据文件: {args.output}")
        else:
            print(f"追加到已有数据文件: {args.output}")

    stats = {}
    do_flip = not args.no_flip

    try:
        if args.videos:
            print("\n===== 视频批量采集 =====")
            print(f"镜像翻转: {'是' if do_flip else '否'}")
            batch_from_videos(tracker, args.videos, writer, stats,
                              flip=do_flip, preview=args.preview)

        if args.camera:
            realtime_from_camera(tracker, writer, stats)
    finally:
        csvfile.close()

    # 统计汇总
    print("\n" + "=" * 50)
    print("采集完成！数据统计：")
    print("-" * 50)
    total = 0
    for label, count in sorted(stats.items()):
        print(f"  {label:20s} → {count:6d} 条")
        total += count
    print("-" * 50)
    print(f"  {'合计':20s} → {total:6d} 条")
    print(f"  保存至: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
