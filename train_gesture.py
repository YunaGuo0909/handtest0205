# 手势训练脚本
# 从 gesture_data.csv 读取关键点序列，训练 LSTM 分类模型
#
# 用法:
#   python train_gesture.py                        # 默认参数训练
#   python train_gesture.py --epochs 100           # 自定义轮数
#   python train_gesture.py --window 20 --stride 5 # 自定义窗口

import argparse
import csv
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ========== 默认超参数 ==========

DEFAULTS = {
    "csv": "gesture_data.csv",
    "window": 30,       # 滑动窗口帧数（30帧≈1秒@30fps）
    "stride": 5,        # 滑动步长
    "batch_size": 32,
    "epochs": 80,
    "lr": 0.001,
    "hidden": 128,      # LSTM 隐藏层维度
    "layers": 2,        # LSTM 层数
    "dropout": 0.3,
    "test_ratio": 0.2,  # 测试集比例
    "model_out": "gesture_model.pt",
}

INPUT_DIM = 126  # 左手63 + 右手63


# ========== 数据集 ==========

class GestureDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


def load_csv(csv_path):
    """加载 CSV，按 (source, label) 分组保持时序"""
    sequences = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            label = row[0]
            source = row[1]
            coords = [float(x) for x in row[3:]]  # 跳过 label, source, frame_idx
            sequences[(source, label)].append(coords)
    return sequences


def normalize_coords(frames):
    """对一个窗口的坐标做归一化：以左手腕为原点，按手掌尺度缩放"""
    frames = np.array(frames, dtype=np.float32)
    for i in range(len(frames)):
        frame = frames[i]
        left_x = frame[0:21]
        left_y = frame[21:42]
        right_x = frame[63:84]
        right_y = frame[84:105]

        # 找一个有效的手作为参考点
        lw_x, lw_y = left_x[0], left_y[0]
        rw_x, rw_y = right_x[0], right_y[0]

        has_left = any(v != 0 for v in left_x)
        has_right = any(v != 0 for v in right_x)

        if has_left:
            ref_x, ref_y = lw_x, lw_y
        elif has_right:
            ref_x, ref_y = rw_x, rw_y
        else:
            continue

        # 平移：以参考手腕为原点
        if has_left:
            frame[0:21] = left_x - ref_x
            frame[21:42] = left_y - ref_y
        if has_right:
            frame[63:84] = right_x - ref_x
            frame[84:105] = right_y - ref_y

        # 缩放：用左手中指根到手腕距离（如果有左手）
        if has_left:
            scale = np.sqrt((left_x[9] - lw_x)**2 + (left_y[9] - lw_y)**2)
        elif has_right:
            scale = np.sqrt((right_x[9] - rw_x)**2 + (right_y[9] - rw_y)**2)
        else:
            scale = 1.0

        if scale > 1e-6:
            frame[0:63] /= scale
            frame[63:126] /= scale

        frames[i] = frame

    return frames


def build_windows(sequences, window_size, stride, augment=True):
    """从按视频分组的序列中构建滑动窗口样本"""
    windows = []
    labels = []
    label_set = sorted(set(label for _, label in sequences.keys()))
    label_to_idx = {name: i for i, name in enumerate(label_set)}

    print(f"\n标签映射: {label_to_idx}")

    for (source, label), frames in sequences.items():
        if len(frames) < window_size:
            continue

        label_idx = label_to_idx[label]

        for start in range(0, len(frames) - window_size + 1, stride):
            window = frames[start:start + window_size]
            normed = normalize_coords(window)
            windows.append(normed)
            labels.append(label_idx)

            if augment:
                # 数据增强1：加随机噪声
                noisy = normed + np.random.normal(0, 0.005, normed.shape).astype(np.float32)
                windows.append(noisy)
                labels.append(label_idx)

                # 数据增强2：随机缩放
                scale = np.random.uniform(0.9, 1.1)
                scaled = normed * scale
                windows.append(scaled)
                labels.append(label_idx)

    return np.array(windows), np.array(labels), label_to_idx


def split_by_source(sequences, test_ratio):
    """按视频源分割训练/测试集，确保同一视频的数据不会同时出现在两个集合"""
    sources_by_label = defaultdict(list)
    for (source, label) in sequences.keys():
        sources_by_label[label].append(source)

    train_seqs = {}
    test_seqs = {}

    for label, sources in sources_by_label.items():
        sources = list(set(sources))
        random.shuffle(sources)
        n_test = max(1, int(len(sources) * test_ratio))
        test_sources = set(sources[:n_test])

        for (src, lbl), frames in sequences.items():
            if lbl != label:
                continue
            if src in test_sources:
                test_seqs[(src, lbl)] = frames
            else:
                train_seqs[(src, lbl)] = frames

    return train_seqs, test_seqs


# ========== 模型 ==========

class GestureLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # 取最后一层的隐藏状态
        out = self.dropout(last_hidden)
        return self.fc(out)


# ========== 训练 ==========

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for windows, labels in loader:
        windows, labels = windows.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(windows)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for windows, labels in loader:
            windows, labels = windows.to(device), labels.to(device)
            outputs = model(windows)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def print_confusion_matrix(preds, labels, label_names):
    """打印混淆矩阵"""
    n = len(label_names)
    matrix = [[0] * n for _ in range(n)]
    for true, pred in zip(labels, preds):
        matrix[true][pred] += 1

    print(f"\n{'':>12s}", end="")
    for name in label_names:
        print(f"{name:>8s}", end="")
    print("   ← predicted")

    for i, name in enumerate(label_names):
        print(f"{name:>12s}", end="")
        for j in range(n):
            val = matrix[i][j]
            if val > 0:
                print(f"{val:>8d}", end="")
            else:
                print(f"{'·':>8s}", end="")
        row_total = sum(matrix[i])
        row_correct = matrix[i][i]
        acc = row_correct / max(row_total, 1) * 100
        print(f"   {acc:.0f}%")
    print("↑ actual")


def main():
    parser = argparse.ArgumentParser(description="手势 LSTM 训练")
    parser.add_argument("--csv", default=DEFAULTS["csv"])
    parser.add_argument("--window", type=int, default=DEFAULTS["window"])
    parser.add_argument("--stride", type=int, default=DEFAULTS["stride"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--hidden", type=int, default=DEFAULTS["hidden"])
    parser.add_argument("--layers", type=int, default=DEFAULTS["layers"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULTS["test_ratio"])
    parser.add_argument("--model-out", default=DEFAULTS["model_out"])
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"错误: {args.csv} 不存在，请先运行 collect_data.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 1. 加载数据
    print(f"\n加载数据: {args.csv}")
    sequences = load_csv(args.csv)
    print(f"视频片段数: {len(sequences)}")

    total_frames = sum(len(f) for f in sequences.values())
    print(f"总帧数: {total_frames}")

    # 2. 按视频源分割
    train_seqs, test_seqs = split_by_source(sequences, args.test_ratio)
    print(f"训练集视频: {len(train_seqs)}, 测试集视频: {len(test_seqs)}")

    # 3. 构建滑动窗口
    print(f"窗口大小: {args.window}帧, 步长: {args.stride}帧")
    train_windows, train_labels, label_map = build_windows(
        train_seqs, args.window, args.stride, augment=True
    )
    test_windows, test_labels, _ = build_windows(
        test_seqs, args.window, args.stride, augment=False
    )

    num_classes = len(label_map)
    label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

    print(f"训练样本: {len(train_labels)}, 测试样本: {len(test_labels)}")
    print(f"类别数: {num_classes}, 标签: {label_names}")

    # 类别分布
    print("\n训练集类别分布:")
    for name in label_names:
        idx = label_map[name]
        count = (train_labels == idx).sum()
        print(f"  {name}: {count}")

    # 4. DataLoader
    train_ds = GestureDataset(train_windows, train_labels)
    test_ds = GestureDataset(test_windows, test_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # 5. 模型
    model = GestureLSTM(
        INPUT_DIM, args.hidden, args.layers, num_classes, args.dropout
    ).to(device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # 6. 训练
    print(f"\n{'='*60}")
    print(f"开始训练 ({args.epochs} epochs)")
    print(f"{'='*60}")

    best_acc = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "label_map": label_map,
                "window_size": args.window,
                "input_dim": INPUT_DIM,
                "hidden_dim": args.hidden,
                "num_layers": args.layers,
                "num_classes": num_classes,
            }, args.model_out)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1%} | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.1%} | "
                  f"LR: {lr_now:.6f}"
                  f"{' ★' if epoch == best_epoch else ''}")

    # 7. 最终评估
    print(f"\n{'='*60}")
    print(f"训练完成! 最佳测试准确率: {best_acc:.1%} (Epoch {best_epoch})")
    print(f"模型保存至: {args.model_out}")
    print(f"{'='*60}")

    checkpoint = torch.load(args.model_out, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    _, final_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"\n最终测试准确率: {final_acc:.1%}")
    print_confusion_matrix(preds, labels, label_names)


if __name__ == "__main__":
    main()
