# 手势识别 + 键盘/鼠标控制

基于 MediaPipe 的手部关键点检测，支持**简单手势**（规则）与**复杂手势**（LSTM），映射到键盘与鼠标，可用于游戏或无障碍控制。

## 环境

- Python 3.8+
- 摄像头（本机或外接）

## 安装

```bash
pip install -r requirements.txt
```

首次运行时会自动下载 MediaPipe 手部模型（约 10MB），无需手动放置。

## 直接运行（克隆即用）

```bash
python demo_gesture.py
```

- **无 `gesture_model.pt`**：仅启用简单手势（移动、指向、握拳、捏合鼠标等），程序会提示“仅使用简单手势识别”。
- **有 `gesture_model.pt`**：同时启用复杂手势（手保健操等），对应键位见下方。

退出：按 **Q**。

## 键位说明（当前配置）

| 类型 | 手势 | 键位/效果 |
|------|------|-----------|
| 左手 | 食指指右/指左 | D / A |
| 左手 | 握拳（正面/背面） | W / S |
| 左手 | pre_pinch / pinch | 鼠标移动 / 左键点击 |
| 右手 | 张开掌心向上 + 四指向上挥 | Space |
| 复杂(1) | leftright | T |
| 复杂(5) | 抓手指 | Space |
| 复杂(6) | 掌根互拍 | E |
| 复杂(7) | 虎口互击 | Q |
| 复杂(2,3,4) | 其余手操 | 2 / 3 / 4 |

## 可选：训练复杂手势

若要使用复杂手势识别，需先采集数据并训练 LSTM（或使用已有的 `gesture_model.pt`）。

1. **准备视频**：按动作类别分子文件夹，例如：
   ```
   training_videos/
   ├── 1/          # leftright
   │   ├── p1_001.mp4
   │   └── ...
   ├── 2/          # 手背互拍
   ├── ...
   └── 7/          # 虎口互击
   ```

2. **从视频生成数据**（会覆盖已有 `gesture_data.csv`）：
   ```bash
   python collect_data.py --videos training_videos/ --overwrite
   ```

3. **训练模型**：
   ```bash
   python train_gesture.py
   ```
   完成后会生成 `gesture_model.pt`，再运行 `demo_gesture.py` 即可使用复杂手势。

## 项目结构（上传到 Git 的部分）

- `demo_gesture.py` — 主程序（识别 + 键盘/鼠标）
- `hand_tracker.py` — 手部追踪（MediaPipe）
- `gesture_recognizer.py` — 简单手势规则
- `train_gesture.py` — 复杂手势 LSTM 训练
- `collect_data.py` — 从视频采集关键点数据
- `keyboard_controller.py` / `config.py` — 键盘与配置
- `requirements.txt` / `README.md`

以下**不会**提交（已在 `.gitignore` 中）：训练视频、`gesture_data.csv`、`gesture_model.pt`、手部模型文件、`__pycache__` 等。
