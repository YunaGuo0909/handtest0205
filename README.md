# Gesture Recognition + Keyboard/Mouse Control

Hand landmark detection via MediaPipe with **simple gestures** (rule-based) and **complex gestures** (LSTM). Maps to keyboard and mouse for games or accessibility.

## Requirements

- Python 3.8+
- Webcam (built-in or external)

## Install

```bash
pip install -r requirements.txt
```

On first run, the MediaPipe hand model (~10MB) is downloaded automatically.

## Run (clone and use)

```bash
python demo_gesture.py
```

- **Without `gesture_model.pt`**: Only simple gestures (move, point, fist, pinch mouse, etc.). The program will report "simple gestures only".
- **With `gesture_model.pt`**: Complex gestures (hand exercises) are also enabled. Key bindings are listed below.

Press **Q** to quit.

## Key bindings (current)

| Hand / type | Gesture | Key / effect |
|-------------|---------|--------------|
| Left | Point right / point left | D / A |
| Left | Fist (palm / back) | W / S |
| Left | pre_pinch / pinch | Mouse move / left click |
| Right | Palm up + swipe up | Space |
| Complex (1) | leftright | T |
| Complex (5) | Grab fingers | Space |
| Complex (6) | Palm-heel clap | E |
| Complex (7) | Thumb-webbing tap | Q |
| Complex (2,3,4) | Other exercises | 2 / 3 / 4 |

## Optional: train complex gestures

To use complex gesture recognition, collect data and train the LSTM (or use an existing `gesture_model.pt`).

1. **Organize videos** by action in subfolders, e.g.:
   ```
   training_videos/
   ├── 1/          # leftright
   │   ├── p1_001.mp4
   │   └── ...
   ├── 2/
   ├── ...
   └── 7/
   ```

2. **Build CSV from videos** (overwrites existing `gesture_data.csv`):
   ```bash
   python collect_data.py --videos training_videos/ --overwrite
   ```

3. **Train the model**:
   ```bash
   python train_gesture.py
   ```
   This produces `gesture_model.pt`. Run `demo_gesture.py` again to use complex gestures.

