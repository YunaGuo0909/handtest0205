# Real-Time Hand Gesture Recognition for Game Control

A webcam-based hand gesture recognition system that maps hand poses and movements to keyboard/mouse input for controlling games — no additional hardware required.

Uses a **dual recognition engine**: rule-based geometry classification for simple single-hand poses, and an LSTM neural network for complex two-hand temporal gestures. Game input is delivered via **Win32 virtual keyboard simulation** (keybd_event with hardware scan codes), making it game-agnostic with near-zero latency.

**Author:** Yutong Guo

## Project Structure

```
├── main.py                  # Entry point: camera loop, dual-engine state machine
├── hand_tracker.py          # MediaPipe HandLandmarker wrapper (detect + draw)
├── gesture_recognizer.py    # Rule-based 14-gesture classifier (3D joint geometry)
├── keyboard_controller.py   # Win32 keybd_event virtual keyboard/mouse simulation
├── command_mapper.py        # Gesture → keyboard/mouse action mapping
├── config.py                # Global parameters & thresholds
├── train_gesture.py         # LSTM model definition + training pipeline
├── collect_data.py          # Training data collection (video batch / live camera)
├── demo_gesture.py          # Alternative demo entry point with different tuning
├── GestureControl.spec      # PyInstaller config for standalone .exe packaging
├── pyproject.toml           # Project metadata & dependencies
└── requirements.txt         # pip dependencies
```

**Not tracked in git** (generated locally):

```
├── hand_landmarker.task     # MediaPipe model (auto-downloaded on first run)
├── gesture_model.pt         # Trained LSTM checkpoint
├── gesture_data.csv         # Training data CSV
├── training_videos/         # Raw training video clips by gesture label
├── build/                   # PyInstaller build artifacts
└── dist/                    # Packaged standalone executable
```

## Requirements

- Python 3.12+
- Windows (virtual keyboard uses Win32 API)
- Webcam

## Install & Run

```bash
pip install -r requirements.txt
python main.py
```

The MediaPipe hand model (~10 MB) is downloaded automatically on first run. Press **ESC** to quit.

- **Without `gesture_model.pt`**: Simple gestures only (movement + mouse control).
- **With `gesture_model.pt`**: Complex two-hand exercise gestures are also enabled.

## How It Works

### Dual-Engine State Machine

| State | Condition | Engine |
|-------|-----------|--------|
| `SIMPLE` | One hand detected | Rule-based geometry classifier |
| `WAIT_LSTM` | Both hands, LSTM buffer filling | Simple gestures suppressed |
| `LSTM` | Both hands, confident prediction | LSTM complex gesture classifier |

### Key Bindings

**Simple gestures (single hand):**

| Hand | Gesture | Output |
|------|---------|--------|
| Left | Point right / left | D / A (hold) |
| Left | Fist (palm facing / back) | W / S (hold) |
| Left | Pre-pinch / pinch | Mouse move / left click |
| Right | Repeated open-fist (grab ×2) | Space (tap) |
| Both | Open palms still 3s | P (tap) |

**Complex gestures (LSTM, both hands):**

| ID | Exercise | Key |
|----|----------|-----|
| 1 | Left-right swing | F |
| 5 | Finger grabbing | Space |
| 6 | Palm-heel clap | E |
| 7 | Tiger-mouth strike | T |

### Anti-Flicker Mechanisms

- Pinch state smoothing (4-frame hysteresis)
- Movement key debounce (5-frame unanimous vote)
- LSTM confidence voting (6-frame sliding window, per-class thresholds)
- LSTM-pinch mutual exclusion during two-hand gestures

## Train Complex Gestures (Optional)

1. **Collect data** — organise videos in labelled subfolders:
   ```
   training_videos/
   ├── 1/    # left-right swing
   ├── 2/    # back-hand clap
   ├── ...
   └── 7/    # tiger-mouth strike
   ```

2. **Extract landmarks from videos:**
   ```bash
   python collect_data.py --videos training_videos/ --overwrite
   ```

3. **Train the LSTM:**
   ```bash
   python train_gesture.py
   ```
   Produces `gesture_model.pt`. Restart `main.py` to enable complex gestures.

## Standalone Executable

```bash
pyinstaller GestureControl.spec
```

Output: `dist/GestureControl/GestureControl.exe` — runs without Python installed.
