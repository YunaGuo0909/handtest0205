# Global config

# ========== Camera ==========
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ========== Hand roles ==========
MOVE_HAND = "Left"              # Left hand: continuous move
ACTION_HAND = "Right"           # Right hand: tap actions

# ========== Debounce ==========
MOVE_STABLE_FRAMES = 5          # Move key debounce frames
ACTION_COOLDOWN = 0.8           # Tap action cooldown (seconds)
COMBO_COOLDOWN = 1.5            # Two-hand combo cooldown (seconds)

# ========== Motion ==========
SWIPE_UP_THRESHOLD = 0.15       # Swipe up: y-delta threshold
SWIPE_HISTORY_FRAMES = 4        # Frames to compute swipe
HANDS_TOUCH_THRESHOLD = 0.08    # Wrist touch distance threshold
POINTING_X_THRESHOLD = 0.04     # Pointing horizontal component (filter vertical)

# ========== MediaPipe model ==========
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# ========== Detection confidence ==========
MIN_DETECTION_CONFIDENCE = 0.3
MIN_PRESENCE_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3

# ========== Hand landmark names (21) ==========
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# ========== Skeleton connections ==========
BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# ========== Finger colors ==========
FINGER_COLORS = [
    (255, 0, 255),
    (255, 165, 0),
    (0, 255, 255),
    (255, 0, 0),
    (0, 255, 128),
]
