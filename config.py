# 全局配置

# ========== 摄像头 ==========
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ========== 手部分工 ==========
MOVE_HAND = "Left"              # 左手控制移动（连续）
ACTION_HAND = "Right"           # 右手控制技能（瞬发）

# ========== 防抖 ==========
MOVE_STABLE_FRAMES = 5          # 移动指令防抖帧数
ACTION_COOLDOWN = 0.8           # 瞬发动作冷却时间(秒)
COMBO_COOLDOWN = 1.5            # 双手组合冷却时间(秒)

# ========== 运动检测 ==========
SWIPE_UP_THRESHOLD = 0.15       # 手背向上挥动的 y 坐标变化阈值
SWIPE_HISTORY_FRAMES = 4        # 追踪最近 N 帧计算挥动
HANDS_TOUCH_THRESHOLD = 0.08    # 双手碰触距离阈值
POINTING_X_THRESHOLD = 0.04     # 食指水平分量阈值（过滤竖直方向指向）

# ========== MediaPipe 模型 ==========
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# ========== 检测置信度 ==========
MIN_DETECTION_CONFIDENCE = 0.3
MIN_PRESENCE_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3

# ========== 手部关键点名称 (21个) ==========
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# ========== 骨架连线 ==========
BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# ========== 手指颜色 ==========
FINGER_COLORS = [
    (255, 0, 255),
    (255, 165, 0),
    (0, 255, 255),
    (255, 0, 0),
    (0, 255, 128),
]
