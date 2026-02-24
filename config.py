# 全局配置

# ========== OSC 设置 ==========
OSC_IP = "127.0.0.1"
OSC_PORT = 7000

# ========== 摄像头 ==========
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ========== 指令控制 ==========
PRIMARY_HAND = "Right"          # 主控手 ("Right" 或 "Left")
COMMAND_STABLE_FRAMES = 5       # 防抖: 连续N帧相同才切换指令

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
    (0, 1), (1, 2), (2, 3), (3, 4),          # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),          # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),     # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),   # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20),   # 小指
    (5, 9), (9, 13), (13, 17),               # 手掌横连
]

# ========== 手指颜色 ==========
FINGER_COLORS = [
    (255, 0, 255),    # 拇指 - 紫色
    (255, 165, 0),    # 食指 - 橙色
    (0, 255, 255),    # 中指 - 黄色
    (255, 0, 0),      # 无名指 - 蓝色
    (0, 255, 128),    # 小指 - 绿色
]
