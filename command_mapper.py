# 指令映射模块
# 将手势 + 手掌朝向映射为 UE 游戏指令，并提供防抖机制

from collections import deque, Counter

from config import COMMAND_STABLE_FRAMES


class CommandMapper:
    """手势 → 游戏指令映射器（含防抖）

    指令映射规则:
        握拳 + 正面(手心朝镜头) → forward  (前进)
        握拳 + 背面(手背朝镜头) → backward (后退)
        张开 + 正面(手心朝镜头) → stop     (停止)

    防抖机制:
        维护一个滑动窗口，连续 N 帧识别结果相同时才切换指令，
        避免单帧误识别导致角色动作抖动。
    """

    # (手势, 是否正面) → 指令
    GESTURE_COMMAND_MAP = {
        ("fist", True):   "forward",
        ("fist", False):  "backward",
        ("open", True):   "stop",
    }

    # 指令 → 移动值 (给 UE 的 AddMovementInput 直接用)
    COMMAND_MOVE_VALUES = {
        "forward":  1.0,
        "backward": -1.0,
        "stop":     0.0,
        "none":     0.0,
    }

    def __init__(self, stable_frames=None):
        self._window_size = stable_frames or COMMAND_STABLE_FRAMES
        self._history = deque(maxlen=self._window_size)
        self._current_command = "none"

    def raw_command(self, gesture, palm_facing):
        """直接查表，不经过防抖（用于调试显示）"""
        return self.GESTURE_COMMAND_MAP.get((gesture, palm_facing), "none")

    def update(self, gesture, palm_facing):
        """输入当前帧的手势和朝向，返回防抖后的稳定指令

        Args:
            gesture: 手势名称
            palm_facing: 手心是否朝镜头

        Returns:
            str: 稳定指令 ("forward" / "backward" / "stop" / "none")
        """
        raw = self.raw_command(gesture, palm_facing)
        self._history.append(raw)

        if len(self._history) < self._window_size:
            return self._current_command

        counts = Counter(self._history)
        most_common, count = counts.most_common(1)[0]

        if count >= self._window_size:
            self._current_command = most_common

        return self._current_command

    def reset(self):
        """重置状态"""
        self._history.clear()
        self._current_command = "none"

    @property
    def command(self):
        """当前稳定指令"""
        return self._current_command
