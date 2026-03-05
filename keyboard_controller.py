# 键盘控制模块
# 根据手势指令模拟键盘按键，直接驱动 UE 角色

from pynput.keyboard import Controller, Key
import time


class KeyboardController:
    """键盘模拟器 - 将手势指令转换为键盘按键

    连续按键（移动）: 手势保持期间持续按住，手势消失时松开
    瞬发按键（技能）: 按一下立即松开
    """

    def __init__(self):
        self.kb = Controller()
        self._held_keys = set()

    def hold_key(self, key_char):
        """按住一个键（连续动作用）"""
        if key_char not in self._held_keys:
            self._release_all_movement()
            self.kb.press(key_char)
            self._held_keys.add(key_char)

    def tap_key(self, key_char):
        """按一下键（瞬发动作用）"""
        if key_char == "shift":
            self.kb.press(Key.shift)
            self.kb.release(Key.shift)
        elif key_char == "space":
            self.kb.press(Key.space)
            self.kb.release(Key.space)
        else:
            self.kb.press(key_char)
            self.kb.release(key_char)

    def release_key(self, key_char):
        """松开一个键"""
        if key_char in self._held_keys:
            self.kb.release(key_char)
            self._held_keys.discard(key_char)

    def _release_all_movement(self):
        """松开所有被按住的移动键"""
        for k in list(self._held_keys):
            self.kb.release(k)
        self._held_keys.clear()

    def stop(self):
        """停止所有输入"""
        self._release_all_movement()

    def close(self):
        """清理：松开所有键"""
        self._release_all_movement()
