# 键盘控制模块
# 根据手势指令模拟键盘按键，直接驱动 UE 角色

from pynput.keyboard import Controller, Key, KeyCode
import time
import sys

# Windows 虚拟键码，确保 W/S 在记事本等窗口能打出字（部分环境用字符会失效）
VK_W = 0x57
VK_S = 0x53
KEYEVENTF_KEYUP = 0x0002


def _keybd_event(vk, key_up=False):
    """Windows keybd_event，仅用于 W/S 在部分环境更可靠"""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP if key_up else 0, 0)
    except Exception:
        pass


class KeyboardController:
    """键盘模拟器 - 将手势指令转换为键盘按键

    连续按键（移动）: 手势保持期间持续按住，手势消失时松开
    瞬发按键（技能）: 按一下立即松开
    """

    def __init__(self):
        self.kb = Controller()
        self._held_keys = set()

    def _key_obj(self, key_char):
        """W/S 用虚拟键码，供 pynput 使用"""
        if key_char == "w":
            return KeyCode.from_vk(VK_W)
        if key_char == "s":
            return KeyCode.from_vk(VK_S)
        return key_char

    def _use_win_keybd(self, key_char):
        """是否用 Windows keybd_event 发送（W/S 在记事本等更可靠）"""
        return key_char in ("w", "s")

    def hold_key(self, key_char):
        """按住一个键（连续动作用）"""
        if key_char not in self._held_keys:
            self._release_all_movement()
            if self._use_win_keybd(key_char):
                vk = VK_W if key_char == "w" else VK_S
                _keybd_event(vk, key_up=False)
            else:
                self.kb.press(self._key_obj(key_char))
            self._held_keys.add(key_char)

    def tap_key(self, key_char):
        """按一下键（瞬发动作用）"""
        if key_char == "shift":
            self.kb.press(Key.shift)
            self.kb.release(Key.shift)
        elif key_char == "space":
            self.kb.press(Key.space)
            self.kb.release(Key.space)
        elif self._use_win_keybd(key_char):
            vk = VK_W if key_char == "w" else VK_S
            _keybd_event(vk, key_up=False)
            _keybd_event(vk, key_up=True)
        else:
            k = self._key_obj(key_char)
            self.kb.press(k)
            self.kb.release(k)

    def release_key(self, key_char):
        """松开一个键"""
        if key_char in self._held_keys:
            if self._use_win_keybd(key_char):
                vk = VK_W if key_char == "w" else VK_S
                _keybd_event(vk, key_up=True)
            else:
                self.kb.release(self._key_obj(key_char))
            self._held_keys.discard(key_char)

    def _release_all_movement(self):
        """松开所有被按住的移动键"""
        for key_char in list(self._held_keys):
            if self._use_win_keybd(key_char):
                vk = VK_W if key_char == "w" else VK_S
                _keybd_event(vk, key_up=True)
            else:
                self.kb.release(self._key_obj(key_char))
        self._held_keys.clear()

    def stop(self):
        """停止所有输入"""
        self._release_all_movement()

    def close(self):
        """清理：松开所有键"""
        self._release_all_movement()
