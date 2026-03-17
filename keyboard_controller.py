# Keyboard controller: simulate key presses from gesture commands (e.g. for UE/games).

from pynput.keyboard import Controller, Key, KeyCode
import time
import sys

# Windows virtual key codes; W/S sent via keybd_event for reliability in Notepad etc.
VK_W = 0x57
VK_S = 0x53
KEYEVENTF_KEYUP = 0x0002


def _keybd_event(vk, key_up=False):
    """Windows keybd_event; used for W/S when pynput character input is unreliable."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP if key_up else 0, 0)
    except Exception:
        pass


class KeyboardController:
    """Simulate keyboard from gesture commands.

    Hold (move): press and hold while gesture is active, release when it ends.
    Tap (action): press once and release.
    """

    def __init__(self):
        self.kb = Controller()
        self._held_keys = set()

    def _key_obj(self, key_char):
        """W/S as VK KeyCode for pynput."""
        if key_char == "w":
            return KeyCode.from_vk(VK_W)
        if key_char == "s":
            return KeyCode.from_vk(VK_S)
        return key_char

    def _use_win_keybd(self, key_char):
        """Whether to send via Windows keybd_event (W/S more reliable in Notepad etc.)."""
        return key_char in ("w", "s")

    def hold_key(self, key_char):
        """Hold a key (for continuous move)."""
        if key_char not in self._held_keys:
            self._release_all_movement()
            if self._use_win_keybd(key_char):
                vk = VK_W if key_char == "w" else VK_S
                _keybd_event(vk, key_up=False)
            else:
                self.kb.press(self._key_obj(key_char))
            self._held_keys.add(key_char)

    def tap_key(self, key_char):
        """Tap a key once (for instant action)."""
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
        """Release a key."""
        if key_char in self._held_keys:
            if self._use_win_keybd(key_char):
                vk = VK_W if key_char == "w" else VK_S
                _keybd_event(vk, key_up=True)
            else:
                self.kb.release(self._key_obj(key_char))
            self._held_keys.discard(key_char)

    def _release_all_movement(self):
        """Release all held movement keys."""
        for key_char in list(self._held_keys):
            if self._use_win_keybd(key_char):
                vk = VK_W if key_char == "w" else VK_S
                _keybd_event(vk, key_up=True)
            else:
                self.kb.release(self._key_obj(key_char))
        self._held_keys.clear()

    def stop(self):
        """Stop all input."""
        self._release_all_movement()

    def close(self):
        """Cleanup: release all keys."""
        self._release_all_movement()
