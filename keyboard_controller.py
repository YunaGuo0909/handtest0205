# Keyboard + mouse controller: simulate input from gesture commands.

from pynput.mouse import Controller as MouseController, Button
import sys
import time

# Windows virtual key codes
_VK = {
    "w": 0x57, "s": 0x53, "a": 0x41, "d": 0x44,
    "q": 0x51, "e": 0x45, "t": 0x54,
    "f": 0x46, "p": 0x50,
    "space": 0x20, "shift": 0x10,
}

KEYEVENTF_KEYUP = 0x0002

if sys.platform == "win32":
    import ctypes
    _user32 = ctypes.windll.user32

    def _send_key(vk, key_up=False):
        """keybd_event with scan code for game compatibility."""
        scan = _user32.MapVirtualKeyW(vk, 0)  # VK → scan code
        flags = KEYEVENTF_KEYUP if key_up else 0
        _user32.keybd_event(vk, scan, flags, 0)
else:
    def _send_key(vk, key_up=False):
        pass


def _get_screen_size():
    try:
        import ctypes
        u32 = ctypes.windll.user32
        return u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


# Delay between press and release for tap (seconds)
_TAP_DELAY = 0.05


class KeyboardController:
    """Simulate keyboard and mouse from gesture commands."""

    def __init__(self):
        self.mouse = MouseController()
        self._held_keys = set()
        self._screen_w, self._screen_h = _get_screen_size()

    # ------------------------------------------------------------------
    # Keyboard: hold / tap / release
    # ------------------------------------------------------------------

    def hold_key(self, key_char):
        """Hold a key (for continuous move)."""
        if key_char not in self._held_keys:
            self._release_all_movement()
            vk = _VK.get(key_char)
            if vk:
                _send_key(vk, key_up=False)
            self._held_keys.add(key_char)

    def tap_key(self, key_char):
        """Tap a key once."""
        vk = _VK.get(key_char)
        if vk:
            _send_key(vk, key_up=False)
            time.sleep(_TAP_DELAY)
            _send_key(vk, key_up=True)

    def release_key(self, key_char):
        """Release a held key."""
        if key_char in self._held_keys:
            vk = _VK.get(key_char)
            if vk:
                _send_key(vk, key_up=True)
            self._held_keys.discard(key_char)

    def _release_all_movement(self):
        """Release all held movement keys."""
        for key_char in list(self._held_keys):
            vk = _VK.get(key_char)
            if vk:
                _send_key(vk, key_up=True)
        self._held_keys.clear()

    # ------------------------------------------------------------------
    # Mouse: move / click
    # ------------------------------------------------------------------

    def mouse_move(self, norm_x, norm_y):
        """Move mouse to normalized position (0~1 → screen pixels)."""
        x = int(norm_x * self._screen_w)
        y = int(norm_y * self._screen_h)
        self.mouse.position = (x, y)

    def mouse_click(self):
        """Left mouse click."""
        self.mouse.click(Button.left)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def stop(self):
        """Stop all input."""
        self._release_all_movement()

    def close(self):
        """Cleanup: release all keys."""
        self._release_all_movement()
