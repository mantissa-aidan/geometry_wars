import ctypes
import time
from ctypes import wintypes
import win32gui
import win32con

# Constants for virtual key codes
VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_SPACE = 0x20
VK_RETURN = 0x0D

def get_window_handle(window_title):
    """Get the window handle for the given window title."""
    return win32gui.FindWindow(None, window_title)

def send_key_to_window(window_handle, key_code, is_press=True):
    """Send a key press or release message to a specific window."""
    if window_handle == 0:
        return False
    
    message = win32con.WM_KEYDOWN if is_press else win32con.WM_KEYUP
    win32gui.PostMessage(window_handle, message, key_code, 0)
    return True

def up(window_handle):
    """Send up movement keys to the window."""
    send_key_to_window(window_handle, VK_W, True)
    send_key_to_window(window_handle, VK_S, False)

def down(window_handle):
    """Send down movement keys to the window."""
    send_key_to_window(window_handle, VK_S, True)
    send_key_to_window(window_handle, VK_W, False)

def left(window_handle):
    """Send left movement keys to the window."""
    send_key_to_window(window_handle, VK_A, True)
    send_key_to_window(window_handle, VK_D, False)

def right(window_handle):
    """Send right movement keys to the window."""
    send_key_to_window(window_handle, VK_D, True)
    send_key_to_window(window_handle, VK_A, False)

def stop(window_handle):
    """Release all movement keys."""
    send_key_to_window(window_handle, VK_W, False)
    send_key_to_window(window_handle, VK_A, False)
    send_key_to_window(window_handle, VK_S, False)
    send_key_to_window(window_handle, VK_D, False)

def shoot_up(window_handle):
    """Send shoot up keys to the window."""
    send_key_to_window(window_handle, VK_UP, True)
    send_key_to_window(window_handle, VK_DOWN, False)

def shoot_down(window_handle):
    """Send shoot down keys to the window."""
    send_key_to_window(window_handle, VK_DOWN, True)
    send_key_to_window(window_handle, VK_UP, False)

def shoot_left(window_handle):
    """Send shoot left keys to the window."""
    send_key_to_window(window_handle, VK_LEFT, True)
    send_key_to_window(window_handle, VK_RIGHT, False)

def shoot_right(window_handle):
    """Send shoot right keys to the window."""
    send_key_to_window(window_handle, VK_RIGHT, True)
    send_key_to_window(window_handle, VK_LEFT, False)

def shoot_stop(window_handle):
    """Release all shooting keys."""
    send_key_to_window(window_handle, VK_UP, False)
    send_key_to_window(window_handle, VK_LEFT, False)
    send_key_to_window(window_handle, VK_DOWN, False)
    send_key_to_window(window_handle, VK_RIGHT, False)

def bomb(window_handle):
    """Send bomb key to the window."""
    send_key_to_window(window_handle, VK_SPACE, True)
    time.sleep(0.01)
    send_key_to_window(window_handle, VK_SPACE, False)

def enter(window_handle):
    """Send enter key to the window."""
    stop(window_handle)
    send_key_to_window(window_handle, VK_RETURN, True)
    time.sleep(0.01)
    send_key_to_window(window_handle, VK_RETURN, False) 