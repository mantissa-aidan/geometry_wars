# Utility for sending window-specific keyboard input.
import ctypes # It's good practice to have it if wintypes is used.
import time
from ctypes import wintypes
import win32gui
import win32con
import win32api # Import win32api

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
VK_I = ord('I')
VK_J = ord('J')
VK_K = ord('K')
VK_L = ord('L')

EXTENDED_KEYS = [VK_LEFT, VK_UP, VK_RIGHT, VK_DOWN] # List of extended keys

# Custom exception for invalid window handles
class InvalidWindowHandleError(Exception):
    """Custom exception raised when an invalid window handle is encountered."""
    pass

def get_window_handle(window_title):
    """Get the window handle for the given window title."""
    try:
        handle = win32gui.FindWindow(None, window_title)
        if handle == 0:
            print(f"[window_input] WARNING: Window '{window_title}' not found.")
        return handle
    except Exception as e:
        print(f"[window_input] ERROR in FindWindow for '{window_title}': {e}")
        return 0

# Restoring original _send_key_event logic that handles opposing key releases
def _send_key_event(window_handle, key_code, press=True):
    # Determine if the key is an extended key
    current_extended_flag = 0
    if key_code in EXTENDED_KEYS:
        current_extended_flag = win32con.KEYEVENTF_EXTENDEDKEY

    if window_handle == 0:
        print("[window_input] WARNING: Invalid window handle (0), cannot send key event.")
        return

    current_foreground_window = win32gui.GetForegroundWindow()
    if current_foreground_window != window_handle:
        try:
            win32gui.SetForegroundWindow(window_handle)
            time.sleep(0.05) 
        except win32gui.error as e:
            if hasattr(e, 'winerror') and e.winerror == 1400:
                raise InvalidWindowHandleError(f"Window handle {window_handle} is invalid (SetForegroundWindow failed with error 1400).") from e
            else:
                print(f"[window_input] WARNING: Could not set window {window_handle} to foreground: {e}")

    scan_code = win32api.MapVirtualKey(key_code, 0) # MAPVK_VK_TO_VSC

    if press:
        win32api.keybd_event(key_code, scan_code, current_extended_flag | 0, 0)
    else:
        win32api.keybd_event(key_code, scan_code, current_extended_flag | win32con.KEYEVENTF_KEYUP, 0)

# Movement actions (remain unchanged, using WASD)
def up(window_handle): 
    _send_key_event(window_handle, VK_W)
def down(window_handle): 
    _send_key_event(window_handle, VK_S)
def left(window_handle): 
    _send_key_event(window_handle, VK_A)
def right(window_handle): 
    _send_key_event(window_handle, VK_D)
def stop(window_handle): 
    _send_key_event(window_handle, VK_W, press=False)
    _send_key_event(window_handle, VK_S, press=False)
    _send_key_event(window_handle, VK_A, press=False)
    _send_key_event(window_handle, VK_D, press=False)

# Shooting actions - Reverted to only manage their own axis + opposing key
def shoot_up(window_handle):
    _send_key_event(window_handle, VK_UP, press=True)
    _send_key_event(window_handle, VK_DOWN, press=False)

def shoot_down(window_handle):
    _send_key_event(window_handle, VK_DOWN, press=True)
    _send_key_event(window_handle, VK_UP, press=False)

def shoot_left(window_handle):
    _send_key_event(window_handle, VK_LEFT, press=True)
    _send_key_event(window_handle, VK_RIGHT, press=False)

def shoot_right(window_handle):
    _send_key_event(window_handle, VK_RIGHT, press=True)
    _send_key_event(window_handle, VK_LEFT, press=False)

# Comprehensive stop for all shooting
def shoot_stop(window_handle): 
    _send_key_event(window_handle, VK_UP, press=False)
    _send_key_event(window_handle, VK_DOWN, press=False)
    _send_key_event(window_handle, VK_LEFT, press=False)
    _send_key_event(window_handle, VK_RIGHT, press=False)

# New specific stop functions for each shooting axis
def shoot_y_stop(window_handle):
    _send_key_event(window_handle, VK_UP, press=False)
    _send_key_event(window_handle, VK_DOWN, press=False)

def shoot_x_stop(window_handle):
    _send_key_event(window_handle, VK_LEFT, press=False)
    _send_key_event(window_handle, VK_RIGHT, press=False)

def bomb(window_handle): 
    _send_key_event(window_handle, VK_SPACE)
    time.sleep(0.05) 
    _send_key_event(window_handle, VK_SPACE, press=False)

def enter(window_handle):
    try:
        stop(window_handle)
        shoot_stop(window_handle)
    except InvalidWindowHandleError:
        pass 
    _send_key_event(window_handle, VK_RETURN)
    time.sleep(0.05) 
    _send_key_event(window_handle, VK_RETURN, press=False)

if __name__ == '__main__':
    game_title = 'Geometry Wars: Retro Evolved' 
    handle = get_window_handle(game_title)

    if handle != 0:
        print(f"Window '{game_title}' found with handle: {handle}")
        print("Attempting to bring game window to foreground...")
        try:
            win32gui.SetForegroundWindow(handle)
            time.sleep(0.5) 
            if win32gui.GetForegroundWindow() != handle:
                print("WARNING: Game window might not be in the foreground.")
            else:
                print("Game window should be in foreground.")
        except Exception as e:
            print(f"Error setting initial foreground window: {e}")

        print("Will test ARROW KEY shooting inputs in 3 seconds... GAME WINDOW MUST BE FOCUSED!")
        time.sleep(3)

        try:
            print("Testing SHOOT_RIGHT (ARROW RIGHT key)...")
            shoot_right(handle) 
            time.sleep(2)
            shoot_stop(handle)
            print("SHOOT_RIGHT test finished.")
            time.sleep(1)

            print("Testing SHOOT_UP (ARROW UP key)...")
            shoot_up(handle)    
            time.sleep(2)
            shoot_stop(handle)
            print("SHOOT_UP test finished.")
            time.sleep(1)
            
            # Optional: Test movement as well
            # print("Testing MOVE_UP (W key)...")
            # up(handle)
            # time.sleep(1)
            # stop(handle)
            # print("MOVE_UP test finished.")
            # time.sleep(1)

            print("ARROW KEY shooting input test complete. Check game window.")

        except InvalidWindowHandleError as iwhe:
            print(f"TEST FAILED: Invalid window handle: {iwhe}")
        except Exception as ex:
            print(f"TEST FAILED: Unexpected error: {ex}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Window '{game_title}' not found. Cannot run input test.") 