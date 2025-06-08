# Placeholder for memory reading logic import ctypes
import ctypes
from ctypes import wintypes
import struct
import os
import sys
import time
import psutil

def find_process_id_by_name(possible_names):
    """Return the PID of the first process matching any name in possible_names, or None if not found."""
    if isinstance(possible_names, str):
        possible_names = [possible_names]
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for name in possible_names:
                if name.lower() in proc.info['name'].lower():
                    return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

ERROR_PARTIAL_COPY = 0x012B
PROCESS_VM_READ = 0x0010

SIZE_T = ctypes.c_size_t
PSIZE_T = ctypes.POINTER(SIZE_T)

def _check_zero(result, func, args):
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
    return args

kernel32.OpenProcess.errcheck = _check_zero
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = (
    wintypes.DWORD, # _In_ dwDesiredAccess
    wintypes.BOOL,  # _In_ bInheritHandle
    wintypes.DWORD) # _In_ dwProcessId

kernel32.ReadProcessMemory.errcheck = _check_zero
kernel32.ReadProcessMemory.argtypes = (
    wintypes.HANDLE,  # _In_  hProcess
    wintypes.LPCVOID, # _In_  lpBaseAddress
    wintypes.LPVOID,  # _Out_ lpBuffer
    SIZE_T,           # _In_  nSize
    PSIZE_T)          # _Out_ lpNumberOfBytesRead

kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

def read_process_memory(pid, address, size, allow_partial=False):
    buf = (ctypes.c_char * size)()
    nread = SIZE_T()
    hProcess = kernel32.OpenProcess(PROCESS_VM_READ, False, pid)
    try:
        kernel32.ReadProcessMemory(hProcess, address, buf, size,
            ctypes.byref(nread))
    except WindowsError as e:
        if not allow_partial or e.winerror != ERROR_PARTIAL_COPY:
            raise
    finally:
        kernel32.CloseHandle(hProcess)
    return buf[:nread.value]

def getScore():
    possible_names = ["GeometryWars", "Geometry Wars: Retro Evolved", "GeometryWars.exe"]
    pid = find_process_id_by_name(possible_names)
    if pid is None:
        raise RuntimeError("Could not find Geometry Wars process! Make sure the game is running.")
    score = read_process_memory(pid, 0x63C890, 4)
    score = int.from_bytes(score, "little")
    return score

def getLives():
    possible_names = ["GeometryWars", "Geometry Wars: Retro Evolved", "GeometryWars.exe"]
    pid = find_process_id_by_name(possible_names)
    if pid is None:
        raise RuntimeError("Could not find Geometry Wars process! Make sure the game is running.")
    lives = read_process_memory(pid, 0x63C8A0, 4)
    lives = int.from_bytes(lives, "little")
    return lives


