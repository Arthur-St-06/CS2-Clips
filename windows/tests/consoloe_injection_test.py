# minimal_hlae_console.py
import os
import subprocess
import time
from pathlib import Path
import ctypes
from ctypes import wintypes

# =========================
# Paths (reuse your env-based config)
# =========================
CS2_EXE = Path(os.environ.get(
    "CS2_EXE",
    r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe"
))
HLAE_EXE = Path(os.environ.get("HLAE_EXE", r"C:\Program Files (x86)\HLAE\HLAE.exe"))
HOOK_DLL = Path(os.environ.get("HOOK_DLL", r"C:\Program Files (x86)\HLAE\x64\AfxHookSource2.dll"))

DEMO_PATH = Path(
    r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\csgo\replays\match730_003792077342759714824_1663050728_392.dem"
)

EXTRA_LAUNCH = os.environ.get("EXTRA_LAUNCH", "-steam -insecure -novid -nojoy -console")

# =========================
# DLLs
# =========================
user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
shell32 = ctypes.WinDLL("shell32", use_last_error=True)

GetCurrentThreadId = kernel32.GetCurrentThreadId
GetCurrentThreadId.argtypes = []
GetCurrentThreadId.restype = wintypes.DWORD

# Admin check (helps diagnose UIPI blocks)
IsUserAnAdmin = shell32.IsUserAnAdmin
IsUserAnAdmin.argtypes = []
IsUserAnAdmin.restype = wintypes.BOOL

# =========================
# WinAPI prototypes
# =========================
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

EnumWindows = user32.EnumWindows
EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
EnumWindows.restype = wintypes.BOOL

GetWindowTextW = user32.GetWindowTextW
GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
GetWindowTextW.restype = ctypes.c_int

GetWindowTextLengthW = user32.GetWindowTextLengthW
GetWindowTextLengthW.argtypes = [wintypes.HWND]
GetWindowTextLengthW.restype = ctypes.c_int

IsWindowVisible = user32.IsWindowVisible
IsWindowVisible.argtypes = [wintypes.HWND]
IsWindowVisible.restype = wintypes.BOOL

ShowWindow = user32.ShowWindow
ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
ShowWindow.restype = wintypes.BOOL

SetForegroundWindow = user32.SetForegroundWindow
SetForegroundWindow.argtypes = [wintypes.HWND]
SetForegroundWindow.restype = wintypes.BOOL

BringWindowToTop = user32.BringWindowToTop
BringWindowToTop.argtypes = [wintypes.HWND]
BringWindowToTop.restype = wintypes.BOOL

SetFocus = user32.SetFocus
SetFocus.argtypes = [wintypes.HWND]
SetFocus.restype = wintypes.HWND

GetForegroundWindow = user32.GetForegroundWindow
GetForegroundWindow.argtypes = []
GetForegroundWindow.restype = wintypes.HWND

GetWindowThreadProcessId = user32.GetWindowThreadProcessId
GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
GetWindowThreadProcessId.restype = wintypes.DWORD

AttachThreadInput = user32.AttachThreadInput
AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
AttachThreadInput.restype = wintypes.BOOL

GetWindowRect = user32.GetWindowRect
GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
GetWindowRect.restype = wintypes.BOOL

GetSystemMetrics = user32.GetSystemMetrics
GetSystemMetrics.argtypes = [ctypes.c_int]
GetSystemMetrics.restype = ctypes.c_int

# Clipboard
OpenClipboard = user32.OpenClipboard
OpenClipboard.argtypes = [wintypes.HWND]
OpenClipboard.restype = wintypes.BOOL

EmptyClipboard = user32.EmptyClipboard
EmptyClipboard.argtypes = []
EmptyClipboard.restype = wintypes.BOOL

SetClipboardData = user32.SetClipboardData
SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
SetClipboardData.restype = wintypes.HANDLE

CloseClipboard = user32.CloseClipboard
CloseClipboard.argtypes = []
CloseClipboard.restype = wintypes.BOOL

GlobalAlloc = kernel32.GlobalAlloc
GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
GlobalAlloc.restype = wintypes.HGLOBAL

GlobalLock = kernel32.GlobalLock
GlobalLock.argtypes = [wintypes.HGLOBAL]
GlobalLock.restype = wintypes.LPVOID

GlobalUnlock = kernel32.GlobalUnlock
GlobalUnlock.argtypes = [wintypes.HGLOBAL]
GlobalUnlock.restype = wintypes.BOOL

GMEM_MOVEABLE = 0x0002
CF_UNICODETEXT = 13

SW_RESTORE = 9

# =========================
# SendInput (keyboard + mouse)
# =========================
INPUT_KEYBOARD = 1
INPUT_MOUSE = 0

KEYEVENTF_KEYUP = 0x0002

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

VK_RETURN = 0x0D
VK_ESCAPE = 0x1B
VK_CONTROL = 0x11
VK_V = 0x56
CONSOLE_VK = 0xC0

ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]
    _anonymous_ = ("i",)
    _fields_ = [("type", wintypes.DWORD), ("i", _I)]

SendInput = user32.SendInput
SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
SendInput.restype = wintypes.UINT

def _send_vk(vk: int, keyup: bool = False) -> None:
    flags = KEYEVENTF_KEYUP if keyup else 0
    inp = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0))
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

def _tap_vk(vk: int) -> None:
    _send_vk(vk, False)
    _send_vk(vk, True)

def _mouse_click_abs(x: int, y: int) -> None:
    SM_CXVIRTUALSCREEN = 78
    SM_CYVIRTUALSCREEN = 79
    SM_XVIRTUALSCREEN = 76
    SM_YVIRTUALSCREEN = 77

    vx = GetSystemMetrics(SM_XVIRTUALSCREEN)
    vy = GetSystemMetrics(SM_YVIRTUALSCREEN)
    vw = GetSystemMetrics(SM_CXVIRTUALSCREEN)
    vh = GetSystemMetrics(SM_CYVIRTUALSCREEN)
    if vw <= 1 or vh <= 1:
        return

    ax = int((x - vx) * 65535 / (vw - 1))
    ay = int((y - vy) * 65535 / (vh - 1))

    move = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dx=ax, dy=ay, mouseData=0, dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, time=0, dwExtraInfo=0))
    down = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dx=ax, dy=ay, mouseData=0, dwFlags=MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE, time=0, dwExtraInfo=0))
    up   = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dx=ax, dy=ay, mouseData=0, dwFlags=MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE, time=0, dwExtraInfo=0))

    SendInput(1, ctypes.byref(move), ctypes.sizeof(INPUT))
    SendInput(1, ctypes.byref(down), ctypes.sizeof(INPUT))
    SendInput(1, ctypes.byref(up), ctypes.sizeof(INPUT))

def _set_clipboard_text(text: str) -> None:
    if not OpenClipboard(None):
        raise RuntimeError("OpenClipboard failed (clipboard busy).")
    try:
        if not EmptyClipboard():
            raise RuntimeError("EmptyClipboard failed.")

        data = (text + "\0").encode("utf-16le")
        hmem = GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not hmem:
            raise RuntimeError("GlobalAlloc failed.")

        ptr = GlobalLock(hmem)
        if not ptr:
            raise RuntimeError("GlobalLock failed.")
        try:
            ctypes.memmove(ptr, data, len(data))
        finally:
            GlobalUnlock(hmem)

        if not SetClipboardData(CF_UNICODETEXT, hmem):
            raise RuntimeError("SetClipboardData failed.")
        # System owns hmem after success.
    finally:
        CloseClipboard()

# =========================
# Find + focus CS2 window
# =========================
def _find_cs2_window() -> int | None:
    targets = ("counter-strike 2", "counter-strike", "cs2")
    found_hwnd = None

    @WNDENUMPROC
    def enum_proc(hwnd, lparam):
        nonlocal found_hwnd
        if not IsWindowVisible(hwnd):
            return True
        length = GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip().lower()
        if any(t in title for t in targets):
            found_hwnd = hwnd
            return False
        return True

    EnumWindows(enum_proc, 0)
    return found_hwnd

def _force_focus_and_click(hwnd: int) -> None:
    ShowWindow(hwnd, SW_RESTORE)

    fg = GetForegroundWindow()
    fg_tid = GetWindowThreadProcessId(fg, ctypes.byref(wintypes.DWORD(0))) if fg else 0
    tgt_tid = GetWindowThreadProcessId(hwnd, ctypes.byref(wintypes.DWORD(0)))
    cur_tid = GetCurrentThreadId()

    if fg_tid:
        AttachThreadInput(cur_tid, fg_tid, True)
    AttachThreadInput(cur_tid, tgt_tid, True)

    try:
        BringWindowToTop(hwnd)
        SetForegroundWindow(hwnd)
        SetFocus(hwnd)
    finally:
        AttachThreadInput(cur_tid, tgt_tid, False)
        if fg_tid:
            AttachThreadInput(cur_tid, fg_tid, False)

    time.sleep(0.10)

# =========================
# Console automation (strict sequence)
# =========================
def _toggle_console() -> None:
    # Use ONLY the configured VK to avoid accidental TAB/scoreboard behavior.
    _tap_vk(CONSOLE_VK)
    time.sleep(0.12)

def _paste_text(text: str) -> None:
    _set_clipboard_text(text)
    time.sleep(0.03)

    # Ctrl+V
    _send_vk(VK_CONTROL, False)
    _tap_vk(VK_V)
    _send_vk(VK_CONTROL, True)
    time.sleep(0.05)

def send_console_cmd_to_cs2(cmd_text: str) -> None:
    hwnd = _find_cs2_window()
    if not hwnd:
        raise RuntimeError("Couldn't find a CS2 window. Is the game running and visible?")

    _force_focus_and_click(hwnd)

    # Clear any UI state that steals keys
    _tap_vk(VK_ESCAPE)
    time.sleep(0.05)
    _tap_vk(VK_ESCAPE)
    time.sleep(0.05)

    # 1) Open console
    _toggle_console()

    # 2) Paste command, press Enter
    _paste_text(cmd_text)
    _tap_vk(VK_RETURN)
    time.sleep(0.10)

    # 3) Close console
    _toggle_console()

# =========================
# Launch via HLAE
# =========================
def require_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(str(p))

def launch_cs2_via_hlae_with_demo() -> subprocess.Popen:
    for p in [CS2_EXE, HLAE_EXE, HOOK_DLL, DEMO_PATH]:
        require_exists(p)

    cmdline = f'{EXTRA_LAUNCH} +playdemo "{DEMO_PATH}"'
    cmd = [
        str(HLAE_EXE),
        "-customLoader",
        "-autoStart",
        "-hookDllPath", str(HOOK_DLL),
        "-programPath", str(CS2_EXE),
        "-cmdLine", cmdline,
    ]
    return subprocess.Popen(cmd)

def main() -> None:
    if not bool(IsUserAnAdmin()):
        print("[warn] Not elevated. If CS2/HLAE is elevated, Windows may block injected input.")
        print("[warn] Run this script as Administrator OR run CS2/HLAE non-elevated.\n")

    proc = launch_cs2_via_hlae_with_demo()
    print("CS2 launched via HLAE and demo started.")
    print(f"Console toggle VK: {hex(CONSOLE_VK)} (override with CONSOLE_VK=0x.. if needed)")
    print("Type CS2 console commands here. Examples:")
    print('  demoui')
    print('  spec_player "Remag"')
    print('  spec_player "Gknight"')
    print("Type: quit")

    time.sleep(7.0)

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            break

        try:
            send_console_cmd_to_cs2(line)
        except Exception as e:
            print(f"[error] {e}")

    if proc.poll() is not None:
        print("HLAE/CS2 already exited.")
    else:
        print("Leaving CS2 running. Close it normally.")

if __name__ == "__main__":
    main()
