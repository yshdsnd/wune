"""Windows Sony Music Center for PC metadata provider."""
from __future__ import annotations

import sys
from typing import Callable

from .now_playing import MetadataProvider, NowPlaying

APP_TITLE_NAMES = {
    "music center for pc",
    "music center",
    "sony music center for pc",
    "sony music center",
}


def parse_music_center_title(raw_title: str) -> NowPlaying | None:
    """Parse Music Center for PC window title into NowPlaying metadata.

    Format is typically: 'Title / Artist / Album' or 'Title / Artist' or 'Title'.
    When no track is active or playback is stopped, the window title is typically
    the application name ('Music Center for PC') or empty.
    """
    if not raw_title:
        return None
    cleaned = raw_title.strip()
    if not cleaned:
        return None
    if cleaned.lower() in APP_TITLE_NAMES:
        return None

    parts = [p.strip() for p in cleaned.split(" / ")]
    if len(parts) == 1:
        title = parts[0]
        artist = ""
        album = ""
    elif len(parts) == 2:
        title = parts[0]
        artist = parts[1]
        album = ""
    else:
        title = parts[0]
        artist = parts[1]
        album = " / ".join(parts[2:])

    if not title and not artist:
        return None

    return NowPlaying(
        title=title,
        artist=artist,
        album=album,
        source="Music Center for PC",
        is_playing=True,
        raw_text=cleaned,
    )


def default_find_music_center_title() -> str | None:
    """Find the visible top-level window title of Music Center for PC."""
    if sys.platform != "win32":
        return None

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def get_proc_name(pid: int) -> str:
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return buf.value.rsplit("\\", 1)[-1].lower()
            return ""
        finally:
            kernel32.CloseHandle(h)

    matched_title: list[str] = []

    def enum_cb(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        cls_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls_buf, 256)
        # Music Center for PC main window class is Chrome_WidgetWin_1 (CEF wrapper)
        if cls_buf.value != "Chrome_WidgetWin_1":
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return True
        pname = get_proc_name(pid.value)
        if pname == "musiccenter.exe":
            title_buf = ctypes.create_unicode_buffer(1024)
            user32.GetWindowTextW(hwnd, title_buf, 1024)
            if title_buf.value:
                matched_title.append(title_buf.value)
                return False  # Stop enumeration
        return True

    proc = WNDENUMPROC(enum_cb)

    # Check Default desktop first (handles isolated desktop threads or services)
    DESKTOP_ENUMERATE = 0x0040
    DESKTOP_READOBJECTS = 0x0001
    h_desk = user32.OpenDesktopW("Default", 0, False, DESKTOP_ENUMERATE | DESKTOP_READOBJECTS)
    if h_desk:
        try:
            user32.EnumDesktopWindows(h_desk, proc, 0)
        finally:
            user32.CloseDesktop(h_desk)

    if not matched_title:
        user32.EnumWindows(proc, 0)

    return matched_title[0] if matched_title else None


class WindowsMusicCenterProvider(MetadataProvider):
    """Retrieves now-playing metadata from Sony Music Center for PC on Windows."""

    def __init__(
        self,
        name: str = "windows_music_center",
        priority: int = 40,
        title_finder: Callable[[], str | None] | None = None,
    ):
        self._name = name
        self._priority = priority
        self._title_finder = title_finder or default_find_music_center_title
        self._started = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def is_available(self) -> bool:
        return sys.platform == "win32"

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def get_now_playing(self) -> NowPlaying | None:
        if not self._started or not self.is_available():
            return None
        try:
            title = self._title_finder()
            if not title:
                return None
            return parse_music_center_title(title)
        except Exception:
            return None
