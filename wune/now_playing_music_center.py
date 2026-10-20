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


def parse_music_center_title(raw_title: str, is_playing: bool = True) -> NowPlaying | None:
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
        is_playing=is_playing,
        raw_text=cleaned,
    )


if sys.platform == "win32":
    import ctypes
    from ctypes import (
        HRESULT,
        POINTER,
        Structure,
        byref,
        c_uint32,
        c_void_p,
        cast,
        wintypes,
    )

    _ole32 = ctypes.windll.ole32
    _kernel32 = ctypes.windll.kernel32
    _user32 = ctypes.windll.user32

    class _GUID(Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", wintypes.BYTE * 8),
        ]

        def __init__(self, guid_str: str):
            super().__init__()
            _ole32.CLSIDFromString(ctypes.c_wchar_p(guid_str), byref(self))

    _CLSID_MMDeviceEnumerator = _GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
    _IID_IMMDeviceEnumerator = _GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
    _IID_IAudioSessionManager2 = _GUID("{77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F}")
    _IID_IAudioSessionControl2 = _GUID("{bfb7ff88-7239-4fc9-8fa2-07c950be9c6d}")
    _CLSCTX_ALL = 23

    class _IUnknownVtbl(Structure):
        _fields_ = [
            ("QueryInterface", c_void_p),
            ("AddRef", c_void_p),
            ("Release", ctypes.WINFUNCTYPE(c_uint32, c_void_p)),
        ]

    def _release_com(ptr: c_void_p) -> None:
        if ptr:
            try:
                cast(ptr, POINTER(POINTER(_IUnknownVtbl))).contents.contents.Release(ptr)
            except Exception:
                pass

    class _IMMDeviceEnumeratorVtbl(Structure):
        _fields_ = [
            ("QueryInterface", c_void_p),
            ("AddRef", c_void_p),
            ("Release", ctypes.WINFUNCTYPE(c_uint32, c_void_p)),
            (
                "EnumAudioEndpoints",
                ctypes.WINFUNCTYPE(
                    HRESULT, c_void_p, wintypes.DWORD, wintypes.DWORD, POINTER(c_void_p)
                ),
            ),
            (
                "GetDefaultAudioEndpoint",
                ctypes.WINFUNCTYPE(
                    HRESULT, c_void_p, wintypes.DWORD, wintypes.DWORD, POINTER(c_void_p)
                ),
            ),
        ]

    class _IMMDeviceCollectionVtbl(Structure):
        _fields_ = [
            ("QueryInterface", c_void_p),
            ("AddRef", c_void_p),
            ("Release", ctypes.WINFUNCTYPE(c_uint32, c_void_p)),
            ("GetCount", ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(c_uint32))),
            ("Item", ctypes.WINFUNCTYPE(HRESULT, c_void_p, c_uint32, POINTER(c_void_p))),
        ]

    class _IMMDeviceVtbl(Structure):
        _fields_ = [
            ("QueryInterface", c_void_p),
            ("AddRef", c_void_p),
            ("Release", ctypes.WINFUNCTYPE(c_uint32, c_void_p)),
            (
                "Activate",
                ctypes.WINFUNCTYPE(
                    HRESULT,
                    c_void_p,
                    POINTER(_GUID),
                    wintypes.DWORD,
                    c_void_p,
                    POINTER(c_void_p),
                ),
            ),
        ]

    class _IAudioSessionManager2Vtbl(Structure):
        _fields_ = [
            ("QueryInterface", c_void_p),
            ("AddRef", c_void_p),
            ("Release", ctypes.WINFUNCTYPE(c_uint32, c_void_p)),
            ("GetAudioSessionControl", c_void_p),
            ("GetSimpleAudioVolume", c_void_p),
            (
                "GetSessionEnumerator",
                ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(c_void_p)),
            ),
        ]

    class _IAudioSessionEnumeratorVtbl(Structure):
        _fields_ = [
            ("QueryInterface", c_void_p),
            ("AddRef", c_void_p),
            ("Release", ctypes.WINFUNCTYPE(c_uint32, c_void_p)),
            ("GetCount", ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(c_uint32))),
            (
                "GetSession",
                ctypes.WINFUNCTYPE(HRESULT, c_void_p, c_uint32, POINTER(c_void_p)),
            ),
        ]

    class _IAudioSessionControl2Vtbl(Structure):
        _fields_ = [
            (
                "QueryInterface",
                ctypes.WINFUNCTYPE(
                    HRESULT, c_void_p, POINTER(_GUID), POINTER(c_void_p)
                ),
            ),
            ("AddRef", c_void_p),
            ("Release", ctypes.WINFUNCTYPE(c_uint32, c_void_p)),
            ("GetState", ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(wintypes.DWORD))),
            ("GetDisplayName", c_void_p),
            ("SetDisplayName", c_void_p),
            ("GetIconPath", c_void_p),
            ("SetIconPath", c_void_p),
            ("GetGroupingParam", c_void_p),
            ("SetGroupingParam", c_void_p),
            ("RegisterAudioSessionNotification", c_void_p),
            ("UnregisterAudioSessionNotification", c_void_p),
            ("GetSessionIdentifier", c_void_p),
            ("GetSessionInstanceIdentifier", c_void_p),
            (
                "GetProcessId",
                ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(wintypes.DWORD)),
            ),
        ]

    def _get_process_image_name(pid: int) -> str:
        if not pid:
            return ""
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if _kernel32.QueryFullProcessImageNameW(h, 0, buf, byref(size)):
                return buf.value.rsplit("\\", 1)[-1].lower()
            return ""
        finally:
            _kernel32.CloseHandle(h)
else:
    def _get_process_image_name(pid: int) -> str:
        return ""


def default_find_music_center_title() -> str | None:
    """Find the visible top-level window title of Music Center for PC."""
    if sys.platform != "win32":
        return None

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    matched_title: list[str] = []

    def enum_cb(hwnd, lparam):
        if not _user32.IsWindowVisible(hwnd):
            return True
        cls_buf = ctypes.create_unicode_buffer(256)
        _user32.GetClassNameW(hwnd, cls_buf, 256)
        # Music Center for PC main window class is Chrome_WidgetWin_1 (CEF wrapper)
        if cls_buf.value != "Chrome_WidgetWin_1":
            return True
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return True
        pname = _get_process_image_name(pid.value)
        if pname == "musiccenter.exe":
            title_buf = ctypes.create_unicode_buffer(1024)
            _user32.GetWindowTextW(hwnd, title_buf, 1024)
            if title_buf.value:
                matched_title.append(title_buf.value)
                return False  # Stop enumeration
        return True

    proc = WNDENUMPROC(enum_cb)

    # Check Default desktop first (handles isolated desktop threads or services)
    DESKTOP_ENUMERATE = 0x0040
    DESKTOP_READOBJECTS = 0x0001
    h_desk = _user32.OpenDesktopW("Default", 0, False, DESKTOP_ENUMERATE | DESKTOP_READOBJECTS)
    if h_desk:
        try:
            _user32.EnumDesktopWindows(h_desk, proc, 0)
        finally:
            _user32.CloseDesktop(h_desk)

    if not matched_title:
        _user32.EnumWindows(proc, 0)

    return matched_title[0] if matched_title else None


def default_is_music_center_playing() -> bool:
    """Check if Sony Music Center for PC has an actively playing audio session."""
    if sys.platform != "win32":
        return False

    hr = _ole32.CoInitialize(None)
    need_uninit = hr in (0, 1)
    if hr < 0 and (hr & 0xFFFFFFFF) != 0x80010106:
        return False

    try:
        p_enum = c_void_p()
        if (
            _ole32.CoCreateInstance(
                byref(_CLSID_MMDeviceEnumerator),
                None,
                _CLSCTX_ALL,
                byref(_IID_IMMDeviceEnumerator),
                byref(p_enum),
            )
            != 0
            or not p_enum
        ):
            return False

        try:
            vtbl_enum = cast(
                p_enum, POINTER(POINTER(_IMMDeviceEnumeratorVtbl))
            ).contents.contents
            p_col = c_void_p()
            # eRender = 0, DEVICE_STATE_ACTIVE = 1
            if vtbl_enum.EnumAudioEndpoints(p_enum, 0, 1, byref(p_col)) != 0 or not p_col:
                return False

            try:
                vtbl_col = cast(
                    p_col, POINTER(POINTER(_IMMDeviceCollectionVtbl))
                ).contents.contents
                dev_count = c_uint32(0)
                vtbl_col.GetCount(p_col, byref(dev_count))

                for d_idx in range(dev_count.value):
                    p_dev = c_void_p()
                    if vtbl_col.Item(p_col, d_idx, byref(p_dev)) != 0 or not p_dev:
                        continue
                    try:
                        vtbl_dev = cast(
                            p_dev, POINTER(POINTER(_IMMDeviceVtbl))
                        ).contents.contents
                        p_mgr = c_void_p()
                        if (
                            vtbl_dev.Activate(
                                p_dev,
                                byref(_IID_IAudioSessionManager2),
                                _CLSCTX_ALL,
                                None,
                                byref(p_mgr),
                            )
                            != 0
                            or not p_mgr
                        ):
                            continue
                        try:
                            vtbl_mgr = cast(
                                p_mgr, POINTER(POINTER(_IAudioSessionManager2Vtbl))
                            ).contents.contents
                            p_sess_enum = c_void_p()
                            if (
                                vtbl_mgr.GetSessionEnumerator(p_mgr, byref(p_sess_enum))
                                != 0
                                or not p_sess_enum
                            ):
                                continue
                            try:
                                vtbl_sess_enum = cast(
                                    p_sess_enum,
                                    POINTER(POINTER(_IAudioSessionEnumeratorVtbl)),
                                ).contents.contents
                                sess_count = c_uint32(0)
                                vtbl_sess_enum.GetCount(p_sess_enum, byref(sess_count))

                                for s_idx in range(sess_count.value):
                                    p_ctrl = c_void_p()
                                    if (
                                        vtbl_sess_enum.GetSession(
                                            p_sess_enum, s_idx, byref(p_ctrl)
                                        )
                                        != 0
                                        or not p_ctrl
                                    ):
                                        continue
                                    try:
                                        p_ctrl2 = c_void_p()
                                        qi = cast(
                                            p_ctrl,
                                            POINTER(POINTER(_IAudioSessionControl2Vtbl)),
                                        ).contents.contents.QueryInterface
                                        if (
                                            qi(
                                                p_ctrl,
                                                byref(_IID_IAudioSessionControl2),
                                                byref(p_ctrl2),
                                            )
                                            == 0
                                            and p_ctrl2
                                        ):
                                            try:
                                                vtbl_ctrl2 = cast(
                                                    p_ctrl2,
                                                    POINTER(
                                                        POINTER(_IAudioSessionControl2Vtbl)
                                                    ),
                                                ).contents.contents
                                                pid = wintypes.DWORD(0)
                                                vtbl_ctrl2.GetProcessId(
                                                    p_ctrl2, byref(pid)
                                                )
                                                if (
                                                    pid.value
                                                    and _get_process_image_name(pid.value)
                                                    == "musiccenter.exe"
                                                ):
                                                    state = wintypes.DWORD(0)
                                                    vtbl_ctrl2.GetState(
                                                        p_ctrl2, byref(state)
                                                    )
                                                    if state.value == 1:  # AudioSessionStateActive
                                                        return True
                                            finally:
                                                _release_com(p_ctrl2)
                                    finally:
                                        _release_com(p_ctrl)
                            finally:
                                _release_com(p_sess_enum)
                        finally:
                            _release_com(p_mgr)
                    finally:
                        _release_com(p_dev)
            finally:
                _release_com(p_col)
        finally:
            _release_com(p_enum)
    except Exception:
        return False
    finally:
        if need_uninit:
            _ole32.CoUninitialize()

    return False


class WindowsMusicCenterProvider(MetadataProvider):
    """Retrieves now-playing metadata from Sony Music Center for PC on Windows."""

    def __init__(
        self,
        name: str = "windows_music_center",
        priority: int = 40,
        title_finder: Callable[[], str | None] | None = None,
        playback_checker: Callable[[], bool] | None = None,
    ):
        self._name = name
        self._priority = priority
        self._title_finder = title_finder or default_find_music_center_title
        self._playback_checker = playback_checker or default_is_music_center_playing
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
            is_playing = False
            if self._playback_checker is not None:
                try:
                    is_playing = bool(self._playback_checker())
                except Exception:
                    is_playing = False
            return parse_music_center_title(title, is_playing=is_playing)
        except Exception:
            return None
