"""Identify connected Windows monitors without persisting transient display indices."""
from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class Monitor:
    identity: str | None
    bounds: tuple
    handle: int


def valid_identity(value):
    return isinstance(value, str) and 0 < len(value) <= 1024 and '\x00' not in value


def matching_monitor(identity, monitors):
    if not valid_identity(identity):
        return None
    matches = [m for m in monitors if m.identity and m.identity.casefold() == identity.casefold()]
    return matches[0] if len(matches) == 1 else None


def connected_monitors():
    if sys.platform != 'win32':
        return []
    import ctypes as ct
    from ctypes import wintypes as wt

    class MonitorInfo(ct.Structure):
        _fields_ = [('size', wt.DWORD), ('monitor', wt.RECT), ('work', wt.RECT),
                    ('flags', wt.DWORD), ('device', wt.WCHAR * 32)]

    class DisplayDevice(ct.Structure):
        _fields_ = [('size', wt.DWORD), ('name', wt.WCHAR * 32),
                    ('description', wt.WCHAR * 128), ('flags', wt.DWORD),
                    ('identity', wt.WCHAR * 128), ('key', wt.WCHAR * 128)]

    api = ct.WinDLL('user32', use_last_error=True)
    callback_type = ct.WINFUNCTYPE(wt.BOOL, wt.HANDLE, wt.HDC, ct.POINTER(wt.RECT), wt.LPARAM)
    api.EnumDisplayMonitors.argtypes = [wt.HDC, ct.POINTER(wt.RECT), callback_type, wt.LPARAM]
    api.EnumDisplayMonitors.restype = wt.BOOL
    api.GetMonitorInfoW.argtypes = [wt.HANDLE, ct.POINTER(MonitorInfo)]
    api.GetMonitorInfoW.restype = wt.BOOL
    api.EnumDisplayDevicesW.argtypes = [wt.LPCWSTR, wt.DWORD, ct.POINTER(DisplayDevice), wt.DWORD]
    api.EnumDisplayDevicesW.restype = wt.BOOL
    found = []

    @callback_type
    def visit(handle, dc, rect, data):
        info = MonitorInfo()
        info.size = ct.sizeof(info)
        if api.GetMonitorInfoW(handle, ct.byref(info)):
            identities = []
            index = 0
            while True:
                device = DisplayDevice()
                device.size = ct.sizeof(device)
                # EDD_GET_DEVICE_INTERFACE_NAME: per-monitor interface path,
                # not the adapter's ordinal (e.g. DISPLAY1) or friendly name.
                if not api.EnumDisplayDevicesW(info.device, index, ct.byref(device), 1):
                    break
                if device.flags & 1 and not device.flags & 8:  # active, not mirroring driver
                    identities.append(device.identity if valid_identity(device.identity) else None)
                index += 1
            identity = identities[0] if len(identities) == 1 else None
            r = info.monitor
            found.append(Monitor(identity, (r.left, r.top, r.right-r.left, r.bottom-r.top), handle))
        return True

    if not api.EnumDisplayMonitors(None, None, visit, 0):
        return []
    return found


def current_monitor_identity(hwnd):
    if sys.platform != 'win32' or not hwnd:
        return None
    import ctypes as ct
    from ctypes import wintypes as wt
    api = ct.WinDLL('user32', use_last_error=True)
    api.MonitorFromWindow.argtypes = [wt.HWND, wt.DWORD]
    api.MonitorFromWindow.restype = wt.HANDLE
    handle = api.MonitorFromWindow(hwnd, 0)  # No nearest-monitor fallback.
    monitors = connected_monitors()
    matches = [m for m in monitors if m.handle == handle]
    if len(matches) == 1 and matching_monitor(matches[0].identity, monitors) is not None:
        return matches[0].identity
    return None
