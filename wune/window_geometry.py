"""Restore windowed geometry into a connected display's usable area."""
import sys
import warnings
from .layout import fit_window_size


def work_areas():
    if sys.platform == "win32":
        import ctypes as ct
        from ctypes import wintypes as wt

        class MonitorInfo(ct.Structure):
            _fields_ = [("size", wt.DWORD), ("monitor", wt.RECT), ("work", wt.RECT), ("flags", wt.DWORD)]

        user32 = ct.WinDLL("user32", use_last_error=True)
        callback_type = ct.WINFUNCTYPE(wt.BOOL, wt.HANDLE, wt.HDC, ct.POINTER(wt.RECT), wt.LPARAM)
        user32.GetMonitorInfoW.argtypes = [wt.HANDLE, ct.POINTER(MonitorInfo)]
        user32.GetMonitorInfoW.restype = wt.BOOL
        user32.EnumDisplayMonitors.argtypes = [wt.HDC, ct.POINTER(wt.RECT), callback_type, wt.LPARAM]
        user32.EnumDisplayMonitors.restype = wt.BOOL
        found = []

        @callback_type
        def visit(monitor, dc, rect, data):
            info = MonitorInfo()
            info.size = ct.sizeof(info)
            if user32.GetMonitorInfoW(monitor, ct.byref(info)):
                r = info.work
                found.append((bool(info.flags & 1), (r.left, r.top, r.right-r.left, r.bottom-r.top)))
            return True

        if user32.EnumDisplayMonitors(None, None, visit, 0) and found:
            return [rect for primary, rect in sorted(found, key=lambda item: not item[0])]
        warnings.warn("Monitor detection failed; using the primary desktop size", RuntimeWarning)
    import pygame as pg
    width, height = pg.display.get_desktop_sizes()[0]
    return [(0, 0, width, height)]


def restore_geometry(cfg, saved, areas):
    width, height = saved.get("width", cfg.width), saved.get("height", cfg.height)
    x, y = saved.get("x"), saved.get("y")

    def overlap(area):
        if x is None or y is None:
            return 0
        ax, ay, aw, ah = area
        return max(0, min(x+width, ax+aw)-max(x, ax)) * max(0, min(y+height, ay+ah)-max(y, ay))

    area = max(areas, key=overlap)
    ax, ay, aw, ah = area
    # Keep room for native borders and a reachable title bar.
    left, top, available_w, available_h = ax+16, ay+40, max(1, aw-32), max(1, ah-56)
    size = fit_window_size((min(width, available_w), min(height, available_h)), cfg)
    if x is None or y is None or overlap(area) == 0:
        x = left + max(0, (available_w-size[0])//2)
        y = top + max(0, (available_h-size[1])//2)
    position = (max(left, min(x, left+max(0, available_w-size[0]))),
                max(top, min(y, top+max(0, available_h-size[1]))))
    return size, position
