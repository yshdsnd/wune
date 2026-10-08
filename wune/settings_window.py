"""Win32 ownership for the independent Tk settings window.

Only native integer handles cross the Tk/pygame thread boundary. Ownership
keeps settings above Wune without making either window globally topmost.
"""
import ctypes as ct
from ctypes import wintypes as wt
import sys


class SettingsWindow:
    def __init__(self, widget_handle):
        self.handle = None
        self.api = None
        if sys.platform != 'win32':
            return
        api = ct.WinDLL('user32', use_last_error=True)
        api.GetAncestor.argtypes = [wt.HWND, wt.UINT]
        api.GetAncestor.restype = wt.HWND
        api.IsWindow.argtypes = [wt.HWND]
        api.IsWindow.restype = wt.BOOL
        api.GetWindowRect.argtypes = [wt.HWND, ct.POINTER(wt.RECT)]
        api.GetWindowRect.restype = wt.BOOL
        api.MonitorFromWindow.argtypes = [wt.HWND, wt.DWORD]
        api.MonitorFromWindow.restype = wt.HANDLE
        api.GetMonitorInfoW.argtypes = [wt.HANDLE, ct.c_void_p]
        api.GetMonitorInfoW.restype = wt.BOOL
        api.SetWindowPos.argtypes = [wt.HWND, wt.HWND, ct.c_int, ct.c_int,
                                     ct.c_int, ct.c_int, wt.UINT]
        api.SetWindowPos.restype = wt.BOOL
        api.GetWindowThreadProcessId.argtypes = [wt.HWND, ct.POINTER(wt.DWORD)]
        api.GetWindowThreadProcessId.restype = wt.DWORD
        api.AttachThreadInput.argtypes = [wt.DWORD, wt.DWORD, wt.BOOL]
        api.AttachThreadInput.restype = wt.BOOL
        self.set_owner = api.SetWindowLongPtrW if ct.sizeof(ct.c_void_p) == 8 else api.SetWindowLongW
        self.set_owner.argtypes = [wt.HWND, ct.c_int, ct.c_ssize_t]
        self.set_owner.restype = ct.c_ssize_t
        self.api = api
        # winfo_id is the Tk client; ownership belongs to its native wrapper.
        self.handle = api.GetAncestor(widget_handle, 2)  # GA_ROOT
        if not self.handle:
            raise ct.WinError(ct.get_last_error())

    def bind(self, owner):
        if self.api is None or not self.api.IsWindow(self.handle):
            return
        ct.set_last_error(0)
        previous = self.set_owner(self.handle, -8, int(owner or 0))  # GWLP_HWNDPARENT
        error = ct.get_last_error()
        if previous == 0 and error and error != 1400:
            raise ct.WinError(error)
        if owner and self.api.IsWindow(owner):
            pid_owner = wt.DWORD()
            tid_owner = self.api.GetWindowThreadProcessId(owner, ct.byref(pid_owner))
            pid_child = wt.DWORD()
            tid_child = self.api.GetWindowThreadProcessId(self.handle, ct.byref(pid_child))
            if tid_owner and tid_child and tid_owner != tid_child:
                # GWLP_HWNDPARENT implicitly attaches thread input queues across threads,
                # which deadlocks Tk and pygame during external focus loss (e.g. Snipping Tool).
                # Explicitly detach the input queues while preserving native window ownership.
                self.api.AttachThreadInput(tid_child, tid_owner, False)

    def detach(self):
        if self.api is not None and self.api.IsWindow(self.handle):
            self.bind(None)

    def position(self, owner):
        if self.api is None:
            return
        class MonitorInfo(ct.Structure):
            _fields_ = [('size', wt.DWORD), ('monitor', wt.RECT),
                        ('work', wt.RECT), ('flags', wt.DWORD)]
        parent, child = wt.RECT(), wt.RECT()
        info = MonitorInfo()
        info.size = ct.sizeof(info)
        monitor = self.api.MonitorFromWindow(owner, 2)
        if not (self.api.GetWindowRect(owner, ct.byref(parent)) and
                self.api.GetWindowRect(self.handle, ct.byref(child)) and
                self.api.GetMonitorInfoW(monitor, ct.byref(info))):
            raise ct.WinError(ct.get_last_error())
        width, height = child.right - child.left, child.bottom - child.top
        x = max(info.work.left, min((parent.left + parent.right - width) // 2,
                                    info.work.right - width))
        y = max(info.work.top, min((parent.top + parent.bottom - height) // 2,
                                   info.work.bottom - height))
        # Move only; initial focus/show is requested on the Tk thread.
        if not self.api.SetWindowPos(self.handle, None, x, y, 0, 0, 0x0015):
            raise ct.WinError(ct.get_last_error())
