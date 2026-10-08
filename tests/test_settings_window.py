"""Exercise actual Win32 ownership with hidden windows (no audio/display takeover)."""
import ctypes as ct
from ctypes import wintypes as wt
import sys
import unittest

from wune.settings_window import SettingsWindow


@unittest.skipUnless(sys.platform == 'win32', 'Windows ownership')
class SettingsWindowTests(unittest.TestCase):
    def test_settings_thread_is_hidden_until_native_owner_is_ready(self):
        # Other tests construct Tk on the unittest thread. Keep their Tcl objects
        # out of the application's dedicated Tk thread and its final GC pass.
        import subprocess
        subprocess.run([sys.executable, "-c", """
import ctypes as ct
from ctypes import wintypes as wt
from wune.settings_dialog import SettingsDialog
from wune.settings_window import SettingsWindow
from wune.appearance import AppearanceState
from wune.config import Config
s = SettingsDialog(AppearanceState.capture(Config(), 'CLASSIC', {}), 'test.json')
try:
    action, handle = s.events.get(timeout=10)
    assert action == 'ready', (action, handle)
    native = SettingsWindow(handle)
    native.api.IsWindowVisible.argtypes = [wt.HWND]
    native.api.IsWindowVisible.restype = wt.BOOL
    assert not native.api.IsWindowVisible(native.handle)
finally:
    s.close()
assert not s.thread.is_alive()
"""], check=True, timeout=20)

    def test_native_owner_detach_and_rebind_without_topmost(self):
        import subprocess
        subprocess.run([sys.executable, "-c", """
import ctypes as ct
from ctypes import wintypes as wt
import tkinter as tk
from wune.settings_window import SettingsWindow

owner = tk.Tk()
owner.withdraw()
child = tk.Toplevel(owner)
child.withdraw()
child.geometry('600x600')
owner.update_idletasks()
native = SettingsWindow(child.winfo_id())
api = native.api
parent = api.GetAncestor(owner.winfo_id(), 2)
api.GetWindow.argtypes = [wt.HWND, wt.UINT]
api.GetWindow.restype = wt.HWND
api.GetWindowLongW.argtypes = [wt.HWND, ct.c_int]
api.GetWindowLongW.restype = wt.LONG
try:
    native.bind(parent)
    native.position(parent)
    assert api.GetWindow(native.handle, 4) == parent  # GW_OWNER
    assert not (api.GetWindowLongW(native.handle, -20) & 8)  # WS_EX_TOPMOST
    native.detach()
    assert not api.GetWindow(native.handle, 4)
    native.bind(parent)
    assert api.GetWindow(native.handle, 4) == parent
    native.detach()
    child.destroy()
    # Delayed close/recreate handling tolerates an already destroyed HWND.
    native.bind(parent)
finally:
    native.detach()
    owner.destroy()
"""], check=True, timeout=20)

    def test_bind_detaches_cross_thread_input_queue(self):
        from unittest.mock import MagicMock
        native = SettingsWindow.__new__(SettingsWindow)
        native.api = MagicMock()
        native.handle = 123
        native.set_owner = MagicMock(return_value=1)
        native.api.IsWindow.return_value = True
        def mock_get_thread(hwnd, byref_pid):
            return 1001 if hwnd == 456 else 2002
        native.api.GetWindowThreadProcessId.side_effect = mock_get_thread
        native.bind(456)
        native.api.AttachThreadInput.assert_called_once_with(2002, 1001, False)
