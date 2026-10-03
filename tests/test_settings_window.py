"""Exercise actual Win32 ownership with hidden windows (no audio/display takeover)."""
import ctypes as ct
from ctypes import wintypes as wt
import sys
import unittest

from wune.settings_window import SettingsWindow


@unittest.skipUnless(sys.platform == 'win32', 'Windows ownership')
class SettingsWindowTests(unittest.TestCase):
    def test_settings_thread_is_hidden_until_native_owner_is_ready(self):
        from wune.settings_dialog import SettingsDialog
        from wune.appearance import AppearanceState
        from wune.config import Config
        dialog = SettingsDialog(AppearanceState.capture(Config(), "CLASSIC", {}), "test.json")
        try:
            action, handle = dialog.events.get(timeout=10)
            self.assertEqual(action, "ready")
            native = SettingsWindow(handle)
            native.api.IsWindowVisible.argtypes = [wt.HWND]
            native.api.IsWindowVisible.restype = wt.BOOL
            self.assertFalse(native.api.IsWindowVisible(native.handle))
        finally:
            dialog.close()
        self.assertFalse(dialog.thread.is_alive())

    def test_native_owner_detach_and_rebind_without_topmost(self):
        import tkinter as tk
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
            self.assertEqual(api.GetWindow(native.handle, 4), parent)  # GW_OWNER
            self.assertFalse(api.GetWindowLongW(native.handle, -20) & 8)  # WS_EX_TOPMOST
            native.detach()
            self.assertFalse(api.GetWindow(native.handle, 4))
            native.bind(parent)
            self.assertEqual(api.GetWindow(native.handle, 4), parent)
            native.detach()
            child.destroy()
            # Delayed close/recreate handling tolerates an already destroyed HWND.
            native.bind(parent)
        finally:
            native.detach()
            owner.destroy()
