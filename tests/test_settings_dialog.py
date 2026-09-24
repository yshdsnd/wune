"""Exercise real Tk widgets without opening a user-visible window or audio."""
from queue import Queue
import unittest
from unittest.mock import patch

from wune.appearance import AppearanceDraft, AppearanceState
from wune.config import Config
from wune.settings_dialog import _Dialog


class DialogTests(unittest.TestCase):
    def setUp(self):
        try:
            import tkinter as tk
        except ImportError:
            self.skipTest("Tk is not installed")
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"No Tk display: {error}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.events, self.commands = Queue(), Queue()
        self.dialog = _Dialog(self.root, AppearanceDraft(AppearanceState.capture(Config(), "CLASSIC", {})),
                              "test/settings.json", self.events, self.commands)
        self.root.update_idletasks()

    def test_color_picker_previews_and_creates_user_copy(self):
        with patch("tkinter.colorchooser.askcolor", return_value=((12, 34, 56), "#0c2238")):
            self.dialog.pick_color()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.preset.theme.green_on, (12, 34, 56))
        self.assertEqual(state.preset.name, "CLASSIC copy")
        self.assertEqual(self.dialog.theme_name.get(), "CLASSIC copy")

    def test_invalid_ratio_blocks_save_without_discarding_input(self):
        self.dialog.ratio.set("nan")
        self.dialog.submit("save")
        self.assertTrue(self.events.empty())
        self.assertFalse(self.dialog.pending)
        self.assertIn("0.25", self.dialog.status.get())

    def test_apply_acknowledgment_reenables_controls(self):
        self.dialog.submit("apply")
        self.assertTrue(self.dialog.pending)
        self.assertEqual(self.events.get_nowait()[0], "apply")
        self.commands.put(("reply", (True, "Applied", False)))
        self.dialog.poll()
        self.assertFalse(self.dialog.pending)
        self.assertEqual(self.dialog.status.get(), "Applied")

    def test_reset_previews_and_cancel_is_allowed_with_invalid_input(self):
        self.dialog.theme_name.set("BLUE")
        self.dialog.select_theme()
        self.events.get_nowait()
        self.dialog.reset()
        self.assertEqual(self.events.get_nowait()[1].preset.name, "CLASSIC")
        self.dialog.ratio.set("invalid")
        self.dialog.submit("cancel")
        self.assertEqual(self.events.get_nowait()[0], "cancel")

    def test_hex_error_does_not_reach_renderer(self):
        self.dialog.hex_color.set("#12345Z")
        self.dialog.set_hex()
        self.assertTrue(self.events.empty())

    def test_create_and_rename_via_dialog(self):
        with patch("tkinter.simpledialog.askstring", return_value="夜"):
            self.dialog.manage_theme("new")
        self.assertEqual(self.events.get_nowait()[1].preset.name, "夜")
        with patch("tkinter.simpledialog.askstring", return_value="夜空"):
            self.dialog.manage_theme("rename")
        self.assertEqual(self.events.get_nowait()[1].preset.name, "夜空")
