"""Exercise real Tk widgets without opening a user-visible window or audio."""
from queue import Queue
import unittest
from unittest.mock import patch

from wune.appearance import AppearanceDraft, AppearanceState
from wune.config import Config
from wune.settings_dialog import _Dialog


class DialogTests(unittest.TestCase):
    def test_led_controls_preview_independently_of_theme_selection(self):
        self.dialog.style("gauge_style", "box")
        self.dialog.style("led_shape", "ellipse")
        self.dialog.ratio.set("1.5")
        self.dialog.set_ratio()
        self.assertEqual(self.dialog.theme_name.get(), "CLASSIC")
        self.assertEqual(self.dialog.draft.state.user_presets, {})
        self.dialog.theme_name.set("BLUE")
        self.dialog.select_theme()
        self.assertEqual(self.dialog.ratio.get(), "1.5")
        self.assertEqual(self.dialog.draft.state.style["led_shape"], "ellipse")
        self.assertEqual(self.dialog.draft.state.user_presets, {})

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
        with patch("wune.localized_dialogs.ask_name", return_value="夜"):
            self.dialog.manage_theme("new")
        self.assertEqual(self.events.get_nowait()[1].preset.name, "夜")
        with patch("wune.localized_dialogs.ask_name", return_value="夜空"):
            self.dialog.manage_theme("rename")
        self.assertEqual(self.events.get_nowait()[1].preset.name, "夜空")

    def test_motion_slider_previews_without_creating_theme_copy(self):
        self.dialog.slide_motion("vis_attack_ms", "40")
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.motion["vis_attack_ms"], 40)
        self.assertEqual(state.preset.name, "CLASSIC")
        self.assertEqual(state.user_presets, {})
        self.assertEqual(self.dialog.motion_variables["vis_attack_ms"].get(), "40")

    def test_motion_numeric_save_commits_input_without_enter(self):
        self.dialog.motion_variables["vis_release_ms"].set("250.5")
        self.dialog.submit("save")
        action, state = self.events.get_nowait()
        self.assertEqual(action, "save")
        self.assertEqual(state.motion["vis_release_ms"], 250.5)

    def test_invalid_motion_blocks_save_but_not_cancel(self):
        self.dialog.motion_variables["peak_hold_ms"].set("nan")
        self.dialog.submit("save")
        self.assertFalse(self.dialog.pending)
        self.assertTrue(self.events.empty())
        self.dialog.submit("cancel")
        self.assertEqual(self.events.get_nowait()[0], "cancel")

    def test_reset_motion_keeps_theme_and_layout(self):
        self.dialog.theme_name.set("BLUE")
        self.dialog.select_theme()
        self.events.get_nowait()
        self.dialog.slide_motion("peak_fall_per_second", "8")
        self.events.get_nowait()
        self.dialog.slide_motion("peak_hold_ms", "200")
        self.assertEqual(self.events.get_nowait()[1].motion["peak_hold_ms"], 200)
        self.dialog.reset_motion()
        state = self.events.get_nowait()[1]
        self.assertEqual(state.preset.name, "BLUE")
        self.assertEqual(state.motion["peak_fall_per_second"], 2.5)
        self.assertEqual(state.motion["peak_hold_ms"], 500)
        self.assertEqual(float(self.dialog.motion_variables["peak_hold_ms"].get()), 500)

    def test_frequency_cap_preview_and_default_reset(self):
        self.dialog.limit_to_20khz.set(True)
        self.dialog.layout("limit_to_20khz", self.dialog.limit_to_20khz.get())
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertTrue(state.layout["limit_to_20khz"])
        self.dialog.reset()
        self.assertFalse(self.events.get_nowait()[1].layout["limit_to_20khz"])
        self.assertFalse(self.dialog.limit_to_20khz.get())
