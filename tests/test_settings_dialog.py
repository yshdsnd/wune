"""Exercise real Tk widgets without opening a user-visible window or audio."""
from queue import Queue
import unittest
from unittest.mock import MagicMock, patch

from wune.appearance import AppearanceDraft, AppearanceState
from wune.config import Config
from wune.settings_dialog import _Dialog, _focus_dialog


class DialogTests(unittest.TestCase):
    def test_channel_mode_preview_and_apply(self):
        self.assertEqual(len(self.dialog.notebook.tabs()), 5)
        for tab_id in self.dialog.notebook.tabs():
            tab_widget = self.dialog.notebook.nametowidget(tab_id)
            cells = {}
            for child in tab_widget.winfo_children():
                info = child.grid_info()
                if not info:
                    continue
                row = int(info["row"])
                col = int(info["column"])
                span = int(info.get("columnspan", 1))
                for c in range(col, col + span):
                    key = (row, c)
                    self.assertNotIn(key, cells, f"Grid collision at row {row}, col {c} between {child} and {cells.get(key)}")
                    cells[key] = child

        lang_combo, _ = self.dialog.combos["language"]
        self.assertEqual(int(lang_combo.grid_info()["row"]), 4)

        self.dialog.layout("channel_mode", "stereo_mix")
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.layout["channel_mode"], "stereo_mix")
        self.dialog.submit("apply")
        action, state = self.events.get_nowait()
        self.assertEqual(action, "apply")
        self.assertEqual(state.layout["channel_mode"], "stereo_mix")

    def test_background_picker_preview_reset_and_cancel(self):
        with patch("tkinter.filedialog.askopenfilename", return_value="C:/Pictures/example.png"):
            self.dialog.choose_background()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.background["background_mode"], "image")
        self.assertTrue(state.background["background_path"].endswith("example.png"))
        self.dialog.background("background_sizing", "fill")
        self.assertEqual(self.events.get_nowait()[1].background["background_sizing"], "fill")
        self.dialog.reset()
        self.assertEqual(self.events.get_nowait()[1].background["background_mode"], "solid")
        with patch("tkinter.filedialog.askopenfilename", return_value=""):
            self.dialog.choose_background()
        self.assertTrue(self.events.empty())
        self.dialog.submit("cancel")
        self.assertEqual(self.events.get_nowait()[0], "cancel")

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

    def test_exit_confirmation_can_be_disabled_and_reenabled_without_theme_copy(self):
        self.dialog.layout("confirm_keyboard_exit", False)
        self.assertFalse(self.events.get_nowait()[1].layout["confirm_keyboard_exit"])
        self.dialog.layout("confirm_keyboard_exit", True)
        self.assertTrue(self.events.get_nowait()[1].layout["confirm_keyboard_exit"])
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
        self.events, self.commands = Queue(), Queue()
        self.dialog = _Dialog(self.root, AppearanceDraft(AppearanceState.capture(Config(), "CLASSIC", {})),
                              "test/settings.json", self.events, self.commands)
        self.root.update_idletasks()

    def tearDown(self):
        if hasattr(self, "dialog") and self.dialog:
            try:
                self.dialog.close()
            except Exception:
                pass
        self.dialog = None
        if hasattr(self, "root") and self.root:
            try:
                self.root.update_idletasks()
            except Exception:
                pass
            try:
                self.root.destroy()
            except Exception:
                pass
            del self.root
            self.root = None
            import gc
            gc.collect()

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

    def test_invalid_label_font_size_blocks_save_without_discarding_input(self):
        self.dialog.label_font_size.set("99")
        self.dialog.submit("save")
        self.assertTrue(self.events.empty())
        self.assertFalse(self.dialog.pending)
        self.assertIn("10", self.dialog.status.get())

    def test_invalid_info_font_size_blocks_save_without_discarding_input(self):
        self.dialog.info_font_size.set("99")
        self.dialog.submit("save")
        self.assertTrue(self.events.empty())
        self.assertFalse(self.dialog.pending)
        self.assertIn("10", self.dialog.status.get())

    def test_invalid_leds_per_bar_blocks_save_without_discarding_input(self):
        self.dialog.leds_per_bar.set("999")
        self.dialog.submit("save")
        self.assertTrue(self.events.empty())
        self.assertFalse(self.dialog.pending)
        self.assertIn("10", self.dialog.status.get())

    def test_leds_per_bar_preview(self):
        self.dialog.leds_per_bar.set("45")
        self.dialog.set_leds_per_bar()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.style["leds_per_bar"], 45)

    def test_auto_adjust_leds_updates_spinbox_and_previews(self):
        # Default layout is vertical (2 channels stacked) in 1280x800
        self.dialog.auto_adjust_leds()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        optimal_vertical = int(self.dialog.leds_per_bar.get())
        self.assertEqual(state.style["leds_per_bar"], optimal_vertical)
        self.assertEqual(optimal_vertical, 24)
        self.assertIn("24", self.dialog.status.get())

        # Switch to horizontal layout (side-by-side L and R)
        self.dialog.layout("channel_layout", "horizontal")
        _ = self.events.get_nowait()  # consume layout preview
        self.dialog.auto_adjust_leds()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        optimal_horizontal = int(self.dialog.leds_per_bar.get())
        self.assertEqual(state.style["leds_per_bar"], optimal_horizontal)
        self.assertEqual(optimal_horizontal, 100)
        self.assertIn("100", self.dialog.status.get())

    def test_label_font_size_and_auto_scale_preview(self):
        self.dialog.label_font_size.set("18")
        self.dialog.set_label_font_size()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.layout["label_font_size"], 18)

        self.dialog.layout("auto_scale_fonts", False)
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.layout["auto_scale_fonts"], False)

    def test_info_font_size_preview(self):
        self.dialog.info_font_size.set("20")
        self.dialog.set_info_font_size()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.layout["info_font_size"], 20)

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

    def test_now_playing_preview_and_default_reset(self):
        self.dialog.show_now_playing.set(False)
        self.dialog.layout("show_now_playing", self.dialog.show_now_playing.get())
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertFalse(state.layout["show_now_playing"])
        self.dialog.reset()
        self.assertTrue(self.events.get_nowait()[1].layout["show_now_playing"])
        self.assertTrue(self.dialog.show_now_playing.get())

    def test_focus_dialog_skips_deiconify_when_normal(self):
        mock_root = MagicMock()
        mock_root.state.return_value = "normal"
        with patch("sys.platform", "win32"):
            _focus_dialog(mock_root)
        mock_root.deiconify.assert_not_called()
        mock_root.lift.assert_called_once()
        mock_root.focus_set.assert_called_once()
        mock_root.focus_force.assert_not_called()

    def test_focus_dialog_deiconifies_when_not_normal(self):
        mock_root = MagicMock()
        mock_root.state.return_value = "withdrawn"
        with patch("sys.platform", "win32"):
            _focus_dialog(mock_root)
        mock_root.deiconify.assert_called_once()
        mock_root.lift.assert_called_once()
        mock_root.focus_set.assert_called_once()

    def test_focus_dialog_darwin_forces_focus(self):
        mock_root = MagicMock()
        mock_root.state.return_value = "normal"
        with patch("sys.platform", "darwin"):
            _focus_dialog(mock_root)
        mock_root.deiconify.assert_not_called()
        mock_root.lift.assert_called_once()
        mock_root.focus_force.assert_called_once()
        mock_root.focus_set.assert_not_called()

    def test_deactivate_unposts_comboboxes_and_releases_grab(self):
        combo, _ = self.dialog.combos["language"]
        combo.tk.eval(f"ttk::combobox::Post {combo._w}")
        self.root.update_idletasks()
        popdown = combo.tk.eval(f"ttk::combobox::PopdownWindow {combo._w}")
        self.assertEqual(self.root.tk.eval(f"winfo ismapped {popdown}"), "1")
        self.dialog._on_deactivate(MagicMock(widget=self.root))
        self.assertEqual(self.root.tk.eval(f"winfo ismapped {popdown}"), "0")
        self.assertEqual(self.root.tk.eval("grab current"), "")



