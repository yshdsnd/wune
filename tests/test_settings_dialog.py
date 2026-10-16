"""Exercise real Tk widgets without opening a user-visible window or audio."""
from queue import Queue
import unittest
from unittest.mock import MagicMock, patch

from wune.appearance import AppearanceDraft, AppearanceState
from wune.config import Config
from wune.settings_dialog import _Dialog, _focus_dialog


class DialogTests(unittest.TestCase):
    def test_channel_mode_preview_and_apply(self):
        self.assertEqual(len(self.dialog.notebook.tabs()), 6)
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
        self.assertEqual(int(lang_combo.grid_info()["row"]), 1)

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

    def test_display_gain_preview_and_validation(self):
        self.dialog.display_gain.set("6.0")
        self.dialog.set_display_gain()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.style["display_gain_db"], 6.0)

        # Invalid value in set_display_gain shows error
        self.dialog.display_gain.set("30.0")
        self.dialog.set_display_gain()
        self.assertTrue(self.events.empty())
        self.assertIn("24", self.dialog.status.get())

        # Invalid value on submit blocks save
        self.dialog.submit("save")
        self.assertTrue(self.events.empty())
        self.assertFalse(self.dialog.pending)

        # Valid value applies correctly
        self.dialog.display_gain.set("-12.0")
        self.dialog.submit("apply")
        action, state = self.events.get_nowait()
        self.assertEqual(action, "apply")
        self.assertEqual(state.style["display_gain_db"], -12.0)

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

    def test_invalid_bar_gap_blocks_save_without_discarding_input(self):
        self.dialog.bar_gap.set("99")
        self.dialog.submit("save")
        self.assertTrue(self.events.empty())
        self.assertFalse(self.dialog.pending)
        self.assertIn("0", self.dialog.status.get())

    def test_bar_gap_preview(self):
        self.dialog.bar_gap.set("8")
        self.dialog.set_bar_gap()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.style["bar_gap"], 8)

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

    def test_auto_adjust_leds_on_resize_toggle_and_triggers(self):
        # Toggle on triggers immediate auto-adjust
        self.dialog.auto_adjust_leds_on_resize.set(True)
        self.dialog.toggle_auto_adjust_leds_on_resize()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertTrue(state.style["auto_adjust_leds_on_resize"])
        self.assertEqual(state.style["leds_per_bar"], 24)

        # Layout change (channel_layout to horizontal) automatically runs auto-adjust when enabled
        self.dialog.layout("channel_layout", "horizontal")
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.style["leds_per_bar"], 100)
        self.assertEqual(int(self.dialog.leds_per_bar.get()), 100)

        # If leds_per_bar was manually changed while auto_adjust_leds_on_resize is active,
        # receiving a window_size event will re-adjust it to optimal
        self.dialog.leds_per_bar.set("50")
        self.dialog.set_leds_per_bar()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.style["leds_per_bar"], 50)

        self.commands.put(("window_size", (1280, 800)))
        self.dialog.poll()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.style["leds_per_bar"], 100)
        self.assertEqual(int(self.dialog.leds_per_bar.get()), 100)

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

    def test_info_font_family_preview(self):
        self.dialog.info_font_family.set("Noto Sans CJK JP")
        self.dialog.set_info_font_family()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.layout["info_font_family"], "Noto Sans CJK JP")

        # System default option sets empty string
        self.dialog.info_font_family.set(self.dialog.t("settings.font_system_default"))
        self.dialog.set_info_font_family()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.layout["info_font_family"], "")

    def test_japanese_font_filter_toggle(self):
        self.dialog.all_font_families = ["Arial", "Courier New", "Noto Sans CJK JP", "Meiryo"]
        self.dialog.lang_font_families = ["Noto Sans CJK JP", "Meiryo"]
        self.dialog.ja_font_families = self.dialog.lang_font_families

        # When filter is enabled, only Japanese fonts (+ system default) are displayed
        self.dialog.filter_lang_fonts.set(True)
        self.dialog._update_font_combo_choices()
        values_filtered = list(self.dialog.font_combo.cget("values"))
        self.assertIn("Noto Sans CJK JP", values_filtered)
        self.assertIn("Meiryo", values_filtered)
        self.assertNotIn("Arial", values_filtered)
        self.assertNotIn("Courier New", values_filtered)

        # When filter is disabled, all fonts are displayed
        self.dialog.filter_lang_fonts.set(False)
        self.dialog._update_font_combo_choices()
        values_all = list(self.dialog.font_combo.cget("values"))
        self.assertIn("Arial", values_all)
        self.assertIn("Courier New", values_all)
        self.assertIn("Noto Sans CJK JP", values_all)

    def test_language_filter_check_visibility_by_language(self):
        # In English mode, font filter checkbox should not be displayed
        cfg_en = Config(language="en")
        draft_en = AppearanceDraft(AppearanceState.capture(cfg_en, "CLASSIC", {}))
        events_en, commands_en = Queue(), Queue()
        dialog_en = _Dialog(self.root, draft_en, "test/settings.json", events_en, commands_en)
        try:
            self.assertIsNone(dialog_en.filter_check)
            self.assertFalse(dialog_en.is_lang_filter_supported)
        finally:
            dialog_en.close()

        # In Japanese mode, font filter checkbox should be displayed
        cfg_ja = Config(language="ja")
        draft_ja = AppearanceDraft(AppearanceState.capture(cfg_ja, "CLASSIC", {}))
        events_ja, commands_ja = Queue(), Queue()
        dialog_ja = _Dialog(self.root, draft_ja, "test/settings.json", events_ja, commands_ja)
        try:
            self.assertIsNotNone(dialog_ja.filter_check)
            self.assertTrue(dialog_ja.is_lang_filter_supported)
        finally:
            dialog_ja.close()

    def test_button_and_badge_font_size_with_sync(self):
        self.dialog.sync_header_font_sizes.set(True)
        self.dialog.toggle_sync_header_font_sizes()
        self.events.get_nowait()
        self.assertEqual(str(self.dialog.badge_size_spin.cget("state")), "disabled")

        self.dialog.button_font_size.set("18")
        self.dialog.set_button_font_size()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertEqual(state.layout["button_font_size"], 18)
        self.assertEqual(state.layout["badge_font_size"], 22)
        self.assertEqual(self.dialog.badge_font_size.get(), "22")

    def test_button_and_badge_font_size_without_sync(self):
        self.dialog.sync_header_font_sizes.set(False)
        self.dialog.toggle_sync_header_font_sizes()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertFalse(state.layout["sync_header_font_sizes"])
        self.assertEqual(str(self.dialog.badge_size_spin.cget("state")), "normal")

        self.dialog.button_font_size.set("16")
        self.dialog.set_button_font_size()
        action, state = self.events.get_nowait()
        self.assertEqual(state.layout["button_font_size"], 16)
        # Badge size should remain unchanged
        self.assertNotEqual(state.layout.get("badge_font_size"), 20)

        self.dialog.badge_font_size.set("26")
        self.dialog.set_badge_font_size()
        action, state = self.events.get_nowait()
        self.assertEqual(state.layout["badge_font_size"], 26)

    def test_invalid_button_and_badge_font_size_shows_status_error(self):
        self.dialog.button_font_size.set("9")
        self.dialog.set_button_font_size()
        self.assertTrue(self.events.empty())
        self.assertIn("10", self.dialog.status.get())

        self.dialog.badge_font_size.set("30")
        self.dialog.set_badge_font_size()
        self.assertTrue(self.events.empty())
        self.assertIn("28", self.dialog.status.get())

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

    def test_adaptive_fill_toggle_and_submit(self):
        self.assertFalse(self.dialog.adaptive_fill.get())
        self.dialog.adaptive_fill.set(True)
        self.dialog.toggle_adaptive_fill()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertTrue(state.style["adaptive_fill"])

        self.dialog.submit("apply")
        action, state = self.events.get_nowait()
        self.assertEqual(action, "apply")
        self.assertTrue(state.style["adaptive_fill"])

    def test_window_size_preset_apply(self):
        values = self.dialog.window_preset_combo["values"]
        self.assertGreater(len(values), 0)
        # Select compact preset
        compact_entry = [v for v in values if "960x540" in v][0]
        self.dialog.window_preset_var.set(compact_entry)
        self.dialog.apply_selected_window_preset()
        action, preset_id = self.events.get_nowait()
        self.assertEqual(action, "window_preset")
        self.assertEqual(preset_id, "compact")
        self.assertEqual(self.dialog.window_size, (960, 540))

    def test_auto_adjust_bar_gap_button_updates_bar_gap_once(self):
        self.assertFalse(self.dialog.auto_adjust_bar_gap_on_resize.get())
        self.dialog.bar_gap.set("15")
        self.dialog.set_bar_gap()
        _ = self.events.get_nowait()  # consume preview

        self.dialog.auto_adjust_bar_gap()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertNotEqual(state.style["bar_gap"], 15)
        self.assertIn(str(state.style["bar_gap"]), self.dialog.status.get())

    def test_auto_adjust_bar_gap_on_resize_toggle_and_submit(self):
        self.assertFalse(self.dialog.auto_adjust_bar_gap_on_resize.get())
        self.dialog.auto_adjust_bar_gap_on_resize.set(True)
        self.dialog.toggle_auto_adjust_bar_gap_on_resize()
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertTrue(state.style["auto_adjust_bar_gap_on_resize"])

        self.dialog.submit("apply")
        action, state = self.events.get_nowait()
        self.assertEqual(action, "apply")
        self.assertTrue(state.style["auto_adjust_bar_gap_on_resize"])

    def test_info_font_bold_toggle_and_submit(self):
        self.assertFalse(self.dialog.info_font_bold.get())
        self.dialog.info_font_bold.set(True)
        self.dialog.layout("info_font_bold", True)
        action, state = self.events.get_nowait()
        self.assertEqual(action, "preview")
        self.assertTrue(state.layout["info_font_bold"])

        self.dialog.submit("apply")
        action, state = self.events.get_nowait()
        self.assertEqual(action, "apply")
        self.assertTrue(state.layout["info_font_bold"])



