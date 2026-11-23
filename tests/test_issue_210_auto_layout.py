from dataclasses import replace
import unittest
from unittest.mock import MagicMock, patch
import pygame as pg

from wune.app import App
from wune.appearance import AppearanceState
from wune.application_menu import ApplicationMenu
from wune.config import Config
from wune.layout import (
    calculate_optimal_bar_gap,
    calculate_optimal_leds_per_bar,
    minimum_window_size,
    resizable_minimum_window_size,
)
from wune.renderer import LedBarRenderer


class Issue210AutoLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.font.init()

    def test_fullscreen_open_and_cancel_settings_preserves_leds_and_gap(self):
        cfg = Config(
            adaptive_fill=True,
            auto_adjust_leds_on_resize=True,
            auto_adjust_bar_gap_on_resize=True,
            leds_per_bar=42,
            bar_gap=5,
        )
        app = App.__new__(App)
        app.cfg = cfg
        app._fullscreen = True
        app._windowed_size = (960, 540)
        app.screen = MagicMock()
        app.screen.get_size.return_value = (1920, 1080)
        app.renderer = MagicMock()
        app.renderer.preset_name = "CLASSIC"
        app.renderer.user_presets = {}
        app.menu = ApplicationMenu()
        app.settings_dialog = None
        app.settings_store = None
        app._appearance_baseline = None
        app._appearance_size = None
        app._settings_closing = False
        app._redraw_requested = False
        app.update_info_text = MagicMock()
        app.update_now_playing_text = MagicMock()
        app.save_settings = MagicMock()
        app.resize_window = MagicMock()

        with patch("wune.settings_dialog.SettingsDialog") as mock_dialog_cls:
            app.open_settings()
            # SettingsDialog should receive the actual fullscreen size
            self.assertEqual(mock_dialog_cls.call_args.kwargs["window_size"], (1920, 1080))

        # Now cancel without modifying anything
        with patch("pygame.display.set_caption"):
            app.cancel_settings()

        app.resize_window.assert_not_called()
        self.assertEqual(app.cfg.leds_per_bar, 42)
        self.assertEqual(app.cfg.bar_gap, 5)

    def test_auto_adjust_leds_allows_shrinking_from_large_count_to_minimum(self):
        cfg = Config(
            adaptive_fill=True,
            auto_adjust_leds_on_resize=True,
            auto_adjust_bar_gap_on_resize=True,
            leds_per_bar=60,
            bar_gap=6,
        )
        locked_min = minimum_window_size(cfg)
        resizable_min = resizable_minimum_window_size(cfg)
        expected_min = minimum_window_size(replace(cfg, leds_per_bar=10, bar_gap=1))

        self.assertEqual(resizable_min, expected_min)
        self.assertLess(resizable_min[1], locked_min[1])

        # Fixed mode keeps the full minimum size for the current leds_per_bar
        fixed_cfg = replace(cfg, auto_adjust_leds_on_resize=False, auto_adjust_bar_gap_on_resize=False)
        self.assertEqual(resizable_minimum_window_size(fixed_cfg), locked_min)

        # Optimal calculation at the smallest resizable size must shrink down rather than staying at 60
        shrunk_gap = calculate_optimal_bar_gap(resizable_min, cfg)
        shrunk_leds = calculate_optimal_leds_per_bar(resizable_min, replace(cfg, bar_gap=shrunk_gap))
        self.assertLess(shrunk_leds, 60)
        self.assertGreaterEqual(shrunk_leds, 10)

    def test_resize_window_shrinks_auto_leds_to_small_target(self):
        cfg = Config(
            adaptive_fill=True,
            auto_adjust_leds_on_resize=True,
            auto_adjust_bar_gap_on_resize=True,
            leds_per_bar=60,
            bar_gap=6,
        )
        target = resizable_minimum_window_size(cfg)

        app = App.__new__(App)
        app.cfg = cfg
        app._fullscreen = False
        app._windowed_size = (1920, 1080)
        app.screen = pg.Surface((1920, 1080))
        app.renderer = MagicMock()
        app.settings_dialog = None
        app.save_settings = MagicMock()
        app._windowed_flags = lambda: pg.RESIZABLE

        def fake_set_mode(size, flags):
            app.screen = pg.Surface(size)

        app._set_mode = fake_set_mode
        app.resize_window(target)

        self.assertEqual(app._windowed_size, target)
        self.assertLess(app.cfg.leds_per_bar, 60)
        self.assertGreaterEqual(app.cfg.leds_per_bar, 10)

    def test_db_labels_thin_and_omit_without_overlap_on_small_heights(self):
        cfg = Config(show_db_scale=True, db_min=-66.0, db_max=0.0, db_step=10.0)
        surf = pg.Surface((800, 600))
        renderer = LedBarRenderer(surf, cfg)

        def collect_rendered_labels(ch_h):
            renderer.ch_h = ch_h
            rendered = []
            real_font = renderer.font_scale

            class FontProxy:
                def get_height(self):
                    return real_font.get_height()

                def render(self, text, antialias, color, *args, **kwargs):
                    rendered.append(text)
                    return real_font.render(text, antialias, color, *args, **kwargs)

            renderer.font_scale = FontProxy()
            try:
                renderer.draw_db_labels_ch(0)
            finally:
                renderer.font_scale = real_font
            return rendered

        # Tall plot: all 7 numeric labels ("0" .. "-60") are drawn
        tall_labels = collect_rendered_labels(260)
        tall_nums = [t for t in tall_labels if t != "dB"]
        self.assertEqual(len(tall_nums), 7)

        # Compact plot: numeric labels are thinned (fewer than 7, still at least 2)
        compact_labels = collect_rendered_labels(65)
        compact_nums = [t for t in compact_labels if t != "dB"]
        self.assertGreaterEqual(len(compact_nums), 2)
        self.assertLess(len(compact_nums), 7)

        # Extreme tiny plot: all numeric labels are omitted
        tiny_labels = collect_rendered_labels(18)
        tiny_nums = [t for t in tiny_labels if t != "dB"]
        self.assertEqual(tiny_nums, [])


if __name__ == "__main__":
    unittest.main()
