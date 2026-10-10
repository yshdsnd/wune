import unittest
import numpy as np
import pygame as pg

from wune.application_menu import ApplicationMenu
from wune.config import Config
from wune.layout import (
    calculate_layout,
    calculate_optimal_leds_per_bar,
    calculate_ui_scale,
    clamp_window_size,
    minimum_window_size,
)
from wune.renderer import LedBarRenderer
from wune.window_geometry import restore_geometry


class UiScalingAndFlexibleWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.font.init()

    def test_calculate_ui_scale(self):
        # Base resolution or smaller should clamp to 1.0
        self.assertEqual(calculate_ui_scale((800, 600)), 1.0)
        self.assertEqual(calculate_ui_scale((1280, 800)), 1.0)
        self.assertEqual(calculate_ui_scale((1280, 720)), 1.0)

        # Full HD (1920x1080): min(1920/1280=1.5, 1080/800=1.35) = 1.35
        self.assertAlmostEqual(calculate_ui_scale((1920, 1080)), 1.35, places=2)

        # WQHD (2560x1440): min(2560/1280=2.0, 1440/800=1.8) = 1.8
        self.assertAlmostEqual(calculate_ui_scale((2560, 1440)), 1.8, places=2)

        # Ultrawide (3440x1440): limited by height scale 1.8
        self.assertAlmostEqual(calculate_ui_scale((3440, 1440)), 1.8, places=2)

        # 4K UHD (3840x2160): min(3840/1280=3.0, 2160/800=2.7) = 2.7
        self.assertAlmostEqual(calculate_ui_scale((3840, 2160)), 2.7, places=2)

        # Ultra high resolutions should clamp to max 3.0
        self.assertEqual(calculate_ui_scale((7680, 4320)), 3.0)

    def test_now_playing_and_info_rect_scaling_and_separation(self):
        cfg = Config(show_now_playing=True, info_position="top")
        
        # Standard size
        layout_base = calculate_layout((1280, 800), cfg)
        self.assertEqual(layout_base.ui_scale, 1.0)
        self.assertEqual(layout_base.now_playing_rect[3], 28)
        self.assertEqual(layout_base.info_rect[3], 28)
        # Verify no overlap between now_playing_rect and info_rect
        np_rect_base = pg.Rect(layout_base.now_playing_rect)
        info_rect_base = pg.Rect(layout_base.info_rect)
        self.assertFalse(np_rect_base.colliderect(info_rect_base))
        self.assertGreaterEqual(info_rect_base.top, np_rect_base.bottom)

        # 4K UHD
        layout_4k = calculate_layout((3840, 2160), cfg)
        self.assertGreater(layout_4k.ui_scale, 2.5)
        self.assertGreater(layout_4k.now_playing_rect[3], 28 * 2.5)
        self.assertGreater(layout_4k.info_rect[3], 28 * 2.5)
        
        np_rect_4k = pg.Rect(layout_4k.now_playing_rect)
        info_rect_4k = pg.Rect(layout_4k.info_rect)
        self.assertFalse(np_rect_4k.colliderect(info_rect_4k))
        self.assertGreaterEqual(info_rect_4k.top, np_rect_4k.bottom)

        for plot in layout_4k.plots:
            plot_rect = pg.Rect(plot)
            self.assertFalse(plot_rect.colliderect(np_rect_4k))
            self.assertFalse(plot_rect.colliderect(info_rect_4k))

    def test_bottom_info_bar_scaling(self):
        cfg = Config(show_now_playing=True, info_position="bottom")
        layout_4k = calculate_layout((3840, 2160), cfg)
        np_rect = pg.Rect(layout_4k.now_playing_rect)
        info_rect = pg.Rect(layout_4k.info_rect)
        self.assertFalse(np_rect.colliderect(info_rect))
        for plot in layout_4k.plots:
            plot_rect = pg.Rect(plot)
            self.assertFalse(plot_rect.colliderect(np_rect))
            self.assertFalse(plot_rect.colliderect(info_rect))

    def test_renderer_font_and_badge_scaling(self):
        cfg = Config()
        base_surf = pg.Surface((1280, 800))
        renderer = LedBarRenderer(base_surf, cfg)
        base_badge_rect = renderer.badge_rect()
        base_font_h = renderer.font_badge.get_height()

        # Resize to 4K
        surf_4k = pg.Surface((3840, 2160))
        renderer.resize(surf_4k)
        scaled_badge_rect = renderer.badge_rect()
        scaled_font_h = renderer.font_badge.get_height()

        # Badge and fonts must scale up significantly
        self.assertGreater(scaled_badge_rect.width, base_badge_rect.width * 2)
        self.assertGreater(scaled_badge_rect.height, base_badge_rect.height * 2)
        self.assertGreater(scaled_font_h, base_font_h * 2)

        # Hit test must work inside scaled badge and fail outside
        self.assertTrue(renderer.badge_contains(scaled_badge_rect.center))
        self.assertFalse(renderer.badge_contains((scaled_badge_rect.left - 10, scaled_badge_rect.top)))

        # Ensure drawing works without errors at 4K resolution
        levels = np.full((2, cfg.bars), 0.7, dtype=np.float32)
        renderer.draw(levels)
        renderer.draw_pause_overlay()

    def test_auto_scale_fonts_disabled_preserves_base_font_sizes(self):
        cfg = Config(auto_scale_fonts=False, label_font_size=14)
        base_surf = pg.Surface((1280, 800))
        renderer = LedBarRenderer(base_surf, cfg)
        base_badge_h = renderer.font_badge.get_height()
        base_scale_h = renderer.font_scale.get_height()

        # Resize to 4K
        surf_4k = pg.Surface((3840, 2160))
        renderer.resize(surf_4k)

        # Fonts must not scale up when auto_scale_fonts is False
        self.assertEqual(renderer.font_badge.get_height(), base_badge_h)
        self.assertEqual(renderer.font_scale.get_height(), base_scale_h)

    def test_configurable_label_font_size(self):
        base_surf = pg.Surface((1280, 800))
        cfg_small = Config(label_font_size=10)
        renderer_small = LedBarRenderer(base_surf, cfg_small)
        small_h = renderer_small.font_scale.get_height()

        cfg_large = Config(label_font_size=24)
        renderer_large = LedBarRenderer(base_surf, cfg_large)
        large_h = renderer_large.font_scale.get_height()

        self.assertGreater(large_h, small_h)

    def test_configurable_info_font_size_and_adaptive_height(self):
        base_surf = pg.Surface((1280, 800))
        cfg_small = Config(info_font_size=10, show_now_playing=True, info_enabled=True, info_position="top")
        renderer_small = LedBarRenderer(base_surf, cfg_small)
        small_h = renderer_small.font_small.get_height()

        cfg_large = Config(info_font_size=24, show_now_playing=True, info_enabled=True, info_position="top")
        renderer_large = LedBarRenderer(base_surf, cfg_large)
        large_h = renderer_large.font_small.get_height()

        self.assertGreater(large_h, small_h)

        layout_small = calculate_layout((1280, 800), cfg_small)
        layout_large = calculate_layout((1280, 800), cfg_large)

        np_small = pg.Rect(layout_small.now_playing_rect)
        np_large = pg.Rect(layout_large.now_playing_rect)
        info_small = pg.Rect(layout_small.info_rect)
        info_large = pg.Rect(layout_large.info_rect)

        self.assertGreater(np_large.height, np_small.height)
        self.assertGreater(info_large.height, info_small.height)

        # Bar heights must comfortably enclose font height
        self.assertGreaterEqual(np_small.height, small_h)
        self.assertGreaterEqual(np_large.height, large_h)

        # No collision between Now Playing and Info bar
        self.assertGreater(info_large.top, np_large.bottom)

        # No collision between info bars and spectrum plots
        header = max(44, cfg_large.header_reserved)
        channel_label_top = layout_large.plots[0][1] - header
        self.assertGreater(channel_label_top, info_large.bottom)

        for plot in layout_large.plots:
            plot_rect = pg.Rect(plot)
            self.assertFalse(np_large.colliderect(plot_rect))
            self.assertFalse(info_large.colliderect(plot_rect))

        # With auto_scale_fonts=False on 4K, bar height stays at base height
        layout_4k_no_scale = calculate_layout((3840, 2160), Config(info_font_size=24, auto_scale_fonts=False, show_now_playing=True))
        np_4k = pg.Rect(layout_4k_no_scale.now_playing_rect)
        self.assertEqual(np_4k.height, np_large.height)

    def test_application_menu_scaling(self):
        menu = ApplicationMenu()
        font_small = pg.font.SysFont("Meiryo,Segoe UI", 12)
        font_large = pg.font.SysFont("Meiryo,Segoe UI", 32)

        small_btn = menu.button_rect(font_small, "en")
        large_btn = menu.button_rect(font_large, "en")

        self.assertGreater(large_btn.width, small_btn.width)
        self.assertGreater(large_btn.height, small_btn.height)

        # Verify geometry with large font fits on large surface
        surface_size = (3840, 2160)
        menu_rect, rows = menu.geometry(surface_size, font_large, "en", False)
        self.assertTrue(pg.Rect((0, 0), surface_size).contains(menu_rect))
        self.assertEqual(len(rows), 3)
        for i in range(len(rows) - 1):
            self.assertGreaterEqual(rows[i + 1].top, rows[i].bottom)

    def test_flexible_window_sizing_preserves_custom_sizes(self):
        cfg = Config(bars=32)
        min_w, min_h = minimum_window_size(cfg)

        custom_sizes = [
            (3440, 1440),  # Ultrawide 21:9
            (2560, 600),   # Very wide banner
            (900, 1800),   # Very tall portrait
            (1920, 1200),  # 16:10
        ]

        for w, h in custom_sizes:
            self.assertGreaterEqual(w, min_w)
            self.assertGreaterEqual(h, min_h)
            # clamp_window_size must preserve custom width and height
            clamped = clamp_window_size((w, h), cfg)
            self.assertEqual(clamped, (w, h))

    def test_restore_geometry_preserves_saved_custom_size(self):
        cfg = Config(bars=32)
        saved = {
            "x": 100,
            "y": 100,
            "width": 3000,
            "height": 1000,
            "maximized": False,
        }
        bounds = ((0, 0, 3840, 2160),)
        size, position = restore_geometry(cfg, saved, bounds)
        self.assertEqual(size, (3000, 1000))
        self.assertEqual(position, (100, 100))


    def test_menu_and_badge_do_not_overlap_with_now_playing(self):
        for size in ((1280, 800), (3840, 2160)):
            with self.subTest(size=size):
                cfg = Config(show_now_playing=True, show_badge=True)
                surf = pg.Surface(size)
                renderer = LedBarRenderer(surf, cfg)
                menu = ApplicationMenu()
                btn_rect = menu.button_rect(renderer.font_small, cfg.language)
                badge_rect = renderer.badge_rect()
                np_rect = renderer.now_playing_rect()

                self.assertIsNotNone(np_rect)
                self.assertIsNotNone(badge_rect)
                # Now playing bar must be strictly below menu button and theme badge
                self.assertGreater(np_rect.top, btn_rect.bottom)
                self.assertGreater(np_rect.top, badge_rect.bottom)

    def test_bottom_info_bar_maintains_bottom_margin(self):
        for size in ((1280, 800), (3840, 2160)):
            with self.subTest(size=size):
                cfg = Config(info_enabled=True, info_position="bottom")
                layout = calculate_layout(size, cfg)
                self.assertIsNotNone(layout.info_rect)
                info_bottom = layout.info_rect[1] + layout.info_rect[3]
                # Guaranteed margin between info bar bottom and window bottom
                self.assertGreaterEqual(size[1] - info_bottom, 16)

    def test_medium_window_expands_led_grid(self):
        # Screenshot 2 dimensions: 935x811 with 64 bars
        cfg = Config(bars=64)
        layout = calculate_layout((935, 811), cfg)
        self.assertGreaterEqual(layout.led_height, 6)
        self.assertGreaterEqual(layout.bar_width, 12)
        self.assertGreaterEqual(layout.plots[0][2], 800)


    def test_configurable_leds_per_bar(self):
        base_size = (1280, 800)
        for count in (10, 20, 40, 60, 100):
            with self.subTest(leds_per_bar=count):
                cfg = Config(leds_per_bar=count)
                size = clamp_window_size(base_size, cfg)
                surf = pg.Surface(size)
                layout = calculate_layout(size, cfg)
                self.assertIsNotNone(layout)
                renderer = LedBarRenderer(surf, cfg)
                levels = np.full((cfg.display_channels, cfg.bars), 0.75, dtype=np.float32)
                renderer.draw(levels)
                # Verify that row_tiles and grid_rects reflect the exact count
                self.assertEqual(len(renderer._row_tiles), count)
                self.assertEqual(len(renderer._grid_rects[0][0]), count)

        # Test live dynamic reconfiguration without restarting
        surf = pg.Surface(base_size)
        cfg_dynamic = Config(leds_per_bar=20)
        renderer = LedBarRenderer(surf, cfg_dynamic)
        levels = np.full((cfg_dynamic.display_channels, cfg_dynamic.bars), 0.8, dtype=np.float32)
        renderer.draw(levels)
        self.assertAlmostEqual(float(renderer.peak_pos[0, 0]), 16.0, delta=0.5)

        # Dynamic increase to 40 leds_per_bar
        cfg_dynamic.leds_per_bar = 40
        renderer.draw(levels)
        self.assertEqual(len(renderer._row_tiles), 40)
        self.assertEqual(len(renderer._grid_rects[0][0]), 40)
        self.assertGreater(float(renderer.peak_pos[0, 0]), 25.0)

        # Dynamic decrease to 10 leds_per_bar
        cfg_dynamic.leds_per_bar = 10
        renderer.draw(levels)
        self.assertEqual(len(renderer._row_tiles), 10)
        self.assertEqual(len(renderer._grid_rects[0][0]), 10)
        self.assertLessEqual(float(renderer.peak_pos[0, 0]), 10.0)

    def test_calculate_optimal_leds_per_bar(self):
        # 1280x800 default window:
        # Vertical layout (2 rows) has limited height per channel -> optimal is 20 leds
        cfg_vertical = Config(channel_layout="vertical", bars=32)
        optimal_v = calculate_optimal_leds_per_bar((1280, 800), cfg_vertical)
        self.assertEqual(optimal_v, 20)

        # Horizontal layout (1 row) has ample vertical space -> optimal expands to 64 leds
        # to fill the huge vertical blank gap without sacrificing bar thickness
        cfg_horizontal = Config(channel_layout="horizontal", bars=32)
        optimal_h = calculate_optimal_leds_per_bar((1280, 800), cfg_horizontal)
        self.assertEqual(optimal_h, 64)

        # When bars=64 in horizontal mode, optimal reaches the upper bound (100)
        cfg_h64 = Config(channel_layout="horizontal", bars=64)
        optimal_h64 = calculate_optimal_leds_per_bar((1280, 800), cfg_h64)
        self.assertEqual(optimal_h64, 100)

        # 4K resolution (3840x2160):
        # Returns a valid number in [20, 100]
        cfg_4k = Config(channel_layout="horizontal", bars=32)
        optimal_4k = calculate_optimal_leds_per_bar((3840, 2160), cfg_4k)
        self.assertGreaterEqual(optimal_4k, 20)
        self.assertLessEqual(optimal_4k, 100)

        # When window size is below minimum, clamp_window_size ensures no crash and returns valid value
        tiny_size = (100, 100)
        optimal_tiny = calculate_optimal_leds_per_bar(tiny_size, cfg_vertical)
        self.assertGreaterEqual(optimal_tiny, 10)
        self.assertLessEqual(optimal_tiny, 100)

        # When current leds_per_bar is high (100), passing a smaller size must NOT
        # inflate the size and must return the lower optimal segment count.
        cfg_high = Config(channel_layout="vertical", bars=32, leds_per_bar=100)
        optimal_shrunk = calculate_optimal_leds_per_bar((1280, 800), cfg_high)
        self.assertEqual(optimal_shrunk, 20)

        # Custom bounds can be respected
        optimal_custom = calculate_optimal_leds_per_bar((1280, 800), cfg_horizontal, min_leds=30, max_leds=50)
        self.assertEqual(optimal_custom, 50)

    def test_app_resize_window_with_auto_adjust(self):
        from unittest.mock import MagicMock
        from wune.app import App
        app = App.__new__(App)
        app.cfg = Config(channel_layout="horizontal", bars=32, leds_per_bar=20, auto_adjust_leds_on_resize=True)
        app._fullscreen = False
        app._redraw_requested = False
        app.screen = pg.Surface((1280, 800))
        app._windowed_size = (1280, 800)
        app._set_mode = MagicMock()
        app.save_settings = MagicMock()
        app.renderer = MagicMock()
        app.settings_dialog = MagicMock()

        # Resize to same size (1280, 800) with auto_adjust enabled -> optimal is 64
        app.resize_window((1280, 800))
        self.assertEqual(app.cfg.leds_per_bar, 64)
        app.save_settings.assert_called_once()
        app.renderer.resize.assert_called_with(app.screen)
        app.settings_dialog.update_window_size.assert_called_with((1280, 800))

        # Shrinking test: when window was enlarged and segment count became high (e.g. 60),
        # shrinking the window down to (1280, 800) in vertical mode must shrink leds_per_bar to 20
        # and must NOT ratchet / get stuck at large window size.
        app.cfg.channel_layout = "vertical"
        app.cfg.leds_per_bar = 60
        app.resize_window((1280, 800))
        self.assertEqual(app.cfg.leds_per_bar, 20)
        self.assertEqual(app._windowed_size, (1280, 800))

        # When auto_adjust_leds_on_resize is False, resizing does not change leds_per_bar
        app.cfg.auto_adjust_leds_on_resize = False
        app.cfg.leds_per_bar = 50
        app.save_settings.reset_mock()
        app.resize_window((1280, 800))
        self.assertEqual(app.cfg.leds_per_bar, 50)
        app.save_settings.assert_not_called()


if __name__ == "__main__":
    unittest.main()
