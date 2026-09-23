import unittest
from dataclasses import replace

import numpy as np
import pygame as pg
from wune.config import Config
from wune.layout import calculate_layout, clamp_window_size, minimum_window_size
from wune.renderer import LedBarRenderer


class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.font.init()

    def test_minimum_layouts_fit_without_changing_counts(self):
        for direction in ("vertical", "horizontal"):
            for channels in (1, 2):
                for bars in (16, 32, 64):
                    for info in ("top", "bottom"):
                        cfg = Config(channel_layout=direction, channels=channels, bars=bars, info_position=info)
                        size = minimum_window_size(cfg)
                        with self.subTest(direction=direction, channels=channels, bars=bars, info=info):
                            r = LedBarRenderer(pg.Surface(size), cfg)
                            self.assertEqual(cfg.leds_per_bar, 20)
                            self.assertGreaterEqual(r.bar_w, 3)
                            self.assertGreaterEqual(r.led_h, 3)
                            for plot in r.plots:
                                self.assertTrue(r.surf.get_rect().contains(plot))
                                self.assertFalse(plot.colliderect(pg.Rect(r._layout.info_rect)))
                            if channels == 2:
                                self.assertFalse(r.plots[0].colliderect(r.plots[1]))
                                if direction == "horizontal":
                                    self.assertEqual(r.plots[0].top, r.plots[1].top)
                                else:
                                    self.assertEqual(r.plots[0].left, r.plots[1].left)
                            r.draw(np.full((channels, bars), 0.8, dtype=np.float32))
                            r.draw_pause_overlay()

    def test_resize_reuses_peak_arrays_and_preserves_preset(self):
        cfg = Config(channel_layout="horizontal", bars=32)
        r = LedBarRenderer(pg.Surface((1280, 480)), cfg)
        r.apply_preset("BLUE")
        r.draw(np.full((2, 32), 0.8, dtype=np.float32))
        positions, holds = r.peak_pos, r.peak_hold
        values, counters = positions.copy(), holds.copy()
        for size in ((1600, 900), minimum_window_size(cfg), (1000, 600)):
            r.resize(pg.Surface(size))
            self.assertIs(r.peak_pos, positions)
            self.assertIs(r.peak_hold, holds)
            np.testing.assert_array_equal(positions, values)
            np.testing.assert_array_equal(holds, counters)
            self.assertEqual(r.trail.get_size(), size)
            self.assertEqual(r.preset_name, "BLUE")
            self.assertTrue(r.badge_contains(r.badge_rect().center))
            self.assertTrue(r.surf.get_rect().contains(r.badge_rect()))

    def test_growth_makes_leds_taller_and_uses_surface_dimensions(self):
        cfg = Config(width=1280, height=480)
        small = calculate_layout((1280, 480), cfg)
        tall = calculate_layout((1280, 900), cfg)
        self.assertGreater(tall.led_height, small.led_height)
        self.assertEqual((cfg.width, cfg.height, cfg.leds_per_bar), (1280, 480, 20))

    def test_orientation_and_resize_do_not_change_ballistics(self):
        cfg = Config(bars=32)
        vertical = LedBarRenderer(pg.Surface((1280, 720)), cfg)
        horizontal = LedBarRenderer(pg.Surface((1280, 720)), replace(cfg, channel_layout="horizontal"))
        for index, value in enumerate([0.9] + [0.1] * 30 + [0.7, 0]):
            if index == 10:
                horizontal.resize(pg.Surface((1000, 600)))
            levels = np.full((2, 32), value, dtype=np.float32)
            vertical.draw(levels)
            horizontal.draw(levels)
            np.testing.assert_array_equal(vertical.peak_pos, horizontal.peak_pos)
            np.testing.assert_array_equal(vertical.peak_hold, horizontal.peak_hold)

    def test_clamp_and_validation(self):
        cfg = Config(channel_layout="horizontal", bars=32)
        self.assertEqual(clamp_window_size((1, 1), cfg), minimum_window_size(cfg))
        with self.assertRaises(ValueError):
            calculate_layout((1, 1), cfg)
        with self.assertRaises(ValueError):
            minimum_window_size(Config(channel_layout="invalid"))
        with self.assertRaises(ValueError):
            minimum_window_size(Config(bars=0))
