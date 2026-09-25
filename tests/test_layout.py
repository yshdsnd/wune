import unittest
from dataclasses import replace

import numpy as np
import pygame as pg
from wune.config import Config
from wune.layout import calculate_layout, clamp_window_size, minimum_window_size, fit_window_size
from wune.renderer import LedBarRenderer


class LayoutTests(unittest.TestCase):
    def test_fit_is_stable_for_all_shapes_and_layouts(self):
        for orientation in ("frequency_horizontal", "frequency_vertical"):
            for channels in ("horizontal", "vertical"):
                for ratio in (0.25, 1.0, 2.0, 2.8, 8.0):
                    cfg = Config(spectrum_orientation=orientation, channel_layout=channels,
                                 led_aspect_ratio=ratio, bars=32)
                    for size in ((1, 1), (960, 2000), (2400, 400), (1800, 1400)):
                        with self.subTest(orientation=orientation, channels=channels, ratio=ratio, size=size):
                            fitted = fit_window_size(size, cfg)
                            self.assertEqual(fit_window_size(fitted, cfg), fitted)
                            self.assertTrue(all(a <= b for a, b in zip(fitted, clamp_window_size(size, cfg))))
                            grid = calculate_layout(fitted, cfg)
                            self.assertLessEqual(abs(grid.bar_width - ratio * grid.led_height), 0.5)
                            self.assertLessEqual(abs(grid.led_gap - grid.led_height / 4), 0.5)

    def test_fullscreen_packs_channels_in_both_orientations(self):
        for orientation in ("frequency_horizontal", "frequency_vertical"):
            for channels in ("horizontal", "vertical"):
                cfg = Config(spectrum_orientation=orientation, channel_layout=channels, bars=32)
                gaps = []
                for size in ((2000, 1400), (3000, 2200)):
                    a, b = calculate_layout(size, cfg).plots
                    gaps.append(b[0] - a[0] - a[2] if channels == "horizontal" else b[1] - a[1] - a[3])
                self.assertEqual(gaps[0], gaps[1])
                self.assertLess(gaps[0], 200)

    def test_height_grows_grid_when_usable_and_stops_at_width_limit(self):
        for orientation in ("frequency_horizontal", "frequency_vertical"):
            for channels in ("horizontal", "vertical"):
                cfg = Config(spectrum_orientation=orientation, channel_layout=channels, bars=32)
                _, minimum_h = minimum_window_size(cfg)
                small = fit_window_size((2200, minimum_h), cfg)
                tall = fit_window_size((2200, minimum_h + 200), cfg)
                self.assertGreater(calculate_layout(tall, cfg).led_height,
                                   calculate_layout(small, cfg).led_height)
                limited = fit_window_size((2200, 10000), cfg)
                self.assertEqual(fit_window_size((limited[0], 20000), cfg), limited)
                self.assertLess(limited[1], 10000)

    def test_corner_resize_is_monotonic_and_scales_gaps(self):
        for orientation in ("frequency_horizontal", "frequency_vertical"):
            for channels in ("horizontal", "vertical"):
                cfg = Config(spectrum_orientation=orientation, channel_layout=channels, bars=32)
                width, height = minimum_window_size(cfg)
                previous = calculate_layout((width, height), cfg)
                first = previous
                for extra in range(0, 1001, 10):
                    grid = calculate_layout(fit_window_size((width + extra, height + extra), cfg), cfg)
                    self.assertGreaterEqual(grid.led_height, previous.led_height)
                    self.assertGreaterEqual(grid.bar_width, previous.bar_width)
                    self.assertGreaterEqual(grid.led_gap, previous.led_gap)
                    self.assertLessEqual(grid.led_height - previous.led_height, 1)
                    previous = grid
                self.assertGreater(grid.led_height, first.led_height)
                self.assertGreater(grid.led_gap, first.led_gap)

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

    def test_tall_narrow_flat_leds_during_resize(self):
        for bars in (32, 64):
            cfg = Config(channel_layout="horizontal", bars=bars)
            width, height = minimum_window_size(cfg)
            r = LedBarRenderer(pg.Surface((1280, 480)), cfg)
            r.apply_preset("CLASSIC")
            for size in ((width, 900), (width + 128, 900), (width, height), (1280, 720)):
                with self.subTest(bars=bars, size=size):
                    r.resize(pg.Surface(size))
                    r.draw(np.full((2, bars), 0.8, dtype=np.float32))

    def test_narrow_leds_keep_their_face_color_in_every_preset(self):
        for preset in ("CLASSIC", "AMBER", "BLUE", "CLASSIC BOX"):
            r = LedBarRenderer(pg.Surface((1280, 480)), Config())
            r.apply_preset(preset)
            for width in range(3, 9):
                for height in (8, 16, 32):
                    with self.subTest(preset=preset, width=width, height=height):
                        r.surf.fill((0, 0, 0))
                        rect = pg.Rect(10, 10, width, height)
                        color = r.cfg.theme.green_on
                        r.draw_led(rect, color, True)
                        pixels = pg.surfarray.array3d(r.surf.subsurface(rect))
                        # Resampling blends edge pixels; the lit face must stay bright
                        # and retain its dominant theme color rather than turn black.
                        channel = int(np.argmax(color))
                        self.assertGreater(pixels[:, :, channel].max(), color[channel] * 0.7)

    def test_clamp_and_validation(self):
        cfg = Config(channel_layout="horizontal", bars=32)
        self.assertEqual(clamp_window_size((1, 1), cfg), minimum_window_size(cfg))
        with self.assertRaises(ValueError):
            calculate_layout((1, 1), cfg)
        with self.assertRaises(ValueError):
            minimum_window_size(Config(channel_layout="invalid"))
        with self.assertRaises(ValueError):
            minimum_window_size(Config(bars=0))
