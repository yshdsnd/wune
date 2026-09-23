import unittest
from dataclasses import replace
import numpy as np
import pygame as pg
from wune.config import Config
from wune.layout import clamp_window_size, minimum_window_size, fit_window_size, calculate_layout
from wune.renderer import LedBarRenderer


class OrientationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.font.init()

    def test_transposed_cells_fit_both_channel_layouts_after_resize(self):
        for layout in ("vertical", "horizontal"):
            for bars in (32, 64):
                for info in ("top", "bottom"):
                    cfg = Config(spectrum_orientation="frequency_vertical", channel_layout=layout,
                                 bars=bars, info_position=info)
                    r = LedBarRenderer(pg.Surface(minimum_window_size(cfg)), cfg)
                    for size in (minimum_window_size(cfg), (1280, 1000), (1800, 800)):
                        with self.subTest(layout=layout, bars=bars, info=info, size=size):
                            r.resize(pg.Surface(clamp_window_size(size, cfg)))
                            for ch, plot in enumerate(r.plots):
                                for band in range(bars):
                                    for led in range(cfg.leds_per_bar):
                                        self.assertTrue(plot.contains(r.cell_rect(ch, band, led)))
                                self.assertLess(r.cell_rect(ch, bars-1, 0).y, r.cell_rect(ch, 0, 0).y)
                                self.assertLess(r.cell_rect(ch, 0, 0).x, r.cell_rect(ch, 0, 19).x)
                                self.assertFalse(plot.colliderect(pg.Rect(r._layout.info_rect)))
                            r.draw(np.full((2, bars), 0.7, dtype=np.float32), dt=1/60)
                            r.draw_pause_overlay()

    def test_orientation_does_not_change_peak_motion_or_input(self):
        cfg = Config(bars=32)
        a = LedBarRenderer(pg.Surface((1280, 1000)), cfg)
        b = LedBarRenderer(pg.Surface((1280, 1000)), replace(cfg, spectrum_orientation="frequency_vertical"))
        for value in [0.9]+[0.1]*30+[0.7, 0]:
            data = np.full((2, 32), value, dtype=np.float32)
            original = data.copy()
            a.draw(data, dt=0.02)
            b.draw(data, dt=0.02)
            np.testing.assert_array_equal(data, original)
            np.testing.assert_array_equal(a.peak_pos, b.peak_pos)
            np.testing.assert_array_equal(a.peak_hold, b.peak_hold)

    def test_transposed_peak_is_a_vertical_marker_at_level_end(self):
        cfg = Config(bars=32, spectrum_orientation="frequency_vertical")
        r = LedBarRenderer(pg.Surface((960, 1000)), cfg)
        data = np.zeros((2, 32), dtype=np.float32)
        data[0, 4] = 0.5
        r.draw(data, dt=0.1)
        cell = r.cell_rect(0, 4, 9)
        self.assertEqual(r.surf.get_at((cell.right, cell.centery))[:3], cfg.theme.peak)

    def test_scale_visibility_and_invalid_orientation(self):
        for freq in (True, False):
            for db in (True, False):
                cfg = Config(bars=32, spectrum_orientation="frequency_vertical",
                             show_freq_scale=freq, show_db_scale=db)
                r = LedBarRenderer(pg.Surface(minimum_window_size(cfg)), cfg)
                r.draw(np.zeros((2, 32), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "spectrum_orientation"):
            minimum_window_size(Config(spectrum_orientation="invalid"))

    def test_window_fit_is_stable_and_removes_excess_space(self):
        for channels in ("horizontal", "vertical"):
            for bars in (32, 64):
                cfg = Config(spectrum_orientation="frequency_vertical", channel_layout=channels, bars=bars)
                for size in ((960, 800), (960, 2000), (2200, 400), (1, 1), (1400, 1000)):
                    with self.subTest(channels=channels, bars=bars, size=size):
                        fitted = fit_window_size(size, cfg)
                        self.assertEqual(fit_window_size(fitted, cfg), fitted)
                        self.assertTrue(all(a <= b for a, b in zip(fitted, clamp_window_size(size, cfg))))
                        layout = calculate_layout(fitted, cfg)
                        if channels == "horizontal":
                            self.assertEqual(layout.plots[1][0] - (layout.plots[0][0]+layout.plots[0][2]), 72+16+cfg.channel_gap)
                        # Fullscreen also packs the channel group instead of spreading it.
                        full = calculate_layout(clamp_window_size((2400, 1600), cfg), cfg)
                        if channels == "horizontal":
                            self.assertEqual(full.plots[1][0] - (full.plots[0][0]+full.plots[0][2]), 72+16+cfg.channel_gap)

    def test_height_cannot_grow_without_room_to_grow_gauges(self):
        cfg = Config(spectrum_orientation="frequency_vertical", channel_layout="horizontal", bars=32)
        fitted = fit_window_size((960, 2000), cfg)
        self.assertLess(fitted[1], 800)
        self.assertEqual(fit_window_size((fitted[0], 4000), cfg), fitted)
