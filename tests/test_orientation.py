import unittest
from dataclasses import replace
import numpy as np
import pygame as pg
from wune.config import Config
from wune.layout import clamp_window_size, minimum_window_size
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
