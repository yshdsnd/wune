import unittest

import numpy as np
import pygame as pg
from wune.config import Config
from wune.layout import minimum_window_size, clamp_window_size, calculate_layout
from wune.renderer import LedBarRenderer


class LedShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.font.init()

    def test_resize_preserves_proportions_and_bounds(self):
        for direction in ("horizontal", "vertical"):
            for shape in ("rectangle", "rounded", "ellipse"):
                for ratio in (1.0, 2.0, 0.5):
                    cfg = Config(channel_layout=direction, led_shape=shape, led_aspect_ratio=ratio)
                    r = LedBarRenderer(pg.Surface(clamp_window_size((1280, 480), cfg)), cfg)
                    for size in (minimum_window_size(cfg), (1280, 480), (800, 1200), (1920, 480)):
                        with self.subTest(direction=direction, shape=shape, ratio=ratio, size=size):
                            r.resize(pg.Surface(clamp_window_size(size, cfg)))
                            cell = pg.Rect(0, 0, r.bar_w, r.led_h)
                            rect = r.led_rect(cell)
                            self.assertTrue(cell.contains(rect))
                            self.assertEqual(cell.center, rect.center)
                            # Integer rasterization may round the constrained axis by half a pixel.
                            self.assertLessEqual(abs(rect.width-ratio*rect.height), max(1, ratio)*0.5)
                            r.draw(np.full((2, 64), 0.8, dtype=np.float32))
                            self.assertEqual(cfg.leds_per_bar, 20)

    def test_shape_silhouettes_and_color(self):
        for shape in ("rectangle", "rounded", "ellipse"):
            r = LedBarRenderer(pg.Surface((1280, 480)), Config(led_shape=shape))
            r.surf.fill((1, 2, 3))
            cell = pg.Rect(100, 100, 80, 80)
            rect = r.led_rect(cell)
            r.draw_led(cell, (80, 160, 100), False)
            self.assertEqual(r.surf.get_at(rect.center)[:3], (80, 160, 100))
            self.assertEqual(r.surf.get_at(cell.topleft)[:3], (1, 2, 3))
            if shape != "rectangle":
                self.assertEqual(r.surf.get_at(rect.topleft)[:3], (1, 2, 3))

    def test_invalid_proportions_are_rejected(self):
        r = LedBarRenderer(pg.Surface((1280, 480)), Config())
        for ratio in (0, -1, float("nan"), float("inf")):
            r.cfg.led_aspect_ratio = ratio
            with self.assertRaisesRegex(ValueError, "led_aspect_ratio"):
                r.led_rect(pg.Rect(0, 0, 20, 20))

    def test_dense_grid_scales_gaps_with_leds(self):
        cfg = Config()
        for size in ((1280, 480), (1280, 1000), (2400, 1500)):
            layout = calculate_layout(size, cfg)
            self.assertEqual(layout.bar_width, 2 * layout.led_height)
            self.assertEqual(layout.bar_gap, layout.led_gap)
            self.assertLessEqual(abs(layout.led_gap-layout.led_height/4), 0.5)
            for x, y, w, h in layout.plots:
                self.assertEqual(w, cfg.bars*layout.bar_width+(cfg.bars-1)*layout.bar_gap)
                self.assertEqual(h, cfg.leds_per_bar*layout.led_height+(cfg.leds_per_bar-1)*layout.led_gap)
