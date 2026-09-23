import unittest
from dataclasses import replace

import numpy as np
import pygame as pg
from wune.colors import Theme
from wune.config import Config
from wune.renderer import LedBarRenderer


class ThemeRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.font.init()

    def renderer(self, **settings):
        cfg = Config(**settings)
        return LedBarRenderer(pg.Surface((cfg.width, cfg.height)), cfg)

    def test_styles_preserve_input_and_peak_evolution(self):
        flat = self.renderer()
        box = self.renderer(gauge_style="box", theme=replace(Theme(), background=(20, 25, 30)))
        for value in [0.9] + [0.1] * 30 + [0.7, 0.0]:
            levels = np.full((2, 64), value, dtype=np.float32)
            before = levels.copy()
            flat.draw(levels)
            box.draw(levels)
            np.testing.assert_array_equal(levels, before)
            np.testing.assert_array_equal(flat.peak_pos, box.peak_pos)
            np.testing.assert_array_equal(flat.peak_hold, box.peak_hold)

    def test_background_and_inactive_led_use_theme(self):
        theme = replace(Theme(), background=(40, 50, 60), green_off=(12, 34, 56))
        for style in ("flat", "box"):
            r = self.renderer(theme=theme, gauge_style=style)
            r.draw_panel()
            self.assertEqual(r.surf.get_at((0, 0))[:3], theme.background)
            rect = pg.Rect(100, 100, 16, 12)
            r.draw_led(rect, theme.green_off, False)
            self.assertEqual(r.surf.get_at(rect.center)[:3], theme.green_off)

    def test_box_has_light_face_and_shadow_inside_bounds(self):
        r = self.renderer(gauge_style="box")
        r.surf.fill((1, 2, 3))
        rect = pg.Rect(100, 100, 16, 12)
        r.draw_led(rect, (80, 160, 100), True)
        self.assertGreater(r.surf.get_at((101, 100))[0], r.surf.get_at(rect.center)[0])
        self.assertLess(r.surf.get_at((101, 111))[0], r.surf.get_at(rect.center)[0])
        self.assertEqual(r.surf.get_at((99, 100))[:3], (1, 2, 3))

    def test_small_box_keeps_color(self):
        r = self.renderer(gauge_style="box")
        for size in ((1, 1), (2, 2), (3, 3)):
            rect = pg.Rect((100, 100), size)
            r.draw_led(rect, (80, 160, 100), True)
            self.assertEqual(r.surf.get_at(rect.center)[:3], (80, 160, 100))

    def test_invalid_style_is_reported(self):
        with self.assertRaisesRegex(ValueError, "gauge_style"):
            self.renderer(gauge_style="unknown")
