import unittest
from dataclasses import replace

import numpy as np
import pygame as pg
from wune.colors import Theme
from wune.config import Config
from wune.renderer import LedBarRenderer
from wune.presets import PRESETS


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
            np.testing.assert_allclose(r.surf.get_at(rect.center)[:3], theme.green_off, atol=3)

    def test_box_has_light_face_and_shadow_inside_bounds(self):
        r = self.renderer(gauge_style="box", led_shape="rectangle")
        r.surf.fill((1, 2, 3))
        rect = pg.Rect(100, 100, 16, 12)
        r.draw_led(rect, (80, 160, 100), True)
        fitted = r.led_rect(rect)
        self.assertGreater(r.surf.get_at((fitted.left + 1, fitted.top))[0], r.surf.get_at(rect.center)[0])
        self.assertLess(r.surf.get_at((fitted.left + 1, fitted.bottom - 1))[0], r.surf.get_at(rect.center)[0])
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

    def test_cycle_preserves_peak_arrays_and_returns_to_classic(self):
        r = self.renderer()
        r.apply_preset("CLASSIC")
        r.draw(np.full((2, 64), 0.8, dtype=np.float32))
        positions, holds = r.peak_pos, r.peak_hold
        previous_positions, previous_holds = positions.copy(), holds.copy()
        for preset in (*PRESETS[1:], PRESETS[0]):
            r.next_preset()
            self.assertEqual(r.preset_name, preset.name)
            self.assertEqual(r.cfg.theme, preset.theme)
            self.assertEqual(r.cfg.gauge_style, preset.gauge_style)
            self.assertIs(r.peak_pos, positions)
            self.assertIs(r.peak_hold, holds)
            np.testing.assert_array_equal(positions, previous_positions)
            np.testing.assert_array_equal(holds, previous_holds)

    def test_badge_hit_area_follows_name_and_visibility(self):
        r = self.renderer()
        for preset in PRESETS:
            r.apply_preset(preset.name)
            rect = r.badge_rect()
            self.assertTrue(r.badge_contains(rect.center))
            self.assertFalse(r.badge_contains((rect.right, rect.bottom)))
            self.assertEqual(rect.width, r.font_badge.size(preset.name)[0] + 16)
        r.cfg.show_badge = False
        self.assertFalse(r.badge_contains(rect.center))

    def test_unknown_preset_does_not_change_appearance(self):
        r = self.renderer()
        r.apply_preset("CLASSIC")
        with self.assertRaises(ValueError):
            r.apply_preset("missing")
        self.assertEqual(r.preset_name, "CLASSIC")
        self.assertEqual(r.cfg.theme, Theme())

    def test_user_theme_cycles_after_builtins_and_badge_stays_in_window(self):
        from wune.appearance import decode_preset, encode_preset
        name = "夜空" * 20
        r = self.renderer(width=800, height=600)
        r.user_presets = {name: decode_preset(name, encode_preset(PRESETS[1]))}
        r.apply_preset("CLASSIC BOX")
        r.next_preset()
        self.assertEqual(r.preset_name, name)
        self.assertEqual(r.cfg.theme, PRESETS[1].theme)
        r.draw(np.full((2, 64), .8, dtype=np.float32))
        rect = r.badge_rect()
        self.assertGreaterEqual(rect.left, 0)
        self.assertLessEqual(rect.right, r.width)
        r.next_preset()
        self.assertEqual(r.preset_name, "CLASSIC")
