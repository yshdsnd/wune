import unittest
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pygame as pg
from wune.config import Config
from wune.presets import PRESETS
from wune.renderer import LedBarRenderer
from wune.layout import clamp_window_size
from renderer_reference import ReferenceRenderer


class RendererCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.font.init()

    def compare(self, actual, expected):
        levels = np.linspace(0, 1, actual.cfg.channels * actual.cfg.bars,
                             dtype=np.float32).reshape(actual.cfg.channels, actual.cfg.bars)
        for frame in (levels, levels * 0.1):
            actual.draw(frame, dt=0.1)
            expected.draw(frame, dt=0.1)
            self.assertEqual(pg.image.tobytes(actual.surf, 'RGB'),
                             pg.image.tobytes(expected.surf, 'RGB'))
            np.testing.assert_array_equal(actual.peak_pos, expected.peak_pos)

    def test_pixels_match_original_across_styles_sizes_and_directions(self):
        for preset in PRESETS:
            for shape in ('rectangle', 'rounded', 'ellipse'):
                for style in ('flat', 'box'):
                    for direction in ('frequency_horizontal', 'frequency_vertical'):
                        with self.subTest(theme=preset.name, shape=shape, style=style, direction=direction):
                            cfg = Config(theme=preset.theme, led_shape=shape,
                                         gauge_style=style, spectrum_orientation=direction)
                            actual = LedBarRenderer(pg.Surface(clamp_window_size((1000, 700), cfg)), deepcopy(cfg))
                            expected = ReferenceRenderer(pg.Surface(clamp_window_size((1000, 700), cfg)), deepcopy(cfg))
                            for size in ((1000, 700), (320, 240), (1920, 1080)):
                                actual.resize(pg.Surface(clamp_window_size(size, cfg)))
                                expected.resize(pg.Surface(clamp_window_size(size, cfg)))
                                self.compare(actual, expected)

    def test_live_changes_and_background_preserve_pixels(self):
        actual = LedBarRenderer(pg.Surface((1000, 700)), Config())
        expected = ReferenceRenderer(pg.Surface((1000, 700)), Config())
        # A patterned background also exposes differences in tile transparency.
        def background(surface, cfg):
            surface.fill((41, 73, 123))
            pg.draw.circle(surface, (190, 30, 80), (420, 250), 170)
        actual.background.draw = expected.background.draw = background
        self.compare(actual, expected)
        for key, value in (('channel_layout', 'horizontal'), ('led_aspect_ratio', 3.1),
                           ('led_shape', 'ellipse'), ('gauge_style', 'box'),
                           ('info_enabled', False)):
            setattr(actual.cfg, key, value)
            setattr(expected.cfg, key, value)
            actual.resize(pg.Surface(clamp_window_size((1000, 700), actual.cfg)))
            expected.resize(pg.Surface(clamp_window_size((1000, 700), expected.cfg)))
            self.compare(actual, expected)
        for renderer in (actual, expected):
            renderer.apply_preset('BLUE')
        self.compare(actual, expected)
        for renderer in (actual, expected):
            renderer.cfg.theme = replace(renderer.cfg.theme, green_on=(245, 20, 90),
                                         highlight=(0, 240, 80))
        self.compare(actual, expected)

    def test_unchanged_frames_reuse_geometry_and_artwork(self):
        renderer = LedBarRenderer(pg.Surface((1000, 700)), Config(show_freq_scale=False))
        levels = np.zeros((2, renderer.cfg.bars), dtype=np.float32)
        renderer.draw(levels)
        with patch.object(renderer, 'cell_rect', side_effect=AssertionError('rebuilt geometry')), \
             patch.object(renderer, '_led_tile', side_effect=AssertionError('rebuilt tiles')):
            renderer.draw(levels)


if __name__ == '__main__':
    unittest.main()
