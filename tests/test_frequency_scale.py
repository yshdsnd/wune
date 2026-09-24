import unittest
from unittest.mock import Mock

import numpy as np
import pygame as pg
from wune.config import Config
from wune.layout import minimum_window_size
from wune.renderer import LedBarRenderer


class FrequencyScaleTests(unittest.TestCase):
    def test_20khz_reference_is_actually_drawn_even_at_minimum_size(self):
        pg.font.init()
        marker = (201, 40, 90)
        for orientation in ("frequency_horizontal", "frequency_vertical"):
            for layout in ("vertical", "horizontal"):
                for maximum in (16000, 20000, 40000):
                    for edges in (True, False):
                        with self.subTest(orientation=orientation, layout=layout, maximum=maximum, edges=edges):
                            cfg = Config(spectrum_orientation=orientation, channel_layout=layout,
                                         max_freq_hz=maximum, show_freq_edge_labels=edges, bars=32)
                            surface = pg.Surface(minimum_window_size(cfg))
                            renderer = LedBarRenderer(surface, cfg)
                            font = renderer.font_scale
                            marked = []
                            def render(text, antialias, color):
                                result = font.render(text, antialias, color)
                                if text == "20kHz":
                                    result.fill(marker)
                                    marked.append(result.get_width()*result.get_height())
                                return result
                            renderer.font_scale = Mock(wraps=font)
                            renderer.font_scale.render.side_effect = render
                            renderer.draw_freq_scale()
                            pixels = pg.surfarray.array3d(surface)
                            count = np.all(pixels == marker, axis=2).sum()
                            if maximum >= 20000:
                                self.assertEqual(len(marked), cfg.channels)
                                self.assertEqual(count, sum(marked))
                            else:
                                self.assertEqual(count, 0)

    def test_frequency_scale_can_still_be_hidden(self):
        pg.font.init()
        cfg = Config(show_freq_scale=False, max_freq_hz=40000)
        renderer = LedBarRenderer(pg.Surface((1280, 800)), cfg)
        renderer.font_scale = Mock()
        renderer.draw_freq_scale()
        renderer.font_scale.render.assert_not_called()
