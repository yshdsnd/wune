import unittest
from unittest.mock import Mock
import numpy as np
import pygame as pg

from wune.config import Config
from wune.layout import minimum_window_size
from wune.renderer import LedBarRenderer


class StereoMixDisplayTests(unittest.TestCase):
    def test_switch_resize_draw_peaks_and_labels_at_minimum_size(self):
        pg.font.init()
        for orientation in ("frequency_horizontal", "frequency_vertical"):
            for arrangement in ("vertical", "horizontal"):
                cfg = Config(spectrum_orientation=orientation, channel_layout=arrangement)
                renderer = LedBarRenderer(pg.Surface(minimum_window_size(cfg)), cfg)
                for mode, channels in (("stereo", 2), ("stereo_mix", 1), ("stereo", 2)):
                    cfg.channel_mode = mode
                    renderer.resize(pg.Surface(minimum_window_size(cfg)))
                    self.assertEqual(len(renderer.plots), channels)
                    self.assertEqual(renderer.peak_pos.shape, (channels, cfg.bars))
                    font = renderer.font_channel
                    renderer.font_channel = Mock(wraps=font)
                    renderer.draw(np.full((channels, cfg.bars), 0.8, dtype=np.float32), dt=1/60)
                    labels = [c.args[0] for c in renderer.font_channel.render.call_args_list]
                    self.assertEqual(labels, ["MIX"] if channels == 1 else ["L", "R"])
                    renderer.font_channel = font
                    for ch, plot in enumerate(renderer.plots):
                        for band in range(cfg.bars):
                            for led in range(cfg.leds_per_bar):
                                self.assertTrue(plot.contains(renderer.cell_rect(ch, band, led)))
