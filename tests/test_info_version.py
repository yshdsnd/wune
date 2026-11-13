import unittest
from unittest.mock import patch
import pygame as pg

from wune.build_identity import application_identity
from wune.config import Config
from wune.renderer import LedBarRenderer


class InfoVersionRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.font.init()

    def setUp(self):
        self.surf = pg.Surface((800, 600))

    def test_default_config_version_settings(self):
        cfg = Config()
        self.assertFalse(cfg.info_show_version)
        self.assertEqual(cfg.info_version_position, "right")

    def test_renderer_renders_version_on_right(self):
        cfg = Config(info_enabled=True, info_show_version=False)
        renderer = LedBarRenderer(self.surf, cfg)
        renderer.info_text = "OUTPUT: Test Device | 48.0 kHz | 2 ch"
        renderer.draw_panel()
        bytes_without_version = pg.image.tobytes(self.surf, "RGB")

        cfg.info_show_version = True
        cfg.info_version_position = "right"
        self.surf.fill((0, 0, 0))
        renderer.draw_panel()
        bytes_with_version_right = pg.image.tobytes(self.surf, "RGB")

        self.assertNotEqual(bytes_without_version, bytes_with_version_right)

    def test_renderer_renders_version_on_left(self):
        cfg = Config(info_enabled=True, info_show_version=True, info_version_position="left")
        renderer = LedBarRenderer(self.surf, cfg)
        renderer.info_text = "OUTPUT: Test Device | 48.0 kHz | 2 ch"
        self.surf.fill((0, 0, 0))
        renderer.draw_panel()
        bytes_left = pg.image.tobytes(self.surf, "RGB")

        cfg.info_version_position = "right"
        self.surf.fill((0, 0, 0))
        renderer.draw_panel()
        bytes_right = pg.image.tobytes(self.surf, "RGB")

        self.assertNotEqual(bytes_left, bytes_right)

    def test_renderer_with_empty_info_text(self):
        cfg = Config(info_enabled=True, info_show_version=True, info_version_position="right")
        renderer = LedBarRenderer(self.surf, cfg)
        renderer.info_text = ""
        # Should not raise exception
        renderer.draw_panel()

    def test_renderer_with_narrow_surface_truncation(self):
        from wune.layout import clamp_window_size
        cfg = Config(info_enabled=True, info_show_version=True, info_version_position="right")
        min_w, min_h = clamp_window_size((1, 1), cfg)
        small_surf = pg.Surface((min_w, min_h))
        renderer = LedBarRenderer(small_surf, cfg)
        renderer.info_text = "OUTPUT: Extremely Long Audio Device Name That Will Need Aggressive Truncation | 192.0 kHz | 8 ch"
        renderer.app_identity = "Wune v1.2.3.4.5.6-very-long-identity-string-to-truncate"
        # Should fit and render safely without exception or overlap
        renderer.draw_panel()

    def test_renderer_disabled_info_bar(self):
        cfg = Config(info_enabled=False, info_show_version=True)
        renderer = LedBarRenderer(self.surf, cfg)
        renderer.info_text = "OUTPUT: Test Device"
        renderer.draw_panel()


if __name__ == "__main__":
    unittest.main()
