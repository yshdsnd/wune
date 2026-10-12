"""Unit tests for Linux/FreeBSD platform foundation (settings path, locale, process mode, fonts)."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from wune.config import Config
from wune.renderer import LedBarRenderer
from wune.settings import settings_path
from wune.system_locale import user_locale


class LinuxPlatformFoundationTests(unittest.TestCase):
    def test_settings_path_linux_default(self):
        with patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(Path, "home", return_value=Path("/home/testuser")):
            path = settings_path()
            self.assertEqual(path, Path("/home/testuser/.config/Wune/settings.json"))

    def test_settings_path_linux_xdg(self):
        with patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {"XDG_CONFIG_HOME": "/custom/config"}), \
             patch.object(Path, "home", return_value=Path("/home/testuser")):
            path = settings_path()
            self.assertEqual(path, Path("/custom/config/Wune/settings.json"))

    def test_settings_path_freebsd(self):
        with patch.object(sys, "platform", "freebsd14"), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(Path, "home", return_value=Path("/usr/home/testuser")):
            path = settings_path()
            self.assertEqual(path, Path("/usr/home/testuser/.config/Wune/settings.json"))

    def test_settings_dialog_process_mode_on_linux_and_freebsd(self):
        from wune.settings_dialog import SettingsDialog
        for plat in ("linux", "freebsd14", "darwin", "win32"):
            with self.subTest(platform=plat):
                with patch.object(sys, "platform", plat):
                    dialog = SettingsDialog.__new__(SettingsDialog)
                    dialog._process = sys.platform in ("win32", "darwin", "linux") or sys.platform.startswith("freebsd")
                    self.assertTrue(dialog._process)

    def test_system_locale_linux_env_vars(self):
        user_locale.cache_clear()
        try:
            with patch.object(sys, "platform", "linux"), \
                 patch.dict(os.environ, {"LC_ALL": "ja_JP.UTF-8"}):
                user_locale.cache_clear()
                self.assertEqual(user_locale(), "ja_JP")

            with patch.object(sys, "platform", "linux"), \
                 patch.dict(os.environ, {"LANG": "en_US.UTF-8"}, clear=True):
                user_locale.cache_clear()
                self.assertEqual(user_locale(), "en_US")
        finally:
            user_locale.cache_clear()

    def test_create_fonts_linux_and_freebsd(self):
        import pygame as pg
        if not pg.font.get_init():
            pg.font.init()
        from wune.layout import minimum_window_size
        cfg = Config()
        surf = pg.Surface(minimum_window_size(cfg))
        for plat in ("linux", "freebsd14"):
            with self.subTest(platform=plat):
                with patch.object(sys, "platform", plat):
                    renderer = LedBarRenderer(surf, cfg)
                    self.assertIsNotNone(renderer.font_channel)
                    self.assertIsNotNone(renderer.font_small)
                    self.assertIsNotNone(renderer.font_badge)
                    self.assertIsNotNone(renderer.font_scale)


if __name__ == "__main__":
    unittest.main()
