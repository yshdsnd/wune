"""Tests for preserving native desktop resolution and avoiding display mode switches in fullscreen (Issue #106)."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pygame as pg

from wune.app import App
from wune.config import Config


class FullscreenResolutionTests(unittest.TestCase):
    def setUp(self):
        self.app = App.__new__(App)
        self.app.cfg = Config()
        self.app._fullscreen = False
        self.app._windowed_size = (1280, 800)
        self.app._windowed_position = (100, 100)
        self.app.menu = MagicMock()
        self.app.renderer = MagicMock()
        self.app.settings_store = None

        self.window = MagicMock()
        self.window.position = (100, 100)
        self.window.size = (1280, 800)
        self.window.display_index = 0
        self.app._display_window = self.window

        self.screen = MagicMock()
        self.screen.get_size.return_value = (1280, 800)
        self.app.screen = self.screen

    def test_dpi_awareness_env_set_on_windows(self):
        def fake_set_mode(app_self, size, flags, **kwargs):
            app_self.screen = MagicMock()

        with patch.dict(os.environ, {}, clear=True), patch("sys.platform", "win32"), \
             patch("pygame.init") as mock_init, \
             patch("wune.app.set_app_id"), \
             patch("wune.app.pygame_icon"), \
             patch("wune.now_playing.create_default_coordinator"), \
             patch("wune.app.LedBarRenderer"), \
             patch.object(App, "_set_mode", new=fake_set_mode), \
             patch.object(App, "_restore_position"), \
             patch.object(App, "_restore_fullscreen"), \
             patch("wune.app.AudioSpectrum") as mock_spectrum:
            mock_spectrum.return_value.sr = 48000
            mock_spectrum.return_value.fmax = 24000
            mock_init.side_effect = lambda: self.assertEqual(
                os.environ.get("SDL_WINDOWS_DPI_AWARENESS"), "permonitorv2"
            )
            App(Config())

    def test_enter_fullscreen_uses_desktop_fullscreen_without_mode_switch(self):
        full_surface = MagicMock()
        full_surface.get_size.return_value = (3840, 2160)

        with patch("pygame.display.get_surface", return_value=full_surface), \
             patch.object(self.app, "_set_mode") as mock_set_mode:
            self.app._enter_fullscreen()

        self.window.set_fullscreen.assert_called_once_with(desktop=True)
        mock_set_mode.assert_not_called()
        self.assertTrue(self.app._fullscreen)
        self.assertIs(self.app.screen, full_surface)
        self.assertEqual(self.app.screen.get_size(), (3840, 2160))

    def test_enter_fullscreen_falls_back_when_desktop_fullscreen_raises(self):
        self.window.set_fullscreen.side_effect = RuntimeError("desktop fullscreen unsupported")

        with patch.object(self.app, "_set_mode") as mock_set_mode:
            with self.assertWarns(RuntimeWarning):
                self.app._enter_fullscreen(display=1)

        mock_set_mode.assert_called_once_with((0, 0), pg.FULLSCREEN, display=1)
        self.assertTrue(self.app._fullscreen)

    def test_leave_fullscreen_restores_windowed_state(self):
        self.app._fullscreen = True

        with patch.object(self.app, "_set_mode") as mock_set_mode, \
             patch.object(self.app, "_restore_position") as mock_restore_pos:
            self.app._leave_fullscreen()

        self.window.set_windowed.assert_called_once()
        self.assertFalse(self.app._fullscreen)
        mock_set_mode.assert_called_once()
        mock_restore_pos.assert_called_once()

    def test_toggle_fullscreen_roundtrip(self):
        full_surface = MagicMock()
        full_surface.get_size.return_value = (3840, 2160)
        windowed_surface = MagicMock()
        windowed_surface.get_size.return_value = (1280, 800)

        with patch("pygame.display.get_surface", return_value=full_surface), \
             patch.object(self.app, "_set_mode") as mock_set_mode, \
             patch.object(self.app, "_restore_position"):
            # 1. Enter fullscreen
            self.app.toggle_fullscreen()
            self.assertTrue(self.app._fullscreen)
            self.window.set_fullscreen.assert_called_once_with(desktop=True)
            self.assertEqual(self.app._windowed_size, (1280, 800))
            self.app.renderer.resize.assert_called_with(full_surface)

            # 2. Leave fullscreen
            self.app.screen = full_surface
            mock_set_mode.side_effect = lambda size, flags: setattr(self.app, "screen", windowed_surface)
            self.app.toggle_fullscreen()
            self.assertFalse(self.app._fullscreen)
            self.window.set_windowed.assert_called_once()
            self.assertEqual(self.app.renderer.resize.call_count, 2)


if __name__ == "__main__":
    unittest.main()
