import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import shutil
import uuid
import pygame as pg

from wune.app import App
from wune.appearance import AppearanceDraft, AppearanceState
from wune.application_menu import ApplicationMenu
from wune.config import Config
from wune.settings import SettingsStore


class BorderlessWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.init()
        pg.font.init()

    def setUp(self):
        self.app = App.__new__(App)
        self.app.cfg = Config(borderless=False)
        self.app._fullscreen = False
        self.app._windowed_size = (1280, 800)
        self.app._windowed_position = (100, 100)
        self.app.menu = ApplicationMenu()
        self.app.renderer = MagicMock()
        self.app.renderer.ui_scale = 1.0
        self.app.renderer.font_menu = pg.font.Font(None, 18)
        self.app.renderer.badge_contains.return_value = False
        self.app.settings_store = None
        self.app.settings_dialog = None
        self.app.exit_confirmation = MagicMock(active=False)
        self.app._redraw_requested = False

        self.window = MagicMock()
        self.window.position = (100, 100)
        self.window.size = (1280, 800)
        self.window.borderless = False
        self.window.resizable = True
        self.app._display_window = self.window

        self.screen = MagicMock()
        self.screen.get_size.return_value = (1280, 800)
        self.app.screen = self.screen

    def test_config_borderless_default_is_false(self):
        cfg = Config()
        self.assertFalse(cfg.borderless)

    def test_settings_store_borderless_persistence(self):
        directory = (Path.cwd() / f"test-settings-{uuid.uuid4().hex}").resolve()
        directory.mkdir()
        self.addCleanup(shutil.rmtree, directory)
        path = directory / "settings.json"

        store = SettingsStore(path)
        cfg = Config(borderless=True)
        self.assertTrue(store.save(cfg, (800, 600), (100, 100), "CLASSIC"))

        loaded_cfg, _ = SettingsStore(path).load(Config())
        self.assertTrue(loaded_cfg.borderless)

        cfg_decorated = Config(borderless=False)
        self.assertTrue(store.save(cfg_decorated, (800, 600), (100, 100), "CLASSIC"))
        loaded_cfg2, _ = SettingsStore(path).load(Config())
        self.assertFalse(loaded_cfg2.borderless)

    def test_appearance_state_capture_and_apply(self):
        cfg = Config(borderless=True)
        state = AppearanceState.capture(cfg, "CLASSIC", {})
        self.assertIn("borderless", state.layout)
        self.assertTrue(state.layout["borderless"])

        target_cfg = Config(borderless=False)
        state.apply(target_cfg)
        self.assertTrue(target_cfg.borderless)

    def test_menu_borderless_item_and_fullscreen_exclusion(self):
        menu = ApplicationMenu()
        # In decorated mode
        items = menu.items("en", fullscreen=False, borderless=False)
        cmd_names = [cmd for cmd, *_ in items]
        self.assertIn("borderless", cmd_names)
        borderless_item = next(item for item in items if item[0] == "borderless")
        self.assertEqual(borderless_item[1], "Borderless Window")
        self.assertEqual(borderless_item[2], "F10")

        # In borderless mode
        items_bl = menu.items("en", fullscreen=False, borderless=True)
        borderless_item_bl = next(item for item in items_bl if item[0] == "borderless")
        self.assertEqual(borderless_item_bl[1], "Standard Window")

        # In Japanese
        items_ja = menu.items("ja", fullscreen=False, borderless=False)
        borderless_item_ja = next(item for item in items_ja if item[0] == "borderless")
        self.assertEqual(borderless_item_ja[1], "ボーダレスウィンドウ")

        items_ja_bl = menu.items("ja", fullscreen=False, borderless=True)
        borderless_item_ja_bl = next(item for item in items_ja_bl if item[0] == "borderless")
        self.assertEqual(borderless_item_ja_bl[1], "標準ウィンドウに戻す")

        # When in fullscreen, borderless option must be excluded
        items_fs = menu.items("en", fullscreen=True, borderless=False)
        self.assertNotIn("borderless", [cmd for cmd, *_ in items_fs])

    def test_app_borderless_lifecycle_and_flags(self):
        self.assertFalse(self.app.cfg.borderless)
        self.assertFalse(bool(self.app._windowed_flags() & pg.NOFRAME))

        # Enable borderless mode
        self.app.set_borderless(True)
        self.assertTrue(self.app.cfg.borderless)
        self.assertTrue(self.app.menu.borderless)
        self.assertTrue(bool(self.app._windowed_flags() & pg.NOFRAME))
        self.assertTrue(self.window.borderless)
        self.assertTrue(self.window.resizable)

        # Disable borderless mode
        self.app.set_borderless(False)
        self.assertFalse(self.app.cfg.borderless)
        self.assertFalse(self.app.menu.borderless)
        self.assertFalse(bool(self.app._windowed_flags() & pg.NOFRAME))
        self.assertFalse(self.window.borderless)

    def test_app_edge_detection(self):
        self.app.cfg.borderless = True
        w, h = self.app.screen.get_size()
        # Top-left corner
        self.assertEqual(self.app._detect_resize_edge((2, 2)), (True, False, True, False))
        # Bottom-right corner
        self.assertEqual(self.app._detect_resize_edge((w - 2, h - 2)), (False, True, False, True))
        # Center of the window (no edge)
        self.assertEqual(self.app._detect_resize_edge((w // 2, h // 2)), (False, False, False, False))
        # Right edge
        self.assertEqual(self.app._detect_resize_edge((w - 2, h // 2)), (False, True, False, False))
        # Left edge
        self.assertEqual(self.app._detect_resize_edge((2, h // 2)), (True, False, False, False))
        # Top edge
        self.assertEqual(self.app._detect_resize_edge((w // 2, 2)), (False, False, True, False))
        # Bottom edge
        self.assertEqual(self.app._detect_resize_edge((w // 2, h - 2)), (False, False, False, True))

        # When decorated, edge detection must return False for all edges
        self.app.cfg.borderless = False
        self.assertEqual(self.app._detect_resize_edge((2, 2)), (False, False, False, False))

    def test_f10_key_toggles_borderless(self):
        self.assertFalse(self.app.cfg.borderless)
        event_f10 = pg.event.Event(pg.KEYDOWN, key=pg.K_F10, mod=0)
        self.app.handle_event(event_f10)
        self.assertTrue(self.app.cfg.borderless)

        # Press again to toggle back
        self.app.handle_event(event_f10)
        self.assertFalse(self.app.cfg.borderless)

    def test_fullscreen_transitions_preserve_borderless(self):
        self.app.cfg.borderless = True
        self.app._set_mode = MagicMock()
        self.app._restore_position = MagicMock()

        # Enter fullscreen
        self.app.toggle_fullscreen()
        self.assertTrue(self.app._fullscreen)

        # Leave fullscreen
        self.app.toggle_fullscreen()
        self.assertFalse(self.app._fullscreen)

        # Borderless state must be preserved
        self.assertTrue(self.app.cfg.borderless)
        self.assertTrue(bool(self.app._windowed_flags() & pg.NOFRAME))


if __name__ == "__main__":
    unittest.main()
