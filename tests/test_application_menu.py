import unittest
import pygame as pg
from wune.application_menu import ApplicationMenu
from wune.config import Config
from wune.layout import minimum_window_size
from wune.renderer import LedBarRenderer


class ApplicationMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.font.init()

    def setUp(self):
        self.menu = ApplicationMenu()
        self.font = pg.font.SysFont("Meiryo,Segoe UI", 15)

    def handle(self, kind, **values):
        return self.menu.handle(pg.event.Event(kind, values), (480, 400), self.font, "en", False)

    def test_button_and_right_click_share_commands(self):
        for event in (dict(button=1, pos=self.menu.button_rect(self.font, "en").center),
                      dict(button=3, pos=(470, 390))):
            for index, expected in enumerate(("settings", "fullscreen", "window_size", "exit")):
                self.menu.close()
                self.assertEqual(self.handle(pg.MOUSEBUTTONDOWN, **event), (True, None))
                rect, rows = self.menu.geometry((480, 400), self.font, "en", False)
                self.assertTrue(pg.Rect(0, 0, 480, 400).contains(rect))
                if expected == "window_size":
                    self.assertEqual(self.handle(pg.MOUSEBUTTONDOWN, button=1, pos=rows[index].center),
                                     (True, None))
                    self.assertEqual(self.menu.submenu, "window_size")
                    self.assertIsNotNone(self.menu.anchor)
                else:
                    self.assertEqual(self.handle(pg.MOUSEBUTTONDOWN, button=1, pos=rows[index].center),
                                     (True, expected))
                    self.assertIsNone(self.menu.anchor)

    def test_window_size_submenu_and_presets(self):
        self.assertEqual(self.handle(pg.MOUSEBUTTONDOWN, button=3, pos=(200, 200)), (True, None))
        rect, rows = self.menu.geometry((480, 400), self.font, "en", False)
        items = self.menu.items("en", False)
        ws_index = [i for i, (cmd, *_) in enumerate(items) if cmd == "window_size"][0]
        self.assertEqual(self.handle(pg.MOUSEBUTTONDOWN, button=1, pos=rows[ws_index].center), (True, None))
        self.assertEqual(self.menu.submenu, "window_size")

        sub_rect, sub_rows = self.menu.geometry((480, 400), self.font, "en", False)
        sub_items = self.menu.items("en", False)
        self.assertEqual(sub_items[0][0], "back")
        self.assertEqual(self.handle(pg.MOUSEBUTTONDOWN, button=1, pos=sub_rows[0].center), (True, None))
        self.assertIsNone(self.menu.submenu)

        rect, rows = self.menu.geometry((480, 400), self.font, "en", False)
        self.assertEqual(self.handle(pg.MOUSEBUTTONDOWN, button=1, pos=rows[ws_index].center), (True, None))
        sub_rect, sub_rows = self.menu.geometry((480, 400), self.font, "en", False)
        sub_items = self.menu.items("en", False)
        self.assertEqual(self.handle(pg.MOUSEBUTTONDOWN, button=1, pos=sub_rows[1].center),
                         (True, sub_items[1][0]))
        self.assertIsNone(self.menu.anchor)
        self.assertIsNone(self.menu.submenu)

    def test_dismissal_and_outside_click_do_not_activate_underlying_controls(self):
        for kind, values, consumed in ((pg.KEYDOWN, dict(key=pg.K_ESCAPE), True),
                                      (pg.MOUSEBUTTONDOWN, dict(button=1, pos=(470, 390)), True),
                                      (pg.WINDOWFOCUSLOST, {}, False),
                                      (pg.WINDOWSIZECHANGED, {}, False)):
            self.menu.anchor = (24, 44)
            self.assertEqual(self.handle(kind, **values), (consumed, None))
            self.assertIsNone(self.menu.anchor)

    def test_keyboard_navigation_and_alt_enter_passthrough(self):
        self.handle(pg.MOUSEBUTTONDOWN, button=3, pos=(24, 44))
        self.handle(pg.KEYDOWN, key=pg.K_DOWN)
        self.assertEqual(self.handle(pg.KEYDOWN, key=pg.K_RETURN, mod=0), (True, "fullscreen"))
        self.handle(pg.MOUSEBUTTONDOWN, button=3, pos=(24, 44))
        self.assertEqual(self.handle(pg.KEYDOWN, key=pg.K_RETURN, mod=pg.KMOD_ALT), (False, None))
        self.assertIsNone(self.menu.anchor)

    def test_localized_state_and_minimum_layouts_with_long_theme_names(self):
        for language in ("en", "ja"):
            self.assertNotEqual(self.menu.items(language, False)[1][1], self.menu.items(language, True)[1][1])
            for orientation in ("frequency_horizontal", "frequency_vertical"):
                for channels in ("horizontal", "vertical"):
                    cfg = Config(language=language, spectrum_orientation=orientation, channel_layout=channels)
                    size = minimum_window_size(cfg)
                    surface = pg.Surface(size)
                    renderer = LedBarRenderer(surface, cfg)
                    renderer.preset_name = "LONG THEME NAME " * 5
                    button = self.menu.button_rect(renderer.font_small, language)
                    self.assertFalse(button.colliderect(renderer.badge_rect()))
                    renderer.draw_panel()
                    self.menu.anchor = (size[0] - 1, size[1] - 1)
                    rect, _ = self.menu.geometry(size, renderer.font_small, language, True)
                    self.assertTrue(surface.get_rect().contains(rect))
                    self.menu.draw(surface, renderer.font_small, language, True, cfg.theme)
