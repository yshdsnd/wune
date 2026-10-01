import json
from pathlib import Path
import unittest
from unittest.mock import patch
import pygame as pg
from wune.exit_confirmation import ExitConfirmation
from wune.config import Config
from wune.settings import SettingsStore, valid_preference
from wune.appearance import AppearanceState, AppearanceDraft
from wune.application_menu import ApplicationMenu
from wune.layout import minimum_window_size


class ExitConfirmationTests(unittest.TestCase):
    def setUp(self):
        pg.font.init()
        self.font = pg.font.SysFont('Meiryo,Segoe UI', 15)
        self.prompt = ExitConfirmation()
        self.prompt.open()

    def key(self, key, **kwargs):
        return self.prompt.handle(pg.event.Event(pg.KEYDOWN, key=key, **kwargs),
                                  (480, 400), self.font, 'en')

    def test_cancel_is_default_and_checkbox_is_not_committed_on_cancel(self):
        self.assertEqual(self.key(pg.K_RETURN), (False, False))
        self.prompt.open()
        self.key(pg.K_LEFT)
        self.key(pg.K_SPACE)
        self.assertTrue(self.prompt.dont_ask)
        self.assertEqual(self.key(pg.K_ESCAPE), (False, False))
        self.prompt.open()
        self.assertFalse(self.prompt.dont_ask)

    def test_keyboard_accept_and_repeat_guard(self):
        self.key(pg.K_LEFT)
        self.key(pg.K_SPACE)
        self.key(pg.K_TAB, mod=pg.KMOD_SHIFT)
        self.assertIsNone(self.key(pg.K_RETURN, repeat=True))
        self.assertTrue(self.prompt.active)
        self.assertEqual(self.key(pg.K_RETURN), (True, True))

    def test_mouse_and_bilingual_minimum_layout(self):
        for language in ('en', 'ja'):
            for orientation in ('frequency_horizontal', 'frequency_vertical'):
                for layout in ('vertical', 'horizontal'):
                    cfg = Config(spectrum_orientation=orientation, channel_layout=layout)
                    size = minimum_window_size(cfg)
                    self.prompt.open()
                    panel, controls = self.prompt.geometry(size, self.font, language)
                    self.assertTrue(pg.Rect((0, 0), size).contains(panel))
                    surface = pg.Surface(size)
                    self.prompt.draw(surface, self.font, language, cfg.theme)
                    for index in (0, 2):
                        result = self.prompt.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1,
                                                   pos=controls[index].center), size, self.font, language)
                    self.assertEqual(result, (True, True))

    def test_settings_default_legacy_validation_and_reenable(self):
        for value, expected in ((False, False), (True, True), ('false', True), (0, True)):
            content = json.dumps({'version': 2, 'appearance': {'confirm_keyboard_exit': value}})
            with patch.object(Path, 'read_text', return_value=content):
                cfg, _ = SettingsStore('unused.json').load(Config())
            self.assertEqual(cfg.confirm_keyboard_exit, expected)
        with patch.object(Path, 'read_text', return_value='{"version": 2}'):
            self.assertTrue(SettingsStore('unused.json').load(Config())[0].confirm_keyboard_exit)
        self.assertFalse(valid_preference('confirm_keyboard_exit', 1))
        draft = AppearanceDraft(AppearanceState.capture(Config(confirm_keyboard_exit=False), 'CLASSIC', {}))
        draft.state.layout['confirm_keyboard_exit'] = True
        cfg = Config(confirm_keyboard_exit=False)
        draft.snapshot().apply(cfg)
        self.assertTrue(cfg.confirm_keyboard_exit)
        self.assertEqual(draft.state.user_presets, {})

    def test_menu_only_advertises_fullscreen_exit_key(self):
        self.assertEqual(ApplicationMenu().items('en', True)[-1][2], 'Q')
        self.assertEqual(ApplicationMenu().items('ja', False)[-1][2], 'Q / Esc')
