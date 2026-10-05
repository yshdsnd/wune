import ctypes
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import pygame as pg
from wune import system_locale
from wune.config import Config
from wune.renderer import LedBarRenderer
from wune.settings import settings_path


class MacBasicsTests(unittest.TestCase):
    def tearDown(self):
        system_locale.user_locale.cache_clear()

    def test_macos_path_ignores_windows_environment(self):
        with patch('wune.settings.sys.platform', 'darwin'), \
             patch('wune.settings.Path.home', return_value=Path('/Users/test')), \
             patch.dict('os.environ', {'LOCALAPPDATA': '/wrong'}):
            self.assertEqual(settings_path(), Path('/Users/test/Library/Application Support/Wune/settings.json'))

    def test_windows_path_is_unchanged(self):
        with patch('wune.settings.sys.platform', 'win32'), \
             patch.dict('os.environ', {'LOCALAPPDATA': '/local'}):
            self.assertEqual(settings_path(), Path('/local/Wune/settings.json'))

    def preferred_language(self, value, count=1, conversion=True):
        cf = Mock()
        cf.CFLocaleCopyPreferredLanguages.return_value = 123
        cf.CFArrayGetCount.return_value = count
        cf.CFArrayGetValueAtIndex.return_value = 456
        def copy_string(ref, buffer, size, encoding):
            buffer.value = value
            return conversion
        cf.CFStringGetCString.side_effect = copy_string
        system_locale.user_locale.cache_clear()
        with patch('wune.system_locale.sys.platform', 'darwin'), \
             patch('ctypes.cdll.LoadLibrary', return_value=cf), \
             patch('wune.system_locale.locale.getlocale', return_value=('en_US', None)):
            language = system_locale.user_locale()
        cf.CFRelease.assert_called_once_with(123)
        self.assertEqual(cf.CFArrayGetValueAtIndex.argtypes, [ctypes.c_void_p, ctypes.c_long])
        return language

    def test_macos_ui_language_takes_precedence_over_process_locale(self):
        self.assertEqual(self.preferred_language(b'ja-JP'), 'ja-JP')

    def test_empty_and_failed_native_conversion_fall_back_and_release(self):
        self.assertEqual(self.preferred_language(b'', count=0), 'en_US')
        self.assertEqual(self.preferred_language(b'ja', conversion=False), 'en_US')

    def test_unavailable_framework_uses_existing_fallback(self):
        with patch('wune.system_locale.sys.platform', 'darwin'), \
             patch('ctypes.cdll.LoadLibrary', side_effect=OSError), \
             patch('wune.system_locale.locale.getlocale', return_value=('en_US', None)):
            self.assertEqual(system_locale.user_locale(), 'en_US')

    def test_platform_font_selection_keeps_windows_preferences(self):
        pg.font.init()
        for platform, title, japanese, scale in (
            ('win32', 'Bahnschrift', 'Meiryo', 'Consolas'),
            ('darwin', 'SF Pro Display', 'Hiragino Sans GB', 'SF Mono')):
            with patch('wune.renderer.sys.platform', platform), \
                 patch('pygame.font.SysFont', wraps=pg.font.SysFont) as fonts:
                LedBarRenderer(pg.Surface((1280, 800)), Config())
                names = [call.args[0] for call in fonts.call_args_list]
                self.assertTrue(names[0].startswith(title))
                self.assertTrue(names[1].startswith(japanese))
                self.assertTrue(names[-1].startswith(scale))


if __name__ == '__main__':
    unittest.main()
