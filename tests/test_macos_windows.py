import ctypes as ct
from pathlib import Path
from queue import Queue
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pygame as pg
from wune.app import App
from wune.appearance import AppearanceState
from wune.config import Config
from wune.settings_dialog import SettingsDialog
from wune.window_geometry import work_areas


class MacWindowTests(unittest.TestCase):
    def test_menu_shows_platform_shortcuts(self):
        from wune.application_menu import ApplicationMenu
        for platform, expected in (('darwin', ('Cmd+,', 'Cmd+F')), ('win32', ('F2', 'Alt+Enter'))):
            with patch('wune.application_menu.sys.platform', platform):
                items = ApplicationMenu().items('en', False)
            self.assertEqual((items[0][2], items[1][2]), expected)

    def test_mac_settings_spawn_and_cleanup(self):
        context = Mock()
        context.Process.return_value.is_alive.return_value = True
        context.Process.return_value.exitcode = None
        queues = [Mock(), Mock()]
        context.Queue.side_effect = queues
        with patch('wune.settings_dialog.sys.platform', 'darwin'), \
             patch('wune.settings_dialog.mp.get_context', return_value=context) as factory:
            dialog = SettingsDialog(AppearanceState.capture(Config(), 'CLASSIC', {}), 'settings.json')
            self.assertFalse(dialog.worker_failed)
            dialog.focus()
            dialog.reply(True, 'status.applied')
            dialog.close()
            dialog.close()
        factory.assert_called_once_with('spawn')
        context.Process.return_value.start.assert_called_once()
        context.Process.return_value.terminate.assert_called_once()
        for queue in queues:
            queue.close.assert_called_once()
            queue.cancel_join_thread.assert_called_once()

    def test_process_start_failure_closes_queues(self):
        context = Mock()
        queues = [Mock(), Mock()]
        context.Queue.side_effect = queues
        context.Process.return_value.start.side_effect = OSError('spawn')
        with patch('wune.settings_dialog.sys.platform', 'darwin'), \
             patch('wune.settings_dialog.mp.get_context', return_value=context):
            with self.assertRaises(OSError):
                SettingsDialog(AppearanceState.capture(Config(), 'CLASSIC', {}), 'settings.json')
        for queue in queues:
            queue.close.assert_called_once()

    def test_mac_ready_avoids_win32_ownership_and_worker_crash_reverts(self):
        app = App.__new__(App)
        dialog = Mock(events=Queue(), worker_failed=False)
        app.settings_dialog = dialog
        app.cancel_settings = Mock()
        dialog.events.put(('ready', None))
        with patch('wune.app.sys.platform', 'darwin'), \
             patch('wune.settings_window.SettingsWindow') as native:
            app.poll_settings()
        native.assert_not_called()
        dialog.focus.assert_called_once()
        dialog.worker_failed = True
        with self.assertWarns(RuntimeWarning):
            app.poll_settings()
        app.cancel_settings.assert_called_once()
        dialog.close.assert_called_once()
        self.assertIsNone(app.settings_dialog)

    def test_command_shortcuts_are_mac_only(self):
        app = App.__new__(App)
        app.screen = Mock()
        app.renderer = Mock()
        app.cfg = Config()
        app._fullscreen = False
        app.exit_confirmation = Mock(active=False)
        app.menu = Mock()
        app.menu.handle.return_value = (False, None)
        app.execute_command = Mock()
        for platform in ('win32', 'darwin'):
            for key, command in ((pg.K_COMMA, 'settings'), (pg.K_f, 'fullscreen')):
                with patch('wune.app.sys.platform', platform):
                    app.handle_event(pg.event.Event(pg.KEYDOWN, key=key, mod=pg.KMOD_META))
                if platform == 'darwin':
                    app.execute_command.assert_called_once_with(command)
                else:
                    app.execute_command.assert_not_called()
                app.execute_command.reset_mock()

    def test_mac_display_enumeration_fallback_and_negative_coordinates(self):
        cg = Mock()
        cg.CGMainDisplayID.return_value = 9
        def active(maximum, displays, count):
            ct.cast(count, ct.POINTER(ct.c_uint32))[0] = 0
            return 0
        def online(maximum, displays, count):
            displays[0], displays[1] = 3, 9
            ct.cast(count, ct.POINTER(ct.c_uint32))[0] = 2
            return 0
        def bounds(display):
            return SimpleNamespace(origin=SimpleNamespace(x=-1920 if display == 3 else 0, y=0),
                                   size=SimpleNamespace(width=1920, height=1080))
        cg.CGGetActiveDisplayList.side_effect = active
        cg.CGGetOnlineDisplayList.side_effect = online
        cg.CGDisplayBounds.side_effect = bounds
        with patch('wune.window_geometry.sys.platform', 'darwin'), \
             patch('ctypes.cdll.LoadLibrary', return_value=cg):
            self.assertEqual(work_areas(), [(0, 0, 1920, 1080), (-1920, 0, 1920, 1080)])
            cg.CGGetOnlineDisplayList.side_effect = active
            self.assertEqual(work_areas(), [(0, 0, 1920, 1080)])

    def test_desktop_initialization_failure_has_safe_fallback(self):
        with patch('wune.window_geometry.sys.platform', 'darwin'), \
             patch('ctypes.cdll.LoadLibrary', side_effect=OSError('framework')), \
             patch('pygame.display.get_init', return_value=False), \
             patch('pygame.display.init', side_effect=pg.error('display')):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(work_areas(), [(0, 0, 1920, 1080)])

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS real spawned Tk process')
    def test_real_hidden_mac_settings_process(self):
        subprocess.run([sys.executable, str(Path(__file__).with_name('macos_settings_probe.py'))],
                       check=True, timeout=45)
