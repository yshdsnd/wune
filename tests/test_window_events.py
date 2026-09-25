"""Use a child process: a dangling SDL PyObject can crash the interpreter."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class WindowEventLifetimeTests(unittest.TestCase):
    def test_geometry_wrapper_survives_collection_and_display_changes(self):
        code = textwrap.dedent('''
            import gc
            import pygame as pg
            from pygame._sdl2.video import Window
            from wune.app import App
            from wune.icons import pygame_icon
            pg.init()
            app = App.__new__(App)
            app._icon = pygame_icon()
            app.settings_store = object()
            app._display_window = None
            app._windowed_position = None
            try:
                app._set_mode((800, 600), pg.RESIZABLE)
                app._remember_position()
                position = app._windowed_position
                for i in range(20):
                    app._set_mode((0, 0) if i % 2 == 0 else (800, 600),
                                  pg.FULLSCREEN if i % 2 == 0 else pg.RESIZABLE)
                    app._restore_position()
                    gc.collect()
                    # Allocate same-sized Python objects to expose freed-memory reuse.
                    churn = [pg.event.Event(pg.USEREVENT, index=n) for n in range(200)]
                    for event in pg.event.get():
                        if hasattr(event, 'window') and event.window is not None:
                            assert isinstance(event.window, Window), repr(event)
                            assert event.window is app._display_window
                        assert event.type != pg.QUIT
                    assert app._windowed_position == position
                pg.event.post(pg.event.Event(pg.QUIT))
                assert any(e.type == pg.QUIT for e in pg.event.get())
                print('20 transitions: live Window references and QUIT preserved')
            finally:
                pg.quit()
        ''')
        env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
        result = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code],
                                cwd=Path(__file__).resolve().parents[1], env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("20 transitions", result.stdout)
