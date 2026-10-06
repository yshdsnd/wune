"""Event-loop regressions for silent rendering; no audio hardware required."""
import types
import unittest
from unittest.mock import MagicMock

import numpy as np
import test_cleanup


class SmartIdleTests(unittest.TestCase):
    setUp = test_cleanup.AppCleanupTests.setUp
    tearDown = test_cleanup.AppCleanupTests.tearDown

    def prepare(self):
        self.app.menu.anchor = None
        self.app.renderer.peak_pos = np.zeros((2, 64))
        self.app.spectrum.step.return_value = np.zeros((2, 64))
        self.app.clock.tick.return_value = 16
        self.app.exit_confirmation.draw = MagicMock()
        self.drawn = []
        self.frame = 0
        self.app.renderer.draw.side_effect = lambda *a, **kw: self.drawn.append(self.frame)

    def run_frames(self, count, change=lambda frame: None):
        def events():
            self.frame += 1
            if self.frame > count:
                return [types.SimpleNamespace(type=self.pg.QUIT)]
            return change(self.frame) or []
        self.pg.event.get.side_effect = events
        self.app.run()

    def test_silence_skips_draws_but_keeps_capture_and_periodic_refresh(self):
        self.prepare()
        self.run_frames(130)
        self.assertLess(len(self.drawn), 40)
        self.assertIn(61, self.drawn)
        self.assertIn(121, self.drawn)
        self.assertEqual(self.app.spectrum.step.call_count, 130)
        self.assertEqual(self.pg.display.flip.call_count, len(self.drawn))
        self.pg.time.wait.assert_not_called()
        self.app.spectrum.close.assert_called_once()
        self.pg.quit.assert_called_once()

    def test_audio_resumes_on_first_nonzero_frame(self):
        self.prepare()
        def change(frame):
            if frame == 45:
                self.app.spectrum.step.return_value = np.full((2, 64), .5)
        self.run_frames(47, change)
        self.assertNotIn(44, self.drawn)
        self.assertEqual(self.drawn[-3:], [45, 46, 47])

    def test_peak_decay_prevents_idle(self):
        self.prepare()
        self.app.renderer.peak_pos.fill(.3)
        self.run_frames(70)
        self.assertEqual(len(self.drawn), 70)

    def test_expose_restore_and_resize_events_wake_drawing(self):
        self.prepare()
        events = {45: types.SimpleNamespace(type=self.pg.WINDOWEXPOSED),
                  80: types.SimpleNamespace(type=self.pg.WINDOWRESTORED),
                  115: types.SimpleNamespace(type=self.pg.VIDEORESIZE, size=(900, 600))}
        self.run_frames(116, lambda frame: [events[frame]] if frame in events else [])
        for frame in events:
            self.assertIn(frame, self.drawn)
        self.assertEqual(self.app.spectrum.step.call_count, 116)

    def test_menu_and_exit_confirmation_keep_rendering(self):
        self.prepare()
        def change(frame):
            if frame == 45:
                self.app.menu.anchor = (10, 10)
            if frame == 85:
                self.app.menu.anchor = None
                self.app.exit_confirmation.open()
        self.run_frames(125, change)
        self.assertNotIn(44, self.drawn)
        self.assertTrue(all(frame in self.drawn for frame in range(45, 126)))

    def test_settings_close_preview_redraws_without_pygame_event(self):
        self.prepare()
        test_cleanup.AppCleanupTests.prepare_settings_session(self)
        dialog = self.app.settings_dialog
        # Finish the dialog inside poll_settings, so it is already absent when
        # the idle decision runs. Rollback must still invalidate the image.
        self.app.settings_dialog = None
        def change(frame):
            if frame == 45:
                self.app.settings_dialog = dialog
                dialog.events.put(("closed", None))
        self.run_frames(46, change)
        self.assertNotIn(44, self.drawn)
        self.assertIn(45, self.drawn)
        self.assertIsNone(self.app.settings_dialog)

    def test_open_settings_keeps_preview_live(self):
        self.prepare()
        test_cleanup.AppCleanupTests.prepare_settings_session(self)
        self.run_frames(70)
        self.assertEqual(len(self.drawn), 70)

    def test_pause_and_unpause_remain_responsive(self):
        self.prepare()
        def change(frame):
            if frame in (45, 85):
                return [types.SimpleNamespace(type=self.pg.KEYDOWN, key=self.pg.K_SPACE)]
        self.run_frames(86, change)
        self.assertTrue(all(frame in self.drawn for frame in range(45, 87)))
        self.assertEqual(self.app.spectrum.step.call_count, 46)
        self.assertFalse(self.app.paused)


if __name__ == "__main__":
    unittest.main()
