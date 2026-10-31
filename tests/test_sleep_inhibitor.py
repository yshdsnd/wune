"""Unit tests for SleepInhibitor and idle power management transitions (Issue #201)."""
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import test_cleanup

from wune.config import Config
from wune.sleep_inhibitor import (
    ES_CONTINUOUS,
    ES_DISPLAY_REQUIRED,
    ES_SYSTEM_REQUIRED,
    SleepInhibitor,
)


class SleepInhibitorTests(unittest.TestCase):
    def test_sleep_inhibitor_windows_api_calls(self):
        mock_kernel32 = MagicMock()
        inhibitor = SleepInhibitor(kernel32=mock_kernel32)

        # Initially uninhibited
        self.assertIsNone(inhibitor.is_inhibited)

        # Enable inhibition (playback active)
        with patch("pygame.display.get_init", return_value=True), \
             patch("pygame.display.set_allow_screensaver") as mock_set_screensaver:
            inhibitor.set_inhibited(True)
            self.assertTrue(inhibitor.is_inhibited)
            mock_set_screensaver.assert_called_once_with(False)
            mock_kernel32.SetThreadExecutionState.assert_called_once_with(
                ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED
            )

        # Re-applying same state is a no-op
        mock_kernel32.reset_mock()
        inhibitor.set_inhibited(True)
        mock_kernel32.SetThreadExecutionState.assert_not_called()

        # Disable inhibition (idle / silent)
        with patch("pygame.display.get_init", return_value=True), \
             patch("pygame.display.set_allow_screensaver") as mock_set_screensaver:
            inhibitor.set_inhibited(False)
            self.assertFalse(inhibitor.is_inhibited)
            mock_set_screensaver.assert_called_once_with(True)
            mock_kernel32.SetThreadExecutionState.assert_called_once_with(ES_CONTINUOUS)

        # Close ensures inhibition is released
        inhibitor.set_inhibited(True)
        self.assertTrue(inhibitor.is_inhibited)
        mock_kernel32.reset_mock()
        with patch("pygame.display.get_init", return_value=True), \
             patch("pygame.display.set_allow_screensaver") as mock_set_screensaver:
            inhibitor.close()
            self.assertFalse(inhibitor.is_inhibited)
            mock_set_screensaver.assert_called_once_with(True)
            mock_kernel32.SetThreadExecutionState.assert_called_once_with(ES_CONTINUOUS)

    def test_sleep_inhibitor_apply_reapplies_state(self):
        mock_kernel32 = MagicMock()
        inhibitor = SleepInhibitor(kernel32=mock_kernel32)
        inhibitor.set_inhibited(True)
        mock_kernel32.reset_mock()

        with patch("pygame.display.get_init", return_value=True), \
             patch("pygame.display.set_allow_screensaver") as mock_set_screensaver:
            inhibitor.apply()
            mock_set_screensaver.assert_called_once_with(False)
            mock_kernel32.SetThreadExecutionState.assert_called_once_with(
                ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED
            )


class AppSleepLifecycleTests(unittest.TestCase):
    setUp = test_cleanup.AppCleanupTests.setUp
    tearDown = test_cleanup.AppCleanupTests.tearDown

    def prepare(self):
        self.app.menu.anchor = None
        self.app.renderer.peak_pos = np.zeros((2, 64))
        self.app.spectrum.step.return_value = np.zeros((2, 64))
        self.app.clock.tick.return_value = 16
        self.app.exit_confirmation.draw = MagicMock()
        self.app.cfg.sleep_inhibit_idle_seconds = 10.0
        self.frame = 0
        self.history = {}

        orig_draw = self.app.renderer.draw
        def record_draw(*args, **kwargs):
            self.history[self.frame] = self.app.sleep_inhibitor.is_inhibited
            return orig_draw(*args, **kwargs)
        self.app.renderer.draw = MagicMock(side_effect=record_draw)

    def run_frames(self, count, change=lambda frame: None):
        def events():
            self.frame += 1
            if self.frame > count:
                return [types.SimpleNamespace(type=self.pg.QUIT)]
            return change(self.frame) or []
        self.pg.event.get.side_effect = events
        self.app.run()

    def test_initial_state_silent_allows_sleep(self):
        """Transition 1: Silent from startup (起動直後から無音)."""
        self.prepare()
        # On startup with no audio, sleep_inhibitor should be initialized to False (sleep allowed)
        self.assertFalse(self.app.sleep_inhibitor.is_inhibited)

        # Run 5 silent frames
        self.run_frames(5)
        # All drawn frames must have is_inhibited == False
        self.assertTrue(all(v is False for v in self.history.values()))

    def test_playback_to_silence_and_resumption_lifecycle(self):
        """Transitions 2, 3, 4, 5:
        Start silent -> Playback -> Short silence (track change) -> Long silence (idle) -> Playback resumes.
        """
        self.prepare()
        mono_time = [100.0]

        def get_mono_time():
            return mono_time[0]

        def change(frame):
            if 1 <= frame <= 20:
                # Active music playback
                mono_time[0] = 100.0 + frame * 0.016
                self.app.spectrum.step.return_value = np.full((2, 64), 0.5)
            elif 21 <= frame <= 30:
                # Short track change (3s silence < 10.0s timeout)
                mono_time[0] = 103.0
                self.app.spectrum.step.return_value = np.zeros((2, 64))
                self.app.renderer.peak_pos = np.zeros((2, 64))
            elif 31 <= frame <= 40:
                # Music stopped (15s silence >= 10.0s timeout)
                mono_time[0] = 120.0
                self.app.spectrum.step.return_value = np.zeros((2, 64))
                self.app.renderer.peak_pos = np.zeros((2, 64))
            elif frame >= 41:
                # Playback resumes
                mono_time[0] = 130.0
                self.app.spectrum.step.return_value = np.full((2, 64), 0.7)

        with patch("wune.app.time.monotonic", side_effect=get_mono_time):
            self.run_frames(45, change)

        # 1. During active playback (frames 1-20): inhibited must be True
        for f in range(1, 21):
            if f in self.history:
                self.assertTrue(self.history[f], f"Frame {f} should be inhibited")

        # 2. During short track change (frames 21-30, 3s silence): inhibited must still be True
        for f in range(21, 31):
            if f in self.history:
                self.assertTrue(self.history[f], f"Frame {f} (track change) should remain inhibited")

        # 3. After 10s idle silence (frames 31-40, 15s silence): inhibited must be False
        for f in range(31, 41):
            if f in self.history:
                self.assertFalse(self.history[f], f"Frame {f} (idle) should allow sleep")

        # 4. Once playback resumes (frames 41-45): inhibited must be True again
        for f in range(41, 46):
            if f in self.history:
                self.assertTrue(self.history[f], f"Frame {f} (playback resumed) should be inhibited")

    def test_paused_releases_suppression_after_timeout(self):
        """Paused state releases suppression after idle timeout."""
        self.prepare()
        mono_time = [100.0]

        def change(frame):
            if 1 <= frame <= 10:
                # Playing
                mono_time[0] = 100.0 + frame * 0.016
                self.app.spectrum.step.return_value = np.full((2, 64), 0.5)
            elif 11 <= frame <= 20:
                # Paused (under 10s)
                self.app.paused = True
                mono_time[0] = 103.0
            elif frame >= 21:
                # Paused (over 10s)
                self.app.paused = True
                mono_time[0] = 120.0

        with patch("wune.app.time.monotonic", side_effect=lambda: mono_time[0]):
            self.run_frames(30, change)

        # While playing: True
        for f in range(1, 11):
            if f in self.history:
                self.assertTrue(self.history[f])

        # Paused under 10s: True
        for f in range(11, 21):
            if f in self.history:
                self.assertTrue(self.history[f])

        # Paused over 10s: False (sleep allowed)
        for f in range(21, 31):
            if f in self.history:
                self.assertFalse(self.history[f])

    def test_system_resume_resets_idle_state(self):
        """スリープ復帰後 (Resume from sleep)."""
        self.prepare()
        self.app.sleep_inhibitor.set_inhibited(True)
        self.app.spectrum.restart_capture = MagicMock()
        self.app.clock.tick = MagicMock()
        self.app.update_info_text = MagicMock()

        self.app.on_system_resume()

        # Capture restarted, suppression set to False until audio starts
        self.app.spectrum.restart_capture.assert_called_once()
        self.assertFalse(self.app.sleep_inhibitor.is_inhibited)
        self.assertIsNone(self.app._silence_start_time)

    def test_exit_releases_suppression(self):
        """終了・エラー時 (Exit / cleanup)."""
        self.prepare()
        self.app.sleep_inhibitor.close = MagicMock()
        self.run_frames(1)
        # On exit, sleep_inhibitor.close() must be called
        self.app.sleep_inhibitor.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
