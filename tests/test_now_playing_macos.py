"""Unit tests for macOS Apple Music metadata provider."""
import subprocess
import sys
import unittest
from unittest.mock import patch, MagicMock

from wune.now_playing import NowPlaying, create_default_coordinator
from wune.now_playing_macos import MacAppleMusicProvider, _default_osascript_runner


class MacAppleMusicProviderTests(unittest.TestCase):
    def test_availability(self):
        provider_no_runner = MacAppleMusicProvider()
        expected = sys.platform == "darwin"
        self.assertEqual(provider_no_runner.is_available(), expected)

        provider_with_runner = MacAppleMusicProvider(runner=lambda _: "")
        self.assertTrue(provider_with_runner.is_available())

    def test_get_now_playing_when_unavailable(self):
        with patch.object(MacAppleMusicProvider, "is_available", return_value=False):
            provider = MacAppleMusicProvider(runner=lambda _: "playing|||Title|||Artist|||Album")
            self.assertIsNone(provider.get_now_playing())

    def test_get_now_playing_when_music_not_running(self):
        provider = MacAppleMusicProvider(runner=lambda _: "")
        self.assertIsNone(provider.get_now_playing())

    def test_get_now_playing_playing_track(self):
        provider = MacAppleMusicProvider(
            runner=lambda _: "playing|||Test Title|||Test Artist|||Test Album"
        )
        now_playing = provider.get_now_playing()
        self.assertIsNotNone(now_playing)
        self.assertEqual(now_playing.title, "Test Title")
        self.assertEqual(now_playing.artist, "Test Artist")
        self.assertEqual(now_playing.album, "Test Album")
        self.assertTrue(now_playing.is_playing)
        self.assertEqual(now_playing.source, "Apple Music")

    def test_get_now_playing_paused_track(self):
        provider = MacAppleMusicProvider(
            runner=lambda _: "paused|||Paused Song|||Artist Name|||Album Name"
        )
        now_playing = provider.get_now_playing()
        self.assertIsNotNone(now_playing)
        self.assertEqual(now_playing.title, "Paused Song")
        self.assertEqual(now_playing.artist, "Artist Name")
        self.assertFalse(now_playing.is_playing)

    def test_get_now_playing_missing_artist_or_album(self):
        provider = MacAppleMusicProvider(
            runner=lambda _: "playing|||Streaming Stream||||||"
        )
        now_playing = provider.get_now_playing()
        self.assertIsNotNone(now_playing)
        self.assertEqual(now_playing.title, "Streaming Stream")
        self.assertEqual(now_playing.artist, "")
        self.assertEqual(now_playing.album, "")
        self.assertTrue(now_playing.is_playing)

    def test_get_now_playing_empty_title_returns_none(self):
        provider = MacAppleMusicProvider(
            runner=lambda _: "playing||||||Artist Only|||Album"
        )
        self.assertIsNone(provider.get_now_playing())

    def test_get_now_playing_runner_exception_handled_gracefully(self):
        def bad_runner(_):
            raise RuntimeError("osascript execution failed")

        provider = MacAppleMusicProvider(runner=bad_runner)
        self.assertIsNone(provider.get_now_playing())

    def test_default_osascript_runner_returns_stdout_on_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "playing|||Song|||Artist|||Album\n"
        with patch("subprocess.run", return_value=mock_result):
            output = _default_osascript_runner("dummy script")
            self.assertEqual(output, "playing|||Song|||Artist|||Album")

    def test_default_osascript_runner_returns_empty_on_error(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["osascript"], 1.5)):
            output = _default_osascript_runner("dummy script")
            self.assertEqual(output, "")

        mock_failed = MagicMock()
        mock_failed.returncode = 1
        with patch("subprocess.run", return_value=mock_failed):
            output = _default_osascript_runner("dummy script")
            self.assertEqual(output, "")

    def test_coordinator_integration_on_darwin(self):
        with patch("sys.platform", "darwin"), \
             patch.object(MacAppleMusicProvider, "is_available", return_value=True):
            coord = create_default_coordinator()
            self.assertTrue(any(isinstance(p, MacAppleMusicProvider) for p in coord._providers))


if __name__ == "__main__":
    unittest.main()
