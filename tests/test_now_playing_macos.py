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
            runner=lambda _: '{"state":"playing","title":"Streaming Stream","artist":"","album":""}'
        )
        now_playing = provider.get_now_playing()
        self.assertIsNotNone(now_playing)
        self.assertEqual(now_playing.title, "Streaming Stream")
        self.assertEqual(now_playing.artist, "")
        self.assertEqual(now_playing.album, "")
        self.assertTrue(now_playing.is_playing)

    def test_get_now_playing_artist_only_supported(self):
        """Issue 114 #7: Artist-only tracks (e.g. streaming/radio) should not be dropped."""
        provider = MacAppleMusicProvider(
            runner=lambda _: '{"state":"playing","title":"","artist":"Artist Only","album":"Album"}'
        )
        now_playing = provider.get_now_playing()
        self.assertIsNotNone(now_playing)
        self.assertEqual(now_playing.title, "")
        self.assertEqual(now_playing.artist, "Artist Only")
        self.assertEqual(now_playing.album, "Album")
        self.assertTrue(now_playing.has_metadata)

    def test_get_now_playing_empty_metadata_returns_none(self):
        provider = MacAppleMusicProvider(
            runner=lambda _: '{"state":"playing","title":"","artist":"","album":""}'
        )
        self.assertIsNone(provider.get_now_playing())

    def test_get_now_playing_delimiter_in_track_title_preserved_via_json(self):
        """Issue 114 #7: Titles containing '|||' or quotes/newlines must not break fields."""
        import json
        payload = json.dumps({
            "state": "playing",
            "title": "Song Title ||| Special Edition",
            "artist": "Artist Name ||| Co-Artist",
            "album": 'Album "Quoted"',
        })
        provider = MacAppleMusicProvider(runner=lambda _: payload)
        now_playing = provider.get_now_playing()
        self.assertIsNotNone(now_playing)
        self.assertEqual(now_playing.title, "Song Title ||| Special Edition")
        self.assertEqual(now_playing.artist, "Artist Name ||| Co-Artist")
        self.assertEqual(now_playing.album, 'Album "Quoted"')

    def test_get_now_playing_legacy_delimiter_fallback(self):
        """Backward compatibility: legacy '|||' format is still parsed if runner emits it."""
        provider = MacAppleMusicProvider(
            runner=lambda _: "playing|||Legacy Song|||Legacy Artist|||Legacy Album"
        )
        now_playing = provider.get_now_playing()
        self.assertIsNotNone(now_playing)
        self.assertEqual(now_playing.title, "Legacy Song")
        self.assertEqual(now_playing.artist, "Legacy Artist")
        self.assertEqual(now_playing.album, "Legacy Album")

    def test_get_now_playing_runner_exception_handled_gracefully(self):
        def bad_runner(_):
            raise RuntimeError("osascript execution failed")

        provider = MacAppleMusicProvider(runner=bad_runner)
        self.assertIsNone(provider.get_now_playing())

    def test_default_osascript_runner_returns_stdout_on_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"state":"playing","title":"Song","artist":"Artist","album":"Album"}\n'
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            output = _default_osascript_runner("dummy script")
            self.assertEqual(output, '{"state":"playing","title":"Song","artist":"Artist","album":"Album"}')
            # Without JXA markers, -l JavaScript is not appended
            self.assertEqual(mock_run.call_args[0][0], ["osascript", "-e", "dummy script"])

    def test_default_osascript_runner_adds_jxa_flag_for_javascript(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"state":"playing"}'
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            output = _default_osascript_runner("Application('Music').playerState()")
            self.assertEqual(output, '{"state":"playing"}')
            self.assertEqual(mock_run.call_args[0][0], ["osascript", "-l", "JavaScript", "-e", "Application('Music').playerState()"])

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
