import sys
import unittest
from unittest.mock import MagicMock

from wune.now_playing import NowPlaying, NowPlayingCoordinator
from wune.now_playing_music_center import (
    WindowsMusicCenterProvider,
    parse_music_center_title,
    default_find_music_center_title,
    default_is_music_center_playing,
)


class MusicCenterParsingTests(unittest.TestCase):
    def test_parse_music_center_title_three_parts(self):
        result = parse_music_center_title("We Can't Stop That Way / TM NETWORK / QUANTUM")
        self.assertIsNotNone(result)
        self.assertEqual(result.title, "We Can't Stop That Way")
        self.assertEqual(result.artist, "TM NETWORK")
        self.assertEqual(result.album, "QUANTUM")
        self.assertEqual(result.source, "Music Center for PC")
        self.assertTrue(result.is_playing)
        self.assertEqual(result.display_text(), "We Can't Stop That Way - TM NETWORK")

    def test_parse_music_center_title_two_parts(self):
        result = parse_music_center_title("Get Wild / TM NETWORK")
        self.assertIsNotNone(result)
        self.assertEqual(result.title, "Get Wild")
        self.assertEqual(result.artist, "TM NETWORK")
        self.assertEqual(result.album, "")
        self.assertTrue(result.is_playing)
        self.assertEqual(result.display_text(), "Get Wild - TM NETWORK")

    def test_parse_music_center_title_one_part(self):
        result = parse_music_center_title("Solo Instrumental Track")
        self.assertIsNotNone(result)
        self.assertEqual(result.title, "Solo Instrumental Track")
        self.assertEqual(result.artist, "")
        self.assertEqual(result.album, "")
        self.assertTrue(result.is_playing)
        self.assertEqual(result.display_text(), "Solo Instrumental Track")

    def test_parse_music_center_title_more_than_three_parts(self):
        result = parse_music_center_title("Beyond the Time / TM NETWORK / CAROL / Disc 2")
        self.assertIsNotNone(result)
        self.assertEqual(result.title, "Beyond the Time")
        self.assertEqual(result.artist, "TM NETWORK")
        self.assertEqual(result.album, "CAROL / Disc 2")

    def test_parse_music_center_title_slash_without_spaces_preserved(self):
        result = parse_music_center_title("Thunderstruck / AC/DC / The Razors Edge")
        self.assertIsNotNone(result)
        self.assertEqual(result.title, "Thunderstruck")
        self.assertEqual(result.artist, "AC/DC")
        self.assertEqual(result.album, "The Razors Edge")

    def test_parse_music_center_title_empty_or_whitespace(self):
        self.assertIsNone(parse_music_center_title(""))
        self.assertIsNone(parse_music_center_title("   "))
        self.assertIsNone(parse_music_center_title(None))

    def test_parse_music_center_title_application_name_ignored(self):
        for name in (
            "Music Center for PC",
            "music center for pc",
            "MUSIC CENTER FOR PC",
            "Music Center",
            "Sony Music Center for PC",
            "   Sony Music Center for PC   ",
        ):
            with self.subTest(name=name):
                self.assertIsNone(parse_music_center_title(name))

    def test_parse_music_center_title_playback_flag(self):
        result_default = parse_music_center_title("Track / Artist / Album")
        self.assertIsNotNone(result_default)
        self.assertTrue(result_default.is_playing)

        result_playing = parse_music_center_title("Track / Artist / Album", is_playing=True)
        self.assertIsNotNone(result_playing)
        self.assertTrue(result_playing.is_playing)

        result_paused = parse_music_center_title("Track / Artist / Album", is_playing=False)
        self.assertIsNotNone(result_paused)
        self.assertFalse(result_paused.is_playing)


class MusicCenterProviderTests(unittest.TestCase):
    def test_provider_metadata_and_availability(self):
        provider = WindowsMusicCenterProvider()
        self.assertEqual(provider.name, "windows_music_center")
        self.assertEqual(provider.priority, 40)
        self.assertEqual(provider.is_available(), sys.platform == "win32")

    def test_provider_lifecycle(self):
        mock_finder = MagicMock(return_value="Track / Artist / Album")
        provider = WindowsMusicCenterProvider(title_finder=mock_finder)

        # Before start: returns None without calling finder
        self.assertIsNone(provider.get_now_playing())
        mock_finder.assert_not_called()

        # After start: queries finder
        provider.start()
        if provider.is_available():
            result = provider.get_now_playing()
            self.assertIsNotNone(result)
            self.assertEqual(result.title, "Track")
            mock_finder.assert_called_once()
        else:
            self.assertIsNone(provider.get_now_playing())

        # After stop: returns None
        mock_finder.reset_mock()
        provider.stop()
        self.assertIsNone(provider.get_now_playing())
        mock_finder.assert_not_called()

    def test_provider_finder_returns_empty_or_app_name(self):
        for mock_val in ("", None, "Music Center for PC"):
            with self.subTest(mock_val=mock_val):
                provider = WindowsMusicCenterProvider(title_finder=lambda: mock_val)
                provider.start()
                self.assertIsNone(provider.get_now_playing())

    def test_provider_finder_exception_handled_gracefully(self):
        def bad_finder():
            raise RuntimeError("Windows API error")

        provider = WindowsMusicCenterProvider(title_finder=bad_finder)
        provider.start()
        self.assertIsNone(provider.get_now_playing())

    def test_coordinator_prioritizes_music_center_over_generic_provider(self):
        mc_mock = MagicMock()
        mc_mock.name = "windows_music_center"
        mc_mock.priority = 40
        mc_mock.is_available.return_value = True
        mc_mock.get_now_playing.return_value = NowPlaying(
            title="Music Center Track", artist="Artist A", is_playing=True
        )

        generic_mock = MagicMock()
        generic_mock.name = "generic_provider"
        generic_mock.priority = 50
        generic_mock.is_available.return_value = True
        generic_mock.get_now_playing.return_value = NowPlaying(
            title="Generic Track", artist="Artist B", is_playing=True
        )

        coordinator = NowPlayingCoordinator(providers=[generic_mock, mc_mock])
        with coordinator:
            coordinator.update()
            self.assertEqual(coordinator.current.title, "Music Center Track")

            # When Music Center is idle / returns None, fallback to generic provider
            mc_mock.get_now_playing.return_value = None
            coordinator.update()
            self.assertEqual(coordinator.current.title, "Generic Track")

            # When Music Center is disabled, generic provider is used
            mc_mock.get_now_playing.return_value = NowPlaying(
                title="Music Center Track", artist="Artist A", is_playing=True
            )
            coordinator.disable_provider("windows_music_center")
            coordinator.update()
            self.assertEqual(coordinator.current.title, "Generic Track")

            # Re-enabling restores priority
            coordinator.enable_provider("windows_music_center")
            coordinator.update()
            self.assertEqual(coordinator.current.title, "Music Center Track")

    def test_provider_playback_checker_integration(self):
        mock_finder = MagicMock(return_value="Track / Artist / Album")
        mock_checker = MagicMock(return_value=True)

        provider = WindowsMusicCenterProvider(
            title_finder=mock_finder, playback_checker=mock_checker
        )
        provider.start()
        if provider.is_available():
            result = provider.get_now_playing()
            self.assertIsNotNone(result)
            self.assertTrue(result.is_playing)
            mock_checker.assert_called_once()

            # When checker returns False (paused / inactive)
            mock_checker.reset_mock()
            mock_checker.return_value = False
            result = provider.get_now_playing()
            self.assertIsNotNone(result)
            self.assertFalse(result.is_playing)

            # When checker raises an exception
            mock_checker.reset_mock()
            mock_checker.side_effect = RuntimeError("WASAPI COM Error")
            result = provider.get_now_playing()
            self.assertIsNotNone(result)
            self.assertFalse(result.is_playing)

            # When finder returns None, checker is not even called
            mock_checker.reset_mock()
            mock_finder.return_value = None
            result = provider.get_now_playing()
            self.assertIsNone(result)
            mock_checker.assert_not_called()

    def test_coordinator_prioritizes_playing_generic_over_paused_music_center(self):
        mc_mock = MagicMock()
        mc_mock.name = "windows_music_center"
        mc_mock.priority = 40
        mc_mock.is_available.return_value = True
        # Music Center is paused with metadata remaining in window
        mc_mock.get_now_playing.return_value = NowPlaying(
            title="Music Center Paused Track", artist="Artist A", is_playing=False
        )

        generic_mock = MagicMock()
        generic_mock.name = "generic_provider"
        generic_mock.priority = 50
        generic_mock.is_available.return_value = True
        # Apple Music / GSMTC is actively playing
        generic_mock.get_now_playing.return_value = NowPlaying(
            title="Apple Music Playing Track", artist="Artist B", is_playing=True
        )

        coordinator = NowPlayingCoordinator(providers=[generic_mock, mc_mock], only_playing=True)
        with coordinator:
            coordinator.update()
            # Actively playing generic provider should be chosen
            self.assertEqual(coordinator.current.title, "Apple Music Playing Track")

        # Even with only_playing=False, actively playing generic track wins over paused Music Center
        coordinator_all = NowPlayingCoordinator(providers=[generic_mock, mc_mock], only_playing=False)
        with coordinator_all:
            coordinator_all.update()
            self.assertEqual(coordinator_all.current.title, "Apple Music Playing Track")

            # But when Music Center starts playing, it takes priority (rank 40 < 50)
            mc_mock.get_now_playing.return_value = NowPlaying(
                title="Music Center Active Track", artist="Artist A", is_playing=True
            )
            coordinator_all.update()
            self.assertEqual(coordinator_all.current.title, "Music Center Active Track")

            # When both are paused, with only_playing=False, higher priority Music Center is used as fallback
            mc_mock.get_now_playing.return_value = NowPlaying(
                title="Music Center Paused Track", artist="Artist A", is_playing=False
            )
            generic_mock.get_now_playing.return_value = NowPlaying(
                title="Apple Music Paused Track", artist="Artist B", is_playing=False
            )
            coordinator_all.update()
            self.assertEqual(coordinator_all.current.title, "Music Center Paused Track")

    @unittest.skipUnless(sys.platform == "win32", "Win32 window API requires Windows")
    def test_default_find_music_center_title_runs_safely(self):
        # Must execute without unhandled exceptions
        title = default_find_music_center_title()
        self.assertTrue(title is None or isinstance(title, str))

    @unittest.skipUnless(sys.platform == "win32", "WASAPI Core Audio API requires Windows")
    def test_default_is_music_center_playing_runs_safely(self):
        is_playing = default_is_music_center_playing()
        self.assertIsInstance(is_playing, bool)


if __name__ == "__main__":
    unittest.main()
