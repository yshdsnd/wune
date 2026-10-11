import sys
import unittest
from unittest.mock import MagicMock

from wune.now_playing import NowPlaying, NowPlayingCoordinator
from wune.now_playing_music_center import (
    WindowsMusicCenterProvider,
    parse_music_center_title,
    default_find_music_center_title,
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

    @unittest.skipUnless(sys.platform == "win32", "Win32 window API requires Windows")
    def test_default_find_music_center_title_runs_safely(self):
        # Must execute without unhandled exceptions
        title = default_find_music_center_title()
        self.assertTrue(title is None or isinstance(title, str))


if __name__ == "__main__":
    unittest.main()
