"""Tests for Linux and Unix MPRIS2 Now Playing metadata provider."""
from unittest.mock import MagicMock, patch
import sys
import unittest

from wune.now_playing import NowPlaying
from wune.now_playing_mpris import (
    LinuxMpris2Provider,
    parse_playerctl_output,
    parse_gdbus_player_names,
    parse_gdbus_player_properties,
    parse_dbus_send_player_names,
    parse_dbus_send_player_properties,
    select_best_track,
)


class MprisParserTests(unittest.TestCase):
    """Test parsing logic for playerctl, gdbus, and dbus-send command outputs."""

    def test_parse_playerctl_output(self):
        output = (
            "Paused\tOld Track\tOld Artist\tOld Album\tvlc\n"
            "Playing\tBlinding Lights\tThe Weeknd\tAfter Hours\tspotify\n"
        )
        tracks = parse_playerctl_output(output)
        self.assertEqual(len(tracks), 2)

        self.assertEqual(tracks[0].title, "Old Track")
        self.assertEqual(tracks[0].artist, "Old Artist")
        self.assertEqual(tracks[0].album, "Old Album")
        self.assertFalse(tracks[0].is_playing)
        self.assertEqual(tracks[0].source, "vlc")

        self.assertEqual(tracks[1].title, "Blinding Lights")
        self.assertEqual(tracks[1].artist, "The Weeknd")
        self.assertEqual(tracks[1].album, "After Hours")
        self.assertTrue(tracks[1].is_playing)
        self.assertEqual(tracks[1].source, "spotify")

        best = select_best_track(tracks)
        self.assertIsNotNone(best)
        self.assertTrue(best.is_playing)
        self.assertEqual(best.title, "Blinding Lights")

    def test_parse_playerctl_output_empty_or_whitespace(self):
        self.assertEqual(parse_playerctl_output(""), [])
        self.assertEqual(parse_playerctl_output("   \n\n  "), [])

    def test_parse_gdbus_player_names(self):
        output = (
            "(['org.freedesktop.DBus', 'org.mpris.MediaPlayer2.spotify', "
            "'org.mpris.MediaPlayer2.rhythmbox.instance1', 'org.gnome.SessionManager'],)"
        )
        names = parse_gdbus_player_names(output)
        self.assertEqual(
            names,
            ["org.mpris.MediaPlayer2.spotify", "org.mpris.MediaPlayer2.rhythmbox.instance1"],
        )

    def test_parse_gdbus_player_properties_playing_with_artist_array(self):
        output = (
            "({'PlaybackStatus': <'Playing'>, 'Metadata': <{"
            "'mpris:trackid': <'spotify:track:123'>, "
            "'xesam:title': <'Stay With Me'>, "
            "'xesam:artist': <['Miki Matsubara']>, "
            "'xesam:album': <'Pocket Park'>"
            "}>, 'Volume': <1.0>},)"
        )
        track = parse_gdbus_player_properties(output, "org.mpris.MediaPlayer2.spotify")
        self.assertIsNotNone(track)
        self.assertEqual(track.title, "Stay With Me")
        self.assertEqual(track.artist, "Miki Matsubara")
        self.assertEqual(track.album, "Pocket Park")
        self.assertTrue(track.is_playing)
        self.assertEqual(track.source, "spotify")

    def test_parse_gdbus_player_properties_multiple_artists_and_paused(self):
        output = (
            "({'PlaybackStatus': <'Paused'>, 'Metadata': <{"
            "'xesam:title': <'Collaboration Song'>, "
            "'xesam:artist': <['Artist A', 'Artist B']>, "
            "'xesam:album': <'Split Album'>"
            "}>},)"
        )
        track = parse_gdbus_player_properties(output, "org.mpris.MediaPlayer2.rhythmbox")
        self.assertIsNotNone(track)
        self.assertEqual(track.title, "Collaboration Song")
        self.assertEqual(track.artist, "Artist A, Artist B")
        self.assertEqual(track.album, "Split Album")
        self.assertFalse(track.is_playing)
        self.assertEqual(track.source, "rhythmbox")

    def test_parse_dbus_send_player_names(self):
        output = """
method return time=1700000000.123 sender=org.freedesktop.DBus -> destination=:1.50
   array [
      string "org.freedesktop.DBus"
      string "org.mpris.MediaPlayer2.audacious"
      string "org.mpris.MediaPlayer2.vlc"
      string "org.freedesktop.Notifications"
   ]
"""
        names = parse_dbus_send_player_names(output)
        self.assertEqual(
            names,
            ["org.mpris.MediaPlayer2.audacious", "org.mpris.MediaPlayer2.vlc"],
        )

    def test_parse_dbus_send_player_properties(self):
        output = """
method return time=1700000000.456 sender=:1.60 -> destination=:1.50
   array [
      dict entry(
         string "PlaybackStatus"
         variant string "Playing"
      )
      dict entry(
         string "Metadata"
         variant array [
            dict entry(
               string "xesam:title"
               variant string "Plastic Love"
            )
            dict entry(
               string "xesam:artist"
               variant array [
                  string "Mariya Takeuchi"
               ]
            )
            dict entry(
               string "xesam:album"
               variant string "Variety"
            )
         ]
      )
   ]
"""
        track = parse_dbus_send_player_properties(output, "org.mpris.MediaPlayer2.audacious")
        self.assertIsNotNone(track)
        self.assertEqual(track.title, "Plastic Love")
        self.assertEqual(track.artist, "Mariya Takeuchi")
        self.assertEqual(track.album, "Variety")
        self.assertTrue(track.is_playing)
        self.assertEqual(track.source, "audacious")


class LinuxMpris2ProviderTests(unittest.TestCase):
    """Test LinuxMpris2Provider lifecycle, availability, and CLI routing."""

    def test_availability_with_injected_runner(self):
        provider = LinuxMpris2Provider(runner=lambda cmd: "")
        self.assertTrue(provider.is_available())

    def test_availability_on_non_linux(self):
        with patch("sys.platform", "win32"):
            provider = LinuxMpris2Provider(command_finder=lambda cmd: "/usr/bin/" + cmd)
            self.assertFalse(provider.is_available())

    def test_availability_on_linux_when_commands_found(self):
        with patch("sys.platform", "linux"):
            provider = LinuxMpris2Provider(
                command_finder=lambda cmd: "/usr/bin/playerctl" if cmd == "playerctl" else None
            )
            self.assertTrue(provider.is_available())

    def test_availability_on_freebsd_when_commands_found(self):
        with patch("sys.platform", "freebsd14"):
            provider = LinuxMpris2Provider(
                command_finder=lambda cmd: "/usr/local/bin/gdbus" if cmd == "gdbus" else None
            )
            self.assertTrue(provider.is_available())

    def test_availability_false_when_no_commands_present(self):
        with patch("sys.platform", "linux"):
            provider = LinuxMpris2Provider(command_finder=lambda cmd: None)
            self.assertFalse(provider.is_available())

    def test_query_via_playerctl(self):
        def fake_runner(cmd):
            if cmd[0] == "playerctl":
                return "Playing\tFlyday Chinatown\tYasuha\tTransit\tplayerctl"
            return ""

        provider = LinuxMpris2Provider(runner=fake_runner)
        track = provider.get_now_playing()
        self.assertIsNotNone(track)
        self.assertEqual(track.title, "Flyday Chinatown")
        self.assertEqual(track.artist, "Yasuha")
        self.assertEqual(track.album, "Transit")
        self.assertTrue(track.is_playing)

    def test_query_fallback_to_gdbus(self):
        def fake_runner(cmd):
            if cmd[0] == "playerctl":
                return ""  # playerctl not returning anything
            if cmd[0] == "gdbus" and "ListNames" in cmd[-1]:
                return "(['org.mpris.MediaPlayer2.clementine'],)"
            if cmd[0] == "gdbus" and "GetAll" in cmd[-2]:
                return (
                    "({'PlaybackStatus': <'Playing'>, 'Metadata': <{"
                    "'xesam:title': <'Midnight Pretenders'>, "
                    "'xesam:artist': <['Tomoko Aran']>, "
                    "'xesam:album': <'Fuyukukan'>"
                    "}>},)"
                )
            return ""

        provider = LinuxMpris2Provider(
            runner=fake_runner,
            command_finder=lambda cmd: "/usr/bin/" + cmd if cmd in ("gdbus",) else None,
        )
        track = provider.get_now_playing()
        self.assertIsNotNone(track)
        self.assertEqual(track.title, "Midnight Pretenders")
        self.assertEqual(track.artist, "Tomoko Aran")
        self.assertTrue(track.is_playing)
        self.assertEqual(track.source, "clementine")

    def test_query_fallback_to_dbus_send(self):
        def fake_runner(cmd):
            if cmd[0] in ("playerctl", "gdbus"):
                return ""
            if cmd[0] == "dbus-send" and "ListNames" in cmd[-1]:
                return 'array [ string "org.mpris.MediaPlayer2.mpv" ]'
            if cmd[0] == "dbus-send" and "GetAll" in cmd[-2]:
                return """
array [
   dict entry(
      string "PlaybackStatus"
      variant string "Playing"
   )
   dict entry(
      string "Metadata"
      variant array [
         dict entry(
            string "xesam:title"
            variant string "Telephone Number"
         )
         dict entry(
            string "xesam:artist"
            variant string "Junko Ohashi"
         )
      ]
   )
]
"""
            return ""

        provider = LinuxMpris2Provider(
            runner=fake_runner,
            command_finder=lambda cmd: "/usr/bin/" + cmd if cmd in ("dbus-send",) else None,
        )
        track = provider.get_now_playing()
        self.assertIsNotNone(track)
        self.assertEqual(track.title, "Telephone Number")
        self.assertEqual(track.artist, "Junko Ohashi")
        self.assertTrue(track.is_playing)
        self.assertEqual(track.source, "mpv")

    def test_returns_none_when_all_runners_fail(self):
        def failing_runner(cmd):
            raise RuntimeError("Command failed")

        provider = LinuxMpris2Provider(runner=failing_runner)
        self.assertIsNone(provider.get_now_playing())

    def test_start_and_stop_noop(self):
        provider = LinuxMpris2Provider(runner=lambda cmd: "")
        provider.start()
        provider.stop()
        self.assertEqual(provider.name, "linux_mpris2")
        self.assertEqual(provider.priority, 50)
