"""Linux and Unix MPRIS2 D-Bus metadata provider."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from typing import Callable

from .now_playing import MetadataProvider, NowPlaying


def _default_cmd_runner(cmd: list[str], timeout: float = 1.0) -> str:
    """Execute a CLI command with a strict timeout and return stdout."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return ""


def parse_playerctl_output(output: str) -> list[NowPlaying]:
    """Parse output from playerctl metadata command.

    Expected format per line:
    status\\ttitle\\tartist\\talbum\\tplayer_name
    """
    tracks: list[NowPlaying] = []
    if not output:
        return tracks
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            status_str = parts[0].strip().lower()
            title = parts[1].strip() if len(parts) > 1 else ""
            artist = parts[2].strip() if len(parts) > 2 else ""
            album = parts[3].strip() if len(parts) > 3 else ""
            player_name = parts[4].strip() if len(parts) > 4 else "MPRIS2"
            is_playing = status_str == "playing"
            if title or artist:
                tracks.append(
                    NowPlaying(
                        title=title,
                        artist=artist,
                        album=album,
                        is_playing=is_playing,
                        source=player_name or "MPRIS2",
                    )
                )
    return tracks


def parse_gdbus_player_names(output: str) -> list[str]:
    """Extract org.mpris.MediaPlayer2.* service names from gdbus ListNames output."""
    if not output:
        return []
    names = re.findall(r"'org\.mpris\.MediaPlayer2\.[a-zA-Z0-9_\.-]+'", output)
    return [name.strip("'") for name in names]


def parse_gdbus_player_properties(output: str, service_name: str = "") -> NowPlaying | None:
    """Parse `gdbus call ... GetAll org.mpris.MediaPlayer2.Player` output into NowPlaying."""
    if not output:
        return None

    status_match = re.search(r"'PlaybackStatus':\s*<['\"]([^'\"]+)['\"]>", output)
    status_str = status_match.group(1).lower() if status_match else ""
    is_playing = status_str == "playing"

    title_match = re.search(r"'xesam:title':\s*<['\"]([^'\"]+)['\"]>", output)
    title = title_match.group(1).strip() if title_match else ""

    artist_list_match = re.search(r"'xesam:artist':\s*<\[(.*?)\]>", output)
    if artist_list_match:
        artists = [
            a.strip().strip("'\"")
            for a in artist_list_match.group(1).split(",")
            if a.strip()
        ]
        artist = ", ".join(artists)
    else:
        artist_match = re.search(r"'xesam:artist':\s*<['\"]([^'\"]+)['\"]>", output)
        artist = artist_match.group(1).strip() if artist_match else ""

    album_match = re.search(r"'xesam:album':\s*<['\"]([^'\"]+)['\"]>", output)
    album = album_match.group(1).strip() if album_match else ""

    source = (
        service_name.replace("org.mpris.MediaPlayer2.", "")
        if service_name
        else "MPRIS2"
    )

    if title or artist:
        return NowPlaying(
            title=title,
            artist=artist,
            album=album,
            is_playing=is_playing,
            source=source,
        )
    return None


def parse_dbus_send_player_names(output: str) -> list[str]:
    """Extract org.mpris.MediaPlayer2.* service names from dbus-send ListNames output."""
    if not output:
        return []
    names = re.findall(r'string "(org\.mpris\.MediaPlayer2\.[a-zA-Z0-9_\.-]+)"', output)
    return names


def parse_dbus_send_player_properties(output: str, service_name: str = "") -> NowPlaying | None:
    """Parse `dbus-send ... GetAll ...` output into NowPlaying."""
    if not output:
        return None

    status_match = re.search(r'string "PlaybackStatus"\s+variant\s+string "([^"]+)"', output)
    status_str = status_match.group(1).lower() if status_match else ""
    is_playing = status_str == "playing"

    title_match = re.search(r'string "xesam:title"\s+variant\s+string "([^"]+)"', output)
    title = title_match.group(1).strip() if title_match else ""

    artist_block = re.search(
        r'string "xesam:artist"\s+variant\s+(?:array \[(.*?)\]|string "([^"]+)")',
        output,
        re.DOTALL,
    )
    if artist_block:
        if artist_block.group(1):
            artists = re.findall(r'string "([^"]+)"', artist_block.group(1))
            artist = ", ".join(artists)
        elif artist_block.group(2):
            artist = artist_block.group(2).strip()
        else:
            artist = ""
    else:
        artist = ""

    album_match = re.search(r'string "xesam:album"\s+variant\s+string "([^"]+)"', output)
    album = album_match.group(1).strip() if album_match else ""

    source = (
        service_name.replace("org.mpris.MediaPlayer2.", "")
        if service_name
        else "MPRIS2"
    )

    if title or artist:
        return NowPlaying(
            title=title,
            artist=artist,
            album=album,
            is_playing=is_playing,
            source=source,
        )
    return None


def select_best_track(tracks: list[NowPlaying]) -> NowPlaying | None:
    """Select the best candidate from a list of tracks, preferring currently playing ones."""
    if not tracks:
        return None
    for track in tracks:
        if track.is_playing:
            return track
    return tracks[0]


class LinuxMpris2Provider(MetadataProvider):
    """Retrieves current track metadata from MPRIS2-compliant players on Linux and Unix."""

    def __init__(
        self,
        name: str = "linux_mpris2",
        priority: int = 50,
        runner: Callable[[list[str]], str] | None = None,
        command_finder: Callable[[str], str | None] | None = None,
    ):
        self._name = name
        self._priority = priority
        self._runner = runner
        self._finder = command_finder or shutil.which

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def is_available(self) -> bool:
        if self._runner is not None:
            return True
        if sys.platform != "linux" and not sys.platform.startswith("freebsd"):
            return False
        return any(self._finder(cmd) for cmd in ("playerctl", "gdbus", "dbus-send"))

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_now_playing(self) -> NowPlaying | None:
        if not self.is_available():
            return None

        runner = self._runner if self._runner is not None else _default_cmd_runner

        # 1. Try playerctl
        if self._runner is not None or self._finder("playerctl"):
            track = self._query_playerctl(runner)
            if track is not None:
                return track

        # 2. Try gdbus
        if self._runner is not None or self._finder("gdbus"):
            track = self._query_gdbus(runner)
            if track is not None:
                return track

        # 3. Try dbus-send
        if self._runner is not None or self._finder("dbus-send"):
            track = self._query_dbus_send(runner)
            if track is not None:
                return track

        return None

    def _query_playerctl(self, runner: Callable[[list[str]], str]) -> NowPlaying | None:
        cmd = [
            "playerctl",
            "-a",
            "metadata",
            "--format",
            "{{status}}\t{{title}}\t{{artist}}\t{{album}}\t{{playerName}}",
        ]
        try:
            output = runner(cmd)
            tracks = parse_playerctl_output(output)
            return select_best_track(tracks)
        except Exception:
            return None

    def _query_gdbus(self, runner: Callable[[list[str]], str]) -> NowPlaying | None:
        cmd_list = [
            "gdbus",
            "call",
            "--session",
            "--dest",
            "org.freedesktop.DBus",
            "--object-path",
            "/org/freedesktop/DBus",
            "--method",
            "org.freedesktop.DBus.ListNames",
        ]
        try:
            list_output = runner(cmd_list)
            services = parse_gdbus_player_names(list_output)
            if not services:
                return None

            tracks: list[NowPlaying] = []
            for service in services:
                cmd_props = [
                    "gdbus",
                    "call",
                    "--session",
                    "--dest",
                    service,
                    "--object-path",
                    "/org/mpris/MediaPlayer2",
                    "--method",
                    "org.freedesktop.DBus.Properties.GetAll",
                    "org.mpris.MediaPlayer2.Player",
                ]
                props_output = runner(cmd_props)
                track = parse_gdbus_player_properties(props_output, service_name=service)
                if track is not None:
                    if track.is_playing:
                        return track
                    tracks.append(track)
            return select_best_track(tracks)
        except Exception:
            return None

    def _query_dbus_send(self, runner: Callable[[list[str]], str]) -> NowPlaying | None:
        cmd_list = [
            "dbus-send",
            "--session",
            "--dest=org.freedesktop.DBus",
            "--type=method_call",
            "--print-reply",
            "/org/freedesktop/DBus",
            "org.freedesktop.DBus.ListNames",
        ]
        try:
            list_output = runner(cmd_list)
            services = parse_dbus_send_player_names(list_output)
            if not services:
                return None

            tracks: list[NowPlaying] = []
            for service in services:
                cmd_props = [
                    "dbus-send",
                    "--session",
                    f"--dest={service}",
                    "--type=method_call",
                    "--print-reply",
                    "/org/mpris/MediaPlayer2",
                    "org.freedesktop.DBus.Properties.GetAll",
                    "string:org.mpris.MediaPlayer2.Player",
                ]
                props_output = runner(cmd_props)
                track = parse_dbus_send_player_properties(props_output, service_name=service)
                if track is not None:
                    if track.is_playing:
                        return track
                    tracks.append(track)
            return select_best_track(tracks)
        except Exception:
            return None
