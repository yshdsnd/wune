"""macOS Apple Music metadata provider."""
from __future__ import annotations

import subprocess
import sys
from typing import Callable

from .now_playing import MetadataProvider, NowPlaying

# Script checks if Music process exists before querying, preventing Music.app from launching
# unintentionally when closed.
_APPLESCRIPT_QUERY = """
tell application "System Events"
    if not (exists (process "Music")) then return ""
end tell
tell application "Music"
    try
        set pState to player state as string
        set trackName to ""
        set trackArtist to ""
        set trackAlbum to ""
        try
            set trackName to name of current track
        end try
        try
            set trackArtist to artist of current track
        end try
        try
            set trackAlbum to album of current track
        end try
        if trackName is not "" then
            return pState & "|||" & trackName & "|||" & trackArtist & "|||" & trackAlbum
        end if
    end try
end tell
return ""
"""


def _default_osascript_runner(script: str, timeout: float = 1.5) -> str:
    """Execute an AppleScript snippet via osascript with a strict timeout."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return ""


class MacAppleMusicProvider(MetadataProvider):
    """Retrieves current track metadata from Apple Music on macOS."""

    def __init__(
        self,
        name: str = "mac_apple_music",
        priority: int = 50,
        runner: Callable[[str], str] | None = None,
    ):
        self._name = name
        self._priority = priority
        self._runner = runner

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def is_available(self) -> bool:
        if self._runner is not None:
            return True
        return sys.platform == "darwin"

    def start(self) -> None:
        """Start provider lifecycle resources."""
        pass

    def stop(self) -> None:
        """Stop provider lifecycle resources."""
        pass

    def get_now_playing(self) -> NowPlaying | None:
        if not self.is_available():
            return None

        runner = self._runner if self._runner is not None else _default_osascript_runner
        try:
            output = runner(_APPLESCRIPT_QUERY)
        except Exception:
            return None

        if not output:
            return None

        parts = output.split("|||")
        if len(parts) >= 2:
            state_str = parts[0].strip().lower()
            title = parts[1].strip()
            artist = parts[2].strip() if len(parts) > 2 else ""
            album = parts[3].strip() if len(parts) > 3 else ""
            is_playing = state_str == "playing"
            if title:
                return NowPlaying(
                    title=title,
                    artist=artist,
                    album=album,
                    is_playing=is_playing,
                    source="Apple Music",
                )
        return None
