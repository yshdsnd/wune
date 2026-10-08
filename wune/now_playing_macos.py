"""macOS Apple Music metadata provider."""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Callable

from .now_playing import MetadataProvider, NowPlaying

# Script checks if Music process exists before querying, preventing Music.app from launching
# unintentionally when closed. Uses JavaScript for Automation (JXA) with native JSON serialization
# to prevent delimiter collisions, escaping errors, and support artist-only tracks.
_JXA_QUERY = """
(() => {
    try {
        const se = Application("System Events");
        if (!se.processes.byName("Music").exists()) {
            return "";
        }
        const music = Application("Music");
        const state = String(music.playerState() || "");
        const track = music.currentTrack;
        let title = "";
        let artist = "";
        let album = "";
        if (track) {
            try { title = String(track.name() || ""); } catch (e) {}
            try { artist = String(track.artist() || ""); } catch (e) {}
            try { album = String(track.album() || ""); } catch (e) {}
        }
        return JSON.stringify({
            state: state,
            title: title,
            artist: artist,
            album: album
        });
    } catch (e) {
        return "";
    }
})()
"""

# Kept for backward compatibility or direct AppleScript runner injection
_APPLESCRIPT_QUERY = _JXA_QUERY


def _parse_track_output(output: str) -> NowPlaying | None:
    """Parse structured JSON or legacy delimiter-separated runner output."""
    if not output:
        return None
    trimmed = output.strip()
    if not trimmed:
        return None

    # 1. Try structured JSON format
    if trimmed.startswith("{") and trimmed.endswith("}"):
        try:
            data = json.loads(trimmed)
            state_str = str(data.get("state", "")).strip().lower()
            title = str(data.get("title", "")).strip()
            artist = str(data.get("artist", "")).strip()
            album = str(data.get("album", "")).strip()
            is_playing = state_str == "playing"
            if title or artist:
                return NowPlaying(
                    title=title,
                    artist=artist,
                    album=album,
                    is_playing=is_playing,
                    source="Apple Music",
                )
            return None
        except Exception:
            pass

    # 2. Legacy delimiter-based format ("|||")
    parts = trimmed.split("|||")
    if len(parts) >= 2:
        state_str = parts[0].strip().lower()
        title = parts[1].strip()
        artist = parts[2].strip() if len(parts) > 2 else ""
        album = parts[3].strip() if len(parts) > 3 else ""
        is_playing = state_str == "playing"
        if title or artist:
            return NowPlaying(
                title=title,
                artist=artist,
                album=album,
                is_playing=is_playing,
                source="Apple Music",
            )
    return None


def _default_osascript_runner(script: str, timeout: float = 1.5) -> str:
    """Execute an AppleScript or JXA snippet via osascript with a strict timeout."""
    try:
        cmd = ["osascript"]
        if "Application(" in script or "JSON.stringify" in script:
            cmd.extend(["-l", "JavaScript"])
        cmd.extend(["-e", script])
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
            output = runner(_JXA_QUERY)
        except Exception:
            return None

        return _parse_track_output(output)
