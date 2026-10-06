"""Windows GSMTC (Global System Media Transport Controls) metadata provider."""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Callable

from .now_playing import MetadataProvider, NowPlaying

try:
    if sys.platform == "win32":
        from winrt.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as _SessionManager,
            GlobalSystemMediaTransportControlsSessionPlaybackStatus as _PlaybackStatus,
        )
        WINRT_AVAILABLE = True
    else:
        _SessionManager = None
        _PlaybackStatus = None
        WINRT_AVAILABLE = False
except (ImportError, ModuleNotFoundError):
    _SessionManager = None
    _PlaybackStatus = None
    WINRT_AVAILABLE = False


class WindowsGsmtcProvider(MetadataProvider):
    """Retrieves current track metadata from Windows GSMTC."""

    def __init__(
        self,
        manager: Any = None,
        name: str = "windows_gsmtc",
        priority: int = 50,
        on_change: Callable[[], None] | None = None,
    ):
        self._name = name
        self._priority = priority
        self._custom_manager = manager
        self._manager: Any = manager
        self._on_change = on_change
        self._session_token: Any = None
        self._started = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def is_available(self) -> bool:
        if self._custom_manager is not None:
            return True
        return WINRT_AVAILABLE

    def start(self) -> None:
        if self._started or not self.is_available():
            return
        self._started = True
        try:
            if self._manager is None and WINRT_AVAILABLE:
                self._manager = asyncio.run(_SessionManager.request_async())
            if self._manager is not None and hasattr(self._manager, "add_current_session_changed"):
                self._session_token = self._manager.add_current_session_changed(
                    self._handle_session_changed
                )
        except Exception:
            pass

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            if self._manager is not None and self._session_token is not None:
                if hasattr(self._manager, "remove_current_session_changed"):
                    self._manager.remove_current_session_changed(self._session_token)
        except Exception:
            pass
        finally:
            self._session_token = None
            if self._custom_manager is None:
                self._manager = None

    def _handle_session_changed(self, sender: Any, args: Any) -> None:
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception:
                pass

    def get_now_playing(self) -> NowPlaying | None:
        if not self.is_available():
            return None
        try:
            return asyncio.run(self._fetch_async())
        except Exception:
            return None

    async def _fetch_async(self) -> NowPlaying | None:
        if self._manager is None:
            if self._custom_manager is not None:
                self._manager = self._custom_manager
            elif WINRT_AVAILABLE:
                self._manager = await _SessionManager.request_async()

        if self._manager is None:
            return None

        session = self._manager.get_current_session()
        if session is None:
            return None

        is_playing = True
        if hasattr(session, "get_playback_info"):
            try:
                playback_info = session.get_playback_info()
                if playback_info is not None:
                    status = playback_info.playback_status
                    if _PlaybackStatus is not None:
                        is_playing = status == _PlaybackStatus.PLAYING
                    else:
                        is_playing = bool(getattr(playback_info, "is_playing", status == 4))
            except Exception:
                pass

        props = None
        if hasattr(session, "try_get_media_properties_async"):
            props = await session.try_get_media_properties_async()
        elif hasattr(session, "get_media_properties"):
            props = session.get_media_properties()

        if props is None:
            return None

        title = getattr(props, "title", "") or ""
        artist = getattr(props, "artist", "") or ""
        album = getattr(props, "album_title", "") or ""

        return NowPlaying(
            title=str(title).strip(),
            artist=str(artist).strip(),
            album=str(album).strip(),
            source="GSMTC",
            is_playing=is_playing,
        )
