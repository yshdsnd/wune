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
        self._subscribed_session: Any = None
        self._media_props_token: Any = None
        self._playback_info_token: Any = None
        self._started = False

    def set_on_change(self, on_change: Callable[[], None] | None) -> None:
        """Set or update the change notification callback."""
        self._on_change = on_change

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
            if self._manager is None and WINRT_AVAILABLE and _SessionManager is not None:
                async def _request_manager():
                    return await _SessionManager.request_async()

                self._manager = asyncio.run(_request_manager())
            if self._manager is not None:
                if hasattr(self._manager, "add_current_session_changed"):
                    self._session_token = self._manager.add_current_session_changed(
                        self._handle_session_changed
                    )
                self._sync_session_subscriptions()
        except Exception:
            pass

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            self._unsubscribe_session_events()
            if self._manager is not None and self._session_token is not None:
                if hasattr(self._manager, "remove_current_session_changed"):
                    self._manager.remove_current_session_changed(self._session_token)
        except Exception:
            pass
        finally:
            self._session_token = None
            if self._custom_manager is None:
                self._manager = None

    def _sync_session_subscriptions(self) -> None:
        """Subscribe to track/playback changes on the currently active media session."""
        if self._manager is None:
            return
        try:
            current_session = self._manager.get_current_session()
        except Exception:
            current_session = None

        if current_session == self._subscribed_session:
            return

        self._unsubscribe_session_events()

        if current_session is not None:
            self._subscribed_session = current_session
            try:
                if hasattr(current_session, "add_media_properties_changed"):
                    self._media_props_token = current_session.add_media_properties_changed(
                        self._handle_media_changed
                    )
            except Exception:
                self._media_props_token = None
            try:
                if hasattr(current_session, "add_playback_info_changed"):
                    self._playback_info_token = current_session.add_playback_info_changed(
                        self._handle_playback_changed
                    )
            except Exception:
                self._playback_info_token = None

    def _unsubscribe_session_events(self) -> None:
        """Unsubscribe from previous active media session events."""
        session = self._subscribed_session
        if session is not None:
            if self._media_props_token is not None and hasattr(session, "remove_media_properties_changed"):
                try:
                    session.remove_media_properties_changed(self._media_props_token)
                except Exception:
                    pass
            if self._playback_info_token is not None and hasattr(session, "remove_playback_info_changed"):
                try:
                    session.remove_playback_info_changed(self._playback_info_token)
                except Exception:
                    pass
        self._subscribed_session = None
        self._media_props_token = None
        self._playback_info_token = None

    def _notify_change(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception:
                pass

    def _handle_session_changed(self, sender: Any = None, args: Any = None) -> None:
        self._sync_session_subscriptions()
        self._notify_change()

    def _handle_media_changed(self, sender: Any = None, args: Any = None) -> None:
        self._notify_change()

    def _handle_playback_changed(self, sender: Any = None, args: Any = None) -> None:
        self._notify_change()

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
            elif WINRT_AVAILABLE and _SessionManager is not None:
                self._manager = await _SessionManager.request_async()

        if self._manager is None:
            return None

        self._sync_session_subscriptions()

        session = self._manager.get_current_session()
        if session is None:
            return None

        is_playing = False
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
                is_playing = False

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
