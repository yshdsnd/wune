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

_BROWSER_KEYWORDS = ("chrome", "firefox", "msedge", "edge", "opera", "brave", "vivaldi")


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
        self._sessions_token: Any = None
        self._subscribed_session: Any = None
        self._subscribed_app_id: str | None = None
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
                if hasattr(self._manager, "add_sessions_changed"):
                    self._sessions_token = self._manager.add_sessions_changed(
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
            if self._manager is not None:
                if self._session_token is not None and hasattr(self._manager, "remove_current_session_changed"):
                    self._manager.remove_current_session_changed(self._session_token)
                if self._sessions_token is not None and hasattr(self._manager, "remove_sessions_changed"):
                    self._manager.remove_sessions_changed(self._sessions_token)
        except Exception:
            pass
        finally:
            self._session_token = None
            self._sessions_token = None
            if self._custom_manager is None:
                self._manager = None

    @staticmethod
    def _is_session_playing(session: Any) -> bool:
        if session is None or not hasattr(session, "get_playback_info"):
            return False
        try:
            playback_info = session.get_playback_info()
            if playback_info is not None:
                status = getattr(playback_info, "playback_status", None)
                if _PlaybackStatus is not None:
                    return status == _PlaybackStatus.PLAYING
                return bool(getattr(playback_info, "is_playing", status == 4))
        except Exception:
            pass
        return False

    @staticmethod
    def _is_browser(app_id: str) -> bool:
        app_lower = (app_id or "").lower()
        return any(keyword in app_lower for keyword in _BROWSER_KEYWORDS)

    def _sync_session_subscriptions(self, target_session: Any = None) -> None:
        """Subscribe to track/playback changes on the target media session."""
        if self._manager is None:
            return

        if target_session is None:
            target_session = self._find_preliminary_session()

        target_app_id = getattr(target_session, "source_app_user_model_id", None) if target_session else None

        if target_session == self._subscribed_session:
            return
        if target_app_id and target_app_id == self._subscribed_app_id and self._media_props_token is not None:
            return

        self._unsubscribe_session_events()

        if target_session is not None:
            self._subscribed_session = target_session
            self._subscribed_app_id = target_app_id
            try:
                if hasattr(target_session, "add_media_properties_changed"):
                    self._media_props_token = target_session.add_media_properties_changed(
                        self._handle_media_changed
                    )
            except Exception:
                self._media_props_token = None
            try:
                if hasattr(target_session, "add_playback_info_changed"):
                    self._playback_info_token = target_session.add_playback_info_changed(
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
        self._subscribed_app_id = None
        self._media_props_token = None
        self._playback_info_token = None

    def _find_preliminary_session(self) -> Any:
        """Quickly pick the best candidate session based on synchronous status."""
        if self._manager is None:
            return None
        current_session = None
        if hasattr(self._manager, "get_current_session"):
            try:
                current_session = self._manager.get_current_session()
            except Exception:
                current_session = None

        candidates = self._get_ranked_candidates(current_session)
        return candidates[0] if candidates else current_session

    def _get_ranked_candidates(self, current_session: Any) -> list[Any]:
        """Rank sessions by playback status, app type, and active focus."""
        sessions: list[Any] = []
        if hasattr(self._manager, "get_sessions"):
            try:
                sessions = list(self._manager.get_sessions())
            except Exception:
                sessions = []

        cur_app_id = getattr(current_session, "source_app_user_model_id", None) if current_session else None
        has_current = any(
            s == current_session or (cur_app_id and getattr(s, "source_app_user_model_id", None) == cur_app_id)
            for s in sessions
        )
        if current_session is not None and not has_current:
            sessions.insert(0, current_session)

        def _score(s: Any) -> int:
            score = 0
            if self._is_session_playing(s):
                score += 1000
            app_id = getattr(s, "source_app_user_model_id", "") or ""
            if not self._is_browser(app_id):
                score += 100
            if s == current_session or (cur_app_id and getattr(s, "source_app_user_model_id", None) == cur_app_id):
                score += 10
            return score

        sessions.sort(key=_score, reverse=True)
        return sessions

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

        current_session = None
        if hasattr(self._manager, "get_current_session"):
            try:
                current_session = self._manager.get_current_session()
            except Exception:
                current_session = None

        candidates = self._get_ranked_candidates(current_session)
        if not candidates:
            return None

        # 1. Try to find a candidate that has non-empty metadata
        for session in candidates:
            now_playing = await self._extract_now_playing(session)
            if now_playing is not None and now_playing.has_metadata:
                self._sync_session_subscriptions(session)
                return now_playing

        # 2. If none has title/artist, fall back to the highest-ranked candidate
        fallback_session = candidates[0]
        fallback_now_playing = await self._extract_now_playing(fallback_session)
        self._sync_session_subscriptions(fallback_session)
        return fallback_now_playing

    async def _extract_now_playing(self, session: Any) -> NowPlaying | None:
        if session is None:
            return None

        is_playing = self._is_session_playing(session)
        props = None
        if hasattr(session, "try_get_media_properties_async"):
            try:
                props = await session.try_get_media_properties_async()
            except Exception:
                props = None
        elif hasattr(session, "get_media_properties"):
            try:
                props = session.get_media_properties()
            except Exception:
                props = None

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
