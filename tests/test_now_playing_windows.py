"""Unit tests for Windows GSMTC metadata provider."""
import asyncio
import unittest
from unittest.mock import MagicMock

from wune.now_playing import NowPlaying
from wune.now_playing_windows import WindowsGsmtcProvider, WINRT_AVAILABLE


class MockMediaProperties:
    def __init__(self, title="Mock Title", artist="Mock Artist", album_title="Mock Album"):
        self.title = title
        self.artist = artist
        self.album_title = album_title


class MockPlaybackInfo:
    def __init__(self, playback_status=4):  # 4 is PLAYING in WinRT
        self.playback_status = playback_status


class MockSession:
    def __init__(self, props=None, playback_status=4, raises=None):
        self._props = props if props is not None else MockMediaProperties()
        self._playback_info = MockPlaybackInfo(playback_status)
        self._raises = raises

    def get_playback_info(self):
        if self._raises:
            raise self._raises
        return self._playback_info

    async def try_get_media_properties_async(self):
        if self._raises:
            raise self._raises
        return self._props


class MockSessionManager:
    def __init__(self, session=None, raises=None):
        self._session = session
        self._raises = raises
        self.session_changed_handlers = []
        self.tokens_removed = []

    def get_current_session(self):
        if self._raises:
            raise self._raises
        return self._session

    def add_current_session_changed(self, handler):
        self.session_changed_handlers.append(handler)
        return len(self.session_changed_handlers)

    def remove_current_session_changed(self, token):
        self.tokens_removed.append(token)


class WindowsGsmtcProviderTests(unittest.TestCase):
    def test_availability(self):
        provider = WindowsGsmtcProvider()
        self.assertEqual(provider.is_available(), WINRT_AVAILABLE)

        injected = WindowsGsmtcProvider(manager=MockSessionManager())
        self.assertTrue(injected.is_available())

    def test_no_active_session_returns_none(self):
        mgr = MockSessionManager(session=None)
        provider = WindowsGsmtcProvider(manager=mgr)
        self.assertIsNone(provider.get_now_playing())

    def test_active_session_returns_metadata(self):
        props = MockMediaProperties(title="Test Song", artist="Test Band", album_title="Test Album")
        session = MockSession(props=props, playback_status=4)
        mgr = MockSessionManager(session=session)

        provider = WindowsGsmtcProvider(manager=mgr)
        now_playing = provider.get_now_playing()

        self.assertIsNotNone(now_playing)
        self.assertEqual(now_playing.title, "Test Song")
        self.assertEqual(now_playing.artist, "Test Band")
        self.assertEqual(now_playing.album, "Test Album")
        self.assertEqual(now_playing.source, "GSMTC")
        self.assertTrue(now_playing.is_playing)
        self.assertEqual(now_playing.display_text(), "Test Song - Test Band")

    def test_paused_status_reflected(self):
        props = MockMediaProperties(title="Paused Track", artist="Artist")
        session = MockSession(props=props, playback_status=5)  # 5 is PAUSED
        mgr = MockSessionManager(session=session)

        provider = WindowsGsmtcProvider(manager=mgr)
        now_playing = provider.get_now_playing()

        self.assertIsNotNone(now_playing)
        self.assertEqual(now_playing.title, "Paused Track")
        self.assertFalse(now_playing.is_playing)

    def test_session_exception_isolated(self):
        session = MockSession(raises=RuntimeError("Transient COM disconnection"))
        mgr = MockSessionManager(session=session)

        provider = WindowsGsmtcProvider(manager=mgr)
        self.assertIsNone(provider.get_now_playing())

    def test_manager_exception_isolated(self):
        mgr = MockSessionManager(raises=RuntimeError("Manager acquisition failed"))

        provider = WindowsGsmtcProvider(manager=mgr)
        self.assertIsNone(provider.get_now_playing())

    def test_lifecycle_and_event_handling(self):
        notifications = []
        mgr = MockSessionManager()
        provider = WindowsGsmtcProvider(manager=mgr, on_change=lambda: notifications.append(True))

        provider.start()
        self.assertEqual(len(mgr.session_changed_handlers), 1)

        # Trigger event handler
        mgr.session_changed_handlers[0](mgr, None)
        self.assertEqual(notifications, [True])

        provider.stop()
        self.assertEqual(len(mgr.tokens_removed), 1)


if __name__ == "__main__":
    unittest.main()
