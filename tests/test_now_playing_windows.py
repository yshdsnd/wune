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
    def __init__(self, props=None, playback_status=4, app_id="mock_app", raises=None):
        self._props = props if props is not None else MockMediaProperties()
        self._playback_info = MockPlaybackInfo(playback_status)
        self.source_app_user_model_id = app_id
        self._raises = raises
        self.media_properties_changed_handlers = []
        self.playback_info_changed_handlers = []

    def get_playback_info(self):
        if self._raises:
            raise self._raises
        return self._playback_info

    async def try_get_media_properties_async(self):
        if self._raises:
            raise self._raises
        return self._props

    def add_media_properties_changed(self, handler):
        self.media_properties_changed_handlers.append(handler)
        return len(self.media_properties_changed_handlers)

    def remove_media_properties_changed(self, token):
        pass

    def add_playback_info_changed(self, handler):
        self.playback_info_changed_handlers.append(handler)
        return len(self.playback_info_changed_handlers)

    def remove_playback_info_changed(self, token):
        pass


class MockSessionManager:
    def __init__(self, session=None, sessions=None, raises=None):
        self._session = session
        self._sessions = sessions if sessions is not None else ([session] if session else [])
        self._raises = raises
        self.session_changed_handlers = []
        self.sessions_changed_handlers = []
        self.tokens_removed = []
        self.sessions_tokens_removed = []

    def get_current_session(self):
        if self._raises:
            raise self._raises
        return self._session

    def get_sessions(self):
        if self._raises:
            raise self._raises
        return self._sessions

    def add_current_session_changed(self, handler):
        self.session_changed_handlers.append(handler)
        return len(self.session_changed_handlers)

    def remove_current_session_changed(self, token):
        self.tokens_removed.append(token)

    def add_sessions_changed(self, handler):
        self.sessions_changed_handlers.append(handler)
        return len(self.sessions_changed_handlers)

    def remove_sessions_changed(self, token):
        self.sessions_tokens_removed.append(token)


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
        self.assertEqual(len(mgr.sessions_changed_handlers), 1)

        # Trigger current_session_changed handler
        mgr.session_changed_handlers[0](mgr, None)
        self.assertEqual(notifications, [True])

        # Trigger sessions_changed handler
        mgr.sessions_changed_handlers[0](mgr, None)
        self.assertEqual(notifications, [True, True])

        provider.stop()
        self.assertEqual(len(mgr.tokens_removed), 1)
        self.assertEqual(len(mgr.sessions_tokens_removed), 1)

    def test_prioritize_playing_app_over_paused_browser_current_session(self):
        browser_props = MockMediaProperties(title="YouTube Video", artist="Channel")
        browser_session = MockSession(props=browser_props, playback_status=5, app_id="chrome.exe")

        foobar_props = MockMediaProperties(title="Foobar Song", artist="Foobar Artist")
        foobar_session = MockSession(props=foobar_props, playback_status=4, app_id="foobar2000.exe")

        mgr = MockSessionManager(session=browser_session, sessions=[browser_session, foobar_session])
        provider = WindowsGsmtcProvider(manager=mgr)

        now_playing = provider.get_now_playing()
        self.assertIsNotNone(now_playing)
        self.assertEqual(now_playing.title, "Foobar Song")
        self.assertEqual(now_playing.artist, "Foobar Artist")
        self.assertTrue(now_playing.is_playing)

    def test_prioritize_dedicated_player_when_both_playing(self):
        browser_props = MockMediaProperties(title="Web Audio", artist="Site")
        browser_session = MockSession(props=browser_props, playback_status=4, app_id="msedge.exe")

        music_props = MockMediaProperties(title="Apple Music Track", artist="Apple Artist")
        music_session = MockSession(props=music_props, playback_status=4, app_id="AppleInc.AppleMusicWin_cw5n1h2txyewy")

        mgr = MockSessionManager(session=browser_session, sessions=[browser_session, music_session])
        provider = WindowsGsmtcProvider(manager=mgr)

        now_playing = provider.get_now_playing()
        self.assertIsNotNone(now_playing)
        self.assertEqual(now_playing.title, "Apple Music Track")
        self.assertEqual(now_playing.artist, "Apple Artist")

    def test_browser_chosen_when_it_is_playing_and_dedicated_is_paused(self):
        browser_props = MockMediaProperties(title="YouTube Music", artist="Web Artist")
        browser_session = MockSession(props=browser_props, playback_status=4, app_id="firefox.exe")

        foobar_props = MockMediaProperties(title="Old Song", artist="Artist")
        foobar_session = MockSession(props=foobar_props, playback_status=5, app_id="foobar2000.exe")

        mgr = MockSessionManager(session=foobar_session, sessions=[foobar_session, browser_session])
        provider = WindowsGsmtcProvider(manager=mgr)

        now_playing = provider.get_now_playing()
        self.assertIsNotNone(now_playing)
        self.assertEqual(now_playing.title, "YouTube Music")
        self.assertTrue(now_playing.is_playing)

    def test_empty_metadata_candidate_skipped_for_candidate_with_metadata(self):
        empty_props = MockMediaProperties(title="", artist="")
        empty_session = MockSession(props=empty_props, playback_status=4, app_id="foobar2000.exe")

        valid_props = MockMediaProperties(title="Valid Title", artist="Valid Artist")
        valid_session = MockSession(props=valid_props, playback_status=4, app_id="chrome.exe")

        mgr = MockSessionManager(session=empty_session, sessions=[empty_session, valid_session])
        provider = WindowsGsmtcProvider(manager=mgr)

        now_playing = provider.get_now_playing()
        self.assertIsNotNone(now_playing)
        self.assertEqual(now_playing.title, "Valid Title")
        self.assertEqual(now_playing.artist, "Valid Artist")


if __name__ == "__main__":
    unittest.main()

