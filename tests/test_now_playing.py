"""Unit tests for the Now Playing metadata model, providers, and coordinator."""
from dataclasses import FrozenInstanceError
import time
import unittest
from unittest.mock import patch

from wune.now_playing import (
    MetadataProvider,
    NowPlaying,
    NowPlayingCoordinator,
    create_default_coordinator,
)


class DummyProvider(MetadataProvider):
    """Configurable mock provider for test cases."""

    def __init__(
        self,
        name: str = "dummy",
        priority: int = 100,
        available: bool = True,
        result: NowPlaying | None = None,
        raises: Exception | None = None,
    ):
        self._name = name
        self._priority = priority
        self.available = available
        self.result = result
        self.raises = raises
        self.started = False
        self.stopped = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def is_available(self) -> bool:
        if self.raises:
            raise self.raises
        return self.available

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def get_now_playing(self) -> NowPlaying | None:
        if self.raises:
            raise self.raises
        return self.result


class NowPlayingModelTests(unittest.TestCase):
    def test_default_values(self):
        np = NowPlaying()
        self.assertEqual(np.title, "")
        self.assertEqual(np.artist, "")
        self.assertEqual(np.album, "")
        self.assertEqual(np.source, "")
        self.assertTrue(np.is_playing)
        self.assertEqual(np.raw_text, "")
        self.assertFalse(np.has_metadata)
        self.assertFalse(bool(np))
        self.assertEqual(np.display_text(), "")

    def test_immutability(self):
        np = NowPlaying(title="Song", artist="Artist")
        with self.assertRaises(FrozenInstanceError):
            np.title = "Another"

    def test_has_metadata_and_bool(self):
        self.assertFalse(NowPlaying(title="   ").has_metadata)
        self.assertFalse(NowPlaying(artist="   ").has_metadata)
        self.assertTrue(NowPlaying(title="Track").has_metadata)
        self.assertTrue(NowPlaying(artist="Artist").has_metadata)
        self.assertTrue(bool(NowPlaying(title="Track")))

    def test_display_text_formatting(self):
        self.assertEqual(
            NowPlaying(title=" Song ", artist=" Artist ").display_text(),
            "Song - Artist",
        )
        self.assertEqual(
            NowPlaying(title="Song", artist="Artist").display_text(separator=" / "),
            "Song / Artist",
        )
        self.assertEqual(
            NowPlaying(title="Song", artist="").display_text(),
            "Song",
        )
        self.assertEqual(
            NowPlaying(title="", artist="Artist").display_text(),
            "Artist",
        )
        self.assertEqual(
            NowPlaying(title="   ", artist="   ").display_text(),
            "",
        )


class NowPlayingCoordinatorTests(unittest.TestCase):
    def test_provider_registration_order_by_priority(self):
        p_low = DummyProvider(name="low", priority=100)
        p_high = DummyProvider(name="high", priority=10)
        p_mid = DummyProvider(name="mid", priority=50)

        coordinator = NowPlayingCoordinator()
        coordinator.register_provider(p_low)
        coordinator.register_provider(p_high)
        coordinator.register_provider(p_mid)

        self.assertEqual(
            [p.name for p in coordinator._providers],
            ["high", "mid", "low"],
        )

    def test_priority_resolution(self):
        track_high = NowPlaying(title="High Priority Song", artist="Artist A")
        track_low = NowPlaying(title="Low Priority Song", artist="Artist B")

        p_low = DummyProvider(name="generic_gsmtc", priority=100, result=track_low)
        p_high = DummyProvider(name="app_specific", priority=10, result=track_high)

        coordinator = NowPlayingCoordinator([p_low, p_high])
        resolved = coordinator.update()
        self.assertEqual(resolved, track_high)
        self.assertEqual(coordinator.current, track_high)

    def test_fallback_when_higher_priority_has_no_metadata(self):
        track_low = NowPlaying(title="Generic Song", artist="Artist")
        p_high = DummyProvider(name="app_specific", priority=10, result=None)
        p_low = DummyProvider(name="generic_gsmtc", priority=100, result=track_low)

        coordinator = NowPlayingCoordinator([p_high, p_low])
        resolved = coordinator.update()
        self.assertEqual(resolved, track_low)

    def test_skips_unavailable_provider(self):
        track_avail = NowPlaying(title="Available Song", artist="Artist")
        track_unavail = NowPlaying(title="Hidden Song", artist="Artist")

        p_unavail = DummyProvider(
            name="p1", priority=10, available=False, result=track_unavail
        )
        p_avail = DummyProvider(
            name="p2", priority=50, available=True, result=track_avail
        )

        coordinator = NowPlayingCoordinator([p_unavail, p_avail])
        resolved = coordinator.update()
        self.assertEqual(resolved, track_avail)

    def test_ignores_non_playing_when_configured(self):
        track_paused = NowPlaying(title="Paused Song", artist="Artist", is_playing=False)
        p = DummyProvider(name="p1", priority=10, result=track_paused)

        coordinator = NowPlayingCoordinator([p], only_playing=True)
        self.assertIsNone(coordinator.update())

        coordinator_all = NowPlayingCoordinator([p], only_playing=False)
        self.assertEqual(coordinator_all.update(), track_paused)

    def test_exception_isolation_does_not_crash_coordinator(self):
        broken = DummyProvider(
            name="broken", priority=10, raises=RuntimeError("COM failure")
        )
        healthy = DummyProvider(
            name="healthy",
            priority=50,
            result=NowPlaying(title="Resilient Track", artist="Artist"),
        )

        coordinator = NowPlayingCoordinator([broken, healthy])
        resolved = coordinator.update()
        self.assertEqual(resolved.title, "Resilient Track")

    def test_provider_enable_disable_and_unregister(self):
        track1 = NowPlaying(title="Track 1", artist="Artist 1")
        track2 = NowPlaying(title="Track 2", artist="Artist 2")
        p1 = DummyProvider(name="p1", priority=10, result=track1)
        p2 = DummyProvider(name="p2", priority=20, result=track2)

        coordinator = NowPlayingCoordinator([p1, p2])
        self.assertEqual(coordinator.update(), track1)

        coordinator.disable_provider("p1")
        self.assertFalse(coordinator.is_provider_enabled("p1"))
        self.assertEqual(coordinator.update(), track2)

        coordinator.enable_provider("p1")
        self.assertTrue(coordinator.is_provider_enabled("p1"))
        self.assertEqual(coordinator.update(), track1)

        coordinator.unregister_provider("p1")
        self.assertEqual(coordinator.update(), track2)

    def test_listener_notifications_on_change(self):
        events = []
        coordinator = NowPlayingCoordinator()
        coordinator.add_listener(events.append)

        p = DummyProvider(name="p", priority=10, result=None)
        coordinator.register_provider(p)

        coordinator.update()
        self.assertEqual(events, [])  # None -> None does not notify

        track = NowPlaying(title="New Track", artist="Artist")
        p.result = track
        coordinator.update()
        self.assertEqual(events, [track])

        # Same track -> no duplicate notification
        coordinator.update()
        self.assertEqual(events, [track])

        # Track stopped
        p.result = None
        coordinator.update()
        self.assertEqual(events, [track, None])

        coordinator.remove_listener(events.append)
        p.result = track
        coordinator.update()
        self.assertEqual(events, [track, None])

    def test_lifecycle_and_background_worker(self):
        track = NowPlaying(title="Async Song", artist="Async Artist")
        p = DummyProvider(name="async_p", priority=10, result=track)

        with NowPlayingCoordinator([p], poll_interval=0.1) as coordinator:
            self.assertTrue(p.started)
            # Give background worker a brief moment to update
            deadline = time.time() + 1.0
            while coordinator.current is None and time.time() < deadline:
                time.sleep(0.02)
            self.assertEqual(coordinator.current, track)

        self.assertTrue(p.stopped)
        self.assertIsNone(coordinator.current)

    def test_create_default_coordinator(self):
        coord = create_default_coordinator()
        self.assertIsInstance(coord, NowPlayingCoordinator)
        # Should gracefully return None when no providers or no media playing
        self.assertIsNone(coord.update())

    def test_default_coordinator_factory_on_darwin(self):
        with patch("sys.platform", "darwin"):
            with patch("wune.now_playing_macos.MacAppleMusicProvider.is_available", return_value=True):
                coord = create_default_coordinator()
                self.assertIsInstance(coord, NowPlayingCoordinator)
                self.assertTrue(any(p.name == "mac_apple_music" for p in coord._providers))


class NowPlayingRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pygame as pg

        pg.font.init()

    def test_renderer_now_playing_drawing(self):
        import pygame as pg
        from wune.config import Config
        from wune.renderer import LedBarRenderer

        cfg = Config(show_now_playing=True)
        surf = pg.Surface((800, 600))
        renderer = LedBarRenderer(surf, cfg)

        self.assertIsNotNone(renderer.now_playing_rect())

        # Baseline: empty text (frame drawn, no text)
        renderer.now_playing_text = ""
        renderer.draw_panel()
        baseline_bytes = pg.image.tobytes(surf, "RGB")

        # Set track text
        renderer.now_playing_text = "Now playing:  Sample Track - Sample Artist"
        renderer.draw_panel()
        with_text_bytes = pg.image.tobytes(surf, "RGB")
        self.assertNotEqual(baseline_bytes, with_text_bytes)

        # Disabled setting: frame removed
        cfg.show_now_playing = False
        self.assertIsNone(renderer.now_playing_rect())
        renderer.draw_panel()
        disabled_bytes = pg.image.tobytes(surf, "RGB")
        self.assertNotEqual(baseline_bytes, disabled_bytes)

    def test_app_update_now_playing_text_format(self):
        from unittest.mock import Mock
        from wune.app import App
        from wune.config import Config
        from wune.now_playing import NowPlaying

        mock_screen = Mock()
        mock_screen.get_size.return_value = (800, 600)

        app = Mock(spec=App)
        app.cfg = Config(show_now_playing=True)
        app.renderer = Mock()
        app.now_playing = Mock()

        # Track with artist
        app.now_playing.current = NowPlaying(title="Song", artist="Artist")
        App.update_now_playing_text(app)
        self.assertEqual(app.renderer.now_playing_text, "Now playing:  Song - Artist")

        # Track without artist
        app.now_playing.current = NowPlaying(title="Song Alone")
        App.update_now_playing_text(app)
        self.assertEqual(app.renderer.now_playing_text, "Now playing:  Song Alone")

        # No track playing
        app.now_playing.current = None
        App.update_now_playing_text(app)
        self.assertEqual(app.renderer.now_playing_text, "Now playing:  -")

        # Disabled setting
        app.cfg.show_now_playing = False
        App.update_now_playing_text(app)
        self.assertEqual(app.renderer.now_playing_text, "")

    def test_renderer_long_text_truncated_safely(self):
        import pygame as pg
        from wune.config import Config
        from wune.renderer import LedBarRenderer

        cfg = Config(show_now_playing=True)
        surf = pg.Surface((800, 600))
        renderer = LedBarRenderer(surf, cfg)

        renderer.now_playing_text = "Now playing:  " + "A" * 1000
        renderer.draw_panel()

    def test_layout_reserves_space_for_now_playing_and_shifts_spectrum(self):
        import pygame as pg
        from wune.config import Config
        from wune.layout import calculate_layout, minimum_window_size

        cfg_without = Config(show_now_playing=False)
        cfg_with = Config(show_now_playing=True)
        size = (1280, 800)

        layout_without = calculate_layout(size, cfg_without)
        layout_with = calculate_layout(size, cfg_with)

        self.assertIsNone(layout_without.now_playing_rect)
        self.assertIsNotNone(layout_with.now_playing_rect)

        np_rect = pg.Rect(layout_with.now_playing_rect)
        # Menu button is at y=14..44; frame must be below it
        self.assertGreaterEqual(np_rect.top, 44)
        self.assertEqual(np_rect.height, cfg_with.info_height)

        # Plots must be shifted down when now_playing is enabled
        self.assertGreater(layout_with.plots[0][1], layout_without.plots[0][1])

        # Channel label "L" is drawn above plots[0] (group_y = plots[0].y - header)
        # Verify that the channel area (including L label) is strictly below now_playing_rect
        header = max(44, cfg_with.header_reserved)
        channel_label_top = layout_with.plots[0][1] - header
        self.assertGreater(channel_label_top, np_rect.bottom)

        # Spectrum plot rects must never intersect now_playing_rect
        for plot in layout_with.plots:
            self.assertFalse(np_rect.colliderect(pg.Rect(plot)))


class MockNonCoroutineAwaitable:
    """Awaitable with __await__ that is deliberately NOT an asyncio coroutine."""

    def __init__(self, result):
        self._result = result

    def __await__(self):
        def _gen():
            yield
            return self._result

        return _gen()


class MockSessionManager:
    def __init__(self, current_session=None):
        self._current_session = current_session
        self.session_changed_handlers = []
        self.session_tokens = []

    def get_current_session(self):
        return self._current_session

    def add_current_session_changed(self, handler):
        self.session_changed_handlers.append(handler)
        token = object()
        self.session_tokens.append(token)
        return token

    def remove_current_session_changed(self, token):
        if token in self.session_tokens:
            idx = self.session_tokens.index(token)
            self.session_tokens.pop(idx)
            self.session_changed_handlers.pop(idx)


class MockMediaSession:
    def __init__(self, title="Mock Title", artist="Mock Artist", is_playing=True, fail_playback=False):
        self.title = title
        self.artist = artist
        self.is_playing = is_playing
        self.fail_playback = fail_playback
        self.media_changed_handlers = []
        self.playback_changed_handlers = []
        self.media_tokens = []
        self.playback_tokens = []

    def get_playback_info(self):
        if self.fail_playback:
            raise RuntimeError("Playback RPC error")

        class Status:
            pass

        s = Status()
        s.playback_status = 4 if self.is_playing else 5
        s.is_playing = self.is_playing
        return s

    async def try_get_media_properties_async(self):
        class Props:
            pass

        p = Props()
        p.title = self.title
        p.artist = self.artist
        p.album_title = "Mock Album"
        return p

    def add_media_properties_changed(self, handler):
        self.media_changed_handlers.append(handler)
        token = object()
        self.media_tokens.append(token)
        return token

    def remove_media_properties_changed(self, token):
        if token in self.media_tokens:
            idx = self.media_tokens.index(token)
            self.media_tokens.pop(idx)
            self.media_changed_handlers.pop(idx)

    def add_playback_info_changed(self, handler):
        self.playback_changed_handlers.append(handler)
        token = object()
        self.playback_tokens.append(token)
        return token

    def remove_playback_info_changed(self, token):
        if token in self.playback_tokens:
            idx = self.playback_tokens.index(token)
            self.playback_tokens.pop(idx)
            self.playback_changed_handlers.pop(idx)


class NowPlayingCoreLogicFixesTests(unittest.TestCase):
    """Targeted regression tests for Issue #114 review findings (Point 1, 2, 3, 6)."""

    def test_gsmtc_start_with_non_coroutine_awaitable_and_event_wiring(self):
        """Issue 114 #1: PyWinRT returns awaitable, not coroutine. Verify start() succeeds and wires events."""
        import threading
        from wune import now_playing_windows
        from wune.now_playing_windows import WindowsGsmtcProvider

        mock_session = MockMediaSession(title="Test Song", artist="Test Artist")
        mock_mgr = MockSessionManager(current_session=mock_session)

        # Mock _SessionManager.request_async returning non-coroutine awaitable
        class FakeSessionManagerClass:
            @classmethod
            def request_async(cls):
                return MockNonCoroutineAwaitable(mock_mgr)

        with patch.object(now_playing_windows, "WINRT_AVAILABLE", True):
            with patch.object(now_playing_windows, "_SessionManager", FakeSessionManagerClass):
                provider = WindowsGsmtcProvider()
                changed_events = []
                provider.set_on_change(lambda: changed_events.append(True))
                provider.start()

                # Verify manager was obtained successfully through coroutine wrapper
                self.assertIsNotNone(provider._manager)
                self.assertEqual(len(mock_mgr.session_changed_handlers), 1)
                self.assertEqual(len(mock_session.media_changed_handlers), 1)
                self.assertEqual(len(mock_session.playback_changed_handlers), 1)

                # Fire media changed event and verify provider callback
                mock_session.media_changed_handlers[0]()
                self.assertEqual(len(changed_events), 1)

                # Switch session and verify subscription transfer
                mock_session2 = MockMediaSession(title="Song 2", artist="Artist 2")
                mock_mgr._current_session = mock_session2
                mock_mgr.session_changed_handlers[0]()

                self.assertEqual(len(mock_session.media_changed_handlers), 0)
                self.assertEqual(len(mock_session2.media_changed_handlers), 1)

                # Stop provider and verify cleanup
                provider.stop()
                self.assertEqual(len(mock_mgr.session_changed_handlers), 0)
                self.assertEqual(len(mock_session2.media_changed_handlers), 0)

    def test_disabled_coordinator_suppresses_provider_calls(self):
        """Issue 114 #2: When disabled, coordinator must never invoke providers."""
        call_count = [0]

        class CountingProvider(DummyProvider):
            def get_now_playing(self):
                call_count[0] += 1
                return NowPlaying(title="Song", artist="Artist")

        provider = CountingProvider(name="counter")
        coordinator = NowPlayingCoordinator([provider], poll_interval=0.05, enabled=False)

        # Direct update() call must return None and not query provider
        self.assertIsNone(coordinator.update())
        self.assertEqual(call_count[0], 0)

        # Running worker loop must remain idle and not query provider
        with coordinator:
            time.sleep(0.15)
            self.assertEqual(call_count[0], 0)
            self.assertIsNone(coordinator.current)

            # Enabling coordinator wakes worker and resumes querying
            coordinator.enabled = True
            deadline = time.time() + 1.0
            while call_count[0] == 0 and time.time() < deadline:
                time.sleep(0.02)
            self.assertGreater(call_count[0], 0)
            self.assertIsNotNone(coordinator.current)

    def test_stop_discards_delayed_results_and_prevents_race(self):
        """Issue 114 #3: Slow provider resolving after stop() must be discarded."""
        import threading

        in_get_event = threading.Event()
        release_event = threading.Event()
        notifications = []

        class SlowProvider(DummyProvider):
            def get_now_playing(self):
                in_get_event.set()
                release_event.wait(timeout=2.0)
                return NowPlaying(title="Late Song", artist="Late Artist")

        provider = SlowProvider(name="slow")
        coordinator = NowPlayingCoordinator([provider], poll_interval=0.05)
        coordinator.add_listener(notifications.append)

        coordinator.start()
        worker_thread = coordinator._thread
        # Wait until worker enters get_now_playing()
        self.assertTrue(in_get_event.wait(timeout=2.0))

        # Stop concurrently while get_now_playing is pending
        stop_thread = threading.Thread(target=coordinator.stop)
        stop_thread.start()

        # Brief pause so stop() acquires lock, increments gen, and resets current
        time.sleep(0.05)
        self.assertIsNone(coordinator.current)

        # Allow delayed provider to finish so worker exits cleanly
        release_event.set()

        stop_thread.join(timeout=2.0)
        if worker_thread:
            worker_thread.join(timeout=2.0)

        # Current must remain None and late result must not be published
        self.assertIsNone(coordinator.current)
        self.assertEqual(notifications, [])

    def test_windows_playback_status_failure_falls_back_to_not_playing(self):
        """Issue 114 #6: If get_playback_info fails, is_playing must default to False."""
        from wune.now_playing_windows import WindowsGsmtcProvider

        mock_session = MockMediaSession(title="Valid Song", artist="Valid Artist", fail_playback=True)
        mock_mgr = MockSessionManager(current_session=mock_session)
        provider = WindowsGsmtcProvider(manager=mock_mgr)

        np = provider.get_now_playing()
        self.assertIsNotNone(np)
        self.assertEqual(np.title, "Valid Song")
        self.assertFalse(np.is_playing)

        # With only_playing=True coordinator, this track must be filtered out
        coordinator = NowPlayingCoordinator([provider], only_playing=True)
        self.assertIsNone(coordinator.update())


if __name__ == "__main__":
    unittest.main()

