"""Unit tests for the Now Playing metadata model, providers, and coordinator."""
from dataclasses import FrozenInstanceError
import time
import unittest

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
        renderer.now_playing_text = "Nowplaying:  Sample Track - Sample Artist"
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
        self.assertEqual(app.renderer.now_playing_text, "Nowplaying:  Song - Artist")

        # Track without artist
        app.now_playing.current = NowPlaying(title="Song Alone")
        App.update_now_playing_text(app)
        self.assertEqual(app.renderer.now_playing_text, "Nowplaying:  Song Alone")

        # No track playing
        app.now_playing.current = None
        App.update_now_playing_text(app)
        self.assertEqual(app.renderer.now_playing_text, "Nowplaying:  -")

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

        renderer.now_playing_text = "Nowplaying:  " + "A" * 1000
        renderer.draw_panel()


if __name__ == "__main__":
    unittest.main()

