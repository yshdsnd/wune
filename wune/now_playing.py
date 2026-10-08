"""Now-playing metadata model, provider abstraction, and coordinator."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys
import threading
from typing import Callable, Sequence


@dataclass(frozen=True)
class NowPlaying:
    """Immutable snapshot of track playback metadata."""

    title: str = ""
    artist: str = ""
    album: str = ""
    source: str = ""
    is_playing: bool = True
    raw_text: str = ""

    @property
    def has_metadata(self) -> bool:
        """True if either title or artist contains non-whitespace text."""
        return bool(self.title.strip() or self.artist.strip())

    def display_text(self, separator: str = " - ") -> str:
        """Formatted representation suitable for unobtrusive UI display.

        Returns:
            'Title - Artist' if both exist,
            'Title' if only title exists,
            'Artist' if only artist exists,
            or '' if neither exists.
        """
        title = self.title.strip()
        artist = self.artist.strip()
        if title and artist:
            return f"{title}{separator}{artist}"
        if title:
            return title
        if artist:
            return artist
        return ""

    def __bool__(self) -> bool:
        return self.has_metadata


class MetadataProvider(ABC):
    """Abstract base provider for obtaining now-playing metadata."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider identifier."""
        ...

    @property
    @abstractmethod
    def priority(self) -> int:
        """Provider priority rank (lower numbers evaluate first)."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider can operate on the current platform/environment."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Initialize provider resources or event listeners."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Release provider resources or event listeners."""
        ...

    @abstractmethod
    def get_now_playing(self) -> NowPlaying | None:
        """Fetch or return the current track metadata, or None if inactive."""
        ...


class NowPlayingCoordinator:
    """Manages metadata providers and publishes atomic snapshots to the UI loop."""

    def __init__(
        self,
        providers: Sequence[MetadataProvider] | None = None,
        poll_interval: float = 1.0,
        only_playing: bool = True,
        enabled: bool = True,
    ):
        self.poll_interval = max(0.1, float(poll_interval))
        self.only_playing = bool(only_playing)
        self._enabled = bool(enabled)
        self._providers: list[MetadataProvider] = []
        self._enabled_names: set[str] = set()
        self._lock = threading.Lock()
        self._current: NowPlaying | None = None
        self._listeners: list[Callable[[NowPlaying | None], None]] = []

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._update_event = threading.Event()
        self._started = False
        self._generation = 0

        if providers:
            for p in providers:
                self.register_provider(p)

    @property
    def enabled(self) -> bool:
        """Return True if now playing retrieval is currently active."""
        with self._lock:
            return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable metadata retrieval without tearing down threads."""
        should_wake = False
        with self._lock:
            was_enabled = self._enabled
            self._enabled = bool(value)
            if not self._enabled:
                self._generation += 1
                self._current = None
            elif not was_enabled:
                should_wake = True
        if should_wake:
            self.request_update()

    def request_update(self) -> None:
        """Request an immediate metadata update (e.g. from provider change event)."""
        self._update_event.set()

    def register_provider(self, provider: MetadataProvider, enabled: bool = True) -> None:
        """Register a provider and insert it in priority order."""
        with self._lock:
            self._providers = [p for p in self._providers if p.name != provider.name]
            self._providers.append(provider)
            self._providers.sort(key=lambda p: p.priority)
            if enabled:
                self._enabled_names.add(provider.name)
            else:
                self._enabled_names.discard(provider.name)

        if hasattr(provider, "set_on_change"):
            try:
                provider.set_on_change(self.request_update)
            except Exception:
                pass

        if self._started and enabled:
            try:
                if provider.is_available():
                    provider.start()
            except Exception:
                pass

    def unregister_provider(self, name: str) -> None:
        """Unregister a provider by name."""
        removed = None
        with self._lock:
            remaining = []
            for p in self._providers:
                if p.name == name:
                    removed = p
                else:
                    remaining.append(p)
            self._providers = remaining
            self._enabled_names.discard(name)

        if removed and self._started:
            try:
                removed.stop()
            except Exception:
                pass

    def enable_provider(self, name: str) -> None:
        """Enable a registered provider."""
        target = None
        with self._lock:
            self._enabled_names.add(name)
            target = next((p for p in self._providers if p.name == name), None)

        if target and self._started:
            try:
                if target.is_available():
                    target.start()
            except Exception:
                pass

    def disable_provider(self, name: str) -> None:
        """Disable a registered provider."""
        target = None
        with self._lock:
            self._enabled_names.discard(name)
            target = next((p for p in self._providers if p.name == name), None)

        if target and self._started:
            try:
                target.stop()
            except Exception:
                pass

    def is_provider_enabled(self, name: str) -> bool:
        """Return True if the provider is currently enabled."""
        with self._lock:
            return name in self._enabled_names

    @property
    def current(self) -> NowPlaying | None:
        """Thread-safe, non-blocking snapshot of current metadata."""
        with self._lock:
            return self._current

    def add_listener(self, listener: Callable[[NowPlaying | None], None]) -> None:
        """Subscribe to metadata change notifications."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[NowPlaying | None], None]) -> None:
        """Unsubscribe from metadata change notifications."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def update(self) -> NowPlaying | None:
        """Synchronously query providers in priority order and update current state."""
        with self._lock:
            if not self._enabled or self._stop_event.is_set():
                return None
            gen = self._generation
            active_providers = [
                p for p in self._providers if p.name in self._enabled_names
            ]

        resolved: NowPlaying | None = None
        for provider in active_providers:
            with self._lock:
                if (
                    not self._enabled
                    or self._generation != gen
                    or self._stop_event.is_set()
                ):
                    return None
            try:
                if not provider.is_available():
                    continue
                candidate = provider.get_now_playing()
                if candidate is None:
                    continue
                if self.only_playing and not candidate.is_playing:
                    continue
                if candidate.has_metadata:
                    resolved = candidate
                    break
            except Exception:
                continue

        listeners_to_notify: list[Callable[[NowPlaying | None], None]] = []
        with self._lock:
            if (
                not self._enabled
                or self._generation != gen
                or self._stop_event.is_set()
            ):
                return None
            changed = self._current != resolved
            self._current = resolved
            if changed:
                listeners_to_notify = list(self._listeners)

        for listener in listeners_to_notify:
            try:
                listener(resolved)
            except Exception:
                pass

        return resolved

    def start(self) -> None:
        """Start provider lifecycle and the background polling worker."""
        with self._lock:
            if self._started:
                return
            self._started = True
            self._generation += 1
            gen = self._generation
            self._stop_event.clear()
            self._update_event.clear()
            providers_to_start = [
                p for p in self._providers if p.name in self._enabled_names
            ]

        for p in providers_to_start:
            try:
                if p.is_available():
                    p.start()
            except Exception:
                pass

        with self._lock:
            if not self._started or self._generation != gen:
                return
            self._thread = threading.Thread(
                target=self._worker_loop,
                args=(gen,),
                name="NowPlayingCoordinator",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop background worker and shut down all providers."""
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._generation += 1
            self._stop_event.set()
            self._update_event.set()
            thread = self._thread
            self._thread = None
            providers_to_stop = list(self._providers)
            self._current = None

        if thread and thread.is_alive() and thread != threading.current_thread():
            thread.join(timeout=2.0)

        for p in providers_to_stop:
            try:
                p.stop()
            except Exception:
                pass

        with self._lock:
            self._current = None

    def _worker_loop(self, generation: int) -> None:
        """Worker loop executing low-frequency metadata updates."""
        while not self._stop_event.is_set():
            with self._lock:
                if not self._started or self._generation != generation:
                    break
                is_enabled = self._enabled

            if is_enabled:
                try:
                    self.update()
                except Exception:
                    pass

            self._update_event.wait(self.poll_interval)
            self._update_event.clear()

    def __enter__(self) -> NowPlayingCoordinator:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def create_default_coordinator() -> NowPlayingCoordinator:
    """Create a coordinator configured with platform-appropriate metadata providers."""
    providers: list[MetadataProvider] = []
    if sys.platform == "win32":
        try:
            from .now_playing_windows import WindowsGsmtcProvider

            provider = WindowsGsmtcProvider()
            if provider.is_available():
                providers.append(provider)
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            from .now_playing_macos import MacAppleMusicProvider

            provider = MacAppleMusicProvider()
            if provider.is_available():
                providers.append(provider)
        except Exception:
            pass
    return NowPlayingCoordinator(providers=providers)

