"""Audio capture backend abstractions for platform-specific capture."""
from __future__ import annotations

from contextlib import ExitStack
import sys
from typing import Protocol, runtime_checkable
import numpy as np

from .config import Config


@runtime_checkable
class CaptureBackend(Protocol):
    """Platform-specific audio capture backend interface."""

    def record(self, numframes: int) -> np.ndarray:
        """Record numframes of audio data. Returns float32 array shaped (numframes, channels)."""
        ...

    @property
    def sample_rate(self) -> int:
        """Effective audio sample rate in Hz."""
        ...

    @property
    def channels(self) -> int:
        """Effective number of captured channels."""
        ...

    @property
    def device_name(self) -> str:
        """Display name of the actual audio capture source."""
        ...

    def close(self) -> None:
        """Release audio capture resources."""
        ...


class WasapiLoopbackBackend:
    """Windows WASAPI loopback capture backend."""

    def __init__(self, cfg: Config, blocksize: int | None = None):
        import soundcard as sc
        from .soundcard_compat import prepare_soundcard, output_sample_rate

        prepare_soundcard()
        speaker = (
            sc.default_speaker()
            if cfg.output_device is None
            else sc.get_speaker(cfg.output_device)
        )
        if speaker is None:
            raise RuntimeError("No Windows playback device is available.")
        if speaker.channels < 2:
            raise RuntimeError("Select a stereo Windows playback device for loopback.")

        requested_rate = cfg.sample_rate
        if requested_rate is None:
            requested_rate = output_sample_rate(speaker)
        self._sample_rate = int(requested_rate)
        if self._sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        loopback = sc.get_microphone(id=speaker.id, include_loopback=True)
        if not loopback.isloopback:
            raise RuntimeError("The selected playback endpoint has no loopback capture.")

        self._device_name = str(speaker.name)
        self._channels = 2
        self._stack = ExitStack()

        nfft = int(cfg.block_size if blocksize is None else blocksize)
        recorder = loopback.recorder(
            samplerate=self._sample_rate,
            channels=[0, 1],
            blocksize=nfft,
            exclusive_mode=False,
        )
        self._stream = self._stack.enter_context(recorder)

    def record(self, numframes: int) -> np.ndarray:
        return self._stream.record(numframes=numframes)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def device_name(self) -> str:
        return self._device_name

    def close(self) -> None:
        self._stack.close()


class MacOsCaptureBackend:
    """macOS capture backend wrapping ScreenCaptureKit or CoreAudio devices."""

    def __init__(self, cfg: Config, blocksize: int | None = None):
        from .capture_macos import open_macos_capture_backend

        self._backend = open_macos_capture_backend(cfg, blocksize=blocksize)

    def record(self, numframes: int) -> np.ndarray:
        return self._backend.record(numframes=numframes)

    @property
    def sample_rate(self) -> int:
        return self._backend.sample_rate

    @property
    def channels(self) -> int:
        return self._backend.channels

    @property
    def device_name(self) -> str:
        return self._backend.device_name

    def close(self) -> None:
        self._backend.close()


def create_capture_backend(cfg: Config, blocksize: int | None = None) -> CaptureBackend:
    """Create the appropriate capture backend for the current operating system."""
    if sys.platform == "darwin":
        return MacOsCaptureBackend(cfg, blocksize=blocksize)
    return WasapiLoopbackBackend(cfg, blocksize=blocksize)
