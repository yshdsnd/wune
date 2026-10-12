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


class PulseAudioLoopbackBackend:
    """Linux and Unix PulseAudio / PipeWire loopback capture backend."""

    def __init__(self, cfg: Config, blocksize: int | None = None):
        from .soundcard_compat import get_soundcard
        sc = get_soundcard()

        speaker = (
            sc.default_speaker()
            if cfg.output_device is None
            else sc.get_speaker(cfg.output_device)
        )
        if speaker is None:
            raise RuntimeError("No playback device is available.")

        loopback, exact_match = self._find_monitor_source(
            sc, speaker, allow_fallback=(cfg.output_device is None)
        )
        if loopback is None:
            raise RuntimeError(
                f"No PulseAudio/PipeWire loopback monitor source found for speaker: {speaker.name}"
            )

        requested_rate = cfg.sample_rate
        if requested_rate is None:
            requested_rate = 48000
        self._sample_rate = int(requested_rate)
        if self._sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        if exact_match:
            self._device_name = str(speaker.name)
        else:
            self._device_name = str(getattr(loopback, "name", speaker.name))
        self._channels = 2
        self._stack = ExitStack()

        nfft = int(cfg.block_size if blocksize is None else blocksize)

        mic_channels = getattr(loopback, "channels", 2)
        if isinstance(mic_channels, int) and mic_channels >= 2:
            stream_channels = [0, 1]
            self._duplicate_mono = False
        else:
            stream_channels = [0]
            self._duplicate_mono = True

        recorder = loopback.recorder(
            samplerate=self._sample_rate,
            channels=stream_channels,
            blocksize=nfft,
        )
        self._stream = self._stack.enter_context(recorder)

    @staticmethod
    def _find_monitor_source(sc, speaker, allow_fallback: bool = True):
        """Find the loopback monitor source corresponding to the given speaker.

        Returns (loopback_mic, is_exact_match).
        """
        speaker_id = getattr(speaker, "id", None)
        speaker_name = getattr(speaker, "name", "")

        # 1. Direct lookup by monitor ID '<speaker.id>.monitor' or '<speaker.id>'
        if speaker_id:
            for candidate_id in (f"{speaker_id}.monitor", str(speaker_id)):
                try:
                    mic = sc.get_microphone(id=candidate_id, include_loopback=True)
                    if mic is not None and getattr(mic, "isloopback", True):
                        return mic, True
                except Exception:
                    pass

        # 2. Iterate all_microphones(include_loopback=True)
        try:
            mics = sc.all_microphones(include_loopback=True)
        except Exception:
            mics = []

        # Exact ID match with monitor suffix or speaker ID
        if speaker_id:
            for m in mics:
                if getattr(m, "isloopback", False):
                    if m.id in (f"{speaker_id}.monitor", str(speaker_id)):
                        return m, True

        # Match by speaker ID prefix / substring or name
        for m in mics:
            if getattr(m, "isloopback", False):
                if (speaker_id and str(speaker_id) in str(m.id)) or m.name == speaker_name or m.name == f"Monitor of {speaker_name}":
                    return m, True

        # Fallback to any loopback device only if allowed (e.g. default speaker)
        if allow_fallback:
            for m in mics:
                if getattr(m, "isloopback", False):
                    return m, False

        return None, False

    def record(self, numframes: int) -> np.ndarray:
        data = self._stream.record(numframes=numframes)
        if self._duplicate_mono or data.ndim == 1 or data.shape[1] == 1:
            if data.ndim == 1:
                data = np.column_stack([data, data])
            else:
                data = np.column_stack([data[:, 0], data[:, 0]])
        return np.ascontiguousarray(data, dtype=np.float32)

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
    """macOS capture backend wrapping Core Audio Process Tap or selected inputs."""

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
    if sys.platform == "linux" or sys.platform.startswith("freebsd"):
        return PulseAudioLoopbackBackend(cfg, blocksize=blocksize)
    return WasapiLoopbackBackend(cfg, blocksize=blocksize)
