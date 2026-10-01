"""macOS audio capture backend supporting ScreenCaptureKit and CoreAudio devices."""
from __future__ import annotations

from contextlib import ExitStack
import sys
from typing import TYPE_CHECKING, Tuple, Any
import numpy as np

if TYPE_CHECKING:
    from .config import Config


def select_macos_output(cfg: Config) -> Any:
    """Select macOS audio endpoint, respecting user configuration."""
    import soundcard as sc

    if cfg.output_device is not None:
        try:
            return sc.get_speaker(cfg.output_device)
        except Exception:
            try:
                return sc.get_microphone(cfg.output_device)
            except Exception:
                pass

    try:
        speaker = sc.default_speaker()
        if speaker is not None:
            return speaker
    except Exception:
        pass

    try:
        mic = sc.default_microphone()
        if mic is not None:
            return mic
    except Exception:
        pass

    raise RuntimeError("No macOS audio device is available.")


class SoundCardCaptureBackend:
    """Capture backend wrapping a soundcard recorder stream."""

    def __init__(
        self,
        stream: Any,
        device_name: str,
        channels: int,
        sample_rate: int,
        exit_stack: ExitStack,
    ):
        self._stream = stream
        self._device_name = str(device_name)
        self._channels = int(channels)
        self._sample_rate = int(sample_rate)
        self._exit_stack = exit_stack

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
        self._exit_stack.close()


def is_macos_tap_supported() -> bool:
    """Return True if running macOS 14.2+ where Core Audio Process Tap is supported."""
    if sys.platform != "darwin":
        return False
    import platform

    ver_str = platform.mac_ver()[0]
    if not ver_str:
        return False
    parts = []
    for x in ver_str.split("."):
        try:
            parts.append(int(x))
        except ValueError:
            break
    return tuple(parts[:2]) >= (14, 2)


def _find_virtual_loopback_microphone() -> Any | None:
    """Search for virtual loopback audio devices (e.g. BlackHole, Soundflower)."""
    import soundcard as sc

    try:
        mics = sc.all_microphones()
    except Exception:
        mics = []
    for m in mics:
        name_lower = getattr(m, "name", "").lower()
        if any(k in name_lower for k in ("blackhole", "loopback", "soundflower")):
            return m
    return None


def _open_macos_virtual_loopback(
    cfg: Config,
    loopback_mic: Any,
    blocksize: int | None,
) -> Any:
    """Open SoundCard capture backend targeting an identified virtual loopback device."""
    requested_rate = cfg.sample_rate or getattr(loopback_mic, "samplerate", 48000)
    samplerate = int(requested_rate)
    stack = ExitStack()
    stream, device_name, channels_eff = open_macos_capture(
        endpoint=loopback_mic,
        samplerate=samplerate,
        blocksize=blocksize,
        exit_stack=stack,
    )
    return SoundCardCaptureBackend(
        stream=stream,
        device_name=device_name,
        channels=channels_eff,
        sample_rate=samplerate,
        exit_stack=stack,
    )


def open_macos_capture_backend(
    cfg: Config, blocksize: int | None = None
) -> Any:
    """Open macOS capture backend using Core Audio Process Tap (preferred) or virtual loopback fallback.

    If system playback capture is requested (output_device is None), does NOT silently fall back
    to physical microphone capture when tap fails or is unsupported.
    """
    import warnings

    # If user did not request an explicit device, prefer driverless native system-audio tap
    if cfg.output_device is None:
        if is_macos_tap_supported():
            try:
                from .tap_macos import CoreAudioTapBackend

                return CoreAudioTapBackend(cfg, blocksize=blocksize)
            except Exception as error:
                warnings.warn(
                    f"Core Audio system audio tap failed: {error}. Checking for virtual loopback fallback...",
                    RuntimeWarning,
                )
                loopback = _find_virtual_loopback_microphone()
                if loopback is not None:
                    return _open_macos_virtual_loopback(cfg, loopback, blocksize)
                raise RuntimeError(
                    f"System audio capture failed: {error}. "
                    "Install BlackHole for virtual loopback, or specify --device to capture from a microphone."
                ) from error
        else:
            # macOS < 14.2: tap is unsupported by the OS
            loopback = _find_virtual_loopback_microphone()
            if loopback is not None:
                return _open_macos_virtual_loopback(cfg, loopback, blocksize)
            raise RuntimeError(
                "Driverless system audio capture requires macOS 14.2+. "
                "Install BlackHole for virtual loopback, or specify --device to capture from a microphone."
            )

    endpoint = select_macos_output(cfg)
    requested_rate = cfg.sample_rate
    if requested_rate is None:
        if hasattr(endpoint, "samplerate"):
            requested_rate = endpoint.samplerate
        else:
            requested_rate = 48000
    samplerate = int(requested_rate)

    stack = ExitStack()
    stream, device_name, channels_eff = open_macos_capture(
        endpoint=endpoint,
        samplerate=samplerate,
        blocksize=blocksize,
        exit_stack=stack,
    )
    return SoundCardCaptureBackend(
        stream=stream,
        device_name=device_name,
        channels=channels_eff,
        sample_rate=samplerate,
        exit_stack=stack,
    )


def open_macos_capture(
    endpoint: Any,
    samplerate: int,
    blocksize: int | None,
    exit_stack: ExitStack,
) -> Tuple[Any, str, int]:
    """Open a capture stream on macOS.

    Attempts ScreenCaptureKit system-audio capture if available,
    falling back to CoreAudio virtual loopback devices (e.g. BlackHole)
    or input endpoints.
    """
    import soundcard as sc

    # 1. Attempt ScreenCaptureKit capture if helper / binding is available
    try:
        from .sck_capture import ScreenCaptureKitBackend  # type: ignore

        backend = ScreenCaptureKitBackend(samplerate=samplerate, blocksize=blocksize)
        exit_stack.callback(backend.close)
        backend.start()
        device_name = "System Audio (ScreenCaptureKit)"
        return backend, device_name, 2
    except (ImportError, RuntimeError):
        pass

    # 2. Check for explicit microphone or virtual loopback devices via CoreAudio
    target_mic = None

    # Review 1B: If endpoint is explicitly a microphone / recording-capable device, use it directly!
    # Even if isloopback is False, a microphone has a recorder method.
    if endpoint is not None and (
        hasattr(endpoint, "recorder") or getattr(endpoint, "isloopback", False)
    ):
        target_mic = endpoint
    else:
        # Look for virtual loopback devices (e.g. BlackHole)
        try:
            mics = sc.all_microphones()
        except Exception:
            mics = []
        for m in mics:
            name_lower = m.name.lower()
            if any(k in name_lower for k in ("blackhole", "loopback", "soundflower")):
                target_mic = m
                break

    if target_mic is None:
        try:
            target_mic = sc.default_microphone()
        except Exception:
            pass

    if target_mic is None:
        raise RuntimeError(
            "No macOS capture device or loopback driver (ScreenCaptureKit / BlackHole) is available."
        )

    channels = [0, 1] if getattr(target_mic, "channels", 2) >= 2 else [0]

    # Review 1C / Error fix:
    # On macOS CoreAudio, the hardware I/O buffer frame size is typically clamped to max 512.
    # Passing the FFT size (e.g. 4096) raises "TypeError: blocksize must be between 15.0 and 512".
    # Passing blocksize=None allows soundcard to use the device's safe default I/O buffer size.
    # The stream.record(numframes) call will safely accumulate multiple I/O chunks.
    recorder_blocksize = None
    if blocksize is not None and blocksize <= 512:
        recorder_blocksize = blocksize

    recorder = target_mic.recorder(
        samplerate=samplerate,
        channels=channels,
        blocksize=recorder_blocksize,
    )
    stream = exit_stack.enter_context(recorder)

    # Review 1A: Report the actual capture source that supplies PCM to Wune
    device_name = getattr(target_mic, "name", "Microphone")
    channels_eff = 2 if getattr(target_mic, "channels", 2) >= 2 else 1
    return stream, device_name, channels_eff
