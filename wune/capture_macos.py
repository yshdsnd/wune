"""macOS audio capture backend supporting ScreenCaptureKit and CoreAudio devices."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Tuple, Any

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


def open_macos_capture(
    endpoint: Any,
    samplerate: int,
    blocksize: int,
    exit_stack: Any,
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

    # 2. Check for virtual loopback devices or microphones via CoreAudio
    target_mic = None
    if endpoint is not None and getattr(endpoint, "isloopback", False):
        target_mic = endpoint
    else:
        mics = sc.all_microphones()
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
    recorder = target_mic.recorder(
        samplerate=samplerate,
        channels=channels,
        blocksize=blocksize,
    )
    stream = exit_stack.enter_context(recorder)
    device_name = getattr(endpoint, "name", target_mic.name)
    channels_eff = 2 if getattr(target_mic, "channels", 2) >= 2 else 1
    return stream, device_name, channels_eff
