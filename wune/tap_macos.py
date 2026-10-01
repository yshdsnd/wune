"""Core Audio Process Tap backend for driverless macOS system audio capture."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .config import Config


def _ensure_dylib() -> str | None:
    """Locate or compile libwune_tap.dylib."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dylib_path = os.path.join(dir_path, "libwune_tap.dylib")
    if os.path.exists(dylib_path):
        return dylib_path

    # Attempt to build from tap_backend.m if clang is available
    src_path = os.path.join(dir_path, "tap_backend.m")
    if os.path.exists(src_path):
        try:
            cmd = [
                "/usr/bin/clang",
                "-dynamiclib",
                "-O3",
                "-fobjc-arc",
                "-arch", "arm64",
                "-arch", "x86_64",
                "-framework", "Foundation",
                "-framework", "CoreAudio",
                src_path,
                "-o", dylib_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=15)
            if os.path.exists(dylib_path):
                return dylib_path
        except Exception:
            pass
    return None


class CoreAudioTapBackend:
    """Driverless system audio playback capture using Apple's Core Audio Process Tap."""

    def __init__(self, cfg: Config | None = None, blocksize: int | None = None):
        if sys.platform != "darwin":
            raise RuntimeError("CoreAudioTapBackend is only supported on macOS.")

        dylib_path = _ensure_dylib()
        if not dylib_path:
            raise RuntimeError("libwune_tap.dylib could not be found or compiled.")

        try:
            self._lib = ctypes.cdll.LoadLibrary(dylib_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load libwune_tap.dylib: {e}") from e

        self._lib.wune_tap_create.restype = ctypes.c_void_p
        self._lib.wune_tap_create.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]

        self._lib.wune_tap_read.restype = ctypes.c_uint32
        self._lib.wune_tap_read.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]

        self._lib.wune_tap_destroy.restype = None
        self._lib.wune_tap_destroy.argtypes = [ctypes.c_void_p]

        sr = ctypes.c_uint32(0)
        ch = ctypes.c_uint32(0)
        self._handle = self._lib.wune_tap_create(ctypes.byref(sr), ctypes.byref(ch))
        if not self._handle:
            raise RuntimeError("Failed to create Core Audio Process Tap aggregate device.")

        self._sample_rate = int(sr.value if sr.value > 0 else 48000)
        self._channels = int(ch.value if ch.value > 0 else 2)
        self._device_name = "System Audio (Core Audio Tap)"
        self._closed = False

    def record(self, numframes: int) -> np.ndarray:
        if self._closed or not self._handle:
            return np.zeros((numframes, self._channels), dtype=np.float32)

        buffer = np.zeros((numframes, self._channels), dtype=np.float32)
        total_read = 0
        while total_read < numframes:
            read = self._lib.wune_tap_read(
                self._handle,
                buffer[total_read:].ctypes.data,
                ctypes.c_uint32(numframes - total_read),
                ctypes.c_uint32(0),
            )
            if read == 0:
                # Silence or timeout: zero-fill remainder
                buffer[total_read:] = 0.0
                break
            total_read += read
        return buffer

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
        if not self._closed:
            self._closed = True
            if self._handle:
                handle = self._handle
                self._handle = None
                self._lib.wune_tap_destroy(handle)
