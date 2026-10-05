"""Tests for capture backend abstractions and AudioSpectrum integration."""
from unittest.mock import MagicMock, patch
import unittest
import numpy as np

from wune.config import Config
from wune.capture import (
    CaptureBackend,
    WasapiLoopbackBackend,
    MacOsCaptureBackend,
    create_capture_backend,
)
from wune.spectrum_audio import AudioSpectrum


class DummyCaptureBackend:
    def __init__(self, sample_rate: int = 48000, channels: int = 2, device_name: str = "Dummy Device"):
        self._sample_rate = sample_rate
        self._channels = channels
        self._device_name = device_name
        self.closed = False

    def record(self, numframes: int) -> np.ndarray:
        return np.zeros((numframes, self._channels), dtype=np.float32)

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
        self.closed = True


class CaptureBackendTests(unittest.TestCase):
    def test_dummy_backend_satisfies_protocol(self):
        backend = DummyCaptureBackend()
        self.assertIsInstance(backend, CaptureBackend)

    def test_create_capture_backend_selects_macos_on_darwin(self):
        with patch("sys.platform", "darwin"), \
             patch("wune.capture.MacOsCaptureBackend") as mock_mac:
            cfg = Config()
            backend = create_capture_backend(cfg, blocksize=4096)
            mock_mac.assert_called_once_with(cfg, blocksize=4096)
            self.assertEqual(backend, mock_mac.return_value)

    def test_create_capture_backend_selects_wasapi_on_windows(self):
        with patch("sys.platform", "win32"), \
             patch("wune.capture.WasapiLoopbackBackend") as mock_win:
            cfg = Config()
            backend = create_capture_backend(cfg, blocksize=4096)
            mock_win.assert_called_once_with(cfg, blocksize=4096)
            self.assertEqual(backend, mock_win.return_value)

    def test_audio_spectrum_uses_injected_capture_backend(self):
        dummy = DummyCaptureBackend(sample_rate=44100, channels=2, device_name="Test Input")
        cfg = Config(sample_rate=44100)
        spectrum = AudioSpectrum(cfg, bars=32, channels=2, capture_backend=dummy)
        self.assertEqual(spectrum.sr, 44100)
        self.assertEqual(spectrum.channels_eff, 2)
        self.assertEqual(spectrum.device, "Test Input")

        # Step invokes backend.record
        levels = spectrum.step(1 / 60)
        self.assertEqual(levels.shape, (2, 32))

        # Close releases backend
        spectrum.close()
        self.assertTrue(dummy.closed)
