"""Tests for capture backend abstractions and AudioSpectrum integration."""
from unittest.mock import MagicMock, patch
import unittest
import numpy as np

from wune.config import Config
from wune.capture import (
    CaptureBackend,
    WasapiLoopbackBackend,
    MacOsCaptureBackend,
    PulseAudioLoopbackBackend,
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

    def test_create_capture_backend_selects_pulseaudio_on_linux(self):
        with patch("sys.platform", "linux"), \
             patch("wune.capture.PulseAudioLoopbackBackend") as mock_pulse:
            cfg = Config()
            backend = create_capture_backend(cfg, blocksize=4096)
            mock_pulse.assert_called_once_with(cfg, blocksize=4096)
            self.assertEqual(backend, mock_pulse.return_value)

    def test_create_capture_backend_selects_pulseaudio_on_freebsd(self):
        with patch("sys.platform", "freebsd14"), \
             patch("wune.capture.PulseAudioLoopbackBackend") as mock_pulse:
            cfg = Config()
            backend = create_capture_backend(cfg, blocksize=4096)
            mock_pulse.assert_called_once_with(cfg, blocksize=4096)
            self.assertEqual(backend, mock_pulse.return_value)

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

    def test_audio_spectrum_step_handles_record_exception_with_zero_levels(self):
        dummy = DummyCaptureBackend(sample_rate=44100, channels=2)
        dummy.record = MagicMock(side_effect=RuntimeError("Device disconnected"))
        cfg = Config(sample_rate=44100)
        spectrum = AudioSpectrum(cfg, bars=32, channels=2, capture_backend=dummy)

        # Step should not raise; instead it should return zero levels
        levels = spectrum.step(1 / 60)
        self.assertEqual(levels.shape, (2, 32))
        self.assertTrue(np.all(levels == 0.0))

    def test_audio_spectrum_close_handles_backend_exception_gracefully(self):
        dummy = DummyCaptureBackend(sample_rate=44100, channels=2)
        dummy.close = MagicMock(side_effect=RuntimeError("Shutdown failed"))
        cfg = Config(sample_rate=44100)
        spectrum = AudioSpectrum(cfg, bars=32, channels=2, capture_backend=dummy)

        # Close should catch exception and not raise
        spectrum.close()
        self.assertTrue(spectrum._closed)

    def test_wasapi_backend_close_safely_handles_stack_exception(self):
        backend = object.__new__(WasapiLoopbackBackend)
        mock_stack = MagicMock()
        mock_stack.close.side_effect = RuntimeError("WASAPI COM Error 0x100000001")
        backend._stack = mock_stack

        # close() should safely catch and warn without raising
        backend.close()
        mock_stack.close.assert_called_once()

