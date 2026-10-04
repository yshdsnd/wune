"""Tests for macOS Core Audio Process Tap backend and status text font rendering."""
from unittest.mock import MagicMock, patch
import sys
import unittest
import numpy as np

from wune.config import Config
from wune.capture_macos import open_macos_capture_backend


class MacOsTapTests(unittest.TestCase):
    def test_core_audio_tap_backend_live(self):
        """Verify CoreAudioTapBackend live creation, record and teardown on macOS."""
        if sys.platform != "darwin":
            self.skipTest("macOS specific test")

        from wune.tap_macos import CoreAudioTapBackend
        try:
            backend = CoreAudioTapBackend()
        except RuntimeError as error:
            self.skipTest(f"Core Audio tap unavailable on this system or headless CI runner: {error}")

        self.assertEqual(backend.device_name, "System Audio (Core Audio Tap)")
        self.assertGreater(backend.sample_rate, 0)
        self.assertEqual(backend.channels, 2)

        data = backend.record(4096)
        self.assertEqual(data.shape, (4096, 2))
        self.assertEqual(data.dtype, np.float32)

        backend.close()

    def test_core_audio_tap_backend_non_mac_raises(self):
        """CoreAudioTapBackend must fail on non-macOS platforms."""
        from wune.tap_macos import CoreAudioTapBackend
        with patch("sys.platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "only supported on macOS"):
                CoreAudioTapBackend()

    def test_open_macos_capture_backend_prefers_core_audio_tap(self):
        """When output_device is None, open_macos_capture_backend prefers CoreAudioTapBackend."""
        mock_backend = MagicMock()
        mock_backend.device_name = "System Audio (Core Audio Tap)"
        mock_backend.sample_rate = 48000
        mock_backend.channels = 2

        with patch("wune.capture_macos.is_macos_tap_supported", return_value=True), \
             patch("wune.tap_macos.CoreAudioTapBackend", return_value=mock_backend):
            cfg = Config(output_device=None)
            backend = open_macos_capture_backend(cfg, blocksize=4096)
            self.assertEqual(backend, mock_backend)
            self.assertEqual(backend.device_name, "System Audio (Core Audio Tap)")

    def test_open_macos_capture_backend_falls_back_when_tap_fails(self):
        """When CoreAudioTapBackend raises an error, falls back to virtual loopback if available."""
        blackhole_mic = MagicMock()
        blackhole_mic.name = "BlackHole 2ch"
        blackhole_mic.samplerate = 48000
        blackhole_mic.channels = 2

        with patch("wune.capture_macos.is_macos_tap_supported", return_value=True), \
             patch("wune.tap_macos.CoreAudioTapBackend", side_effect=RuntimeError("tap unavailable")), \
             patch("soundcard.all_microphones", return_value=[blackhole_mic]), \
             patch("wune.capture_macos.open_macos_capture") as mock_open:
            mock_stream = MagicMock()
            mock_open.return_value = (mock_stream, "BlackHole 2ch", 2)
            cfg = Config(output_device=None)
            backend = open_macos_capture_backend(cfg, blocksize=4096)
            self.assertEqual(backend.device_name, "BlackHole 2ch")
            self.assertEqual(backend.channels, 2)

    def test_open_macos_capture_backend_refuses_silent_microphone_fallback(self):
        """Review P1: Never silently fall back to physical microphone when system audio tap fails."""
        internal_mic = MagicMock()
        internal_mic.name = "MacBook Pro Microphone"
        internal_mic.channels = 1

        with patch("wune.capture_macos.is_macos_tap_supported", return_value=True), \
             patch("wune.tap_macos.CoreAudioTapBackend", side_effect=RuntimeError("permission denied")), \
             patch("soundcard.all_microphones", return_value=[internal_mic]):
            cfg = Config(output_device=None)
            with self.assertRaisesRegex(RuntimeError, "System audio capture failed: permission denied"):
                open_macos_capture_backend(cfg, blocksize=4096)

    def test_open_macos_capture_backend_unsupported_on_old_macos(self):
        """When macOS < 14.2, tap is unsupported; falls back to virtual loopback or raises clear error."""
        internal_mic = MagicMock()
        internal_mic.name = "MacBook Pro Microphone"

        with patch("wune.capture_macos.is_macos_tap_supported", return_value=False), \
             patch("soundcard.all_microphones", return_value=[internal_mic]):
            cfg = Config(output_device=None)
            with self.assertRaisesRegex(RuntimeError, "Driverless system audio capture requires macOS 14.2+"):
                open_macos_capture_backend(cfg, blocksize=4096)

    def test_core_audio_tap_backend_accumulates_until_complete_block(self):
        """Review P1: Backend accumulates chunks until requested numframes is reached."""
        from wune.tap_macos import CoreAudioTapBackend
        backend = CoreAudioTapBackend.__new__(CoreAudioTapBackend)
        backend._closed = False
        backend._channels = 2
        backend._handle = 12345
        backend._lib = MagicMock()

        # Simulate wune_tap_read returning 2400 frames first, then remaining 1696 frames
        backend._lib.wune_tap_read.side_effect = [2400, 1696]

        data = backend.record(4096)
        self.assertEqual(data.shape, (4096, 2))
        self.assertEqual(backend._lib.wune_tap_read.call_count, 2)

    def test_core_audio_tap_backend_silence_timeout_zero_fills(self):
        """Review P1: Zero-fill when wune_tap_read times out with 0 frames."""
        from wune.tap_macos import CoreAudioTapBackend
        backend = CoreAudioTapBackend.__new__(CoreAudioTapBackend)
        backend._closed = False
        backend._channels = 2
        backend._handle = 12345
        backend._lib = MagicMock()

        # Simulate timeout returning 0 frames
        backend._lib.wune_tap_read.return_value = 0

        data = backend.record(4096)
        self.assertEqual(data.shape, (4096, 2))
        self.assertTrue((data == 0).all())

    def test_renderer_status_text_renders_japanese_device_names(self):
        """Verify Japanese text in status area renders successfully without glyph failure."""
        import pygame as pg
        pg.init()
        from wune.renderer import LedBarRenderer

        surf = pg.Surface((800, 600))
        cfg = Config()
        renderer = LedBarRenderer(surf, cfg)

        japanese_strings = [
            "出力: MacBook Airのマイク (48.0 kHz, 1 ch)",
            "出力: System Audio (Core Audio Tap) (48.0 kHz, 2 ch)",
            "出力: 外付けDAC (96.0 kHz, 2 ch)",
        ]
        levels = np.zeros((2, 64), dtype=np.float32)

        for s in japanese_strings:
            renderer.info_text = s
            # Must render without error
            renderer.draw(levels, 0.0)
            rendered_surf = renderer.font_small.render(s, True, (255, 255, 255))
            self.assertGreater(rendered_surf.get_width(), 100)
            self.assertGreater(rendered_surf.get_height(), 10)
