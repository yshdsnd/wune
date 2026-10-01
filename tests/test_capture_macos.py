"""Tests for macOS audio capture selection, actual source reporting, and explicit microphone routing."""
from contextlib import ExitStack
import types
import unittest
from unittest.mock import MagicMock, patch

from wune.config import Config
from wune.capture_macos import (
    select_macos_output,
    open_macos_capture,
    open_macos_capture_backend,
)


class MacOsCaptureTests(unittest.TestCase):
    def test_select_macos_output_finds_default_speaker(self):
        speaker = types.SimpleNamespace(id="mac-speaker", name="Built-in Output", channels=2)
        with patch("soundcard.default_speaker", return_value=speaker):
            endpoint = select_macos_output(Config())
            self.assertEqual(endpoint.id, "mac-speaker")

    def test_select_macos_output_respects_explicit_config(self):
        speaker = types.SimpleNamespace(id="external-dac", name="External DAC", channels=2)
        with patch("soundcard.get_speaker", return_value=speaker) as mock_get:
            endpoint = select_macos_output(Config(output_device="external-dac"))
            self.assertEqual(endpoint.id, "external-dac")
            mock_get.assert_called_once_with("external-dac")

    def test_select_macos_output_respects_explicit_microphone(self):
        mic = types.SimpleNamespace(id="usb-mic", name="USB Microphone", channels=1)
        with patch("soundcard.get_speaker", side_effect=Exception("not a speaker")), \
             patch("soundcard.get_microphone", return_value=mic) as mock_get_mic:
            endpoint = select_macos_output(Config(output_device="usb-mic"))
            self.assertEqual(endpoint.id, "usb-mic")
            mock_get_mic.assert_called_once_with("usb-mic")

    def test_select_macos_output_raises_when_no_device_found(self):
        with patch("soundcard.default_speaker", side_effect=Exception("none")), \
             patch("soundcard.default_microphone", side_effect=Exception("none")):
            with self.assertRaisesRegex(RuntimeError, "No macOS audio device"):
                select_macos_output(Config())

    def test_open_macos_capture_reports_actual_source_with_blackhole(self):
        """Review 1A: Report actual capture source (BlackHole) instead of speaker name."""
        blackhole_mic = MagicMock()
        blackhole_mic.name = "BlackHole 2ch"
        blackhole_mic.channels = 2
        blackhole_recorder = MagicMock()
        mock_stream = MagicMock()
        blackhole_recorder.__enter__.return_value = mock_stream
        blackhole_mic.recorder.return_value = blackhole_recorder

        other_mic = MagicMock()
        other_mic.name = "Internal Mic"
        other_mic.channels = 1

        speaker = types.SimpleNamespace(name="MacBook Speakers", isloopback=False)

        stack = ExitStack()
        with stack:
            with patch("soundcard.all_microphones", return_value=[other_mic, blackhole_mic]):
                stream, name, ch_eff = open_macos_capture(speaker, 48000, 4096, stack)
                self.assertEqual(stream, mock_stream)
                # Actual capture source must be reported, not the output endpoint
                self.assertEqual(name, "BlackHole 2ch")
                self.assertEqual(ch_eff, 2)
                # blocksize must not exceed CoreAudio limits (safe None used for > 512)
                blackhole_mic.recorder.assert_called_once_with(
                    samplerate=48000,
                    channels=[0, 1],
                    blocksize=None,
                )

    def test_open_macos_capture_reports_actual_source_with_default_microphone(self):
        """Review 1A: Report actual capture source when falling back to default mic."""
        default_mic = MagicMock()
        default_mic.name = "Built-in Microphone"
        default_mic.channels = 1
        recorder = MagicMock()
        mock_stream = MagicMock()
        recorder.__enter__.return_value = mock_stream
        default_mic.recorder.return_value = recorder

        speaker = types.SimpleNamespace(name="Headphones", isloopback=False)

        stack = ExitStack()
        with stack:
            with patch("soundcard.all_microphones", return_value=[]), \
                 patch("soundcard.default_microphone", return_value=default_mic):
                stream, name, ch_eff = open_macos_capture(speaker, 44100, 2048, stack)
                self.assertEqual(stream, mock_stream)
                # Actual capture source must be reported
                self.assertEqual(name, "Built-in Microphone")
                self.assertEqual(ch_eff, 1)
                default_mic.recorder.assert_called_once_with(
                    samplerate=44100,
                    channels=[0],
                    blocksize=None,
                )

    def test_open_macos_capture_uses_explicitly_selected_microphone(self):
        """Review 1B: Verify explicit microphone selection is not replaced by BlackHole/default mic."""
        explicit_mic = MagicMock()
        explicit_mic.name = "USB Podcast Mic"
        explicit_mic.channels = 1
        explicit_mic.isloopback = False
        recorder = MagicMock()
        mock_stream = MagicMock()
        recorder.__enter__.return_value = mock_stream
        explicit_mic.recorder.return_value = recorder

        blackhole_mic = MagicMock()
        blackhole_mic.name = "BlackHole 2ch"

        stack = ExitStack()
        with stack:
            with patch("soundcard.all_microphones", return_value=[blackhole_mic]):
                # Passing the explicitly selected microphone endpoint
                stream, name, ch_eff = open_macos_capture(explicit_mic, 48000, 4096, stack)
                self.assertEqual(stream, mock_stream)
                self.assertEqual(name, "USB Podcast Mic")
                self.assertEqual(ch_eff, 1)
                explicit_mic.recorder.assert_called_once_with(
                    samplerate=48000,
                    channels=[0],
                    blocksize=None,
                )
                # BlackHole should not have been used
                blackhole_mic.recorder.assert_not_called()

    def test_open_macos_capture_backend_end_to_end_explicit_mic(self):
        """Review 1B: End-to-end select + open path with explicit microphone config."""
        mic = MagicMock()
        mic.id = "usb-mic"
        mic.name = "Studio Condenser Mic"
        mic.channels = 2
        mic.isloopback = False
        mic.samplerate = 48000
        recorder = MagicMock()
        mock_stream = MagicMock()
        mock_stream.record.return_value = "pcm_data"
        recorder.__enter__.return_value = mock_stream
        mic.recorder.return_value = recorder

        cfg = Config(output_device="usb-mic", sample_rate=48000)

        with patch("soundcard.get_speaker", side_effect=Exception("not speaker")), \
             patch("soundcard.get_microphone", return_value=mic):
            backend = open_macos_capture_backend(cfg, blocksize=4096)
            self.assertEqual(backend.device_name, "Studio Condenser Mic")
            self.assertEqual(backend.channels, 2)
            self.assertEqual(backend.sample_rate, 48000)
            data = backend.record(4096)
            self.assertEqual(data, "pcm_data")
            mock_stream.record.assert_called_once_with(numframes=4096)
            backend.close()
