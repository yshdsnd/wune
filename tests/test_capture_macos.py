"""Tests for macOS audio capture selection and fallback routing."""
from contextlib import ExitStack
import types
import unittest
from unittest.mock import MagicMock, patch

from wune.config import Config
from wune.capture_macos import select_macos_output, open_macos_capture


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

    def test_select_macos_output_raises_when_no_device_found(self):
        with patch("soundcard.default_speaker", side_effect=Exception("none")), \
             patch("soundcard.default_microphone", side_effect=Exception("none")):
            with self.assertRaisesRegex(RuntimeError, "No macOS audio device"):
                select_macos_output(Config())

    def test_open_macos_capture_uses_blackhole_loopback_when_available(self):
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
                self.assertEqual(name, "MacBook Speakers")
                self.assertEqual(ch_eff, 2)
                blackhole_mic.recorder.assert_called_once_with(
                    samplerate=48000,
                    channels=[0, 1],
                    blocksize=4096,
                )

    def test_open_macos_capture_falls_back_to_default_microphone(self):
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
                self.assertEqual(name, "Headphones")
                self.assertEqual(ch_eff, 1)
                default_mic.recorder.assert_called_once_with(
                    samplerate=44100,
                    channels=[0],
                    blocksize=2048,
                )
