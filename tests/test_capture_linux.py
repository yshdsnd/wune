"""Tests for Linux and Unix PulseAudio / PipeWire capture backend."""
from unittest.mock import MagicMock, patch
import sys
import unittest
import numpy as np

from wune.config import Config
from wune.capture import (
    CaptureBackend,
    PulseAudioLoopbackBackend,
    create_capture_backend,
)
from wune.soundcard_compat import get_soundcard


class PulseAudioCaptureBackendTests(unittest.TestCase):
    """Test PulseAudioLoopbackBackend and Linux loopback device resolution."""

    def _create_mock_speaker(self, name="Built-in Audio Analog Stereo", id_val="alsa_output.pci-0000_00_1f.3.analog-stereo", channels=2):
        speaker = MagicMock()
        speaker.name = name
        speaker.id = id_val
        speaker.channels = channels
        return speaker

    def _create_mock_loopback(self, name="Monitor of Built-in Audio", id_val="alsa_output.pci-0000_00_1f.3.analog-stereo.monitor", channels=2):
        mic = MagicMock()
        mic.name = name
        mic.id = id_val
        mic.channels = channels
        mic.isloopback = True

        recorder_ctx = MagicMock()
        stream_mock = MagicMock()
        stream_mock.record.side_effect = lambda numframes: np.ones((numframes, channels), dtype=np.float32) * 0.5
        recorder_ctx.__enter__.return_value = stream_mock
        recorder_ctx.__exit__.return_value = None
        mic.recorder.return_value = recorder_ctx
        mic._stream_mock = stream_mock
        return mic

    def test_satisfies_capture_backend_protocol(self):
        mock_sc = MagicMock()
        speaker = self._create_mock_speaker()
        loopback = self._create_mock_loopback()
        mock_sc.default_speaker.return_value = speaker
        mock_sc.get_microphone.return_value = loopback

        with patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            backend = PulseAudioLoopbackBackend(Config())
            self.assertIsInstance(backend, CaptureBackend)
            backend.close()

    def test_default_speaker_and_default_sample_rate(self):
        mock_sc = MagicMock()
        speaker = self._create_mock_speaker()
        loopback = self._create_mock_loopback()
        mock_sc.default_speaker.return_value = speaker
        mock_sc.get_microphone.return_value = loopback

        with patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            cfg = Config(sample_rate=None)
            backend = PulseAudioLoopbackBackend(cfg)

            mock_sc.default_speaker.assert_called_once()
            self.assertEqual(backend.sample_rate, 48000)
            self.assertEqual(backend.channels, 2)
            self.assertEqual(backend.device_name, speaker.name)

            loopback.recorder.assert_called_once_with(
                samplerate=48000,
                channels=[0, 1],
                blocksize=cfg.block_size,
            )
            backend.close()

    def test_custom_output_device_and_custom_sample_rate(self):
        mock_sc = MagicMock()
        speaker = self._create_mock_speaker(name="USB DAC", id_val="alsa_output.usb-dac")
        loopback = self._create_mock_loopback(name="Monitor of USB DAC", id_val="alsa_output.usb-dac.monitor")
        mock_sc.get_speaker.return_value = speaker
        mock_sc.get_microphone.return_value = loopback

        with patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            cfg = Config(output_device="alsa_output.usb-dac", sample_rate=44100)
            backend = PulseAudioLoopbackBackend(cfg, blocksize=2048)

            mock_sc.get_speaker.assert_called_once_with("alsa_output.usb-dac")
            self.assertEqual(backend.sample_rate, 44100)
            loopback.recorder.assert_called_once_with(
                samplerate=44100,
                channels=[0, 1],
                blocksize=2048,
            )
            backend.close()

    def test_record_stereo(self):
        mock_sc = MagicMock()
        speaker = self._create_mock_speaker()
        loopback = self._create_mock_loopback(channels=2)
        mock_sc.default_speaker.return_value = speaker
        mock_sc.get_microphone.return_value = loopback

        with patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            backend = PulseAudioLoopbackBackend(Config())
            data = backend.record(512)
            self.assertEqual(data.shape, (512, 2))
            self.assertEqual(data.dtype, np.float32)
            self.assertTrue(np.allclose(data, 0.5))
            backend.close()

    def test_record_mono_duplicated_to_stereo(self):
        mock_sc = MagicMock()
        speaker = self._create_mock_speaker(channels=1)
        loopback = self._create_mock_loopback(channels=1)
        # 1D array returned by mono recorder
        loopback._stream_mock.record.side_effect = lambda numframes: np.ones((numframes, 1), dtype=np.float32) * 0.75
        mock_sc.default_speaker.return_value = speaker
        mock_sc.get_microphone.return_value = loopback

        with patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            backend = PulseAudioLoopbackBackend(Config())
            loopback.recorder.assert_called_once_with(
                samplerate=48000,
                channels=[0],
                blocksize=Config().block_size,
            )
            data = backend.record(256)
            self.assertEqual(data.shape, (256, 2))
            self.assertEqual(data.dtype, np.float32)
            self.assertTrue(np.allclose(data[:, 0], 0.75))
            self.assertTrue(np.allclose(data[:, 1], 0.75))
            backend.close()

    def test_record_mono_1d_array_duplicated_to_stereo(self):
        mock_sc = MagicMock()
        speaker = self._create_mock_speaker(channels=1)
        loopback = self._create_mock_loopback(channels=1)
        # Flat 1D array (numframes,)
        loopback._stream_mock.record.side_effect = lambda numframes: np.ones(numframes, dtype=np.float32) * 0.25
        mock_sc.default_speaker.return_value = speaker
        mock_sc.get_microphone.return_value = loopback

        with patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            backend = PulseAudioLoopbackBackend(Config())
            data = backend.record(128)
            self.assertEqual(data.shape, (128, 2))
            self.assertEqual(data.dtype, np.float32)
            self.assertTrue(np.allclose(data[:, 0], 0.25))
            self.assertTrue(np.allclose(data[:, 1], 0.25))
            backend.close()

    def test_find_monitor_source_by_all_microphones_id_match(self):
        mock_sc = MagicMock()
        speaker = self._create_mock_speaker(id_val="sink_test")
        loopback = self._create_mock_loopback(id_val="sink_test.monitor")
        mock_sc.default_speaker.return_value = speaker
        # get_microphone raises or returns None
        mock_sc.get_microphone.side_effect = IndexError("Not found")
        mock_sc.all_microphones.return_value = [loopback]

        with patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            backend = PulseAudioLoopbackBackend(Config())
            self.assertEqual(backend.device_name, speaker.name)
            backend.close()

    def test_find_monitor_source_by_name_match(self):
        mock_sc = MagicMock()
        speaker = self._create_mock_speaker(name="Speaker Name", id_val="sink_other")
        loopback = self._create_mock_loopback(name="Monitor of Speaker Name", id_val="source_other_mon")
        mock_sc.default_speaker.return_value = speaker
        mock_sc.get_microphone.side_effect = IndexError("Not found")
        mock_sc.all_microphones.return_value = [loopback]

        with patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            backend = PulseAudioLoopbackBackend(Config())
            self.assertEqual(backend.device_name, speaker.name)
            backend.close()

    def test_find_monitor_source_fallback_to_any_loopback(self):
        mock_sc = MagicMock()
        speaker = self._create_mock_speaker(name="Unknown Sink", id_val="sink_unknown")
        loopback = self._create_mock_loopback(name="Generic Monitor", id_val="generic.monitor")
        mock_sc.default_speaker.return_value = speaker
        mock_sc.get_microphone.side_effect = IndexError("Not found")
        mock_sc.all_microphones.return_value = [loopback]

        with patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            backend = PulseAudioLoopbackBackend(Config())
            self.assertEqual(backend.device_name, speaker.name)
            backend.close()

    def test_no_speaker_raises_error(self):
        mock_sc = MagicMock()
        mock_sc.default_speaker.return_value = None

        with patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            with self.assertRaises(RuntimeError) as ctx:
                PulseAudioLoopbackBackend(Config())
            self.assertIn("No playback device is available", str(ctx.exception))

    def test_no_monitor_source_raises_error(self):
        mock_sc = MagicMock()
        speaker = self._create_mock_speaker()
        mock_sc.default_speaker.return_value = speaker
        mock_sc.get_microphone.side_effect = IndexError("Not found")
        mock_sc.all_microphones.return_value = []

        with patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            with self.assertRaises(RuntimeError) as ctx:
                PulseAudioLoopbackBackend(Config())
            self.assertIn("No PulseAudio/PipeWire loopback monitor source found", str(ctx.exception))

    def test_invalid_sample_rate_raises_error(self):
        mock_sc = MagicMock()
        speaker = self._create_mock_speaker()
        loopback = self._create_mock_loopback()
        mock_sc.default_speaker.return_value = speaker
        mock_sc.get_microphone.return_value = loopback

        with patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            with self.assertRaises(ValueError) as ctx:
                PulseAudioLoopbackBackend(Config(sample_rate=-1))
            self.assertIn("Sample rate must be positive", str(ctx.exception))

    def test_close_releases_stream_context(self):
        mock_sc = MagicMock()
        speaker = self._create_mock_speaker()
        loopback = self._create_mock_loopback()
        mock_sc.default_speaker.return_value = speaker
        mock_sc.get_microphone.return_value = loopback

        with patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            backend = PulseAudioLoopbackBackend(Config())
            backend.close()
            # __exit__ should have been called on recorder context
            recorder_ctx = loopback.recorder.return_value
            recorder_ctx.__exit__.assert_called_once()

    def test_audio_spectrum_integration_on_linux(self):
        from wune.spectrum_audio import AudioSpectrum

        mock_sc = MagicMock()
        speaker = self._create_mock_speaker()
        loopback = self._create_mock_loopback()
        mock_sc.default_speaker.return_value = speaker
        mock_sc.get_microphone.return_value = loopback

        with patch("sys.platform", "linux"), \
             patch("wune.soundcard_compat.get_soundcard", return_value=mock_sc):
            cfg = Config(sample_rate=48000)
            spectrum = AudioSpectrum(cfg, bars=16, channels=2)
            self.assertEqual(spectrum.sr, 48000)
            self.assertEqual(spectrum.channels_eff, 2)
            self.assertEqual(spectrum.device, speaker.name)
            levels = spectrum.step(1 / 60)
            self.assertEqual(levels.shape, (2, 16))
            spectrum.close()


class SoundCardCompatTests(unittest.TestCase):
    """Test soundcard_compat FreeBSD platform adaptation."""

    def test_get_soundcard_on_freebsd_switches_platform_temporarily(self):
        with patch("sys.platform", "freebsd14"):
            with patch.dict("sys.modules", clear=False):
                if "soundcard" in sys.modules:
                    del sys.modules["soundcard"]
                # Verify that get_soundcard executes without error
                with patch("builtins.__import__") as mock_import:
                    fake_sc = MagicMock()
                    mock_import.return_value = fake_sc
                    sc = get_soundcard()
                    self.assertIsNotNone(sc)
                    # Platform should be restored
                    self.assertEqual(sys.platform, "freebsd14")
