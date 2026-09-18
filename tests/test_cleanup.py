"""Hardware-free regression checks; native VB-CABLE testing is still required."""

import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from wune.config import Config


def input_frames(channels=2):
    """Fixed 48 kHz tones, quiet input and silence after a transient."""
    t = np.arange(4096) / 48000
    left = 0.2 * np.sin(2 * np.pi * 1000 * t)
    right = 0.12 * np.sin(2 * np.pi * 12000 * t)
    tone = np.column_stack((left, right)).astype(np.float32)
    if channels == 1:
        tone = tone[:, :1]
    return [tone.copy() for _ in range(4)] + [tone * 0.02, tone * 0, tone * 0]


class AudioCleanupTests(unittest.TestCase):
    def setUp(self):
        self.sd = types.ModuleType("sounddevice")
        self.sd.PortAudioError = type("PortAudioError", (Exception,), {})
        self.sd.query_devices = MagicMock(return_value={"max_input_channels": 2})
        self.sd.query_hostapis = MagicMock()
        self.sd.default = types.SimpleNamespace(device=(0, 1), hostapi=0)
        self.stream = MagicMock()
        self.sd.InputStream = MagicMock(return_value=self.stream)
        self.modules = patch.dict(sys.modules, {"sounddevice": self.sd})
        self.modules.start()
        sys.modules.pop("wune.spectrum_audio", None)
        self.audio = importlib.import_module("wune.spectrum_audio")

    def tearDown(self):
        sys.modules.pop("wune.spectrum_audio", None)
        self.modules.stop()

    def make_spectrum(self):
        cfg = Config()
        spectrum = self.audio.AudioSpectrum(cfg, cfg.bars, cfg.channels)
        spectrum.set_range(cfg.min_freq_hz, cfg.max_freq_hz)
        self.addCleanup(spectrum.close)
        return spectrum

    def test_working_input_settings(self):
        spectrum = self.make_spectrum()
        self.sd.InputStream.assert_called_once_with(
            device="CABLE Output (VB-Audio Virtual Cable), Windows WASAPI",
            channels=2, samplerate=48000, blocksize=4096, dtype="float32",
        )
        self.assertEqual(spectrum.fmax, 23999)

    def test_analysis_matches_pre_cleanup_reference(self):
        reference = json.loads((Path(__file__).parent / "fixtures" / "spectrum_48k.json").read_text())
        for channels in (1, 2):
            with self.subTest(input_channels=channels):
                spectrum = self.make_spectrum()
                actual = []
                for frame in input_frames(channels):
                    self.stream.read.return_value = (frame, False)
                    actual.append(spectrum.step(1 / 60).copy())
                np.testing.assert_allclose(actual, reference[str(channels)], rtol=1e-6, atol=1e-7)
                self.assertTrue(spectrum.gated)
                self.assertTrue(np.any(actual[0]))

    def test_configured_rate_and_block_are_passed_through(self):
        cfg = Config(sample_rate=44100, block_size=2048)
        spectrum = self.audio.AudioSpectrum(cfg, cfg.bars, cfg.channels)
        self.addCleanup(spectrum.close)
        self.assertEqual(self.sd.InputStream.call_args.kwargs["samplerate"], 44100)
        self.assertEqual(self.sd.InputStream.call_args.kwargs["blocksize"], 2048)

    def test_failed_start_is_closed_before_mono_retry(self):
        failed, retry = MagicMock(), MagicMock()
        failed.start.side_effect = self.sd.PortAudioError("start failed")
        self.sd.InputStream.side_effect = [failed, retry]
        spectrum = self.make_spectrum()
        failed.close.assert_called_once()
        self.assertEqual(self.sd.InputStream.call_args.kwargs["channels"], 1)
        self.assertEqual(spectrum.channels_eff, 1)

    def test_final_start_error_is_not_hidden(self):
        failed, retry = MagicMock(), MagicMock()
        failed.start.side_effect = self.sd.PortAudioError("stereo failed")
        retry.start.side_effect = self.sd.PortAudioError("mono failed")
        self.sd.InputStream.side_effect = [failed, retry]
        with self.assertRaises(self.sd.PortAudioError):
            self.make_spectrum()
        failed.close.assert_called_once()
        retry.close.assert_called_once()

    def test_close_runs_even_if_stop_fails(self):
        spectrum = self.audio.AudioSpectrum(Config(), 64)
        self.stream.stop.side_effect = self.sd.PortAudioError("stop failed")
        with self.assertRaises(self.sd.PortAudioError):
            spectrum.close()
        self.stream.close.assert_called_once()


class AppCleanupTests(unittest.TestCase):
    def setUp(self):
        self.pg = MagicMock()
        for i, name in enumerate(("QUIT", "KEYDOWN", "K_ESCAPE", "K_q", "K_F11", "K_SPACE", "K_i")):
            setattr(self.pg, name, i + 1)
        self.backend = MagicMock()
        self.modules = patch.dict(sys.modules, {
            "pygame": self.pg,
            "wune.spectrum_audio": self.backend,
        })
        self.modules.start()
        for name in ("wune.app", "wune.renderer"):
            sys.modules.pop(name, None)
        self.app_module = importlib.import_module("wune.app")
        self.renderer_patch = patch.object(self.app_module, "LedBarRenderer")
        self.renderer_patch.start()
        self.backend.AudioSpectrum.return_value.device = Config().input_device
        self.backend.AudioSpectrum.return_value.sr = 48000
        self.backend.AudioSpectrum.return_value.fmax = 23999
        self.backend.AudioSpectrum.return_value.gated = False
        self.backend.AudioSpectrum.return_value.last_rms = 0.0
        self.app = self.app_module.App(Config())

    def tearDown(self):
        self.renderer_patch.stop()
        for name in ("wune.app", "wune.renderer"):
            sys.modules.pop(name, None)
        self.modules.stop()

    def test_pause_keeps_last_frame_and_does_not_read_audio(self):
        frame = np.full((2, 64), 0.5, dtype=np.float32)
        self.app.spectrum.step.return_value = frame
        self.pg.event.get.side_effect = [[], [types.SimpleNamespace(type=self.pg.KEYDOWN, key=self.pg.K_SPACE)], [types.SimpleNamespace(type=self.pg.QUIT)]]
        self.app.run()
        self.app.spectrum.step.assert_called_once()
        calls = self.app.renderer.draw.call_args_list
        self.assertEqual(len(calls), 2)
        for call in calls:
            np.testing.assert_array_equal(call.args[0], frame)
        self.app.spectrum.close.assert_called_once()
        self.pg.quit.assert_called_once()

    def test_pause_before_first_read_draws_zero_array(self):
        self.pg.event.get.side_effect = [[types.SimpleNamespace(type=self.pg.KEYDOWN, key=self.pg.K_SPACE)], [types.SimpleNamespace(type=self.pg.QUIT)]]
        self.app.run()
        self.app.spectrum.step.assert_not_called()
        np.testing.assert_array_equal(self.app.renderer.draw.call_args.args[0], np.zeros((2, 64)))

    def test_read_error_closes_stream_and_display(self):
        self.pg.event.get.return_value = []
        self.app.spectrum.step.side_effect = RuntimeError("input lost")
        with self.assertRaisesRegex(RuntimeError, "input lost"):
            self.app.run()
        self.app.spectrum.close.assert_called_once()
        self.pg.quit.assert_called_once()

    def test_info_uses_configured_input_instead_of_placeholder(self):
        self.assertIn("CABLE Output", self.app.renderer.info_text)
        self.assertIn("48.0 kHz", self.app.renderer.info_text)
        self.assertIn("float32", self.app.renderer.info_text)
        self.assertNotIn("24-bit", self.app.renderer.info_text)


if __name__ == "__main__":
    unittest.main()
