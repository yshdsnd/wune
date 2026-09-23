"""Hardware-free regression checks; native WASAPI loopback testing is still required."""

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
        compat = patch("wune.soundcard_compat.prepare_soundcard")
        compat.start()
        self.addCleanup(compat.stop)
        rate_patch = patch("wune.soundcard_compat.output_sample_rate", return_value=48000)
        self.detect_rate = rate_patch.start()
        self.addCleanup(rate_patch.stop)
        self.sc = types.ModuleType("soundcard")
        self.speaker = types.SimpleNamespace(id="render-id", name="HDMI Speakers", channels=2)
        self.sc.default_speaker = MagicMock(return_value=self.speaker)
        self.sc.get_speaker = MagicMock(return_value=self.speaker)
        self.loopback = MagicMock()
        self.loopback.isloopback = True
        self.sc.get_microphone = MagicMock(return_value=self.loopback)
        self.stream = MagicMock()
        self.recorder = self.loopback.recorder.return_value
        self.recorder.__enter__.return_value = self.stream
        self.modules = patch.dict(sys.modules, {"soundcard": self.sc})
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

    def test_default_render_endpoint_in_shared_stereo_mode(self):
        spectrum = self.make_spectrum()
        self.sc.default_speaker.assert_called_once_with()
        self.sc.get_microphone.assert_called_once_with(id="render-id", include_loopback=True)
        self.loopback.recorder.assert_called_once_with(
            channels=[0, 1], samplerate=48000, blocksize=4096, exclusive_mode=False,
        )
        self.assertEqual(spectrum.device, "HDMI Speakers")
        self.assertEqual(spectrum.fmax, 23999)

    def test_selected_render_endpoint(self):
        cfg = Config(output_device="Headphones")
        spectrum = self.audio.AudioSpectrum(cfg, cfg.bars, cfg.channels)
        self.addCleanup(spectrum.close)
        self.sc.get_speaker.assert_called_once_with("Headphones")
        self.sc.default_speaker.assert_not_called()
        self.sc.get_microphone.assert_called_once_with(id="render-id", include_loopback=True)

    def test_missing_default_endpoint_is_reported(self):
        self.sc.default_speaker.return_value = None
        with self.assertRaisesRegex(RuntimeError, "No Windows playback"):
            self.make_spectrum()
        self.loopback.recorder.assert_not_called()

    def test_missing_selected_endpoint_does_not_fall_back(self):
        self.sc.get_speaker.side_effect = IndexError("device not found")
        with self.assertRaises(IndexError):
            self.audio.AudioSpectrum(Config(output_device="missing"), 64)
        self.sc.default_speaker.assert_not_called()

    def test_non_loopback_microphone_is_rejected(self):
        self.loopback.isloopback = False
        with self.assertRaisesRegex(RuntimeError, "no loopback"):
            self.make_spectrum()
        self.loopback.recorder.assert_not_called()

    def test_mono_endpoint_is_rejected_instead_of_recording_garbage(self):
        self.speaker.channels = 1
        with self.assertRaisesRegex(RuntimeError, "stereo"):
            self.make_spectrum()
        self.loopback.recorder.assert_not_called()

    def test_mono_display_still_captures_stereo(self):
        spectrum = self.audio.AudioSpectrum(Config(channels=1), 64, 1)
        self.addCleanup(spectrum.close)
        self.assertEqual(self.loopback.recorder.call_args.kwargs["channels"], [0, 1])

    def test_analysis_matches_pre_cleanup_reference(self):
        reference = json.loads((Path(__file__).parent / "fixtures" / "spectrum_48k.json").read_text())
        for channels in (1, 2):
            with self.subTest(input_channels=channels):
                spectrum = self.make_spectrum()
                actual = []
                for frame in input_frames(channels):
                    self.stream.record.return_value = frame
                    actual.append(spectrum.step(1 / 60).copy())
                np.testing.assert_allclose(actual, reference[str(channels)], rtol=1e-6, atol=1e-7)
                self.assertTrue(spectrum.gated)
                self.assertTrue(np.any(actual[0]))

    def test_configured_rate_and_block_are_passed_through(self):
        cfg = Config(sample_rate=44100, block_size=2048)
        spectrum = self.audio.AudioSpectrum(cfg, cfg.bars, cfg.channels)
        self.addCleanup(spectrum.close)
        self.assertEqual(self.loopback.recorder.call_args.kwargs["samplerate"], 44100)
        self.assertEqual(self.loopback.recorder.call_args.kwargs["blocksize"], 2048)
        self.detect_rate.assert_not_called()

    def test_auto_rate_drives_capture_and_fft_for_selected_endpoint(self):
        for rate in (44100, 48000, 96000):
            with self.subTest(rate=rate):
                self.detect_rate.return_value = rate
                spectrum = self.audio.AudioSpectrum(Config(output_device="HDMI"), 64)
                self.addCleanup(spectrum.close)
                self.detect_rate.assert_called_with(self.speaker)
                self.assertEqual(spectrum.sr, rate)
                self.assertEqual(spectrum.freqs[-1], rate / 2)
                self.assertEqual(self.loopback.recorder.call_args.kwargs["samplerate"], rate)

    def test_mix_rate_failure_does_not_start_capture(self):
        self.detect_rate.side_effect = RuntimeError("mix format unavailable")
        with self.assertRaisesRegex(RuntimeError, "mix format unavailable"):
            self.make_spectrum()
        self.loopback.recorder.assert_not_called()

    def test_capture_start_failure_is_reported_without_input_fallback(self):
        self.recorder.__enter__.side_effect = RuntimeError("capture failed")
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            self.make_spectrum()
        self.loopback.recorder.assert_called_once()

    def test_close_exits_recorder_once(self):
        spectrum = self.make_spectrum()
        spectrum.close()
        spectrum.close()
        self.recorder.__exit__.assert_called_once()
        self.assertEqual(self.recorder.__exit__.call_args.args[-3:], (None, None, None))

    def test_record_requests_exact_fft_frame_count(self):
        spectrum = self.make_spectrum()
        self.stream.record.return_value = input_frames()[0]
        spectrum.step(1 / 60)
        self.stream.record.assert_called_once_with(numframes=4096)

    def test_sample_rate_range_is_applied_to_fft_bands(self):
        for rate, expected in ((44100, 20000), (48000, 20000), (96000, 40000)):
            with self.subTest(rate=rate):
                cfg = Config(sample_rate=rate)
                spectrum = self.audio.AudioSpectrum(cfg, cfg.bars, cfg.channels)
                self.addCleanup(spectrum.close)
                spectrum.set_range(cfg.min_freq_hz, cfg.spectrum_upper_hz(spectrum.sr))
                self.assertEqual(spectrum.fmax, expected)
                used = np.concatenate(spectrum._bin_idx)
                self.assertTrue(np.all(spectrum.freqs[used] < expected))
                if rate == 96000:
                    t = np.arange(cfg.block_size) / rate
                    tone = (0.2 * np.sin(2 * np.pi * 30000 * t)).astype(np.float32)
                    self.stream.record.return_value = np.column_stack((tone, tone))
                    levels = spectrum.step(1 / 60)
                    band = next(i for i, bins in enumerate(spectrum._bin_idx)
                                if np.any(np.abs(spectrum.freqs[bins] - 30000) < rate / cfg.block_size))
                    self.assertGreater(levels[0, band], 0)

    def test_range_policy_honors_configured_and_nyquist_limits(self):
        self.assertEqual(Config(max_freq_hz=16000).spectrum_upper_hz(48000), 16000)
        self.assertEqual(Config(max_freq_hz=100000).spectrum_upper_hz(96000), 40000)
        self.assertLess(Config().spectrum_upper_hz(32000), 16000)
        self.assertLess(Config().spectrum_upper_hz(64000), 32000)


class AppCleanupTests(unittest.TestCase):
    def setUp(self):
        self.pg = MagicMock()
        for i, name in enumerate(("QUIT", "KEYDOWN", "K_ESCAPE", "K_q", "K_F11", "K_SPACE", "K_i", "K_t", "MOUSEBUTTONDOWN", "VIDEORESIZE", "WINDOWSIZECHANGED")):
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
        self.backend.AudioSpectrum.return_value.device = "HDMI Speakers"
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
        self.assertIn("LOOPBACK:HDMI Speakers", self.app.renderer.info_text)
        self.assertIn("48.0 kHz", self.app.renderer.info_text)
        self.assertIn("float32", self.app.renderer.info_text)
        self.assertNotIn("24-bit", self.app.renderer.info_text)

    def test_app_passes_policy_range_and_displays_effective_limit(self):
        self.app.spectrum.set_range.assert_called_once_with(20.0, 20000.0)
        self.assertEqual(self.app.cfg.max_freq_hz, self.app.spectrum.fmax)
        self.app.spectrum.sr = 96000
        self.app.spectrum.fmax = 40000
        self.app.spectrum.set_range.reset_mock()
        app = self.app_module.App(Config(sample_rate=48000))
        app.spectrum.set_range.assert_called_once_with(20.0, 40000.0)
        self.assertEqual(app.cfg.max_freq_hz, 40000)

    def test_mouse_and_keyboard_share_switch_without_touching_audio(self):
        self.app.renderer.badge_contains.return_value = True
        self.app.spectrum.reset_mock()
        levels = self.app.levels
        self.app.paused = True
        self.app.handle_event(types.SimpleNamespace(type=self.pg.MOUSEBUTTONDOWN, button=1, pos=(100, 20)))
        self.app.handle_event(types.SimpleNamespace(type=self.pg.KEYDOWN, key=self.pg.K_t))
        self.assertEqual(self.app.renderer.next_preset.call_count, 2)
        self.assertIs(self.app.levels, levels)
        self.assertTrue(self.app.paused)
        self.assertEqual(self.app.spectrum.mock_calls, [])
        self.backend.AudioSpectrum.assert_called_once()

    def test_other_mouse_clicks_do_not_switch(self):
        self.app.renderer.badge_contains.return_value = False
        self.app.handle_event(types.SimpleNamespace(type=self.pg.MOUSEBUTTONDOWN, button=1, pos=(0, 0)))
        self.app.renderer.badge_contains.return_value = True
        self.app.handle_event(types.SimpleNamespace(type=self.pg.MOUSEBUTTONDOWN, button=3, pos=(100, 20)))
        self.app.renderer.next_preset.assert_not_called()

    def test_initial_preset_is_applied(self):
        self.app.renderer.apply_preset.assert_called_once_with("CLASSIC")
        self.app.renderer.apply_preset.reset_mock()
        self.app_module.App(Config(initial_preset="BLUE"))
        self.app.renderer.apply_preset.assert_called_once_with("BLUE")

    def test_fullscreen_restores_previous_window_size(self):
        window = MagicMock()
        window.get_size.return_value = (960, 600)
        fullscreen = MagicMock()
        fullscreen.get_size.return_value = (1920, 1080)
        self.app.screen = window
        self.pg.display.set_mode.side_effect = [fullscreen, window]
        self.app.toggle_fullscreen()
        self.assertTrue(self.app._fullscreen)
        self.app.toggle_fullscreen()
        self.assertFalse(self.app._fullscreen)
        self.pg.display.set_mode.assert_called_with((960, 600), self.pg.RESIZABLE)
        self.assertEqual(self.app.renderer.resize.call_count, 2)

    def test_info_toggle_exits_fullscreen_if_it_no_longer_fits(self):
        from wune.layout import minimum_window_size
        self.app.cfg.info_enabled = False
        self.app.screen.get_size.return_value = minimum_window_size(self.app.cfg)
        self.app._fullscreen = True
        self.app.handle_event(types.SimpleNamespace(type=self.pg.KEYDOWN, key=self.pg.K_i))
        self.assertTrue(self.app.cfg.info_enabled)
        self.assertFalse(self.app._fullscreen)
        self.pg.display.set_mode.assert_called_with(self.app._windowed_size, self.pg.RESIZABLE)

    def test_resize_clamps_window_without_reopening_audio(self):
        from wune.layout import minimum_window_size
        self.app.screen.get_size.return_value = (100, 100)
        self.app.spectrum.reset_mock()
        levels = self.app.levels
        self.app.handle_event(types.SimpleNamespace(type=self.pg.VIDEORESIZE, size=(100, 100)))
        self.pg.display.set_mode.assert_called_with(minimum_window_size(self.app.cfg), self.pg.RESIZABLE)
        self.app.renderer.resize.assert_called_once_with(self.app.screen)
        self.assertIs(self.app.levels, levels)
        self.assertEqual(self.app.spectrum.mock_calls, [])
        self.backend.AudioSpectrum.assert_called_once()


if __name__ == "__main__":
    unittest.main()
