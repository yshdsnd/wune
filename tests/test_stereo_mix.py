"""Power-domain stereo mix, persistence and compact geometry without hardware."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from wune.appearance import AppearanceDraft, AppearanceState
from wune.config import Config
from wune.layout import calculate_layout, minimum_window_size, fit_window_size
from wune.settings import SettingsStore
from wune.spectrum_audio import AudioSpectrum


class StereoMixTests(unittest.TestCase):
    def make_spectrum(self, mode="stereo", channels=2):
        cfg = Config(channel_mode=mode, channels=channels, output_floor=0)
        backend = Mock(sample_rate=48000, channels=2, device_name="test")
        spectrum = AudioSpectrum(cfg, cfg.bars, cfg.channels, capture_backend=backend)
        spectrum.set_range(20, 20000)
        self.addCleanup(spectrum.close)
        return spectrum

    def tone(self):
        # Bin-centred tone safely inside one band, below clipping.
        return (0.25 * np.sin(2*np.pi*96*np.arange(4096)/4096)).astype(np.float32)

    def test_identical_and_opposite_phase_match_separate_level(self):
        mono = self.tone()
        stereo = self.make_spectrum()
        mix = self.make_spectrum("stereo_mix")
        reference = stereo._map_levels(np.column_stack((mono, mono)))
        for right in (mono, -mono):
            actual = mix._map_levels(np.column_stack((mono, right)))
            self.assertEqual(actual.shape, (1, 64))
            np.testing.assert_allclose(actual[0], reference[0], atol=1e-7)
        self.assertAlmostEqual(float(reference.max()) * 66 - 66,
                               20*np.log10(0.25), places=3)

    def test_either_single_active_side_is_three_db_lower(self):
        mono = self.tone()
        mix = self.make_spectrum("stereo_mix")
        reference = mix._map_levels(np.column_stack((mono, mono)))
        for data in (np.column_stack((mono, mono*0)), np.column_stack((mono*0, mono))):
            actual = mix._map_levels(data)
            self.assertAlmostEqual(float(actual.max()-reference.max())*66,
                                   10*np.log10(0.5), places=4)

    def test_distinct_left_and_right_frequencies_both_survive(self):
        left = self.tone()
        right = (0.1*np.sin(2*np.pi*512*np.arange(4096)/4096)).astype(np.float32)
        data = np.column_stack((left, right))
        separate = self.make_spectrum()._map_levels(data)
        mixed = self.make_spectrum("stereo_mix")._map_levels(data)[0]
        for ch in range(2):
            band = separate[ch].argmax()
            self.assertAlmostEqual(float(mixed[band]-separate[ch, band])*66,
                                   10*np.log10(0.5), places=3)

    def test_switch_keeps_capture_mapping_and_restores_stereo(self):
        spectrum = self.make_spectrum()
        backend = spectrum.capture
        data = np.column_stack((self.tone(), -self.tone()))
        backend.record.return_value = data
        original = spectrum._map_levels(data).copy()
        bins = spectrum._bin_idx
        spectrum.step(1/60)
        spectrum.set_display_mode("stereo_mix")
        self.assertIsNone(spectrum._vis_env.y)
        self.assertEqual(spectrum.step(1/60).shape, (1, 64))
        self.assertIs(spectrum.capture, backend)
        self.assertIs(spectrum._bin_idx, bins)
        spectrum.set_display_mode("stereo")
        np.testing.assert_array_equal(spectrum._map_levels(data), original)
        backend.record.return_value = data*0
        self.assertEqual(spectrum.step(1/60).shape, (2, 64))
        backend.close.assert_not_called()

    def test_mix_reads_both_sides_even_with_legacy_one_display_channel(self):
        spectrum = self.make_spectrum("stereo_mix", channels=1)
        spectrum.capture.record.return_value = np.column_stack((self.tone()*0, self.tone()))
        self.assertGreater(spectrum.step(1).max(), 0.5)
        spectrum.capture.record.return_value = self.tone()[:, None]
        self.assertEqual(spectrum.step(1).shape, (1, 64))

    def test_settings_round_trip_preview_reset_and_old_defaults(self):
        cfg = Config()
        baseline = AppearanceState.capture(cfg, "CLASSIC", {})
        draft = AppearanceDraft(baseline)
        draft.state.layout["channel_mode"] = "stereo_mix"
        draft.snapshot().apply(cfg)
        self.assertEqual(cfg.display_channels, 1)
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory)/"settings.json")
            store.save(cfg, (800, 400), (0, 0), "CLASSIC")
            loaded, _ = store.load(Config())
            self.assertEqual(loaded.channel_mode, "stereo_mix")
            self.assertEqual(loaded.channels, 2)
            store.path.write_text('{"version":2,"appearance":{}}', encoding="utf-8")
            loaded, _ = store.load(Config())
            self.assertEqual(loaded.channel_mode, "stereo")
        baseline.apply(cfg)
        self.assertEqual(cfg.display_channels, 2)
        draft.reset()
        self.assertEqual(draft.state.layout["channel_mode"], "stereo")

    def test_single_plot_removes_second_channel_space_in_both_orientations(self):
        for orientation in ("frequency_horizontal", "frequency_vertical"):
            for arrangement in ("vertical", "horizontal"):
                cfg = Config(spectrum_orientation=orientation, channel_layout=arrangement)
                stereo = minimum_window_size(cfg)
                mix = replace(cfg, channel_mode="stereo_mix")
                minimum = minimum_window_size(mix)
                axis = 1 if arrangement == "vertical" else 0
                self.assertLessEqual(minimum[axis], stereo[axis])
                if axis == 1 or stereo[axis] > 480:
                    self.assertLess(minimum[axis], stereo[axis])
                for size in (minimum, (1280, 800), (2000, 1600)):
                    fitted = fit_window_size(size, mix)
                    self.assertEqual(fit_window_size(fitted, mix), fitted)
                    self.assertEqual(len(calculate_layout(fitted, mix).plots), 1)
