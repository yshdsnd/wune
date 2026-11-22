import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
import pygame as pg

from wune.app import App
from wune.application_menu import ApplicationMenu
from wune.capture import MacOsCaptureBackend, PulseAudioLoopbackBackend, WasapiLoopbackBackend
from wune.config import Config
from wune.spectrum_audio import AudioSpectrum


class WasapiNeedsRestartTests(unittest.TestCase):
    def _make_backend(self, *, device_id="dev-1", device_name="Speakers", sample_rate=48000,
                      follow_default=True, follow_sample_rate=True):
        backend = WasapiLoopbackBackend.__new__(WasapiLoopbackBackend)
        backend._device_id = device_id
        backend._device_name = device_name
        backend._sample_rate = sample_rate
        backend._follow_default = follow_default
        backend._follow_sample_rate = follow_sample_rate
        return backend

    def test_returns_false_when_unchanged(self):
        backend = self._make_backend()
        fake_sc = SimpleNamespace(default_speaker=lambda: SimpleNamespace(id="dev-1", name="Speakers"))
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc), \
             patch("wune.soundcard_compat.output_sample_rate", return_value=48000):
            self.assertFalse(backend.needs_restart())

    def test_returns_true_when_default_speaker_id_changes(self):
        backend = self._make_backend()
        fake_sc = SimpleNamespace(default_speaker=lambda: SimpleNamespace(id="dev-2", name="Headphones"))
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc), \
             patch("wune.soundcard_compat.output_sample_rate", return_value=48000):
            self.assertTrue(backend.needs_restart())

    def test_returns_true_when_mix_sample_rate_changes(self):
        backend = self._make_backend(sample_rate=48000)
        fake_sc = SimpleNamespace(default_speaker=lambda: SimpleNamespace(id="dev-1", name="Speakers"))
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc), \
             patch("wune.soundcard_compat.output_sample_rate", return_value=192000):
            self.assertTrue(backend.needs_restart())

    def test_returns_false_when_output_device_is_explicitly_pinned(self):
        backend = self._make_backend(follow_default=False)
        self.assertFalse(backend.needs_restart())

    def test_returns_false_when_probe_raises(self):
        backend = self._make_backend()
        with patch("wune.soundcard_compat.get_soundcard", side_effect=RuntimeError("COM error")):
            self.assertFalse(backend.needs_restart())


class PulseAudioNeedsRestartTests(unittest.TestCase):
    def _make_backend(
        self,
        *,
        device_id="alsa_output.pci-0000_00_1f.3.analog-stereo",
        speaker_name="Built-in Audio Analog Stereo",
        device_name=None,
        sample_rate=48000,
        follow_default=True,
        follow_sample_rate=True,
        exact_match=True,
        closed=False,
    ):
        backend = PulseAudioLoopbackBackend.__new__(PulseAudioLoopbackBackend)
        backend._device_id = device_id
        backend._speaker_name = speaker_name
        backend._device_name = speaker_name if device_name is None else device_name
        backend._sample_rate = sample_rate
        backend._follow_default = follow_default
        backend._follow_sample_rate = follow_sample_rate
        backend._exact_match = exact_match
        backend._closed = closed
        return backend

    def test_returns_false_when_unchanged(self):
        backend = self._make_backend()
        speaker = SimpleNamespace(
            id="alsa_output.pci-0000_00_1f.3.analog-stereo",
            name="Built-in Audio Analog Stereo",
        )
        fake_sc = SimpleNamespace(default_speaker=lambda: speaker)
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc), \
             patch("wune.soundcard_compat.pulseaudio_output_sample_rate", return_value=48000):
            self.assertFalse(backend.needs_restart())

    def test_returns_true_when_default_speaker_id_changes(self):
        backend = self._make_backend()
        speaker = SimpleNamespace(id="alsa_output.usb-DAC-00.analog-stereo", name="USB DAC")
        fake_sc = SimpleNamespace(default_speaker=lambda: speaker)
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc), \
             patch("wune.soundcard_compat.pulseaudio_output_sample_rate", return_value=48000):
            self.assertTrue(backend.needs_restart())

    def test_returns_true_when_default_speaker_name_changes(self):
        backend = self._make_backend()
        speaker = SimpleNamespace(
            id="alsa_output.pci-0000_00_1f.3.analog-stereo",
            name="Built-in Audio Headphones",
        )
        fake_sc = SimpleNamespace(default_speaker=lambda: speaker)
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc), \
             patch("wune.soundcard_compat.pulseaudio_output_sample_rate", return_value=48000):
            self.assertTrue(backend.needs_restart())

    def test_returns_true_when_sink_sample_rate_changes(self):
        backend = self._make_backend(sample_rate=48000)
        speaker = SimpleNamespace(
            id="alsa_output.pci-0000_00_1f.3.analog-stereo",
            name="Built-in Audio Analog Stereo",
        )
        fake_sc = SimpleNamespace(default_speaker=lambda: speaker)
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc), \
             patch("wune.soundcard_compat.pulseaudio_output_sample_rate", return_value=96000):
            self.assertTrue(backend.needs_restart())

    def test_returns_false_when_sample_rate_is_explicitly_pinned(self):
        backend = self._make_backend(sample_rate=48000, follow_sample_rate=False)
        speaker = SimpleNamespace(
            id="alsa_output.pci-0000_00_1f.3.analog-stereo",
            name="Built-in Audio Analog Stereo",
        )
        fake_sc = SimpleNamespace(default_speaker=lambda: speaker)
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc), \
             patch("wune.soundcard_compat.pulseaudio_output_sample_rate", return_value=96000):
            self.assertFalse(backend.needs_restart())

    def test_returns_false_when_sample_rate_probe_unavailable(self):
        backend = self._make_backend(sample_rate=48000)
        speaker = SimpleNamespace(
            id="alsa_output.pci-0000_00_1f.3.analog-stereo",
            name="Built-in Audio Analog Stereo",
        )
        fake_sc = SimpleNamespace(default_speaker=lambda: speaker)
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc), \
             patch("wune.soundcard_compat.pulseaudio_output_sample_rate", return_value=0):
            self.assertFalse(backend.needs_restart())

    def test_returns_false_when_output_device_is_explicitly_pinned(self):
        backend = self._make_backend(follow_default=False)
        self.assertFalse(backend.needs_restart())

    def test_returns_false_when_closed(self):
        backend = self._make_backend(closed=True)
        self.assertFalse(backend.needs_restart())

    def test_returns_false_when_default_speaker_is_none(self):
        backend = self._make_backend()
        fake_sc = SimpleNamespace(default_speaker=lambda: None)
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc):
            self.assertFalse(backend.needs_restart())

    def test_returns_false_when_probe_raises(self):
        backend = self._make_backend()
        with patch("wune.soundcard_compat.get_soundcard", side_effect=IndexError("no soundcard with id @DEFAULT_SINK@")):
            self.assertFalse(backend.needs_restart())

    def test_fallback_monitor_does_not_loop_when_no_exact_monitor_exists(self):
        backend = self._make_backend(
            device_id="sink_unknown",
            speaker_name="Unknown Sink",
            device_name="Generic Monitor",
            exact_match=False,
        )
        speaker = SimpleNamespace(id="sink_unknown", name="Unknown Sink")
        fake_sc = MagicMock()
        fake_sc.default_speaker.return_value = speaker
        fake_sc.get_microphone.side_effect = IndexError("Not found")
        fake_sc.all_microphones.return_value = [
            SimpleNamespace(id="generic.monitor", name="Generic Monitor", isloopback=True)
        ]
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc), \
             patch("wune.soundcard_compat.pulseaudio_output_sample_rate", return_value=48000):
            self.assertFalse(backend.needs_restart())

    def test_fallback_monitor_restarts_when_exact_monitor_appears(self):
        backend = self._make_backend(
            device_id="sink_unknown",
            speaker_name="Unknown Sink",
            device_name="Generic Monitor",
            exact_match=False,
        )
        speaker = SimpleNamespace(id="sink_unknown", name="Unknown Sink")
        exact_mon = SimpleNamespace(id="sink_unknown.monitor", name="Monitor of Unknown Sink", isloopback=True)
        fake_sc = MagicMock()
        fake_sc.default_speaker.return_value = speaker
        fake_sc.get_microphone.return_value = exact_mon
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc), \
             patch("wune.soundcard_compat.pulseaudio_output_sample_rate", return_value=48000):
            self.assertTrue(backend.needs_restart())

    def test_uninitialized_instance_returns_false(self):
        pa = PulseAudioLoopbackBackend.__new__(PulseAudioLoopbackBackend)
        self.assertFalse(pa.needs_restart())


class MacOsNeedsRestartTests(unittest.TestCase):
    def test_macos_capture_backend_delegates_to_inner_backend(self):
        backend = MacOsCaptureBackend.__new__(MacOsCaptureBackend)
        backend._follow_default = True
        mock_inner = MagicMock()
        mock_inner.needs_restart.return_value = True
        backend._backend = mock_inner
        self.assertTrue(backend.needs_restart())

        mock_inner.needs_restart.return_value = False
        self.assertFalse(backend.needs_restart())

    def test_macos_capture_backend_returns_false_when_pinned(self):
        backend = MacOsCaptureBackend.__new__(MacOsCaptureBackend)
        backend._follow_default = False
        mock_inner = MagicMock()
        mock_inner.needs_restart.return_value = True
        backend._backend = mock_inner
        self.assertFalse(backend.needs_restart())

    def test_macos_capture_backend_safe_on_inner_exception(self):
        backend = MacOsCaptureBackend.__new__(MacOsCaptureBackend)
        backend._follow_default = True
        mock_inner = MagicMock()
        mock_inner.needs_restart.side_effect = RuntimeError("CoreAudio error")
        backend._backend = mock_inner
        self.assertFalse(backend.needs_restart())

    def test_core_audio_tap_calls_native_dylib_checker(self):
        from wune.tap_macos import CoreAudioTapBackend

        tap = CoreAudioTapBackend.__new__(CoreAudioTapBackend)
        tap._follow_default = True
        tap._closed = False
        tap._handle = 0x1234
        mock_lib = MagicMock()
        mock_lib.wune_tap_needs_restart.return_value = True
        tap._lib = mock_lib

        self.assertTrue(tap.needs_restart())
        mock_lib.wune_tap_needs_restart.assert_called_once_with(0x1234)

        mock_lib.wune_tap_needs_restart.return_value = False
        self.assertFalse(tap.needs_restart())

    def test_core_audio_tap_returns_false_when_pinned(self):
        from wune.tap_macos import CoreAudioTapBackend

        tap = CoreAudioTapBackend.__new__(CoreAudioTapBackend)
        tap._follow_default = False
        tap._closed = False
        tap._handle = 0x1234
        mock_lib = MagicMock()
        mock_lib.wune_tap_needs_restart.return_value = True
        tap._lib = mock_lib
        self.assertFalse(tap.needs_restart())

    def test_core_audio_tap_fallback_detects_device_change(self):
        from wune.tap_macos import CoreAudioTapBackend

        tap = CoreAudioTapBackend.__new__(CoreAudioTapBackend)
        tap._follow_default = True
        tap._follow_sample_rate = True
        tap._closed = False
        tap._handle = 0x1234
        tap._sample_rate = 48000
        tap._device_id = "71"
        tap._device_name = "MacBook Air Speakers"
        tap._lib = MagicMock(spec=[])  # No wune_tap_needs_restart attribute

        fake_speaker = SimpleNamespace(id="85", name="AirPods")
        fake_sc = SimpleNamespace(default_speaker=lambda: fake_speaker)
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc), \
             patch("wune.soundcard_compat.output_sample_rate", return_value=48000):
            self.assertTrue(tap.needs_restart())

    def test_core_audio_tap_fallback_detects_sample_rate_change(self):
        from wune.tap_macos import CoreAudioTapBackend

        tap = CoreAudioTapBackend.__new__(CoreAudioTapBackend)
        tap._follow_default = True
        tap._follow_sample_rate = True
        tap._closed = False
        tap._handle = 0x1234
        tap._sample_rate = 48000
        tap._device_id = "71"
        tap._device_name = "MacBook Air Speakers"
        tap._lib = MagicMock(spec=[])

        fake_speaker = SimpleNamespace(id="71", name="MacBook Air Speakers")
        fake_sc = SimpleNamespace(default_speaker=lambda: fake_speaker)
        with patch("wune.soundcard_compat.get_soundcard", return_value=fake_sc), \
             patch("wune.soundcard_compat.output_sample_rate", return_value=96000):
            self.assertTrue(tap.needs_restart())

    def test_soundcard_backend_detects_device_and_rate_changes(self):
        from wune.capture_macos import SoundCardCaptureBackend

        backend = SoundCardCaptureBackend.__new__(SoundCardCaptureBackend)
        backend._follow_default = True
        backend._follow_sample_rate = True
        backend._device_id = "1"
        backend._device_name = "Built-in"
        backend._sample_rate = 44100

        # Unchanged
        same_sp = SimpleNamespace(id="1", name="Built-in")
        with patch("wune.soundcard_compat.get_soundcard", return_value=SimpleNamespace(default_speaker=lambda: same_sp)), \
             patch("wune.soundcard_compat.output_sample_rate", return_value=44100):
            self.assertFalse(backend.needs_restart())

        # ID changed
        diff_sp = SimpleNamespace(id="2", name="Headphones")
        with patch("wune.soundcard_compat.get_soundcard", return_value=SimpleNamespace(default_speaker=lambda: diff_sp)), \
             patch("wune.soundcard_compat.output_sample_rate", return_value=44100):
            self.assertTrue(backend.needs_restart())

        # Rate changed
        with patch("wune.soundcard_compat.get_soundcard", return_value=SimpleNamespace(default_speaker=lambda: same_sp)), \
             patch("wune.soundcard_compat.output_sample_rate", return_value=48000):
            self.assertTrue(backend.needs_restart())


class SpectrumRestartDetectionTests(unittest.TestCase):
    def _make_fake_backend(self, sr=48000, name="Speakers", needs_restart=False, fail_record=False):
        backend = MagicMock()
        backend.sample_rate = sr
        backend.channels = 2
        backend.device_name = name
        backend.needs_restart.return_value = needs_restart
        if fail_record:
            backend.record.side_effect = RuntimeError("Device disconnected")
        else:
            backend.record.return_value = np.zeros((256, 2), dtype=np.float32)
        return backend

    def test_stream_error_triggers_needs_capture_restart_and_clears_after_restart(self):
        cfg = Config(block_size=256)
        b1 = self._make_fake_backend(sr=48000, name="Old Speakers", fail_record=True)
        b2 = self._make_fake_backend(sr=96000, name="USB DAC", fail_record=False)
        with patch("wune.spectrum_audio.create_capture_backend", side_effect=[b1, b2]):
            spec = AudioSpectrum(cfg, bars=16, channels=2, blocksize=256)
            self.assertFalse(spec.needs_capture_restart())
            with self.assertWarns(RuntimeWarning):
                spec.step(1 / 60)
            self.assertTrue(spec.needs_capture_restart())

            restarted = spec.restart_capture()
            self.assertTrue(restarted)
            self.assertFalse(spec.needs_capture_restart())
            self.assertEqual(spec.device, "USB DAC")
            self.assertEqual(spec.sr, 96000)

    def test_backend_needs_restart_propagates(self):
        cfg = Config(block_size=256)
        b1 = self._make_fake_backend(sr=48000, name="Speakers", needs_restart=True)
        with patch("wune.spectrum_audio.create_capture_backend", return_value=b1):
            spec = AudioSpectrum(cfg, bars=16, channels=2, blocksize=256)
            self.assertTrue(spec.needs_capture_restart())


class AppReloadAndMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.font.init()

    def test_menu_contains_reload_audio_in_windowed_and_fullscreen(self):
        menu = ApplicationMenu()
        for fs in (False, True):
            items = menu.items("en", fs)
            reload_items = [item for item in items if item[0] == "reload_audio"]
            self.assertEqual(len(reload_items), 1)
            self.assertEqual(reload_items[0][2], "F5")

    def test_f5_triggers_reload_audio_capture(self):
        app = App.__new__(App)
        app.cfg = Config()
        app._fullscreen = False
        app.screen = MagicMock()
        app.screen.get_size.return_value = (1280, 800)
        app.renderer = MagicMock()
        app.renderer.font_menu = pg.font.SysFont(None, 16)
        app.menu = ApplicationMenu()
        app.exit_confirmation = MagicMock(active=False)
        app.settings_dialog = None
        app.reload_audio_capture = MagicMock()

        event = SimpleNamespace(type=pg.KEYDOWN, key=pg.K_F5, mod=0, repeat=False)
        app.handle_event(event)
        app.reload_audio_capture.assert_called_once()

    def test_reload_audio_capture_updates_range_and_info_text(self):
        app = App.__new__(App)
        app.cfg = Config(min_freq_hz=20.0, max_freq_hz=24000.0, limit_to_20khz=False)
        app._requested_max_freq_hz = 48000.0
        app.levels = np.ones((2, 16), dtype=np.float32)
        app.renderer = MagicMock()
        app.update_info_text = MagicMock()

        spectrum = MagicMock()
        spectrum.sr = 192000
        spectrum.fmax = 23999.0

        def set_range(fmin, fmax):
            spectrum.fmax = fmax

        spectrum.set_range.side_effect = set_range
        app.spectrum = spectrum

        app.reload_audio_capture()

        expected_max = app.cfg.spectrum_upper_hz(192000, requested_max_hz=48000.0)
        spectrum.restart_capture.assert_called_once()
        spectrum.set_range.assert_called_once_with(20.0, expected_max)
        self.assertEqual(app.cfg.max_freq_hz, expected_max)
        self.assertFalse(np.any(app.levels))
        app.renderer.reset_peaks.assert_called_once()
        self.assertTrue(app._redraw_requested)
        self.assertEqual(app._idle_frames, 0)
        app.update_info_text.assert_called_once()


if __name__ == "__main__":
    unittest.main()
