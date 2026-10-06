import unittest
from unittest.mock import Mock, patch

from wune.config import Config
from wune.spectrum_audio import AudioSpectrum
from wune.capture_macos import open_macos_capture_backend
from wune.tap_macos import _ensure_dylib


class CaptureLifecycleTests(unittest.TestCase):
    def backend(self):
        return Mock(sample_rate=48000, channels=2, device_name='test')

    def test_analysis_failure_closes_owned_capture_once(self):
        backend = self.backend()
        with patch.object(AudioSpectrum, '_rebuild_bins', side_effect=RuntimeError('bins')):
            with self.assertRaisesRegex(RuntimeError, 'bins'):
                AudioSpectrum(Config(), 64, capture_backend=backend)
        backend.close.assert_called_once()

    def test_bad_backend_metadata_is_released(self):
        for key in ('sample_rate', 'channels'):
            backend = self.backend()
            setattr(backend, key, 0)
            with self.assertRaises(ValueError):
                AudioSpectrum(Config(), 64, capture_backend=backend)
            backend.close.assert_called_once()

    def test_explicit_rate_reaches_capture_without_mutating_config(self):
        cfg = Config(sample_rate=48000)
        backend = self.backend()
        backend.sample_rate = 44100
        with patch('wune.capture.sys.platform', 'win32'), \
             patch('wune.capture.WasapiLoopbackBackend', return_value=backend) as factory:
            spectrum = AudioSpectrum(cfg, 64, samplerate=44100, blocksize=2048)
        self.assertEqual(factory.call_args.args[0].sample_rate, 44100)
        self.assertEqual(factory.call_args.kwargs['blocksize'], 2048)
        self.assertEqual(spectrum.sr, 44100)
        self.assertEqual(spectrum.freqs[-1], 22050)
        self.assertEqual(cfg.sample_rate, 48000)
        spectrum.close()
        spectrum.close()
        backend.close.assert_called_once()

    def test_actual_backend_rate_drives_fft(self):
        backend = self.backend()
        backend.sample_rate = 96000
        spectrum = AudioSpectrum(Config(sample_rate=44100), 64, capture_backend=backend)
        self.addCleanup(spectrum.close)
        self.assertEqual(spectrum.freqs[-1], 48000)

    def test_unknown_explicit_mac_device_does_not_use_defaults(self):
        with patch('soundcard.get_speaker', side_effect=IndexError), \
             patch('soundcard.get_microphone', side_effect=IndexError), \
             patch('soundcard.default_microphone') as default:
            with self.assertRaisesRegex(RuntimeError, 'Selected macOS'):
                open_macos_capture_backend(Config(output_device='missing'))
            default.assert_not_called()

    def test_mac_open_failure_releases_registered_resources(self):
        close = Mock()
        def fail(**kwargs):
            kwargs['exit_stack'].callback(close)
            raise RuntimeError('open failed')
        with patch('wune.capture_macos.select_macos_output', return_value=Mock()), \
             patch('wune.capture_macos.open_macos_capture', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'open failed'):
                open_macos_capture_backend(Config(output_device='input', sample_rate=48000))
        close.assert_called_once()

    def test_missing_dylib_builds_to_defined_output_path(self):
        built = set()
        def exists(path):
            return path.endswith('tap_backend.m') or path in built
        def compile(cmd, **kwargs):
            built.add(cmd[cmd.index('-o') + 1])
        with patch('wune.tap_macos.os.path.exists', side_effect=exists), \
             patch('wune.tap_macos.subprocess.run', side_effect=compile) as run, \
             patch('wune.tap_macos.sys.frozen', False, create=True):
            self.assertTrue(_ensure_dylib().endswith('libwune_tap.dylib'))
        run.assert_called_once()
        self.assertIn('arm64', run.call_args.args[0])

    def test_frozen_bundle_never_compiles_missing_library(self):
        with patch('wune.tap_macos.os.path.exists', return_value=False), \
             patch('wune.tap_macos.subprocess.run') as run, \
             patch('wune.tap_macos.sys.frozen', True, create=True):
            self.assertIsNone(_ensure_dylib())
            run.assert_not_called()
