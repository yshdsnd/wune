# Audio capture backends

`AudioSpectrum` consumes PCM through `CaptureBackend`: `record(numframes)`,
`sample_rate`, `channels`, `device_name`, and `close()`. FFT calibration,
band allocation, silence gating and level envelopes remain shared.
The backend's actual sample rate determines the FFT frequency axis.
Passing a backend into `AudioSpectrum` transfers its lifetime to the spectrum:
initialization failure and normal shutdown both close it.

- Windows: WASAPI shared stereo loopback, using the selected render endpoint
  and its mix rate unless an explicit rate is configured. The existing
  SoundCard compatibility workaround remains active.
- macOS 14.2+: Core Audio Process Tap captures system playback by default.
  If unavailable, an installed virtual loopback input may be used. Failure
  never silently selects a physical microphone. An input can be selected
  explicitly with the existing device configuration.
- The native tap reports the output's actual rate; it does not resample to
  `Config.sample_rate`. Explicit SoundCard input capture uses the configured
  rate, or its existing default when no rate is supplied.

## macOS helper

The Apple Silicon helper source is `wune/tap_backend.m`. For source execution,
`tap_macos.py` compiles a missing `libwune_tap.dylib` using `/usr/bin/clang` and
the macOS SDK (Xcode Command Line Tools required). No prebuilt binary is added
by this integration. Frozen applications must bundle the helper: they never
attempt compilation into the signed application. Bundling and the remaining
macOS window/settings-process integration are later stages of Issue #98.

The macOS audio workflow compiles the helper and tests capture routing,
buffer accumulation, fallback policy and cleanup with mocked audio devices.
It does not validate playback capture or grant recording permissions.
An optional hardware test can be run on a Mac with:

```sh
WUNE_TEST_LIVE_AUDIO=1 python -m unittest discover -s tests -p test_tap_macos.py
```

Real-device checks still include permission denial, silence/playback,
44.1/48/96 kHz outputs, stereo input and shutdown. Full application validation
on macOS also requires the subsequent settings/window integration.
