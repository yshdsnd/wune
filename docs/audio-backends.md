# Audio capture backends

`AudioSpectrum` consumes PCM through `CaptureBackend`: `record(numframes)`,
`sample_rate`, `channels`, `device_name`, and `close()`. FFT calibration,
band allocation, silence gating and level envelopes remain shared across all platforms.
The backend's actual sample rate determines the FFT frequency axis.
Passing a backend into `AudioSpectrum` transfers its lifetime to the spectrum:
initialization failure and normal shutdown both close it.

- **Windows**: WASAPI shared stereo loopback, using the selected render endpoint
  and its mix rate unless an explicit rate is configured. The SoundCard
  compatibility workaround (`wune/soundcard_compat.py`) resolves PROPVARIANT
  allocation and memory leaks, permits non-negative COM codes such as `S_FALSE`
  (0x00000001) on `IAudioClient::Stop` to prevent crashes when sleeping/suspending,
  and safely reinitializes the capture stream upon system resume so new playback
  attaches immediately.
- **macOS 14.2+**: Core Audio Process Tap captures system playback by default.
  If unavailable, an installed virtual loopback input may be used. Failure
  never silently selects a physical microphone. An input can be selected
  explicitly with the device configuration.
  The native tap reports the output's actual rate; it does not resample to
  `Config.sample_rate`.
- **Linux / Unix**: PulseAudio / PipeWire monitor loopback capture (`wune/capture_linux.py`).
  Automatically identifies the default audio sink's monitor source via `libpulse-simple`
  C API or CLI fallback (`parec`, `pw-record`). It captures uncompressed stereo PCM
  at the output's native sample rate without requiring virtual audio cables.

## macOS helper

The Apple Silicon helper source is `wune/tap_backend.m`. For source execution,
`tap_macos.py` compiles a missing `libwune_tap.dylib` using `/usr/bin/clang` and
the macOS SDK (Xcode Command Line Tools required).
Frozen applications bundle the helper inside `Contents/Frameworks/libwune_tap.dylib`
and perform ad-hoc code signing.

The macOS audio workflow compiles the helper and tests capture routing,
buffer accumulation, fallback policy and cleanup with mocked audio devices.
It does not validate playback capture or grant recording permissions on headless CI.
An optional hardware test can be run on a Mac with:

```sh
WUNE_TEST_LIVE_AUDIO=1 python -m unittest discover -s tests -p test_tap_macos.py
```

Real-device verification includes permission prompt handling, silence/playback transitions,
44.1/48/96 kHz outputs, stereo separation, and clean shutdown.

## Linux audio capture

Linux capture uses `wune/capture_linux.py`, which supports both native PulseAudio
servers and PipeWire implementations with `pipewire-pulse`:
- **Native C library binding**: Uses `ctypes` to link `libpulse-simple.so.0` and
  `libpulse.so.0` for minimal latency and low CPU overhead.
- **Subprocess fallback**: If `libpulse-simple` is unavailable, it falls back to
  streaming PCM frames from `parec` or `pw-record`.
- **Automatic monitor detection**: Queries the default audio sink and connects to its
  `.monitor` source. If audio playback routes to a different card or device, Wune
  should be restarted to track the new default device.
