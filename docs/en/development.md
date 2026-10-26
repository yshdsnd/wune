# Development and Running from Source

**English** / [日本語](../development.md)

[User README](../../README.en.md) / [Packaging Guide](packaging.md)

## Environment and Launching

Wune supports Python 3.10+ on Windows x64, macOS Apple Silicon (arm64), Linux x64 (Fedora, Ubuntu, etc.), and FreeBSD (amd64). A Python environment that includes Tkinter is required.
End users using prebuilt binary packages do not need this setup.
Obtain the source from the [GitHub repository](https://github.com/yshdsnd/wune).

### Launching on Windows
~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
~~~

### Launching on Linux (Fedora / Ubuntu etc.)
~~~bash
# Fedora / RHEL dependencies
sudo dnf install -y python3 python3-devel python3-tkinter pulseaudio-libs \
    sdl2-compat-devel SDL2_image-devel SDL2_ttf-devel freetype-devel playerctl

# Ubuntu / Debian dependencies
sudo apt install -y python3 python3-dev python3-pip python3-venv python3-tk \
    libpulse0 libpulse-dev libasound2-dev libsdl2-dev libsdl2-image-dev \
    libsdl2-ttf-dev libfreetype6-dev playerctl

# Virtual environment and launching
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
~~~

### Launching on macOS
~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
~~~

### Launching on FreeBSD
See the [FreeBSD Installation Guide](../../INSTALL_FREEBSD.en.md) for dependencies, venv setup, and startup steps on FreeBSD.

Refer to `requirements-build.txt` for pinned dependencies used during release packaging.
Built-in configuration defaults reside in `Config` inside `wune/config.py`; test values can be specified in `CFG` at the bottom of `config.py` during development.
Saved user settings take precedence over default values. To reset settings on launch:
**This deletes existing user themes and saved window geometry. Back up settings.json if needed.**

~~~powershell
.\.venv\Scripts\python.exe main.py --reset-settings
~~~

Launching with `--reset-settings` ignores custom `CFG` and restores `Config()` defaults.
Normal source launches share the same per-user configuration as packaged distribution builds.
Unlike `Wune.log` in packaged builds, errors during source execution are logged to standard console output.

### Microsoft Store Python Configuration Path

Windows may virtualize local configuration files to the following path when using Microsoft Store Python:

~~~text
%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\Local\Wune\settings.json
~~~

The `Python.3.13` part depends on your installed Python version.

## Audio Input

- `output_device = None`: Default OS audio output device at launch.
- `sample_rate = None`: Acquires the mix rate of the shared mode audio output device at launch.
- `block_size = 4096`: Read / FFT buffer size.
- `channels = 2`, `bars = 64`: Default stereo display and frequency bands.

Wune does not inspect file bitrates or DAC internal clock rates.
Device or sample rate changes while Wune is running require restarting the app.
If capture initialization fails, Wune exits with an error without silently falling back to a microphone or synthetic audio.
Windows uses WASAPI loopback via SoundCard, macOS uses Core Audio Process Tap, Linux uses PulseAudio/PipeWire monitor capture, and FreeBSD uses SoundCard's PulseAudio backend.

To test different output endpoints, list available devices with:

~~~powershell
.\.venv\Scripts\python.exe -c "from wune.soundcard_compat import prepare_soundcard; prepare_soundcard(); import soundcard as sc; [print(s.id, s.name) for s in sc.all_speakers()]"
~~~

Playback apps must route output through the same target device.

## Visual and Analysis Settings

See the [README](../../README.en.md) for settings configurable via the graphical settings window.
In windowed mode, identity information is consolidated into the window title bar, with no watermark brand label inside the spectrum area.
The top preset badge can be clicked to cycle visual presets.
Example `CFG` development override:

~~~python
CFG = Config(
    spectrum_orientation="frequency_vertical",
    channel_layout="horizontal",
    bars=32,
    width=960,
    height=800,
)
~~~

- `spectrum_orientation`: `frequency_horizontal` (frequencies across X, levels on Y) or `frequency_vertical` (frequencies across Y, levels on X).
- `channel_layout`: `vertical` (L/R stacked) or `horizontal` (L/R side by side).
- `led_shape`: `rectangle` / `rounded` / `ellipse`. `led_aspect_ratio`: width / height (0.25 to 8.0).
- `gauge_style`: `flat` / `box`.
- `initial_preset`: `CLASSIC` / `BLUE` / `AMBER`, or `None` to supply custom `Theme(...)`.

### Frequency Range and Level Calibration

Outputs with sample rates $\le 48\text{ kHz}$ display up to 20 kHz; higher rates display up to 40 kHz.
`limit_to_20khz=True` restricts the upper bound to 20 kHz.
Band power is computed by summing FFT bins across Hann-windowed blocks and converted to dB.
A 1.0 peak sine wave corresponds to 0 dB, mapping `db_min` to `db_max` (default $-66$ to $0\text{ dB}$) across the meter scale.

| Ballistics Setting | Range | Default |
| --- | --- | --- |
| `vis_attack_ms` | 1–1000 ms | 5 ms |
| `vis_release_ms` | 1–5000 ms | 120 ms |
| `peak_hold_ms` | 0–5000 ms | 500 ms |
| `peak_fall_per_second` | 0–20 full spans/sec | 2.5 |

## Settings File

Saved in version 2 JSON format containing `window`, `appearance`, and `user_themes`.
Keys correspond directly to `Config` attributes.

~~~json
{
  "version": 2,
  "window": {"width": 916, "height": 504, "x": 100, "y": 100},
  "appearance": {
    "language": "en",
    "initial_preset": "BLUE",
    "spectrum_orientation": "frequency_vertical",
    "channel_layout": "horizontal",
    "bars": 32
  }
}
~~~

## Localization and Icons

Translations are managed in `wune/locales/en.json` and `ja.json`.
Missing keys fall back to English.
Icon origins, generation prompts, and asset tooling are documented in [`wune/assets/README.md`](https://github.com/yshdsnd/wune/blob/main/wune/assets/README.md).

## Documentation Updates

When modifying user-facing functionality, update `README.md` (Japanese) and `README.en.md` (English) in tandem.

## Running Tests

Execute from the repository root:

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

## macOS Development

Requires Apple Silicon, Python 3.10+ with Tkinter, and Xcode Command Line Tools:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

Settings are stored in `~/Library/Application Support/Wune/settings.json`.
Settings dialogs run in spawned child processes across all platforms.

## Linux Development

Requires Linux x64 (Fedora, Ubuntu, etc.) with Tkinter:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

Captures audio via PulseAudio / PipeWire monitor sources. Media metadata is queried via MPRIS2 (`playerctl`).
Settings are stored in `~/.config/Wune/settings.json`, and logs in `~/.config/Wune/Wune.log`.

## FreeBSD Development

Requires FreeBSD 14.x / 15.x (amd64) with Tkinter:

```bash
# Install dependencies
sudo pkg install -y py312-tkinter py312-cffi py312-numpy pulseaudio playerctl \
    v4l_compat noto-basic dejavu pkgconf sdl2 sdl2_image sdl2_mixer sdl2_ttf

# Virtual environment and launching
python3.12 -m venv --system-site-packages .venv
.venv/bin/pip install SoundCard==0.4.6
CFLAGS="-I/usr/local/include" LDFLAGS="-L/usr/local/lib" .venv/bin/pip install pygame
.venv/bin/python main.py
```

See the [FreeBSD Ports / Pkg Guide](../../packaging/freebsd/README.md) for ports and package generation.
Settings are stored in `~/.config/Wune/settings.json`, and logs in `~/.config/Wune/Wune.log`.

The active development branch is `main` (Next release: v1.3.0). See the [Integration Log](integration-v1.3.0.md).
