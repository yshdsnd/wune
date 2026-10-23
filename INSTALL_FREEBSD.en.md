# FreeBSD Installation and User Guide for Wune

**English** / [日本語](INSTALL_FREEBSD.md)

Installation and usage guide for **Wune**, the 80s-90s mini-hifi-inspired LED spectrum analyzer, on FreeBSD.

## System Requirements

- **Supported OS & Architecture**: **FreeBSD 14.x / 15.x (amd64)** (Verified on physical FreeBSD 15.1-STABLE)
  *(Hardware test setup: i915kms + modesetting Xorg, Openbox, Firefox + PulseAudio, initial port on `devel/freebsd` branch)*
- **Audio Server**: **PulseAudio** or **PipeWire** (PulseAudio compatibility layer)
- **Display**: **X11** or **Wayland**
- **Audio Output**: Any stereo playback device (speakers, headphones, etc.)
- **Recommended**: Japanese fonts (Google Noto Fonts), `playerctl` (for Now Playing track titles), `pavucontrol` (for GUI audio routing and capture source switching)

## Installation Steps

### 1. Install System Dependencies

Use FreeBSD `pkg` to install required runtimes, libraries, fonts, and utilities:

```bash
sudo pkg install py312-tkinter py312-cffi py312-numpy pulseaudio pavucontrol playerctl \
    v4l_compat noto-basic noto-jp dejavu pkgconf sdl2 sdl2_image sdl2_mixer sdl2_ttf
```

> [!NOTE]
> - `pavucontrol` provides a convenient GUI to manage playback and recording streams (switching Wune's capture source).
> - `v4l_compat` provides Video4Linux2 compatibility headers required for building Pygame.
> - `noto-basic`, `noto-jp`, and `dejavu` ensure crisp font rendering for the UI and Japanese audio device names.

### 2. Clone Repository and Create Virtual Environment

Create a Python virtual environment with `--system-site-packages` to leverage system packages (NumPy, CFFI, Tkinter):

```bash
git clone https://github.com/yshdsnd/wune.git
cd wune
python3.12 -m venv --system-site-packages .venv
```

### 3. Install Python Dependencies

```bash
# Install SoundCard
.venv/bin/pip install SoundCard==0.4.6

# Build and install Pygame with FreeBSD include/library paths
CFLAGS="-I/usr/local/include" LDFLAGS="-L/usr/local/lib" .venv/bin/pip install pygame
```

### 4. Launch

```bash
.venv/bin/python main.py
```

## Audio Capture

- On launch, Wune detects your default PulseAudio / PipeWire playback sink and automatically captures audio from its **loopback monitor source** (`<device>.monitor`).
- Wune never outputs audio or modifies your playback route. Play audio using your favorite media player or browser.
- **Default Sample Rate**: The default capture rate is **48 kHz**.
- Press the `I` key to toggle the info panel displaying active device name, sampling rate, and channels.

### Configuring Default Output and Monitor Source (pactl)

To ensure Wune captures from a specific audio output (for example, HDMI audio `oss_output.dsp3`), set both the default playback sink and loopback monitor source explicitly using `pactl`:

```bash
# List available sinks (playback) and sources (recording/monitor)
pactl list short sinks
pactl list short sources

# Set default sink and monitor source (example: oss_output.dsp3)
pactl set-default-sink oss_output.dsp3
pactl set-default-source oss_output.dsp3.monitor
```

> [!IMPORTANT]
> - In some FreeBSD configurations, setting `set-default-source` alone may not select the intended monitor. Always configure **both** `set-default-sink` and `set-default-source`.
> - Setting both ensures that Wune will reliably select the intended HDMI/speaker monitor stream even after quitting and restarting Wune.

### Switching Capture Source at Runtime (pavucontrol)

You can change Wune's capture source in real time using the `pavucontrol` GUI:

1. Launch `pavucontrol`.
2. Go to the **Recording** tab.
3. Find **Wune** in the application list and select the desired monitor (e.g., `Monitor of oss_output.dsp3`) from the dropdown.

> [!NOTE]
> When switching capture sources via `pavucontrol`, the spectrum analyzer visualization updates immediately, but the monitor device name displayed in Wune's info panel (`I` key) will not dynamically update (planned for future improvement).

### Current Verification Status and Scope

At present, hardware testing has verified PulseAudio startup, stream capture, and spectrum animation under the specified setup.
Preserving configuration across OS reboots, direct OSS (`/dev/dsp`), `virtual_oss`, PipeWire, settings persistence, and resizing/fullscreen stability remain subjects for ongoing verification.

## Now Playing Track Titles

Wune integrates with MPRIS2 via `playerctl` to display playback titles and artists from Spotify, browsers, and media players.
Installing `playerctl` as described above enables automatic title and artist display at the top of the window.

## Keyboard Shortcuts and Controls

Click the window to focus before using keyboard shortcuts.

| Shortcut | Description |
| --- | --- |
| F2 | Open Settings dialog |
| Top-left Menu or Right-Click | Settings, Fullscreen toggle, Quit |
| Alt+Enter / F11 | Toggle Fullscreen |
| Space | Pause / Resume spectrum display |
| T (or click theme badge) | Cycle visual themes |
| I | Toggle audio stream info bar |
| Esc | Exit fullscreen, or show exit confirmation when windowed |
| Q | Show exit confirmation dialog |
| Window Drag | Resize window dynamically with smooth LED scaling |

## Settings File Location

Settings are saved automatically per user:

- **Location**: `~/.config/Wune/settings.json`
- To reset settings to defaults, launch with: `main.py --reset-settings`
