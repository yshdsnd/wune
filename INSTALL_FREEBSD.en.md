# FreeBSD Installation and User Guide for Wune

**English** / [日本語](INSTALL_FREEBSD.md)

Installation and usage guide for **Wune**, the 80s-90s mini-hifi-inspired LED spectrum analyzer, on FreeBSD.

## System Requirements

- **Supported OS & Architecture**: **FreeBSD 14.x / 15.x (amd64)** (Tested on FreeBSD 15.1-STABLE)
- **Audio Server**: **PulseAudio** or **PipeWire** (PulseAudio compatibility layer)
- **Display**: **X11** or **Wayland**
- **Audio Output**: Any stereo playback device (speakers, headphones, etc.)
- **Recommended**: Japanese fonts (Google Noto Fonts), `playerctl` (for Now Playing track titles)

## Installation Steps

### 1. Install System Dependencies

Use FreeBSD `pkg` to install required runtimes, libraries, and fonts:

```bash
sudo pkg install py312-tkinter py312-cffi py312-numpy pulseaudio playerctl \
    v4l_compat noto-basic noto-jp dejavu pkgconf sdl2 sdl2_image sdl2_mixer sdl2_ttf
```

> [!NOTE]
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

## Now Playing Track Titles

Wune integrates with MPRIS2 via `playerctl` to display playback titles and artists from Spotify, browsers, and media players.

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
