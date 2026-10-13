# Linux Installation Guide for Wune

**English** / [日本語](INSTALL_LINUX.md)

Installation and setup instructions for Wune, an LED audio spectrum analyzer inspired by 1980s and 1990s compact stereo systems, on Linux.

## System Requirements

- **Supported Architecture**: **x86_64 (64-bit PC)**
- **Hardware-Verified Distribution**:
  - **Fedora 44 (x86_64)** (tested and verified on physical hardware)
  - **RHEL / CentOS Stream, Ubuntu / Debian, and others**: While RPM/DEB packages and standalone archives are provided and smoke-tested in CI (Ubuntu 24.04), they have not been verified on physical hardware. We welcome users who can test Wune on these distributions—if you encounter any issues, please report them via **[GitHub Issues](https://github.com/yshdsnd/wune/issues)**.
  - *Note*: By design, standard Linux desktop environments running PulseAudio or PipeWire (`pipewire-pulseaudio`) are supported.
- **Audio Server**: **PulseAudio** or **PipeWire** (with PulseAudio compatibility layer)
- **Display Server**: **X11** or **Wayland** (via XWayland)
- Audio output device capable of stereo playback (speakers, headphones, external DAC, etc.)
- **Recommended**: Japanese fonts (e.g. Google Noto Sans CJK JP), `playerctl` (for Now Playing metadata)

## Choosing a Package and Installation

Download the appropriate package for your Linux distribution from **[GitHub Releases](https://github.com/yshdsnd/wune/releases)**.

### 1. Fedora / RHEL / CentOS Stream (RPM Package)

Download the RPM package and install it using `dnf`. Desktop menu integration and `/usr/bin/wune` command link are installed automatically.

```bash
# Install package
sudo dnf install ./Wune-vX.Y.Z-1.x86_64.rpm
```

- **Launch**: Open "Wune" from your desktop application launcher or run `wune` in your terminal.
- **Uninstall**: `sudo dnf remove wune`

### 2. Debian / Ubuntu / Linux Mint (DEB Package)

Download the DEB package and install it using `apt`.

```bash
# Install package
sudo apt install ./wune_X.Y.Z_amd64.deb
```

- **Launch**: Open "Wune" from your desktop application launcher or run `wune` in your terminal.
- **Uninstall**: `sudo apt remove wune`

### 3. Standalone Archive (.tar.gz / .zip)

A standalone archive that runs on 64-bit Linux (glibc 2.35+) with PulseAudio or PipeWire without requiring Python to be installed. Audio capture requires `pulseaudio-libs` (or `libpulse0` on Debian/Ubuntu).

```bash
# Extract tar.gz and run
tar -xzf Wune-vX.Y.Z-linux-x64.tar.gz
cd Wune
./Wune
```

To integrate with your desktop application launcher, copy the bundled `wune.desktop` file into `~/.local/share/applications/` and adjust `Exec=` and `Icon=` to the absolute path of the extracted `Wune` executable, or create a symlink in a directory on your PATH (e.g. `~/.local/bin/wune`).

## Audio Capture Notes

- At startup, Wune automatically detects the default Linux audio output device (speakers, headphones, etc.) and connects to its **loopback monitor source** (`<device>.monitor`) to capture playback audio in real time.
- Both PipeWire (`pipewire-pulseaudio`) and PulseAudio configurations are supported.
- Wune does not output sound or alter your default audio device. Play music or video in your preferred media player or web browser to see the visualization.
- **Default Sample Rate**: When unspecified, the capture sample rate defaults to **48 kHz** (max display range up to 20 kHz). To observe frequencies beyond 20 kHz on high-resolution audio sources (such as 96 kHz), specify `sample_rate` explicitly in `settings.json`.
- The information bar (toggle with `I`) displays the active device name, sample rate, and channels.
- If you change your default audio output device, restart Wune.

## Now Playing Track Information

On Linux, Wune uses the MPRIS2 D-Bus interface to fetch the title and artist of currently playing tracks from Spotify, Rhythmbox, VLC, browsers, and other players.

- **`playerctl` (Recommended)**:
  Installing `playerctl` provides the most reliable track metadata integration:
  ```bash
  # Fedora / RHEL
  sudo dnf install playerctl

  # Debian / Ubuntu
  sudo apt install playerctl
  ```
- If `playerctl` is not installed, Wune automatically falls back to system D-Bus queries via `gdbus` or `dbus-send`.

## Basic Controls and Shortcuts

| Key / Action | Function |
| --- | --- |
| S or F2 | Open Settings dialog |
| F or F11 (or Alt+Enter) | Toggle fullscreen / windowed mode |
| Space | Pause / resume visualization (audio playback continues) |
| T or click theme badge | Cycle to next color theme |
| I | Toggle info bar (output device, sample rate, channels) |
| Esc | Exit fullscreen in fullscreen mode; exit confirmation dialog in windowed mode |
| Ctrl+Q or Q | Exit confirmation dialog |
| Drag window edges | Resize window (LED grid scales smoothly) |

## Configuration and Logs

Following the XDG Base Directory specification, configuration files and logs are stored in:

- **Settings**: `~/.config/Wune/settings.json` (or `$XDG_CONFIG_HOME/Wune/settings.json`)
- **Log file**: `~/.config/Wune/Wune.log`

### Resetting Preferences
To reset all preferences to their built-in defaults:
```bash
wune --reset-settings
# Standalone archive:
./Wune --reset-settings
```

## Running from Source

Follow these steps to run Wune directly from source for development.

### 1. Install System Dependencies

#### Fedora / RHEL
```bash
# Required packages and build dependencies
sudo dnf install -y python3 python3-devel python3-tkinter pulseaudio-libs \
    sdl2-compat-devel SDL2_image-devel SDL2_ttf-devel freetype-devel

# Recommended
sudo dnf install -y playerctl google-noto-sans-cjk-fonts
```

#### Debian / Ubuntu
```bash
# Required packages and build dependencies
sudo apt update
sudo apt install -y python3 python3-dev python3-pip python3-venv python3-tk \
    libpulse0 libpulse-dev libasound2-dev libsdl2-dev libsdl2-image-dev \
    libsdl2-ttf-dev libfreetype6-dev

# Recommended
sudo apt install -y playerctl fonts-noto-cjk
```

### 2. Set Up Virtual Environment and Launch
```bash
# Clone repository
git clone https://github.com/yshdsnd/wune.git
cd wune

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python3 main.py
```

## Troubleshooting

| Issue | What to Check |
| --- | --- |
| Spectrum is static / no audio captured | Confirm that audio is actively playing and visualization is not paused (Space). Verify PipeWire/PulseAudio is running (`pactl info`). |
| Settings dialog does not open | Ensure `python3-tkinter` (`python3-tk` on Debian/Ubuntu) is installed. (Included automatically in standalone packages). |
| Japanese characters display as tofu/squares | Ensure Japanese fonts are installed (`google-noto-sans-cjk-fonts` or `fonts-noto-cjk`). |
| Track info not displayed | Check if `playerctl` is installed, and verify if your player supports MPRIS2 (`playerctl status`). |

For bug reports and feature proposals, please visit **[GitHub Issues](https://github.com/yshdsnd/wune/issues)**.
