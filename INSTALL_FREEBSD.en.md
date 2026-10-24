# FreeBSD Installation and User Guide for Wune

**English** / [日本語](INSTALL_FREEBSD.md)

Installation and usage guide for **Wune**, the 80s-90s mini-hifi-inspired LED spectrum analyzer, on FreeBSD.

## System Requirements

- **Supported OS & Architecture**: **FreeBSD 14.x / 15.x (amd64)** (Verified on physical FreeBSD 15.1-STABLE)
  *(Hardware test setup: i915kms + modesetting Xorg, Openbox, Firefox + PulseAudio)*
- **Audio Server**: **PulseAudio** or **PipeWire** (PulseAudio compatibility layer)
- **Display**: **X11** or **Wayland**
- **Audio Output**: Any stereo playback device (speakers, headphones, etc.)
- **Recommended**: Japanese fonts (Google Noto Fonts), `playerctl` (for Now Playing track titles), `pavucontrol` (for GUI audio routing and capture source switching)

## Installation Options

There are two ways to install and run Wune on FreeBSD:

1. **FreeBSD Ports / Pkg (Recommended)**
   Using the Ports skeleton included in this repository, you can build and install Wune natively as a FreeBSD port or generate `.pkg` binary packages.
   See the **[FreeBSD Ports / Pkg Guide](packaging/freebsd/README.md)** for detailed instructions.

2. **Run from Source via Virtual Environment**
   Set up a Python virtual environment leveraging system packages and run Wune directly. Follow the steps below.

## Running from Source Steps

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

On physical hardware (FreeBSD 15.1-STABLE, i915kms + modesetting Xorg, Openbox, Firefox + PulseAudio), the following have been verified:
- Startup, audio capture via PulseAudio, and spectrum analyzer visualization
- Window resizing (scaling up and down)
- Settings saving and persistence across application restarts
- Settings dialog layout and font rendering (proper window dimensions without overlapping text)

*Note: Preserving PulseAudio default settings across OS reboots, direct OSS (`/dev/dsp`), `virtual_oss`, PipeWire compatibility, and fullscreen stability remain subjects for ongoing verification.*

### Troubleshooting & Tips: Handling Audio Lockups and HDMI Sound Loss

#### Symptom: Wune visualizer responds, but no physical sound comes from speakers/HDMI
If Wune's spectrum analyzer is bouncing, it means the audio application is outputting to PulseAudio and PulseAudio's loopback monitor (`<device>.monitor`) is functioning normally.  
The issue is stalled transmission from PulseAudio to the physical hardware device (e.g., HDMI / `/dev/dsp3`).

On FreeBSD systems with Intel graphics (`i915kms`) and HDMI audio, **power to the audio circuit (Power Well) may be cut off during periods of silence or display power-saving, and subsequent attempts to resume playback fail to re-power or re-synchronize the hardware clock, causing the device to become silent until reboot**.

#### Prevention (Root Cause Mitigation & Recommended Settings)

1. **Keep Intel GPU (i915kms) audio power wells always powered**  
   Add the following line to `/boot/loader.conf` and reboot (or apply immediately in the current session via `sudo sysctl compat.linuxkpi.i915_disable_power_well=0`):
   ```text
   compat.linuxkpi.i915_disable_power_well="0"
   ```
   > [!TIP]
   > With the default value `-1` (auto), power to the HDMI audio circuit is cut during silence and may fail to restore. Setting it to `0` (power wells always on) prevents this hardware sleep lockup.

2. **Disable PulseAudio idle suspend**  
   Prevent PulseAudio from releasing and closing `/dev/dsp` when audio is paused:
   ```bash
   mkdir -p ~/.config/pulse
   cp /usr/local/etc/pulse/default.pa ~/.config/pulse/
   ```
   Comment out `load-module module-suspend-on-idle` in `~/.config/pulse/default.pa`:
   ```diff
   - load-module module-suspend-on-idle
   + # load-module module-suspend-on-idle
   ```
   Apply the change by restarting PulseAudio: `pulseaudio -k && pulseaudio --start`.

#### Recovery Steps (Restoring Sound Without an OS Reboot)

On FreeBSD, sound drivers such as `snd_hda` are statically compiled into the `GENERIC` kernel and cannot be unloaded via `kldunload`. Furthermore, restarting Xorg does not affect the kernel sound subsystem.  
To attempt recovery without rebooting the OS, use FreeBSD's `devctl(8)` utility or reconfigure the HDA codec sysctl:

```bash
# 1. Stop PulseAudio which holds the device open
pulseaudio -k

# 2. Force reset or detach/attach the PCM device with devctl
sudo devctl reset pcm3
# or
sudo devctl detach -f pcm3 && sudo devctl attach pcm3

# (or reconfigure the parent HDA codec device hdaa1)
sudo sysctl dev.hdaa.1.reconfig=1

# 3. Restart PulseAudio and restore default sink/source
pulseaudio --start
pactl set-default-sink oss_output.dsp3
pactl set-default-source oss_output.dsp3.monitor
```

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
