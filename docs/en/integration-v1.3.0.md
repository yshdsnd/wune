# v1.3.0 Main Integration Log

**English** / [日本語](../integration-v1.3.0.md)

Record of changes and integrations in the `main` branch toward the minor update v1.3.0.
The major highlights are native support for FreeBSD 14.x / 15.x (Issue #77), building the FreeBSD Ports / pkg binary packaging infrastructure, full CI automation via FreeBSD 15.1-RELEASE VM on GitHub Actions, and improved playback state detection for Sony Music Center for PC on Windows (Issue #190).

| Feature | Integration Details | Related Issue / PR |
| --- | --- | --- |
| Music Center for PC Playback Detection | Directly monitors Windows Core Audio (`IAudioSessionControl2`) to determine playback/pause state accurately based on active audio output sessions | #190, PR #191 |
| FreeBSD Platform Foundation & SoundCard Compatibility | Implemented platform spoofing and `sys.argv` guard in `wune/soundcard_compat.py` to enable SoundCard 0.4.6's PulseAudio backend on FreeBSD | #77, PR #189, PR #193 |
| FreeBSD Hardware Testing & Setup Guide | Verified on FreeBSD 15.1-STABLE (Xorg/Openbox/PulseAudio), authored bilingual setup guides (`INSTALL_FREEBSD.md`, `INSTALL_FREEBSD.en.md`), and documented `pactl`/`pavucontrol` routing | #77, PR #189, PR #192, PR #194 |
| FreeBSD HDMI Audio Troubleshooting | Documented root causes and recovery steps for HDMI audio disappearance upon display sleep/standby (`snd_hda` / `i915kms` power well issues) | PR #195 |
| FreeBSD Ports Skeleton | Created complete Ports definitions under `packaging/freebsd/` for `audio/py-soundcard` and `audio/wune`, supporting hybrid direct checkout packaging and non-root `make package` execution | PR #195, PR #196 |
| Python Packaging Standardization | Configured `pyproject.toml` and `setup.py` defining the `wune` CLI entry point, desktop entry (`wune.desktop`), icons, and title bar version identification on pkg installation | PR #195 |
| Documentation Installation to DOCSDIR | Configured ports to install README and FreeBSD installation guides (Japanese and English) to `/usr/local/share/doc/wune-<version>/` | PR #195 |
| FreeBSD 15.1 VM CI Workflow | Created `.github/workflows/freebsd-package.yml` using `vmactions/freebsd-vm@v1` on FreeBSD 15.1-RELEASE VM to automate testing, package builds, and validation on every push/PR | PR #197 |
| Headless Audio Daemon & Test Resilience | Configured headless PulseAudio daemon (UNIX socket, anonymous auth, null sink `wune_dummy`) inside FreeBSD VM; added `soundcard` fallback stub for headless test discovery | PR #197 |
| Release Artifact Generation | Added SHA-256 checksum generation and upload for `Wune-freebsd-pkg` artifacts, with automatic attachment to GitHub Releases on version tags (`v*`) | PR #197 |
| Bilingual Documentation Reorganization | Organized Japanese documentation under `docs/` and English documentation under `docs/en/`, translating missing counterparts to ensure complete bilingual coverage | PR #198 |

## Key Integration Commits

| Commit | Description |
| --- | --- |
| `30aac56` | feat: add FreeBSD support and documentation (#77) |
| `d56129d` | fix(now-playing): determine Music Center playback state via Core Audio session (#190) |
| `efe9179` | docs(freebsd): document audio sink/source configuration and pavucontrol routing (#77) |
| `04986de` | docs: note verified FreeBSD features (resize, settings save, dialog layout) (#77) |
| `1f8f337` | docs: clean up branch references and align documentation across platforms (#77) |
| `6047977` | feat(freebsd): provide FreeBSD ports skeleton and python package setup (#77) |
| `90e6c0b` | feat(freebsd): enable hybrid ports build with local checkout and git snapshot |
| `0633586` | fix(freebsd): align ports distfile naming and non-root package build |
| `5f4fa5d` | feat(freebsd): show package version in title bar when installed via pkg |
| `296b921` | docs(freebsd): add troubleshooting tips for HDMI audio and power well recovery |
| `cf6c11c` | feat(freebsd): install README and FreeBSD install guides to DOCSDIR |
| `dfbc837` | fix(freebsd): refine packaging docs and remove optional pavucontrol dependency |
| `ff6d377` | ci(freebsd): add FreeBSD 15.1 VM packaging workflow and tests |
| `4738d7c` | fix(ci): install fonts and build packages before running unit tests in FreeBSD VM |
| `4113ae1` | fix(ci): launch PulseAudio daemon with null sink and stub soundcard in headless tests |

## File Group Verification

- **wune/now_playing_music_center.py**: Traverses Windows Core Audio's `IAudioSessionManager2` / `IAudioSessionEnumerator` / `IAudioSessionControl2` to accurately determine Music Center's output state (`AudioSessionStateActive`).
- **wune/soundcard_compat.py**: Implements `sys.platform = "linux"` spoofing and `sys.argv` guard on FreeBSD, and provides fallback mock stubs when audio daemons are unavailable.
- **wune/capture.py / capture_macos.py**: Unified direct `soundcard` imports to go through `get_soundcard()`.
- **wune/build_identity.py**: Reads package versions (e.g. `1.3.0`) via `pkg info` or `build-info.json` to display `Wune v1.3.0` in the title bar.
- **pyproject.toml / setup.py**: Standard packaging metadata for FreeBSD and definition of `wune = wune.cli:main` entry point.
- **packaging/freebsd/**:
  - `ports/audio/wune/Makefile`: Wune port definition with options (DOCS, NLS, ALSA, JACK, PULSEAUDIO, PIPEWIRE) and desktop/doc placement.
  - `ports/audio/py-soundcard/Makefile`: Port definition for py-soundcard on FreeBSD.
  - `Makefile`: Helper to package both ports into `packages/`.
- **.github/workflows/freebsd-package.yml**: Automated test, packaging, checksumming, and release attachment workflow on FreeBSD 15.1-RELEASE VM.
- **tests/test_freebsd_ports.py**: Static validation of port files, metadata, and CI workflow definition.
- **tests/test_now_playing_music_center.py**: Unit tests for WASAPI audio session enumeration and playback state detection.

## Pre-Release Checklist

1. Unit tests pass on all platforms (Windows 11, macOS Apple Silicon, Linux Ubuntu/Fedora, FreeBSD 15.1 VM): 536 tests passing.
2. Verify Music Center for PC playback/pause/stop state synchronization on physical Windows machine.
3. Verify package installation (`pkg add packages/*.pkg`), startup, audio capture, and exit on physical FreeBSD 15.1-STABLE machine.
4. Verify existing features on macOS and Linux (fullscreen, settings dialog, audio capture, Now Playing) without regression.
