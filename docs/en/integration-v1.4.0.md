# v1.4.0 Main Integration Log

**English** / [日本語](../integration-v1.4.0.md)

Integration and change log for the `main` branch ahead of minor release v1.4.0.
Key highlights include unifying the header into a single row with listening-mode styling (PR #205), optional borderless window mode (PR #206), version display in Settings and the output info bar alongside automatic default output tracking and manual audio reload across all platforms (PR #207), single-click focus clickthrough on unfocused windows (PR #209), and fixes for fullscreen settings cancel, auto-LED shrinking limits, and overlapping dB scale labels (PR #211).

| Feature | Summary of Changes | Related Issue / PR |
| --- | --- | --- |
| Unified Single-Row Header & Listening Mode | Replaced the text menu button with a three-line hamburger icon, unified left-aligned Now Playing text and a fixed-width centered theme badge (with ellipsis truncation) into a single top row, and dimmed header borders/text when not hovered | PR #205 |
| Borderless Window Mode | Added optional `borderless` mode (Settings → General, Menu, `F10`) hiding the OS title bar and frame, with in-window drag-to-move and edge/corner drag-to-resize | PR #206 |
| Version Display (Info Tab & Info Bar) | Displayed `application_identity()` inline in the Settings Info & Text tab, and added a toggleable `info_show_version` option (default OFF) to display the version on the right side of the output info bar with a dynamic `\| ` separator when narrow | PR #207 |
| Automatic Output Device Tracking & Manual Reload | When tracking the OS default output (`output_device = None`), automatically detects default playback device and sample rate changes every ~2 seconds across Windows (WASAPI), macOS (Core Audio Tap / SoundCard), and Linux/FreeBSD (PulseAudio/PipeWire), auto-recovers on stream disconnect errors, and adds manual reload (`F5` / Menu) | PR #207 |
| Unfocused Window Clickthrough | Enabled `SDL_MOUSE_FOCUS_CLICKTHROUGH=1` before `pg.init()` so clicking buttons/menus on an unfocused window focuses the window and triggers the button in a single click | #208, PR #209 |
| Auto-Layout & dB Scale Fixes | Fixed fullscreen Settings cancel recalculating auto-LED/bar-gap against windowed size, introduced `resizable_minimum_window_size()` so auto-adjust modes can shrink down to minimum segments/gap (10 LEDs / 1 px), and added adaptive thinning and omission for dB scale labels on compact vertical heights | #210, PR #211 |

## Key Merged Commits

| Commit | Description |
| --- | --- |
| `1c1f2b6` | feat(ui): unify header into single row with listening-mode styling (#205) |
| `4380249` | feat(ui): add optional borderless window mode with edge resizing and dragging (#206) |
| `be52f93` | feat: add version display in Info tab/info bar and automatic/manual audio device reload (#207) |
| `6bc3931` | Merge pull request #209 from yshdsnd/fix/issue-208-mouse-focus-clickthrough |
| `135533c` | Merge pull request #211 from yshdsnd/fix/issue-210-auto-layout-and-db-labels |

## File Overview

- **wune/config.py / wune/settings.py / wune/appearance.py**: Added persistence and preview support for `borderless` and `info_show_version`.
- **wune/application_menu.py / wune/locales/en.json / wune/locales/ja.json**: Hamburger menu icon, `borderless` (`F10`) and `reload_audio` (`F5`) menu items, and EN/JA localization keys.
- **wune/app.py**: Borderless toggling/dragging/edge-resizing, `SDL_MOUSE_FOCUS_CLICKTHROUGH`, 2-second default output polling and `F5` manual reload, and fullscreen cancel / shrink-resize consistency fixes.
- **wune/capture.py / wune/capture_macos.py / wune/tap_macos.py / wune/tap_backend.m / wune/soundcard_compat.py / wune/spectrum_audio.py**: Per-platform `needs_restart()` change detection and stream error auto-recovery.
- **wune/layout.py / wune/renderer.py / wune/settings_dialog.py**: Unified 1-row header, `resizable_minimum_window_size()`, right-aligned info bar version with dynamic separator, and non-overlapping adaptive dB scale rendering.
- **tests/**: Added `test_borderless.py`, `test_info_version.py`, `test_audio_reload.py`, `test_issue_210_auto_layout.py`, and updated existing menu/scaling/cleanup tests.

## Pre-Release Verification Checklist

1. All 604 unit tests pass across Windows, macOS, Linux, and FreeBSD.
2. Borderless window mode (`F10` / Menu / Settings), drag-to-move, edge-resizing, and info bar version display work cleanly.
3. OS default output device switches and sample rate changes are followed automatically on real hardware, and `F5` / Menu manual reload works immediately.
4. Shrinking with auto-LED and auto-bar-gap enabled reaches the minimum layout smoothly without overlapping dB scale labels.
