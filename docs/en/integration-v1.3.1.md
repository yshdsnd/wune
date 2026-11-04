# v1.3.1 Main Integration Log

**English** / [日本語](../integration-v1.3.1.md)

Record of changes and integrations in the `main` branch toward the patch update v1.3.1.
The key fixes address visual clipping of right-edge spectrum bars and frequency labels in 4K/maximized display modes (Issue #200), and release display-off and system sleep suppression during idle/silent periods to protect displays and conserve power (Issue #201).

| Feature | Integration Details | Related Issue / PR |
| --- | --- | --- |
| Fix Spectrum Right-Edge Overflow on 4K/Maximized Displays | Resolved an issue in vertical channel layout where rounding errors and insufficient margins on wide 4K displays caused right-most LED columns and the "20k" scale label to extend ~36px beyond the window boundary and clip. Implemented adaptive layout width optimization and strict boundary clamping | #200, PR #202 |
| Release Display Turn-Off & Sleep Suppression During Idle | Introduced `SleepInhibitor` to control Pygame/SDL2's default persistent screen turn-off inhibition (`ES_DISPLAY_REQUIRED`). Prevents screen turn-off while actively visualizing audio, but cleanly releases sleep and display suppression after 10 seconds of silence (or when paused/idle at launch), allowing native OS display sleep and screen burnout prevention | #201, PR #203 |

## Key Integration Commits

| Commit | Description |
| --- | --- |
| `6e2b9ec` | fix(layout): prevent spectrum clipping on right edge when maximized or fullscreen (#200) |
| `bc56f27` | Merge pull request #202 from yshdsnd/fix/issue-200-vertical-layout-overflow |
| `a75f62e` | fix(power): release screen turn-off and sleep suppression during idle (#201) |
| `98ef881` | Merge pull request #203 from yshdsnd/fix/issue-201-idle-sleep |

## File Group Verification

- **wune/layout.py**: Right-edge clipping prevention and boundary clamping for column groups during fullscreen/maximized layouts.
- **wune/sleep_inhibitor.py**: Clean state management for screensaver/display-off suppression (`pg.display.set_allow_screensaver`) and Win32 `SetThreadExecutionState`.
- **wune/config.py**: Added `sleep_inhibit_idle_seconds` (default: 10.0 seconds).
- **wune/app.py**: Dynamic `SleepInhibitor` engagement on active playback, timeout release during silence, and safe release on pause/teardown.
- **tests/test_layout.py**: 4K (3840x2160) maximized/fullscreen right-edge bounds regression test.
- **tests/test_sleep_inhibitor.py**: Unit tests verifying inhibitor state transitions across silence, playback restart, pause, sleep resume, and teardown (7 tests).

## Pre-Release Checklist

1. All unit tests pass across all platforms (544 tests).
2. Rightmost spectrum columns and the "20k" label remain fully visible on 4K / maximized displays.
3. Windows physical verification: Display turn-off and system sleep proceed normally after 10 seconds of silence or when playback is paused.
