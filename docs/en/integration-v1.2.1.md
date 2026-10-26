# v1.2.1 Main Integration Log

**English** / [日本語](../integration-v1.2.1.md)

Record of changes and integrations in the `main` branch toward the patch update v1.2.1.
The major highlights are preventing crashes on Windows when resuming from system sleep/suspend while Wune is running, and resolving the issue where the spectrum analyzer became unresponsive on the first playback after resume (Issue #185).

| Feature | Integration Details | Related Issue / PR |
| --- | --- | --- |
| WASAPI COM `S_FALSE` Tolerance | Allowed non-negative HRESULTs (such as `S_FALSE` = 0x00000001) in SoundCard's `check_error` as successful completions, preventing crash exceptions (`Error 0x100000001`) on sleep resume | #185, PR #186 |
| Safe Recorder Teardown Guarantee | Patched `_Recorder.__exit__` so that exceptions during `Stop()` produce warnings while `finally` guarantees releasing COM interface pointers | #185, PR #186 |
| Sleep Resume Auto-Detection & Stream Reinit | Detected wall-clock time jumps (`wall_dt > 2.0s`) in main loop and recreated the loopback stream via `AudioSpectrum.restart_capture()`, ensuring immediate response on first audio playback | #185, PR #186 |
| Capture Exception Silence Fallback | Caught transient audio capture exceptions in `AudioSpectrum.step()` and returned zeroed (silent) arrays to prevent rendering loop termination | #185, PR #186 |
| Cross-Platform Safe Close Operations | Safely caught exceptions in `close()` across WASAPI, PulseAudio, and macOS backends, `AudioSpectrum.close()`, and `App.run()` `finally` clauses | #185, PR #186 |
| Startup Window Jump Prevention | Kept window hidden (`pg.HIDDEN`) during initialization and displayed it (`show()`) only after restoring geometry and rendering the first frame, eliminating window repositioning flashes | #184 |

## Key Integration Commits

| Commit | Description |
| --- | --- |
| `0aad278` | fix(capture): prevent crash on sleep resume by permitting WASAPI Stop S_FALSE (#185) |
| `59e521f` | fix(capture): refresh audio capture backend on system sleep resume |
| `06dcc17` | Merge pull request #186 from yshdsnd/fix/issue-185-wasapi-sleep-s-false-crash |

## File Group Verification

- **wune/soundcard_compat.py**: `S_FALSE` tolerance filter for WASAPI COM invocations, exception resilience and COM release guarantees in `_Recorder.__exit__`.
- **wune/capture.py**: Exception safety during backend `close()`.
- **wune/spectrum_audio.py**: Safe backend reinitialization via `restart_capture()`, silence fallback in `step()`, safe `close()`.
- **wune/app.py**: Wall-clock jump detection in main loop, stream refresh and clock reset via `on_system_resume()`.
- **tests/test_soundcard_compat.py**: COM error filter tests for various HRESULTs, exception handling and COM release tests in `_Recorder.__exit__`.
- **tests/test_capture.py**: Silence fallback tests during capture exceptions, `restart_capture()` reconfiguration tests.
- **tests/test_smart_idle.py**: Sleep resume detection via time jumps and event loop integration tests for `App.on_system_resume()`.

## Pre-Release Checklist

1. Unit tests pass on all platforms (Windows 11, macOS Apple Silicon, Linux).
2. Physical Windows verification of sleep and sleep resume (no crash, spectrum responds immediately on first playback).
3. Normal startup, capture, and exit verification on macOS and Linux.
