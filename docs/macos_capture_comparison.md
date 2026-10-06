# macOS Native Audio Capture: Core Audio Process Tap vs ScreenCaptureKit

Historical design notes imported from v1.0.1. For current routing, supported
inputs, and limitations, use [audio-backends.md](audio-backends.md).
The relative performance statements below are design estimates, not benchmarks.

This document compares Apple's **Core Audio Process Tap** API and **ScreenCaptureKit** for driverless system audio capture in Wune (Issue #75).

---

## Executive Summary

| Evaluation Dimension | **Core Audio Process Tap** | **ScreenCaptureKit (SCStream)** |
| :--- | :--- | :--- |
| **Primary Design Intent** | **Audio-only tap** for processes / system output | Screen & window recording (with audio stream) |
| **Virtual Driver Required?** | ❌ **No** (Direct native system PCM capture) | ❌ **No** (Direct native system PCM capture) |
| **Runtime Overhead & Latency** | **Lowest** (Operates directly at Core Audio HAL) | Moderate (Runs via ScreenCaptureKit daemon) |
| **Stereo / Sample Rate** | Native device ASBD (44.1k / 48k float32 stereo) | Configurable (`sampleRate=48000`, `channels=2`) |
| **Minimum macOS Version** | **macOS 14.2 (Sonoma)** | **macOS 13.0 (Ventura)** |
| **TCC Permission Prompt** | **System Audio Recording** (`NSAudioCaptureUsageDescription`) | **Screen & System Audio Recording** |
| **User Privacy Perception** | Clear (audio only, no camera/screen fear) | Concerning (asks for "Screen Recording" access) |
| **Python Implementation** | C API (`ctypes` or small C/Swift helper dylib) | Objective-C (`pyobjc-framework-ScreenCaptureKit`) |

---

## Detailed Comparison

### 1. Suitability for Spectrum Analyzer (Low-overhead continuous PCM)
- **Core Audio Process Tap**:
  - Purpose-built for tapping outgoing audio without creating virtual devices.
  - Feeds PCM frames directly through a lightweight Core Audio IO block / callback.
  - Zero involvement of WindowServer, GPU, or video frame allocators.
- **ScreenCaptureKit**:
  - Primarily a screen and window capture pipeline. Even with video output disabled (`capturesAudio = true`), the capture session coordinates with the display server daemon.
  - Slightly higher memory and process overhead for a background visualizer.

### 2. Permissions and User Experience (TCC)
- **Core Audio Process Tap**:
  - Requires `NSAudioCaptureUsageDescription` in `Info.plist`.
  - On macOS 14.2+, prompts specifically for permission to record system audio.
  - Users understand why an audio visualizer requests audio permissions.
- **ScreenCaptureKit**:
  - Triggers the system **"Screen & System Audio Recording"** permission prompt.
  - Users are frequently uncomfortable granting screen-recording permissions to an audio-only tool.

### 3. Implementation Path in Python / Native Bridge
- **Core Audio Process Tap**:
  - C functions: `AudioHardwareCreateProcessTap`, `AudioHardwareDestroyProcessTap`, `AudioDeviceCreateIOProcIDWithBlock`.
  - Accessible via Python's standard `ctypes` library calling `CoreAudio.framework`, or via a compiled thin C/Swift helper library packaged inside `wune/`.
- **ScreenCaptureKit**:
  - Requires `pyobjc-framework-ScreenCaptureKit` and Objective-C runloop/dispatch queue integration.
  - Binary size and dependencies of PyObjC wheels in PyInstaller packages are substantial.

---

## Recommendation for Wune

1. **Production Backend**: Adopt **Core Audio Process Tap** as the primary native driverless capture backend for macOS 14.2+.
2. **Fallback Strategy**:
   - macOS 14.2+: Core Audio Process Tap.
   - macOS 14.1 and below (or developer fallback): Existing CoreAudio / SoundCard loopback (e.g. BlackHole) or microphone input.
3. **Packaging**:
   - Provide `NSAudioCaptureUsageDescription` in the `.app` bundle `Info.plist`.
   - Provide a friendly, actionable error dialog when permissions are denied instead of silently capturing silence or falling back to a microphone.
