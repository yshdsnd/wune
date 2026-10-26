# Smart Idle

**English** / [日本語](../smart-idle.md)

When audio levels and peaks all drop to zero, and no user interaction, menu, or settings window is active,
Wune continues rendering for 30 frames before omitting subsequent frame drawing and screen blits.
It subsequently redraws once every 60 frames (approximately once per second at 60 FPS,
depending on audio capture loop timing).

Audio capture, FFT analysis, window events, and settings process IPC continue running uninterrupted.
The instant sound returns, full-rate rendering resumes on the very first frame without latency.
FFT omission on silent blocks is handled by existing audio processing logic.

Full rendering is maintained while menus, quit confirmations, or settings dialogs are displayed, or when paused.
Window resizing, restoration, or exposure events trigger an immediate redraw.
Settings preview and cancellation request redraws even in the absence of user events,
ensuring clean restoration as soon as the settings dialog closes.

Continuous capture during idle, sound resumption, peak decay, menus, quit confirmations, settings, pause,
and resume events are verified in Windows, macOS, Linux, and FreeBSD CI.
Exact real-world CPU and power savings depend on display hardware and system configuration.
