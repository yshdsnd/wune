# Cross-platform window and settings integration

Settings dialogs run in a spawned child process (`multiprocessing.get_context("spawn")`)
across Windows, macOS, and Linux so Tkinter owns that process's main thread without
interfering with pygame's rendering loop or audio capture.
The main application loop retains ownership of preferences:
preview, apply, save, and cancel exchange serializable snapshots over multiprocessing queues.
An unexpected child-process failure rolls back any uncommitted preview.
Closing settings joins the worker process and releases its queues; a worker that
does not exit within the shutdown timeout is terminated.

## Platform Differences

### Windows
- Window ownership: Passes the Win32 window handle (`winfo_id()`) to maintain proper
  z-order above fullscreen pygame surfaces and safe ownership rebinding during display changes.
- Dialog sizing: Default minimum window size is 580x480 px, tailored to Segoe UI font metrics.
- Keyboard shortcuts: F2 (settings), Alt+Enter / F11 (fullscreen), Space (pause), Esc / Q (quit).

### macOS
- Frontmost activation: Settings do not use Win32 handles; they activate their own process
  via `SetFrontProcessWithOptions` and `focus_force()`.
- Keyboard shortcuts: Adds Command+F (fullscreen) and Command+, (settings) alongside standard shortcuts.
- Monitor bounds: Obtained via CoreGraphics (`CGGetActiveDisplayList`), with fallback to online
  displays and pygame. Bounds are desktop coordinates rather than Retina physical pixel sizes.
- Fullscreen restoration: macOS currently launches windowed on startup.

### Linux / Unix
- Dialog sizing: Minimum window size is 640x500 px to accommodate Linux desktop font metrics
  (Noto Sans / DejaVu Sans) and prevent premature line wraps on localized text.
- Window management: Respects X11 and Wayland window manager placement.
- Shortcuts: Standard F2, Alt+Enter / F11, Space, Esc / Q.

## Multiprocessing and Frozen Bundles

Source and packaged entry points invoke `multiprocessing.freeze_support()`.
In packaged builds, the child process must enter multiprocessing routines before log
initialization to prevent recursive application startup or truncation of the parent's `Wune.log`.
