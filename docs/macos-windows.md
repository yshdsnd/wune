# macOS settings and window integration

Settings run in a spawned child process on macOS so Tk owns that process's
main thread. The main pygame/audio loop retains ownership of preferences:
preview, apply, save and cancel exchange serializable snapshots over queues.
An unexpected child-process failure rolls back the uncommitted preview.
Closing settings joins the worker and releases its queues; a worker that
does not exit within the shutdown timeout is terminated.

Windows retains its existing Tk thread and native owner relationship, including
settings above fullscreen and safe ownership rebinding during display changes.
Mac settings send no Win32 handle. On request they activate their own process.
Fullscreen/Spaces and focus behavior still require physical Mac verification.

Mac shortcuts add Command+F (fullscreen) and Command+, (settings).
F2, F11, Alt+Enter, the application menu, and existing exit confirmation remain.
The Windows key behavior is unchanged.

Mac monitor bounds come from CoreGraphics, with active, online, main-display,
then pygame fallback. Bounds are desktop coordinates, not Retina pixel sizes.
CoreGraphics bounds include the menu bar and Dock; the existing restoration
margins are retained. Accurate per-display usable-area/Dock handling is not
introduced here. Windows monitor identity and conditional fullscreen restoration
remain Windows-specific; Mac starts windowed, as in the v1.0.1 implementation.

Source and packaged entry points use multiprocessing.freeze_support(). The
packaged child must enter multiprocessing before log initialization, avoiding
recursive application startup or truncation of the parent's log. Full macOS
packaging verification belongs to the packaging stage of Issue #98.

CI runs an actual hidden Tk process on macOS, checking startup, acknowledged
close, direct close and reopen. Real-device checks still include live preview,
apply/cancel, native file/color dialogs, full-screen settings visibility,
multiple displays, focus switching and application shutdown.
