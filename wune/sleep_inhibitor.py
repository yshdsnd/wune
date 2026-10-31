"""System sleep and display turn-off inhibitor management."""
from __future__ import annotations

import sys
import pygame as pg

# Windows API constants for SetThreadExecutionState
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


class SleepInhibitor:
    """Controls OS screen turn-off and system sleep suppression during audio playback."""

    def __init__(self, kernel32=None):
        self.is_inhibited: bool | None = None
        self._kernel32 = kernel32
        if self._kernel32 is None and sys.platform == "win32":
            import ctypes
            try:
                self._kernel32 = ctypes.windll.kernel32
                self._kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
                self._kernel32.SetThreadExecutionState.restype = ctypes.c_uint
            except Exception:
                self._kernel32 = None

    def apply(self) -> None:
        """Re-apply current inhibition state (e.g. after display mode change)."""
        if self.is_inhibited is not None:
            self._apply_state(self.is_inhibited)

    def set_inhibited(self, inhibit: bool) -> None:
        """Update sleep inhibition state."""
        if self.is_inhibited == inhibit:
            return
        self.is_inhibited = inhibit
        self._apply_state(inhibit)

    def _apply_state(self, inhibit: bool) -> None:
        # 1. Pygame / SDL level screensaver management (cross-platform: Windows, macOS, Linux)
        try:
            if pg.display.get_init():
                pg.display.set_allow_screensaver(not inhibit)
        except Exception:
            pass

        # 2. Windows-specific: explicitly manage thread execution state
        if self._kernel32 is not None:
            try:
                if inhibit:
                    self._kernel32.SetThreadExecutionState(
                        ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED
                    )
                else:
                    self._kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            except Exception:
                pass

    def close(self) -> None:
        """Ensure suppression is completely cleared on exit."""
        self.set_inhibited(False)
