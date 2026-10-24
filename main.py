"""Wune: System audio capture with an LED spectrum display.

Menu button/right-click: Settings, fullscreen, Exit.
ESC: leave fullscreen or ask to quit; Q: ask to quit, Alt+Enter (also F11): fullscreen, Space: pause, I: input information,
T or left-click the preset badge: next visual preset.
macOS: Command+F toggles fullscreen; Command+, opens settings.
"""
import sys

import pygame as pg

from wune.cli import main

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as ex:
        print("Error:", ex)
        pg.quit()
        sys.exit(1)
