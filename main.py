"""Wune: Windows WASAPI loopback with an LED spectrum display.

ESC/Q: quit, F11: fullscreen, Space: pause, I: input information,
T or left-click the preset badge: next visual preset.
"""
# main.py
import pygame as pg
import sys

from wune.config import CFG
from wune.app import App

if __name__ == "__main__":
    try:
        App(CFG).run()
    except Exception as ex:
        print("Error:", ex)
        pg.quit()
        sys.exit(1)
