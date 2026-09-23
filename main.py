"""Wune: Windows WASAPI loopback with an LED spectrum display.

ESC/Q: quit, F11: fullscreen, Space: pause, I: input information,
T or left-click the preset badge: next visual preset.
"""
# main.py
import pygame as pg
import sys
import argparse

from wune.config import CFG
from wune.app import App
from wune.settings import SettingsStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-settings", action="store_true",
                        help="Reset saved preferences and geometry, then launch with built-in defaults")
    args = parser.parse_args(argv)
    store = SettingsStore()
    cfg, geometry = store.reset() if args.reset_settings else store.load(CFG)
    App(cfg, settings_store=store, saved_geometry=geometry).run()

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print("Error:", ex)
        pg.quit()
        sys.exit(1)
