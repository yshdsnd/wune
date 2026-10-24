"""Command line interface and main entry point for Wune."""
import argparse
import os
import sys
import warnings

from wune.config import CFG
from wune.app import App
from wune.settings import SettingsStore


def _apply_platform_environment():
    """Apply platform-specific environment settings before initializing display/audio."""
    if sys.platform == "win32":
        os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
    elif sys.platform == "darwin":
        os.environ.setdefault("SDL_VIDEO_MAC_FULLSCREEN_SPACES", "0")

    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*avx2 capable.*")


def main(argv=None):
    """Launch Wune with parsed command-line arguments."""
    _apply_platform_environment()

    parser = argparse.ArgumentParser(description="Wune: System audio capture with an LED spectrum display.")
    parser.add_argument(
        "--reset-settings", action="store_true",
        help="Reset saved preferences and geometry, then launch with built-in defaults"
    )
    args = parser.parse_args(argv)
    store = SettingsStore()
    cfg, geometry = store.reset() if args.reset_settings else store.load(CFG)
    App(cfg, settings_store=store, saved_geometry=geometry).run()
