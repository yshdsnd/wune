"""Hardware-free checks executed by the frozen executable, not host Python."""
import json
import os
from pathlib import Path
from queue import Queue
import sys


def run(report):
    if not getattr(sys, "frozen", False):
        raise RuntimeError("Run the smoke test through the packaged Wune.exe")
    import numpy as np
    import pygame as pg
    import tkinter as tk
    from pygame._sdl2.video import Window
    from .soundcard_compat import prepare_soundcard
    from .settings import SettingsStore, settings_path
    from .config import Config
    from .i18n import Translator
    from .appearance import AppearanceDraft, AppearanceState
    from .settings_dialog import _Dialog
    from .renderer import LedBarRenderer
    from .icons import ASSETS, set_app_id, pygame_icon
    prepare_soundcard()  # Includes metadata, CFFI, WASAPI headers and COM loading.
    import soundcard
    bundle = Path(sys._MEIPASS).resolve()
    for module in (np, pg, tk, soundcard):
        if not Path(module.__file__).resolve().is_relative_to(bundle):
            raise RuntimeError(f"Dependency escaped the bundle: {module.__name__}")
    if settings_path().resolve().is_relative_to(Path(sys.executable).parent.resolve()):
        raise RuntimeError("Settings must live outside the application directory")
    np.fft.rfft(np.zeros(4096))
    if not ASSETS.resolve().is_relative_to(bundle) or not (ASSETS / "Wune.ico").is_file():
        raise RuntimeError("Missing bundled icon resources")
    set_app_id()
    pg.display.init()
    pg.display.set_icon(pygame_icon())
    pg.display.set_mode((320, 200), pg.HIDDEN)
    pg.font.init()
    for language in ("en", "ja"):
        cfg = Config(language=language)
        renderer = LedBarRenderer(pg.Surface((1280, 800)), cfg)
        renderer.draw(np.zeros((cfg.channels, cfg.bars), dtype=np.float32))
        renderer.draw_pause_overlay()
        if Translator(language)("app.paused") == "app.paused":
            raise RuntimeError("Missing locale data")
        root = tk.Tk()
        root.withdraw()
        try:
            dialog = _Dialog(root, AppearanceDraft(AppearanceState.capture(cfg, "CLASSIC", {})),
                             str(settings_path()), Queue(), Queue())
            root.update_idletasks()
            dialog.style("led_shape", "ellipse")
        finally:
            root.destroy()
    store = SettingsStore()
    cfg = Config(language="ja", led_shape="ellipse")
    if not store.save(cfg, (1000, 700), (40, 50), "BLUE"):
        raise RuntimeError("Cannot save settings")
    restored, geometry = store.load(Config())
    if restored.language != "ja" or restored.led_shape != "ellipse" or geometry["x"] != 40:
        raise RuntimeError("Settings round trip failed")
    pg.quit()
    report.write_text(json.dumps({"ok": True, "frozen": True, "languages": ["en", "ja"],
                                 "settings": str(settings_path())}), encoding="utf-8")
