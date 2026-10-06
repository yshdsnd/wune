"""Hardware-free checks executed by the frozen executable, not host Python."""
import json
import os
from pathlib import Path
from queue import Queue
import sys


def check_macos_settings(cfg):
    """Exercise the frozen spawn entry, not Tk in the pygame process."""
    from .appearance import AppearanceState
    from .settings import settings_path
    from .settings_dialog import SettingsDialog
    dialog = SettingsDialog(AppearanceState.capture(cfg, "CLASSIC", {}), settings_path())
    try:
        action, payload = dialog.events.get(timeout=30)
        if action != "ready":
            raise RuntimeError(f"Settings process failed: {action}: {payload}")
        dialog.reply(True, close=True)
        action, payload = dialog.events.get(timeout=15)
        if action != "closed":
            raise RuntimeError(f"Settings process did not close: {action}: {payload}")
    finally:
        dialog.close()
    if dialog.thread.is_alive() or dialog.thread.exitcode != 0:
        raise RuntimeError("Settings process did not exit cleanly")


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
    from .build_identity import application_identity, window_title
    from .i18n import Translator
    from .appearance import AppearanceDraft, AppearanceState
    from .settings_dialog import _Dialog
    from .renderer import LedBarRenderer
    from .application_menu import ApplicationMenu
    from .exit_confirmation import ExitConfirmation
    from .icons import ASSETS, set_app_id, pygame_icon
    prepare_soundcard()  # Includes metadata, CFFI, WASAPI headers and COM loading.
    import soundcard
    bundle = Path(sys._MEIPASS).resolve()
    bundle_root = bundle.parent if sys.platform == "darwin" and bundle.name == "Frameworks" else bundle
    for module in (np, pg, tk, soundcard):
        if not Path(module.__file__).resolve().is_relative_to(bundle_root):
            raise RuntimeError(f"Dependency escaped the bundle: {module.__name__}")
    if settings_path().resolve().is_relative_to(Path(sys.executable).parent.resolve()):
        raise RuntimeError("Settings must live outside the application directory")
    np.fft.rfft(np.zeros(4096))
    if not ASSETS.resolve().is_relative_to(bundle_root) or not (ASSETS / "Wune.ico").is_file():
        raise RuntimeError("Missing bundled icon resources")
    if sys.platform == "darwin":
        import ctypes
        from .tap_macos import _ensure_dylib
        dylib = Path(_ensure_dylib()).resolve()
        if not dylib.is_relative_to(bundle_root) or not (ASSETS / "Wune.icns").is_file():
            raise RuntimeError("Missing bundled macOS audio/icon resources")
        ctypes.CDLL(str(dylib))  # Load only: never request audio permission in CI.
    set_app_id()
    pg.display.init()
    pg.display.set_icon(pygame_icon())
    pg.display.set_mode((320, 200), pg.HIDDEN)
    pg.font.init()
    for language in ("en", "ja"):
        pg.display.set_caption(window_title(language))
        if pg.display.get_caption()[0] != window_title(language):
            raise RuntimeError("Window title differs from build identity")
        cfg = Config(language=language)
        image_path = report.with_name("background.png")
        image = pg.Surface((40, 20))
        image.fill((30, 50, 90))
        pg.image.save(image, str(image_path))
        cfg.background_mode = "image"
        cfg.background_path = str(image_path)
        renderer = LedBarRenderer(pg.Surface((1280, 800)), cfg)
        renderer.draw(np.zeros((cfg.channels, cfg.bars), dtype=np.float32))
        renderer.draw_pause_overlay()
        menu = ApplicationMenu()
        menu.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=3, pos=(1270, 790)),
                    renderer.surf.get_size(), renderer.font_small, language, False)
        menu.draw(renderer.surf, renderer.font_small, language, False, cfg.theme)
        menu.handle(pg.event.Event(pg.WINDOWFOCUSLOST), renderer.surf.get_size(),
                    renderer.font_small, language, False)
        if menu.anchor is not None:
            raise RuntimeError("Application menu did not dismiss on focus loss")
        if renderer.background.source is None:
            raise RuntimeError("Packaged image background could not load")
        prompt = ExitConfirmation()
        prompt.open()
        prompt.draw(renderer.surf, renderer.font_small, language, cfg.theme)
        if Translator(language)("app.paused") == "app.paused":
            raise RuntimeError("Missing locale data")
        if sys.platform == "darwin":
            check_macos_settings(cfg)
            continue
        root = tk.Tk()
        root.withdraw()
        try:
            dialog = _Dialog(root, AppearanceDraft(AppearanceState.capture(cfg, "CLASSIC", {})),
                             str(settings_path()), Queue(), Queue())
            root.update_idletasks()
            from .settings_window import SettingsWindow
            native = SettingsWindow(root.winfo_id())
            try:
                native.bind(pg.display.get_wm_info().get("window"))
                native.position(pg.display.get_wm_info().get("window"))
                dialog.style("led_shape", "ellipse")
            finally:
                native.detach()
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
    report.write_text(json.dumps({"ok": True, "frozen": True, "identity": application_identity(), "languages": ["en", "ja"],
                                 "settings": str(settings_path())}), encoding="utf-8")
