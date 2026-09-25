"""Application lifecycle and pygame event loop."""

import numpy as np
import pygame as pg
from copy import deepcopy
import warnings

from .config import Config
from .i18n import Translator
from .icons import set_app_id, pygame_icon
from .layout import clamp_window_size, fit_window_size
from .renderer import LedBarRenderer
from .spectrum_audio import AudioSpectrum


class App:
    def __init__(self, cfg: Config, settings_store=None, saved_geometry=None):
        cfg = deepcopy(cfg)
        set_app_id()
        pg.init()
        self._icon = pygame_icon()
        pg.display.set_caption(Translator(cfg.language)("app.title"))
        self.cfg = cfg
        self._requested_max_freq_hz = cfg.max_freq_hz
        self.settings_store = settings_store
        self.settings_dialog = None
        self._appearance_baseline = None
        self._settings_closing = False
        self._display_window = None
        self._windowed_position = None
        self._fullscreen = False
        self._windowed_size = fit_window_size((cfg.width, cfg.height), cfg)
        if settings_store is not None:
            from .window_geometry import restore_geometry, work_areas
            self._windowed_size, self._windowed_position = restore_geometry(cfg, saved_geometry or {}, work_areas())
        self._set_mode(self._windowed_size, pg.RESIZABLE)
        self._restore_position()
        self.clock = pg.time.Clock()
        self.renderer = LedBarRenderer(self.screen, cfg)
        self.renderer.user_presets = deepcopy(settings_store.user_presets) if settings_store is not None else {}
        if cfg.initial_preset is not None:
            self.renderer.apply_preset(cfg.initial_preset)
        # Audio errors must remain visible rather than silently showing fake data.
        self.spectrum = AudioSpectrum(cfg, cfg.bars, cfg.channels)
        self.spectrum.set_range(cfg.min_freq_hz, cfg.spectrum_upper_hz(self.spectrum.sr))
        cfg.max_freq_hz = self.spectrum.fmax

        self.running = True
        self.paused = False
        self.levels = np.zeros((cfg.channels, cfg.bars), dtype=np.float32)
        self.update_info_text()

    def _set_mode(self, size, flags):
        pg.display.set_icon(self._icon)
        self.screen = pg.display.set_mode(size, flags)
        if self._display_window is not None:
            # set_mode may replace the SDL window. Bind the new wrapper before
            # releasing the old one; queued events may still reference it.
            from pygame._sdl2.video import Window
            self._display_window = Window.from_display_module()

    def _geometry_window(self):
        if self._display_window is None:
            from pygame._sdl2.video import Window
            # pygame 2.6.1 stores a borrowed PyObject pointer in SDL window data.
            # A temporary wrapper leaves event.get() dereferencing freed memory.
            self._display_window = Window.from_display_module()
        return self._display_window

    def toggle_fullscreen(self):
        if self._fullscreen:
            self._set_mode(fit_window_size(self._windowed_size, self.cfg), pg.RESIZABLE)
            self._fullscreen = False
            self._restore_position()
        else:
            self._windowed_size = self.screen.get_size()
            self._remember_position()
            self._set_mode((0, 0), pg.FULLSCREEN)
            self._fullscreen = True
            if self.screen.get_size() != clamp_window_size(self.screen.get_size(), self.cfg):
                self._set_mode(fit_window_size(self._windowed_size, self.cfg), pg.RESIZABLE)
                self._fullscreen = False
                self._restore_position()
        self.renderer.resize(self.screen)

    def _remember_position(self):
        if self.settings_store is not None:
            self._windowed_position = tuple(self._geometry_window().position)

    def _restore_position(self):
        if self._windowed_position is not None:
            self._geometry_window().position = self._windowed_position

    def save_settings(self):
        if self.settings_store is None:
            return False
        try:
            if not self._fullscreen:
                self._windowed_size = self.screen.get_size()
                self._remember_position()
            self.settings_store.user_presets = deepcopy(self.renderer.user_presets)
            return self.settings_store.save(self.cfg, self._windowed_size, self._windowed_position,
                                            self.renderer.preset_name)
        except (OSError, pg.error) as error:
            warnings.warn(f"Cannot save window settings: {error}", RuntimeWarning)
            return False

    def open_settings(self):
        if self.settings_dialog is not None:
            self.settings_dialog.focus()
            return
        from .appearance import AppearanceState
        from .settings_dialog import SettingsDialog
        from .settings import settings_path
        if self._fullscreen:
            self.toggle_fullscreen()
        state = AppearanceState.capture(self.cfg, self.renderer.preset_name, self.renderer.user_presets)
        self._appearance_baseline = state
        self._appearance_size = self.screen.get_size()
        self._settings_closing = False
        path = self.settings_store.path if self.settings_store is not None else settings_path()
        self.settings_dialog = SettingsDialog(state, path)

    def preview_appearance(self, state, size=None):
        from .appearance import AppearanceState
        previous = AppearanceState.capture(self.cfg, self.renderer.preset_name, self.renderer.user_presets)
        previous.layout["language"] = state.layout["language"]
        language_only = previous == state
        previous_cap = self.cfg.limit_to_20khz
        state.apply(self.cfg)
        pg.display.set_caption(Translator(self.cfg.language)("app.title"))
        self.update_info_text()
        if language_only and size is None:
            return
        if self.cfg.limit_to_20khz != previous_cap:
            maximum = self.cfg.spectrum_upper_hz(self.spectrum.sr,
                                                requested_max_hz=self._requested_max_freq_hz)
            if maximum != self.spectrum.fmax:
                self.spectrum.set_range(self.cfg.min_freq_hz, maximum)
                self.cfg.max_freq_hz = self.spectrum.fmax
                # Bars now describe different frequencies; discard old trails.
                self.levels.fill(0)
                self.renderer.reset_peaks()
        self.renderer.user_presets = deepcopy(state.user_presets)
        self.renderer.preset_name = state.preset.name
        self.renderer._led_cache.clear()
        self.resize_window(size or self.screen.get_size())

    def cancel_settings(self):
        if self._appearance_baseline is not None:
            self.preview_appearance(self._appearance_baseline, self._appearance_size)
            self._appearance_baseline = None

    def poll_settings(self):
        from queue import Empty
        dialog = self.settings_dialog
        if dialog is None:
            return
        try:
            while True:
                action, state = dialog.events.get_nowait()
                if action == "closed":
                    self.cancel_settings()
                    self.settings_dialog = None
                    return
                if action == "error":
                    warnings.warn(f"Cannot open settings: {state}", RuntimeWarning)
                    continue
                if self._settings_closing:
                    continue
                if action == "cancel":
                    self.cancel_settings()
                    self._settings_closing = True
                    dialog.reply(True, close=True)
                elif action in ("preview", "apply", "save"):
                    self.preview_appearance(state)
                    if action == "preview":
                        continue
                    if action == "save" and not self.save_settings():
                        dialog.reply(False, "error.save")
                        continue
                    self._appearance_baseline = deepcopy(state)
                    self._appearance_size = self.screen.get_size()
                    if action == "save":
                        self._appearance_baseline = None
                        self._settings_closing = True
                    dialog.reply(True, "status.applied", close=action == "save")
        except Empty:
            pass

    def resize_window(self, size):
        if self._fullscreen and self.screen.get_size() != clamp_window_size(self.screen.get_size(), self.cfg):
            self.toggle_fullscreen()
            return
        if not self._fullscreen:
            size = fit_window_size(size, self.cfg)
            if self.screen.get_size() != size:
                self._set_mode(size, pg.RESIZABLE)
            self._windowed_size = size
        self.renderer.resize(self.screen)

    def handle_event(self, event: pg.event.Event):
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.VIDEORESIZE:
            self.resize_window(event.size)
        elif event.type == pg.WINDOWSIZECHANGED:
            # pygame 2 updates the display Surface when the native window resizes.
            self.resize_window(self.screen.get_size())
        elif event.type == pg.MOUSEBUTTONDOWN:
            if self.settings_dialog is None and event.button == 1 and self.renderer.badge_contains(event.pos):
                self.renderer.next_preset()
                self.resize_window(self.screen.get_size())
        elif event.type == pg.KEYDOWN:
            if event.key in (pg.K_ESCAPE, pg.K_q):
                self.running = False
            elif event.key == pg.K_F11:
                self.toggle_fullscreen()
            elif event.key == pg.K_SPACE:
                self.paused = not self.paused
            elif event.key == pg.K_i:
                if self.settings_dialog is None:
                    self.cfg.info_enabled = not self.cfg.info_enabled
                    self.resize_window(self.screen.get_size())
            elif event.key == pg.K_t:
                if self.settings_dialog is None:
                    self.renderer.next_preset()
                    self.resize_window(self.screen.get_size())
            elif event.key == pg.K_F2:
                self.open_settings()

    def update_info_text(self):
        # Report actual capture channels, not endpoint capacity or display rows.
        spectrum = self.spectrum
        t = Translator(self.cfg.language)
        device = spectrum.device if spectrum.device is not None else t("app.default_output")
        self.renderer.info_text = t("app.output", device=device, rate=spectrum.sr / 1000,
                                    channels=spectrum.channels_eff)

    def run(self):
        try:
            while self.running:
                dt = self.clock.tick(self.cfg.fps) / 1000.0
                for event in pg.event.get():
                    self.handle_event(event)
                if not self.running:
                    break
                self.poll_settings()

                if not self.paused:
                    self.levels = self.spectrum.step(dt)
                # Pause reuses the last levels and freezes peak timers.
                self.update_info_text()
                self.renderer.draw(self.levels, dt=0.0 if self.paused else dt)
                if self.paused:
                    self.renderer.draw_pause_overlay()
                pg.display.flip()
            self.cancel_settings()
            self.save_settings()
        finally:
            if self.settings_dialog is not None:
                self.settings_dialog.close()
            try:
                self.spectrum.close()
            finally:
                pg.quit()
