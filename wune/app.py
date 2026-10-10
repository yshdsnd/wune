"""Application lifecycle and pygame event loop."""

import numpy as np
import pygame as pg
from copy import deepcopy
import warnings
import os
import sys

from .config import Config
from .application_menu import ApplicationMenu
from .exit_confirmation import ExitConfirmation
from .build_identity import window_title
from .i18n import Translator
from .icons import set_app_id, pygame_icon
from .layout import clamp_window_size, fit_window_size, channel_mode_window_size
from .monitor_identity import connected_monitors, matching_monitor, current_monitor_identity
from .renderer import LedBarRenderer
from .spectrum_audio import AudioSpectrum


class App:
    def __init__(self, cfg: Config, settings_store=None, saved_geometry=None):
        cfg = deepcopy(cfg)
        set_app_id()
        # Keep fullscreen visible when another monitor receives keyboard focus.
        # Set before SDL initialization; do not raise the window or force topmost.
        os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"
        if sys.platform == "win32":
            os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
        pg.init()
        self._icon = pygame_icon()
        pg.display.set_caption(window_title(cfg.language))
        self.cfg = cfg
        self._requested_max_freq_hz = cfg.max_freq_hz
        self.settings_store = settings_store
        self.settings_dialog = None
        self._settings_window = None
        self._appearance_baseline = None
        self._settings_closing = False
        self._display_window = None
        self._windowed_position = None
        self._fullscreen = False
        self._windowed_size = clamp_window_size((cfg.width, cfg.height), cfg)
        if settings_store is not None:
            from .window_geometry import restore_geometry, work_areas
            self._windowed_size, self._windowed_position = restore_geometry(cfg, saved_geometry or {}, work_areas())
        self._set_mode(self._windowed_size, pg.RESIZABLE)
        self._restore_position()
        self.clock = pg.time.Clock()
        self.renderer = LedBarRenderer(self.screen, cfg)
        self.menu = ApplicationMenu()
        self.exit_confirmation = ExitConfirmation()
        self._disable_exit_confirmation = False
        self.renderer.user_presets = deepcopy(settings_store.user_presets) if settings_store is not None else {}
        if cfg.initial_preset is not None:
            self.renderer.apply_preset(cfg.initial_preset)
        self._restore_fullscreen(saved_geometry or {})
        # Audio errors must remain visible rather than silently showing fake data.
        self.spectrum = AudioSpectrum(cfg, cfg.bars, cfg.channels)
        self.spectrum.set_range(cfg.min_freq_hz, cfg.spectrum_upper_hz(self.spectrum.sr))
        cfg.max_freq_hz = self.spectrum.fmax

        self.running = True
        self.paused = False
        self._idle_frames = 0
        self._redraw_requested = True
        self.levels = np.zeros((cfg.display_channels, cfg.bars), dtype=np.float32)
        from .now_playing import create_default_coordinator
        self.now_playing = create_default_coordinator()
        self.now_playing.enabled = bool(self.cfg.show_now_playing)
        self.now_playing.add_listener(lambda _: setattr(self, "_redraw_requested", True))
        self.now_playing.start()
        self.update_info_text()
        self.update_now_playing_text()

    def _set_mode(self, size, flags, *, display=None):
        pg.display.set_icon(self._icon)
        # Detach before SDL can replace/destroy its HWND (which destroys owned windows).
        if self._settings_window is not None:
            self._settings_window.detach()
        try:
            self.screen = pg.display.set_mode(size, flags, **({"display": display} if display is not None else {}))
        finally:
            if self._settings_window is not None:
                self._settings_window.bind(pg.display.get_wm_info().get("window"))
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

    def _enter_fullscreen(self, *, display=None):
        # Prefer desktop fullscreen (SDL_WINDOW_FULLSCREEN_DESKTOP) to preserve
        # the display's current desktop resolution and refresh rate without
        # changing physical video modes or causing monitor resync blackouts.
        try:
            window = self._geometry_window()
            window.set_fullscreen(desktop=True)
            surface = pg.display.get_surface()
            if surface is not None:
                self.screen = surface
                self._fullscreen = True
                return
        except Exception as error:
            warnings.warn(f"Desktop fullscreen unavailable ({error}); falling back to mode switch", RuntimeWarning)
        self._set_mode((0, 0), pg.FULLSCREEN, display=display)
        self._fullscreen = True

    def _leave_fullscreen(self):
        try:
            self._geometry_window().set_windowed()
        except Exception:
            pass
        self._set_mode(clamp_window_size(self._windowed_size, self.cfg), pg.RESIZABLE)
        self._fullscreen = False
        self._restore_position()

    def toggle_fullscreen(self):
        self.menu.close()
        if self._fullscreen:
            self._leave_fullscreen()
        else:
            self._windowed_size = self.screen.get_size()
            self._remember_position()
            self._enter_fullscreen()
            if self.screen.get_size() != clamp_window_size(self.screen.get_size(), self.cfg):
                self._leave_fullscreen()
        self.renderer.resize(self.screen)

    def _restore_fullscreen(self, saved):
        if saved.get("fullscreen") is not True:
            return
        try:
            target = matching_monitor(saved.get("fullscreen_monitor"), connected_monitors())
            if target is None:
                return
            # Resolve SDL's current index by placing the window on the identified
            # monitor. Keep the saved normal-window geometry untouched.
            x, y, width, height = target.bounds
            if (width, height) != clamp_window_size((width, height), self.cfg):
                return
            placement_size = (min(width, self._windowed_size[0]), min(height, self._windowed_size[1]))
            if placement_size != self.screen.get_size():
                self._set_mode(placement_size, pg.RESIZABLE)
            window = self._geometry_window()
            window.position = (x + max(0, (width - placement_size[0]) // 2),
                               y + max(0, (height - placement_size[1]) // 2))
            self._enter_fullscreen(display=window.display_index)
            actual = current_monitor_identity(pg.display.get_wm_info().get("window"))
            if not actual or actual.casefold() != target.identity.casefold():
                raise ValueError("Fullscreen monitor changed during startup")
            self.renderer.resize(self.screen)
            return
        except (OSError, pg.error, ValueError) as error:
            warnings.warn(f"Cannot restore fullscreen: {error}. Using a normal window.", RuntimeWarning)
        self._fullscreen = False
        try:
            self._geometry_window().set_windowed()
        except Exception:
            pass
        self._set_mode(self._windowed_size, pg.RESIZABLE)
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
            monitor = None
            if self._fullscreen:
                try:
                    monitor = current_monitor_identity(pg.display.get_wm_info().get("window"))
                except OSError as error:
                    warnings.warn(f"Cannot identify fullscreen monitor: {error}", RuntimeWarning)
            return self.settings_store.save(self.cfg, self._windowed_size, self._windowed_position,
                                            self.renderer.preset_name, fullscreen=self._fullscreen,
                                            fullscreen_monitor=monitor)
        except (OSError, pg.error) as error:
            warnings.warn(f"Cannot save window settings: {error}", RuntimeWarning)
            return False

    def open_settings(self):
        self.menu.close()
        if self.settings_dialog is not None:
            self.settings_dialog.focus()
            return
        from .appearance import AppearanceState
        from .settings_dialog import SettingsDialog
        from .settings import settings_path
        state = AppearanceState.capture(self.cfg, self.renderer.preset_name, self.renderer.user_presets)
        self._appearance_baseline = state
        self._appearance_size = self._windowed_size if self._fullscreen else self.screen.get_size()
        self._settings_closing = False
        path = self.settings_store.path if self.settings_store is not None else settings_path()
        self.settings_dialog = SettingsDialog(state, path)

    def preview_appearance(self, state, size=None):
        self._redraw_requested = True
        from .appearance import AppearanceState
        previous = AppearanceState.capture(self.cfg, self.renderer.preset_name, self.renderer.user_presets)
        previous.layout["language"] = state.layout["language"]
        previous.background = deepcopy(state.background)
        previous.layout["confirm_keyboard_exit"] = state.layout["confirm_keyboard_exit"]
        presentation_only = previous == state
        previous_cfg = deepcopy(self.cfg)
        previous_mode = self.cfg.channel_mode
        previous_layout = self.cfg.channel_layout
        previous_orientation = self.cfg.spectrum_orientation
        previous_cap = self.cfg.limit_to_20khz
        state.apply(self.cfg)
        if self.cfg.channel_mode != previous_mode:
            if size is None and not self._fullscreen:
                size = channel_mode_window_size(self.screen.get_size(), previous_cfg, self.cfg)
            self.spectrum.set_display_mode(self.cfg.channel_mode)
            self.levels = np.zeros((self.cfg.display_channels, self.cfg.bars), dtype=np.float32)
        elif (self.cfg.channel_layout != previous_layout or
              self.cfg.spectrum_orientation != previous_orientation or
              self.cfg.leds_per_bar != previous_cfg.leds_per_bar):
            if size is None and not self._fullscreen:
                size = clamp_window_size(self.screen.get_size(), self.cfg)
        pg.display.set_caption(window_title(self.cfg.language))
        self.update_info_text()
        self.update_now_playing_text()
        if presentation_only and size is None:
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
        if size is not None:
            self.resize_window(size)
        else:
            self._redraw_requested = True
            if self._fullscreen and self.screen.get_size() != clamp_window_size(self.screen.get_size(), self.cfg):
                self.toggle_fullscreen()
            else:
                clamped = clamp_window_size(self.screen.get_size(), self.cfg)
                if not self._fullscreen and self.screen.get_size() != clamped:
                    self._set_mode(clamped, pg.RESIZABLE)
                    self._windowed_size = clamped
                self.renderer.resize(self.screen)

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
                if action == "ready":
                    if sys.platform == "win32":
                        from .settings_window import SettingsWindow
                        try:
                            self._settings_window = SettingsWindow(state)
                            owner = pg.display.get_wm_info().get("window")
                            self._settings_window.bind(owner)
                            self._settings_window.position(owner)
                        except OSError as error:
                            warnings.warn(f"Cannot attach settings window: {error}", RuntimeWarning)
                    dialog.focus()
                    continue
                if action == "closed":
                    dialog.close()
                    self.cancel_settings()
                    self.settings_dialog = None
                    self._settings_window = None
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
                    self._appearance_size = self._windowed_size if self._fullscreen else self.screen.get_size()
                    if action == "save":
                        self._appearance_baseline = None
                        self._settings_closing = True
                    dialog.reply(True, "status.applied", close=action == "save")
        except Empty:
            if getattr(dialog, "worker_failed", False) is True:
                warnings.warn("Settings process exited unexpectedly; reverting preview", RuntimeWarning)
                dialog.close()
                self.cancel_settings()
                self.settings_dialog = None
                self._settings_window = None

    def resize_window(self, size):
        self._redraw_requested = True
        if self._fullscreen and self.screen.get_size() != clamp_window_size(self.screen.get_size(), self.cfg):
            self.toggle_fullscreen()
            return
        if not self._fullscreen:
            size = clamp_window_size(size, self.cfg)
            if self.screen.get_size() != size:
                self._set_mode(size, pg.RESIZABLE)
            self._windowed_size = size
        self.renderer.resize(self.screen)

    def handle_event(self, event: pg.event.Event):
        if event.type == pg.QUIT:
            self.execute_command("exit")
            return
        if self.exit_confirmation.active:
            result = self.exit_confirmation.handle(event, self.screen.get_size(),
                                                   self.renderer.font_small, self.cfg.language)
            if result is not None:
                confirmed, dont_ask = result
                if confirmed:
                    self._disable_exit_confirmation = dont_ask
                    self.execute_command("exit")
            if event.type not in (pg.VIDEORESIZE, pg.WINDOWSIZECHANGED):
                return
        consumed, command = self.menu.handle(event, self.screen.get_size(), self.renderer.font_small,
                                             self.cfg.language, self._fullscreen)
        if command is not None:
            self.execute_command(command)
        if consumed:
            return
        if event.type == pg.QUIT:
            self.execute_command("exit")
        elif event.type == pg.VIDEORESIZE:
            self.resize_window(event.size)
        elif event.type == pg.WINDOWSIZECHANGED:
            # pygame 2 updates the display Surface when the native window resizes.
            self.resize_window(self.screen.get_size())
        elif event.type == pg.MOUSEBUTTONDOWN:
            if self.settings_dialog is None and event.button == 1 and self.renderer.badge_contains(event.pos):
                self.renderer.next_preset()
                self._redraw_requested = True
        elif event.type == pg.KEYDOWN:
            mac_cmd = sys.platform == "darwin" and bool(getattr(event, "mod", 0) & pg.KMOD_META)
            if mac_cmd and event.key == pg.K_COMMA:
                self.execute_command("settings")
                return
            if mac_cmd and event.key == pg.K_f:
                self.execute_command("fullscreen")
                return
            if event.key in (pg.K_ESCAPE, pg.K_q):
                if getattr(event, "repeat", False):
                    return
                if event.key == pg.K_ESCAPE and self._fullscreen:
                    self.toggle_fullscreen()
                elif self.cfg.confirm_keyboard_exit:
                    self.menu.close()
                    self.exit_confirmation.open()
                else:
                    self.execute_command("exit")
            elif event.key == pg.K_F11 or (event.key in (pg.K_RETURN, pg.K_KP_ENTER)
                                           and getattr(event, "mod", 0) & pg.KMOD_ALT):
                self.execute_command("fullscreen")
            elif event.key == pg.K_SPACE:
                self.paused = not self.paused
            elif event.key == pg.K_i:
                if self.settings_dialog is None:
                    self.cfg.info_enabled = not self.cfg.info_enabled
                    if self._fullscreen and self.screen.get_size() != clamp_window_size(self.screen.get_size(), self.cfg):
                        self.toggle_fullscreen()
                    else:
                        clamped = clamp_window_size(self.screen.get_size(), self.cfg)
                        if not self._fullscreen and self.screen.get_size() != clamped:
                            self._set_mode(clamped, pg.RESIZABLE)
                            self._windowed_size = clamped
                        self._redraw_requested = True
                        self.renderer.resize(self.screen)
            elif event.key == pg.K_t:
                if self.settings_dialog is None:
                    self.renderer.next_preset()
                    self._redraw_requested = True
            elif event.key == pg.K_F2:
                self.execute_command("settings")

    def execute_command(self, command):
        self.menu.close()
        if command == "settings":
            self.open_settings()
        elif command == "fullscreen":
            self.toggle_fullscreen()
        elif command == "exit":
            self.running = False

    def update_info_text(self):
        # Report actual capture channels, not endpoint capacity or display rows.
        spectrum = self.spectrum
        t = Translator(self.cfg.language)
        device = spectrum.device if spectrum.device is not None else t("app.default_output")
        self.renderer.info_text = t("app.output", device=device, rate=spectrum.sr / 1000,
                                    channels=spectrum.channels_eff)

    def update_now_playing_text(self):
        if hasattr(self, "now_playing") and self.now_playing is not None:
            self.now_playing.enabled = bool(self.cfg.show_now_playing)
        if not self.cfg.show_now_playing:
            self.renderer.now_playing_text = ""
            return
        t = Translator(self.cfg.language)
        current = self.now_playing.current if hasattr(self, "now_playing") and self.now_playing is not None else None
        track_info = current.display_text() if current else ""
        if track_info:
            self.renderer.now_playing_text = t("app.now_playing", track=track_info)
        else:
            self.renderer.now_playing_text = t("app.now_playing_empty")


    def run(self):
        try:
            while self.running:
                dt = self.clock.tick(self.cfg.fps) / 1000.0
                had_events = False
                for event in pg.event.get():
                    had_events = True
                    self.handle_event(event)
                if not self.running:
                    break
                self.poll_settings()

                if not self.paused:
                    self.levels = self.spectrum.step(dt)
                # Keep capture and event/settings polling at their normal cadence.
                # Wait for both the envelope and peak animation to finish, then
                # allow 30 draws for the trail to settle before skipping frames.
                silent = not np.any(self.levels) and not np.any(self.renderer.peak_pos)
                interactive = (had_events or self._redraw_requested or self.paused
                               or self.settings_dialog is not None
                               or self.menu.anchor is not None
                               or self.exit_confirmation.active)
                self._idle_frames = self._idle_frames + 1 if silent and not interactive else 0
                if self._idle_frames > 30 and self._idle_frames % 60 != 0:
                    continue
                # Pause reuses the last levels and freezes peak timers.
                self.update_info_text()
                self.update_now_playing_text()
                self.renderer.draw(self.levels, dt=0.0 if self.paused else dt)
                if self.paused:
                    self.renderer.draw_pause_overlay()
                self.menu.draw(self.screen, self.renderer.font_small, self.cfg.language,
                               self._fullscreen, self.cfg.theme)
                self.exit_confirmation.draw(self.screen, self.renderer.font_small,
                                            self.cfg.language, self.cfg.theme)
                pg.display.flip()
                self._redraw_requested = False
            self.cancel_settings()
            # Apply the explicit exit choice after reverting any uncommitted settings preview.
            if self._disable_exit_confirmation:
                self.cfg.confirm_keyboard_exit = False
            self.save_settings()
        finally:
            if hasattr(self, "now_playing") and self.now_playing is not None:
                self.now_playing.stop()
            if self._settings_window is not None:
                self._settings_window.detach()

            if self.settings_dialog is not None:
                self.settings_dialog.close()
            try:
                self.spectrum.close()
            finally:
                pg.quit()
