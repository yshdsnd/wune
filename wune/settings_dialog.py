"""Tk owns a Windows thread or a macOS child process; pygame/audio only receive immutable draft snapshots.

Native color/name dialogs can run their modal loops without blocking capture.
No Tk widget or pygame object crosses the queues.
"""
from dataclasses import replace
from queue import Empty, Queue
from threading import Thread
import sys
import multiprocessing as mp

from .appearance import AppearanceDraft, COLOR_FIELDS
from .ballistics import MOTION_LIMITS
from .i18n import Translator, languages, catalog
from .icons import set_tk_icon

MOTION_LABELS = {
    'vis_attack_ms': (
        'settings.attack_time_ms',
        'settings.lower_values_respond_faster_to_beats', 1),
    'vis_release_ms': (
        'settings.release_time_ms',
        'settings.higher_values_make_bars_decay_more_slowly', 1),
    'peak_hold_ms': (
        'settings.peak_hold_time_ms',
        'settings.time_to_hold_a_peak_applies_to_the_next_peak', 1),
    'peak_fall_per_second': (
        'settings.peak_fall_speed_full_scale_s',
        'settings.higher_values_fall_faster_zero_stops_falling', 0.1),
}


COLOR_LABELS = {key: "color." + key for key in COLOR_FIELDS}


def _run_dialog(state, path, events, commands, window_size=None):
    root = dialog = None
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        dialog = _Dialog(root, AppearanceDraft(state), path, events, commands, window_size=window_size)
        root.update_idletasks()
        # Only Windows transfers a native handle for ownership. Mac Tk lives
        # on the child process's main thread and is activated by that process.
        events.put(("ready", root.winfo_id() if sys.platform == "win32" else None))
        root.mainloop()
    except Exception as error:
        events.put(("error", str(error)))
    finally:
        if dialog is not None:
            dialog.close()
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        dialog = root = None
        import gc
        gc.collect()
        events.put(("closed", None))


def _focus_dialog(root):
    if root.state() != "normal":
        root.deiconify()
    root.lift()
    if sys.platform == "darwin":
        # Activate the settings process rather than the pygame process.
        import ctypes as ct
        try:
            class ProcessSerialNumber(ct.Structure):
                _fields_ = [("high", ct.c_uint32), ("low", ct.c_uint32)]
            api = ct.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
            api.SetFrontProcessWithOptions.argtypes = [ct.POINTER(ProcessSerialNumber), ct.c_uint32]
            api.SetFrontProcessWithOptions.restype = ct.c_int32
            api.SetFrontProcessWithOptions(ct.byref(ProcessSerialNumber(0, 2)), 1)
        except (OSError, AttributeError):
            pass
        root.focus_force()
    else:
        root.focus_set()


class SettingsDialog:
    def __init__(self, state, path, window_size=None):
        self._process = sys.platform in ("win32", "darwin")
        self._closed = False
        if self._process:
            context = mp.get_context("spawn")
            self.events, self.commands = context.Queue(), context.Queue()
            worker = context.Process
        else:
            self.events, self.commands = Queue(), Queue()
            worker = Thread
        self.thread = worker(target=_run_dialog,
                             args=(state, str(path), self.events, self.commands, window_size),
                             daemon=True, name="Wune settings")
        try:
            self.thread.start()
        except BaseException:
            self._release_queues()
            raise

    @property
    def worker_failed(self):
        return self._process and self.thread.exitcode not in (None, 0)

    def _release_queues(self):
        if self._process:
            for queue in (self.events, self.commands):
                queue.cancel_join_thread()
                queue.close()

    def focus(self):
        if not self._closed:
            self.commands.put(("focus", None))

    def update_window_size(self, size):
        if not self._closed:
            self.commands.put(("window_size", size))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.commands.put(("close", None))
            self.thread.join(timeout=1.0)
            if self._process and self.thread.is_alive():
                self.thread.terminate()
                self.thread.join(timeout=1.0)
        finally:
            self._release_queues()

    def reply(self, success, message="", close=False):
        if not self._closed:
            self.commands.put(("reply", (success, message, close)))


class _Dialog:
    def __init__(self, root, draft, path, events, commands, window_size=None):
        import tkinter as tk
        from tkinter import ttk
        self.root, self.draft = root, draft
        self.events, self.commands = events, commands
        self.window_size = window_size or (1280, 800)
        self.t = Translator(draft.state.layout["language"])
        self.loading = False
        self.pending = False
        root.title(self.t('settings.wune_display_settings'))
        set_tk_icon(root)
        root.resizable(True, True)
        root.minsize(570, 560)
        root.protocol("WM_DELETE_WINDOW", lambda: self.submit("cancel"))
        root.bind("<Escape>", lambda event: self.submit("cancel"))
        root.bind("<Deactivate>", self._on_deactivate)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=self.t('settings.preview_changes_on_the_playing_spectrum'), font=("Hiragino Sans" if sys.platform == "darwin" else "Yu Gothic UI", 11, "bold")).pack(anchor="w")
        notebook = ttk.Notebook(frame)
        self.notebook = notebook
        notebook.pack(fill="both", expand=True, pady=10)
        spectrum = ttk.Frame(notebook, padding=12)
        info_tab = ttk.Frame(notebook, padding=12)
        colors = ttk.Frame(notebook, padding=12)
        motion = ttk.Frame(notebook, padding=12)
        background = ttk.Frame(notebook, padding=12)

        notebook.add(spectrum, text=self.t('settings.tab_spectrum'))
        notebook.add(info_tab, text=self.t('settings.tab_info_text'))
        notebook.add(colors, text=self.t('settings.themes_and_colors'))
        notebook.add(motion, text=self.t('settings.motion'))
        notebook.add(background, text=self.t('settings.tab_background_general'))

        motion.columnconfigure(0, weight=1)
        self.motion_variables = {}
        self.motion_scales = {}
        for row, (key, (label, hint, increment)) in enumerate(MOTION_LABELS.items()):
            low, high = MOTION_LIMITS[key]
            ttk.Label(motion, text=self.t(label)).grid(row=row*3, column=0, sticky="w", pady=(6, 0))
            variable = tk.StringVar(root)
            self.motion_variables[key] = variable
            entry = ttk.Spinbox(motion, from_=low, to=high, increment=increment, width=10,
                                textvariable=variable, command=self.set_motion)
            entry.grid(row=row*3, column=1, padx=(12, 0))
            entry.bind("<Return>", lambda event: self.set_motion())
            entry.bind("<FocusOut>", lambda event: self.set_motion())
            slider = ttk.Scale(motion, from_=low, to=high,
                               command=lambda value, k=key: self.slide_motion(k, value))
            slider.grid(row=row*3+1, column=0, columnspan=2, sticky="ew", pady=2)
            self.motion_scales[key] = slider
            ttk.Label(motion, text=self.t("motion.range_hint", hint=self.t(hint), low=low, high=high), wraplength=500).grid(row=row*3+2, column=0, columnspan=2, sticky="w")
        ttk.Button(motion, text=self.t('settings.reset_motion_only'), command=self.reset_motion).grid(row=12, column=0, sticky="w", pady=8)
        ttk.Label(motion, text=self.t('motion.help'), wraplength=500).grid(row=13, column=0, columnspan=2, sticky="w")
        self.variables = {}
        self.combos = {}

        def choice(parent, row, key, label, options, callback):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=8)
            variable = tk.StringVar(root)
            self.variables[key] = variable
            combo = ttk.Combobox(parent, textvariable=variable, values=list(options), state="readonly", width=max(32, max(map(len, options))))
            combo.grid(row=row, column=1, sticky="ew", padx=(14, 0))
            combo.bind("<<ComboboxSelected>>", lambda event: callback(options[variable.get()]))
            self.combos[key] = (combo, options)

        # Tab 0: Spectrum
        spectrum.columnconfigure(1, weight=1)
        choice(spectrum, 0, "channel_mode", self.t("settings.channel_mode"), {
            self.t("settings.stereo_separate"): "stereo",
            self.t("settings.stereo_mix"): "stereo_mix"},
            lambda value: self.layout("channel_mode", value))
        choice(spectrum, 1, "channel_layout", self.t('settings.l_r_arrangement'), {
            self.t('settings.stacked'): "vertical", self.t('settings.side_by_side'): "horizontal"},
            lambda value: self.layout("channel_layout", value))
        choice(spectrum, 2, "spectrum_orientation", self.t('settings.spectrum_direction'), {
            self.t('settings.frequency_horizontal_level_vertical'): "frequency_horizontal",
            self.t('settings.frequency_vertical_level_horizontal'): "frequency_vertical"},
            lambda value: self.layout("spectrum_orientation", value))
        self.limit_to_20khz = tk.BooleanVar(root)
        ttk.Checkbutton(spectrum, text=self.t('settings.limit_display_to_20_khz_keep_capture_rate'),
                        variable=self.limit_to_20khz,
                        command=lambda: self.layout("limit_to_20khz", self.limit_to_20khz.get())).grid(
                            row=3, column=0, columnspan=2, sticky="w", pady=8)
        choice(spectrum, 4, "gauge_style", self.t('settings.led_rendering'), {
            self.t('settings.flat'): "flat", self.t('settings.beveled_rectangle'): "box"},
            lambda value: self.style("gauge_style", value))
        choice(spectrum, 5, "led_shape", self.t('settings.led_shape'), {
            self.t('settings.rectangle'): "rectangle", self.t('settings.rounded'): "rounded", self.t('settings.ellipse'): "ellipse"},
            lambda value: self.style("led_shape", value))
        ttk.Label(spectrum, text=self.t('settings.led_aspect_ratio_width_height')).grid(row=6, column=0, sticky="w", pady=8)
        self.ratio = tk.StringVar(root)
        ratio = ttk.Spinbox(spectrum, from_=0.25, to=8, increment=0.25, textvariable=self.ratio,
                            command=self.set_ratio, width=10)
        ratio.grid(row=6, column=1, sticky="w", padx=(14, 0))
        ratio.bind("<Return>", lambda event: self.set_ratio())
        ratio.bind("<FocusOut>", lambda event: self.set_ratio())
        ttk.Label(spectrum, text=self.t('settings.leds_per_bar')).grid(row=7, column=0, sticky="w", pady=8)
        leds_frame = ttk.Frame(spectrum)
        leds_frame.grid(row=7, column=1, sticky="w", padx=(14, 0))
        self.leds_per_bar = tk.StringVar(root)
        leds_spin = ttk.Spinbox(leds_frame, from_=10, to=100, increment=1, textvariable=self.leds_per_bar,
                                command=self.set_leds_per_bar, width=8)
        leds_spin.pack(side="left")
        leds_spin.bind("<Return>", lambda event: self.set_leds_per_bar())
        leds_spin.bind("<FocusOut>", lambda event: self.set_leds_per_bar())
        self.auto_adjust_button = ttk.Button(leds_frame, text=self.t('settings.auto_adjust_leds'),
                                             command=self.auto_adjust_leds)
        self.auto_adjust_button.pack(side="left", padx=(8, 0))
        self.auto_adjust_leds_on_resize = tk.BooleanVar(root)
        ttk.Checkbutton(spectrum, text=self.t('settings.auto_adjust_leds_on_resize'),
                        variable=self.auto_adjust_leds_on_resize,
                        command=self.toggle_auto_adjust_leds_on_resize).grid(
                            row=8, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(spectrum, text=self.t('settings.bar_gap')).grid(row=9, column=0, sticky="w", pady=8)
        self.bar_gap = tk.StringVar(root)
        bar_gap_spin = ttk.Spinbox(spectrum, from_=0, to=20, increment=1, textvariable=self.bar_gap,
                                   command=self.set_bar_gap, width=8)
        bar_gap_spin.grid(row=9, column=1, sticky="w", padx=(14, 0))
        bar_gap_spin.bind("<Return>", lambda event: self.set_bar_gap())
        bar_gap_spin.bind("<FocusOut>", lambda event: self.set_bar_gap())
        self.adaptive_fill = tk.BooleanVar(root)
        ttk.Checkbutton(spectrum, text=self.t('settings.adaptive_fill'),
                        variable=self.adaptive_fill,
                        command=self.toggle_adaptive_fill).grid(
                            row=10, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(spectrum, text=self.t('settings.spectrum_help'),
                  wraplength=500).grid(row=11, column=0, columnspan=2, sticky="w", pady=12)

        # Tab 1: Info & Text
        info_tab.columnconfigure(1, weight=1)
        ttk.Label(info_tab, text=self.t('settings.label_font_size')).grid(row=0, column=0, sticky="w", pady=8)
        self.label_font_size = tk.StringVar(root)
        label_size_spin = ttk.Spinbox(info_tab, from_=10, to=24, increment=1, textvariable=self.label_font_size,
                                      command=self.set_label_font_size, width=10)
        label_size_spin.grid(row=0, column=1, sticky="w", padx=(14, 0))
        label_size_spin.bind("<Return>", lambda event: self.set_label_font_size())
        label_size_spin.bind("<FocusOut>", lambda event: self.set_label_font_size())
        ttk.Label(info_tab, text=self.t('settings.info_font_size')).grid(row=1, column=0, sticky="w", pady=8)
        self.info_font_size = tk.StringVar(root)
        info_size_spin = ttk.Spinbox(info_tab, from_=10, to=24, increment=1, textvariable=self.info_font_size,
                                     command=self.set_info_font_size, width=10)
        info_size_spin.grid(row=1, column=1, sticky="w", padx=(14, 0))
        info_size_spin.bind("<Return>", lambda event: self.set_info_font_size())
        info_size_spin.bind("<FocusOut>", lambda event: self.set_info_font_size())
        self.auto_scale_fonts = tk.BooleanVar(root)
        ttk.Checkbutton(info_tab, text=self.t('settings.auto_scale_fonts'), variable=self.auto_scale_fonts,
                        command=lambda: self.layout("auto_scale_fonts", self.auto_scale_fonts.get())).grid(
                            row=2, column=0, columnspan=2, sticky="w", pady=8)
        self.show_now_playing = tk.BooleanVar(root)
        ttk.Checkbutton(info_tab, text=self.t('settings.show_now_playing'), variable=self.show_now_playing,
                        command=lambda: self.layout("show_now_playing", self.show_now_playing.get())).grid(
                            row=3, column=0, columnspan=2, sticky="w", pady=8)
        self.info = tk.BooleanVar(root)
        ttk.Checkbutton(info_tab, text=self.t('settings.show_output_information'), variable=self.info,
                        command=lambda: self.layout("info_enabled", self.info.get())).grid(
                            row=4, column=0, sticky="w", pady=8)
        choice(info_tab, 5, "info_position", self.t('settings.information_position'), {
            self.t('settings.bottom'): "bottom", self.t('settings.top'): "top"},
            lambda value: self.layout("info_position", value))
        ttk.Label(info_tab, text=self.t('settings.info_text_help'),
                  wraplength=500).grid(row=6, column=0, columnspan=2, sticky="w", pady=12)

        # Tab 4: Background & General
        background.columnconfigure(1, weight=1)
        choice(background, 0, "background_mode", self.t("background.mode"),
               {self.t("background.solid"): "solid", self.t("background.image"): "image"},
               lambda value: self.background("background_mode", value))
        choice(background, 1, "background_sizing", self.t("background.sizing"),
               {self.t("background.fit"): "fit", self.t("background.fill"): "fill"},
               lambda value: self.background("background_sizing", value))
        self.background_path = tk.StringVar(root)
        ttk.Entry(background, textvariable=self.background_path, state="readonly").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(background, text=self.t("background.choose"), command=self.choose_background).grid(
            row=3, column=0, sticky="w")
        language_options = {self.t("language.auto"): "auto"}
        language_options.update({catalog(code).get("language.name", code): code for code in languages() if code != "auto"})
        choice(background, 4, "language", self.t("settings.language"), language_options,
               lambda value: self.layout("language", value))
        self.confirm_keyboard_exit = tk.BooleanVar()
        ttk.Checkbutton(background, text=self.t("exit.confirm_setting"),
                        variable=self.confirm_keyboard_exit,
                        command=lambda: self.layout("confirm_keyboard_exit", self.confirm_keyboard_exit.get())).grid(
                            row=5, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Label(background, text=self.t('settings.window_size_preset')).grid(row=6, column=0, sticky="w", pady=8)
        preset_frame = ttk.Frame(background)
        preset_frame.grid(row=6, column=1, sticky="w", padx=(14, 0))
        self.window_preset_var = tk.StringVar(root)
        self.window_preset_combo = ttk.Combobox(preset_frame, textvariable=self.window_preset_var, state="readonly", width=24)
        self.window_preset_combo.pack(side="left")
        ttk.Button(preset_frame, text=self.t('settings.apply_window_preset'),
                   command=self.apply_selected_window_preset).pack(side="left", padx=(8, 0))
        ttk.Label(background, text=self.t("background.help"), wraplength=500).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=12)

        theme_row = ttk.Frame(colors)
        theme_row.pack(fill="x")
        ttk.Label(theme_row, text=self.t('settings.theme')).pack(side="left", padx=(0, 12))
        self.theme_name = tk.StringVar(root)
        self.theme_combo = ttk.Combobox(theme_row, textvariable=self.theme_name, state="readonly")
        self.theme_combo.pack(side="left", fill="x", expand=True)
        self.theme_combo.bind("<<ComboboxSelected>>", lambda event: self.select_theme())
        buttons = ttk.Frame(colors)
        buttons.pack(fill="x", pady=8)
        for label, action in ((self.t('settings.new'), "new"), (self.t('settings.duplicate'), "copy"), (self.t('settings.rename'), "rename"), (self.t('settings.delete'), "delete")):
            ttk.Button(buttons, text=label, command=lambda a=action: self.manage_theme(a)).pack(side="left", padx=(0, 5))
        table = ttk.Frame(colors)
        table.pack(fill="both", expand=True)
        self.colors = ttk.Treeview(table, columns=("color",), show="tree headings", height=9, selectmode="browse")
        self.colors.heading("#0", text=self.t('settings.color_role'))
        self.colors.heading("color", text="RGB")
        self.colors.column("#0", width=220)
        self.colors.column("color", width=100, stretch=False)
        self.colors.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.colors.yview)
        scroll.pack(side="right", fill="y")
        self.colors.configure(yscrollcommand=scroll.set)
        for key in COLOR_FIELDS:
            self.colors.insert("", "end", iid=key, text=self.t(COLOR_LABELS[key]))
        self.colors.selection_set("green_on")
        self.colors.bind("<<TreeviewSelect>>", lambda event: self.refresh_color())
        self.colors.bind("<Double-1>", lambda event: self.pick_color())
        edit = ttk.Frame(colors)
        edit.pack(fill="x", pady=8)
        self.swatch = tk.Label(edit, width=4, relief="sunken")
        self.swatch.pack(side="left", padx=(0, 8))
        self.hex_color = tk.StringVar(root)
        self.hex_entry = ttk.Entry(edit, width=10, textvariable=self.hex_color)
        self.hex_entry.pack(side="left")
        self.hex_entry.bind("<Return>", lambda event: self.set_hex())
        ttk.Button(edit, text=self.t('settings.apply_color'), command=self.set_hex).pack(side="left", padx=6)
        ttk.Button(edit, text=self.t('settings.color_picker'), command=self.pick_color).pack(side="left")
        ttk.Label(colors, text=self.t('colors.help'), wraplength=500).pack(anchor="w")

        self.status = tk.StringVar(root, self.t('settings.previewing_changes_save_to_keep_them_for_the_next_launch'))
        ttk.Label(frame, textvariable=self.status, wraplength=540).pack(anchor="w", pady=(0, 8))
        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Button(row, text=self.t('settings.reset_display'), command=self.reset).pack(side="left")
        self.action_buttons = []
        for label, action in ((self.t('settings.cancel'), "cancel"), (self.t('settings.apply'), "apply"), (self.t('settings.save'), "save")):
            button = ttk.Button(row, text=label, command=lambda a=action: self.submit(a))
            button.pack(side="right", padx=(6, 0))
            self.action_buttons.append(button)
        ttk.Label(frame, text=self.t('settings.actions_help'), wraplength=540).pack(anchor="w", pady=(8, 0))
        ttk.Label(frame, text=self.t("settings.path", path=path), wraplength=540).pack(anchor="w", pady=(6, 0))
        self.refresh()
        root.update_idletasks()
        root.minsize(max(560, root.winfo_reqwidth()), max(480, root.winfo_reqheight()))
        self._poll_id = root.after(30, self.poll)

    def refresh(self):
        self.loading = True
        state = self.draft.state
        self.theme_combo.configure(values=self.draft.names())
        self.theme_name.set(state.preset.name)
        for key, (_, choices) in self.combos.items():
            value = (state.background[key] if key in state.background else
                     state.layout[key] if key in state.layout else state.style[key])
            self.variables[key].set(next(label for label, item in choices.items() if item == value))
        self.ratio.set(str(state.style["led_aspect_ratio"]))
        self.leds_per_bar.set(str(state.style.get("leds_per_bar", 20)))
        self.auto_adjust_leds_on_resize.set(bool(state.style.get("auto_adjust_leds_on_resize", False)))
        self.bar_gap.set(str(state.style.get("bar_gap", 2)))
        self.adaptive_fill.set(bool(state.style.get("adaptive_fill", False)))
        self.combos["channel_layout"][0].configure(
            state="disabled" if state.layout["channel_mode"] == "stereo_mix" else "readonly")
        self.background_path.set(state.background["background_path"])
        self.confirm_keyboard_exit.set(state.layout["confirm_keyboard_exit"])
        self.info.set(state.layout["info_enabled"])
        self.show_now_playing.set(state.layout.get("show_now_playing", True))
        self.limit_to_20khz.set(state.layout["limit_to_20khz"])
        self.label_font_size.set(str(state.layout.get("label_font_size", 14)))
        self.info_font_size.set(str(state.layout.get("info_font_size", 14)))
        self.auto_scale_fonts.set(state.layout.get("auto_scale_fonts", True))
        self.update_window_preset_choices()
        for key, value in state.motion.items():
            self.motion_variables[key].set(f"{value:g}")
            self.motion_scales[key].set(value)
        for key in COLOR_FIELDS:
            self.colors.item(key, values=(self.color_hex(key),))
        self.refresh_color()
        self.loading = False

    def color_hex(self, key):
        return "#%02X%02X%02X" % getattr(self.draft.state.preset.theme, key)

    def refresh_color(self):
        selection = self.colors.selection()
        if selection:
            color = self.color_hex(selection[0])
            self.hex_color.set(color)
            self.swatch.configure(background=color)

    def preview(self):
        self.refresh()
        self.events.put(("preview", self.draft.snapshot()))
        self.status.set(self.t('settings.previewing_save_or_apply_to_confirm'))

    def layout(self, key, value):
        self.draft.state.layout[key] = value
        if key in ("channel_layout", "channel_mode") and getattr(self, "auto_adjust_leds_on_resize", None) and self.auto_adjust_leds_on_resize.get():
            self.auto_adjust_leds()
        else:
            self.preview()

    def style(self, key, value):
        if self.loading or self.draft.state.style[key] == value:
            return
        try:
            self.draft.edit_style(**{key: value})
            self.preview()
        except ValueError as error:
            self.status.set(self.t.error(error))

    def set_ratio(self):
        try:
            self.style("led_aspect_ratio", float(self.ratio.get()))
        except ValueError:
            self.status.set(self.t('settings.enter_an_led_aspect_ratio_between_0_25_and_8'))

    def set_leds_per_bar(self):
        from .settings import valid_preference
        try:
            val = int(self.leds_per_bar.get())
            if not valid_preference("leds_per_bar", val):
                raise ValueError()
            self.style("leds_per_bar", val)
        except ValueError:
            self.status.set(self.t('settings.enter_leds_per_bar_between_10_and_100'))

    def set_bar_gap(self):
        from .settings import valid_preference
        try:
            val = int(self.bar_gap.get())
            if not valid_preference("bar_gap", val):
                raise ValueError()
            self.style("bar_gap", val)
        except ValueError:
            self.status.set(self.t('settings.enter_a_bar_gap_between_0_and_20'))

    def auto_adjust_leds(self):
        from .layout import calculate_optimal_leds_per_bar
        from .config import Config
        cfg = Config()
        self.draft.state.apply(cfg)
        size = getattr(self, "window_size", None) or (1280, 800)
        optimal = calculate_optimal_leds_per_bar(size, cfg)
        self.leds_per_bar.set(str(optimal))
        self.set_leds_per_bar()
        self.status.set(self.t('settings.auto_adjusted_leds_to', count=optimal))

    def toggle_auto_adjust_leds_on_resize(self):
        if self.loading:
            return
        enabled = bool(self.auto_adjust_leds_on_resize.get())
        self.draft.state.style["auto_adjust_leds_on_resize"] = enabled
        if enabled:
            self.auto_adjust_leds()
        else:
            self.preview()

    def toggle_adaptive_fill(self):
        if self.loading:
            return
        enabled = bool(self.adaptive_fill.get())
        self.draft.state.style["adaptive_fill"] = enabled
        self.preview()

    def update_window_preset_choices(self):
        if not hasattr(self, "window_preset_combo"):
            return
        from .window_presets import get_all_window_presets
        user_presets = getattr(self.draft.state, "user_window_presets", None)
        presets = get_all_window_presets(user_presets)
        self.preset_map = {}
        labels = []
        current_w, current_h = getattr(self, "window_size", (1280, 800))
        selected_label = None
        for p in presets:
            label_name = self.t(p.name_key) if getattr(p, "name_key", None) else p.name
            label = f"{label_name} ({p.width}x{p.height})"
            self.preset_map[label] = p.id
            labels.append(label)
            if p.width == current_w and p.height == current_h and selected_label is None:
                selected_label = label
        self.window_preset_combo.configure(values=labels)
        if selected_label:
            self.window_preset_var.set(selected_label)
        elif labels and not self.window_preset_var.get():
            self.window_preset_var.set(labels[0])

    def apply_selected_window_preset(self):
        if self.loading or self.pending:
            return
        from .window_presets import find_window_preset
        label = self.window_preset_var.get()
        preset_id = getattr(self, "preset_map", {}).get(label)
        if not preset_id:
            return
        preset = find_window_preset(preset_id, getattr(self.draft.state, "user_window_presets", None))
        if preset is not None:
            self.window_size = (preset.width, preset.height)
            if preset.leds_per_bar is not None:
                self.draft.state.style["leds_per_bar"] = preset.leds_per_bar
            if preset.led_aspect_ratio is not None:
                self.draft.state.style["led_aspect_ratio"] = preset.led_aspect_ratio
            if preset.bar_gap is not None:
                self.draft.state.style["bar_gap"] = preset.bar_gap
            if preset.adaptive_fill is not None:
                self.draft.state.style["adaptive_fill"] = preset.adaptive_fill
            self.refresh()
        self.events.put(("window_preset", preset_id))

    def set_label_font_size(self):
        from .settings import valid_preference
        try:
            val = int(self.label_font_size.get())
            if not valid_preference("label_font_size", val):
                raise ValueError()
            self.layout("label_font_size", val)
        except ValueError:
            self.status.set(self.t('settings.enter_a_label_font_size_between_10_and_24'))

    def set_info_font_size(self):
        from .settings import valid_preference
        try:
            val = int(self.info_font_size.get())
            if not valid_preference("info_font_size", val):
                raise ValueError()
            self.layout("info_font_size", val)
        except ValueError:
            self.status.set(self.t('settings.enter_an_info_font_size_between_10_and_24'))

    def background(self, key, value):
        if self.loading or self.pending:
            return
        self.draft.edit_background(**{key: value})
        self.preview()

    def choose_background(self):
        from tkinter import filedialog
        from pathlib import Path
        path = filedialog.askopenfilename(parent=self.root, title=self.t("background.choose"),
            filetypes=[(self.t("background.images"), "*.png *.jpg *.jpeg"),
                       (self.t("background.all_files"), "*.*")])
        if path:
            self.draft.edit_background(background_mode="image", background_path=str(Path(path).resolve()))
            self.preview()

    def select_theme(self):
        self.draft.select(self.theme_name.get())
        self.preview()

    def manage_theme(self, action):
        from .localized_dialogs import confirm, ask_name
        try:
            if action == "delete":
                if self.draft.state.preset.name not in self.draft.state.user_presets:
                    raise ValueError(self.t('settings.built_in_themes_cannot_be_deleted'))
                if not confirm(self.root, self.t, self.t('settings.delete_theme'), self.t('settings.delete_the_selected_user_theme')):
                    return
                self.draft.delete()
            else:
                source = self.draft.state.preset
                if action == "rename" and source.name not in self.draft.state.user_presets:
                    raise ValueError(self.t('settings.duplicate_a_built_in_theme_before_renaming_it'))
                initial = source.name if action == "rename" else self.draft.available_name("My theme" if action == "new" else source.name + " copy")
                name = ask_name(self.root, self.t, self.t('settings.theme_name'), self.t('settings.name_1_40_characters'), initial=initial)
                if name is None:
                    return
                if action == "rename":
                    self.draft.rename(name)
                else:
                    self.draft.create(name, source if action == "copy" else None)
            self.preview()
        except ValueError as error:
            self.status.set(self.t.error(error))

    def set_hex(self):
        text = self.hex_color.get().strip().lstrip("#")
        try:
            if len(text) != 6:
                raise ValueError()
            rgb = tuple(int(text[i:i+2], 16) for i in (0, 2, 4))
            key = self.colors.selection()[0]
            theme = replace(self.draft.state.preset.theme, **{key: rgb})
            self.draft.edit(theme=theme)
            self.preview()
        except (ValueError, IndexError):
            self.status.set(self.t('settings.enter_six_hexadecimal_digits_in_rrggbb_format'))

    def pick_color(self):
        from tkinter import colorchooser
        _, color = colorchooser.askcolor(self.hex_color.get(), parent=self.root, title=self.t('settings.select_color'))
        if color:
            self.hex_color.set(color)
            self.set_hex()

    def reset(self):
        self.draft.reset()
        self.preview()
        self.status.set(self.t('settings.display_defaults_restored_user_themes_are_kept_cancel_to_undo'))

    def read_motion(self):
        from .ballistics import valid_motion
        values = {}
        for key, variable in self.motion_variables.items():
            low, high = MOTION_LIMITS[key]
            try:
                value = float(variable.get())
                if not valid_motion(key, value):
                    raise ValueError()
            except ValueError:
                raise ValueError(self.t("error.motion_range", label=self.t(MOTION_LABELS[key][0]), low=low, high=high)) from None
            values[key] = value
        return values

    def set_motion(self):
        if self.loading or self.pending:
            return
        try:
            values = self.read_motion()
            if values != self.draft.state.motion:
                self.draft.edit_motion(values)
                self.preview()
        except ValueError as error:
            self.status.set(self.t.error(error))

    def slide_motion(self, key, value):
        if self.loading or self.pending:
            return
        increment = MOTION_LABELS[key][2]
        value = round(round(float(value) / increment) * increment, 1)
        if value != self.draft.state.motion[key]:
            self.draft.edit_motion({key: value})
            self.preview()

    def reset_motion(self):
        self.draft.reset_motion()
        self.preview()
        self.status.set(self.t('settings.motion_defaults_restored_save_or_apply_to_confirm_cancel_to_undo'))

    def submit(self, action):
        if not self.pending:
            if action != "cancel":
                try:
                    motion = self.read_motion()
                except ValueError as error:
                    self.status.set(self.t.error(error))
                    return
                from .settings import valid_preference
                try:
                    ratio = float(self.ratio.get())
                    if not valid_preference("led_aspect_ratio", ratio):
                        raise ValueError()
                except ValueError:
                    self.status.set(self.t('settings.enter_an_led_aspect_ratio_between_0_25_and_8'))
                    return
                try:
                    leds = int(self.leds_per_bar.get())
                    if not valid_preference("leds_per_bar", leds):
                        raise ValueError()
                except ValueError:
                    self.status.set(self.t('settings.enter_leds_per_bar_between_10_and_100'))
                    return
                try:
                    font_size = int(self.label_font_size.get())
                    if not valid_preference("label_font_size", font_size):
                        raise ValueError()
                except ValueError:
                    self.status.set(self.t('settings.enter_a_label_font_size_between_10_and_24'))
                    return
                try:
                    info_size = int(self.info_font_size.get())
                    if not valid_preference("info_font_size", info_size):
                        raise ValueError()
                except ValueError:
                    self.status.set(self.t('settings.enter_an_info_font_size_between_10_and_24'))
                    return
                try:
                    bar_gap = int(self.bar_gap.get())
                    if not valid_preference("bar_gap", bar_gap):
                        raise ValueError()
                except ValueError:
                    self.status.set(self.t('settings.enter_a_bar_gap_between_0_and_20'))
                    return
                self.draft.edit_motion(motion)
                self.draft.edit_style(
                    led_aspect_ratio=ratio,
                    leds_per_bar=leds,
                    auto_adjust_leds_on_resize=bool(self.auto_adjust_leds_on_resize.get()),
                    bar_gap=bar_gap,
                    adaptive_fill=bool(self.adaptive_fill.get()),
                )
                self.draft.state.layout["label_font_size"] = font_size
                self.draft.state.layout["info_font_size"] = info_size
            self.pending = True
            for tab in self.notebook.tabs():
                self.notebook.tab(tab, state="disabled")
            for button in self.action_buttons:
                button.configure(state="disabled")
            self.events.put((action, self.draft.snapshot()))

    def _on_deactivate(self, event=None):
        if event is None or event.widget == self.root:
            self.unpost_combos()

    def unpost_combos(self):
        for combo, _ in self.combos.values():
            try:
                combo.tk.eval(f"ttk::combobox::Unpost {combo._w}")
            except Exception:
                pass
        try:
            current_grab = self.root.tk.eval("grab current")
            if current_grab:
                self.root.tk.eval(f"grab release {current_grab}")
        except Exception:
            pass

    def close(self):
        self.unpost_combos()
        if hasattr(self, "_poll_id") and self._poll_id is not None:
            try:
                self.root.after_cancel(self._poll_id)
            except Exception:
                pass
            self._poll_id = None

    def poll(self):
        try:
            if not self.root.winfo_exists():
                return
            while True:
                action, payload = self.commands.get_nowait()
                if action == "close":
                    self.close()
                    self.root.destroy()
                    return
                if action == "focus":
                    _focus_dialog(self.root)
                elif action == "window_size":
                    self.window_size = payload
                    self.update_window_preset_choices()
                    if getattr(self, "auto_adjust_leds_on_resize", None) and self.auto_adjust_leds_on_resize.get():
                        self.auto_adjust_leds()
                elif action == "reply":
                    success, message, close = payload
                    if success and close:
                        self.close()
                        self.root.destroy()
                        return
                    self.pending = False
                    for tab in self.notebook.tabs():
                        self.notebook.tab(tab, state="normal")
                    for button in self.action_buttons:
                        button.configure(state="normal")
                    self.status.set(self.t(message))
        except Empty:
            pass
        except Exception:
            return
        try:
            if self.root.winfo_exists():
                self._poll_id = self.root.after(30, self.poll)
        except Exception:
            self._poll_id = None
