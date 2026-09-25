"""Tk owns its own thread; pygame/audio only receive immutable draft snapshots.

Native color/name dialogs can run their modal loops without blocking capture.
No Tk widget or pygame object crosses the queues.
"""
from dataclasses import replace
from queue import Empty, Queue
from threading import Thread

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


class SettingsDialog:
    def __init__(self, state, path):
        self.events = Queue()
        self.commands = Queue()
        self.thread = Thread(target=self._run, args=(state, str(path)), daemon=True, name="Wune settings")
        self.thread.start()

    def _run(self, state, path):
        root = dialog = None
        try:
            import tkinter as tk
            root = tk.Tk()
            dialog = _Dialog(root, AppearanceDraft(state), path, self.events, self.commands)
            root.mainloop()
        except Exception as error:
            self.events.put(("error", str(error)))
        finally:
            if root is not None:
                try:
                    root.destroy()
                except Exception:
                    pass
            # Tcl objects must be finalized on the thread that created them.
            dialog = root = None
            import gc
            gc.collect()
            self.events.put(("closed", None))

    def focus(self):
        self.commands.put(("focus", None))

    def close(self):
        self.commands.put(("close", None))
        self.thread.join(timeout=1.0)

    def reply(self, success, message="", close=False):
        self.commands.put(("reply", (success, message, close)))


class _Dialog:
    def __init__(self, root, draft, path, events, commands):
        import tkinter as tk
        from tkinter import ttk
        self.root, self.draft = root, draft
        self.events, self.commands = events, commands
        self.t = Translator(draft.state.layout["language"])
        self.loading = False
        self.pending = False
        root.title(self.t('settings.wune_display_settings'))
        set_tk_icon(root)
        root.resizable(True, True)
        root.minsize(570, 560)
        root.protocol("WM_DELETE_WINDOW", lambda: self.submit("cancel"))
        root.bind("<Escape>", lambda event: self.submit("cancel"))
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=self.t('settings.preview_changes_on_the_playing_spectrum'), font=("Yu Gothic UI", 11, "bold")).pack(anchor="w")
        notebook = ttk.Notebook(frame)
        self.notebook = notebook
        notebook.pack(fill="both", expand=True, pady=10)
        general = ttk.Frame(notebook, padding=12)
        colors = ttk.Frame(notebook, padding=12)
        notebook.add(general, text=self.t('settings.layout_and_leds'))
        notebook.add(colors, text=self.t('settings.themes_and_colors'))
        motion = ttk.Frame(notebook, padding=12)
        notebook.add(motion, text=self.t('settings.motion'))
        motion.columnconfigure(0, weight=1)
        self.motion_variables = {}
        self.motion_scales = {}
        for row, (key, (label, hint, increment)) in enumerate(MOTION_LABELS.items()):
            low, high = MOTION_LIMITS[key]
            ttk.Label(motion, text=self.t(label)).grid(row=row*3, column=0, sticky="w", pady=(8, 0))
            variable = tk.StringVar(root)
            self.motion_variables[key] = variable
            entry = ttk.Spinbox(motion, from_=low, to=high, increment=increment, width=10,
                                textvariable=variable, command=self.set_motion)
            entry.grid(row=row*3, column=1, padx=(12, 0))
            entry.bind("<Return>", lambda event: self.set_motion())
            entry.bind("<FocusOut>", lambda event: self.set_motion())
            slider = ttk.Scale(motion, from_=low, to=high,
                               command=lambda value, k=key: self.slide_motion(k, value))
            slider.grid(row=row*3+1, column=0, columnspan=2, sticky="ew", pady=4)
            self.motion_scales[key] = slider
            ttk.Label(motion, text=self.t("motion.range_hint", hint=self.t(hint), low=low, high=high), wraplength=500).grid(row=row*3+2, column=0, columnspan=2, sticky="w")
        ttk.Button(motion, text=self.t('settings.reset_motion_only'), command=self.reset_motion).grid(row=12, column=0, sticky="w", pady=12)
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

        language_options = {self.t("language.auto"): "auto"}
        language_options.update({catalog(code).get("language.name", code): code for code in languages() if code != "auto"})
        choice(general, 9, "language", self.t("settings.language"), language_options,
               lambda value: self.layout("language", value))
        ttk.Label(general, text=self.t("language.help"), wraplength=500).grid(
            row=10, column=0, columnspan=2, sticky="w", pady=8)
        general.columnconfigure(1, weight=1)
        choice(general, 0, "spectrum_orientation", self.t('settings.spectrum_direction'), {
            self.t('settings.frequency_horizontal_level_vertical'): "frequency_horizontal", self.t('settings.frequency_vertical_level_horizontal'): "frequency_vertical"},
            lambda value: self.layout("spectrum_orientation", value))
        choice(general, 1, "channel_layout", self.t('settings.l_r_arrangement'), {self.t('settings.stacked'): "vertical", self.t('settings.side_by_side'): "horizontal"},
            lambda value: self.layout("channel_layout", value))
        choice(general, 2, "gauge_style", self.t('settings.led_rendering'), {self.t('settings.flat'): "flat", self.t('settings.beveled_rectangle'): "box"},
            lambda value: self.style("gauge_style", value))
        choice(general, 3, "led_shape", self.t('settings.led_shape'), {self.t('settings.rectangle'): "rectangle", self.t('settings.rounded'): "rounded", self.t('settings.ellipse'): "ellipse"},
            lambda value: self.style("led_shape", value))
        ttk.Label(general, text=self.t('settings.led_aspect_ratio_width_height')).grid(row=4, column=0, sticky="w", pady=8)
        self.ratio = tk.StringVar(root)
        ratio = ttk.Spinbox(general, from_=0.25, to=8, increment=0.25, textvariable=self.ratio,
                            command=self.set_ratio, width=10)
        ratio.grid(row=4, column=1, sticky="w", padx=(14, 0))
        ratio.bind("<Return>", lambda event: self.set_ratio())
        ratio.bind("<FocusOut>", lambda event: self.set_ratio())
        self.info = tk.BooleanVar(root)
        ttk.Checkbutton(general, text=self.t('settings.show_output_information'), variable=self.info,
                        command=lambda: self.layout("info_enabled", self.info.get())).grid(row=5, column=0, sticky="w", pady=8)
        choice(general, 6, "info_position", self.t('settings.information_position'), {self.t('settings.bottom'): "bottom", self.t('settings.top'): "top"},
            lambda value: self.layout("info_position", value))
        self.limit_to_20khz = tk.BooleanVar(root)
        ttk.Checkbutton(general, text=self.t('settings.limit_display_to_20_khz_keep_capture_rate'),
                        variable=self.limit_to_20khz,
                        command=lambda: self.layout("limit_to_20khz", self.limit_to_20khz.get())).grid(
                            row=7, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Label(general, text=self.t('settings.layout_help'),
                  wraplength=500).grid(row=8, column=0, columnspan=2, sticky="w", pady=16)

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
        root.minsize(max(570, root.winfo_reqwidth()), max(560, root.winfo_reqheight()))
        root.after(30, self.poll)

    def refresh(self):
        self.loading = True
        state = self.draft.state
        self.theme_combo.configure(values=self.draft.names())
        self.theme_name.set(state.preset.name)
        for key, (_, choices) in self.combos.items():
            value = state.layout[key] if key in state.layout else state.style[key]
            self.variables[key].set(next(label for label, item in choices.items() if item == value))
        self.ratio.set(str(state.style["led_aspect_ratio"]))
        self.info.set(state.layout["info_enabled"])
        self.limit_to_20khz.set(state.layout["limit_to_20khz"])
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
                self.draft.edit_motion(motion)
                self.style("led_aspect_ratio", ratio)
            self.pending = True
            for tab in self.notebook.tabs():
                self.notebook.tab(tab, state="disabled")
            for button in self.action_buttons:
                button.configure(state="disabled")
            self.events.put((action, self.draft.snapshot()))

    def poll(self):
        try:
            while True:
                action, payload = self.commands.get_nowait()
                if action == "close":
                    self.root.destroy()
                    return
                if action == "focus":
                    self.root.deiconify()
                    self.root.lift()
                elif action == "reply":
                    success, message, close = payload
                    if success and close:
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
        self.root.after(30, self.poll)
