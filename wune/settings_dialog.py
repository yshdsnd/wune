"""Tk owns its own thread; pygame/audio only receive immutable draft snapshots.

Native color/name dialogs can run their modal loops without blocking capture.
No Tk widget or pygame object crosses the queues.
"""
from dataclasses import replace
from queue import Empty, Queue
from threading import Thread

from .appearance import AppearanceDraft, COLOR_FIELDS
from .ballistics import MOTION_LIMITS

MOTION_LABELS = {
    "vis_attack_ms": ("立ち上がり時間 (ms)", "小さいほどビートに素早く反応", 1),
    "vis_release_ms": ("下降時間 (ms)", "大きいほどバーの余韻が長い", 1),
    "peak_hold_ms": ("ピーク保持時間 (ms)", "ピーク線を留める時間。次のピークから反映", 1),
    "peak_fall_per_second": ("ピーク落下速度 (表示全幅/秒)", "大きいほど速く落下。0は落下停止", 0.1),
}


COLOR_LABELS = {
    "green_on": "下部LED・点灯", "green_off": "下部LED・消灯",
    "yellow_on": "中部LED・点灯", "yellow_off": "中部LED・消灯",
    "red_on": "上部LED・点灯", "red_off": "上部LED・消灯",
    "background": "背景", "border": "外枠", "led_border": "LED枠",
    "led_outline": "LED輪郭", "highlight": "LED光沢", "shadow": "LED影",
    "peak": "ピーク", "peak_cutout": "ピークの縁", "scale_line": "目盛り線",
    "scale_text": "目盛り文字", "edge_text": "端の目盛り", "db_text": "dB文字",
    "unit_text": "単位", "logo_text": "ロゴ", "badge_text": "テーマ名",
    "badge_glow": "バッジの縁", "badge_background": "バッジ背景",
    "info_background": "情報欄背景", "info_border": "情報欄枠", "info_text": "情報欄文字",
    "pause_text": "一時停止文字", "overlay": "残像オーバーレイ",
}


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
        self.loading = False
        self.pending = False
        root.title("Wune — 表示設定")
        root.resizable(True, True)
        root.minsize(570, 560)
        root.protocol("WM_DELETE_WINDOW", lambda: self.submit("cancel"))
        root.bind("<Escape>", lambda event: self.submit("cancel"))
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="再生中のスペクトラムで見た目を確認できます。", font=("Yu Gothic UI", 11, "bold")).pack(anchor="w")
        notebook = ttk.Notebook(frame)
        self.notebook = notebook
        notebook.pack(fill="both", expand=True, pady=10)
        general = ttk.Frame(notebook, padding=12)
        colors = ttk.Frame(notebook, padding=12)
        notebook.add(general, text="配置・LED")
        notebook.add(colors, text="テーマ・配色")
        motion = ttk.Frame(notebook, padding=12)
        notebook.add(motion, text="動作")
        motion.columnconfigure(0, weight=1)
        self.motion_variables = {}
        self.motion_scales = {}
        for row, (key, (label, hint, increment)) in enumerate(MOTION_LABELS.items()):
            low, high = MOTION_LIMITS[key]
            ttk.Label(motion, text=label).grid(row=row*3, column=0, sticky="w", pady=(8, 0))
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
            ttk.Label(motion, text=f"{hint}（{low:g}～{high:g}）").grid(row=row*3+2, column=0, columnspan=2, sticky="w")
        ttk.Button(motion, text="動きだけ既定に戻す", command=self.reset_motion).grid(row=12, column=0, sticky="w", pady=12)
        ttk.Label(motion, text="数値はEnterまたは入力欄から移動して反映。スライダーは即反映。\n"
                  "取得間隔にも制約があります（4096サンプル / 48 kHz：約85 ms）。\n"
                  "時間設定を短くしても、取得間隔自体は短くなりません。", wraplength=500).grid(row=13, column=0, columnspan=2, sticky="w")
        self.variables = {}
        self.combos = {}

        def choice(parent, row, key, label, options, callback):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=8)
            variable = tk.StringVar(root)
            self.variables[key] = variable
            combo = ttk.Combobox(parent, textvariable=variable, values=list(options), state="readonly", width=32)
            combo.grid(row=row, column=1, sticky="ew", padx=(14, 0))
            combo.bind("<<ComboboxSelected>>", lambda event: callback(options[variable.get()]))
            self.combos[key] = (combo, options)

        general.columnconfigure(1, weight=1)
        choice(general, 0, "spectrum_orientation", "表示方向", {
            "周波数：横 / レベル：縦": "frequency_horizontal", "周波数：縦 / レベル：横": "frequency_vertical"},
            lambda value: self.layout("spectrum_orientation", value))
        choice(general, 1, "channel_layout", "L / R の配置", {"上下": "vertical", "左右": "horizontal"},
            lambda value: self.layout("channel_layout", value))
        choice(general, 2, "gauge_style", "LEDの表現", {"フラット": "flat", "立体（長方形）": "box"},
            lambda value: self.style("gauge_style", value))
        choice(general, 3, "led_shape", "LEDの形", {"長方形": "rectangle", "角丸": "rounded", "楕円": "ellipse"},
            lambda value: self.style("led_shape", value))
        ttk.Label(general, text="LEDの幅 / 高さ").grid(row=4, column=0, sticky="w", pady=8)
        self.ratio = tk.StringVar(root)
        ratio = ttk.Spinbox(general, from_=0.25, to=8, increment=0.25, textvariable=self.ratio,
                            command=self.set_ratio, width=10)
        ratio.grid(row=4, column=1, sticky="w", padx=(14, 0))
        ratio.bind("<Return>", lambda event: self.set_ratio())
        ratio.bind("<FocusOut>", lambda event: self.set_ratio())
        self.info = tk.BooleanVar(root)
        ttk.Checkbutton(general, text="入力情報を表示", variable=self.info,
                        command=lambda: self.layout("info_enabled", self.info.get())).grid(row=5, column=0, sticky="w", pady=8)
        choice(general, 6, "info_position", "情報欄の位置", {"下": "bottom", "上": "top"},
            lambda value: self.layout("info_position", value))
        self.limit_to_20khz = tk.BooleanVar(root)
        ttk.Checkbutton(general, text="表示上限を20 kHzに制限（取得レートは変更しない）",
                        variable=self.limit_to_20khz,
                        command=lambda: self.layout("limit_to_20khz", self.limit_to_20khz.get())).grid(
                            row=7, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Label(general, text="形・比率・配色はテーマに保存します。\n組み込みテーマの編集時はユーザー用コピーを作ります。\n表示方向やLED比率に応じてメインウィンドウの寸法も調整します。",
                  wraplength=500).grid(row=8, column=0, columnspan=2, sticky="w", pady=16)

        theme_row = ttk.Frame(colors)
        theme_row.pack(fill="x")
        ttk.Label(theme_row, text="テーマ").pack(side="left", padx=(0, 12))
        self.theme_name = tk.StringVar(root)
        self.theme_combo = ttk.Combobox(theme_row, textvariable=self.theme_name, state="readonly")
        self.theme_combo.pack(side="left", fill="x", expand=True)
        self.theme_combo.bind("<<ComboboxSelected>>", lambda event: self.select_theme())
        buttons = ttk.Frame(colors)
        buttons.pack(fill="x", pady=8)
        for label, action in (("新規", "new"), ("複製", "copy"), ("名前変更", "rename"), ("削除", "delete")):
            ttk.Button(buttons, text=label, command=lambda a=action: self.manage_theme(a)).pack(side="left", padx=(0, 5))
        table = ttk.Frame(colors)
        table.pack(fill="both", expand=True)
        self.colors = ttk.Treeview(table, columns=("color",), show="tree headings", height=9, selectmode="browse")
        self.colors.heading("#0", text="色の用途")
        self.colors.heading("color", text="RGB")
        self.colors.column("#0", width=220)
        self.colors.column("color", width=100, stretch=False)
        self.colors.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.colors.yview)
        scroll.pack(side="right", fill="y")
        self.colors.configure(yscrollcommand=scroll.set)
        for key in COLOR_FIELDS:
            self.colors.insert("", "end", iid=key, text=COLOR_LABELS[key])
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
        ttk.Button(edit, text="色を反映", command=self.set_hex).pack(side="left", padx=6)
        ttk.Button(edit, text="カラーピッカー…", command=self.pick_color).pack(side="left")
        ttk.Label(colors, text="色選択の確定・「色を反映」で即プレビュー。変更後は保存してください。", wraplength=500).pack(anchor="w")

        self.status = tk.StringVar(root, "変更はプレビュー中です。「保存」で次回の起動にも使用します。")
        ttk.Label(frame, textvariable=self.status, wraplength=540).pack(anchor="w", pady=(0, 8))
        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Button(row, text="表示を既定に戻す", command=self.reset).pack(side="left")
        self.action_buttons = []
        for label, action in (("キャンセル", "cancel"), ("適用", "apply"), ("保存", "save")):
            button = ttk.Button(row, text=label, command=lambda a=action: self.submit(a))
            button.pack(side="right", padx=(6, 0))
            self.action_buttons.append(button)
        ttk.Label(frame, text="適用：現在の変更を確定（正常終了時に保存）\nキャンセル：最後に適用した状態へ戻す。未適用なら開く前へ戻す。", wraplength=540).pack(anchor="w", pady=(8, 0))
        ttk.Label(frame, text=f"保存先: {path}", wraplength=540).pack(anchor="w", pady=(6, 0))
        self.refresh()
        root.after(30, self.poll)

    def refresh(self):
        self.loading = True
        state = self.draft.state
        self.theme_combo.configure(values=self.draft.names())
        self.theme_name.set(state.preset.name)
        for key, (_, choices) in self.combos.items():
            value = state.layout[key] if key in state.layout else getattr(state.preset, key)
            self.variables[key].set(next(label for label, item in choices.items() if item == value))
        self.ratio.set(str(state.preset.led_aspect_ratio))
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
        self.status.set("プレビュー中 — 保存または適用で確定します。")

    def layout(self, key, value):
        self.draft.state.layout[key] = value
        self.preview()

    def style(self, key, value):
        if self.loading or getattr(self.draft.state.preset, key) == value:
            return
        try:
            self.draft.edit(**{key: value})
            self.preview()
        except ValueError as error:
            self.status.set(str(error))

    def set_ratio(self):
        try:
            self.style("led_aspect_ratio", float(self.ratio.get()))
        except ValueError:
            self.status.set("LEDの幅 / 高さは0.25～8の数値で入力してください。")

    def select_theme(self):
        self.draft.select(self.theme_name.get())
        self.preview()

    def manage_theme(self, action):
        from tkinter import messagebox, simpledialog
        try:
            if action == "delete":
                if self.draft.state.preset.name not in self.draft.state.user_presets:
                    raise ValueError("組み込みテーマは削除できません。")
                if not messagebox.askyesno("テーマを削除", "選択中のユーザーテーマを削除しますか？", parent=self.root):
                    return
                self.draft.delete()
            else:
                source = self.draft.state.preset
                if action == "rename" and source.name not in self.draft.state.user_presets:
                    raise ValueError("組み込みテーマは名前を変更できません。複製してください。")
                initial = source.name if action == "rename" else self.draft.available_name("My theme" if action == "new" else source.name + " copy")
                name = simpledialog.askstring("テーマ名", "名前（1～40文字）", initialvalue=initial, parent=self.root)
                if name is None:
                    return
                if action == "rename":
                    self.draft.rename(name)
                else:
                    self.draft.create(name, source if action == "copy" else None)
            self.preview()
        except ValueError as error:
            self.status.set(str(error))

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
            self.status.set("色は #RRGGBB 形式の6桁の16進数で入力してください。")

    def pick_color(self):
        from tkinter import colorchooser
        _, color = colorchooser.askcolor(self.hex_color.get(), parent=self.root, title="色を選択")
        if color:
            self.hex_color.set(color)
            self.set_hex()

    def reset(self):
        self.draft.reset()
        self.preview()
        self.status.set("表示を既定値に戻しました。ユーザーテーマは保持します。キャンセルで取り消せます。")

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
                raise ValueError(f"{MOTION_LABELS[key][0]}は{low:g}～{high:g}で入力してください。") from None
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
            self.status.set(str(error))

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
        self.status.set("動きを既定値に戻しました。保存・適用で確定、キャンセルで取り消せます。")

    def submit(self, action):
        if not self.pending:
            if action != "cancel":
                try:
                    motion = self.read_motion()
                except ValueError as error:
                    self.status.set(str(error))
                    return
                from .settings import valid_preference
                try:
                    ratio = float(self.ratio.get())
                    if not valid_preference("led_aspect_ratio", ratio):
                        raise ValueError()
                except ValueError:
                    self.status.set("LEDの幅 / 高さは0.25～8の数値で入力してください。")
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
                    self.status.set(message)
        except Empty:
            pass
        self.root.after(30, self.poll)
