"""Small Tk prompts whose buttons follow the selected application language."""
from tkinter import simpledialog, ttk


class Prompt(simpledialog.Dialog):
    def __init__(self, parent, translator, title, message, initial=None):
        self.t, self.message, self.initial = translator, message, initial
        super().__init__(parent, title)

    def body(self, master):
        ttk.Label(master, text=self.message, wraplength=420).pack(anchor="w", padx=8, pady=8)
        if self.initial is not None:
            self.entry = ttk.Entry(master, width=40)
            self.entry.insert(0, self.initial)
            self.entry.selection_range(0, "end")
            self.entry.pack(fill="x", padx=8, pady=8)
            return self.entry

    def buttonbox(self):
        box = ttk.Frame(self)
        ttk.Button(box, text=self.t("common.ok"), command=self.ok).pack(side="left", padx=5, pady=5)
        ttk.Button(box, text=self.t("settings.cancel"), command=self.cancel).pack(side="left", padx=5, pady=5)
        box.pack()
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def apply(self):
        self.result = self.entry.get() if self.initial is not None else True


def ask_name(parent, translator, title, message, initial):
    return Prompt(parent, translator, title, message, initial).result


def confirm(parent, translator, title, message):
    return bool(Prompt(parent, translator, title, message).result)
