"""Application identity and icons shared by source and frozen launches."""
from pathlib import Path
import sys


ASSETS = Path(__file__).resolve().with_name("assets")
APP_ID = "yshdsnd.Wune"


def set_app_id():
    """Keep source launches grouped as Wune instead of the Python interpreter."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        set_id = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        set_id.argtypes = [wintypes.LPCWSTR]
        set_id.restype = ctypes.c_long
        result = set_id(APP_ID)
        if result < 0:
            raise OSError(f"Cannot set Wune application identity: 0x{result & 0xffffffff:08x}")


def pygame_icon():
    """Load the text-free variant, readable in small title bars and taskbars."""
    import pygame as pg
    return pg.image.load(str(ASSETS / "wune-window.png"))


def set_tk_icon(root):
    """Call on the Tk owning thread; defaults also cover child dialogs."""
    if sys.platform == "win32":
        root.iconbitmap(default=str(ASSETS / "Wune.ico"))
    else:
        import tkinter as tk
        root._wune_icon = tk.PhotoImage(master=root, file=str(ASSETS / "wune-window.png"))
        root.iconphoto(True, root._wune_icon)
