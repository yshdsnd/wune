"""Windows x64 onedir bundle. Run via tools/build_windows.py."""
from pathlib import Path
import json
from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

root = Path(SPECPATH)
datas = collect_data_files("soundcard") + copy_metadata("SoundCard")
datas += [(str(root / "wune" / "locales"), "wune/locales")]
datas += [(str(root / "wune" / "assets"), "wune/assets")]
a = Analysis(
    [str(root / "packaging" / "entry.py")],
    pathex=[str(root)],
    binaries=[], datas=datas,
    hiddenimports=["pygame._sdl2.video", "_cffi_backend", "soundcard.mediafoundation"],
    hookspath=[], runtime_hooks=[], excludes=[], noarchive=False,
)
# Replace pygame's historical fallback font with a source-matched GNU FreeFont.
# System font selection is unchanged; only pygame's packaged fallback changes.
a.datas = [entry for entry in a.datas if entry[0].replace("\\", "/") != "pygame/freesansbold.ttf"]
a.datas += [("pygame/freesansbold.ttf", str(root / "packaging" / "fonts" / "FreeSansBold.ttf"), "DATA")]
# Retain provenance before PYZ embeds Python modules inside the executable.
(Path(CONF["workpath"]) / "license-inputs.json").write_text(
    json.dumps(list(a.pure) + list(a.binaries) + list(a.datas) + list(a.scripts)), encoding="utf-8")
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Wune",
          icon=str(root / "wune" / "assets" / "Wune.ico"),
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Wune")
