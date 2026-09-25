"""Windows x64 onedir bundle. Run via tools/build_windows.py."""
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

root = Path(SPECPATH)
datas = collect_data_files("soundcard") + copy_metadata("SoundCard")
datas += [(str(root / "wune" / "locales"), "wune/locales")]
a = Analysis(
    [str(root / "packaging" / "entry.py")],
    pathex=[str(root)],
    binaries=[], datas=datas,
    hiddenimports=["pygame._sdl2.video", "_cffi_backend", "soundcard.mediafoundation"],
    hookspath=[], runtime_hooks=[], excludes=[], noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Wune",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Wune")
