"""macOS bundle spec. Run via tools/build_macos.py."""
from pathlib import Path
import json
from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

root = Path(SPECPATH)
datas = collect_data_files("soundcard") + copy_metadata("SoundCard")
datas += [(str(root / "wune" / "locales"), "wune/locales")]
datas += [(str(root / "wune" / "assets"), "wune/assets")]

binaries = [(str(root / "wune" / "libwune_tap.dylib"), "wune")]

a = Analysis(
    [str(root / "packaging" / "entry.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["pygame._sdl2.video", "_cffi_backend", "soundcard.coreaudio"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Replace pygame's historical fallback font with a source-matched GNU FreeFont.
a.datas = [entry for entry in a.datas if entry[0].replace("\\", "/") != "pygame/freesansbold.ttf"]
a.datas += [("pygame/freesansbold.ttf", str(root / "packaging" / "fonts" / "FreeSansBold.ttf"), "DATA")]
(Path(CONF["workpath"]) / "license-inputs.json").write_text(
    json.dumps(list(a.pure) + list(a.binaries) + list(a.datas) + list(a.scripts)), encoding="utf-8")

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Wune",
    icon=str(root / "wune" / "assets" / "Wune.icns"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Wune",
)

app = BUNDLE(
    coll,
    name="Wune.app",
    icon=str(root / "wune" / "assets" / "Wune.icns"),
    bundle_identifier="yshdsnd.Wune",
    info_plist={
        "CFBundleName": "Wune",
        "CFBundleDisplayName": "Wune",
        "CFBundleIdentifier": "yshdsnd.Wune",
        "CFBundlePackageType": "APPL",
        "CFBundleIconFile": "Wune.icns",
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "14.2",
        "NSAudioCaptureUsageDescription": "Wune needs permission to capture system audio to display real-time spectrum visualization.",
        "NSMicrophoneUsageDescription": "Wune needs microphone access to visualize sound input.",
    },
)
