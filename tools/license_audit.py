"""Collect notices from the inputs PyInstaller actually redistributes.

Unknown native libraries and missing notices are errors, not best-effort skips.
The checked-in supplement records the provenance of wheel-external notices.
"""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT / "packaging" / "licenses"
MICROSOFT_DLL = r"(?:vcruntime140(?:_1)?|ucrtbase|api-ms-win-[a-z0-9-]+|msvcp140-[a-f0-9]+)\.dll"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_notice(path):
    return any(re.search(r"(^|[_.-])(licen[cs]es?|copying|copyright|notice|[al]?gpl)([.-]|$)",
                         part, re.I) for part in path.parts)


def copy_required(source, target):
    if not source.is_file() or not source.stat().st_size:
        raise RuntimeError(f"Missing or empty license material: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def is_namespace_entry(source, kind):
    # PyInstaller's module graph represents a namespace package without code
    # using '-', not a filesystem path. Its real child modules are audited.
    return source == "-" and kind == "PYMODULE"


def collect_notices(bundle, inputs):
    if sys.version_info[:3] != (3, 13, 14):
        raise RuntimeError("License audit targets Python 3.13.14; review notices before updating Python")
    for item in json.loads((SUPPLEMENT / "provenance.json").read_text(encoding="utf-8")):
        if sha256(SUPPLEMENT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Vendored notice changed: {item['path']}")
    target = bundle / "licenses"
    shutil.copytree(SUPPLEMENT, target)
    entries = json.loads(inputs.read_text(encoding="utf-8"))
    distributions = list(metadata.distributions())
    owners = {}
    for dist in distributions:
        for file in dist.files or ():
            owners[Path(dist.locate_file(file)).resolve()] = dist
    selected = {}
    modules = []
    for name, source, kind in entries:
        if is_namespace_entry(source, kind):
            modules.append({"name": name, "kind": kind, "component": "namespace package (no code)"})
            continue
        path = Path(source).resolve()
        dist = owners.get(path)
        if dist:
            owner = dist.metadata["Name"]
            selected[owner] = dist
        elif path == ROOT / "packaging" / "fonts" / "FreeSansBold.ttf":
            owner = "GNU FreeFont 20120503"
        elif re.fullmatch(MICROSOFT_DLL, path.name.lower()):
            owner = "Microsoft runtime"
        elif path.is_relative_to(Path(sys.base_prefix).resolve()) or name == "base_library.zip":
            owner = "Python"
        elif (path == ROOT / "main.py" or path.is_relative_to(ROOT / "wune")
              or path.is_relative_to(ROOT / "packaging")):
            owner = "Wune"
        elif kind == "SYMLINK":
            continue
        else:
            raise RuntimeError(f"Unidentified PyInstaller input: {name} ({source})")
        modules.append({"name": name, "kind": kind, "component": owner})

    # The bootloader is embedded in Wune.exe, not in Analysis.binaries.
    bootloader = metadata.distribution("PyInstaller")
    selected[bootloader.metadata["Name"]] = bootloader
    components = {}
    pinned = {}
    for requirements in (ROOT / "requirements.txt", ROOT / "requirements-build.txt"):
        for line in requirements.read_text(encoding="utf-8").splitlines():
            if "==" in line:
                package, version = line.split("==")
                pinned[package.lower()] = version
    for name, dist in sorted(selected.items()):
        if pinned.get(name.lower()) != dist.version:
            raise RuntimeError(f"Unreviewed bundled package/version: {name} {dist.version}")
        notices = []
        for file in dist.files or ():
            if is_notice(file):
                relative = Path(*("_" if p == ".." else p for p in file.parts))
                dest = target / name / relative
                copy_required(Path(dist.locate_file(file)), dest)
                notices.append(dest.relative_to(bundle).as_posix())
        if not notices:
            raise RuntimeError(f"No license/notice text found for bundled package {name}")
        components[name] = {"version": dist.version, "notices": sorted(notices)}
        # Setuptools carries separately licensed distributions inside its wheel.
        vendor = Path(dist.locate_file("setuptools/_vendor"))
        if name == "setuptools" and vendor.is_dir():
            components[name]["vendored_distributions"] = [
                {"name": d.metadata["Name"], "version": d.version}
                for d in sorted(metadata.distributions(path=[str(vendor)]), key=lambda d: d.metadata["Name"])]

    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    copy_required(python_license, target / "Python" / "LICENSE.txt")
    tcl = Path(sys.base_prefix) / "tcl"
    # CPython Windows installations omit Tcl's license. The pinned upstream
    # Tcl/Tk texts are in SUPPLEMENT; additionally preserve installation notices.
    for source in tcl.rglob("license*"):
        if source.is_file():
            copy_required(source, target / "Tcl-Tk" / "installation" / source.relative_to(tcl))
    components["Python"] = {"version": sys.version.split()[0], "notices": [
        "licenses/Python/LICENSE.txt", "licenses/Python/history-and-license.rst",
        "licenses/Python/OpenSSL-LICENSE.txt", "licenses/Python/XZ-COPYING",
        "licenses/README.md"]}
    components["Tcl-Tk"] = {"notices": [
        f"licenses/Tcl-Tk/{library}/license.terms" for library in ("tcl8.6", "tk8.6")]}

    binaries = audit_native_files(bundle, components)
    binaries.append({"path": "Wune.exe", "sha256": sha256(bundle / "Wune.exe"),
                     "component": "Wune and PyInstaller bootloader/loader",
                     "notices": ["LICENSE", *components[bootloader.metadata["Name"]]["notices"]]})

    # Preserve MPL-covered setuptools source as well as its nested notices.
    if "setuptools" in selected:
        dist = selected["setuptools"]
        with zipfile.ZipFile(target / "setuptools-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for file in dist.files or ():
                if str(file).startswith("setuptools/") and (file.suffix in (".py", ".json") or is_notice(file)):
                    archive.write(dist.locate_file(file), str(file))

    inventory = {"components": components, "native_files": binaries, "collected_inputs": modules}
    (target / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return inventory


def collect_sources(bundle, cache):
    """Ship exact pinned corresponding sources rather than a future source offer."""
    target = bundle / "licenses" / "sources"
    target.mkdir()
    specs = json.loads((SUPPLEMENT / "sources.json").read_text(encoding="utf-8"))
    cache.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        source = cache / spec["filename"]
        if not source.exists():
            with urllib.request.urlopen(spec["url"], timeout=120) as response:
                source.write_bytes(response.read())
        if sha256(source) != spec["sha256"]:
            raise RuntimeError(f"Source archive checksum mismatch: {source}")
        copy_required(source, target / source.name)
    # Provide the application's relink/rebuild material for LGPL 2.1 section 6(a).
    with zipfile.ZipFile(target / "Wune-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in ("wune", "tools", "packaging", "tests", "docs", ".github"):
            for source in sorted((ROOT / name).rglob("*")):
                if source.is_file() and "__pycache__" not in source.parts:
                    archive.write(source, source.relative_to(ROOT))
        for name in ("LICENSE", "README.md", "Wune.spec", "main.py", "requirements.txt", "requirements-build.txt",
                     ".gitattributes", ".gitignore"):
            archive.write(ROOT / name, name)


def verify_archive(archive, bundle):
    """Check the final ZIP, including exact notice and corresponding-source bytes."""
    with zipfile.ZipFile(archive) as zipped:
        required = [bundle / "LICENSE", *[p for p in (bundle / "licenses").rglob("*") if p.is_file()]]
        for file in required:
            member = file.relative_to(bundle.parent).as_posix()
            if hashlib.sha256(zipped.read(member)).hexdigest() != sha256(file):
                raise RuntimeError(f"Release ZIP license material differs: {member}")
        inventory = bundle / "licenses" / "inventory.json"
        if inventory.exists():
            for entry in json.loads(inventory.read_text(encoding="utf-8"))["native_files"]:
                if hashlib.sha256(zipped.read(bundle.name + "/" + entry["path"])).hexdigest() != entry["sha256"]:
                    raise RuntimeError(f"Release ZIP native file differs: {entry['path']}")


def audit_native_files(bundle, components):
    native = json.loads((bundle / "licenses" / "native-libraries.json").read_text(encoding="utf-8"))
    binaries = []
    python_extensions = set("_asyncio _bz2 _ctypes _decimal _hashlib _lzma _multiprocessing "
                            "_overlapped _queue _socket _ssl _tkinter _wmi pyexpat select unicodedata".split())
    for file in sorted((bundle / "_internal").rglob("*")):
        if file.suffix.lower() not in (".dll", ".pyd", ".ttf"):
            continue
        relative = file.relative_to(bundle).as_posix()
        key = file.name.lower()
        if key in native:
            rule = native[key]
            if sha256(file) != rule["sha256"]:
                raise RuntimeError(f"Native library changed; review its notices: {relative}")
            owner = rule["component"]
            notices = rule["notices"]
        elif file.suffix.lower() == ".pyd" and relative.startswith("_internal/pygame/"):
            owner, notices = "pygame", components["pygame"]["notices"]
        elif file.suffix.lower() == ".pyd" and relative.startswith("_internal/numpy/"):
            owner, notices = "numpy", components["numpy"]["notices"]
        elif key.startswith("_cffi_backend.") and key.endswith(".pyd"):
            owner, notices = "cffi", components["cffi"]["notices"]
        elif key in {name + ".pyd" for name in python_extensions} or key == "python313.dll":
            owner, notices = "Python", components["Python"]["notices"]
        elif key in ("libcrypto-3.dll", "libssl-3.dll", "libffi-8.dll"):
            owner, notices = "Python runtime libraries", components["Python"]["notices"]
        elif key in ("tcl86t.dll", "tk86t.dll"):
            owner, notices = "Tcl-Tk", components["Tcl-Tk"]["notices"]
        elif re.fullmatch(MICROSOFT_DLL, key):
            owner, notices = "Microsoft runtime", ["licenses/Python/LICENSE.txt", "licenses/README.md"]
        elif re.fullmatch(r"libscipy_openblas64_-[a-f0-9]+\.dll", key):
            owner, notices = "OpenBLAS, LAPACK and GCC runtime", components["numpy"]["notices"]
        else:
            raise RuntimeError(f"Unreviewed native library or font: {relative}")
        for notice in notices:
            if not (bundle / notice).is_file() or not (bundle / notice).stat().st_size:
                raise RuntimeError(f"Missing notice for {relative}: {notice}")
        binaries.append({"path": relative, "sha256": sha256(file), "component": owner, "notices": notices})

    return binaries
