"""Collect notices and an inventory for Linux standalone packaging.

Wheel notices (including their vendored-library notices) are retained in full.
"""
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys
import zipfile

try:
    from tools.license_audit import ROOT, SUPPLEMENT, copy_required, is_notice, sha256
except ModuleNotFoundError:
    from license_audit import ROOT, SUPPLEMENT, copy_required, is_notice, sha256


def collect_notices(bundle: Path, inputs: Path) -> dict:
    target = bundle / "licenses"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(SUPPLEMENT, target)
    for item in json.loads((SUPPLEMENT / "provenance.json").read_text(encoding="utf-8")):
        if sha256(SUPPLEMENT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Vendored notice changed: {item['path']}")
    owners = {}
    for dist in metadata.distributions():
        for file in dist.files or ():
            owners[Path(dist.locate_file(file)).resolve()] = dist
    entries = json.loads(inputs.read_text(encoding="utf-8")) if inputs.is_file() else []
    selected = {}
    try:
        selected["PyInstaller"] = metadata.distribution("PyInstaller")
    except metadata.PackageNotFoundError:
        pass
    for _, source, _ in entries:
        dist = owners.get(Path(source).resolve())
        if dist is not None:
            selected[dist.metadata["Name"]] = dist
    pinned = {}
    for requirements in (ROOT / "requirements.txt", ROOT / "requirements-build.txt"):
        for raw_line in requirements.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            spec = line.split(";", 1)[0].strip()
            if "==" in spec:
                name, version = spec.split("==", 1)
                pinned[name.strip().lower()] = version.strip()
    components = {}
    for name, dist in sorted(selected.items()):
        if pinned.get(name.lower()) is not None and pinned.get(name.lower()) != dist.version:
            raise RuntimeError(f"Unreviewed bundled package/version: {name} {dist.version}")
        notices = []
        for file in dist.files or ():
            if is_notice(file):
                relative = Path(*("_" if part == ".." else part for part in file.parts))
                dest = target / name / relative
                copy_required(Path(dist.locate_file(file)), dest)
                notices.append(dest.relative_to(bundle).as_posix())
        if not notices:
            raise RuntimeError(f"Missing bundled package notices: {name}")
        components[name] = {"version": dist.version, "notices": sorted(notices)}
        if name.lower() == "setuptools":
            with zipfile.ZipFile(target / "setuptools-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
                for file in dist.files or ():
                    if str(file).startswith("setuptools/") and (file.suffix in (".py", ".json") or is_notice(file)):
                        archive.write(dist.locate_file(file), str(file))
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        copy_required(python_license, target / "Python" / "LICENSE.txt")
    components["Python"] = {
        "version": sys.version.split()[0],
        "notices": ["licenses/Python/history-and-license.rst"],
    }
    files = [
        {"path": p.relative_to(bundle).as_posix(), "sha256": sha256(p)}
        for p in sorted(bundle.rglob("*")) if p.is_file() and not p.is_symlink()
    ]
    inventory = {
        "platform": "Linux",
        "components": components,
        "bundle_files": files,
        "collected_inputs": entries,
        "note": "Standalone bundle inventory; Windows DLL hash allowlist does not apply to ELF files.",
    }
    (target / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return inventory
