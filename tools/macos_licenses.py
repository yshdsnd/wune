"""Collect notices and a signed-bundle inventory for macOS packaging.

The Windows DLL hash allowlist is deliberately not used for Mach-O binaries.
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


def collect_notices(bundle, inputs):
    target = bundle / "licenses"
    shutil.copytree(SUPPLEMENT, target)
    for item in json.loads((SUPPLEMENT / "provenance.json").read_text(encoding="utf-8")):
        if sha256(SUPPLEMENT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Vendored notice changed: {item['path']}")
    owners = {}
    for dist in metadata.distributions():
        for file in dist.files or ():
            owners[Path(dist.locate_file(file)).resolve()] = dist
    entries = json.loads(inputs.read_text(encoding="utf-8"))
    selected = {"PyInstaller": metadata.distribution("PyInstaller")}
    for _, source, _ in entries:
        dist = owners.get(Path(source).resolve())
        if dist is not None:
            selected[dist.metadata["Name"]] = dist
    pinned = {}
    for requirements in (ROOT / "requirements.txt", ROOT / "requirements-build.txt"):
        for line in requirements.read_text(encoding="utf-8").splitlines():
            if "==" in line:
                name, version = line.split("==")
                pinned[name.lower()] = version
    components = {}
    for name, dist in sorted(selected.items()):
        if pinned.get(name.lower()) != dist.version:
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
    # Python/Tcl/Tk/FreeFont notices are also in the verified supplement.
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        copy_required(python_license, target / "Python" / "LICENSE.txt")
    components["Python"] = {"version": sys.version.split()[0],
                            "notices": ["licenses/Python/history-and-license.rst"]}
    files = [{"path": p.relative_to(bundle).as_posix(), "sha256": sha256(p)}
             for p in sorted((bundle / "Wune.app").rglob("*")) if p.is_file() and not p.is_symlink()]
    (target / "inventory.json").write_text(json.dumps({
        "platform": "macOS", "components": components, "bundle_files": files,
        "collected_inputs": entries,
        "note": "Signed bundle inventory; Windows DLL hash allowlist does not apply to Mach-O files.",
    }, indent=2), encoding="utf-8")
