"""Build, probe and zip a standalone Windows x64 directory."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile

try:
    from tools.license_audit import collect_notices, collect_sources, verify_archive
except ModuleNotFoundError:
    from license_audit import collect_notices, collect_sources, verify_archive

ROOT = Path(__file__).resolve().parents[1]


def validate_version(value):
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?", value):
        raise argparse.ArgumentTypeError("Use X.Y.Z or X.Y.Z-rc.1 (without the v prefix)")
    return value


def smoke_test(bundle):
    # Launch a relocated executable outside the checkout, with no Python search paths.
    probe = Path(tempfile.gettempdir()) / ("Wune-package-check-" + uuid.uuid4().hex)
    probe.mkdir()
    relocated = probe / "Wune"
    shutil.copytree(bundle, relocated)
    home = probe / "user-data"
    home.mkdir()
    report = probe / "smoke.json"
    env = {key.upper(): value for key, value in os.environ.items()}
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["LOCALAPPDATA"] = str(home)
    env["PATH"] = str(Path(env["SYSTEMROOT"]) / "System32")
    result = subprocess.run([str(relocated / "Wune.exe"), "--package-smoke-test", str(report)],
                            cwd=probe, env=env, timeout=90)
    if result.returncode or not report.exists() or not json.loads(report.read_text(encoding="utf-8")).get("ok"):
        log = home / "Wune" / "Wune.log"
        if log.exists():
            print(log.read_text(encoding="utf-8", errors="replace"))
        raise RuntimeError(f"Packaged smoke test failed; inspect {home / 'Wune' / 'Wune.log'}")
    print(f"Packaged smoke test passed: {report}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, type=validate_version)
    args = parser.parse_args(argv)
    if sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
        parser.error("Build on 64-bit Windows with x64 Python")
    if sys.maxsize <= 2**32:
        parser.error("Use 64-bit Python")
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    archive = output / f"Wune-v{args.version}-win64.zip"
    if archive.exists():
        raise FileExistsError(f"Will not replace an existing package: {archive}")
    work = ROOT / "build" / ("windows-" + uuid.uuid4().hex)
    work.mkdir(parents=True)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"], cwd=ROOT, check=True)
    # Do not collect unrelated runtime DLLs from tools on the caller's PATH.
    build_env = os.environ.copy()
    build_env["PATH"] = os.pathsep.join((str(Path(sys.executable).parent), sys.base_prefix,
                                        str(Path(sys.base_prefix) / "DLLs"),
                                        str(Path(os.environ["SYSTEMROOT"]) / "System32")))
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "--workpath", str(work / "work"), "--distpath", str(work / "output"),
                    str(ROOT / "Wune.spec")], cwd=ROOT, env=build_env, check=True)
    bundle = work / "output" / "Wune"
    shutil.copy2(ROOT / "packaging" / "README.txt", bundle / "README.txt")
    shutil.copy2(ROOT / "README.md", bundle / "README.md")
    shutil.copytree(ROOT / "docs", bundle / "docs")  # Keep README links/images usable offline.
    # Preserve the source README's third-party-notice link in the offline package.
    (bundle / "packaging" / "licenses").mkdir(parents=True)
    shutil.copy2(ROOT / "packaging" / "licenses" / "README.md", bundle / "packaging" / "licenses" / "README.md")
    shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE")
    collect_notices(bundle, work / "work" / "Wune" / "license-inputs.json")
    collect_sources(bundle, ROOT / "build" / "license-sources")
    info = {"version": args.version, "python": sys.version,
            "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()}}
    (bundle / "build-info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    smoke_test(bundle)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zipped:
        for file in sorted(bundle.rglob("*")):
            if file.is_file():
                zipped.write(file, file.relative_to(bundle.parent))
    verify_archive(archive, bundle)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(archive)


if __name__ == "__main__":
    main()
