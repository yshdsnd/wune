"""Build, probe and package a standalone Linux x64 directory."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wune.build_identity import VERSION_PATTERN, format_identity, source_commit


def validate_version(value):
    if not re.fullmatch(VERSION_PATTERN, value):
        raise argparse.ArgumentTypeError("Use X.Y.Z or X.Y.Z-rc.1 (without the v prefix)")
    return value


def smoke_test(bundle):
    # Launch relocated executable outside checkout with clean environment.
    probe = Path(tempfile.gettempdir()) / ("Wune-package-check-" + uuid.uuid4().hex)
    probe.mkdir()
    relocated = probe / "Wune"
    shutil.copytree(bundle, relocated)
    home = probe / "user-data"
    home.mkdir()
    report = probe / "smoke.json"
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["HOME"] = str(home)
    env["XDG_CONFIG_HOME"] = str(home / ".config")

    executable = relocated / "Wune"
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    result = subprocess.run(
        [str(executable), "--package-smoke-test", str(report)],
        cwd=probe,
        env=env,
        timeout=90,
    )

    if result.returncode != 0 or not report.exists() or not json.loads(report.read_text(encoding="utf-8")).get("ok"):
        log_file = home / ".config" / "Wune" / "Wune.log"
        if log_file.exists():
            print(log_file.read_text(encoding="utf-8", errors="replace"))
        raise RuntimeError(f"Packaged smoke test failed; returncode={result.returncode}")

    print(f"Packaged smoke test passed: {report}")
    expected = format_identity(json.loads((bundle / "build-info.json").read_text(encoding="utf-8")))
    if json.loads(report.read_text(encoding="utf-8")).get("identity") != expected:
        raise RuntimeError("Packaged build identity differs from build metadata")


def create_desktop_file(target_dir):
    desktop_content = """[Desktop Entry]
Type=Application
Name=Wune
Comment=LED Bar Audio Spectrum Visualizer
Exec=Wune
Icon=wune
Categories=AudioVideo;Audio;
Terminal=false
"""
    (target_dir / "wune.desktop").write_text(desktop_content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, type=validate_version)
    parser.add_argument("--release", action="store_true",
                        help="Use --version as official release identity; otherwise identify a development build")
    args = parser.parse_args(argv)

    if sys.platform != "linux" or platform.machine().lower() not in ("amd64", "x86_64"):
        parser.error("Build on 64-bit Linux with x64 Python")
    if sys.maxsize <= 2**32:
        parser.error("Use 64-bit Python")

    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    tar_archive = output / f"Wune-v{args.version}-linux-x64.tar.gz"
    zip_archive = output / f"Wune-v{args.version}-linux-x64.zip"

    if tar_archive.exists():
        raise FileExistsError(f"Will not replace an existing package: {tar_archive}")

    work = ROOT / "build" / ("linux-" + uuid.uuid4().hex)
    work.mkdir(parents=True)

    # 1. Run full test suite
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, check=True)

    # 2. Run PyInstaller
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--workpath", str(work / "work"), "--distpath", str(work / "output"),
        str(ROOT / "Wune-linux.spec")
    ], cwd=ROOT, check=True)

    bundle = work / "output" / "Wune"
    executable = bundle / "Wune"
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    # 3. Documentation and desktop integration
    shutil.copy2(ROOT / "packaging" / "README.txt", bundle / "README.txt")
    shutil.copy2(ROOT / "README.md", bundle / "README.md")
    shutil.copy2(ROOT / "README.en.md", bundle / "README.en.md")

    install_doc = ROOT / "INSTALL_LINUX.md"
    install_doc_en = ROOT / "INSTALL_LINUX.en.md"
    if install_doc.exists():
        shutil.copy2(install_doc, bundle / "INSTALL.md")
        shutil.copy2(install_doc, bundle / "INSTALL_LINUX.md")
    if install_doc_en.exists():
        shutil.copy2(install_doc_en, bundle / "INSTALL.en.md")
        shutil.copy2(install_doc_en, bundle / "INSTALL_LINUX.en.md")

    shutil.copytree(ROOT / "docs", bundle / "docs")
    (bundle / "packaging" / "licenses").mkdir(parents=True)
    shutil.copy2(ROOT / "packaging" / "licenses" / "README.md", bundle / "packaging" / "licenses" / "README.md")
    shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE")

    create_desktop_file(bundle)
    if (ROOT / "wune" / "assets" / "wune.png").is_file():
        shutil.copy2(ROOT / "wune" / "assets" / "wune.png", bundle / "wune.png")

    # 4. Build info
    info = {
        "version": args.version,
        "release_version": args.version if args.release else None,
        "commit": source_commit(ROOT),
        "python": sys.version,
        "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
    }
    (bundle / "build-info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # 5. Smoke test
    smoke_test(bundle)

    # 6. Create tar.gz archive (preserving executable permissions)
    with tarfile.open(tar_archive, "w:gz") as tar:
        tar.add(bundle, arcname="Wune")

    digest = hashlib.sha256(tar_archive.read_bytes()).hexdigest()
    tar_archive.with_suffix(".tar.gz.sha256").write_text(f"{digest}  {tar_archive.name}\n", encoding="ascii")
    print(tar_archive)

    # 7. Also create zip archive for convenience
    with zipfile.ZipFile(zip_archive, "x", zipfile.ZIP_DEFLATED) as zipped:
        for file in sorted(bundle.rglob("*")):
            if file.is_file():
                zipped.write(file, file.relative_to(bundle.parent))
    zip_digest = hashlib.sha256(zip_archive.read_bytes()).hexdigest()
    zip_archive.with_suffix(".zip.sha256").write_text(f"{zip_digest}  {zip_archive.name}\n", encoding="ascii")
    print(zip_archive)


if __name__ == "__main__":
    main()
