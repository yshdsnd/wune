"""Build, probe and zip a standalone macOS .app bundle."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wune.build_identity import VERSION_PATTERN, format_identity, source_commit


def validate_version(value):
    if not re.fullmatch(VERSION_PATTERN, value):
        raise argparse.ArgumentTypeError("Use X.Y.Z or X.Y.Z-rc.1 (without the v prefix)")
    return value


def smoke_test(app_bundle, build_info):
    # Launch relocated application outside checkout, with isolated environment.
    probe = Path(tempfile.gettempdir()) / ("Wune-package-check-" + uuid.uuid4().hex)
    probe.mkdir()
    relocated = probe / "Wune.app"
    shutil.copytree(app_bundle, relocated, symlinks=True)
    home = probe / "user-data"
    home.mkdir()
    report = probe / "smoke.json"
    env = {key: value for key, value in os.environ.items()}
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["HOME"] = str(home)
    executable = relocated / "Contents" / "MacOS" / "Wune"
    result = subprocess.run([str(executable), "--package-smoke-test", str(report)],
                            cwd=probe, env=env, timeout=90)
    if result.returncode or not report.exists() or not json.loads(report.read_text(encoding="utf-8")).get("ok"):
        log = home / "Library" / "Application Support" / "Wune" / "Wune.log"
        if log.exists():
            print(log.read_text(encoding="utf-8", errors="replace"))
        raise RuntimeError(f"Packaged smoke test failed; inspect {log}")
    print(f"Packaged smoke test passed: {report}")
    expected = format_identity(build_info)
    actual = json.loads(report.read_text(encoding="utf-8")).get("identity")
    if actual != expected:
        raise RuntimeError(f"Packaged build identity ({actual}) differs from build metadata ({expected})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, type=validate_version)
    parser.add_argument("--release", action="store_true",
                        help="Use --version as official release identity; otherwise identify a development build")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        parser.error("Build on macOS")
    arch = platform.machine().lower()
    if arch not in ("arm64", "x86_64"):
        parser.error("Build on 64-bit macOS (arm64 or x86_64)")

    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    archive = output / f"Wune-v{args.version}-macos-{arch}.zip"
    if archive.exists():
        raise FileExistsError(f"Will not replace an existing package: {archive}")

    work = ROOT / "build" / ("macos-" + uuid.uuid4().hex)
    work.mkdir(parents=True)

    # 1. Run full test suite before bundling
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"], cwd=ROOT, check=True)

    # 2. Build .app bundle with PyInstaller
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--workpath", str(work / "work"), "--distpath", str(work / "output"),
        str(ROOT / "Wune-macos.spec")
    ], cwd=ROOT, check=True)

    app_source = work / "output" / "Wune.app"
    bundle_dir = work / "bundle"
    bundle_dir.mkdir()

    app_target = bundle_dir / "Wune.app"
    shutil.copytree(app_source, app_target, symlinks=True)

    # 3. Copy documentation and licenses
    shutil.copy2(ROOT / "packaging" / "README.txt", bundle_dir / "README.txt")
    shutil.copy2(ROOT / "README.md", bundle_dir / "README.md")
    shutil.copy2(ROOT / "README.en.md", bundle_dir / "README.en.md")
    shutil.copy2(ROOT / "INSTALL_MACOS.md", bundle_dir / "INSTALL.md")
    shutil.copy2(ROOT / "INSTALL_MACOS.en.md", bundle_dir / "INSTALL.en.md")
    shutil.copytree(ROOT / "docs", bundle_dir / "docs")
    (bundle_dir / "packaging" / "licenses").mkdir(parents=True)
    shutil.copy2(ROOT / "packaging" / "licenses" / "README.md", bundle_dir / "packaging" / "licenses" / "README.md")
    shutil.copy2(ROOT / "LICENSE", bundle_dir / "LICENSE")

    # 4. Update Info.plist with exact version
    plist_path = app_target / "Contents" / "Info.plist"
    with plist_path.open("rb") as stream:
        plist = plistlib.load(stream)
    plist["CFBundleVersion"] = args.version
    plist["CFBundleShortVersionString"] = args.version
    with plist_path.open("wb") as stream:
        plistlib.dump(plist, stream)

    # 5. Write build identity metadata
    info = {
        "version": args.version,
        "release_version": args.version if args.release else None,
        "commit": source_commit(ROOT),
        "python": sys.version,
        "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
    }
    info_json = json.dumps(info, indent=2)
    (bundle_dir / "build-info.json").write_text(info_json, encoding="utf-8")
    (app_target / "Contents" / "MacOS" / "build-info.json").write_text(info_json, encoding="utf-8")

    # 6. Apply ad-hoc code signing to the .app bundle
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app_target)], check=True)

    # 7. Execute isolated smoke test
    smoke_test(app_target, info)

    # 8. Create ZIP archive preserving permissions and symlinks
    subprocess.run(["zip", "-q", "-r", "-y", str(archive), "."], cwd=bundle_dir, check=True)

    # 9. Write SHA256 checksum
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(archive)


if __name__ == "__main__":
    main()
