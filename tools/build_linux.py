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
try:
    from tools.license_audit import collect_sources
    from tools.linux_licenses import collect_notices
except ModuleNotFoundError:
    from license_audit import collect_sources
    from linux_licenses import collect_notices


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
Exec=wune
Icon=wune
Categories=AudioVideo;Audio;
Terminal=false
StartupWMClass=Wune
"""
    (target_dir / "wune.desktop").write_text(desktop_content, encoding="utf-8")


def create_rpm_package(bundle: Path, output_dir: Path, version: str) -> Path | None:
    """Package the standalone bundle into a Fedora/RHEL/openSUSE RPM package."""
    if not shutil.which("rpmbuild"):
        print("rpmbuild not found; skipping RPM package creation.")
        return None

    rpm_version = version.replace("-", "_")
    work_dir = Path(tempfile.mkdtemp(prefix="wune-rpm-"))
    try:
        topdir = work_dir / "rpmbuild"
        for sub in ("BUILD", "RPMS", "SOURCES", "SPECS", "SRPMS"):
            (topdir / sub).mkdir(parents=True, exist_ok=True)

        spec_file = topdir / "SPECS" / "wune.spec"
        spec_content = f"""%define _topdir {topdir}
%define _build_id_links none
%define debug_package %{{nil}}
%global __os_install_post %{{nil}}

Name:           wune
Version:        {rpm_version}
Release:        1
Summary:        LED Bar Audio Spectrum Visualizer
License:        BSD-2-Clause
URL:            https://github.com/yshdsnd/wune
AutoReqProv:    no
Requires:       pulseaudio-libs

%description
Wune is a customizable LED-bar audio spectrum visualizer featuring real-time
frequency analysis, smooth ballistics, theme presets, and Now Playing metadata.

%install
rm -rf %{{buildroot}}
mkdir -p %{{buildroot}}/opt/wune
cp -a {bundle}/* %{{buildroot}}/opt/wune/

mkdir -p %{{buildroot}}/usr/bin
ln -s /opt/wune/Wune %{{buildroot}}/usr/bin/wune
ln -s /opt/wune/Wune %{{buildroot}}/usr/bin/Wune

mkdir -p %{{buildroot}}/usr/share/applications
if [ -f {bundle}/wune.desktop ]; then
    cp {bundle}/wune.desktop %{{buildroot}}/usr/share/applications/
fi

mkdir -p %{{buildroot}}/usr/share/pixmaps
if [ -f {bundle}/wune.png ]; then
    cp {bundle}/wune.png %{{buildroot}}/usr/share/pixmaps/wune.png
fi

%clean
rm -rf %{{buildroot}}

%files
/opt/wune
/usr/bin/wune
/usr/bin/Wune
/usr/share/applications/wune.desktop
{"/usr/share/pixmaps/wune.png" if (bundle / "wune.png").is_file() else ""}
"""
        spec_file.write_text(spec_content, encoding="utf-8")

        subprocess.run(["rpmbuild", "-bb", str(spec_file)], check=True)

        rpm_files = list((topdir / "RPMS").glob("**/*.rpm"))
        if not rpm_files:
            raise RuntimeError("RPM package was not created by rpmbuild")

        built_rpm = rpm_files[0]
        target_rpm = output_dir / f"Wune-v{version}-1.x86_64.rpm"
        shutil.copy2(built_rpm, target_rpm)

        digest = hashlib.sha256(target_rpm.read_bytes()).hexdigest()
        target_rpm.with_suffix(".rpm.sha256").write_text(f"{digest}  {target_rpm.name}\n", encoding="ascii")
        print(f"Built RPM package: {target_rpm}")
        return target_rpm
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def create_deb_package(bundle: Path, output_dir: Path, version: str) -> Path | None:
    """Package the standalone bundle into a Debian/Ubuntu DEB package."""
    if not shutil.which("dpkg-deb"):
        print("dpkg-deb not found; skipping DEB package creation.")
        return None

    deb_version = version.replace("-", "~")
    work_dir = Path(tempfile.mkdtemp(prefix="wune-deb-"))
    try:
        pkg_root = work_dir / "pkg"
        debian_dir = pkg_root / "DEBIAN"
        debian_dir.mkdir(parents=True)

        control_content = f"""Package: wune
Version: {deb_version}
Section: sound
Priority: optional
Architecture: amd64
Depends: libpulse0
Maintainer: Yoshihide Sonoda <yshd.snd@gmail.com>
Description: LED Bar Audio Spectrum Visualizer
 Wune is a customizable LED-bar audio spectrum visualizer featuring real-time
 frequency analysis, smooth ballistics, theme presets, and Now Playing metadata.
"""
        (debian_dir / "control").write_text(control_content, encoding="utf-8")

        opt_dir = pkg_root / "opt" / "wune"
        opt_dir.mkdir(parents=True)
        for item in bundle.iterdir():
            if item.is_dir():
                shutil.copytree(item, opt_dir / item.name)
            else:
                shutil.copy2(item, opt_dir / item.name)

        bin_dir = pkg_root / "usr" / "bin"
        bin_dir.mkdir(parents=True)
        wune_bin = bin_dir / "wune"
        wune_upper_bin = bin_dir / "Wune"
        try:
            wune_bin.symlink_to("/opt/wune/Wune")
            wune_upper_bin.symlink_to("/opt/wune/Wune")
        except OSError:
            script = '#!/bin/sh\nexec /opt/wune/Wune "$@"\n'
            wune_bin.write_text(script, encoding="utf-8")
            wune_bin.chmod(wune_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            wune_upper_bin.write_text(script, encoding="utf-8")
            wune_upper_bin.chmod(wune_upper_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        if (bundle / "wune.desktop").is_file():
            apps_dir = pkg_root / "usr" / "share" / "applications"
            apps_dir.mkdir(parents=True)
            shutil.copy2(bundle / "wune.desktop", apps_dir / "wune.desktop")

        if (bundle / "wune.png").is_file():
            pixmaps_dir = pkg_root / "usr" / "share" / "pixmaps"
            pixmaps_dir.mkdir(parents=True)
            shutil.copy2(bundle / "wune.png", pixmaps_dir / "wune.png")

        target_deb = output_dir / f"wune_{deb_version}_amd64.deb"
        subprocess.run(["dpkg-deb", "--build", "--root-owner-group", str(pkg_root), str(target_deb)], check=True)

        digest = hashlib.sha256(target_deb.read_bytes()).hexdigest()
        target_deb.with_suffix(".deb.sha256").write_text(f"{digest}  {target_deb.name}\n", encoding="ascii")
        print(f"Built DEB package: {target_deb}")
        return target_deb
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


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

    collect_notices(bundle, work / "work" / "license-inputs.json")
    collect_sources(bundle, ROOT / "build" / "license-sources")

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

    # 8. Create distro packages (RPM for Fedora/RHEL, DEB for Debian/Ubuntu)
    create_rpm_package(bundle, output, args.version)
    create_deb_package(bundle, output, args.version)


if __name__ == "__main__":
    main()
