# Creating Windows, macOS, Linux, and FreeBSD Packages

**English** / [日本語](../packaging.md)

Distribution formats are PyInstaller onedir archives/packages (Windows, macOS, Linux) and FreeBSD Ports binary packages (`.pkg`):
- Windows: `Wune-vX.Y.Z-win64.zip` (bundles `Wune/Wune.exe` and `_internal/`)
- macOS: `Wune-vX.Y.Z-macos-arm64.zip` (bundles `Wune.app`)
- Linux: `Wune-vX.Y.Z-linux-x64.tar.gz`, `Wune-vX.Y.Z-linux-x64.zip`, `Wune-vX.Y.Z-1.x86_64.rpm`, `wune_X.Y.Z_amd64.deb`
- FreeBSD: `wune-X.Y.Z.pkg`, `py312-soundcard-0.4.6.pkg` (built via Ports)

## Local Builds for Developers

### Windows (x64)
Use Windows x64 with Python 3.10+ (tested with Python 3.13.14 x64) inside a dedicated virtual environment:

```powershell
python -m venv .venv-package
.\.venv-package\Scripts\python -m pip install -r requirements-build.txt
.\.venv-package\Scripts\python tools/build_windows.py --version 1.3.0
```

### Linux (x86_64)
Execute on Fedora / RHEL or Ubuntu / Debian (`rpm-build` required for `.rpm`, `dpkg-deb` for `.deb`):

```bash
python3 -m venv .venv-package
source .venv-package/bin/activate
pip install -r requirements-build.txt
python3 tools/build_linux.py --version 1.3.0
```

The script runs unit tests, invokes PyInstaller, checks dependencies of the relocated binary, builds packages, and generates SHA-256 checksums.
Existing artifacts with the same name are not overwritten. Output goes to `dist/`, working files to `build/`.

### FreeBSD (amd64)
Binary packages (`.pkg`) can be created directly from the repository tree using the FreeBSD Ports skeleton.
See the **[FreeBSD Ports / Pkg Guide](../../packaging/freebsd/README.md)** for details.

```bash
cd packaging/freebsd
make package
```

Generated `.pkg` packages are placed under `packaging/freebsd/packages/`.

## License and Corresponding Source Verification

Wune's application code is licensed under the BSD 2-Clause license (`LICENSE`).
Third-party software terms are documented in [`packaging/licenses/README.md`](../../packaging/licenses/README.md).
The build records all modules, data files, and DLLs actually collected by PyInstaller, gathering notices from target wheels.
Build-only tool notices are not unconditionally mixed into redistribution packages.
SDL DLLs are byte-verified against official Windows release archives.

Inside the ZIP, `licenses/inventory.json` logs bundled modules, package versions, and SHA-256 hashes of all DLLs, PYDs, fonts, and EXEs mapped to their notices.
`licenses/sources/` bundles corresponding source code for pygame, GNU FreeFont, and the Wune source tree used for that build, along with MPL-covered setuptools sources.
Upstream sources are downloaded and verified against SHA-256 hashes defined in `packaging/licenses/sources.json`.

## GitHub Actions

### Build Identity

Standard source launches and development ZIPs display `Wune dev (commit-id)` in the window title.
Environments without Git or distribution artifacts missing metadata display `Wune dev (unknown)`.
Builds triggered by a release tag `vX.Y.Z` display `Wune vX.Y.Z`.
Pull requests and manual runs produce development builds. Enabling `release_identity` on manual workflow dispatch produces release-identified packages without creating a tag or GitHub release.

Package metadata is saved to `build-info.json` beside the executable.

## Candidate Verification and Publishing

1. Merge release preparation changes to `main`.
2. Manually trigger the Windows, macOS, Linux, and FreeBSD packaging workflows on the same `main` commit with `version` set to `1.3.0` and `release_identity` enabled.
3. Validate candidate packages, SHA-256 checksums, bundled files, and perform physical device verification below.
4. Create tag `v1.3.0` on the verified commit. The resulting GitHub Release remains in draft status.
5. Review tag artifacts, release notes, and publish the release.

Never publish a release without completing physical machine verification.

## Physical Machine Verification

Verify audio playback/pause, sample rates, settings preview/apply/save/cancel, fullscreen toggling, localization (English/Japanese), backgrounds, menus, quit confirmations, and silence recovery on real hardware (Windows 11, macOS Apple Silicon, Linux Fedora, FreeBSD 15.1).

## macOS Apple Silicon

Requires Apple Silicon, Python with Tkinter, and Xcode Command Line Tools:

```sh
python3 -m venv .venv-package
.venv-package/bin/python -m pip install -r requirements-build.txt
.venv-package/bin/python tools/build_macos.py --version 1.3.0
```

Produces `dist/Wune-vX.Y.Z-macos-arm64.zip` and SHA-256 checksums containing `Wune.app`, bilingual README/guides (`INSTALL_MACOS.md`, `INSTALL_MACOS.en.md`), LICENSE, and licenses.
macOS builds are ad-hoc signed. See the [macOS Installation Guide](../../INSTALL_MACOS.en.md) for initial launch permissions.
The **macOS package** workflow runs on `macos-15` (Apple Silicon) runners, saving artifacts to `Wune-macos`.
Prior to publishing v1.3.0, verify all OS workflows from the same commit.
