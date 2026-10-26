# Issue #58: Windows Distribution License Audit

**English** / [日本語](../license-audit.md)

On 2026-09-26, using Windows 11 x64 / Python 3.13.14 / pinned versions in `requirements-build.txt`,
a real distribution ZIP was built using `tools/build_windows.py --version 0.0.0-issue58` and audited.
Wune's own BSD 2-Clause license (Copyright 2026 Yoshihide Sonoda) is preserved in `LICENSE`.

## Audited Bundled Assets

The PyInstaller collection inputs were cross-referenced against the actual data extracted from the ZIP.
Python packages included: SoundCard, CFFI, NumPy, packaging, pycparser, pygame, setuptools,
and the PyInstaller bootloader embedded in the executable. Python and Tcl/Tk were recorded separately.
Nested dependencies of setuptools retain their complete wheel notice trees with names and versions recorded.
Build-time tools (pip, altgraph, pefile, pywin32-ctypes, pyinstaller-hooks-contrib) reside in the build environment
but are not collected into the redistribution package.

The audited ZIP contained 97 native files (including identical DLLs placed at distinct paths):

| Category | File Count |
| --- | ---: |
| Python core and standard extensions | 17 |
| OpenSSL and libffi | 3 |
| Tcl/Tk | 2 |
| CFFI extension | 1 |
| NumPy extensions | 13 |
| OpenBLAS (including LAPACK and GCC runtimes) | 1 |
| pygame extensions | 30 |
| SDL2 | 2 |
| SDL_image with JPEG/PNG/TIFF/WebP/zlib | 10 |
| SDL_mixer with modplug/Ogg/Opus/opusfile | 8 |
| SDL_ttf (including FreeType and HarfBuzz) | 2 |
| FreeType DLL | 2 |
| PortMidi | 1 |
| GNU FreeFont fallback font | 1 |
| Microsoft C/C++ runtimes | 3 |
| Wune.exe (including PyInstaller bootloader) | 1 |

Actual file paths, SHA-256 digests, and corresponding notices are canonically recorded in each ZIP's `licenses/inventory.json`.
The number of Microsoft runtime files may vary depending on Python distributors and Windows environments.
Satisfying table counts alone is not the pass criterion.

## Identified Discrepancies and Remediation

- While pygame wheels include the LGPL license text, notices for Windows-dependent DLLs were incomplete.
  The 15 bundled DLLs were SHA-256 matched against SDL distribution archives referenced by official pygame build configs.
  Missing notices, static SDL_mixer decoders, and FreeType/HarfBuzz notices inside SDL_ttf were supplemented.
- Python Windows installations lacked Tcl notices. The Tcl/Tk 8.6.15 notice text from Python's official source dependencies was pinned and saved.
- Editable sources for pygame's legacy fallback font could not be reliably traced.
  Replaced with GNU FreeFont 20120503's FreeSansBold and its matching SFD source, preserving standard Windows font selection.
- For LGPL-licensed pygame, the library source code and Wune rebuilding materials are bundled directly inside the ZIP.
  Python/JSON sources for MPL-covered setuptools are also bundled.
- Previous builds picked up unrelated UCRT/API-set DLLs from the caller's PATH.
  PyInstaller execution PATH was restricted to the active Python environment and System32.

Provenances, upstream archive member names, and notice digests are stored in `packaging/licenses/provenance.json`.
Corresponding source URLs and digests are recorded in `sources.json`.
For individual license conditions, acknowledgments, and rebuild instructions, see [Third-Party Software Notices](../../packaging/licenses/README.md).

## Verification and Release Policy

- All 167 tests passed (including license collection, unknown DLL detection, missing notice detection, hash alteration, and ZIP audit regression tests).
- Clean smoke test of relocated executable outside the Python search path.
- Hash verification of all notices, source archives, and enumerated native binaries inside the ZIP.
- Verified GNU FreeFont fallback font matches known good byte sequences rather than old pygame binaries.
- Confirmed inclusion of 3 corresponding source archives and setuptools source tree (242 files).

This document serves as an audit record for the specified pinned build.
Audits are updated whenever dependencies or build environments change.
Physical audio input and tests on clean Windows machines without Python are required per Issue #38 pre-release checklist.
