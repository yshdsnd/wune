# Third-party software in Wune's Windows distribution

Wune itself is BSD-2-Clause (see `../../LICENSE` in the source tree, or
`../LICENSE` in the release). Each component below keeps its own license.
This directory does not relicense third-party software under Wune's license.

`inventory.json`, generated from PyInstaller's collected inputs and the actual
distribution directory, lists collected Python modules, package versions,
native files, SHA-256 hashes, and the notices for each native file. Build-only
packages are not automatically listed as redistributed packages. Notice files
within a redistributed wheel are intentionally retained as a superset, including
vendored subcomponents. `native-libraries.json` is the reviewed SDL/font mapping;
new native libraries or changed SDL/font bytes stop the build.

## Components and acknowledgements

| Redistributed material | License / notice location |
| --- | --- |
| Python 3.13.14, standard library, bundled extensions | Python/LICENSE.txt and Python/history-and-license.rst; PSF terms and embedded third-party notices |
| OpenSSL 3, libffi, bzip2, Expat, zlib, libmpdec, XZ/LZMA and other Python embedded code | Python/; preserve the complete Windows and documentation license texts |
| Tcl 8.6.15 and Tk 8.6.15 scripts and DLLs | Tcl-Tk/; pinned upstream license.terms plus any installation notices |
| pygame 2.6.1, including embedded SDL_gfx code | pygame/ and pygame-supplement/; LGPL-2.1 and individual embedded-code notices |
| SDL 2.28.4, SDL_image 2.0.5, SDL_mixer 2.6.2, SDL_ttf 2.20.1 | SDL2*/; zlib license and dependency notices |
| libjpeg, libpng, libtiff, libwebp, zlib | SDL2_image/; individual upstream licenses |
| libmodplug, Ogg, Opus, opusfile, embedded dr_libs, stb_vorbis and TiMidity | SDL2_mixer/; individual upstream licenses |
| FreeType (standalone and embedded in SDL_ttf), HarfBuzz (embedded in SDL_ttf) | pygame-supplement/LICENSE.freetype.txt and SDL2_ttf/ |
| PortMidi | pygame-supplement/LICENSE.portmidi.txt |
| GNU FreeFont FreeSansBold, release 20120503 (pygame fallback only) | FreeFont/; GPL-3.0-or-later with font embedding exception |
| NumPy 2.5.3, OpenBLAS, LAPACK, GCC runtime and embedded algorithms | numpy/; complete wheel license tree, including GCC runtime exception |
| SoundCard 0.4.6, CFFI 2.1.1, pycparser 3.0 | SoundCard/, cffi/, pycparser/; BSD/MIT-family terms |
| setuptools 84.0.0 and its vendored packages, packaging 26.3 | setuptools/ and packaging/; nested notices include MPL-2.0-covered validate-pyproject code |
| PyInstaller 6.22.3 bootloader/loader and runtime hooks | pyinstaller/; GPL with bootloader exception and Apache-2.0 runtime hooks |
| Microsoft VC runtime / Universal CRT / API-set runtime DLLs | Python/LICENSE.txt, Windows binary build additional conditions; terms below |

This software is based in part on the work of the Independent JPEG Group.
OpenSSL 3.0.21: Copyright 1998-2026 The OpenSSL Authors. All rights reserved.
Portions of this software are copyright of the FreeType Project
(https://www.freetype.org). All rights reserved.
The original copyright notices and full terms are retained in the files above.

## Corresponding source, modification and rebuilding

The release includes `sources/pygame-2.6.1.zip` (unmodified library source and
build configuration), `sources/freefont-src-20120503.tar.gz` (the font's editable
SFD sources and build instructions), and `sources/Wune-source.zip` (the exact
application source and packaging scripts for this build). `sources.json` records
the official URLs and SHA-256 values of the upstream archives. They are downloaded
and verified at build time and included inside every release ZIP, not just linked
from a website or promised on request. The source tree has the same pinned URLs
and hashes; generated archives are not committed to Git.

pygame's Python modules are frozen in Wune.exe. To use a modified pygame, unpack
Wune-source.zip, follow docs/packaging.md to create the Python build environment,
unpack pygame-2.6.1.zip, follow its buildconfig instructions to build/install your
modified library into that environment, and rebuild Wune. Update the reviewed
notice/native-file mapping when intentionally modifying native library bytes.
The native DLLs also remain separate under `_internal`; ABI-compatible modified
libraries can replace them there (replace all copies listed in inventory.json).
Modification for your own use and reverse engineering for debugging modifications
to LGPL-covered libraries are permitted. Wune imposes no additional restriction
on those rights. The bundled source/build material supports LGPL-2.1 section 6(a).

`setuptools-source.zip` contains the original installed Python/JSON source,
including its MPL-covered configuration validators, and their notices. These
files are unmodified. Sources and their original licenses remain available for
modification and redistribution under their respective terms.

## Microsoft runtime terms

Microsoft Distributable Code is independently licensed. Its copyright, trademark
and patent notices must be preserved. Redistribution is permitted only for use
on Microsoft operating systems, runtime technologies or application platforms;
it must not be included in malicious, deceptive or unlawful programs. Do not use
Microsoft trademarks to imply Microsoft authorship or endorsement. Recipients
redistributing those runtime components must pass on these protections to their
distributors and end users. These conditions apply only to Microsoft
Distributable Code, not Wune's BSD-licensed code or other independent components.
See the complete Windows binary build conditions in Python/LICENSE.txt.

## Provenance and updates

`provenance.json` records the exact upstream archive/member or source URL and
SHA-256 for each checked-in notice. SDL DLLs were byte-compared with the upstream
Windows archives selected by pygame 2.6.1's buildconfig/download_win_prebuilt.py.
SDL_ttf's archive includes FreeType and HarfBuzz notices even though those libraries
are embedded in SDL2_ttf.dll. SDL_mixer's decoder notices also cover embedded code.
The extra pygame license directory is retained in full as upstream provides it;
it does not mean every optional pygame dependency is in this distribution.

When changing Python, wheels, fonts or packaging hooks, rebuild and inspect the
actual inventory, including statically embedded dependencies. Do not add a generic
catch-all rule to silence an unknown-library error. Review upstream source/build
configuration, preserve new notices and any corresponding-source obligations,
then update the mappings, hashes, table, and tests together.
