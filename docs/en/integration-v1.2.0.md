# v1.2.0 Main Integration Log

**English** / [日本語](../integration-v1.2.0.md)

Record of changes and integrations in the `main` branch toward the major/minor update v1.2.0.
The major highlights are native support for Linux platforms (Issue #76), font family and size customization for the info bar (Issue #176), and improvements to media metadata display and UI layouts.

| Feature | Integration Details | Related Issue / PR |
| --- | --- | --- |
| Linux Platform Foundation | XDG configuration directory support, system locale detection, platform identification | #76, PR #165 |
| Linux Audio Capture | PulseAudio / PipeWire monitor loopback audio capture implementation | #76, PR #166 |
| Linux Now Playing (MPRIS2) | Active playback track metadata acquisition via `playerctl` / DBus | #76, PR #167 |
| Linux Packaging & CI | `.tar.gz`, `.zip`, RPM (`.rpm`), and Debian (`.deb`) package generation and GitHub Actions CI workflow | #76, PR #168, #169, #170, #171, #175 |
| Linux Installation Guides | Bilingual setup guides (`INSTALL_LINUX.md`, `INSTALL_LINUX.en.md`) and README updates | #76, PR #170, #172, #173, #174 |
| Windows GSMTC Playback Prioritization | Improved active player prioritization in Windows Global System Media Transport Controls | #159, PR #160 |
| Now Playing Overlap Prevention | Layout adjustments to prevent long track/artist titles from overlapping menu buttons and preset badges | #163, PR #164 |
| Menu Button Size Decoupling | Decoupled menu button size from info font size to prevent button oversizing | #176, PR #177 |
| Button & Badge Font Size Settings | Configurable font sizes for menu button and theme badge with "sync with text size" option | #176, PR #178 |
| Font Family Selection & Filtering | Info bar font family selection (system font enumeration, direct input, fallback) and CJK language filtering | #176, PR #179 |
| Settings Dialog Slimming & Optimization | Reduced Windows dialog initial/minimum size to 580x460 px (~31% reduction) and removed rigid wraplengths | #176, PR #179, PR #181 |
| Info Bar Bold Font Option | "Bold text" option (`info_font_bold`) for improved legibility | #176, PR #181 |
| Spectrum Settings 4-Column Grid | Unified LED aspect ratio, row count, and bar gap controls into a 4-column grid aligned vertically | PR #181 |
| Font Size Alignment in Info & Text Tab | Laid out font size controls one per row aligned with the font family combobox | PR #181 |
| Combobox Widths & Preset Labels | Compacted combobox widths to avoid stretching and eliminated duplicate resolution numbers in presets | PR #181 |
| Linux UI Spacing & Treeview Fixes | Dynamic Treeview row heights based on font metrics (preventing text overlap) and optimized window padding | PR #182 |

## Key Integration Commits

| Commit | Description |
| --- | --- |
| `9499707` | fix: prioritize active playback session in Windows GSMTC (#159) |
| `6255a8b` | fix: prevent Now Playing bar overlap with menu button and badge (#163) |
| `799cfd1` | feat: add Linux and Unix platform foundation (#76) |
| `66a80c8` | feat: add PulseAudio and PipeWire loopback capture for Linux (#76) |
| `32425f2` | feat: add Linux and Unix MPRIS2 metadata provider (#76) |
| `768fa64` | ci: add Linux packaging workflow and build scripts (#76) |
| `7d9eef8` | ci: add RPM and DEB package generation for Linux (#76) |
| `33744cd` | docs: add Linux installation guides and update README (#76) |
| `80af946` | fix: decouple menu button size from info font size (#176) |
| `408f0f7` | feat: add configurable button and badge font sizes with sync option (#176) |
| `64a4b26` | feat: add info font family selection (#176) |
| `450fe18` | fix: show font filter checkbox only for non-Latin languages (#176) |
| `a822d61` | fix: restore compact windows size and remove rigid label wraplengths (#176) |
| `feb1bac` | feat(ui): add bold font option and info bar color customization (#176) |
| `3f5153d` | refactor(ui): remove inline info bar color section from Info & Text tab (#176) |
| `6def429` | refactor(ui): compact settings dialog layout and eliminate excess vertical space (#176) |
| `0cb62b2` | Align spectrum settings controls into a unified 4-column grid |
| `6904a81` | Clarify that auto-adjust checkboxes take effect on window resize |
| `03a865f` | Lay out font sizes into 1 item per row aligned with font combo |
| `62174be` | Compact combobox widths and remove duplicate resolution in window preset labels |
| `d1ffe0a` | Fix Treeview text clipping and improve UI spacing on Linux |

## File Group Verification

- **wune/platform_linux.py / capture_linux.py / now_playing_mpris.py**: Linux-specific audio loopback capture and MPRIS2 metadata acquisition.
- **wune/font_finder.py**: Cross-platform system font enumeration and language-specific font filtering.
- **wune/settings_dialog.py**: Font family/size settings, bold toggle, 4-column grid layout, compacted comboboxes, and OS-specific dialog sizing.
- **wune/renderer.py**: Dynamic Pygame font rendering with bold and fallback support.
- **wune/config.py / settings.py**: Persistence for `info_font_bold`.
- **tools/build_linux.py / Wune-linux.spec / .github/workflows/linux-package.yml**: Linux package builds (tar.gz, zip, rpm, deb) and CI.
- **INSTALL_LINUX.md / INSTALL_LINUX.en.md**: Installation instructions for Fedora, RHEL, Ubuntu, and Debian.
