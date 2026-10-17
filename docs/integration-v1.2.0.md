# v1.2.0 main統合記録

次期メジャー/マイナーアップデート v1.2.0 に向けた main ブランチの変更・統合記録です。
主な変更点は、Linux プラットフォームの正式サポート（Issue #76）、ヘッダーフォントおよび情報バーフォントのカスタマイズ機能（Issue #176）、メディア情報取得と UI レイアウトの改善です。

| 機能 | 統合内容・対応 | 関連 Issue / PR |
| --- | --- | --- |
| Linux プラットフォーム基盤 | XDG 設定ディレクトリ対応、システムロケール取得、プラットフォーム判定 | #76, PR #165 |
| Linux 音声キャプチャ | PulseAudio / PipeWire monitor loopback 音声キャプチャ実装 | #76, PR #166 |
| Linux Now Playing (MPRIS2) | `playerctl` / DBus を用いた再生中楽曲メタデータ取得 | #76, PR #167 |
| Linux パッケージング & CI | `.tar.gz`, `.zip`, RPM (`.rpm`), Debian (`.deb`) パッケージ生成および GitHub Actions ワークフロー追加 | #76, PR #168, #169, #170, #171, #175 |
| Linux インストールガイド | 日英導入ガイド（`INSTALL_LINUX.md`, `INSTALL_LINUX.en.md`）の作成と README 更新 | #76, PR #170, #172, #173, #174 |
| Windows GSMTC 再生判定 | Windows システムメディアセッションでアクティブ再生中プレイヤーの優先判定を改善 | #159, PR #160 |
| Now Playing 表示重なり防止 | 長い曲名・アーティスト名表示時にメニューボタンやテーマバッジと重ならないようレイアウト調整 | #163, PR #164 |
| メニューボタンサイズの独立化 | 情報バーのフォントサイズ変更時にメニューボタンが過剰に肥大化しないよう分離 | #176, PR #177 |
| ボタン・バッジフォントサイズ設定 | メニューボタンおよびテーマバッジのフォントサイズ個別指定と「文字サイズに連動」オプション | #176, PR #178 |
| フォントファミリ選択 & 言語別絞り込み | 情報バーのフォントファミリ選択（システムフォント一覧、直接入力、フォールバック）、非ラテン言語（日本語・中国語・韓国語等）環境向けフォントフィルタ | #176, PR #179 |
| 設定ダイアログの縦幅スリム化 & 最適化 | Windows でのダイアログ初期・最小サイズを 580x460 に短縮（約31%削減）、固定 wraplength 撤廃によるテキスト折り返し適正化 | #176, PR #179, PR #181 |
| 情報バーの太字設定 | 「文字を太字にする」オプション（`info_font_bold`）による視認性向上 | #176, PR #181 |
| スペクトラム設定の4列グリッド整列 | LED 縦横比、LED 段数、バー間隔の3行を統一された4列グリッドに集約し、スピンボックス・自動調整ボタン・チェックボックスを縦一列に整列。「リサイズ時に〜を自動調整」文言の明確化 | PR #181 |
| 情報・文字設定のフォントサイズ整列 | 目盛り、情報バー、メニューボタン、テーマバッジのフォントサイズ設定を1行1項目に改行し、フォント名コンボボックスと左端整列 | PR #181 |
| コンボボックス幅適正化 & プリセット表記整理 | テーマ、背景、一般タブのプルダウン幅の適正化（右端引き伸ばし解消・左揃え）およびウィンドウサイズプリセットの重複解像度表記解消 | PR #181 |
| Linux UI改善 & Treeview文字潰れ解消 | Linux 環境でのテーマ一覧 Treeview 行高をフォントメトリクスに基づき動的拡大（文字の上下重なり解消）、大きめのデスクトップフォントに合わせた余白（pady）・最小サイズ（640x540）の最適化 | PR #182 |

## 主要な統合コミット

| コミット | 内容 |
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

## ファイル群の確認

- **wune/platform_linux.py / capture_linux.py / now_playing_mpris.py**: Linux プラットフォーム固有の音声ループバックキャプチャと MPRIS2 メディア情報取得。
- **wune/font_finder.py**: プラットフォーム別のシステムフォント一覧列挙と、指定言語（日本語・中国語・韓国語等）の対応フォント判定。
- **wune/settings_dialog.py**: 「情報・文字」タブでのフォントファミリ・フォントサイズ設定・太字切り替え、スペクトラムタブの4列グリッド整列、コンボボックス幅の適正化、OS 別の設定ダイアログサイズ（Windows 580x460, Linux 640x540）および Linux 向けの動的 Treeview 行高と UI スペーシング最適化。
- **wune/renderer.py**: 情報バーフォントファミリ指定および太字指定（`info_font_bold`）に基づく Pygame フォント動的生成とフォールバック。
- **wune/config.py / settings.py**: `info_font_bold` 設定の保持と永続化。
- **tools/build_linux.py / Wune-linux.spec / .github/workflows/linux-package.yml**: Linux 向けバイナリ・パッケージ生成（tar.gz, zip, rpm, deb）および CI ビルド。
- **INSTALL_LINUX.md / INSTALL_LINUX.en.md**: Fedora, RHEL, Ubuntu, Debian 向けの導入手順および実機検証情報。

## リリース前の確認事項

1. 各 OS（Windows 11、macOS Apple Silicon、Linux Fedora 44）でのパッケージビルドとスモークテスト。
2. 音声ループバックキャプチャ（WASAPI / Core Audio Process Tap / PulseAudio・PipeWire）の動作確認。
3. 再生中楽曲（GSMTC / Music Center / Apple Music / MPRIS2）のメタデータ表示確認。
4. 設定画面（フォントサイズ・ファミリ・太字、スペクトラムの自動調整ボタンおよびリサイズ時チェックボックス、コンボボックス表示、言語切替、テーマ、ウィンドウサイズプリセット）の変更・保存・キャンセル動作確認。
5. Linux 環境（Fedora 44 / Ubuntu 等）での設定画面表示（テーマ一覧 Treeview の文字潰れがないこと、UI 余白の確認）。
6. 全単体テスト（tests/*.py）が正常に PASS することの確認。
