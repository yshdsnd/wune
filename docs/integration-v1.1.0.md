# v1.1.0 main統合記録

Issue #98。統合元は公開済みv1.0.1、`91975d05fcf1880bb650253c103fe6bf09e085e4`。
共通祖先 `b8cc6a5b45ba03ed330e5ba6e3163da695444279` からの22コミット（マージを含む）、41変更ファイルを確認しました。
最終段階の基点mainは `399a57a9a82ad5f15f22c3862ef7bd41a3568c9f` です。

| 機能 | 統合先／扱い |
| --- | --- |
| LEDタイル／座標キャッシュ | #99。mainの背景描画を維持し、表示比較テストを追加 |
| 設定保存先・言語・フォント | #100。Windowsの設定項目・保存形式を維持 |
| 音声バックエンド・Core Audio Tap | #101。初期化途中の解放・ライブラリ自動ビルド・エラー処理を修正 |
| 設定用プロセス・前面表示・画面領域・Macショートカット | #102。Macのみspawn、Windowsの所有ウィンドウを維持 |
| スマートアイドル | #103。mainのメニュー・終了確認・設定通信での復帰条件を追加。追加15ms待機は取り込まず入力周期を維持 |
| Macアプリ・ICNS・日英導入文書・ビルドCI | 今回。現在のソースからCore Audioライブラリを再生成。両OSで配布アプリを移動して起動検証 |
| 同梱済み libwune_tap.dylib | バイナリ自体は取り込まない。ソースとビルド手順を採用し、古いバイナリの混入を防ぐ |
| Macの全画面解除を伴う設定表示 | mainの全画面維持方式を優先。Macの同一モニター全画面復元は未実装で文書化 |
| MacのRelease asset上書き | --clobberを取り込まず、既存assetを保持 |
| tests全般 | 片側で置換せず追加・統合。Macで全テスト実行、Windows固有の実ネイティブ検証のみ条件付き |

## 統合元コミット

以下の通常コミットと、そのマージコミットを上表の機能単位で再点検しました。

| コミット | 内容 |
| --- | --- |
| `666cf28` | Support macOS settings path, fonts, shortcuts, and capture abstraction (#75) |
| `d9deace` | Introduce capture backend boundary and fix macOS capture issues (#75) |
| `d73ef37` | Implement native Core Audio Process Tap capture and fix macOS text rendering (#75) |
| `8bdc6cb` | Isolate Tkinter settings dialog in spawned process on non-Windows platforms (#75) |
| `bf05ad1` | Run SettingsDialogLifecycleTests in isolated subprocess to prevent cross-thread Tcl crash on Windows |
| `36d71c9` | Address PR #89 review comments for macOS Core Audio tap capture (#75) |
| `f81072a` | Ensure macOS routing tests mock tap capability appropriately across host platforms |
| `6f10759` | Detect macOS system UI language via CoreFoundation preferred languages (#75) |
| `8fa5caf` | Add standalone macOS packaging, application icon, and CI workflow (#75) |
| `91d0dd5` | Skip live Core Audio tap test when audio hardware is unavailable on headless CI |
| `15a1cc5` | Bring settings dialog window to front and activate process on open |
| `9b089eb` | Split platform installation instructions and reorganize README (#92) |
| `86173cc` | Restrict macOS builds to Apple Silicon arm64 and document verification on macOS 27 |
| `fa13320` | Fall back to online displays and safe pygame init in macOS work_areas |
| `f20241f` | Update macOS Gatekeeper bypass steps for modern macOS in documentation |
| `b99875d` | perf(renderer): precompute LED grid and tiles to reduce CPU usage |
| `91db07d` | perf(app): add smart idle throttling to eliminate WindowServer and idle CPU load |

## ファイル群の確認

- app / renderer / settings / settings_dialog / soundcard_compat / spectrum_audio / system_locale / window_geometry / main / packaging entry: #99〜103の調整済み実装を維持。
- capture / capture_macos / tap_macos / tap_backend と音声テスト: #101の改良済み実装を維持。
- i18n / icons / settings / dialog / cleanupテスト: 両系統を維持し、Mac基礎・ウィンドウ・キャッシュの独立した回帰テストも追加済み。
- Wune-macos.spec / build_macos / workflow / package_smoke / icon / packagingテスト: 今回取り込み、現在のmainのメニュー・背景・設定プロセスまで検証。
- README / INSTALL / development / packaging / packaging README / assets README: Windows固有機能を残して日英を統合。
- macos_capture_comparison: 統合元の設計比較資料として保存。現在の動作条件はaudio-backends.mdを優先。

## リリース前に残る確認

このPRは配布物生成までで、タグ・Release公開・参照ブランチ削除は行いません。
両OSの候補ZIPを同じコミットから作り、実機の音声権限・再生・停止・全画面／Spaces・設定・背景・復帰・既存設定引継ぎを確認します。
Macのネイティブ依存物の通知確認、配布ZIPの相対リンク、SHA-256、署名の表示も確認します。
確認後にv1.1.0を公開し、devel/v2.0の先端が統合元から進んでいないことを再確認して整理します。
