# v1.4.0 main統合記録

[English](en/integration-v1.4.0.md) / **日本語**

マイナーアップデート v1.4.0 に向けた main ブランチの変更・統合記録です。
主な変更点は、ヘッダーの1行統合とリスニングモード向けデザイン刷新（PR #205）、ボーダレスウィンドウモードの実装（PR #206）、設定「情報」タブおよび出力情報窓のバージョン表示と全OS対応の既定出力デバイス自動追従／手動再読み込み（PR #207）、非フォーカスウィンドウのワンクリック反応（PR #209）、および自動レイアウト・縮小下限・dB目盛り重なりの修正（PR #211）です。

| 機能 | 統合内容・対応 | 関連 Issue / PR |
| --- | --- | --- |
| ヘッダー1行統合・デザイン刷新 | メニューボタンを三本線（ハンバーガー）アイコン化し、Now Playing（左詰め）と固定幅センタリングのテーマバッジ（超過時は省略）を最上部1行に統合。通常時は控えめなリスニングモード配色とし、ホバー時のみ強調表示 | PR #205 |
| ボーダレスウィンドウモード | タイトルバーと枠を非表示にする `borderless` 設定（「一般」タブ / メニュー / `F10` キー）を追加。ボーダレス表示中はウィンドウ内ドラッグによる移動と、四辺・四隅ドラッグによるリサイズに対応 | PR #206 |
| バージョン表示（設定「情報」タブ・出力情報窓） | ボーダレス時でもバージョンを確認できるよう、設定「情報・文字」タブにバージョン表記をインライン表示。さらに出力情報窓の右端へのバージョン表示（`info_show_version`、既定 OFF）を追加し、狭幅時は動的に `\| ` 区切りを挿入 | PR #207 |
| 既定出力デバイスの自動追従 & 手動再読み込み | OS既定出力（`output_device = None`）使用時、Windows（WASAPI）・macOS（Core Audio Tap / SoundCard）・Linux/FreeBSD（PulseAudio/PipeWire）で既定出力デバイスやサンプルレートの変更を約2秒周期で自動検知し再接続。ストリーム切断時の自動復帰に加え、全OS共通の手動再読み込み（`F5` / メニュー項目）を追加 | PR #207 |
| 非フォーカス時のワンクリック反応 | `SDL_MOUSE_FOCUS_CLICKTHROUGH=1` を `pg.init()` 前に有効化し、ウィンドウにフォーカスがない状態でもメニューボタンやテーマバッジを1クリックで前面表示＋即時反応するよう改善 | #208, PR #209 |
| 自動レイアウト不具合と dB 目盛り重なりの修正 | 全画面→設定→キャンセル時にウィンドウサイズ基準で自動調整が走る不具合を修正。LED段数／バー間隔の自動調整ON時は最小構成（10段／1px）まで縮小可能にし、縦幅が狭い際の dB 目盛りを動的間引き・極小時は数値省略するよう改善 | #210, PR #211 |

## 主要な統合コミット

| コミット | 内容 |
| --- | --- |
| `1c1f2b6` | feat(ui): unify header into single row with listening-mode styling (#205) |
| `4380249` | feat(ui): add optional borderless window mode with edge resizing and dragging (#206) |
| `be52f93` | feat: add version display in Info tab/info bar and automatic/manual audio device reload (#207) |
| `6bc3931` | Merge pull request #209 from yshdsnd/fix/issue-208-mouse-focus-clickthrough |
| `135533c` | Merge pull request #211 from yshdsnd/fix/issue-210-auto-layout-and-db-labels |

## ファイル群の確認

- **wune/config.py / wune/settings.py / wune/appearance.py**: `borderless` および `info_show_version` 設定の追加・永続化・プレビュー対応。
- **wune/application_menu.py / wune/locales/en.json / wune/locales/ja.json**: 三本線メニューアイコン化、`borderless` (`F10`) および `reload_audio` (`F5`) メニュー項目と日英翻訳キー追加。
- **wune/app.py**: ボーダレス切替・移動・端リサイズ、`SDL_MOUSE_FOCUS_CLICKTHROUGH` 有効化、2秒周期の自動デバイス追従・`F5` 手動再読み込み、全画面キャンセル・縮小リサイズ時の自動調整整合性修正。
- **wune/capture.py / wune/capture_macos.py / wune/tap_macos.py / wune/tap_backend.m / wune/soundcard_compat.py / wune/spectrum_audio.py**: 各OSバックエンドでの `needs_restart()` 変更検知およびストリームエラー自動復帰の実装。
- **wune/layout.py / wune/renderer.py / wune/settings_dialog.py**: ヘッダー1行統合、`resizable_minimum_window_size()` 導入、出力情報窓の動的セパレータ付きバージョン描画、dB 目盛りの動的間引き・省略。
- **tests/**: `test_borderless.py`, `test_info_version.py`, `test_audio_reload.py`, `test_issue_210_auto_layout.py` 等の新規テスト追加および既存テスト更新。

## リリース前の確認事項

1. 全単体テスト（604件）が全プラットフォーム（Windows / macOS / Linux / FreeBSD）で PASS すること。
2. ボーダレスモードの切替（`F10` / メニュー / 設定）、ドラッグ移動、端リサイズ、および出力情報窓のバージョン表示が正常に動作すること。
3. 実機にて OS 既定出力デバイスの切り替え・サンプルレート変更時の自動追従、および `F5` / メニューからの手動再読み込みが機能すること。
4. LED段数・バー間隔の自動調整ON時に最小サイズまでスムーズに縮小でき、狭い縦幅でも dB 目盛りが重ならないこと。
