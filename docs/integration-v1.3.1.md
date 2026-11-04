# v1.3.1 main統合記録

[English](en/integration-v1.3.1.md) / **日本語**

パッチアップデート v1.3.1 に向けた main ブランチの変更・統合記録です。
主な変更点は、高解像度（4K等）での最大化・全画面表示時にスペクトラム右端および目盛りラベルが見切れる不具合の解消（Issue #200）、および無音・アイドル時にディスプレイ消灯やシステムスリープが恒常的に阻害されていた問題の解消（Issue #201）です。

| 機能 | 統合内容・対応 | 関連 Issue / PR |
| --- | --- | --- |
| 4K/最大化時のスペクトラム右端見切れ修正 | 縦配置（垂直チャンネル配置・水平周波数）において、4Kディスプレイ等の大画面最大化時に丸め誤差やマージン不足によって右端のLEDバーおよび「20k」目盛りラベルがウィンドウ枠外へ約36pxはみ出しクリッピングされる問題を解消。配置幅計算の最適化と境界クランプ処理を導入 | #200, PR #202 |
| 無音・アイドル時の画面消灯／スリープ抑止解除 | Pygame/SDL2が初期化時に永続設定していた画面消灯・スリープ抑止（`ES_DISPLAY_REQUIRED`）を制御する `SleepInhibitor` を導入。音楽再生中は快適な鑑賞のため消灯を抑止し、無音状態が10秒継続した場合（および起動時無音・一時停止時）は即座に抑止を解除してOSの画面消灯・省電力スリープ（有機EL焼き付き防止）を正常に機能させるよう改善 | #201, PR #203 |

## 主要な統合コミット

| コミット | 内容 |
| --- | --- |
| `6e2b9ec` | fix(layout): prevent spectrum clipping on right edge when maximized or fullscreen (#200) |
| `bc56f27` | Merge pull request #202 from yshdsnd/fix/issue-200-vertical-layout-overflow |
| `a75f62e` | fix(power): release screen turn-off and sleep suppression during idle (#201) |
| `98ef881` | Merge pull request #203 from yshdsnd/fix/issue-201-idle-sleep |

## ファイル群の確認

- **wune/layout.py**: 最大化・大画面表示時における右端クリッピング防止、グループX座標の境界クランプ処理。
- **wune/sleep_inhibitor.py**: OS画面セーバー／消灯（`pg.display.set_allow_screensaver`）および Win32 `SetThreadExecutionState` によるスリープ抑止の安全な状態管理クラス。
- **wune/config.py**: アイドル判定タイムアウト設定 `sleep_inhibit_idle_seconds`（デフォルト 10.0 秒）の追加。
- **wune/app.py**: 音声再生検知と連動した `SleepInhibitor` の起動・更新・無音時解放、一時停止および終了時の確実な解放処理。
- **tests/test_layout.py**: 4K（3840x2160）最大化・全画面時におけるスペクトラム右端・目盛りラベルのはみ出し防止リグレッションテスト。
- **tests/test_sleep_inhibitor.py**: 無音継続・再生再開・一時停止・スリープ復帰・終了処理におけるスリープ抑止状態遷移の単体テスト（7件）。

## リリース前の確認事項

1. 全単体テスト（544件）が全プラットフォームで PASS すること。
2. 4Kディスプレイ環境でウィンドウを最大化した際にスペクトラム右端と「20k」ラベルが欠けずに表示されること。
3. Windows実機にて音楽再生を停止・無音にして10秒経過後、OSのディスプレイ電源オフおよびスリープが正常に動作すること。
