# v1.2.1 main統合記録

パッチアップデート v1.2.1 に向けた main ブランチの変更・統合記録です。
主な変更点は、Windows 環境で Wune を起動したままスリープ（サスペンド）に入り復帰した際のクラッシュ防止、および復帰直後の初回再生時にスペクトラムが反応しなくなる問題の解消（Issue #185）です。

| 機能 | 統合内容・対応 | 関連 Issue / PR |
| --- | --- | --- |
| WASAPI COM `S_FALSE` 許容 | `SoundCard` の `check_error` において非負の HRESULT（`S_FALSE` = 0x00000001 など）を正常終了として許容し、スリープ復帰時の一時的停止例外（`Error 0x100000001`）によるクラッシュを防止 | #185, PR #186 |
| 安全なレコーダー終了保証 | `_Recorder.__exit__` のパッチにより `Stop()` 例外時も警告にとどめ、`finally` で COM インターフェースポインタを確実に解放 | #185, PR #186 |
| スリープ復帰の自動検知 & ストリーム再初期化 | メインループでウォールクロックの時間跳躍（`wall_dt > 2.0s`）を検知し、復帰直後に `AudioSpectrum.restart_capture()` でループバックストリームを再生成。復帰直後の初回再生から即座にスペクトラムが反応するよう改善 | #185, PR #186 |
| キャプチャ例外時の無音フォールバック | `AudioSpectrum.step()` において、デバイス切断等の一時的例外をキャッチしてゼロ（無音）配列を返し、描画ループの中断を防止 | #185, PR #186 |
| クローズ処理の全プラットフォーム安全化 | WASAPI, PulseAudio, macOS 各バックエンドの `close()`、`AudioSpectrum.close()`、および `App.run()` の `finally`句で例外を捕捉し、プロセス終了を安全に保護 | #185, PR #186 |
| 起動時のチラつき（ウィンドウジャンプ）防止 | 初期化中はウィンドウを非表示（`pg.HIDDEN`）のまま保持し、保存位置の復元および初回フレームの描画完了後に表示（`show()`）することで、画面中央から保存位置への移動チラつきを解消 | #184 |

## 主要な統合コミット

| コミット | 内容 |
| --- | --- |
| `0aad278` | fix(capture): prevent crash on sleep resume by permitting WASAPI Stop S_FALSE (#185) |
| `59e521f` | fix(capture): refresh audio capture backend on system sleep resume |
| `06dcc17` | Merge pull request #186 from yshdsnd/fix/issue-185-wasapi-sleep-s-false-crash |

## ファイル群の確認

- **wune/soundcard_compat.py**: WASAPI COM 呼び出しにおける `S_FALSE` 許容フィルター、`_Recorder.__exit__` での例外耐性と COM 解放保証。
- **wune/capture.py**: 各バックエンドの `close()` における例外保護。
- **wune/spectrum_audio.py**: `restart_capture()` による安全なキャプチャバックエンド再初期化、`step()` での無音フォールバック、`close()` の安全化。
- **wune/app.py**: メインループにおけるスリープ復帰時間跳躍検知、`on_system_resume()` によるストリームリフレッシュとクロックリセット。
- **tests/test_soundcard_compat.py**: COM エラーフィルタの各種 HRESULT テスト、`_Recorder.__exit__` の例外耐性とポインタ解放テスト。
- **tests/test_capture.py**: キャプチャ例外発生時の無音フォールバック、`restart_capture()` の再構築・カスタムバックエンド除外・エラーハンドリングテスト。
- **tests/test_smart_idle.py**: 時間跳躍によるスリープ復帰検知および `App.on_system_resume()` のイベントループ連携テスト。

## リリース前の確認事項

1. 各 OS（Windows 11、macOS Apple Silicon、Linux）での単体テスト全件 PASS。
2. Windows 実機でのスリープおよびスリープ復帰確認（クラッシュしないこと、復帰後の初回再生からスペクトラムが反応すること）。
3. macOS / Linux での通常起動・音声キャプチャ・終了処理が正常に動作すること。
