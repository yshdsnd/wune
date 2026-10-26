# macOS ネイティブ音声キャプチャの比較: Core Audio Process Tap 対 ScreenCaptureKit

[English](en/macos_capture_comparison.md) / **日本語**

v1.0.1 時点での設計比較記録です。現在のルーティング、対応入力、制限事項については [音声バックエンド](audio-backends.md) を参照してください。
以下の相対的な性能に関する記述はベンチマーク値ではなく設計上の推定値です。

本書は、Wune における仮想ドライバ不要のシステム音声キャプチャ実現（Issue #75）に向けて、Apple の **Core Audio Process Tap** API と **ScreenCaptureKit** を比較検討した記録です。

---

## 比較概要

| 評価軸 | **Core Audio Process Tap** | **ScreenCaptureKit (SCStream)** |
| :--- | :--- | :--- |
| **主たる設計意図** | プロセス／システム出力向けの**音声専用 Tap** | 画面・ウィンドウ録画（音声ストリーム付き） |
| **仮想ドライバが必要か？** | ❌ **不要**（直接ネイティブ PCM を取得） | ❌ **不要**（直接ネイティブ PCM を取得） |
| **実行時負荷・遅延** | **最小**（Core Audio HAL レベルで直結） | 中程度（ScreenCaptureKit デーモン経由） |
| **ステレオ／サンプルレート** | 出力先ネイティブ ASBD（44.1k/48k float32 ステレオ） | 設定可能（`sampleRate=48000`, `channels=2`） |
| **対応 macOS 最小バージョン** | **macOS 14.2 (Sonoma)** | **macOS 13.0 (Ventura)** |
| **TCC 権限プロンプト** | **システム音声の録音**（`NSAudioCaptureUsageDescription`） | **画面およびシステム音声の録画** |
| **プライバシー面での印象** | 明確（音声のみでカメラや画面への不安なし） | 懸念されやすい（「画面収録」権限を要求される） |
| **Python 実装方法** | C API（`ctypes` または薄い C/Swift 補助 dylib） | Objective-C（`pyobjc-framework-ScreenCaptureKit`） |

---

## 詳細比較

### 1. スペクトラムアナライザへの適合性（低負荷・連続 PCM 取得）
- **Core Audio Process Tap**:
  - 仮想デバイスを作成せず、出力音声を直接分岐取得することに特化。
  - 軽量な Core Audio IO ブロック／コールバックを通じて PCM フレームを直接供給。
  - WindowServer、GPU、ビデオフレームアロケータは一切関与しない。
- **ScreenCaptureKit**:
  - 主に画面やウィンドウのキャプチャパイプライン。映像出力を無効（`capturesAudio = true`）にしても、キャプチャセッションはディスプレイサーバーデーモンと連携。
  - バックグラウンドで常駐するビジュアライザーとしては、メモリおよびプロセス負荷がやや高め。

### 2. 権限とユーザー体験（TCC）
- **Core Audio Process Tap**:
  - `Info.plist` に `NSAudioCaptureUsageDescription` の記述が必要。
  - macOS 14.2+ では、明示的にシステム音声の録音許可のみを求めるダイアログが表示される。
  - 音声ビジュアライザーが音声権限を要求する理由はユーザーにとっても直感的で納得しやすい。
- **ScreenCaptureKit**:
  - システム標準の **「画面およびシステム音声録音」** 許可プロンプトをトリガー。
  - 音声のみのツールに対して画面録画権限を付与することに抵抗を感じるユーザーが多い。

### 3. Python 実装／ネイティブ連携経路
- **Core Audio Process Tap**:
  - C 関数群: `AudioHardwareCreateProcessTap`, `AudioHardwareDestroyProcessTap`, `AudioDeviceCreateIOProcIDWithBlock`。
  - Python 標準の `ctypes` から `CoreAudio.framework` を直接呼ぶか、`wune/` 内にコンパイル済みの薄い C/Swift 補助ライブラリを同梱することで容易に呼び出し可能。
- **ScreenCaptureKit**:
  - `pyobjc-framework-ScreenCaptureKit` および Objective-C のランループ／ディスパッチキュー連携が必要。
  - PyInstaller パッケージに PyObjC ホイールを含めるとバイナリサイズと依存関係が大幅に肥大化。

---

## Wune での採用方針

1. **本番バックエンド**: macOS 14.2+ 向けの主要ネイティブキャプチャ方式として **Core Audio Process Tap** を採用。
2. **フォールバック方針**:
   - macOS 14.2+: Core Audio Process Tap を使用。
   - macOS < 14.2: サードパーティ製仮想オーディオループバックドライバ（BlackHole 等）の導入を推奨。
   - 内蔵マイクへの無断フォールバックは絶対に行わない。
