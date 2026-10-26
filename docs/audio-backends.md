# 音声キャプチャバックエンド

[English](en/audio-backends.md) / **日本語**

`AudioSpectrum` は `CaptureBackend` インターフェースを通じて PCM データを取得します：`record(numframes)`、`sample_rate`、`channels`、`device_name`、および `close()`。FFT のキャリブレーション、周波数帯の割り当て、無音ゲーティング、レベルエンベロープなどの処理は全プラットフォームで共通です。
バックエンドの実際のサンプルレートによって FFT の周波数軸が決定されます。
バックエンドを `AudioSpectrum` に渡すと、そのライフタイムはスペクトラム側に委譲され、初期化失敗時および通常のシャットダウン時の両方で適切にクローズされます。

- **Windows**: WASAPI 共有ステレオループバック。明示的なレートが指定されていない限り、選択されたレンダリングエンドポイントのミックスレートを使用します。SoundCard 互換レイヤー（`wune/soundcard_compat.py`）により、PROPVARIANT の確保サイズ不足とメモリリークを解消し、`IAudioClient::Stop` 呼び出し時の非負 COM コード（スリープ／サスペンド復帰時の一時的停止例外 `S_FALSE` = 0x00000001 など）を正常として許容してクラッシュを防止。さらにシステム復帰時にキャプチャストリームを安全に再初期化し、即座に新しい再生音を拾えるようにしています。
- **macOS 14.2+**: Core Audio Process Tap によりシステムの再生音を標準でキャプチャ。利用できない場合は仮想ループバック入力を使用可能。障害時でも内蔵マイクを無断で選択することはありません。デバイス設定で入力を明示的に選択することも可能です。ネイティブ Tap は出力先の実効レートを報告し、`Config.sample_rate` へのリサンプリングは行いません。
- **Linux**: PulseAudio / PipeWire モニターのループバックキャプチャ（`wune/capture_linux.py`）。`libpulse-simple` C API または CLI フォールバック（`parec`, `pw-record`）経由でデフォルト出力シンクのモニターソースを自動判別。仮想オーディオケーブル不要で、出力ネイティブのサンプルレートで非圧縮ステレオ PCM を取り込みます。
- **FreeBSD**: SoundCard の PulseAudio バックエンドを用いたループバックモニターキャプチャ。`wune/soundcard_compat.py` によるプラットフォーム偽装とガードを適用し、FreeBSD 上で Pure-Python の SoundCard 0.4.6 から PulseAudio（基盤の `hw.snd` / OSS サブシステム）を安定して直接呼び出します。取り込み元モニターは `pactl` または `pavucontrol` で設定します。

## macOS 補助ライブラリ

Apple Silicon 向け補助ライブラリのソースは `wune/tap_backend.m` です。ソース実行時は、`tap_macos.py` が `/usr/bin/clang` と macOS SDK を用いて未生成の `libwune_tap.dylib` を自動コンパイルします（Xcode Command Line Tools が必要です）。
配布用バイナリでは、このライブラリを `Contents/Frameworks/libwune_tap.dylib` 内に同梱し、アドホックコード署名を実施します。

macOS の音声ワークフローでは、補助ライブラリをビルドした上で、モックされたオーディオデバイスを用いてキャプチャルーティング、バッファ蓄積、フォールバックポリシー、クローズ処理をテストします。ヘッドレス CI では実際の再生音キャプチャや録音権限の付与は行われません。
実機 Mac でのハードウェアテストは以下で実行できます：

```sh
WUNE_TEST_LIVE_AUDIO=1 python -m unittest discover -s tests -p test_tap_macos.py
```

実機検証項目には、権限プロンプトのハンドリング、無音／再生状態の遷移、44.1 / 48 / 96 kHz 出力、ステレオ分離、正常終了が含まれます。

## Linux 音声キャプチャ

Linux での音声キャプチャは `wune/capture_linux.py` を使用し、ネイティブ PulseAudio サーバーおよび `pipewire-pulse` を用いた PipeWire 環境の両方をサポートします：
- **ネイティブ C ライブラリバインディング**: `ctypes` を用いて `libpulse-simple.so.0` および `libpulse.so.0` に直接リンクし、最小の遅延と低 CPU 負荷を実現。
- **サブプロセスフォールバック**: `libpulse-simple` が利用できない場合、`parec` または `pw-record` から PCM フレームをストリーミング取得するフォールバックを搭載。
- **自動モニター検出**: デフォルトのオーディオシンクを照会し、その `.monitor` ソースへ自動接続。音声出力先が別のカードやデバイスへ切り替わった場合、Wune を再起動することで新しいデフォルトデバイスへ追従します。

## FreeBSD 音声キャプチャ

FreeBSD でのキャプチャは、`wune/soundcard_compat.py` で `SoundCard` ライブラリをラップして利用します：
- **プラットフォーム互換性**: SoundCard のインポート時に一時的に Linux プラットフォーム識別子を適用し、FreeBSD 上でネイティブの PulseAudio クライアントバックエンドをロード。
- **PulseAudio / OSS 連携**: FreeBSD の音声スタックと PulseAudio または PipeWire 経由で連携。デフォルトの出力シンクのモニターソース（`<sink>.monitor`）からリアルタイムにシステム音声をキャプチャ。
- **デバイスルーティング**: `pactl set-default-sink` / `pactl set-default-source` で設定するか、GUI ツール `pavucontrol` で動作中にインタラクティブに取り込み元を切り替え可能。
