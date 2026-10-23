# FreeBSD版 Wune 導入ガイド

[English](INSTALL_FREEBSD.en.md) / **日本語**

80～90年代のミニコンポをイメージしたLEDスペクトラムアナライザー「Wune」の、FreeBSD 向けインストールと利用手順です。

## 動作環境

- **対応OS / アーキテクチャ**: **FreeBSD 14.x / 15.x（amd64）**（FreeBSD 15.1-STABLE 実機にて動作確認済み）
  *(実機検証構成: i915kms + modesetting Xorg, Openbox, Firefox + PulseAudio, `devel/freebsd` ブランチ)*
- **音声サーバー**: **PulseAudio** または **PipeWire**（PulseAudio互換レイヤー）
- **ディスプレイ**: **X11** または **Wayland**
- **ステレオ音声を再生できる出力デバイス**（スピーカー、ヘッドホン等）
- **推奨**: 日本語フォント（Google Noto Fonts）、`playerctl`（Now Playing 曲名表示用）、`pavucontrol`（音声入出力・録音ストリームの切り替え用）

## インストール手順

### 1. 依存パッケージの導入

FreeBSD の `pkg` を使用して、必要なランタイム、ライブラリ、フォント、管理ツールをインストールします。

```bash
sudo pkg install py312-tkinter py312-cffi py312-numpy pulseaudio pavucontrol playerctl \
    v4l_compat noto-basic noto-jp dejavu pkgconf sdl2 sdl2_image sdl2_mixer sdl2_ttf
```

> [!NOTE]
> - `pavucontrol` は GUI で出力デバイスや録音ストリーム（Wune の取り込み元）を切り替える際に便利です。
> - `v4l_compat` は Pygame ビルド時に必要な Video4Linux2 互換ヘッダーを提供します。
> - `noto-basic`, `noto-jp`, `dejavu` は UI や日本語デバイス名のフォント描画に使用されます。

### 2. ソースコードの取得と仮想環境の作成

システムにインストールされた NumPy や CFFI、Tkinter を活用するため、`--system-site-packages` を指定して仮想環境を作成します。

```bash
git clone https://github.com/yshdsnd/wune.git
cd wune
python3.12 -m venv --system-site-packages .venv
```

### 3. Python パッケージのインストール

```bash
# SoundCard のインストール
.venv/bin/pip install SoundCard==0.4.6

# Pygame のビルドとインストール（FreeBSD のインクルード/ライブラリパスを指定）
CFLAGS="-I/usr/local/include" LDFLAGS="-L/usr/local/lib" .venv/bin/pip install pygame
```

### 4. 起動

```bash
.venv/bin/python main.py
```

## 音声の取得について

- 起動時に FreeBSD 上の PulseAudio / PipeWire の既定の出力先（スピーカーやヘッドホン）を検出し、その **ループバックモニターソース**（`<device>.monitor`）から音声をリアルタイムキャプチャします。
- Wune 自身は音を再生したり、出力先を変更したりしません。普段お使いの音楽プレイヤーやブラウザで音声を再生してください。
- **サンプリングレートの既定値**: 未指定時のキャプチャサンプリングレートは **48 kHz** です。
- 情報欄（`I`キーで表示／非表示）で、取得中のデバイス名・サンプルレート・チャンネル数を確認できます。

### 既定の出力先と取り込み用モニターの設定（pactl）

PulseAudio 経由で特定の出力先（例: HDMI 出力の `oss_output.dsp3`）の音声を確実に Wune で取り込むには、デフォルトの出力先（sink）とモニターソース（source）の両方を `pactl` で明示的に指定します。

```bash
# 利用可能な sink（出力先）と source（モニター元）の一覧を確認
pactl list short sinks
pactl list short sources

# 既定の出力先とループバックモニターを設定（例: oss_output.dsp3 の場合）
pactl set-default-sink oss_output.dsp3
pactl set-default-source oss_output.dsp3.monitor
```

> [!IMPORTANT]
> - 環境によっては `set-default-source` のみでは適切にモニターが選択されない場合があります。必ず `set-default-sink` と `set-default-source` の**両方を設定**してください。
> - 両方を設定しておくことで、Wune を終了・再起動しても起動時から意図したモニター（例: `oss_output.dsp3.monitor`）が正しく選択されます。

### 動作中の取り込み先切り替え（pavucontrol）

GUI ボリューム調整ツール `pavucontrol` を使用して、Wune の動作中にも取り込み先を切り替えることができます。

1. `pavucontrol` を起動します。
2. **「録音」タブ** を開きます。
3. アプリケーション一覧に表示されている **Wune** のドロップダウンメニューから、目的の出力デバイスの Monitor（例: `Monitor of oss_output.dsp3`）を選択します。

> [!NOTE]
> `pavucontrol` で取り込み元を切り替えた場合、スペクトラムアナライザーの描画はリアルタイムに切り替わりますが、Wune の情報表示（`I` キーで表示されるモニター名）は起動時の取得名のまま追従しません（今後の改善課題）。

### 動作確認状況と今後の検証範囲

実機環境（FreeBSD 15.1-STABLE, i915kms + modesetting Xorg, Openbox, Firefox + PulseAudio）にて、以下が確認済みです。
- PulseAudio 経由での起動、再生音取り込み、スペアナ描画
- ウィンドウのリサイズ（拡大・縮小）
- 設定の保存および次回起動時の復元
- 設定画面のレイアウト・フォント表示（ダイアログサイズが過大にならず、文字の重なりがないこと）

※ OS 再起動後の PulseAudio 既定設定の維持、OSS 直接（`/dev/dsp`）や `virtual_oss`、PipeWire 経由での動作、フルスクリーン表示の安定性については今後の検証範囲です。

## Now Playing（再生中楽曲表示）について

FreeBSD 上で Spotify、ブラウザ、メディアプレイヤー等の再生中情報を取得するために、MPRIS2（`playerctl`）を利用します。
上記パッケージ導入で `playerctl` をインストールしておくと、曲名やアーティスト名が画面上部に自動表示されます。

## 基本操作とキーボードショートカット

メイン画面をクリックしてからキーを押してください。

| 操作 | 機能 |
| --- | --- |
| F2 | 設定画面を開く |
| 左上のメニュー、または画面を右クリック | 設定・全画面切り替え・終了 |
| Alt+Enter / F11 | 全画面／通常ウィンドウの切り替え |
| Space | 表示の一時停止／再開（音楽再生は止まりません） |
| T または右上のテーマバッジをクリック | 次の配色テーマへ切り替え |
| I | 出力デバイスやサンプルレートの情報欄を表示／非表示 |
| Esc | 全画面時は通常表示に戻る。通常表示時は終了確認ダイアログ |
| Q | 終了確認ダイアログ（全画面中も同じ） |
| ウィンドウの端をドラッグ | ウィンドウのリサイズ（LED表示も滑らかにスケーリング） |

## 設定ファイルの保存先

設定はユーザーディレクトリに自動保存されます。

- **設定ファイル**: `~/.config/Wune/settings.json`
- 設定を初期化したい場合は、`main.py --reset-settings` を指定して起動してください。
