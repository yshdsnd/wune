# FreeBSD版 Wune 導入ガイド

[English](INSTALL_FREEBSD.en.md) / **日本語**

80～90年代のミニコンポをイメージしたLEDスペクトラムアナライザー「Wune」の、FreeBSD 向けインストールと利用手順です。

## 動作環境

- **対応OS / アーキテクチャ**: **FreeBSD 14.x / 15.x（amd64）**（FreeBSD 15.1-STABLE 実機にて動作確認済み）
- **音声サーバー**: **PulseAudio** または **PipeWire**（PulseAudio互換レイヤー）
- **ディスプレイ**: **X11** または **Wayland**
- **ステレオ音声を再生できる出力デバイス**（スピーカー、ヘッドホン等）
- **推奨**: 日本語フォント（Google Noto Fonts）、`playerctl`（Now Playing 曲名表示用）

## インストール手順

### 1. 依存パッケージの導入

FreeBSD の `pkg` を使用して、必要なランタイム、ライブラリ、フォントをインストールします。

```bash
sudo pkg install py312-tkinter py312-cffi py312-numpy pulseaudio playerctl \
    v4l_compat noto-basic noto-jp dejavu pkgconf sdl2 sdl2_image sdl2_mixer sdl2_ttf
```

> [!NOTE]
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
