# Linux版 Wune 導入ガイド

[English](INSTALL_LINUX.en.md) / **日本語**

80～90年代のミニコンポをイメージしたLEDスペクトラムアナライザー「Wune」の、Linux向けインストールと利用手順です。

## 動作環境

- **対応アーキテクチャ**: **x86_64（64ビット PC）**
- **実機確認済みディストリビューション**:
  - **Fedora 44 (x86_64)**（実機にて動作確認済み）
  - **RHEL / CentOS Stream、Ubuntu / Debian 等**: パッケージ（RPM / DEB）やスタンドアロンアーカイブの提供および CI（Ubuntu 24.04）でのビルド・スモークテストは実施していますが、実機環境での確認は行っていません。これらのディストリビューションで動かしていただける方を募集しています。問題や気づいた点があれば、ぜひ **[GitHub Issues](https://github.com/yshdsnd/wune/issues)** までご報告ください。
  - ※ 仕組み上は PulseAudio または PipeWire（`pipewire-pulseaudio`）が動作する一般的な Linux デスクトップ環境に対応しています。
- **音声サーバー**: **PulseAudio** または **PipeWire**（PulseAudio互換レイヤー）
- **ディスプレイ**: **X11** または **Wayland**（XWayland）
- **ステレオ音声を再生できる出力デバイス**（スピーカー、ヘッドホン等）
- **推奨**: 日本語フォント（Google Noto Sans CJK JP など）、`playerctl`（Now Playing 曲名表示用）

## パッケージの選択とインストール

[GitHub Releases](https://github.com/yshdsnd/wune/releases) からお使いのディストリビューションに合わせたパッケージをダウンロードしてください。

### 1. Fedora / RHEL / CentOS Stream（RPM パッケージ）

RPM パッケージをダウンロードして `dnf` でインストールします。デスクトップメニューや `/usr/bin/wune` コマンドが自動登録されます。

```bash
# パッケージのインストール
sudo dnf install ./Wune-vX.Y.Z-1.x86_64.rpm
```

- **起動**: アプリケーションランチャーから「Wune」を選択するか、ターミナルで `wune` を実行します。
- **アンインストール**: `sudo dnf remove wune`

### 2. Debian / Ubuntu / Linux Mint（DEB パッケージ）

DEB パッケージをダウンロードして `apt` でインストールします。

```bash
# パッケージのインストール
sudo apt install ./wune_X.Y.Z_amd64.deb
```

- **起動**: アプリケーションランチャーから「Wune」を選択するか、ターミナルで `wune` を実行します。
- **アンインストール**: `sudo apt remove wune`

### 3. スタンドアロンアーカイブ（tar.gz / zip）

Python のインストールは不要で、PulseAudio / PipeWire が動作する 64ビット Linux 環境で動作するスタンドアロンアーカイブです（Fedora 44 実機、および Ubuntu 24.04 CI にて動作確認。音声キャプチャ用に `pulseaudio-libs` または `libpulse0` が必要です）。

```bash
# tar.gz の展開と起動
tar -xzf Wune-vX.Y.Z-linux-x64.tar.gz
cd Wune
./Wune
```

デスクトップランチャーへ登録したい場合は、同梱されている `wune.desktop` を `~/.local/share/applications/` に配置し、`Exec=` および `Icon=` のパスを展開先の絶対パス（例: `/home/user/apps/Wune/Wune`）に合わせて書き換えるか、PATHの通ったディレクトリ（例: `~/.local/bin/wune`）へシンボリックリンクを作成してください。

## 音声の取得について

- 起動時に Linux の既定の出力先（スピーカーやヘッドホン）を検出し、その **ループバックモニターソース**（`<device>.monitor`）から音声をリアルタイムキャプチャします。
- Fedora や Ubuntu 等の標準構成である PipeWire（`pipewire-pulseaudio`）および PulseAudio の双方に対応しています。
- Wune 自身は音を再生したり、出力先を変更したりしません。普段お使いの音楽プレイヤーやブラウザで音声を再生してください。
- **サンプリングレートの既定値**: Linux/macOS での未指定時のキャプチャサンプリングレートは **48 kHz** です（表示上限は最大 20 kHz）。96 kHz などのハイレゾ音源で 20 kHz 超の高域まで観察したい場合は、ソースコードからの実行時に `main.py` の `Config(sample_rate=...)` を指定して起動してください（配布パッケージ版では現状サンプルレートの個別指定には対応していません）。
- 情報欄（`I`キーで表示／非表示）で、取得中のデバイス名・サンプルレート・チャンネル数を確認できます。
- 出力先を変更した場合は、Wune を再起動してください。

## Now Playing（再生中楽曲表示）について

Linux 上で Spotify、Rhythmbox、VLC、ブラウザ等のメディアプレイヤーから再生中の曲名・アーティスト名を取得するために、MPRIS2 インターフェースを利用します。

- **`playerctl`（推奨）**:
  システムに `playerctl` コマンドがインストールされていると、最も安定して Now Playing 情報を取得できます。
  ```bash
  # Fedora / RHEL
  sudo dnf install playerctl

  # Debian / Ubuntu
  sudo apt install playerctl
  ```
- `playerctl` がない場合も、システム標準の `gdbus` または `dbus-send` を通じて MPRIS2 情報を自動取得します。

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

## 設定ファイルの保存先と初期化

XDG Base Directory 仕様に準拠し、設定やログは以下の場所に保存されます。

- **設定ファイル**: `~/.config/Wune/settings.json`（または `$XDG_CONFIG_HOME/Wune/settings.json`）
- **ログファイル**: `~/.config/Wune/Wune.log`

### 初期化手順
設定を既定値にリセットしたい場合は、以下のコマンドで起動します：
```bash
wune --reset-settings
# スタンドアロンアーカイブの場合:
./Wune --reset-settings
```

## 開発・ソースコードからの実行

開発目的でリポジトリからソースコードを直接実行する場合の手順です。

### 1. 依存パッケージのインストール

#### Fedora / RHEL の場合
```bash
# 必須・ビルド依存パッケージ
sudo dnf install -y python3 python3-devel python3-tkinter pulseaudio-libs \
    sdl2-compat-devel SDL2_image-devel SDL2_ttf-devel freetype-devel

# 推奨
sudo dnf install -y playerctl google-noto-sans-cjk-fonts
```

#### Debian / Ubuntu の場合
```bash
# 必須・ビルド依存パッケージ
sudo apt update
sudo apt install -y python3 python3-dev python3-pip python3-venv python3-tk \
    libpulse0 libpulse-dev libasound2-dev libsdl2-dev libsdl2-image-dev \
    libsdl2-ttf-dev libfreetype6-dev

# 推奨
sudo apt install -y playerctl fonts-noto-cjk
```

### 2. 仮想環境の準備と起動
```bash
# リポジトリの取得
git clone https://github.com/yshdsnd/wune.git
cd wune

# 仮想環境の作成と依存ライブラリの導入
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 起動
python3 main.py
```

## 困ったとき（トラブルシューティング）

| 症状 | 確認すること |
| --- | --- |
| 音声がキャプチャされない／スペクトラムが動かない | 音楽が再生されているか、Spaceキーで一時停止になっていないか確認してください。PipeWire/PulseAudioが動作しているか（`pactl info`）を確認してください。 |
| 設定画面が開かない | `python3-tkinter`（Debian/Ubuntuでは `python3-tk`）がインストールされているか確認してください。スタンドアロンパッケージ版では同梱されているため追加不要です。 |
| 日本語フォントが四角（豆腐）になる | システムに日本語フォント（`google-noto-sans-cjk-fonts` や `fonts-noto-cjk`）がインストールされているか確認してください。 |
| 曲名が表示されない | `playerctl` がインストールされているか確認し、再生中のアプリが MPRIS2 に対応しているか（`playerctl status` で確認）をチェックしてください。 |

解決しない場合や機能の提案は **[GitHub Issues](https://github.com/yshdsnd/wune/issues)** へお願いします。
