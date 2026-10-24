# FreeBSD版 Wune 導入ガイド

[English](INSTALL_FREEBSD.en.md) / **日本語**

80～90年代のミニコンポをイメージしたLEDスペクトラムアナライザー「Wune」の、FreeBSD 向けインストールと利用手順です。

## 動作環境

- **対応OS / アーキテクチャ**: **FreeBSD 14.x / 15.x（amd64）**（FreeBSD 15.1-STABLE 実機にて動作確認済み）
  *(実機検証構成: i915kms + modesetting Xorg, Openbox, Firefox + PulseAudio)*
- **音声サーバー**: **PulseAudio** または **PipeWire**（PulseAudio互換レイヤー）
- **ディスプレイ**: **X11** または **Wayland**
- **ステレオ音声を再生できる出力デバイス**（スピーカー、ヘッドホン等）
- **推奨**: 日本語フォント（Google Noto Fonts）、`playerctl`（Now Playing 曲名表示用）、`pavucontrol`（音声入出力・録音ストリームの切り替え用）

## インストール方法の選択

FreeBSD への導入には、以下の 2 つの方法があります。

1. **FreeBSD Ports / pkg によるパッケージ導入（推奨）**
   リポジトリに含まれる Ports スケルトンを利用して、システム標準の Ports ツリーへの登録や、バイナリパッケージ（`.pkg`）の生成・インストールが可能です。パッケージ導入時には各種ドキュメント（README や本トラブルシューティング・操作ガイド）が `/usr/local/share/doc/wune-<version>/` にも配置されます。
   詳細は **[FreeBSD Ports / Pkg ガイド](packaging/freebsd/README.md)** を参照してください。

2. **ソースコードからの直接導入（仮想環境）**
   システムのパッケージと Python 仮想環境（venv）を組み合わせて直接実行する方法です。以下に手順を記載します。

## ソースコードからの直接導入手順

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

### トラブルシューティング & Tips：音が出なくなった場合の対処と予防

#### 症状：Wune のスペアナは反応しているのにスピーカーから音が出ない
Wune のスペアナが振れているということは、再生アプリから PulseAudio への出力、およびループバックモニター（`<device>.monitor`）からの取り込みは正常に動いています。  
止まっているのは「PulseAudio から物理デバイス（HDMI / `/dev/dsp3` 等）」への伝送です。

FreeBSD + Intel グラフィックス（`i915kms`）の HDMI 音声環境では、**無音時や画面スリープ時にオーディオ回路の給電（Power Well）が落とされ、次の再生時にハードウェアの復帰・再同期に失敗してデバイスが沈黙する** という現象が知られています。

#### 予防策（根本対策・推奨設定）

1. **Intel GPU（i915kms）のオーディオ給電を常時維持する**  
   `/boot/loader.conf` に以下を追記して再起動します（現在のセッションで即時適用する場合は `sudo sysctl compat.linuxkpi.i915_disable_power_well=0`）：
   ```text
   compat.linuxkpi.i915_disable_power_well="0"
   ```
   > [!TIP]
   > デフォルト値 `-1`（auto）では無音時に HDMI 音声回路が電源断され、復帰不能になる場合があります。`0` に設定することで常に給電（power wells always on）され、突然死を防ぐことができます。

2. **PulseAudio のアイドルサスペンドを無効化する**  
   無音時に PulseAudio が `/dev/dsp` を閉じる動作を防止します：
   ```bash
   mkdir -p ~/.config/pulse
   cp /usr/local/etc/pulse/default.pa ~/.config/pulse/
   ```
   `~/.config/pulse/default.pa` 内の `load-module module-suspend-on-idle` をコメントアウトします：
   ```diff
   - load-module module-suspend-on-idle
   + # load-module module-suspend-on-idle
   ```
   設定後、`pulseaudio -k && pulseaudio --start` で反映します。

#### 発生時の復旧手順（OS 再起動なしで復旧を試みる）

FreeBSD では `snd_hda` などのサウンドドライバが GENERIC カーネルに静的組み込みされているため `kldunload` できません。また Xorg の再起動も音声サブシステムには影響しません。  
OS 再起動を行わずに復帰を試みるには、FreeBSD のデバイス制御ツール `devctl(8)` または HDA コーデック再構成 sysctl を使用します：

```bash
# 1. デバイスを掴んでいる PulseAudio を一度停止
pulseaudio -k

# 2. devctl によるデバイスの強制リセットまたはデタッチ・再アタッチ
sudo devctl reset pcm3
# または
sudo devctl detach -f pcm3 && sudo devctl attach pcm3

# （または親の HDA コーデック hdaa1 を再構成）
sudo sysctl dev.hdaa.1.reconfig=1

# 3. PulseAudio を再開して既定値を再指定
pulseaudio --start
pactl set-default-sink oss_output.dsp3
pactl set-default-source oss_output.dsp3.monitor
```

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
