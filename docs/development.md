# 開発・ソースからの実行

[利用者向けREADME](../README.md) / [Windowsパッケージの作成](packaging.md)

## 環境と起動

Windows x64、Python 3.13 x64を使用して検証しています。Tkinterを含むPython環境が必要です。
配布EXEの利用には、このセットアップは不要です。
ソースは[GitHubリポジトリ](https://github.com/yshdsnd/wune)から取得してください。

リポジトリのルートで実行します。仮想環境の有効化は不要です。

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
~~~

ビルド時の固定依存物はrequirements-build.txtを参照してください。
アプリ設定の組み込み既定値はwune/config.pyのConfigにあり、末尾のCFGで開発用の値を指定できます。
保存済みの設定がある項目はそちらを優先します。リセットして起動する場合は次を使います。
**既存のユーザーテーマや位置も削除します。必要ならsettings.jsonをバックアップしてください。**

~~~powershell
.\.venv\Scripts\python.exe main.py --reset-settings
~~~

この起動では末尾の独自CFGも使用せず、Config()の既定値から開始します。
通常のソース起動は配布EXEと同じユーザー別設定を使います。
配布EXEのWune.logとは異なり、ソース実行時のエラーはコンソールへ出力します。

### Microsoft Store版Pythonの保存先

Windowsによってファイルが次の場所へ転送される場合があります。
通常のLocal\Wuneが見つからない場合はこちらを確認してください。
設定画面にはPythonから使用する論理パスを表示します。

~~~text
%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\Local\Wune\settings.json
~~~

Python.3.13の部分は使用中のPythonで変わります。この説明はStore版Pythonでのソース起動向けです。

## 音声入力

- output_device = None：起動時のWindows既定の出力デバイス。
- sample_rate = None：起動時にその出力先の共有モードのミックスレートを取得。
- block_size = 4096：読み取り／FFTサイズ。
- channels = 2、bars = 64：既定のステレオ表示とバンド数。

音源ファイルのレートやDAC内部の動作レートを検出するものではありません。
起動後のデバイス・レート変更には追従せず、再起動が必要です。
自動取得に失敗した場合はエラーで終了し、別のレート・マイク・疑似音声へ切り替えません。
SoundCardによる共有モードのWASAPIループバックを使います。

別の出力先を検証するには、次で一覧を取得し、IDまたは名前をConfigのoutput_deviceに指定します。
確実な指定にはIDを使用してください。再生アプリも同じ出力先を使う必要があります。

~~~powershell
.\.venv\Scripts\python.exe -c "from wune.soundcard_compat import prepare_soundcard; prepare_soundcard(); import soundcard as sc; [print(s.id, s.name) for s in sc.all_speakers()]"
~~~

取得はチャンネル0/1です。channels=1は表示上のモノラル化で、モノラル専用デバイスへの対応ではありません。
固定レートの検証にはsample_rate=48000などを指定できます。
SoundCard 0.4.6に固定し、wune/soundcard_compat.pyでWindowsのPROPVARIANTの確保サイズ・初期化・解放を補正しています。
インストール済みライブラリのファイルは書き換えません。

## 表示・解析設定

GUIで変更できる内容は[README](../README.md)を参照してください。
バンド数などを開発時に指定する例です。該当項目の保存済み設定があれば、そちらが優先されます。

~~~python
CFG = Config(
    spectrum_orientation="frequency_vertical",
    channel_layout="horizontal",
    bars=32,
    width=960,
    height=800,
)
~~~

- spectrum_orientation：frequency_horizontalは横に周波数、縦にレベル。frequency_verticalは下から上に周波数、左から右にレベル。
- channel_layout：verticalはL/R上下、horizontalはL/R左右。軸の向きとは独立。
- led_shape：rectangle／rounded／ellipse。led_aspect_ratioは幅÷高さ（0.25～8）。
- gauge_style：flat／box。比率・形はテーマから独立して保存。
- initial_preset：CLASSIC／BLUE／AMBER／CLASSIC BOX、またはNoneで直接指定したThemeを使用。

CLASSIC BOXは配色の互換名で、現在のテーマ選択はLEDの形や立体表現を変えません。
配色はwune/colors.pyのTheme、組み込みテーマはwune/presets.pyで定義します。
独自色をCFGで使う場合はinitial_preset=Noneとtheme=Theme(...)を指定します。
色はRGBの各0～255で、green/yellow/redは低・中・高レベル領域の色名です。
ウィンドウ全体の拡大率はリサイズで決まり、LED比率は拡大率ではありません。

### 表示範囲とレベルの基準

取得レートが48 kHz以下なら最大20 kHz、超える場合は最大40 kHzを表示します。
limit_to_20khz=Trueで最大20 kHzに制限できます。
どちらもmax_freq_hzとナイキスト周波数の安全限界を超えません。
この制限は取得レートを変えず、解析・表示範囲を変更します。
範囲変更時は帯域の意味が変わるためバーとピークの履歴をクリアします。

帯域ごとのFFTパワーを合計し、Hann窓とFFTサイズを補正してdBへ変換します。
ピーク振幅1.0の正弦波を0 dBの基準とし、db_min～db_max（既定は−66～0 dB）をゲージへ対応させます。
フレーム内の最大値や音量履歴での自動正規化、高域強調は行いません。
帯域境界ではエネルギーが分かれ、最大帯域も約3 dB低くなる場合があります。
音楽の各バーは波形全体のピークとは一致しません。

| 動作設定 | 範囲 | 既定値 |
| --- | --- | --- |
| vis_attack_ms | 1～1000 ms | 5 ms |
| vis_release_ms | 1～5000 ms | 120 ms |
| peak_hold_ms | 0～5000 ms | 120 ms |
| peak_fall_per_second | 0～20 表示全幅/秒 | 2.5 |

アタック／リリースは指数応答の時定数です。4096サンプルの取得は48 kHzで約85 msかかるため、
実際の反応は入力間隔にも制限されます。ピーク落下速度0は落下停止です。
動作設定はテーマから独立し、保持時間変更は次のピークから適用します。

## 設定ファイル

version 2のJSONでwindow、appearance、user_themesを保存します。
外観・動作のキーはConfigに対応します。手動編集は終了中に行ってください。

~~~json
{
  "version": 2,
  "window": {"width": 916, "height": 504, "x": 100, "y": 100},
  "appearance": {
    "language": "ja",
    "initial_preset": "BLUE",
    "spectrum_orientation": "frequency_vertical",
    "channel_layout": "horizontal",
    "bars": 32
  }
}
~~~

version 1の選択中テーマのLED設定は独立設定へ引き継ぎ、次回保存からversion 2になります。
ユーザーテーマは配色だけを保存します。不正な項目は無視し、壊れたJSONや未対応のバージョンは上書きを抑止します。
保存は一時ファイルを置き換える方式で、実行エラーで終了した場合は更新しません。

## 翻訳・アイコン

翻訳はwune/locales/en.jsonとja.jsonのキーで管理します。
同じキーとlanguage.name（自称）を持つ言語JSONを追加すると、選択肢に現れます。
不足した訳や不正な置換フィールドは英語へフォールバックします。
デバイス名・テーマ名・開発ログは翻訳しません。
アイコンの出典、生成プロンプトと再エンコード方法は
[こちら](https://github.com/yshdsnd/wune/blob/main/wune/assets/README.md)を参照してください。

## テスト

リポジトリのルートで実行します。Tkのテストにはデスクトップ環境が必要です。

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

固定音声でのFFT／レベル・動作、設定保存、翻訳、描画、ウィンドウ寿命、アイコンなどを検証します。
音声制御のテストでは実機入力の代替を使います。自動テストだけでは音声経路の動作確認はできません。
配布ビルドは依存物を同梱したEXEを別フォルダーで起動して検証します。
実機での再生・停止、出力先・レート、F2、F11、テーマ、設定保存と再起動も確認してください。
