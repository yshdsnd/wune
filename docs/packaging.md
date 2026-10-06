# Windows・macOSパッケージの作成

配布形式はPyInstallerのonedirをZIP化したWindows x64版です。
`Wune-vX.Y.Z-win64.zip` の中に `Wune/Wune.exe`、`_internal/`、README、Wune本体のLICENSE、依存物のライセンス、
`build-info.json` を含みます。Pythonを含むランタイムを同梱し、通常利用時の管理者権限は要求しません。
設定とログはユーザーの `%LOCALAPPDATA%/Wune` に保存します。

## 開発者向けローカルビルド

Windows x64とPython 3.13.14 x64を使い、専用の仮想環境で実行します。

```powershell
python -m venv .venv-package
.venv-package\Scripts\python -m pip install -r requirements-build.txt
.venv-package\Scripts\python tools/build_windows.py --version 1.1.0-rc.1
```

テスト、PyInstaller、別フォルダーへコピーしたEXEの依存物チェック、ZIPとSHA-256作成を順に行います。
同じ名前の既存ZIPは上書きしません。出力は `dist/`、作業ファイルは `build/` です。
依存物チェックの結果とログは `%TEMP%/Wune-package-check-*` に残します。
音声機器のないCIでもチェックできるよう、この確認では録音せずWASAPIライブラリの読み込みまでを検証します。

## ライセンスと対応ソースの検証

`LICENSE`はWune本体のBSD 2-Clauseです。第三者ソフトウェアの条件は
[`packaging/licenses/README.md`](../packaging/licenses/README.md)に分けています。
ビルドはPyInstallerが実際に収集したモジュール・データ・DLLを記録し、対象wheelの通知を収集します。
ビルド専用ツールの通知を無条件で混ぜる方式ではありません。
SDL関連DLLは公式Windowsアーカイブとのバイト一致を確認した一覧で照合し、
未知のDLL・フォント、変更されたSDL DLL、空／欠落した通知、未確認のパッケージ版ではビルドを停止します。
呼び出し元のPATHにある無関係なツールのDLLを拾わないよう、PyInstallerのPATHも限定します。

ZIP内の`licenses/inventory.json`に同梱モジュール、パッケージ版、各DLL・PYD・フォント・EXEの
SHA-256と通知の対応を記録します。`licenses/sources/`にはpygame、GNU FreeFontの対応ソースと
そのビルドで使ったWuneソースを含めます。MPL対象のsetuptoolsのソースも同梱します。
上流ソースは`packaging/licenses/sources.json`のURLから取得してSHA-256を照合するため、
初回ビルドにはネット接続が必要です。キャッシュは`build/license-sources/`です。
ZIP作成後に通知・対応ソース・ネイティブファイルの実データを再照合します。

pygameの古い予備フォントは、公式バイナリと編集用SFDソースがそろうGNU FreeFont 20120503の
FreeSansBoldに置き換えています。Windowsの通常のフォント選択は変えません。
Python・wheel・フックを更新するときは、実際の配布物と静的に組み込まれる依存物を再確認し、
通知・対応ソース・出典・ハッシュを更新してください。単にエラーを無視するルールを追加しないでください。

## GitHub Actions

### ビルド識別情報

通常のソース起動と開発用ZIPは、タイトルに `Wune dev (コミットID)` を表示します。
Gitを利用できないソース環境や識別情報が欠けた配布物では `Wune dev (unknown)` になります。
リリースタグ `vX.Y.Z` のActionsビルドはタグから版を取得し、`Wune vX.Y.Z` を表示します。
PR・通常の手動実行は開発ビルドです。手動実行で `release_identity` をオンにすると、タグやReleaseを作成せず正式版と同じ識別表示の候補ZIPを作成できます。PRでは実際にビルドしたマージコミットのIDを記録します。
ローカルで正式リリースとしてビルドする場合のみ、`--version X.Y.Z --release` を指定します。
`--version` だけではZIP名の版を指定するだけで、正式リリース表示にはなりません。

EXEと同じフォルダーの `build-info.json` に `release_version` と完全な `commit` を保存します。
タイトルと起動ログはこの情報を共用するため、配布先にGitやソースは不要です。
ZIPはこのファイルも含めて展開してください。移動後のEXEのスモークテストで識別情報と英日タイトルを確認します。

PRと手動実行はZIPをActionsの `Wune-win64` artifactに保存します。
`vX.Y.Z` または `vX.Y.Z-rc.1` タグを作成するとビルド後にReleaseへ添付します。
新規Releaseはdraftに留め、公開操作は別途行います。既存の同名assetは上書きしません。
利用したPythonとパッケージの版はZIP内の `build-info.json` で確認できます。

## 最終候補と公開

1. リリース準備の変更をmainへ取り込む。
2. ActionsのWindows packageをmainで手動実行し、versionを1.1.0、release_identityをオンにする。
3. 候補ZIP・SHA-256と同梱ファイルを確認し、下記の実機確認を完了する。
4. 確認したコミットにv1.1.0タグを作成する。タグのビルド後に生成されるReleaseはdraftのまま保持する。
5. タグ版ZIPのSHA-256・識別情報・ライセンスを再確認し、リリース本文を記入して公開する。

実機確認が未完了の候補を、確認済みとしてタグ付け・公開しないこと。

## 実機確認

両OSで再生／停止、出力レート、設定のプレビュー・保存・取消、全画面、英日表示、背景、メニュー、終了確認、無音復帰を確認します。Windowsは同一モニター全画面復元、MacはSpaces／設定の前面表示と音声権限も確認します。CIの非録音テストはこれらの代わりにはなりません。

参考: [PyInstaller spec files](https://pyinstaller.org/en/stable/spec-files.html)、
[windowedの標準入出力](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html)、
[GitHub Release upload](https://cli.github.com/manual/gh_release_upload)。

## macOS Apple Silicon版

Python 3.13.14（Tk対応）、Xcode Command Line Toolsを用意します。

```sh
python3 -m venv .venv-package
.venv-package/bin/python -m pip install -r requirements-build.txt
.venv-package/bin/python tools/build_macos.py --version 1.1.0-rc.1
```

arm64専用です。Core Audioライブラリをソースから再生成し、全テスト、PyInstaller、アドホック署名・検証、移動後のアプリ起動チェックを実行します。チェックは音声権限を要求せず、ライブラリ読み込み、背景・メニュー・確認画面、日本語／英語、設定保存、実際の設定用子プロセスを検証します。

出力は `dist/Wune-vX.Y.Z-macos-arm64.zip` とSHA-256です。ZIP直下に `Wune.app`、日英README・導入ガイド、LICENSE、licenses、build-info.jsonを含みます。全OSのガイドも元のファイル名で同梱し、相対リンクを維持します。Macのアプリ内識別情報は `Contents/MacOS/build-info.json`、Info.plistには数値版のみを記録します。

Macの `licenses/inventory.json` は署名後アプリのファイルハッシュ、PyInstaller収集元、依存wheelの通知を記録します。Windows専用のDLLハッシュ照合をMacへ流用しません。Macのネイティブ依存物の最終ライセンス確認はリリース確認項目として残ります。通知と対応ソースはZIP内の実データでも照合します。

MacはDeveloper ID署名・公証済みではありません。初回起動は[導入ガイド](../INSTALL_MACOS.md)の手順で許可します。

Actionsの **macOS package** は `macos-15` のApple Siliconで実行し、artifact `Wune-macos` に保存します。Windowsと同様、PR・手動実行は公開せず、タグ時のみReleaseに添付します。同名assetは上書きしません。v1.1.0公開前には両ワークフローを同じコミットから実行し、実機確認を終えてからタグ・公開へ進みます。
