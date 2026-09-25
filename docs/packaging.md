# Windowsパッケージの作成

配布形式はPyInstallerのonedirをZIP化したWindows x64版です。
`Wune-vX.Y.Z-win64.zip` の中に `Wune/Wune.exe`、`_internal/`、README、依存物のライセンス、
`build-info.json` を含みます。Pythonを含むランタイムを同梱し、通常利用時の管理者権限は要求しません。
設定とログはユーザーの `%LOCALAPPDATA%/Wune` に保存します。

## 開発者向けローカルビルド

Windows x64とPython 3.13 x64を使い、専用の仮想環境で実行します。

```powershell
python -m venv .venv-package
.venv-package\Scripts\python -m pip install -r requirements-build.txt
.venv-package\Scripts\python tools/build_windows.py --version 1.0.0-rc.1
```

テスト、PyInstaller、別フォルダーへコピーしたEXEの依存物チェック、ZIPとSHA-256作成を順に行います。
同じ名前の既存ZIPは上書きしません。出力は `dist/`、作業ファイルは `build/` です。
依存物チェックの結果とログは `%TEMP%/Wune-package-check-*` に残します。
音声機器のないCIでもチェックできるよう、この確認では録音せずWASAPIライブラリの読み込みまでを検証します。

## GitHub Actions

PRと手動実行はZIPをActionsの `Wune-win64` artifactに保存します。
`vX.Y.Z` または `vX.Y.Z-rc.1` タグを作成するとビルド後にReleaseへ添付します。
新規Releaseはdraftに留め、公開操作は別途行います。既存の同名assetは上書きしません。
利用したPythonとパッケージの版はZIP内の `build-info.json` で確認できます。

## v1.0公開まで

1. #38のZIPを、Pythonや開発用パッケージのないWindows環境で確認する。
2. #39でREADMEを利用者向けに整え、スクリーンショット・導入・操作・設定保存先を確認する。
3. README更新後のコミットから再ビルドする（最新README.mdを毎回同梱）。
4. 展開・ダブルクリック起動、実際のWASAPI入力、F2、テーマ変更、設定保存・再起動、F11、英日表示を確認する。
5. 同梱ライセンスと配布条件を確認し、Release本文・ZIPを最終確認して公開する。

CIの依存物チェックは、実機での音声入力やPython未導入環境の操作確認の代わりにはなりません。
コード署名・インストーラー・単一EXE化はこの初期パッケージの対象外です。

参考: [PyInstaller spec files](https://pyinstaller.org/en/stable/spec-files.html)、
[windowedの標準入出力](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html)、
[GitHub Release upload](https://cli.github.com/manual/gh_release_upload)。
