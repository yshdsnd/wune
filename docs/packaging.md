# Windowsパッケージの作成

配布形式はPyInstallerのonedirをZIP化したWindows x64版です。
`Wune-vX.Y.Z-win64.zip` の中に `Wune/Wune.exe`、`_internal/`、README、Wune本体のLICENSE、依存物のライセンス、
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

- [x] Wune本体をBSD 2-Clauseとし、リポジトリ直下のLICENSEを毎回ZIPへ同梱する。
- [ ] Issue #58：実際のZIPに含まれる第三者コンポーネント・DLLを列挙し、各ライセンス・通知・追加の再配布条件を照合する。既存のlicenses/自動収集だけで監査済みとはしない。
- [ ] READMEの「公開準備中」を公開時の確定情報へ更新する。動作確認環境の記載を確認する。
- [ ] README更新後のコミットから候補ZIPを再ビルドする（README.md、docs/内の画像とガイドを同梱）。
- [ ] Pythonや開発用パッケージのないWindows環境で、展開・ダブルクリック起動・実際のWASAPI入力を確認する。
- [ ] F2、テーマ変更、保存と再起動、F11、英日表示、初期化、EXE・タイトルバー・タスクバーのアイコンを確認する。
- [ ] 候補ZIP内のREADMEと相対リンク・画像、フォルダー構成、設定／ログ保存先を実物と再照合する。
- [ ] v1.0タグ作成前に上記の利用者向け手順を最終レビューする。タグから再ビルドされたZIPも同じ手順で最終確認する。
- [ ] draft Releaseの本文、バージョン、ZIP・SHA-256、READMEのダウンロード案内を確認して公開する。

CIの依存物チェックは、実機での音声入力やPython未導入環境の操作確認の代わりにはなりません。
コード署名・インストーラー・単一EXE化はこの初期パッケージの対象外です。

参考: [PyInstaller spec files](https://pyinstaller.org/en/stable/spec-files.html)、
[windowedの標準入出力](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html)、
[GitHub Release upload](https://cli.github.com/manual/gh_release_upload)。
