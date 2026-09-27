# Issue #58: Windows配布物のライセンス確認

2026-09-26、Windows 11 x64 / Python 3.13.14 / `requirements-build.txt`の固定版を用い、
`tools/build_windows.py --version 0.0.0-issue58`で実際のZIPを生成して確認した。
Wune本体のBSD 2-Clause（Copyright 2026 Yoshihide Sonoda）は既存のLICENSEを維持した。

## 確認した同梱物

PyInstallerの収集入力とZIPの実データを突き合わせた。
PythonパッケージはSoundCard、CFFI、NumPy、packaging、pycparser、pygame、setuptools、
およびEXEに組み込まれるPyInstaller。PythonとTcl/Tkも別に記録した。
setuptoolsの入れ子の依存物はそのwheelの通知ツリーを丸ごと保持し、名前・版も一覧に記録する。
pip、altgraph、pefile、pywin32-ctypes、pyinstaller-hooks-contribはビルド環境にあるが、
収集入力にないため再配布パッケージの一覧には入らない。

確認したZIPのネイティブファイルは97件（同じDLLの別パスへの配置を含む）。

| 分類 | ファイル数 |
| --- | ---: |
| Python本体・標準拡張 | 17 |
| OpenSSL・libffi | 3 |
| Tcl/Tk | 2 |
| CFFI拡張 | 1 |
| NumPy拡張 | 13 |
| OpenBLAS（LAPACK・GCCランタイムを含む） | 1 |
| pygame拡張 | 30 |
| SDL2 | 2 |
| SDL_imageとJPEG/PNG/TIFF/WebP/zlib | 10 |
| SDL_mixerとmodplug/Ogg/Opus/opusfile | 8 |
| SDL_ttf（FreeType・HarfBuzzを含む） | 2 |
| FreeType DLL | 2 |
| PortMidi | 1 |
| GNU FreeFontの予備フォント | 1 |
| Microsoftランタイム | 3 |
| Wune.exe（PyInstallerブートローダーを含む） | 1 |

実際のパス・SHA-256・対応する通知は各ZIPの`licenses/inventory.json`を正とする。
Python配布元・Windows環境が変わるとMicrosoftランタイムなどのファイル数は変わりうる。
表の件数を満たすだけで合格とする検証ではない。

## 発見した不足と対応

- pygame wheelはLGPL本文を含むが、Windows依存DLLの通知がそろっていなかった。
  pygame公式ビルド設定が指すSDL配布アーカイブと、同梱15種類のDLLをSHA-256で照合した。
  そこに含まれる通知、および静的なSDL_mixerデコーダー、SDL_ttf内のFreeType/HarfBuzzの通知を補完した。
- PythonのWindowsインストールにはTclの通知がなかった。Pythonの公式依存ソースの
  Tcl/Tk 8.6.15の本文を固定し、インストールにある通知も追加保存する。
- 古いpygame予備フォントは編集用ソースの対応を確認できなかった。
  GNU公式のFreeFont 20120503のFreeSansBoldと対応するSFDソースに置き換えた。
  通常のWindowsフォント選択は維持した。
- LGPLのpygameにはライブラリのソースとWuneの再ビルド材料をZIP内に同梱する。
  MPL対象のsetuptoolsのPython/JSONソースも同梱する。
- 旧ビルドは呼び出し元のPATHから別ツールのUCRT/API-set DLLを拾っていた。
  PyInstaller起動時のPATHをPython環境とSystem32に限定した。

出典、元アーカイブのメンバー名、通知のハッシュは`packaging/licenses/provenance.json`に、
対応ソースの取得先とハッシュは`sources.json`に保存した。
ライセンスの個別条件・謝辞・再ビルド方法は
[第三者ソフトウェアの説明](../packaging/licenses/README.md)を参照。

## 検証と公開時の扱い

- 全テスト167件成功（ライセンス収集・未知DLL・欠落通知・変更ハッシュ・ZIP照合の回帰検証を含む）。
- EXEを別フォルダーへ移し、Python検索パスを除いたスモークテスト成功。
- ZIP内の通知・対応ソース全ファイルと、一覧のネイティブファイル全件をハッシュ照合。
- GNU FreeFontの予備フォントが旧pygame版ではなく、確認済みバイト列であることを検証。
- 対応ソース3アーカイブとsetuptoolsソース（242ファイル）の同梱を確認。

これは上記固定版での監査記録。v1.0の最終候補ZIPでも一覧を確認し、依存物やビルド条件を
変えた場合は監査を更新する。音声の実入力・Python未導入端末での操作確認はIssue #38の
公開前チェックとして別途必要。
