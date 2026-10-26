# v1.3.0 main統合記録

[English](en/integration-v1.3.0.md) / **日本語**

マイナーアップデート v1.3.0 に向けた main ブランチの変更・統合記録です。
主な変更点は、FreeBSD 14.x / 15.x プラットフォームの正式サポート（Issue #77）、FreeBSD Ports / pkg バイナリパッケージング体系の構築、FreeBSD 15.1-RELEASE VM による GitHub Actions CI 自動化、および Windows 環境における Music Center for PC の再生状態判定改善（Issue #190）です。

| 機能 | 統合内容・対応 | 関連 Issue / PR |
| --- | --- | --- |
| Music Center for PC 再生判定改善 | Windows Core Audio (`IAudioSessionControl2`) を直接監視し、プロセスに紐づくアクティブな音声セッションの有無によって演奏停止／一時停止状態を正確に判定 | #190, PR #191 |
| FreeBSD プラットフォーム基盤 & SoundCard 互換 | `wune/soundcard_compat.py` におけるプラットフォーム偽装および `sys.argv` ガードにより、未対応の SoundCard 0.4.6 を FreeBSD 上で PulseAudio バックエンドとして安定動作化 | #77, PR #189, PR #193 |
| FreeBSD 実機検証 & 導入ガイド | FreeBSD 15.1-STABLE（Xorg/Openbox/PulseAudio）での実機動作検証、日英導入ガイド（`INSTALL_FREEBSD.md`, `INSTALL_FREEBSD.en.md`）の作成、`pactl` および `pavucontrol` による取り込み元設定手順の確立 | #77, PR #189, PR #192, PR #194 |
| FreeBSD HDMI音声トラブルシューティング | HDMI ディスプレイの省電力・スリープ復帰時に音声デバイス（`snd_hda` / `i915kms`）が認識されなくなる現象の原因分析と復旧手順をドキュメント化 | PR #195 |
| FreeBSD Ports スケルトン作成 | `packaging/freebsd/` 配下に `audio/py-soundcard` および `audio/wune` の Ports 定義一式を整備。リポジトリ作業ツリーからの直接パッケージ化（ハイブリッド方式）と一般ユーザー権限での `make package` 生成に対応 | PR #195, PR #196 |
| Python パッケージング標準化 | `pyproject.toml` および `setup.py` を整備し、CLI コマンド `wune` のエントリポイント、デスクトップエントリ（`wune.desktop`）、アイコンを定義。pkg インストール時のタイトルバーバージョン識別表示（`wune/build_identity.py`）に対応 | PR #195 |
| ドキュメントの DOCSDIR 自動配置 | pkg インストール時に README や各種導入ガイド（日本語・英語）が `/usr/local/share/doc/wune-<version>/` 配下に自動インストールされるよう Ports を設定 | PR #195 |
| FreeBSD 15.1 VM CI ワークフロー | `vmactions/freebsd-vm@v1` を用いて FreeBSD 15.1-RELEASE VM 上でビルド・テスト・パッケージングを実行する `.github/workflows/freebsd-package.yml` を新設。コミットや PR ごとに実機 VM で全テストと `.pkg` 生成を自動検証 | PR #197 |
| ヘッドレス音声デーモン & テスト耐性強化 | FreeBSD VM 内でヘッドレス PulseAudio デーモン（UNIX ソケット、匿名認証、null-sink `wune_dummy`）を起動。音声デーモン非稼働時も `soundcard` スタブフォールバックによりモックテストが中断しないよう堅牢化 | PR #197 |
| リリースアーティファクト自動生成 | FreeBSD パッケージ成果物（`Wune-freebsd-pkg`）の SHA-256 チェックサム生成とアップロード、およびバージョンダグ（`v*`）プッシュ時の GitHub Release 自動添付に対応 | PR #197 |
| ドキュメント構造の日英統一 | `docs/` 直下を日本語ドキュメント、`docs/en/` を英語ドキュメントとして整理し、欠落していた翻訳を補完して全資料の日英ペアを完備 | PR #198 |

## 主要な統合コミット

| コミット | 内容 |
| --- | --- |
| `30aac56` | feat: add FreeBSD support and documentation (#77) |
| `d56129d` | fix(now-playing): determine Music Center playback state via Core Audio session (#190) |
| `efe9179` | docs(freebsd): document audio sink/source configuration and pavucontrol routing (#77) |
| `04986de` | docs: note verified FreeBSD features (resize, settings save, dialog layout) (#77) |
| `1f8f337` | docs: clean up branch references and align documentation across platforms (#77) |
| `6047977` | feat(freebsd): provide FreeBSD ports skeleton and python package setup (#77) |
| `90e6c0b` | feat(freebsd): enable hybrid ports build with local checkout and git snapshot |
| `0633586` | fix(freebsd): align ports distfile naming and non-root package build |
| `5f4fa5d` | feat(freebsd): show package version in title bar when installed via pkg |
| `296b921` | docs(freebsd): add troubleshooting tips for HDMI audio and power well recovery |
| `cf6c11c` | feat(freebsd): install README and FreeBSD install guides to DOCSDIR |
| `dfbc837` | fix(freebsd): refine packaging docs and remove optional pavucontrol dependency |
| `ff6d377` | ci(freebsd): add FreeBSD 15.1 VM packaging workflow and tests |
| `4738d7c` | fix(ci): install fonts and build packages before running unit tests in FreeBSD VM |
| `4113ae1` | fix(ci): launch PulseAudio daemon with null sink and stub soundcard in headless tests |

## ファイル群の確認

- **wune/now_playing_music_center.py**: Windows Core Audio の `IAudioSessionManager2` / `IAudioSessionEnumerator` / `IAudioSessionControl2` を巡回し、Music Center プロセスの音声出力状態（`AudioSessionStateActive`）を正確に判定。
- **wune/soundcard_compat.py**: FreeBSD プラットフォーム判定時の `sys.platform = "linux"` 偽装、`sys.argv` ガード、音声デーモン未稼働環境下での `soundcard` モックスタブ生成。
- **wune/capture.py / capture_macos.py**: 直接 `soundcard` を import せず `get_soundcard()` を経由するよう統一。
- **wune/build_identity.py**: `pkg info` または `build-info.json` からパッケージバージョン（例: `1.3.0`）を取得し、タイトルバーに `Wune v1.3.0` を表示。
- **pyproject.toml / setup.py**: FreeBSD 向け標準パッケージングメタデータおよび `wune = wune.cli:main` エントリポイントの定義。
- **packaging/freebsd/**:
  - `ports/audio/wune/Makefile`: Wune Ports 定義（DOCS, NLS, ALSA, JACK, PULSEAUDIO, PIPEWIRE 等のオプションとデスクトップ・ドキュメント配置）。
  - `ports/audio/py-soundcard/Makefile`: FreeBSD 向け SoundCard Ports 定義。
  - `Makefile`: `make package` で `packages/` 配下に `.pkg` を一括出力するヘルパー。
- **.github/workflows/freebsd-package.yml**: FreeBSD 15.1-RELEASE VM 環境での自動テスト、パッケージング、SHA-256 チェックサム生成、Release 添付ワークフロー。
- **tests/test_freebsd_ports.py**: Ports 定義ファイル、pyproject/setup メタデータ、CI ワークフロー定義の静的整合性テスト。
- **tests/test_now_playing_music_center.py**: WASAPI 音声セッション列挙による再生・一時停止・停止状態判定の単体テスト。

## リリース前の確認事項

1. 全プラットフォーム（Windows 11、macOS Apple Silicon、Linux Ubuntu/Fedora、FreeBSD 15.1）での CI 単体テスト全件 PASS（536 tests）。
2. Windows 実機での Music Center for PC 再生・停止・一時停止時の表示連動確認。
3. FreeBSD 実機（15.1-STABLE）でのパッケージ導入（`pkg add packages/*.pkg`）と起動・音声取り込み・終了確認。
4. macOS および Linux での既存機能（全画面、設定ダイアログ、音声取り込み、Now Playing）がリグレッションなく動作すること。
