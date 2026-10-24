# FreeBSD Ports / Pkg ガイド (Packaging Guide for FreeBSD)

[English](#english) / [日本語](#japanese)

---

<a id="japanese"></a>
## 日本語

このディレクトリには、FreeBSD の Ports ツリーおよび `pkg` パッケージを作成するための Ports スケルトン一式が含まれています。

### ハイブリッド動作（ローカル直接ビルド & スナップショット取得）
本 Ports 定義は、**鶏卵問題を回避するハイブリッド方式** で動作します：
1. **リポジトリクローン内でビルドする場合（推奨・即テスト可能）**:
   - `packaging/freebsd/` 配下で `make package` を実行すると、親ディレクトリ（手元の作業ツリー）のソースを直接使用して `.pkg` を生成します。
   - 外部からのダウンロードを行わないため、最新の修正や手元の変更を即座にバイナリパッケージ化してテストできます。
2. **`/usr/ports/` など外部ツリーに配置してビルドする場合**:
   - GitHub から FreeBSD 対応済みの最新スナップショット（コミット `9e49cd3`）のアーカイブを自動フェッチし、チェックサム検証の上でビルドします。
   - 正式リリース（例: v1.3.0）後は、`DISTVERSION= 1.3.0` と `DISTVERSIONPREFIX= v` を指定して公式タグリファレンスに切り替えることができます。

### 構成内容
- **`ports/audio/wune/`**: Wune 本体の Ports 定義（`Makefile`, `distinfo`, `pkg-descr`, `files/wune.desktop`）。
- **`ports/audio/py-soundcard/`**: 未登録の依存ライブラリ `SoundCard`（PyPI 配布版）用の Ports 定義（`Makefile`, `distinfo`, `pkg-descr`）。
- **`Makefile`**: 一括ビルド用ヘルパー Makefile。

---

### パッケージのビルドとインストール手順

#### 方法 1: リポジトリ内から直接 `.pkg` バイナリパッケージをビルド（最も手軽）
```bash
# 本リポジトリの packaging/freebsd/ ディレクトリへ移動
cd packaging/freebsd

# py-soundcard と wune の .pkg パッケージを一括生成（packages/ 配下に出力）
make package

# 生成されたパッケージを pkg add でシステムに導入
sudo pkg add packages/*.pkg
```

個別に追加する場合は以下の順序で行います：
```bash
sudo pkg add packages/py*-soundcard-*.pkg
sudo pkg add packages/wune-*.pkg
```

インストール後、`/usr/local/bin/wune` またはデスクトップ環境のアプリケーションメニューから直接起動できます。

#### 方法 2: システムの Ports ツリー（`/usr/ports/`）に配置してビルド
```bash
# 1. Ports ツリーへコピー
sudo cp -R packaging/freebsd/ports/audio/py-soundcard /usr/ports/audio/
sudo cp -R packaging/freebsd/ports/audio/wune /usr/ports/audio/

# 2. py-soundcard をビルド・インストール
cd /usr/ports/audio/py-soundcard
sudo make install clean

# 3. wune をビルド・インストール
cd /usr/ports/audio/wune
sudo make install clean
```

---

<a id="english"></a>
## English

This directory contains the FreeBSD Ports skeleton to build and package Wune natively using FreeBSD's Ports framework and `pkg`.

### Hybrid Build Mechanism
This port is designed with a **hybrid build mode** to solve the chicken-and-egg problem during active development:
1. **Building within the repository clone (Recommended)**:
   - When running `make package` from `packaging/freebsd/`, the port automatically detects the local source tree and packages directly from it without downloading.
   - Any local edits are immediately reflected in the resulting `.pkg`.
2. **Building from an external Ports tree (e.g., `/usr/ports/`)**:
   - Automatically downloads the snapshot tarball from GitHub (`9e49cd3`) and verifies it using `distinfo`.
   - After a formal release (e.g., v1.3.0), update `DISTVERSION` to point to the release tag.

### Structure
- **`ports/audio/wune/`**: Port definition for Wune application (`Makefile`, `distinfo`, `pkg-descr`, `files/wune.desktop`).
- **`ports/audio/py-soundcard/`**: Port definition for the pure-Python `SoundCard` audio library from PyPI (`Makefile`, `distinfo`, `pkg-descr`).
- **`Makefile`**: Helper Makefile for building both packages in one step.

---

### Quick Packaging and Installation

```bash
cd packaging/freebsd
make package

# Install the created packages (collected under packages/):
sudo pkg add packages/*.pkg
```

Or install individually in dependency order:
```bash
sudo pkg add packages/py*-soundcard-*.pkg
sudo pkg add packages/wune-*.pkg
```
