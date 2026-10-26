# FreeBSD Ports / Pkg ガイド (Packaging Guide for FreeBSD)

[English](#english) / [日本語](#japanese)

---

<a id="japanese"></a>
## 日本語

このディレクトリには、FreeBSD の Ports ツリーおよび `pkg` パッケージを作成するための Ports スケルトン一式が含まれています。

### ソース取得モード（ローカル作業ツリー / 上流アーカイブ）
本 Ports 定義は、ビルドを実行する場所に応じて自動的にソース取得方法を切り替えます：

1. **リポジトリクローン内でビルドする場合（作業ツリーを直接パッケージ化）**:
   - `packaging/freebsd/` 配下で `make package` を実行すると、手元の作業ツリーのソースコードを直接使用して `.pkg` を生成します。
   - 外部からのダウンロードを行わないため、手元の変更内容を即座にバイナリパッケージ化してテストできます。
2. **`/usr/ports/` など外部ツリーに配置してビルドする場合（上流アーカイブを取得）**:
   - GitHub からリリースアーカイブまたはスナップショット（デフォルトではコミット `9e49cd3`）を自動的にダウンロードし、チェックサム検証の上でビルドします。
   - `USE_UPSTREAM_DIST=1` を指定して強制的に上流アーカイブからのビルドを行うことも可能です。

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
各種ドキュメント（`README.md`, `README.en.md`, `INSTALL_FREEBSD.md`, `INSTALL_FREEBSD.en.md`）は `/usr/local/share/doc/wune-<version>/` 配下にインストールされます。

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

### Source Acquisition Modes
This port automatically switches its source acquisition mode based on where it is built:

1. **Building within the repository clone (Direct working tree packaging)**:
   - Running `make package` under `packaging/freebsd/` packages the local working tree directly without downloading external distribution archives.
   - Local edits can be packaged and tested immediately.
2. **Building from an external Ports tree (Upstream distribution archive)**:
   - When placed in `/usr/ports/` or when `USE_UPSTREAM_DIST=1` is specified, the port downloads and verifies the source tarball from GitHub (default: snapshot `9e49cd3`).

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

After installation, Wune can be launched via `/usr/local/bin/wune` or desktop application menus.  
Documentation files (`README.md`, `README.en.md`, `INSTALL_FREEBSD.md`, `INSTALL_FREEBSD.en.md`) are installed to `/usr/local/share/doc/wune-<version>/`.
