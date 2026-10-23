# FreeBSD Ports / Pkg ガイド (Packaging Guide for FreeBSD)

[English](#english) / [日本語](#japanese)

---

<a id="japanese"></a>
## 日本語

このディレクトリには、FreeBSD の Ports ツリーおよび `pkg` パッケージを作成するための Ports スケルトン一式が含まれています。

### 構成内容
- **`ports/audio/wune/`**: Wune 本体の Ports 定義（`Makefile`, `pkg-descr`, `files/wune.desktop`）。
- **`ports/audio/py-soundcard/`**: 未登録の依存ライブラリ `SoundCard`（PyPI 配布版）用の Ports 定義（`Makefile`, `distinfo`, `pkg-descr`）。
- **`Makefile`**: 一括ビルド用ヘルパー Makefile。

---

### パッケージのビルドとインストール手順

#### 前提条件
FreeBSD の Ports ツリー（`/usr/ports`）が存在することを確認します。ない場合は `git clone https://git.freebsd.org/ports.git /usr/ports` 等で取得してください。

#### 方法 1: ローカル Ports ツリーに配置してインストール
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

インストール後、`/usr/local/bin/wune` またはデスクトップ環境のアプリケーションメニューから直接起動できます。

#### 方法 2: バイナリパッケージ（`.pkg`）を作成して `pkg add` で導入
```bash
# 本リポジトリの packaging/freebsd/ から直接ビルド
cd packaging/freebsd

# py-soundcard と wune の .pkg をビルド
make package

# 生成されたパッケージを pkg add でインストール
sudo pkg add ./ports/audio/py-soundcard/work/pkg/py*-soundcard-*.pkg
sudo pkg add ./ports/audio/wune/work/pkg/wune-*.pkg
```

#### ローカルソースツリーから直接テストする場合
GitHub からのアーカイブダウンロードを行わず、手元の作業ツリーで直接 Ports ビルドをテストするには `WRKSRC` を指定します：

```bash
cd packaging/freebsd/ports/audio/wune
make WRKSRC=/path/to/your/wune/checkout stage
```

---

<a id="english"></a>
## English

This directory contains the FreeBSD Ports skeleton to build and package Wune natively using FreeBSD's Ports framework and `pkg`.

### Structure
- **`ports/audio/wune/`**: Port definition for Wune application (`Makefile`, `pkg-descr`, `files/wune.desktop`).
- **`ports/audio/py-soundcard/`**: Port definition for the pure-Python `SoundCard` audio library from PyPI (`Makefile`, `distinfo`, `pkg-descr`).
- **`Makefile`**: Helper Makefile for building packages in one step.

---

### Building and Installing

#### Prerequisites
Ensure the FreeBSD ports tree is present at `/usr/ports`.

#### Option 1: Install via Ports Tree
```bash
# 1. Copy port skeletons to /usr/ports/audio
sudo cp -R packaging/freebsd/ports/audio/py-soundcard /usr/ports/audio/
sudo cp -R packaging/freebsd/ports/audio/wune /usr/ports/audio/

# 2. Build and install py-soundcard
cd /usr/ports/audio/py-soundcard
sudo make install clean

# 3. Build and install wune
cd /usr/ports/audio/wune
sudo make install clean
```

After installation, launch with `/usr/local/bin/wune` or select Wune from your desktop application launcher.

#### Option 2: Build Binary `.pkg` Packages
```bash
cd packaging/freebsd
make package

# Install the created packages:
sudo pkg add ./ports/audio/py-soundcard/work/pkg/py*-soundcard-*.pkg
sudo pkg add ./ports/audio/wune/work/pkg/wune-*.pkg
```
