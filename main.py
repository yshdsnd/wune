"""
ウネウネLEDスペアナ v0.1
- ステップ1〜3を実装：バー描画 / 色分け / ピークホールド
- 90年代ミニコンポ風の見た目を意識したLEDセグメント描画
- いまは『それっぽく動く』ための疑似スペクトラム（擬似ノイズ＋波）

終了: ESC / Q / 閉じるボタン
フルスクリーン切替: F11
一時停止: Space

※ pygame の Python 3.13 サポートは環境により差異がある可能性があります。
   もしインストールでコケる場合は 3.12 を一時的に利用してください。
"""
# main.py
import pygame as pg
import sys

from wune.config import CFG
from wune.app import App

if __name__ == "__main__":
    try:
        App(CFG).run()
    except Exception as ex:
        print("Error:", ex)
        pg.quit()
        sys.exit(1)
