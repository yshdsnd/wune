# utils.py
import os, sys

# ==========================
# ユーティリティ
# ==========================
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def check_supported_environment():
    # Windows の RDP 判定（例: "RDP-Tcp#0"）
    if os.environ.get("SESSIONNAME", "").startswith("RDP-"):
        print("[Wune] Remote Desktop (RDP) 環境では利用できません。ローカルで実行してください。")
        sys.exit(1)
    # 将来的に Linux の X dummy とかも追加してもいい