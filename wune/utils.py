# utils.py
# ==========================
# ユーティリティ
# ==========================
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
