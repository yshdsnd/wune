# utils.py
import os, sys
import numpy as np

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

def build_band_map(sr: int, nfft: int, bars: int,
                   fmin: float = 20.0, fmax: float | None = None):
    """
    返り値:
      slices: 長さbarsの (start, stop) 配列（stopは排他、必ず stop>start）
      mask:   True=有効バー（≥1ビン）、False=無効（起動時に詰める）
      labels: 表示用の代表周波数（中心周波数）
    """
    if fmax is None:
        fmax = 0.5 * sr * 0.98  # Nyquistに余裕をもたせて空帯域を避ける
    # FFT周波数軸（片側）
    freqs = np.fft.rfftfreq(nfft, d=1.0/sr)  # 0..Nyquist, len=nfft//2+1

    # ログ等分の境界（bars本 → bars+1個の境界）
    edges = np.geomspace(fmin, fmax, bars+1)
    # 各境界を最も近いビンに割り当て
    edge_bins = np.searchsorted(freqs, edges, side='left')

    # 境界が重なってゼロ幅にならないよう強制的に単調増加＆最低幅=1を保証
    edge_bins = np.clip(edge_bins, 1, len(freqs)-1)  # DC(0)は避ける
    for i in range(1, len(edge_bins)):
        if edge_bins[i] <= edge_bins[i-1]:
            edge_bins[i] = min(edge_bins[i-1] + 1, len(freqs)-1)

    # (start, stop) を作る（stopは排他）
    starts = edge_bins[:-1]
    stops  = edge_bins[1:]
    spans  = stops - starts

    # マスク：幅0は無効（上のループで原則防げるが念のため）
    mask = spans > 0
    slices = np.stack([starts, stops], axis=1)

    # ラベル用（各帯域の幾何平均を代表値に）
    f_lo = freqs[starts]
    f_hi = freqs[stops-1]
    labels = np.sqrt(f_lo * f_hi)

    return slices, mask, labels
