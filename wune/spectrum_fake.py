# spectrum_fake.py

import numpy as np
import math
from .config import Config

# ==========================
# スペクトラムの『それっぽい』生成器
# ==========================
class FakeSpectrum:
    """雰囲気重視のフェイクスペクトラム発生器。
    複数の波＋ノイズを合成して、帯域方向にウネウネさせる。
    """
    def __init__(self, cfg: Config, bars: int, channels: int = 2):
        self.cfg = cfg
        self.bars = bars
        self.channels = channels
        self.t = 0.0
        rng = np.random.default_rng()
        # chごとにパラメータを変える
        self.phase1 = rng.uniform(0, math.tau, size=channels)
        self.phase2 = rng.uniform(0, math.tau, size=channels)
        self.speed1 = rng.uniform(0.4, 0.8, size=channels)
        self.speed2 = rng.uniform(0.8, 1.6, size=channels)
        self.band_phase = rng.uniform(0.1, 0.35, size=channels)
        self.noise = np.zeros((channels, bars), dtype=np.float32)
        self.smooth = np.zeros((channels, bars), dtype=np.float32)

    def step(self, dt: float) -> np.ndarray:
        self.t += dt
        i = np.arange(self.bars, dtype=np.float32)
        out = np.zeros_like(self.smooth)
        for ch in range(self.channels):
            w1 = 0.5 * (1.0 + np.sin(self.t * self.speed1[ch] + i * self.band_phase[ch] + self.phase1[ch]))
            w2 = 0.5 * (1.0 + np.sin(self.t * self.speed2[ch] - i * (self.band_phase[ch]*0.7) + self.phase2[ch]))
            base = 0.55 * w1 + 0.45 * w2
            self.noise[ch] = 0.98 * self.noise[ch] + 0.02 * np.random.normal(0.0, 0.9, size=self.bars)
            raw = clamp_array(base + 0.07 * self.noise[ch], 0.0, 1.0)
            self.smooth[ch] = 0.85 * self.smooth[ch] + 0.15 * raw
            out[ch] = self.smooth[ch]
        return out
    
def clamp_array(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)
