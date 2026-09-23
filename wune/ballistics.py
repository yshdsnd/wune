"""Time-based meter motion, independent of FFT, calibration and drawing."""
import math
import numpy as np


def elapsed(dt):
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and non-negative")
    return dt


class LevelEnvelope:
    def __init__(self, attack_ms=5, release_ms=120):
        if any(not math.isfinite(t) or t <= 0 for t in (attack_ms, release_ms)):
            raise ValueError("Envelope times must be finite and positive")
        self.attack = attack_ms / 1000
        self.release = release_ms / 1000
        self.y = None

    def step(self, target, dt):
        dt = elapsed(dt)
        if self.y is None or self.y.shape != target.shape:
            self.y = np.zeros_like(target, dtype=np.float32)
        tau = np.where(target > self.y, self.attack, self.release)
        self.y += (target - self.y) * (-np.expm1(-dt / tau))
        return self.y


class PeakEnvelope:
    def __init__(self, shape, hold_ms=120, fall_per_second=2.5):
        if any(not math.isfinite(v) or v < 0 for v in (hold_ms, fall_per_second)):
            raise ValueError("Peak hold/fall must be finite and non-negative")
        self.hold_seconds = hold_ms / 1000
        self.fall = fall_per_second
        self.positions = np.zeros(shape, dtype=np.float32)
        self.remaining = np.zeros(shape, dtype=np.float64)

    def step(self, levels, dt):
        dt = elapsed(dt)
        if dt == 0:
            return self.positions
        rising = levels >= self.positions
        falling_time = np.maximum(0, dt - self.remaining)
        self.positions[:] = np.maximum(levels, self.positions - self.fall * falling_time)
        self.remaining[:] = np.maximum(0, self.remaining - dt)
        self.positions[rising] = levels[rising]
        self.remaining[rising] = self.hold_seconds
        return self.positions
