"""Time-based meter motion, independent of FFT, calibration and drawing."""
import math
import numpy as np


# Shared limits for the settings UI and persisted preferences.
MOTION_LIMITS = {
    "vis_attack_ms": (1.0, 1000.0),
    "vis_release_ms": (1.0, 5000.0),
    "peak_hold_ms": (0.0, 5000.0),
    "peak_fall_per_second": (0.0, 20.0),
}


def valid_motion(key, value):
    low, high = MOTION_LIMITS[key]
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def elapsed(dt):
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and non-negative")
    return dt


class LevelEnvelope:
    def __init__(self, attack_ms=5, release_ms=120):
        self.y = None
        self.configure(attack_ms, release_ms)

    def configure(self, attack_ms, release_ms):
        """Change time constants while retaining the current envelope level."""
        if any(not math.isfinite(t) or t <= 0 for t in (attack_ms, release_ms)):
            raise ValueError("Envelope times must be finite and positive")
        self.attack = attack_ms / 1000
        self.release = release_ms / 1000

    def step(self, target, dt):
        dt = elapsed(dt)
        if self.y is None or self.y.shape != target.shape:
            self.y = np.zeros_like(target, dtype=np.float32)
        tau = np.where(target > self.y, self.attack, self.release)
        self.y += (target - self.y) * (-np.expm1(-dt / tau))
        return self.y


class PeakEnvelope:
    def __init__(self, shape, hold_ms=120, fall_per_second=2.5):
        self.configure(hold_ms, fall_per_second)
        self.positions = np.zeros(shape, dtype=np.float32)
        self.remaining = np.zeros(shape, dtype=np.float64)

    def configure(self, hold_ms, fall_per_second):
        """Keep current positions/timers; new hold duration starts at next peak."""
        if any(not math.isfinite(v) or v < 0 for v in (hold_ms, fall_per_second)):
            raise ValueError("Peak hold/fall must be finite and non-negative")
        self.hold_seconds = hold_ms / 1000
        self.fall = fall_per_second

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
