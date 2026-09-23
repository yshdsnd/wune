import math
import unittest
import numpy as np
from wune.ballistics import LevelEnvelope, PeakEnvelope


class BallisticsTests(unittest.TestCase):
    def test_attack_is_fast_without_overshoot_or_gain(self):
        envelope = LevelEnvelope()
        target = np.array([[0.4, 0.9]], dtype=np.float32)
        result = envelope.step(target, 1/60)
        self.assertTrue(np.all(result > target * 0.96))
        self.assertTrue(np.all(result <= target))
        for _ in range(60):
            result = envelope.step(target, 1/60)
        np.testing.assert_allclose(result, target, atol=1e-6)

    def test_step_response_matches_elapsed_time_at_different_rates(self):
        for rate in (15, 30, 60, 120):
            envelope = LevelEnvelope()
            one = np.ones((2, 4), dtype=np.float32)
            for _ in range(rate):
                envelope.step(one, 1/rate)
            for _ in range(rate):
                result = envelope.step(one * 0, 1/rate)
            np.testing.assert_allclose(result, math.exp(-1/0.120), atol=1e-6)
            # Silence updates the actual state: the next quiet beat stays quiet.
            quiet = envelope.step(one * 0.2, 1/rate)
            self.assertTrue(np.all(quiet <= 0.2))

    def test_irregular_intervals_and_zero_dt(self):
        target = np.array([0.8], dtype=np.float32)
        envelope = LevelEnvelope()
        for dt in (0.001, 0.002, 0.003, 0.004):
            value = envelope.step(target, dt).copy()
        np.testing.assert_allclose(value, target*(1-math.exp(-0.01/0.005)), atol=1e-6)
        np.testing.assert_array_equal(envelope.step(target*0, 0), value)

    def test_peak_hold_boundary_and_fall_are_time_based(self):
        for intervals in ([0.01]*40, [0.1]*4, [0.05, 0.15, 0.2]):
            peak = PeakEnvelope((1,))
            peak.step(np.array([1.0]), 0.01)
            for dt in intervals:
                peak.step(np.array([0.0]), dt)
            # 120 ms hold plus 280 ms fall at 2.5 full-scale spans/s.
            self.assertAlmostEqual(float(peak.positions[0]), 0.3, places=5)

    def test_peaks_follow_new_attacks_and_pause(self):
        peak = PeakEnvelope((2,))
        peak.step(np.array([0.8, 0.3]), 0.02)
        positions, hold = peak.positions.copy(), peak.remaining.copy()
        peak.step(np.array([0.0, 1.0]), 0)
        np.testing.assert_array_equal(peak.positions, positions)
        np.testing.assert_array_equal(peak.remaining, hold)
        peak.step(np.array([0.9, 0.2]), 0.2)
        self.assertAlmostEqual(float(peak.positions[0]), 0.9, places=6)
        self.assertGreaterEqual(peak.positions[1], 0.2)

    def test_invalid_time_is_rejected(self):
        for dt in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                LevelEnvelope().step(np.zeros(1), dt)
            with self.assertRaises(ValueError):
                PeakEnvelope((1,)).step(np.zeros(1), dt)
