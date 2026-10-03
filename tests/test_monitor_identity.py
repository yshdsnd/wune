import unittest
from wune.monitor_identity import Monitor, matching_monitor, valid_identity


class MonitorIdentityTests(unittest.TestCase):
    def test_display_reordering_relocation_and_resolution_changes_keep_identity(self):
        first = Monitor('monitor-a', (0, 0, 1920, 1080), 1)
        saved = Monitor('monitor-b', (-2560, 0, 2560, 1440), 2)
        moved = Monitor('MONITOR-B', (1920, -200, 1920, 1080), 99)
        self.assertIs(matching_monitor(saved.identity, [moved, first]), moved)

    def test_same_position_or_model_does_not_substitute_another_identity(self):
        replacement = Monitor('other-device', (0, 0, 1920, 1080), 1)
        self.assertIsNone(matching_monitor('original-device', [replacement]))

    def test_missing_ambiguous_and_unidentified_monitors_do_not_restore(self):
        a = Monitor('same', (0, 0, 1920, 1080), 1)
        b = Monitor('SAME', (1920, 0, 1920, 1080), 2)
        for monitors in ([], [a, b], [Monitor(None, a.bounds, 1)]):
            self.assertIsNone(matching_monitor('same', monitors))
        for identity in (None, 1, {}, '', 'bad\x00id', 'x' * 1025):
            self.assertFalse(valid_identity(identity))
            self.assertIsNone(matching_monitor(identity, [a]))
