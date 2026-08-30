import unittest

import numpy as np

from prospective_v2_primary import HORIZON, prospective_state_machine
from replay_smooth_mlp_thresholds import required_counts


class ProspectivePrimaryTests(unittest.TestCase):
    def test_never_crossed_gate_remains_trigger_eligible(self):
        required = required_counts(36)

        def provider(step):
            self.assertEqual(step, 0)
            counts = np.full(HORIZON + 1, 21, dtype=np.int64)
            counts[10:] = 22
            return counts

        triggers, offsets, crossings, anchors = prospective_state_machine(
            [(0, 21), (1, 20), (250, 21), (500, 20)],
            required,
            provider,
        )
        self.assertEqual(triggers["0.60"], 0)
        self.assertEqual(offsets["0.60"], 10)
        self.assertIsNone(crossings["0.60"])
        self.assertEqual(anchors, [0])

    def test_prior_crossing_permanently_disables_trigger(self):
        required = required_counts(36)
        provider_calls = []

        def provider(step):
            provider_calls.append(step)
            return np.zeros(HORIZON + 1, dtype=np.int64)

        triggers, _, crossings, _ = prospective_state_machine(
            [(0, 20), (1, 22), (250, 21), (500, 21)],
            required,
            provider,
        )
        self.assertEqual(crossings["0.60"], 1)
        self.assertIsNone(triggers["0.60"])
        self.assertEqual(provider_calls, [])

    def test_first_qualifying_checkpoint_is_frozen(self):
        required = required_counts(36)

        def provider(step):
            counts = np.zeros(HORIZON + 1, dtype=np.int64)
            counts[5 if step == 0 else 2 :] = 22
            return counts

        triggers, offsets, crossings, anchors = prospective_state_machine(
            [(0, 21), (250, 21), (500, 22)],
            required,
            provider,
        )
        self.assertEqual(triggers["0.60"], 0)
        self.assertEqual(offsets["0.60"], 5)
        self.assertEqual(crossings["0.60"], 500)
        self.assertEqual(anchors, [0])

    def test_anchor_requires_a_fully_observable_window(self):
        required = required_counts(36)
        provider_calls = []

        def provider(step):
            provider_calls.append(step)
            counts = np.full(HORIZON + 1, 21, dtype=np.int64)
            counts[1:] = 22
            return counts

        triggers, _, crossings, anchors = prospective_state_machine(
            [(179750, 20), (180000, 21)],
            required,
            provider,
            last_anchor_step=179750,
        )
        self.assertIsNone(triggers["0.60"])
        self.assertIsNone(crossings["0.60"])
        self.assertEqual(anchors, [])
        self.assertEqual(provider_calls, [])


if __name__ == "__main__":
    unittest.main()
