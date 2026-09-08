"""Optimizer/forecast separation and fail-closed anchor barrier tests."""
from dataclasses import asdict
import unittest
from unittest.mock import patch

import torch

from final_scale_engine_v1 import Engine, array_sha, production_config
from final_scale_protocol_v1 import scientific_settings
from transformer_hvp_grokking import TransformerConfig, flat_spec, make_template, gradient


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.use_deterministic_algorithms(True)

    def fixture(self, record=None, guard=lambda: None, maximum_updates=2):
        cfg = TransformerConfig(modulus=3, model_dim=4, hidden_dim=16, heads=2,
            depth=2, normalization="layernorm", seed=0, train_fraction=.6,
            learning_rate=.003, momentum=.9, weight_decay=.01, dtype="float64")
        n = sum(flat_spec(make_template(cfg)).sizes)
        records = []
        engine = Engine(cfg, expected_parameters=n,
            expected_counts={"training": 5, "trigger": 2, "certification": 2},
            maximum_updates=maximum_updates, guard=guard,
            record=record if record is not None else lambda kind, payload: records.append((kind, payload)))
        return engine, records

    def test_registered_configs_resolve_without_constructing_scale_models(self):
        settings = scientific_settings()
        for index, width in enumerate((144, 208, 320)):
            cfg = production_config(settings, index)
            self.assertEqual((cfg.model_dim, cfg.hidden_dim, cfg.depth, cfg.seed), (width, 4*width, 4, 94101))
            self.assertEqual(cfg.momentum, .9)
        with self.assertRaises(ValueError):
            production_config(settings, True)
        settings["optimizer"]["momentum"] = .8
        with self.assertRaises(ValueError):
            production_config(settings, 0)

    def test_true_updates_match_independent_unscaled_momentum(self):
        engine, records = self.fixture()
        p, v = engine.parameter.clone(), engine.velocity.clone()
        for j in (1, 2):
            v = engine.config.momentum*v + gradient(p, engine.data[0], engine.data[1],
                engine.template, engine.spec, engine.config)
            p = p - engine.config.learning_rate*v
            engine.advance()
            self.assertTrue(torch.equal(engine.parameter, p))
            self.assertTrue(torch.equal(engine.velocity, v))
            self.assertEqual(engine.step, j)
        self.assertEqual([kind for kind, _ in records],
                         ["engine_ready", "update_intent", "update_completed", "update_intent", "update_completed"])
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            engine.advance()

    def test_forecast_does_not_continue_or_modify_optimizer(self):
        engine, records = self.fixture()
        identity = engine.state_identity()
        counts = engine.clock_counts(3, 4, 1e6)
        self.assertEqual(len(counts), 4)
        self.assertEqual(engine.state_identity(), identity)
        self.assertEqual([kind for kind, _ in records], ["engine_ready"])
        self.assertEqual(engine.last_clock["reference_sha256"], array_sha(engine.last_clock["scaled_reference"]))
        engine.advance()
        self.assertIsNone(engine.last_clock)

    def test_invalid_runtime_refused_before_model_construction(self):
        torch.backends.mha.set_fastpath_enabled(True)
        try:
            with self.assertRaisesRegex(ValueError, "runtime"):
                self.fixture()
        finally:
            torch.backends.mha.set_fastpath_enabled(False)

    def test_failed_intent_write_stops_before_true_update(self):
        def recorder(kind, payload):
            if kind == "update_intent":
                raise OSError("injected intent-write failure")
        engine, _ = self.fixture(record=recorder)
        identity = engine.state_identity()
        with self.assertRaisesRegex(OSError, "injected intent"):
            engine.advance()
        self.assertEqual(engine.state_identity(), identity)
        with self.assertRaisesRegex(ValueError, "failed engine"):
            engine.advance()

    def test_failed_completion_write_does_not_permit_another_update(self):
        def recorder(kind, payload):
            if kind == "update_completed":
                raise OSError("injected completion-write failure")
        engine, _ = self.fixture(record=recorder)
        with self.assertRaisesRegex(OSError, "completion-write"):
            engine.advance()
        self.assertEqual(engine.step, 1)
        with self.assertRaisesRegex(ValueError, "failed engine"):
            engine.advance()

    def test_guard_failure_poisoned_engine(self):
        engine, _ = self.fixture()
        def fail():
            raise RuntimeError("injected resource cap")
        engine.guard = fail
        with self.assertRaisesRegex(RuntimeError, "resource cap"):
            engine.current_counts()
        engine.guard = lambda: None
        with self.assertRaisesRegex(ValueError, "failed engine"):
            engine.current_counts()

    def selected_fixture(self, recorder=None):
        engine, records = self.fixture(record=recorder)
        # Synthetic count seam tests the gate, not scientific event timing.
        # The neural reference is computed normally and no true update occurs.
        with patch.object(engine, "_count", side_effect=[0, 1, 1]):
            engine.clock_counts(2, 4, 1e6)
        selection = {"status": "candidate_selected", "anchor": 0, "predicted_offset": 1,
                     "future_continuation_executed": False}
        return engine, records, selection

    def test_one_way_freeze_prevents_all_selector_continuation(self):
        engine, records, selection = self.selected_fixture()
        identity = engine.state_identity()
        exported = engine.freeze_candidate(selection, target=1, persistence=2)
        self.assertEqual(engine.state_identity(), identity)
        self.assertEqual(array_sha(exported["parameter"]), identity["parameter_sha256"])
        self.assertEqual(array_sha(exported["unscaled_velocity"]), identity["unscaled_velocity_sha256"])
        self.assertEqual(exported["config"], asdict(engine.config))
        self.assertEqual([kind for kind, _ in records], ["engine_ready", "candidate_frozen"])
        for operation in (engine.advance, engine.current_counts, lambda: engine.clock_counts(2, 4, 1e6)):
            with self.assertRaisesRegex(ValueError, "frozen anchor"):
                operation()

    def test_changed_selected_timing_or_stale_clock_refused(self):
        engine, _, selection = self.selected_fixture()
        with self.assertRaisesRegex(ValueError, "recorded clock"):
            engine.freeze_candidate({**selection, "predicted_offset": 2}, target=1, persistence=2)
        engine.parameter[0] += 1
        with self.assertRaisesRegex(ValueError, "stale"):
            engine.freeze_candidate(selection, target=1, persistence=2)

    def test_failed_freeze_record_returns_no_snapshot_and_cannot_retry(self):
        def recorder(kind, payload):
            if kind == "candidate_frozen":
                raise OSError("injected freeze write")
        engine, _, selection = self.selected_fixture(recorder)
        with self.assertRaisesRegex(OSError, "freeze write"):
            engine.freeze_candidate(selection, target=1, persistence=2)
        with self.assertRaisesRegex(ValueError, "failed engine"):
            engine.freeze_candidate(selection, target=1, persistence=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
