"""Full selection-phase fixture and adversarial semantic replay checks."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from final_scale_artifacts_v1 import BINDINGS, file_identity
from final_scale_engine_v1 import Engine
from final_scale_protocol_v1 import scientific_settings
from final_scale_selection_v1 import SCHEMA, audit_phase, raw_sha, run_phase, sha, validate_policy
from prospective_ledger_v1 import encode, sync_directory
from transformer_hvp_grokking import TransformerConfig


def noop(*args):
    pass


SYNC = sync_directory if os.name == "posix" else noop


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="greencert-selection-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = TransformerConfig(modulus=3, model_dim=4, hidden_dim=16, heads=2, depth=2,
            train_fraction=.6, learning_rate=.003, momentum=.9, weight_decay=.01, seed=0,
            normalization="layernorm", dtype="float64", steps=2, log_every=1, checkpoint_every=1)
        settings = scientific_settings()
        settings["rungs"] = [{"index": 0, "model_dim": 4, "hidden_dim": 16, "parameters": 528, "seed": 0}]
        settings["architecture"].update(modulus=3, depth=2, heads=2)
        settings["data"] = {"training_count": 5, "trigger_count": 2, "certification_count": 2}
        settings["selection"].update(maximum_updates=2, inspection_stride=1, horizon=2,
            persistence=2, target_correct=1, trigger_minimum_correct=2)
        self.policy = {"schema": SCHEMA, "scope": "tiny_development_fixture", "settings": settings,
                       "rung_index": 0, "config": asdict(self.cfg)}
        self.runtime = {"purpose": "tiny_fixture_only", "dtype": "float64"}
        self.bindings = {key: hashlib.sha256(key.encode()).hexdigest() for key in BINDINGS}
        self.bindings.update(phase_input_sha256=sha(encode(self.policy)), runtime_manifest_sha256=sha(encode(self.runtime)))
        self.engines = []

    def factory(self, candidate_anchor=0, real_counts=False):
        def build(record):
            engine = Engine(self.cfg, expected_parameters=528,
                expected_counts={"training": 5, "trigger": 2, "certification": 2},
                maximum_updates=2, guard=noop, record=record)
            self.engines.append(engine)
            if not real_counts:
                engine.current_counts = lambda: {"training": 5, "trigger": 2 if engine.step == candidate_anchor else 0,
                                                  "certification": 0}
                original = engine.clock_counts
                def synthetic_clock(horizon, sweeps, cap):
                    # Gate seam only; reference computed normally, no true future updates.
                    with patch.object(engine, "_count", side_effect=[0, 1, 1]):
                        return original(horizon, sweeps, cap)
                engine.clock_counts = synthetic_clock
            return engine
        return build

    def run_fixture(self, name="phase", **changes):
        return run_phase(self.root / name, policy=changes.get("policy", self.policy),
            runtime_record=changes.get("runtime", self.runtime), bindings=self.bindings,
            engine_factory=changes.get("factory", self.factory()), guard=changes.get("guard", noop),
            maximum_bytes=10000000, sync=SYNC, allow_fixture=changes.get("allow_fixture", True))

    def audit(self, result, name="phase"):
        return audit_phase(self.root / name, expected_manifest_sha256=result["manifest_sha256"],
            policy=self.policy, runtime_record=self.runtime, bindings=self.bindings, guard=noop, allow_fixture=True)

    def reseal_corrupted_fixture(self, name="phase", payload_changes=None):
        """Test-only adversarial rehash: require semantics, not just old checksums."""
        root = self.root / name
        path = root / "manifest.json"
        manifest = json.loads(path.read_bytes())
        if payload_changes:
            manifest["payload"].update(payload_changes)
        manifest["files"] = [file_identity(root, p.relative_to(root).as_posix())
                             for p in sorted(root.rglob("*")) if p.is_file() and p != path]
        manifest["files"].sort(key=lambda row: row["path"])
        manifest["total_file_bytes"] = sum(r["bytes"] for r in manifest["files"])
        payload = encode(manifest)
        path.write_bytes(payload)
        return {"manifest_sha256": sha(payload)}

    def test_selected_anchor_zero_one_and_two_replayed_without_later_update(self):
        for anchor in (0, 1, 2):
            name = f"anchor{anchor}"
            result = self.run_fixture(name, factory=self.factory(anchor))
            audit = self.audit(result, name)
            self.assertEqual(audit["selection"]["anchor"], anchor)
            self.assertEqual(audit["selection"]["predicted_offset"], 1)
            self.assertEqual(self.engines[-1].step, anchor)
            self.assertTrue(self.engines[-1].frozen)
            self.assertEqual(audit["reference"].shape, (3, 1056))
            self.assertFalse(audit["future_outcome_accessed"])
            with self.assertRaises(ValueError):
                self.engines[-1].advance()
            del audit

    def test_no_candidate_at_exact_budget_keeps_data_and_final_state(self):
        result = self.run_fixture(factory=self.factory(None))
        audit = self.audit(result)
        self.assertEqual(audit["selection"]["status"], "no_candidate")
        self.assertEqual(audit["selection"]["completed_updates"], 2)
        self.assertIsNone(audit["reference"])
        self.assertEqual(self.engines[-1].step, 2)

    def test_unmodified_neural_engine_phase_roundtrip(self):
        result = self.run_fixture(factory=self.factory(real_counts=True))
        audit = self.audit(result)
        self.assertEqual(audit["selection"], result["selection"])
        self.assertFalse(audit["prefix_training_and_logits_independently_recomputed"])

    def test_fixture_not_accessible_by_default(self):
        with self.assertRaisesRegex(ValueError, "not authorized"):
            self.run_fixture(allow_fixture=False)
        self.assertEqual(self.engines, [])
        self.assertFalse((self.root / "phase").exists())

    def test_scale_seed_cannot_be_smuggled_into_fixture(self):
        policy = json.loads(json.dumps(self.policy))
        policy["config"]["seed"] = 94101
        with self.assertRaisesRegex(ValueError, "seed zero"):
            validate_policy(policy, allow_fixture=True)

    def test_changed_runtime_refused_before_engine_creation(self):
        with self.assertRaisesRegex(ValueError, "input/runtime"):
            self.run_fixture(runtime={"purpose": "changed"})
        self.assertEqual(self.engines, [])

    def test_existing_phase_is_not_restarted(self):
        self.run_fixture()
        with self.assertRaises(FileExistsError):
            self.run_fixture()
        self.assertEqual(len(self.engines), 1)

    def test_changed_timing_rejected_even_after_full_rehash(self):
        self.run_fixture()
        path = self.root / "phase" / "selection.json"
        data = json.loads(path.read_bytes())
        data["result"]["predicted_offset"] = 2
        path.write_bytes(encode(data))
        seal = self.reseal_corrupted_fixture(payload_changes={"selection_record_sha256": sha(path.read_bytes())})
        with self.assertRaisesRegex(ValueError, "first eligible event"):
            self.audit(seal)

    def test_post_selection_update_intent_refused_after_full_rehash(self):
        self.run_fixture()
        root = self.root / "phase"
        manifest = json.loads((root / "manifest.json").read_bytes())
        count, head = manifest["payload"]["record_count"], manifest["payload"]["record_head_sha256"]
        row = {"sequence": count, "previous_sha256": head, "kind": "update_intent",
               "payload": {"next_step": 1, **self.engines[0].state_identity()}}
        raw = encode(row)
        (root / "records" / f"record_{count:06d}.json").write_bytes(raw)
        seal = self.reseal_corrupted_fixture(payload_changes={"record_count": count+1, "record_head_sha256": sha(raw)})
        with self.assertRaisesRegex(ValueError, "after terminal selection"):
            self.audit(seal)

    def test_overlap_refused_even_after_array_metadata_and_manifest_rehash(self):
        self.run_fixture()
        root = self.root / "phase"
        data = json.loads((root / "data.json").read_bytes())
        for part in ("pairs", "labels"):
            path = root / "data" / f"trigger_{part}.npy"
            path.write_bytes((root / "data" / f"certification_{part}.npy").read_bytes())
            values = np.load(path, allow_pickle=False)
            data["trigger"][part].update(file_identity(root, f"data/trigger_{part}.npy"), raw_sha256=raw_sha(values))
        (root / "data.json").write_bytes(encode(data))
        seal = self.reseal_corrupted_fixture()
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.audit(seal)

    def test_interrupted_factory_keeps_reservation_without_phase_seal(self):
        def fail(record):
            raise OSError("injected initialization failure")
        with self.assertRaises(OSError):
            self.run_fixture(factory=fail)
        self.assertTrue((self.root / "phase" / "method.json").exists())
        self.assertFalse((self.root / "phase" / "manifest.json").exists())
        with self.assertRaises(FileExistsError):
            self.run_fixture()


if __name__ == "__main__":
    unittest.main(verbosity=2)
