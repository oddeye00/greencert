"""Independent replay regressions, including fully rehashed corrupt fixtures."""
import copy
from dataclasses import asdict
import hashlib
import json
import shutil
import unittest

from final_scale_artifacts_v1 import PhaseReader, file_identity
from final_scale_numeric_replay_v1 import audit_numeric, raw_sha
from prospective_ledger_v1 import encode
import test_final_scale_numeric_core_v1 as core


class ReplayTests(core.NumericCoreTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.expected = {"parameters": len(cls.parameter), "horizon": 2,
            "config": asdict(cls.config), "spec": {"names": list(cls.spec.names), "sizes": list(cls.spec.sizes),
                "shapes": [list(s) for s in cls.spec.shapes]}, "normalization_eps": [list(p) for p in cls.eps],
            "options": cls.options, "probe_seed": 3, "probability_ratio": [1, 30000000],
            "training_count": 5, "evaluation_count": 2, "input_raw_sha256": {
                name: raw_sha(value) for name, value in {"parameter": cls.parameter,
                    "unscaled_velocity": cls.velocity, "clock": cls.clock,
                    "train_pairs": cls.data[0].numpy(), "train_labels": cls.data[1].numpy(),
                    "evaluation_pairs": cls.data[4].numpy(), "evaluation_labels": cls.data[5].numpy()}.items()}}
        cls.audit = audit_numeric(cls.reader, expected=cls.expected, target=1, persistence=2, guard=core.noop)

    def mutate(self, name, path, change):
        """Rehash all explicit references and the outer seal after corruption.

Only disposable test copies are edited. This does not modify producer files
or repair any historical/public evidence graph.
"""
        root = self.root.parent / name
        shutil.copytree(self.root, root)
        target = root / path
        data = json.loads(target.read_bytes())
        change(data)
        target.write_bytes(encode(data))
        def refresh(value):
            if isinstance(value, dict):
                if {"path", "bytes", "sha256"} <= set(value):
                    value.update(file_identity(root, value["path"]))
                for child in value.values():
                    refresh(child)
            elif isinstance(value, list):
                for child in value:
                    refresh(child)
        for _ in range(8):
            changes = 0
            for item in sorted(root.rglob("*.json")):
                if item.name == "manifest.json":
                    continue
                raw = item.read_bytes()
                value = json.loads(raw)
                refresh(value)
                updated = encode(value)
                if raw != updated:
                    item.write_bytes(updated)
                    changes += 1
            if not changes:
                break
        else:
            raise AssertionError("test reference rehash failed to converge")
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["files"] = sorted([file_identity(root, p.relative_to(root).as_posix())
            for p in root.rglob("*") if p.is_file() and p != manifest_path], key=lambda r: r["path"])
        manifest["total_file_bytes"] = sum(r["bytes"] for r in manifest["files"])
        raw = encode(manifest)
        manifest_path.write_bytes(raw)
        return PhaseReader(root, expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            expected_role="construction", expected_bindings=self.bindings, guard=core.noop)

    def test_complete_replay_closes_without_authorizing_a_future(self):
        self.assertEqual(self.audit["status"], "numeric_graph_and_local_arithmetic_replayed")
        self.assertEqual(self.audit["green_hvp_records"], 16)
        self.assertEqual(self.audit["green_product_rows"], 32)
        self.assertEqual(self.audit["point_logits_replayed"], 18)
        self.assertTrue(self.audit["assembly"]["state_closure"]["closure"])
        self.assertIsNone(self.audit["assembly"]["bracket"])
        self.assertFalse(self.audit["event_certificate_issued"])
        self.assertFalse(self.audit["future_observation_authorized"])
        self.assertFalse(self.audit["neural_kernels_independently_reexecuted"])

    def test_external_checkpoint_hash_change_refused(self):
        expected = copy.deepcopy(self.expected)
        expected["input_raw_sha256"]["parameter"] = "0"*64
        with self.assertRaisesRegex(ValueError, "external input identity"):
            audit_numeric(self.reader, expected=expected, target=1, persistence=2, guard=core.noop)

    def test_rehashed_response_error_rejected_by_arithmetic(self):
        reader = self.mutate("bad_response", "response/step_001.json",
            lambda row: row["recurrence"].update(recurrence_error_upper=0.0))
        with self.assertRaisesRegex(ValueError, "response local arithmetic"):
            audit_numeric(reader, expected=self.expected, target=1, persistence=2, guard=core.noop)

    def test_rehashed_green_direction_rejected(self):
        reader = self.mutate("bad_direction", "green_kernels/query_000000.json",
            lambda row: row.update(encoded_direction_sha256="0"*64))
        with self.assertRaisesRegex(ValueError, "Green kernel binding"):
            audit_numeric(reader, expected=self.expected, target=1, persistence=2, guard=core.noop)

    def test_rehashed_training_mean_recomputed(self):
        reader = self.mutate("bad_mean", "neural/step_001/summary.json",
            lambda row: row["result"].update(mean_training_drift_upper=0.0))
        with self.assertRaisesRegex(ValueError, "neural mean"):
            audit_numeric(reader, expected=self.expected, target=1, persistence=2, guard=core.noop)

    def test_rehashed_point_parameter_change_refused(self):
        reader = self.mutate("bad_point", "point_logits/step_002.json",
            lambda row: row.update(parameter_sha256="0"*64))
        with self.assertRaisesRegex(ValueError, "point output binding"):
            audit_numeric(reader, expected=self.expected, target=1, persistence=2, guard=core.noop)

    def test_rehashed_missing_example_refused(self):
        def change(row):
            row["records"] = [ref for ref in row["records"] if not ref["path"].endswith("evaluation_001.json")]
        reader = self.mutate("bad_population", "neural/step_001/summary.json", change)
        with self.assertRaisesRegex(ValueError, "incomplete neural population"):
            audit_numeric(reader, expected=self.expected, target=1, persistence=2, guard=core.noop)


if __name__ == "__main__":
    unittest.main(verbosity=2)
