"""Full small lifecycle with synthetic selection gates and real future updates.

All numerical producers are unchanged. The inherited gate seam is explicitly
mocked in named tests, the real constructor abstains, and only then do the
two CPU-float64 fixture updates run. No registered scale seed is instantiated.
"""
import json
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from final_scale_artifacts_v1 import file_identity
from final_scale_construction_v1 import sha
from final_scale_observation_v1 import audit_observation, gradient, observe_phase
from final_scale_selection_v1 import raw_sha
from prospective_ledger_v1 import encode
import test_final_scale_construction_v1 as construction
import test_final_scale_selection_v1 as selection


class ObservationTests(construction.ConstructionTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.case = {k: cls.common[k] for k in ("selection_root", "selection_manifest", "policy", "selection_runtime", "selection_bindings")}
        cls.case.update(construction_root=cls.constructed_root, construction_manifest=cls.constructed["manifest_sha256"],
            disposition_root=cls.disposition_root, disposition_manifest=cls.disposed["manifest_sha256"],
            construction_runtime=cls.runtime, disposition_runtime=cls.runtime)
        cls.observation_root = cls.fixture.root / "observation"
        cls.update_calls = []
        def checked_gradient(*args, **kwargs):
            step = len(cls.update_calls)+1
            last_intent = cls.observation_root / "ledger" / f"record_{1+2*step:06d}.json"
            row = json.loads(last_intent.read_bytes())
            if row["kind"] != "update_intent" or row["payload"]["step"] != step:
                raise AssertionError("true fixture update preceded its durable intent")
            cls.update_calls.append(step)
            return gradient(*args, **kwargs)
        with cls.gate_seam(), patch("final_scale_observation_v1.gradient", side_effect=checked_gradient):
            cls.observed = observe_phase(cls.observation_root, case=cls.case, runtime_record=cls.runtime,
                guard=selection.noop, maximum_bytes=100000000, sync=selection.SYNC, allow_fixture=True)
        with cls.gate_seam(), patch("final_scale_observation_v1.gradient", side_effect=AssertionError("audit repeated optimizer update")):
            cls.observation_audit = audit_observation(cls.observation_root,
                observation_manifest=cls.observed["manifest_sha256"], case=cls.case,
                observation_runtime=cls.runtime, guard=selection.noop, allow_fixture=True)

    def rehash_observer(self, root):
        previous = None
        for path in sorted((root / "ledger").glob("record_*.json")):
            row = json.loads(path.read_bytes())
            row["previous_sha256"] = previous
            if row["kind"] == "observation":
                for key in ("parameter", "unscaled_velocity", "logits"):
                    ref = row["payload"][key]
                    value = np.load(root / ref["path"], allow_pickle=False)
                    ref.update(file_identity(root, ref["path"]), raw_sha256=raw_sha(value))
            path.write_bytes(encode(row))
            previous = sha(row)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["payload"]["ledger_head_sha256"] = previous
        manifest["files"] = sorted([file_identity(root, p.relative_to(root).as_posix())
            for p in root.rglob("*") if p.is_file() and p != manifest_path], key=lambda r: r["path"])
        manifest["total_file_bytes"] = sum(r["bytes"] for r in manifest["files"])
        manifest_path.write_bytes(encode(manifest))
        return sha(manifest)

    def test_true_updates_once_and_only_after_durable_intents(self):
        self.assertEqual(self.update_calls, [1, 2])
        summary = self.observed["summary"]
        self.assertEqual(summary["optimizer_updates"], 2)
        self.assertEqual(summary["observations"], 3)
        self.assertFalse(summary["certificate_issued"])
        self.assertEqual(summary["first_persistent_offset"], 0)
        self.assertTrue(summary["all_count_bounds_contained"])

    def test_independent_output_audit_does_not_rerun_updates(self):
        self.assertEqual(self.observation_audit["status"], "complete_observation_and_point_outputs_replayed")
        self.assertEqual(self.observation_audit["point_logits_independently_recomputed"], 18)
        self.assertFalse(self.observation_audit["optimizer_updates_reexecuted"])
        self.assertFalse(self.observation_audit["exact_real_state_tubes_independently_validated"])
        self.assertEqual(self.observed["summary"], self.observation_audit["summary"])

    def test_completed_observer_cannot_restart_even_admission(self):
        with patch("final_scale_observation_v1.admit") as admission:
            with self.assertRaises(FileExistsError):
                observe_phase(self.observation_root, case=self.case, runtime_record=self.runtime,
                    guard=selection.noop, maximum_bytes=100000000, sync=selection.SYNC, allow_fixture=True)
            admission.assert_not_called()

    def test_failed_update_keeps_one_shot_reservation_and_no_seal(self):
        root = self.fixture.root / "failed_observer_fixture"
        with self.gate_seam(), patch("final_scale_observation_v1.gradient", side_effect=OSError("injected fixture kernel failure")) as kernel:
            with self.assertRaisesRegex(OSError, "injected fixture"):
                observe_phase(root, case=self.case, runtime_record=self.runtime, guard=selection.noop,
                    maximum_bytes=100000000, sync=selection.SYNC, allow_fixture=True)
            self.assertEqual(kernel.call_count, 1)
        self.assertFalse((root / "manifest.json").exists())
        records = sorted((root / "ledger").glob("record_*.json"))
        self.assertEqual(json.loads(records[-1].read_bytes())["kind"], "interrupted")
        self.assertEqual(json.loads(records[-2].read_bytes())["kind"], "update_intent")
        with self.assertRaises(FileExistsError):
            observe_phase(root, case=self.case, runtime_record=self.runtime, guard=selection.noop,
                maximum_bytes=100000000, sync=selection.SYNC, allow_fixture=True)

    def test_rehashed_forged_logits_rejected_by_point_reexecution(self):
        root = self.fixture.root / "forged_logits"
        shutil.copytree(self.observation_root, root)
        path = root / "states" / "step_001_logits.npy"
        values = np.load(path, allow_pickle=False)
        values[0, 0] += 1.0
        np.save(path, values, allow_pickle=False)
        pin = self.rehash_observer(root)
        with self.gate_seam(), self.assertRaisesRegex(ValueError, "independent observed logits"):
            audit_observation(root, observation_manifest=pin, case=self.case,
                observation_runtime=self.runtime, guard=selection.noop, allow_fixture=True)

    def test_rehashed_foreign_update_intent_rejected(self):
        root = self.fixture.root / "forged_intent"
        shutil.copytree(self.observation_root, root)
        path = root / "ledger" / "record_000003.json"
        row = json.loads(path.read_bytes())
        row["payload"]["parameter_sha256"] = "0"*64
        path.write_bytes(encode(row))
        pin = self.rehash_observer(root)
        with self.gate_seam(), self.assertRaisesRegex(ValueError, "observed update intent"):
            audit_observation(root, observation_manifest=pin, case=self.case,
                observation_runtime=self.runtime, guard=selection.noop, allow_fixture=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
