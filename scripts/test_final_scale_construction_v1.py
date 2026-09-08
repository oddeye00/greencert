"""Construction/disposition integration on an explicitly synthetic gate.

The tiny selector fixture supplies synthetic gate counts [0,1,1]. The real
numerical graph is built and replayed unchanged, and it abstains. Gate replay
is mocked ONLY in named integration tests; an unmocked replay must reject it.
These tests do not establish a new scientific transition or certificate.
"""
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from final_scale_artifacts_v1 import file_identity
from final_scale_construction_v1 import (audit_disposition, construct_phase, disposition_phase,
    replay_construction, selected_context, sha)
from prospective_ledger_v1 import encode
import test_final_scale_selection_v1 as selection


class ConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selection.SelectionTests.setUpClass()
        cls.fixture = f = selection.SelectionTests()
        f.setUp()
        cls.addClassCleanup(f.doCleanups)
        cls.selected = f.run_fixture()
        cls.common = {"selection_root": f.root / "phase", "selection_manifest": cls.selected["manifest_sha256"],
            "policy": f.policy, "selection_runtime": f.runtime, "selection_bindings": f.bindings,
            "guard": selection.noop, "allow_fixture": True}
        cls.runtime = {"purpose": "tiny_integration_fixture", "threads": 1}
        cls.constructed_root = f.root / "constructed"
        cls.disposition_root = f.root / "disposition"
        with cls.gate_seam():
            cls.constructed = construct_phase(cls.constructed_root, **cls.common,
                runtime_record=cls.runtime, maximum_bytes=100000000, sync=selection.SYNC)
            cls.disposed = disposition_phase(cls.disposition_root, **cls.common,
                construction_root=cls.constructed_root, construction_manifest=cls.constructed["manifest_sha256"],
                construction_runtime=cls.runtime, runtime_record=cls.runtime,
                maximum_bytes=10000000, sync=selection.SYNC)
            cls.audit = cls.replay_disposition()

    @classmethod
    def gate_seam(cls):
        return patch("final_scale_construction_v1.replay_clock_counts", return_value=[0, 1, 1])

    @classmethod
    def replay_disposition(cls, root=None, pin=None, **changes):
        return audit_disposition(root or cls.disposition_root,
            disposition_manifest=pin or cls.disposed["manifest_sha256"],
            construction_root=cls.constructed_root, construction_manifest=cls.constructed["manifest_sha256"],
            construction_runtime=cls.runtime, disposition_runtime=changes.pop("disposition_runtime", cls.runtime),
            **{**cls.common, **changes})

    def test_separate_numeric_construction_and_abstention_disposition(self):
        self.assertFalse(self.constructed["event_certificate_issued"])
        self.assertFalse(self.constructed["future_observation_authorized"])
        self.assertEqual(self.disposed["disposition"]["status"], "scientific_abstention")
        self.assertIsNone(self.disposed["disposition"]["bracket"])
        self.assertFalse(self.disposed["disposition"]["event_certificate_issued"])
        self.assertTrue(self.disposed["disposition"]["future_observation_authorized"])
        self.assertEqual(self.audit["status"], "complete_disposition_recomputed")
        self.assertFalse(self.audit["future_observation_executed"])
        self.assertTrue(self.fixture.engines[0].frozen)
        self.assertEqual(self.fixture.engines[0].step, 0)

    def test_real_count_replay_rejects_the_synthetic_gate(self):
        with self.assertRaisesRegex(ValueError, "independent clock count"):
            replay_construction(self.constructed_root, construction_manifest=self.constructed["manifest_sha256"],
                                construction_runtime=self.runtime, **self.common)

    def test_no_candidate_refused_before_model_reconstruction(self):
        result = self.fixture.run_fixture("no_candidate", factory=self.fixture.factory(None))
        args = {**self.common, "selection_root": self.fixture.root / "no_candidate", "selection_manifest": result["manifest_sha256"]}
        with patch("final_scale_construction_v1.make_template") as model:
            with self.assertRaisesRegex(ValueError, "no selected candidate"):
                construct_phase(self.fixture.root / "no_candidate_construct", **args,
                    runtime_record=self.runtime, maximum_bytes=100000000, sync=selection.SYNC)
            model.assert_not_called()
        self.assertFalse((self.fixture.root / "no_candidate_construct" / "manifest.json").exists())

    def test_constructor_cannot_restart_a_reserved_phase(self):
        with self.assertRaises(FileExistsError):
            construct_phase(self.constructed_root, **self.common, runtime_record=self.runtime,
                            maximum_bytes=100000000, sync=selection.SYNC)

    def test_fixture_option_is_not_implicitly_enabled(self):
        with self.assertRaisesRegex(ValueError, "not authorized"):
            construct_phase(self.fixture.root / "not_allowed", **{**self.common, "allow_fixture": False},
                runtime_record=self.runtime, maximum_bytes=100000000, sync=selection.SYNC)
        self.assertFalse((self.fixture.root / "not_allowed").exists())

    def test_actual_layernorm_epsilon_checked_after_selection_rehash(self):
        root = self.fixture.root / "changed_eps"
        shutil.copytree(self.fixture.root / "phase", root)
        path = root / "clock" / "metadata.json"
        data = json.loads(path.read_bytes())
        data["normalization_eps"][0][0] *= 2
        path.write_bytes(encode(data))
        seal = self.fixture.reseal_corrupted_fixture("changed_eps")
        with self.assertRaisesRegex(ValueError, "actual LayerNorm epsilon"):
            with selected_context(**{**self.common, "selection_root": root, "selection_manifest": seal["manifest_sha256"]}):
                self.fail("changed epsilon admitted")

    def test_rehashed_issued_flag_cannot_authorize_a_certificate(self):
        root = self.fixture.root / "forged_disposition"
        shutil.copytree(self.disposition_root, root)
        path = root / "disposition.json"
        record = json.loads(path.read_bytes())
        record.update(event_certificate_issued=True, status="certificate_issued", bracket=[1, 1], reason=None)
        path.write_bytes(encode(record))
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["payload"].update(disposition_sha256=sha(record), status="certificate_issued")
        manifest["files"] = sorted([file_identity(root, p.relative_to(root).as_posix())
            for p in root.rglob("*") if p.is_file() and p != manifest_path], key=lambda r: r["path"])
        manifest["total_file_bytes"] = sum(r["bytes"] for r in manifest["files"])
        manifest_path.write_bytes(encode(manifest))
        with self.gate_seam(), self.assertRaisesRegex(ValueError, "disposition decision"):
            self.replay_disposition(root, sha(manifest))

    def test_foreign_disposition_runtime_refused(self):
        with self.assertRaisesRegex(ValueError, "binding differs"):
            self.replay_disposition(disposition_runtime={"purpose": "foreign_runtime"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
