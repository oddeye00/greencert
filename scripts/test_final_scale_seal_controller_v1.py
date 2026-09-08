"""Pure protocol/budget/controller tests; no registered model is constructed."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from final_scale_controller_v1 import run_ladder
from final_scale_entry_v1 import expected_phase_binding, phase_runtime, selection_bindings
from final_scale_protocol_v1 import scientific_settings
from final_scale_runtime_v1 import ResourceLimit
from final_scale_seal_v1 import Budget, ENVIRONMENT, GROUPS, protocol_record, read_pinned, sha, verify_protocol, verify_publication
from final_scale_selection_v1 import registered_policy
from prospective_ledger_v1 import encode


PIN = "a"*64


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.settings = scientific_settings()
        self.budget = Budget(self.settings, clock=lambda: self.now)

    def test_disposition_does_not_reset_construction_clock(self):
        first = self.budget.deadlines(0, "construction")
        self.now += 86390
        second = self.budget.deadlines(0, "disposition")
        self.assertEqual(first["group_started"], second["group_started"])
        self.assertEqual(second["remaining_seconds"], 10)
        self.now += 10
        with self.assertRaises(ResourceLimit):
            self.budget.deadlines(0, "disposition")

    def test_audit_does_not_reset_observation_clock(self):
        self.budget.deadlines(1, "observation")
        self.now += 3599
        self.assertEqual(self.budget.envelope(1, "audit").seconds, 1)
        self.now += 1
        with self.assertRaises(ResourceLimit):
            self.budget.deadlines(1, "audit")

    def test_new_rung_does_not_reset_overall_clock(self):
        self.now += self.settings["execution"]["overall_seconds"]-3
        self.assertEqual(self.budget.deadlines(2, "construction")["remaining_seconds"], 3)
        self.now += 3
        with self.assertRaises(ResourceLimit):
            self.budget.guard()

    def test_reentering_same_group_cannot_extend_deadline(self):
        first = self.budget.deadlines(0, "selection")
        self.now += 10
        self.assertEqual(self.budget.deadlines(0, "selection")["deadline"], first["deadline"])

    def test_invalid_phase_rung_and_clock_refused(self):
        for rung, phase in ((-1, "selection"), (3, "selection"), (True, "selection"), (0, "resume")):
            with self.assertRaises(ValueError):
                self.budget.deadlines(rung, phase)
        self.now = 99
        with self.assertRaises(ValueError):
            self.budget.guard()


class SealTests(unittest.TestCase):
    def test_every_protocol_field_is_fixed(self):
        record = protocol_record(PIN, "b"*64, "c"*64)
        self.assertEqual(verify_protocol(record), record)
        for key, value in (("fixture_execution_permitted", True), ("automatic_retry_or_resume", True),
                           ("construction_and_disposition_share_budget", False)):
            bad = copy.deepcopy(record)
            bad[key] = value
            with self.assertRaises(ValueError):
                verify_protocol(bad)
        bad = copy.deepcopy(record)
        bad["scientific_settings"]["rungs"][0]["seed"] += 1
        with self.assertRaises(ValueError):
            verify_protocol(bad)

    def test_publication_requires_exact_immutable_commit_bytes(self):
        raw = b"frozen test protocol bytes"
        pin = hashlib.sha256(raw).hexdigest()
        record = {"repository": "oddeye00/greencert", "commit": "a"*40,
            "protocol_path": "protocols/final_scale_ladder_v1/protocol.json", "protocol_sha256": pin}
        urls = []
        result = verify_publication(record, pin, fetch=lambda url: urls.append(url) or raw)
        self.assertTrue(result["public_bytes_checked"])
        self.assertIn("/"+"a"*40+"/", urls[0])
        for key, value in (("commit", "main"), ("repository", "foreign/repo"), ("protocol_path", "../protocol.json")):
            with self.assertRaises(ValueError):
                verify_publication({**record, key: value}, pin, fetch=lambda _: raw)
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            verify_publication(record, pin, fetch=lambda _: b"changed")

    def test_external_hash_and_duplicate_json_keys_checked(self):
        with tempfile.TemporaryDirectory(prefix="greencert-seal-tests-") as name:
            path = Path(name) / "record.json"
            path.write_bytes(encode({"x": 1}))
            self.assertEqual(read_pinned(path, sha({"x": 1})), {"x": 1})
            with self.assertRaises(ValueError):
                read_pinned(path, PIN)
            raw = b'{"x":1,"x":2}'
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_pinned(path, hashlib.sha256(raw).hexdigest())

    def test_runtime_groups_and_observer_input_pins(self):
        protocol = protocol_record(PIN, "b"*64, "c"*64)
        runtime = {"test_runtime_identity": True}
        records = {phase: phase_runtime(protocol, runtime, phase) for phase in GROUPS}
        self.assertEqual(records["construction"], records["disposition"])
        self.assertEqual(records["observation"], records["audit"])
        self.assertEqual(records["selection"]["threads"], 4)
        self.assertEqual(records["construction"]["threads"], 1)
        policy = registered_policy(scientific_settings(), 0)
        bindings = selection_bindings(PIN, protocol, records["selection"], policy)
        prior = {phase: {"manifest_sha256": str(j+1)*64} for j, phase in enumerate(GROUPS)}
        first = expected_phase_binding("/pure-test-root", 0, "observation", prior, policy, records, bindings)
        prior["disposition"]["manifest_sha256"] = "f"*64
        changed = expected_phase_binding("/pure-test-root", 0, "observation", prior, policy, records, bindings)
        self.assertNotEqual(first["phase_input_sha256"], changed["phase_input_sha256"])


class ControllerTests(unittest.TestCase):
    def fixture(self, decisions, fail=None, bad_coverage=None, count_mismatch=False):
        calls, records = [], []
        def execute(rung, phase, prior):
            calls.append((rung, phase))
            self.assertEqual(set(prior), set(list(GROUPS)[:list(GROUPS).index(phase)]))
            if (rung, phase) == fail:
                raise OSError("injected execution failure")
            issued = decisions[rung] == "issued"
            value = {"manifest_sha256": str(rung+1)*64}
            if phase == "selection":
                value["status"] = "no_candidate" if decisions[rung] == "none" else "candidate_selected"
            elif phase == "construction":
                value["status"] = "complete_numeric_evidence_constructed"
            elif phase == "disposition":
                value.update(status="certificate_issued" if issued else "scientific_abstention", certificate_issued=issued)
            elif phase == "observation":
                value["status"] = "one_shot_float64_observation_complete"
            else:
                value.update(status="complete_observation_and_point_outputs_replayed",
                    summary={"certificate_issued": issued, "issued_bracket_covered": (rung != bad_coverage) if issued else None,
                             "all_count_bounds_contained": not count_mismatch})
            return value
        result = run_ladder(scientific_settings(), execute=execute, record=lambda k, p: records.append((k, p)), guard=lambda: None)
        return result, calls, records

    def test_all_rungs_run_in_order_without_replacement_seeds(self):
        result, calls, _ = self.fixture(["issued", "abstain", "issued"])
        self.assertEqual(calls, [(rung, phase) for rung in range(3) for phase in GROUPS])
        self.assertEqual(result["status"], "ladder_completed")
        self.assertEqual(result["largest_issued_and_covered_parameters"], 4944000)
        self.assertEqual([r["seed"] for r in result["rungs"]], [94101]*3)

    def test_no_candidate_never_constructs_or_observes(self):
        result, calls, _ = self.fixture(["none"]*3)
        self.assertEqual(calls, [(rung, "selection") for rung in range(3)])
        self.assertIsNone(result["largest_issued_and_covered_parameters"])

    def test_execution_failure_stops_without_retry_or_next_rung(self):
        result, calls, _ = self.fixture(["issued"]*3, fail=(1, "construction"))
        self.assertEqual(result["status"], "ladder_stopped")
        self.assertEqual(calls[-1], (1, "construction"))
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(result["largest_issued_and_covered_parameters"], 1008288)

    def test_uncovered_issued_bracket_stops_at_audit(self):
        result, calls, _ = self.fixture(["issued"]*3, bad_coverage=0)
        self.assertEqual(result["reason"], "uncovered_issued_bracket")
        self.assertEqual(calls[-1], (0, "audit"))
        self.assertEqual(len(calls), 5)

    def test_observed_count_mismatch_is_retained_and_stops(self):
        result, calls, _ = self.fixture(["issued"]*3, count_mismatch=True)
        self.assertEqual(result["status"], "ladder_stopped")
        self.assertEqual(result["failed_phase"], "audit")
        self.assertFalse(result["rungs"][0]["phases"]["audit"]["summary"]["all_count_bounds_contained"])
        self.assertEqual(len(calls), 5)

    def test_zero_exit_like_result_is_not_a_phase_certificate(self):
        result = run_ladder(scientific_settings(), execute=lambda *args: {"returncode": 0},
                            record=lambda *args: None, guard=lambda: None)
        self.assertEqual(result["status"], "ladder_stopped")
        self.assertEqual(result["failed_phase"], "selection")

    def test_guard_failure_before_first_phase_cannot_launch(self):
        def fail():
            raise ResourceLimit("injected overall deadline")
        with patch("final_scale_controller_v1.validate_settings", wraps=lambda value: value):
            with patch("final_scale_entry_v1.execute_scientific_phase") as worker:
                result = run_ladder(scientific_settings(), execute=worker, record=lambda *args: None, guard=fail)
                worker.assert_not_called()
        self.assertEqual(result["status"], "ladder_stopped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
