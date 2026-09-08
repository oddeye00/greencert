"""Pure cap checks and Linux-only tiny supervised subprocess regressions."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from final_scale_runtime_v1 import (Envelope, ResourceLimit, hard_resource_policy,
    install_address_space_limit, phase_envelope, run_once)


def noop():
    pass


class EnvelopeTests(unittest.TestCase):
    def fixture(self):
        return Envelope(seconds=10.0, minimum_free_memory_bytes=10,
                        minimum_free_disk_bytes=20, maximum_rss_bytes=30)

    def test_boundary_values(self):
        self.fixture().check(elapsed=9.0, available_memory=10, free_disk=20, rss=30)

    def test_each_cap_is_enforced(self):
        good = dict(elapsed=9.0, available_memory=10, free_disk=20, rss=30)
        for key, value in (("elapsed", 10.0), ("available_memory", 9), ("free_disk", 19), ("rss", 31)):
            with self.assertRaises(ResourceLimit):
                self.fixture().check(**{**good, key: value})

    def test_invalid_samples_cannot_disable_caps(self):
        good = dict(elapsed=9.0, available_memory=10, free_disk=20, rss=30)
        for key, value in (("elapsed", float("nan")), ("available_memory", True), ("free_disk", -1), ("rss", 1.0)):
            with self.assertRaises(ValueError):
                self.fixture().check(**{**good, key: value})
        for invalid in (replace(self.fixture(), seconds=True), replace(self.fixture(), poll_seconds=10),
                        replace(self.fixture(), maximum_rss_bytes=0)):
            with self.assertRaises(ValueError):
                invalid.validate()

    def test_phase_budgets_and_no_retry_policy(self):
        settings = {"execution": {"selection_seconds_per_rung": 7200,
                    "construction_seconds_per_rung": 86400, "observation_audit_seconds_per_rung": 3600}}
        self.assertEqual(phase_envelope(settings, "selection").seconds, 7200)
        self.assertEqual(phase_envelope(settings, "construction").seconds, 86400)
        self.assertEqual(phase_envelope(settings, "observation_audit").seconds, 3600)
        self.assertFalse(hard_resource_policy()["automatic_retry_or_resume"])
        with self.assertRaises(ValueError):
            phase_envelope(settings, "unlimited")


@unittest.skipUnless(os.name == "posix" and hasattr(os, "pidfd_open"), "Linux pidfd supervision required")
class LinuxWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="greencert-runtime-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bindings = {key: hashlib.sha256(key.encode()).hexdigest() for key in
                         ("protocol_sha256", "source_manifest_sha256", "phase_input_sha256")}
        self.envelope = Envelope(seconds=5.0, minimum_free_memory_bytes=1,
            minimum_free_disk_bytes=1, maximum_rss_bytes=256*1024**2,
            poll_seconds=.05, termination_grace_seconds=.5)

    def run_worker(self, name, program, **changes):
        return run_once(self.root / name, [sys.executable, "-I", "-B", "-c", program],
            working_directory=self.root, bindings=self.bindings,
            envelope=changes.get("envelope", self.envelope), guard=changes.get("guard", noop))

    def test_completed_worker_is_not_a_scientific_certificate(self):
        result = self.run_worker("success", "print('tiny worker completed')")
        self.assertEqual(result["status"], "worker_completed")
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(result["scientific_completion_authenticated"])
        self.assertIn("tiny worker completed", (self.root / "success" / "worker.log").read_text())
        with self.assertRaises(FileExistsError):
            self.run_worker("success", "print('must not run')")

    def test_failure_is_retained_without_retry(self):
        result = self.run_worker("failure", "raise SystemExit(7)")
        self.assertEqual(result["status"], "worker_failed")
        self.assertEqual(result["returncode"], 7)
        with self.assertRaises(FileExistsError):
            self.run_worker("failure", "print('must not run')")

    def test_timeout_terminates_only_owned_child_and_keeps_reservation(self):
        with self.assertRaisesRegex(ResourceLimit, "wall-time"):
            self.run_worker("timeout", "import time; time.sleep(30)",
                            envelope=replace(self.envelope, seconds=.2))
        record = json.loads((self.root / "timeout" / "interrupted.json").read_text())
        self.assertNotIn("stop_error", record)
        self.assertFalse((self.root / "timeout" / "terminal.json").exists())
        with self.assertRaises(FileExistsError):
            self.run_worker("timeout", "print('must not run')")

    def test_source_guard_failure_stops_worker(self):
        checks = []
        def fail_after_launch():
            checks.append(True)
            if len(checks) > 1:
                raise ValueError("injected source identity change")
        with self.assertRaisesRegex(ValueError, "source identity"):
            self.run_worker("guard", "import time; time.sleep(30)", guard=fail_after_launch)
        record = json.loads((self.root / "guard" / "interrupted.json").read_text())
        self.assertNotIn("stop_error", record)
        self.assertFalse((self.root / "guard" / "terminal.json").exists())

    def test_address_space_limit_in_separate_pinned_child(self):
        # A tiny source-only fixture bundle, not a scientific protocol seal.
        source = self.root / "child_sources"
        source.mkdir()
        names = ("pinned_source_bundle_v1.py", "prospective_ledger_v1.py",
                 "final_scale_runtime_v1.py", "test_final_scale_runtime_v1.py")
        sources = {}
        for name in names:
            payload = Path(__file__).with_name(name).read_bytes()
            (source / name).write_bytes(payload)
            sources[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        manifest = {"schema": "pinned_python_bundle_v1", "entry_module": "test_final_scale_runtime_v1",
                    "sources": sources, "external_top_level": []}
        payload = json.dumps(manifest).encode()
        path = self.root / "child_manifest.json"
        path.write_bytes(payload)
        command = [sys.executable, "-I", "-B", str(source / "pinned_source_bundle_v1.py"),
                   "--source-root", str(source), "--manifest", str(path),
                   "--manifest-sha256", hashlib.sha256(payload).hexdigest(), "--", "--address-limit-child"]
        result = run_once(self.root / "memory", command, working_directory=self.root,
                         bindings=self.bindings, envelope=self.envelope, guard=noop)
        self.assertEqual(result["returncode"], 0)
        self.assertIn("EXPECTED_MEMORY_ERROR", (self.root / "memory" / "worker.log").read_text())


if __name__ == "__main__":
    if sys.argv[1:] == ["--address-limit-child"]:
        install_address_space_limit(64)
        try:
            bytearray(128*1024**2)
        except MemoryError:
            print("EXPECTED_MEMORY_ERROR")
        else:
            raise AssertionError("address-space limit did not reject allocation")
    else:
        unittest.main(verbosity=2)
