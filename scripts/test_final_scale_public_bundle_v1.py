"""Check the public frozen-source package without constructing a scale model."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from final_scale_seal_v1 import verify_protocol


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "protocols" / "final_scale_ladder_v1"


class PublicBundleTests(unittest.TestCase):
    def test_protocol_runtime_source_and_working_copy_byte_bindings(self):
        protocol = verify_protocol(json.loads((BUNDLE / "protocol.json").read_bytes()))
        for name, pin in (("source_manifest.json", protocol["source_manifest_sha256"]), ("runtime.json", protocol["runtime_sha256"])):
            self.assertEqual(hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest(), pin)
        manifest = json.loads((BUNDLE / "source_manifest.json").read_bytes())
        self.assertEqual(manifest["entry_module"], "final_scale_entry_v1")
        self.assertFalse(any(name.startswith("test_") for name in manifest["sources"]))
        self.assertEqual({p.name for p in (BUNDLE/"sources").iterdir()}, set(manifest["sources"]))
        for name, record in manifest["sources"].items():
            raw = (BUNDLE/"sources"/name).read_bytes()
            self.assertEqual(len(raw), record["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["sha256"])
            self.assertEqual(raw, (ROOT/"scripts"/name).read_bytes())

    def command(self):
        protocol = json.loads((BUNDLE/"protocol.json").read_bytes())
        return [sys.executable, "-I", "-B", str(BUNDLE/"sources"/"pinned_source_bundle_v1.py"),
            "--source-root", str(BUNDLE/"sources"), "--manifest", str(BUNDLE/"source_manifest.json"),
            "--manifest-sha256", protocol["source_manifest_sha256"]]

    def test_bootstrap_authenticates_all_sources_without_execution(self):
        result = subprocess.run(self.command()+["--check-only"], capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(value["status"], "authenticated_without_execution")
        self.assertFalse(value["bundle_code_executed"])

    def test_production_cli_has_no_fixture_or_resume_option(self):
        result = subprocess.run(self.command()+["--", "--help"], capture_output=True, text=True, check=True)
        self.assertIn("{check,run,worker}", result.stdout)
        self.assertNotIn("--allow-fixture", result.stdout)
        self.assertNotIn("--resume", result.stdout)

    def test_integer_looking_float_conversion_is_not_silently_admitted(self):
        value = json.loads((BUNDLE/"protocol.json").read_bytes())
        value["scientific_settings"]["architecture"]["dropout"] = 0
        with self.assertRaises(ValueError):
            verify_protocol(value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
