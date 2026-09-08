"""Require the Linux CI lock correction to preserve every existing version/hash."""
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LockTests(unittest.TestCase):
    def test_only_the_documented_linux_hash_was_added(self):
        original = (ROOT / 'requirements.txt').read_text()
        linux = (ROOT / 'requirements-linux-ci.txt').read_text()
        record = json.loads((ROOT / 'results/final_scale_linux_ci_lock_20260908_v1.json').read_text())
        extra = '    --hash=sha256:' + record['publisher_and_downloaded_sha256']
        self.assertEqual(linux.count(extra), 1)
        restored = linux.split('\n', 3)[3].replace(' \\\n' + extra, '')
        self.assertEqual(restored, original)

    def test_both_lock_bytes_match_the_provenance_record(self):
        record = json.loads((ROOT / 'results/final_scale_linux_ci_lock_20260908_v1.json').read_text())
        for filename, key in [('requirements.txt', 'original_windows_lock_sha256'),
                              ('requirements-linux-ci.txt', 'linux_ci_lock_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(), record[key])


if __name__ == '__main__':
    unittest.main(verbosity=2)
