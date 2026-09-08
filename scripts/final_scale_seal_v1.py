"""Protocol/runtime admission and cumulative budgets for the final ladder.

These are cooperative reproducibility controls, not remote attestation or a
sandbox. The interpreter, bootstrap and installed dependency implementations
remain trusted. Scientific settings are compared exactly before any model
may be instantiated; the public executable bundle has no fixture mode.
"""
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import urllib.request

from final_scale_protocol_v1 import scientific_settings, validate_settings
from final_scale_runtime_v1 import ResourceLimit, hard_resource_policy, phase_envelope
from prospective_ledger_v1 import encode, finite_float, invalid_constant, is_hash, unique_object


SCHEMA = "final_scale_public_protocol_v1"
ENVIRONMENT = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
GROUPS = {"selection": "selection", "construction": "construction", "disposition": "construction",
          "observation": "observation_audit", "audit": "observation_audit"}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha(value):
    return hashlib.sha256(encode(value)).hexdigest()


def same(actual, expected, message):
    require(encode(actual) == encode(expected), message)


def read_pinned(path, expected):
    path = Path(path)
    require(is_hash(expected) and path.is_file() and not path.is_symlink(), "regular externally pinned file required")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "external file digest differs")
    value = json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_constant, parse_float=finite_float)
    require(type(value) is dict, "pinned object root required")
    return value


def runtime_snapshot():
    # Dependency import does not construct a candidate or evaluate its future.
    import numpy
    import torch
    import flint
    return {"schema": "final_scale_runtime_identity_v1", "python": platform.python_version(),
        "implementation": platform.python_implementation(), "machine": platform.machine(), "platform": platform.platform(),
        "python_executable_sha256": digest(sys.executable),
        "distributions": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "python-flint", "matplotlib")},
        "numpy_build": numpy.__version__, "torch_build": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "flint_build": flint.__version__, "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "execution_device": "cpu", "installed_dependencies_independently_attested": False}


def protocol_record(source_manifest_sha256, bootstrap_sha256, runtime_sha256):
    require(all(is_hash(v) for v in (source_manifest_sha256, bootstrap_sha256, runtime_sha256)), "invalid public protocol pins")
    return {"schema": SCHEMA, "scientific_settings": scientific_settings(), "resources": hard_resource_policy(),
        "source_manifest_sha256": source_manifest_sha256, "bootstrap_sha256": bootstrap_sha256,
        "runtime_sha256": runtime_sha256, "worker_environment": ENVIRONMENT,
        "phase_groups": GROUPS, "phase_order": list(GROUPS), "one_canonical_observer_root_per_rung": True,
        "construction_and_disposition_share_budget": True, "observation_and_audit_share_budget": True,
        "automatic_retry_or_resume": False, "fixture_execution_permitted": False,
        "public_seal_required_before_controller_start": True,
        "numerical_scope": "outward supplied neural bounds and local replay; ideal Gaussian event; float64 outcome observation",
        "trusted_base": ["bootstrap", "Python interpreter", "installed dependencies", "operating system", "cooperative storage"]}


def verify_protocol(value):
    require(type(value) is dict and value.get("schema") == SCHEMA, "unknown public protocol")
    validate_settings(value["scientific_settings"])
    same(value, protocol_record(value["source_manifest_sha256"], value["bootstrap_sha256"], value["runtime_sha256"]),
         "public protocol policy differs")
    return value


def check_loaded_bundle(source_root, manifest, expected_bootstrap, *, entry_module="final_scale_entry_v1"):
    """Require the already-authenticated in-memory loader, and recheck disk.

This check does not retroactively establish pre-import timing. That property
comes from the externally checked -I -B bootstrap used to enter this module.
"""
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode, "public entry requires -I -B bootstrap")
    root = Path(source_root).resolve(strict=True)
    require(manifest["entry_module"] == entry_module and not any(name.startswith("test_") for name in manifest["sources"]),
            "production bundle cannot contain test entries")
    require({p.name for p in root.iterdir()} == set(manifest["sources"]), "public source population differs")
    finders = [v for v in sys.meta_path if type(v).__name__ == "PinnedFinder" and
               getattr(v, "root", None) == root and isinstance(getattr(v, "payloads", None), dict)]
    require(len(finders) == 1, "authenticated in-memory source loader absent or ambiguous")
    payloads = finders[0].payloads
    require(set(payloads) == {name[:-3] for name in manifest["sources"]}, "loaded source population differs")
    for name, identity in manifest["sources"].items():
        path = root / name
        raw = payloads[name[:-3]]
        require(path.is_file() and not path.is_symlink() and len(raw) == identity["bytes"] and
                hashlib.sha256(raw).hexdigest() == identity["sha256"] == digest(path), "loaded or on-disk source differs")
    require(digest(root / "pinned_source_bundle_v1.py") == expected_bootstrap, "public bootstrap differs")


def admit_protocol(*, protocol_path, protocol_sha256, runtime_path, source_root, source_manifest_path):
    protocol = verify_protocol(read_pinned(protocol_path, protocol_sha256))
    manifest = read_pinned(source_manifest_path, protocol["source_manifest_sha256"])
    check_loaded_bundle(source_root, manifest, protocol["bootstrap_sha256"])
    runtime = read_pinned(runtime_path, protocol["runtime_sha256"])
    same(runtime, runtime_snapshot(), "registered runtime identity differs")
    require(os.name == "posix" and runtime["machine"] == "aarch64", "registered CPU ARM Linux runtime required")
    return protocol, runtime


def fetch_public_protocol(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(1024*1024+1)
    require(len(data) <= 1024*1024, "oversized public protocol")
    return data


def verify_publication(record, protocol_sha256, fetch=fetch_public_protocol):
    """Require immutable public commit bytes before starting the controller."""
    require(type(record) is dict and set(record) == {"repository", "commit", "protocol_path", "protocol_sha256"},
            "invalid publication receipt")
    require(record["repository"] == "oddeye00/greencert" and
            type(record["commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", record["commit"]) is not None and
            record["protocol_path"] == "protocols/final_scale_ladder_v1/protocol.json" and
            is_hash(protocol_sha256) and record["protocol_sha256"] == protocol_sha256,
            "publication is not the immutable registered protocol")
    url = "https://raw.githubusercontent.com/"+record["repository"]+"/"+record["commit"]+"/"+record["protocol_path"]
    require(hashlib.sha256(fetch(url)).hexdigest() == protocol_sha256, "published protocol bytes differ")
    return {"url": url, "protocol_sha256": protocol_sha256, "public_bytes_checked": True}


class Budget:
    """A subphase cannot reset its group's clock or the overall clock."""
    def __init__(self, settings, clock=time.monotonic):
        self.settings, self.clock, self.started, self.groups = settings, clock, clock(), {}

    def deadlines(self, rung, phase):
        require(type(rung) is int and 0 <= rung < len(self.settings["rungs"]) and phase in GROUPS, "invalid budget phase")
        now = self.clock()
        require(math.isfinite(now) and now >= self.started, "monotonic clock reversed")
        group = GROUPS[phase]
        key = (rung, group)
        start = self.groups.setdefault(key, now)
        overall = self.started + self.settings["execution"]["overall_seconds"]
        grouped = start + self.settings["execution"][group+"_seconds_per_rung"]
        deadline = min(overall, grouped)
        if now >= deadline:
            raise ResourceLimit("cumulative experiment/group budget exhausted")
        return {"overall_started": self.started, "group_started": start,
                "overall_deadline": overall, "group_deadline": grouped, "deadline": deadline,
                "remaining_seconds": deadline-now}

    def envelope(self, rung, phase):
        return replace(phase_envelope(self.settings, GROUPS[phase]), seconds=self.deadlines(rung, phase)["remaining_seconds"])

    def guard(self):
        now = self.clock()
        require(math.isfinite(now) and now >= self.started, "monotonic clock reversed")
        if now >= self.started + self.settings["execution"]["overall_seconds"]:
            raise ResourceLimit("overall experiment budget exhausted")
