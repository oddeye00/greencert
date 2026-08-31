#!/usr/bin/env python3
"""Verify the exact sealed source and its post-seal finite-input bug fix."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from structured_parameter_green import structured_quadratic_root as current_root
from structured_parameter_green_sealed_v1 import structured_quadratic_root as sealed_root


ROOT = Path(__file__).resolve().parents[1]
CURRENT_RELATIVE = "scripts/structured_parameter_green.py"
CURRENT = ROOT / CURRENT_RELATIVE
SEALED = ROOT / "scripts" / "structured_parameter_green_sealed_v1.py"
SEALED_SHA256 = "0E9561B61F4E76E368A272B28398C04156447B6D3318662F946BDA3164514D86"
CURRENT_SHA256 = "69BA0C19E6A8A34CDAF293F0DE0D58959EEF7B09BC5E53C17B0E3E17DCDABA47"
CURRENT_TEST_RELATIVE = "scripts/test_structured_parameter_green.py"
CURRENT_TEST = ROOT / CURRENT_TEST_RELATIVE
SEALED_TEST = ROOT / "scripts" / "test_structured_parameter_green_sealed_v1.py"
SEALED_TEST_SHA256 = "8D53A19E247FDFD4E67FF74B87A9E8194C6601ED8D799058F09F230CCF5F1EB8"
CURRENT_TEST_SHA256 = "7EB51F57E4EEBA531FC041C8CC01AAC19C1E628F3A343E12FEB5A93E2604C3C9"
AUDIT_RESULTS = (
    ROOT / "results" / "structured_parameter_green_transformer_audit.json",
    ROOT / "results" / "anchor_fixed_structured_parameter_green_transformer_audit.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def attempt_arguments() -> list[tuple[float, float, float]]:
    arguments: list[tuple[float, float, float]] = []
    for result in AUDIT_RESULTS:
        payload = json.loads(result.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            for stage in row["stages"]:
                for route in ("direct", "gram"):
                    attempt = stage.get(route)
                    if attempt is None:
                        continue
                    arguments.append(
                        (
                            float(attempt["parameter_response_upper"]),
                            float(attempt["structured_gain_upper"]),
                            float(attempt["objective_hessian_lipschitz_upper"]),
                        )
                    )
    return arguments


@lru_cache(maxsize=1)
def verify_source_bridge() -> dict:
    require(sha256(SEALED) == SEALED_SHA256, "sealed v1 source snapshot changed")
    require(sha256(CURRENT) == CURRENT_SHA256, "current structured source changed")
    require(sha256(SEALED_TEST) == SEALED_TEST_SHA256, "sealed v1 test snapshot changed")
    require(sha256(CURRENT_TEST) == CURRENT_TEST_SHA256, "current structured test changed")

    arguments = attempt_arguments()
    require(arguments, "no stored structured-Green closure attempts found")
    for response, gain, lipschitz in arguments:
        old = sealed_root(response, gain, lipschitz)
        new = current_root(response, gain, lipschitz)
        require(old == new, "post-seal finite check changed stored closure arithmetic")

    # The patch changes only validation: the sealed implementation silently
    # rounded Python binary64 bounds through float32 and rejected large finite
    # coefficients.  The current implementation keeps the binary64 contract.
    response, gain, lipschitz = 1.0e-50, 1.0e40, 1.0
    sealed_rejected = False
    try:
        sealed_root(response, gain, lipschitz)
    except ValueError:
        sealed_rejected = True
    current = current_root(response, gain, lipschitz)
    require(sealed_rejected, "sealed implementation no longer exhibits the recorded rejection")
    require(current is not None and math.isfinite(current), "current binary64 extension failed")

    return {
        "status": "structured-parameter source supersession verified",
        "sealed_source_sha256": SEALED_SHA256,
        "current_source_sha256": CURRENT_SHA256,
        "sealed_test_sha256": SEALED_TEST_SHA256,
        "current_test_sha256": CURRENT_TEST_SHA256,
        "stored_closure_attempts_compared": len(arguments),
        "stored_closure_results_bitwise_equal": True,
        "large_finite_binary64_input": {
            "response": response,
            "gain": gain,
            "lipschitz": lipschitz,
            "sealed_float32_recheck_rejected": sealed_rejected,
            "current_binary64_check_accepted": True,
        },
    }


def verify_dependency(name: str, expected_sha256: str) -> tuple[Path, bool]:
    """Resolve one protocol dependency, preserving its exact sealed bytes."""

    path = ROOT / Path(name)
    require(path.is_file(), f"sealed dependency missing: {name}")
    if sha256(path) == expected_sha256:
        return path, False
    substitutions = {
        (CURRENT_RELATIVE, SEALED_SHA256): SEALED,
        (CURRENT_TEST_RELATIVE, SEALED_TEST_SHA256): SEALED_TEST,
    }
    require((name, expected_sha256) in substitutions, f"sealed dependency hash mismatch: {name}")
    verify_source_bridge()
    return substitutions[(name, expected_sha256)], True


def main() -> None:
    print(json.dumps(verify_source_bridge(), indent=2))


if __name__ == "__main__":
    main()
