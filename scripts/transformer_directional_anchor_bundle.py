#!/usr/bin/env python3
"""Build and verify compact sparse checkpoint archives for directional replay."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARENT_NAME = "transformer_fully_recentered_three_sweep_audit.json"
MANIFEST_NAME = "transformer_directional_anchor_states_manifest.json"
AGGREGATE_NAME = "transformer_directional_anchor_states.npz"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest().upper()


def archive_relative(seed: int) -> str:
    return f"results/transformer_hvp_prospective_seed_{seed}.checkpoints.npz"


def identities(parent_path: Path) -> list[tuple[int, int]]:
    payload = json.loads(parent_path.read_text(encoding="utf-8"))
    return sorted(
        {
            (int(row["candidate"]["seed"]), int(row["candidate"]["anchor"]))
            for row in payload["rows"]
        }
    )


def _npy_payload(array: np.ndarray) -> bytes:
    destination = io.BytesIO()
    np.lib.format.write_array(
        destination, np.ascontiguousarray(array), allow_pickle=False
    )
    return destination.getvalue()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                _npy_payload(arrays[name]),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    temporary.replace(path)


def build(source_root: Path, output_root: Path = ROOT) -> dict:
    """Extract only claim-used anchors from complete frozen checkpoint archives."""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root:
        raise ValueError(
            "refusing to replace complete source archives with sparse archives; "
            "choose a separate --output-root"
        )
    parent_path = source_root / "results" / PARENT_NAME
    grouped: dict[int, dict[str, np.ndarray]] = {}
    aggregate_arrays: dict[str, np.ndarray] = {}
    rows = []
    source_hashes: dict[int, tuple[str, str]] = {}
    for seed, anchor in identities(parent_path):
        checkpoint_relative = archive_relative(seed)
        blind_relative = f"results/transformer_hvp_prospective_seed_{seed}.json"
        checkpoint_path = source_root / checkpoint_relative
        blind_path = source_root / blind_relative
        if not checkpoint_path.is_file() or not blind_path.is_file():
            missing = checkpoint_path if not checkpoint_path.is_file() else blind_path
            raise FileNotFoundError(missing)
        source_hashes.setdefault(
            seed, (digest(checkpoint_path), digest(blind_path))
        )
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            parameter = np.ascontiguousarray(checkpoint[f"step_{anchor}"])
            velocity = np.ascontiguousarray(checkpoint[f"velocity_{anchor}"])
        grouped.setdefault(seed, {})[f"step_{anchor}"] = parameter
        grouped[seed][f"velocity_{anchor}"] = velocity
        aggregate_parameter_key = f"seed_{seed}_anchor_{anchor}_parameter"
        aggregate_velocity_key = f"seed_{seed}_anchor_{anchor}_velocity"
        aggregate_arrays[aggregate_parameter_key] = parameter
        aggregate_arrays[aggregate_velocity_key] = velocity
        rows.append(
            {
                "seed": seed,
                "anchor": anchor,
                "archive": checkpoint_relative,
                "parameter_key": f"step_{anchor}",
                "velocity_key": f"velocity_{anchor}",
                "aggregate_parameter_key": aggregate_parameter_key,
                "aggregate_velocity_key": aggregate_velocity_key,
                "parameter_dtype": str(parameter.dtype),
                "velocity_dtype": str(velocity.dtype),
                "parameter_shape": list(parameter.shape),
                "velocity_shape": list(velocity.shape),
                "parameter_sha256": array_digest(parameter),
                "velocity_sha256": array_digest(velocity),
                "source_complete_checkpoint_sha256": source_hashes[seed][0],
                "blind_record": blind_relative,
                "blind_record_sha256": source_hashes[seed][1],
            }
        )

    output_results = output_root / "results"
    output_results.mkdir(parents=True, exist_ok=True)
    archives = []
    for seed in sorted(grouped):
        relative = archive_relative(seed)
        destination = output_root / relative
        _write_deterministic_npz(destination, grouped[seed])
        archives.append(
            {
                "seed": seed,
                "path": relative,
                "sha256": digest(destination),
                "bytes": destination.stat().st_size,
                "arrays": len(grouped[seed]),
                "source_complete_checkpoint_sha256": source_hashes[seed][0],
            }
        )

    aggregate_path = output_results / AGGREGATE_NAME
    _write_deterministic_npz(aggregate_path, aggregate_arrays)

    manifest = {
        "status": "deterministic sparse directional checkpoint bundle",
        "format": "NumPy .npy members in timestamp-fixed ZIP containers",
        "scope": (
            "Exact parameter and velocity arrays for every anchor used by the "
            "v1.3 directional replay; these are intentionally sparse archives."
        ),
        "source_parent": f"results/{PARENT_NAME}",
        "source_parent_sha256": digest(parent_path),
        "anchors": len(rows),
        "arrays": 2 * len(rows),
        "archives": archives,
        "aggregate_bytes": sum(int(row["bytes"]) for row in archives),
        "aggregate_archive": {
            "path": f"results/{AGGREGATE_NAME}",
            "sha256": digest(aggregate_path),
            "bytes": aggregate_path.stat().st_size,
            "arrays": len(aggregate_arrays),
        },
        "rows": rows,
    }
    manifest_path = output_results / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _verified_manifest.cache_clear()
    return manifest


@lru_cache(maxsize=8)
def _verified_manifest(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifest_path = root / "results" / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total = 0
    for row in manifest["archives"]:
        archive = root / row["path"]
        if digest(archive) != row["sha256"]:
            raise AssertionError(f"directional sparse archive hash mismatch: {archive}")
        if archive.stat().st_size != int(row["bytes"]):
            raise AssertionError(f"directional sparse archive size mismatch: {archive}")
        total += archive.stat().st_size
    if total != int(manifest["aggregate_bytes"]):
        raise AssertionError("directional sparse archive aggregate size mismatch")
    aggregate = manifest["aggregate_archive"]
    aggregate_path = root / aggregate["path"]
    if digest(aggregate_path) != aggregate["sha256"]:
        raise AssertionError("directional aggregate anchor archive hash mismatch")
    if aggregate_path.stat().st_size != int(aggregate["bytes"]):
        raise AssertionError("directional aggregate anchor archive size mismatch")
    return manifest


def load_anchor(
    seed: int, anchor: int, root: Path = ROOT
) -> tuple[np.ndarray, np.ndarray]:
    root = root.resolve()
    manifest = _verified_manifest(root)
    matches = [
        row
        for row in manifest["rows"]
        if int(row["seed"]) == int(seed) and int(row["anchor"]) == int(anchor)
    ]
    if len(matches) != 1:
        raise KeyError(f"anchor bundle has no unique seed={seed}, anchor={anchor}")
    row = matches[0]
    with np.load(root / row["archive"], allow_pickle=False) as archive:
        parameter = np.ascontiguousarray(archive[row["parameter_key"]])
        velocity = np.ascontiguousarray(archive[row["velocity_key"]])
    checks = (
        (str(parameter.dtype), row["parameter_dtype"], "parameter dtype"),
        (str(velocity.dtype), row["velocity_dtype"], "velocity dtype"),
        (list(parameter.shape), row["parameter_shape"], "parameter shape"),
        (list(velocity.shape), row["velocity_shape"], "velocity shape"),
        (array_digest(parameter), row["parameter_sha256"], "parameter hash"),
        (array_digest(velocity), row["velocity_sha256"], "velocity hash"),
    )
    for observed, expected, label in checks:
        if observed != expected:
            raise AssertionError(f"directional anchor {label} mismatch")
    return parameter, velocity


def verify(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifest = _verified_manifest(root)
    for row in manifest["rows"]:
        load_anchor(int(row["seed"]), int(row["anchor"]), root)
    return {
        "status": "directional sparse checkpoint bundle verified",
        "anchors": len(manifest["rows"]),
        "arrays": int(manifest["arrays"]),
        "archives": len(manifest["archives"]),
        "aggregate_bytes": int(manifest["aggregate_bytes"]),
        "aggregate_archive_bytes": int(manifest["aggregate_archive"]["bytes"]),
        "aggregate_archive_sha256": manifest["aggregate_archive"]["sha256"],
        "archive_sha256": {
            row["path"]: row["sha256"] for row in manifest["archives"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-from-checkpoints", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.build_from_checkpoints is not None:
        build(args.build_from_checkpoints, args.output_root)
    print(json.dumps(verify(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
