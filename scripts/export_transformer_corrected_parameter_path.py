#!/usr/bin/env python3
"""Export the exact sealed parameter inputs used by a causal-path audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_transformer_direct_image_green_panel import tensor_sha256
from diagnose_transformer_segmented_resolvent import (
    CANDIDATE,
    rebuild_corrected_path,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    (
        row,
        _certificate,
        _config,
        _template,
        _spec,
        _train_pairs,
        _train_labels,
        corrected,
        correction,
        _cert_pairs,
        _cert_labels,
    ) = rebuild_corrected_path()
    dimension = corrected.shape[1] // 2
    corrected_parameter = (
        corrected[:, :dimension].detach().cpu().contiguous().numpy()
    )
    correction_parameter = (
        correction[:, :dimension].detach().cpu().contiguous().numpy()
    )
    prefix = ROOT / "data" / (
        f"transformer_seed_{CANDIDATE.seed}_anchor_{CANDIDATE.anchor}"
    )
    corrected_path = prefix.with_name(
        prefix.name + "_corrected_parameter.npy"
    )
    correction_path = prefix.with_name(
        prefix.name + "_correction_parameter.npy"
    )
    metadata_path = prefix.with_name(prefix.name + "_corrected_path.json")
    outputs = (corrected_path, correction_path, metadata_path)
    if not args.force and any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite an existing corrected-path export")
    np.save(corrected_path, corrected_parameter, allow_pickle=False)
    np.save(correction_path, correction_parameter, allow_pickle=False)
    metadata = {
        "status": "exact sealed Transformer corrected-parameter path export",
        "candidate": CANDIDATE.__dict__,
        "dtype": str(corrected_parameter.dtype),
        "checkpoints": int(corrected_parameter.shape[0]),
        "parameter_count": int(corrected_parameter.shape[1]),
        "corrected_parameter_file": corrected_path.relative_to(ROOT).as_posix(),
        "corrected_parameter_file_sha256": sha256(corrected_path),
        "corrected_parameter_tensor_sha256": tensor_sha256(
            corrected[:, :dimension]
        ),
        "correction_parameter_file": correction_path.relative_to(ROOT).as_posix(),
        "correction_parameter_file_sha256": sha256(correction_path),
        "correction_parameter_tensor_sha256": tensor_sha256(
            correction[:, :dimension]
        ),
        "source_full_corrected_path_sha256": row["corrected_path_sha256"],
        "source_certificate_sha256": row["certificate_sha256"],
        "outcome_files_read": 0,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
