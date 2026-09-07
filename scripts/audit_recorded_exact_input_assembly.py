"""Reassemble recorded bounds without changing frozen sources or issuance.

Response/Green premises come from a hash-pinned completed replay. Output
and derivative records are rechecked. No future trajectory is accessed.
"""
import argparse
from dataclasses import asdict, fields, is_dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import exact_input_window_event_assembly as exact
from bound_row_evidence import authenticate_bound_row
from portable_reference_margin_evidence import authenticate_reference_window, intersect_with_derivative_row
from replay_recorded_window import load_context
from roundtrip_evidence_reader import RoundtripEvidenceReader, NUMERIC_CONTRACT
from test_exact_input_scalar_closure import verify_exact_supersolution
from verified_artifact_io import unique_object, invalid_constant, finite_json_float

if not __debug__:
    raise RuntimeError("This audit requires enabled assertions; do not use python -O.")


class ExactConversionGuard:
    """Refuse lossy conversions before using legacy bound extractors.

    This narrow recorded-JSON adapter supports finite dyadic inputs. It does
    not invent a rational JSON format for frozen producer records.
    """
    def __init__(self, reader):
        self.reader = reader
        self.numeric_visits = 0

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def check(self, value):
        if type(value) in (int, float):
            converted = float(value)
            if not math.isfinite(converted) or Fraction(converted) != Fraction(value):
                raise ValueError("legacy bound extractor would receive a lossy numeric input")
            self.numeric_visits += 1
        elif isinstance(value, dict):
            for item in value.values():
                self.check(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.check(item)

    def read_json(self, *args, **kwargs):
        record, digest = self.reader.read_json(*args, **kwargs)
        self.check(record)
        return record, digest


def typed_copy(value):
    if is_dataclass(value):
        cls = getattr(exact, type(value).__name__)
        return cls(**{f.name: typed_copy(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, tuple):
        return tuple(typed_copy(v) for v in value)
    return value


def audit(args):
    started = time.perf_counter()
    raw = args.replay_report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.replay_sha256
    replay = json.loads(raw, object_pairs_hook=unique_object,
                        parse_constant=invalid_constant, parse_float=finite_json_float)
    assert replay["status"] == "PASS" and replay["complete_recorded_graph_rehashed"] is True
    assert replay["manifest_sha256"] == args.manifest_sha256
    assert replay["future_outcome_access"] is False
    reader, request, terminal, saved, _, _, _, barrier = load_context(
        args.root, args.manifest, args.manifest_sha256)
    assert replay["assembly"] == saved["result"]["assembly"]
    strict_reader = RoundtripEvidenceReader(args.root, reader.recorded,
                                            numeric_contract=args.numeric_contract)
    # The original context loader is frozen. Before using its derived
    # values, reparse every JSON record it consumed under the explicit
    # new contract, from the same hash-pinned bytes.
    for name, digest in reader.observed.items():
        if Path(name).suffix.lower() == ".json":
            strict_reader.read_json(name, digest)
        else:
            strict_reader.check_blob(name, digest)
    reader = strict_reader
    guard = ExactConversionGuard(reader)
    guard.check(asdict(request))
    guard.check(saved)
    guard.check(replay["green"]["bound"])
    margins = authenticate_reference_window(guard, request.reference_margin_folder,
        identity=request.identity, evaluation=request.evaluation_examples,
        example_ids=request.example_ids, classes=request.classes,
        normalization_eps=request.normalization_eps)
    outputs, drifts = [typed_copy(margins["anchor"])], []
    for location in request.row_plan:
        row = authenticate_bound_row(guard, location.folder, identity=request.identity,
            step=location.step, domain=request.domain, producer=location.producer,
            training_examples=request.training_examples, evaluation_examples=request.evaluation_examples,
            example_ids=request.example_ids)
        outputs.append(typed_copy(intersect_with_derivative_row(
            margins["rows"][location.step], row["output"])))
        if location.step < request.identity.horizon:
            assert row["drift"] is not None
            drifts.append(typed_copy(row["drift"]))
        else:
            assert row["drift"] is None
    identity = typed_copy(request.identity)
    profile = saved["result"]["provenance"]["known_anchor_response"]["profile"]
    response = exact.ResponseBound(identity, tuple(profile["parameter_norms"]),
        profile["residual_upper"], profile["encoding_injection_tail_upper"], True, "outward")
    green = exact.GreenBound(**{**replay["green"]["bound"], "identity": identity})
    result = exact.assemble(identity=identity, training_count=len(request.training_examples),
        example_ids=request.example_ids, domain=request.domain, drift_rows=drifts,
        output_rows=outputs, response=response, green=green,
        target=request.target, persistence=request.persistence, precision_bits=256)
    original = replay["assembly"]
    assert result["inputs_compatible"] and result["state_closure"]["closure"]
    for key in ("bracket", "lower_counts", "upper_counts"):
        assert result[key] == original[key]
    assert result["bracket"] == terminal["decision"]["bracket"] == [44, 44]
    assert result["state_closure"]["radius"] == original["state_closure"]["radius"]
    assert result["probability_scope"] == original["probability_scope"]
    assert result["failure_probability"] == original["failure_probability"]
    verify_exact_supersolution(dict(gain=green.gain_upper,
        drift_by_input=[row.mean_upper for row in drifts], parameter_response_norms=response.parameter_norms,
        response_residual=response.residual_upper, first_injection_error=response.first_injection_error_upper,
        domain=request.domain), result["state_closure"])
    barrier()
    return dict(schema="recorded_exact_input_assembly_audit_v2", status="PASS",
        numeric_contract=args.numeric_contract,
        manifest_sha256=args.manifest_sha256, reference_replay_sha256=args.replay_sha256,
        assembly=result, bracket_unchanged=True, both_count_paths_unchanged=True,
        radius_unchanged=True, probability_scope_unchanged=True,
        checked_numeric_visits=guard.numeric_visits,
        dependency_files_rechecked=len(reader.observed),
        all_legacy_numeric_conversions_exact_on_consumed_records=True,
        supersolution_independently_verified=True, event_certificate_issued=False,
        future_outcome_access=False, frozen_producers_modified=False,
        adapter_sources={name: hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
            for name in ("audit_recorded_exact_input_assembly.py", "roundtrip_evidence_reader.py",
                         "exact_input_scalar_closure.py", "exact_input_window_event_assembly.py",
                         "test_exact_input_scalar_closure.py")},
        elapsed_seconds=time.perf_counter()-started,
        scope="Recorded-bound reassembly, conditional on the hash-pinned completed replay and original neural/Green premises; not a new end-to-end certificate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--replay-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--numeric-contract", choices=(NUMERIC_CONTRACT,), required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("fresh report required")
    result = audit(args)
    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "assembly"}, indent=2))
