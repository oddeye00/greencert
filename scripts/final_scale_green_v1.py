"""One-shot, case-independent wrapper for the existing outward Green bound.

No optimizer continuation is performed. HVPs must enclose the objective at
the frozen reference; authenticating that callback is a caller obligation.
The stored-graph audit checks identities, probes and scalar aggregation; it
does not independently re-execute neural HVPs. Gaussian probability retains
the project's explicitly conditional ideal-PRNG interpretation.
"""
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np
from flint import arb, ctx

from checkpointed_green_products import StoredRows, digest, product, store_rows
from green_local_residual_bounds import gram_operator_upper
from outward_inexact_anytime_gram import folded_normal_calibration_lower
from prospective_ledger_v1 import (finite_float, invalid_constant, is_hash,
                                   publish_new, sync_directory, unique_object)


SCHEMA = "final_scale_green_v1"
PIN_FIELDS = {"protocol_sha256", "source_manifest_sha256", "reference_sha256",
              "training_set_sha256", "checkpoint_sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probability_down(ratio):
    """Encode a registered rational failure budget without rounding it up."""
    require(type(ratio) in (list, tuple) and len(ratio) == 2 and
            all(type(v) is int and v > 0 for v in ratio), "invalid failure ratio")
    exact = Fraction(*ratio)
    require(0 < exact < 1, "failure probability must be in (0,1)")
    result = float(exact)
    if Fraction.from_float(result) > exact:
        result = math.nextafter(result, -math.inf)
    require(0 < result < 1 and Fraction.from_float(result) <= exact,
            "failure budget cannot be represented positively")
    return result


def method_record(*, parameters, horizon, learning_rate, momentum, probes,
                  powers, seed, failure_ratio, precision_bits, scalar_bits, contract):
    require(type(contract) is dict and set(contract) == PIN_FIELDS and
            all(is_hash(v) for v in contract.values()), "incomplete operator binding")
    require(all(type(v) is int and v > 0 for v in (parameters, horizon, probes)),
            "positive integer shape/probe count required")
    require(type(seed) is int and 0 <= seed < 2**64, "invalid probe seed")
    require(type(powers) in (list, tuple) and powers and
            all(type(v) is int and v > 0 for v in powers) and
            list(powers) == sorted(set(powers)), "powers must be sorted unique positive integers")
    require(type(precision_bits) is int and precision_bits >= 64 and
            type(scalar_bits) is int and scalar_bits >= max(256, precision_bits), "invalid precision")
    require(type(learning_rate) in (int, float) and math.isfinite(learning_rate) and learning_rate > 0
            and type(momentum) in (int, float) and math.isfinite(momentum) and 0 <= momentum < 1,
            "unsupported momentum constants")
    return {"schema": SCHEMA, "parameters": parameters, "horizon": horizon,
            "learning_rate": float(learning_rate), "momentum": float(momentum), "probes": probes,
            "powers": list(powers), "seed": seed, "failure_ratio": list(failure_ratio),
            "failure_probability": probability_down(failure_ratio), "precision_bits": precision_bits,
            "scalar_bits": scalar_bits, "contract": dict(contract),
            "generator": "numpy.Generator(numpy.PCG64(seed+probe_index)); standard_normal(2*n) per row",
            "probability_scope": "ideal_gaussian_probes", "same_projection_event_all_powers": True,
            "automatic_retry_or_resume": False}


def probe_rows(method, index, guard):
    generator = np.random.Generator(np.random.PCG64(method["seed"] + index))
    for _ in range(method["horizon"]):
        guard()
        yield generator.standard_normal(2 * method["parameters"], dtype=np.float64)


def aggregate(method, chains):
    """Recompute every registered power on the same Gaussian projection event."""
    require(len(chains) == method["probes"] and
            all(len(chain) == max(method["powers"]) for chain in chains), "incomplete probe family")
    calibration = folded_normal_calibration_lower(delta=method["failure_probability"],
        probes=method["probes"], precision_bits=method["scalar_bits"])
    estimates = []
    for q in method["powers"]:
        inputs = {"shift": 0.0, "terminal_upper": max(chain[q-1][1].summary[
                       "terminal_sequence_norm_upper"] for chain in chains),
                  "calibration_lower": calibration,
                  "forward_residuals": [max(chain[l][0].summary["residual_sequence_upper"]
                      for chain in chains) for l in range(q)],
                  "adjoint_residuals": [max(chain[l][1].summary["residual_sequence_upper"]
                      for chain in chains) for l in range(q)]}
        result = gram_operator_upper(**inputs, precision_bits=method["scalar_bits"])
        estimates.append({"power": q, "inputs": inputs, "result": result})
    finite = [row["result"]["operator_upper"] for row in estimates if row["result"]["enclosed"]]
    return {"status": "green_bound_enclosed" if finite else "green_bound_abstention",
            "gain_upper": max(1.0, min(finite)) if finite else None,
            "calibration_lower": calibration, "estimates": estimates,
            "complete_probes": len(chains), "failure_probability": method["failure_probability"],
            "probability_scope": "ideal_gaussian_probes", "event_certificate_issued": False}


def construct(folder, hvp, *, parameters, horizon, learning_rate, momentum, probes,
              powers, seed, failure_ratio, precision_bits=128, scalar_bits=256,
              contract, guard, progress, sync=sync_directory):
    """Create one complete family or leave a non-reusable failed reservation.

The caller must provide resource/source guards and an outward HVP callback.
The production directory-sync primitive is POSIX-only. Tests on other
platforms must supply an explicitly non-production sync callback.
"""
    require(callable(hvp) and callable(guard) and callable(progress), "callbacks required")
    method = method_record(parameters=parameters, horizon=horizon, learning_rate=learning_rate,
        momentum=momentum, probes=probes, powers=powers, seed=seed, failure_ratio=failure_ratio,
        precision_bits=precision_bits, scalar_bits=scalar_bits, contract=contract)
    folder = Path(folder)
    require(folder.parent.is_dir() and not folder.parent.is_symlink(), "invalid family parent")
    guard()
    sync(folder.parent)
    folder.mkdir(exist_ok=False)  # Intentionally no resume, including an empty reservation.
    sync(folder.parent)
    method_sha = publish_new(folder / "method.json", method, sync=sync)
    product_contract = {"family_method_sha256": method_sha, **contract}

    def checked_hvp(step, direction):
        guard()
        require(type(step) is int and 1 <= step < horizon, "noncausal HVP step")
        require(ctx.prec == precision_bits, "incorrect HVP entry precision")
        result = hvp(step, direction)
        require(ctx.prec == precision_bits, "HVP callback changed precision")
        require(len(result) == parameters and all(isinstance(v, arb) and v.is_finite() for v in result),
                "HVP callback did not return complete finite enclosures")
        guard()
        return result

    chains, manifests = [], []
    for index in range(probes):
        name = f"probe_{index:03d}"
        current = store_rows(folder / name, probe_rows(method, index, guard),
            shape=(horizon, 2 * parameters), contract=product_contract)
        sync(current.folder)
        sync(folder)
        manifests.append({"folder": name, "summary_sha256": current.identity})
        chain = []
        for power in range(1, max(powers) + 1):
            pair = []
            for transpose in (False, True):
                name = f"probe_{index:03d}_power_{power:02d}_{'adjoint' if transpose else 'forward'}"
                destination = folder / name
                require(not destination.exists(), "product destination already exists")

                def row_progress(row):
                    sync(destination)
                    sync(folder)
                    progress({"probe": index, "power": power, "transpose": transpose, "row": row})

                product(destination, current, checked_hvp, learning_rate=learning_rate,
                    momentum=momentum, transpose=transpose, precision_bits=precision_bits,
                    contract=product_contract, guard=guard, progress=row_progress)
                sync(destination)
                sync(folder)
                current = StoredRows(destination)
                pair.append(current)
                manifests.append({"folder": name, "summary_sha256": current.identity})
            chain.append(tuple(pair))
        chains.append(chain)
    result = aggregate(method, chains)
    guard()
    complete = {"schema": SCHEMA, "method_sha256": method_sha, "sequences": manifests,
                "hvp_calls": probes * 2 * max(powers) * (horizon - 1), "result": result,
                "numerical_scope": "outward supplied-HVP recurrence and scalar arithmetic; ideal Gaussian event"}
    complete_sha = publish_new(folder / "complete.json", complete, sync=sync)
    return {**result, "completion_sha256": complete_sha, "hvp_calls": complete["hvp_calls"]}


def read_record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing or linked record")
    return json.loads(path.read_bytes(), object_pairs_hook=unique_object,
                      parse_constant=invalid_constant, parse_float=finite_float)


def audit(folder, *, expected_completion_sha256, expected_contract, guard):
    """Rehash the complete stored graph, regenerate probes, and replay scalars.

This does not certify source execution, timestamps or the neural HVP values.
Those producer obligations and a separately sealed complete graph remain
necessary. A completion flag or scalar gain alone is never accepted.
"""
    require(is_hash(expected_completion_sha256), "external completion digest required")
    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink(), "invalid family root")
    require(digest(folder / "complete.json") == expected_completion_sha256, "completion digest differs")
    complete, method = read_record(folder / "complete.json"), read_record(folder / "method.json")
    require(complete["schema"] == SCHEMA and digest(folder / "method.json") == complete["method_sha256"],
            "family method differs")
    rebuilt = method_record(**{key: method[key] for key in
        ("parameters", "horizon", "learning_rate", "momentum", "probes", "powers", "seed",
         "failure_ratio", "precision_bits", "scalar_bits", "contract")})
    require(method == rebuilt and method["contract"] == expected_contract, "operator binding differs")
    product_contract = {"family_method_sha256": complete["method_sha256"], **expected_contract}
    entries = complete["sequences"]
    require(type(entries) is list, "invalid sequence manifest")
    names = [entry["folder"] for entry in entries]
    require(len(names) == len(set(names)) and
            set(p.name for p in folder.iterdir()) == set(names) | {"method.json", "complete.json"},
            "missing, extra or duplicate family entries")
    observed, chains = [], []

    def sequence(name, *, kind, input_rows=None):
        guard()
        path = folder / name
        require(path.is_dir() and not path.is_symlink(), "invalid product folder")
        result = StoredRows(path)
        expected_files = {"method.json", "summary.json"} | {
            f"row_{j:03d}.npy" for j in range(method["horizon"])}
        if kind != "input_rows":
            expected_files |= {f"row_{j:03d}.json" for j in range(method["horizon"])}
        require(set(p.name for p in path.iterdir()) == expected_files and
                all(p.is_file() and not p.is_symlink() for p in path.iterdir()), "product file population differs")
        require((result.H, result.d) == (method["horizon"], 2 * method["parameters"]), "product dimensions differ")
        expected_method = {"kind": kind, "shape": [result.H, result.d], "contract": product_contract}
        if kind == "input_rows":
            expected_method["expected_flat_npy_sha256"] = None
        else:
            expected_method.update(input_manifest_sha256=input_rows.identity,
                precision_bits=method["precision_bits"], learning_rate=method["learning_rate"], momentum=method["momentum"])
            require(result.summary["input_manifest_sha256"] == input_rows.identity, "input identity differs")
        require(result.method == expected_method, "product method differs")
        for j in range(result.H):
            guard()
            result.row(j)  # Validate every stored array, not only the summary flags.
            if input_rows is not None:
                require(result.summary["rows"][j]["input_row_sha256"] == input_rows.summary["rows"][j]["sha256"],
                        "row input identity differs")
                endpoint = j == (result.H - 1 if kind == "adjoint" else 0)
                expected_step = None if endpoint else (j + 1 if kind == "adjoint" else j)
                require(result.summary["rows"][j]["hvp_step"] == expected_step, "product HVP index differs")
                if endpoint:
                    require(result.summary["rows"][j]["local_residual_norm_upper"] == 0 and
                            np.array_equal(result.row(j), input_rows.row(j)), "nonexact endpoint copy")
        observed.append({"folder": name, "summary_sha256": result.identity})
        return result

    for index in range(method["probes"]):
        current = sequence(f"probe_{index:03d}", kind="input_rows")
        for j, expected in enumerate(probe_rows(method, index, guard)):
            require(np.array_equal(current.row(j), expected), "Gaussian probe regeneration differs")
        require(current.flat_npy_sha256() == current.summary["flat_npy_sha256"], "probe identity differs")
        chain = []
        for power in range(1, max(method["powers"]) + 1):
            forward = sequence(f"probe_{index:03d}_power_{power:02d}_forward", kind="forward", input_rows=current)
            adjoint = sequence(f"probe_{index:03d}_power_{power:02d}_adjoint", kind="adjoint", input_rows=forward)
            chain.append((forward, adjoint))
            current = adjoint
        chains.append(chain)
    require(observed == entries, "sequence manifest differs")
    result = aggregate(method, chains)
    require(result == complete["result"], "saved scalar result differs from replay")
    expected_calls = method["probes"] * 2 * max(method["powers"]) * (method["horizon"] - 1)
    require(complete["hvp_calls"] == expected_calls, "incomplete HVP count")
    guard()
    return {"status": "stored_green_graph_replayed", "result": result,
            "completion_sha256": expected_completion_sha256, "hvp_calls": expected_calls,
            "neural_hvps_independently_reexecuted": False, "source_execution_independently_attested": False}
