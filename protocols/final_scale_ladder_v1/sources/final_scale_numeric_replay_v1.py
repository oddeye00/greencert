"""File-backed local-arithmetic replay and conditional event assembly.

The phase manifest and expected inputs must be authenticated by the caller.
This auditor recomputes response recurrences, Green recurrences and norms,
training means, output margins, closure, and first-passage brackets. Captured
neural gradients/HVPs/jets remain supplied outward bounds: this is not an
independent re-execution of neural differentiation or source attestation.
"""
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import math

import numpy as np
from flint import arb, arb_mat, ctx

from arb_matrix_bounds import upper_float
from checkpointed_green_products import StoredRows
from final_scale_arb_codec_v1 import decode_vector, encode_vector, layout
from final_scale_green_v1 import audit as audit_green, method_record
from final_scale_neural_v1 import margins_from_logits
from final_scale_response_v1 import propagate
from final_scale_selection_v1 import load_array_reference
from outward_green_momentum import adjoint_hvp_direction, rounded_recurrence_step
from outward_green_products import streaming_norm_upper
from prospective_ledger_v1 import encode
from window_event_assembly import (AnchorOutputRow, DriftRow, GreenBound, Identity,
    MarginBound, OutputRow, PointMargin, ResponseBound, assemble)


EXPECTED = {"parameters", "horizon", "config", "spec", "normalization_eps", "options",
            "probe_seed", "probability_ratio", "input_raw_sha256", "training_count", "evaluation_count"}


def require(value, message):
    if not value:
        raise ValueError(message)


def same(actual, expected, message):
    require(encode(actual) == encode(expected), message)


def raw_sha(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def object_sha(value):
    return hashlib.sha256(encode(value)).hexdigest()


def nonnegative(value):
    require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
            "invalid nonnegative neural bound")
    return value


class Graph:
    def __init__(self, reader, guard, stack):
        self.reader, self.guard, self.stack = reader, guard, stack
        self.seen = set()

    def record(self, path, reference=None):
        if reference is not None:
            same(reference, self.reader.files[path], "record reference differs: " + path)
        result = self.reader.record(path)
        self.seen.add(path)
        return result

    def array(self, reference, path, shape, dtype):
        value = load_array_reference(self.reader, reference, path=path, shape=shape, dtype=dtype)
        self.seen.add(path)
        self.stack.callback(value._mmap.close)
        return value

    def balls(self, reference, path, length, bits):
        same(reference["codec"], layout(bits), "interval codec policy differs")
        value = load_array_reference(self.reader, reference["array"], path=path,
            shape=(length, len(layout(bits)["columns"])), dtype="<i8")
        self.seen.add(path)
        try:
            return decode_vector(value, precision_bits=bits, guard=self.guard)
        finally:
            value._mmap.close()


def audit_numeric(reader, *, expected, target, persistence, guard):
    """Authenticate expected numerical inputs, then replay without any future.

`expected` is reconstructed from the externally validated selected case and
frozen method, never accepted merely because it matches a producer flag.
This function does not authorize an observer; a separate sealed disposition
must establish source/runtime obligations and interpret the returned bracket.
"""
    require(type(expected) is dict and set(expected) == EXPECTED, "incomplete external numerical contract")
    old = ctx.prec
    try:
        with ExitStack() as stack:
            return _audit(Graph(reader, guard, stack), expected, target, persistence)
    finally:
        ctx.prec = old


def _audit(g, expected, target, persistence):
    reader, guard = g.reader, g.guard
    method = g.record("numeric_method.json")
    rebuilt = {key: expected[key] for key in EXPECTED - {"input_raw_sha256"}}
    rebuilt.update(schema="final_scale_numeric_core_v1", inputs=method["inputs"],
        known_physical_anchor=True, local_derivatives_losslessly_captured=True, event_certificate_issued=False)
    same(method, rebuilt, "numerical method differs from external contract")
    n, H, nt, ne = (expected[k] for k in ("parameters", "horizon", "training_count", "evaluation_count"))
    require(all(type(v) is int and v > 0 for v in (n, H, nt, ne)), "invalid external dimensions")
    cfg, options = method["config"], method["options"]
    eta, mu = cfg["learning_rate"], cfg["momentum"]
    classes = cfg["modulus"]
    shapes = {"parameter": ((n,), "<f8"), "unscaled_velocity": ((n,), "<f8"),
        "clock": ((H+1, 2*n), "<f8"), "train_pairs": ((nt, 2), "<i8"),
        "train_labels": ((nt,), "<i8"), "evaluation_pairs": ((ne, 2), "<i8"),
        "evaluation_labels": ((ne,), "<i8")}
    require(set(method["inputs"]) == set(shapes) == set(expected["input_raw_sha256"]), "input population differs")
    inputs = {}
    for name, (shape, dtype) in shapes.items():
        inputs[name] = g.array(method["inputs"][name], f"inputs/{name}.npy", shape, dtype)
        require(raw_sha(inputs[name]) == expected["input_raw_sha256"][name], "external input identity differs: " + name)
    result = g.record("numeric_result.json")
    g.record("numeric_method.json", result["numeric_method"])
    reference = g.array(result["reference"], "reference/scaled.npy", (H+1, 2*n), "<f8")
    correction = g.array(result["float_correction"], "reference/float_correction.npy", (H+1, 2*n), "<f8")
    require(np.array_equal(reference[0, :n], inputs["parameter"]) and
            np.array_equal(reference[0], inputs["clock"][0]) and np.count_nonzero(correction[0]) == 0,
            "reference physical anchor differs")
    for j in range(H+1):
        guard()
        require(np.array_equal(reference[j], inputs["clock"][j] + correction[j]), "reference correction sum differs")
    population = lambda role: object_sha({"pairs": raw_sha(inputs[role+"_pairs"]),
                                         "labels": raw_sha(inputs[role+"_labels"])})
    identity = Identity(object_sha({"parameter": raw_sha(inputs["parameter"]),
        "unscaled_velocity": raw_sha(inputs["unscaled_velocity"])}), raw_sha(reference),
        population("train"), population("evaluation"), n, H, "scaled_momentum_euclidean")
    same(g.record("identity.json"), asdict(identity), "numeric identity differs")
    same(result["identity"], asdict(identity), "result identity differs")
    method_sha = reader.files["numeric_method.json"]["sha256"]
    response_refs = []
    response_bits = options["response_bits"]

    def derivative(j, direction):
        path = f"response_kernels/step_{j:03d}"
        row = g.record(path + ".json")
        chunks = [[k, min(k+options["hvp_chunk_size"], nt)] for k in range(0, nt, options["hvp_chunk_size"])]
        same(row, {"producer": "chunked_objective_gradient_hvp", "numeric_method_sha256": method_sha,
            "step": j, "precision_bits": response_bits, "parameter_sha256": raw_sha(reference[j, :n]),
            "direction_sha256": raw_sha(direction), "gradient": row["gradient"], "hvp": row["hvp"],
            "chunks": chunks, "examples": nt, "parameters": n}, "response kernel binding differs")
        return (g.balls(row["gradient"], path+"_gradient.npy", n, response_bits),
                g.balls(row["hvp"], path+"_hvp.npy", n, response_bits))

    def response_row(j, values, recurrence):
        path = f"response/step_{j:03d}"
        stored = g.record(path+".json")
        same(stored, {"numeric_method_sha256": method_sha, "index": j,
            "array": stored["array"], "recurrence": recurrence}, "response local arithmetic differs")
        array = g.array(stored["array"], path+".npy", (2*n,), "<f8")
        require(np.array_equal(array, values), "response rounded state differs")
        # Do not keep all response rows memory-mapped during a large replay.
        array._mmap.close()
        response_refs.append(reader.files[path+".json"])

    response = propagate(reference, inputs["parameter"], inputs["unscaled_velocity"], derivative,
        learning_rate=eta, momentum=mu, precision_bits=response_bits, guard=guard, persist=response_row)
    same(g.record("response/summary.json", result["response"]),
         {"result": response, "records": response_refs}, "response aggregate differs")
    green_capture = g.record("green_capture.json", result["green"])
    bindings = reader.manifest["method"]["bindings"]
    contract = {"protocol_sha256": bindings["protocol_sha256"],
        "source_manifest_sha256": bindings["source_manifest_sha256"], "reference_sha256": identity.reference_sha256,
        "training_set_sha256": identity.training_set_sha256, "checkpoint_sha256": identity.candidate_sha256}
    same(green_capture["contract"], contract, "Green contract differs")
    green_method = method_record(parameters=n, horizon=H, learning_rate=eta, momentum=mu,
        probes=options["green_probes"], powers=options["green_powers"], seed=expected["probe_seed"],
        failure_ratio=expected["probability_ratio"], precision_bits=options["green_hvp_bits"],
        scalar_bits=options["scalar_bits"], contract=contract)
    same(g.record("green/method.json"), green_method, "Green numerical policy differs")
    green_replay = audit_green(reader.root / "green",
        expected_completion_sha256=green_capture["result"]["completion_sha256"], expected_contract=contract, guard=guard)
    green = green_replay["result"]
    same(green_capture["result"], {**green, "completion_sha256": green_replay["completion_sha256"],
         "hvp_calls": green_replay["hvp_calls"]}, "Green aggregate capture differs")
    g.seen.update(path for path in reader.files if path.startswith("green/"))
    bits, query, row_count, kernel_refs = options["green_hvp_bits"], 0, 0, []
    ctx.prec = bits
    for probe in range(options["green_probes"]):
        current = StoredRows(reader.root / "green" / f"probe_{probe:03d}")
        for power in range(1, max(options["green_powers"])+1):
            for transpose in (False, True):
                kind = "adjoint" if transpose else "forward"
                output = StoredRows(reader.root / "green" / f"probe_{probe:03d}_power_{power:02d}_{kind}")
                previous = None
                for j in (range(H-1, -1, -1) if transpose else range(H)):
                    guard()
                    actual, injection = output.row(j), current.row(j)
                    error = 0.0
                    if previous is None:
                        require(np.array_equal(actual, injection), "Green endpoint copy differs")
                    else:
                        step = j+1 if transpose else j
                        direction = adjoint_hvp_direction(previous) if transpose else [arb(float(v)) for v in previous[:n]]
                        path = f"green_kernels/query_{query:06d}"
                        row = g.record(path+".json")
                        same(row, {"producer": "factored_chunked_hvp", "numeric_method_sha256": method_sha,
                            "query": query, "probe": probe, "power": power, "transpose": transpose,
                            "step": step, "precision_bits": bits, "parameter_sha256": raw_sha(reference[step, :n]),
                            "encoded_direction_sha256": raw_sha(encode_vector(direction, precision_bits=bits, guard=guard)),
                            "examples": nt, "parameters": n, "chunk_size": options["hvp_chunk_size"], "hvp": row["hvp"]},
                            "Green kernel binding differs")
                        hvp = g.balls(row["hvp"], path+".npy", n, bits)
                        item = rounded_recurrence_step(previous, injection, hvp, learning_rate=eta, momentum=mu, transpose=transpose)
                        require(np.array_equal(actual, item["next_state"]), "Green rounded recurrence differs")
                        error = item["local_residual_norm_upper"]
                        kernel_refs.append(reader.files[path+".json"])
                        query += 1
                        del hvp, direction, item
                    require(output.summary["rows"][j]["local_residual_norm_upper"] == error,
                            "Green local residual differs")
                    ctx.prec = max(256, bits)
                    require(output.summary["rows"][j]["norm_upper"] == streaming_norm_upper(actual), "Green row norm differs")
                    ctx.prec = bits
                    previous = actual
                    row_count += 1
                current = output
    same(green_capture["kernels"], kernel_refs, "Green captured query population differs")
    require(query == green_replay["hvp_calls"], "Green replay query count differs")
    ctx.prec = options["transport_bits"]
    ids = tuple(f"evaluation_{j}" for j in range(ne))
    outputs, drifts, neural_refs, point_refs = [], [], [], []
    for j in range(H+1):
        guard()
        ctx.prec = options["transport_bits"]
        path = f"point_logits/step_{j:03d}"
        point = g.record(path+".json")
        same(point, {"producer": "arb_deep_logits", "numeric_method_sha256": method_sha, "step": j,
            "parameter_sha256": raw_sha(reference[j, :n]), "shape": [ne, classes],
            "evaluation_set_sha256": identity.evaluation_set_sha256, "values": point["values"]}, "point output binding differs")
        values = g.balls(point["values"], path+".npy", ne*classes, options["transport_bits"])
        margins = margins_from_logits(arb_mat(ne, classes, values), inputs["evaluation_labels"])
        point_refs.append(reader.files[path+".json"])
        if j == 0:
            outputs.append(AnchorOutputRow(identity, tuple(PointMargin(ids[k], m["lower"], m["upper"])
                for k, m in enumerate(margins)), "outward"))
            continue
        summary_path = f"neural/step_{j:03d}/summary.json"
        summary = g.record(summary_path)
        records, weights, population_rows = summary["records"], {}, {"training": [], "evaluation": []}
        require(type(records) is list and records and len({r["path"] for r in records}) == len(records),
                "missing or duplicate neural records")
        for ref in records:
            row = g.record(ref["path"], ref)
            role, index = row["role"], row["index"]
            require(type(index) is int and index >= 0 and role in ("weights", "training", "evaluation"), "invalid neural record role")
            require(ref["path"] == f"neural/step_{j:03d}/{role}_{index:03d}.json", "neural row path differs")
            same(row, {"numeric_method_sha256": method_sha, "step": j, "parameter_sha256": raw_sha(reference[j, :n]),
                "role": role, "index": index, "bound": row["bound"]}, "neural row binding differs")
            bound = row["bound"]
            if role == "weights":
                require(index in options["geometry_precision_ladder"] and bound["geometry_bits"] == index,
                        "weight precision differs")
                require(type(bound["weights"]) is dict and bound["weights"], "empty weight enclosure")
                weights[index] = bound["weights"]
                continue
            key = "train" if role == "training" else "evaluation"
            require(index == len(population_rows[role]) and index < len(inputs[key+"_labels"]), "neural population order differs")
            require(bound["status"] == "verified" and bound["role"] == role and bound["index"] == index and
                bound["pair"] == inputs[key+"_pairs"][index].tolist() and bound["label"] == int(inputs[key+"_labels"][index]),
                "neural example identity differs")
            require(bound["geometry_bits"] in weights and bound["transport_bits"] == options["transport_bits"],
                    "neural precision or weight dependency differs")
            for value in bound["logit_jet"].values():
                nonnegative(value)
            nonnegative(bound["point_gains"]["logits"])
            require(bound["margin_lower"] <= bound["margin_upper"], "invalid producer margin")
            if role == "training":
                nonnegative(bound["drift_upper"])
            else:
                require(bound["drift_upper"] is None, "evaluation row has training drift")
            population_rows[role].append(bound)
        train, evaluation = population_rows["training"], population_rows["evaluation"]
        require(len(train) == (nt if j < H else 0) and len(evaluation) == ne, "incomplete neural population")
        ctx.prec = max(256, options["transport_bits"])
        mean = upper_float(sum((arb(row["drift_upper"]) for row in train), arb(0))/len(train)) if train else None
        producer_outputs = [{"index": k, "label": row["label"], "lower": row["margin_lower"], "upper": row["margin_upper"],
            "point_jacobian_upper": row["point_gains"]["logits"], "ball_hessian_upper": row["logit_jet"]["second"]}
            for k, row in enumerate(evaluation)]
        same(summary["result"], {"status": "complete_neural_row", "training_count": len(train), "evaluation_count": ne,
            "mean_training_drift_upper": mean, "radius": options["outer_radius"], "output_bounds": producer_outputs,
            "examples": len(train)+ne, "parameter_sha256": raw_sha(reference[j, :n]), "event_certificate_issued": False},
            "neural mean or output summary differs")
        if train:
            drifts.append(DriftRow(identity, j, options["outer_radius"], nt, mean, "outward"))
        outputs.append(OutputRow(identity, j, options["outer_radius"], tuple(MarginBound(ids[k], m["lower"], m["upper"],
            evaluation[k]["point_gains"]["logits"], evaluation[k]["logit_jet"]["second"])
            for k, m in enumerate(margins)), "outward"))
        neural_refs.append(reader.files[summary_path])
    same(result, {"status": "complete_numeric_evidence_constructed", "numeric_method": reader.files["numeric_method.json"],
        "identity": asdict(identity), "reference": result["reference"], "float_correction": result["float_correction"],
        "response": reader.files["response/summary.json"], "neural_rows": neural_refs, "point_logits": point_refs,
        "green": reader.files["green_capture.json"], "event_certificate_issued": False,
        "independent_semantic_replay_required": True, "future_observation_executed": False}, "numeric completion graph differs")
    prefixes = ("inputs/", "reference/", "response/", "response_kernels/", "green/", "green_kernels/", "neural/", "point_logits/")
    require(all(path in g.seen for path in reader.files if path.startswith(prefixes)), "unconsumed numerical file")
    if green["gain_upper"] is None:
        assembly = {"certificate_issued": False, "bracket": None, "reason": "green_bound_abstention"}
    else:
        assembly = assemble(identity=identity, training_count=nt, example_ids=ids, domain=options["outer_radius"],
            drift_rows=drifts, output_rows=outputs,
            response=ResponseBound(identity, response["parameter_norms"], response["residual_upper"],
                response["first_injection_error_upper"], True, "outward"),
            green=GreenBound(identity, green["gain_upper"], "outward", green["probability_scope"],
                green["failure_probability"], options["green_probes"], green["complete_probes"]),
            target=target, persistence=persistence, precision_bits=options["scalar_bits"])
        require(assembly["inputs_compatible"], "replayed assembly inputs incompatible")
    guard()
    return {"status": "numeric_graph_and_local_arithmetic_replayed", "phase_manifest_sha256": reader.identity,
        "identity": asdict(identity), "response_steps": H, "green_hvp_records": query, "green_product_rows": row_count,
        "neural_training_records": nt*(H-1), "point_logits_replayed": (H+1)*ne*classes, "assembly": assembly,
        "event_certificate_issued": False, "future_observation_authorized": False,
        "neural_kernels_independently_reexecuted": False, "float64_reference_construction_reexecuted": False,
        "scope": "authenticated supplied neural enclosures; outward local replay and event assembly; ideal Gaussian event"}
