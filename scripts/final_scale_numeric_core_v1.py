"""Generic outward constructor with losslessly captured local derivatives.

This producer builds evidence, not an issued certificate. The production
phase must first authenticate a selected checkpoint, then independently
replay this evidence before sealing a certificate/abstention disposition.
No true optimizer continuation is evaluated by this module.
"""
from dataclasses import asdict
import hashlib

import numpy as np
import torch
from flint import arb, ctx

from arb_chunked_objective import chunked_objective_gradient_hvp
from arb_deep_transformer_forward import arb_deep_logits
from arb_factored_regularizer_hvp import factored_chunked_hvp
from final_scale_arb_codec_v1 import encode_vector, layout
from final_scale_green_v1 import construct as construct_green
from final_scale_neural_v1 import recenter_scaled, envelope_row, validate_population, validate_parameter
from final_scale_response_v1 import propagate
from prospective_ledger_v1 import encode
from window_event_assembly import Identity


SCHEMA = "final_scale_numeric_core_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def raw_sha(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def digest_record(value):
    return hashlib.sha256(encode(value)).hexdigest()


def population_identity(pairs, labels):
    return digest_record({"pairs": raw_sha(pairs), "labels": raw_sha(labels)})


def capture(writer, path, values, bits, guard):
    return {"codec": layout(bits), "array": writer.array(path,
            encode_vector(values, precision_bits=bits, guard=guard))}


def build(writer, *, parameter, unscaled_velocity, clock, train_pairs, train_labels,
          evaluation_pairs, evaluation_labels, template, spec, config, normalization_eps,
          options, probe_seed, probability_ratio, guard):
    """Stream the complete numerical input graph for a later semantic replay.

The caller must authenticate all arguments and enforce the frozen options.
The original numerical helpers are used unchanged. Captured midpoint/radius
vectors allow local response/Green arithmetic to be replayed without rerunning
neural differentiation; validity of the supplied neural enclosures remains
an obligation of the authenticated outward producers.
"""
    validate_parameter(parameter, spec)
    require(isinstance(unscaled_velocity, np.ndarray) and unscaled_velocity.dtype == np.float64 and
            unscaled_velocity.shape == parameter.shape and np.isfinite(unscaled_velocity).all(), "invalid physical velocity")
    require(isinstance(clock, np.ndarray) and clock.dtype == np.float64 and clock.ndim == 2 and
            clock.shape[1] == 2*len(parameter) and len(clock) >= 2 and np.isfinite(clock).all(), "invalid clock")
    require(np.array_equal(clock[0, :len(parameter)], parameter) and
            np.array_equal(clock[0, len(parameter):], config.learning_rate*unscaled_velocity), "clock physical anchor differs")
    require(config.loss == "cross_entropy" and config.normalization == "layernorm" and config.dtype == "float64",
            "unsupported complete numerical model")
    require(options["additional_float64_recenterings"] == 1 and
            type(options["hvp_chunk_size"]) is int and options["hvp_chunk_size"] > 0, "unsupported constructor options")
    require(len(normalization_eps) == config.depth and all(len(p) == 2 for p in normalization_eps), "normalization metadata incomplete")
    train_pairs, train_labels = validate_population(train_pairs, train_labels, config.modulus)
    evaluation_pairs, evaluation_labels = validate_population(evaluation_pairs, evaluation_labels, config.modulus)
    n, H = len(parameter), len(clock)-1
    guard()
    inputs = {"parameter": writer.array("inputs/parameter.npy", parameter),
              "unscaled_velocity": writer.array("inputs/unscaled_velocity.npy", unscaled_velocity),
              "clock": writer.array("inputs/clock.npy", clock),
              "train_pairs": writer.array("inputs/train_pairs.npy", np.ascontiguousarray(train_pairs)),
              "train_labels": writer.array("inputs/train_labels.npy", np.ascontiguousarray(train_labels)),
              "evaluation_pairs": writer.array("inputs/evaluation_pairs.npy", np.ascontiguousarray(evaluation_pairs)),
              "evaluation_labels": writer.array("inputs/evaluation_labels.npy", np.ascontiguousarray(evaluation_labels))}
    method = {"schema": SCHEMA, "parameters": n, "horizon": H, "config": asdict(config),
              "spec": {"names": list(spec.names), "sizes": list(spec.sizes), "shapes": [list(s) for s in spec.shapes]},
              "normalization_eps": [list(pair) for pair in normalization_eps], "options": options,
              "probe_seed": probe_seed, "probability_ratio": list(probability_ratio), "inputs": inputs,
              "training_count": len(train_pairs), "evaluation_count": len(evaluation_pairs),
              "known_physical_anchor": True, "local_derivatives_losslessly_captured": True,
              "event_certificate_issued": False}
    method_reference = writer.record("numeric_method.json", method)
    train_x = torch.tensor(np.asarray(train_pairs), dtype=torch.long)
    train_y = torch.tensor(np.asarray(train_labels), dtype=torch.long)
    reference, correction = recenter_scaled(torch.tensor(np.asarray(clock), dtype=torch.float64),
        train_x, train_y, template, spec, config, guard=guard)
    reference = reference.detach().numpy()
    correction = correction.detach().numpy()
    reference_info = writer.array("reference/scaled.npy", reference)
    correction_info = writer.array("reference/float_correction.npy", correction)
    identity = Identity(digest_record({"parameter": raw_sha(parameter), "unscaled_velocity": raw_sha(unscaled_velocity)}),
        raw_sha(reference), population_identity(train_pairs, train_labels),
        population_identity(evaluation_pairs, evaluation_labels), n, H, "scaled_momentum_euclidean")
    writer.record("identity.json", asdict(identity))
    del correction
    response_records = []
    response_bits = options["response_bits"]
    chunk_size = options["hvp_chunk_size"]

    def response_derivatives(j, direction):
        guard()
        require(ctx.prec == response_bits and 0 <= j < H, "response derivative context differs")
        values = chunked_objective_gradient_hvp(reference[j, :n], train_pairs, train_labels, spec, config,
            normalization_eps=normalization_eps, direction=direction, chunk_size=chunk_size,
            progress=lambda start, end: guard())
        gradient_record = capture(writer, f"response_kernels/step_{j:03d}_gradient.npy", values["gradient"], response_bits, guard)
        hvp_record = capture(writer, f"response_kernels/step_{j:03d}_hvp.npy", values["hvp"], response_bits, guard)
        writer.record(f"response_kernels/step_{j:03d}.json", {"producer": "chunked_objective_gradient_hvp",
            "numeric_method_sha256": method_reference["sha256"], "step": j, "precision_bits": response_bits,
            "parameter_sha256": raw_sha(reference[j, :n]), "direction_sha256": raw_sha(direction),
            "gradient": gradient_record, "hvp": hvp_record, "chunks": values["chunks"],
            "examples": values["examples"], "parameters": values["parameters"]})
        return values["gradient"], values["hvp"]

    def persist_response(j, values, row):
        array = writer.array(f"response/step_{j:03d}.npy", values)
        record = writer.record(f"response/step_{j:03d}.json", {"numeric_method_sha256": method_reference["sha256"],
            "index": j, "array": array, "recurrence": row})
        response_records.append(record)

    response = propagate(reference, parameter, unscaled_velocity, response_derivatives,
        learning_rate=config.learning_rate, momentum=config.momentum, precision_bits=response_bits,
        guard=guard, persist=persist_response)
    response_info = writer.record("response/summary.json", {"result": response, "records": response_records})
    neural_rows, point_rows = [], []
    old = ctx.prec
    try:
        for j in range(H+1):
            guard()
            ctx.prec = options["transport_bits"]
            point = arb_deep_logits(reference[j, :n], evaluation_pairs, spec, config, normalization_eps=normalization_eps)
            values = capture(writer, f"point_logits/step_{j:03d}.npy", point.entries(), options["transport_bits"], guard)
            point_rows.append(writer.record(f"point_logits/step_{j:03d}.json", {
                "producer": "arb_deep_logits", "numeric_method_sha256": method_reference["sha256"],
                "step": j, "parameter_sha256": raw_sha(reference[j, :n]), "shape": [len(evaluation_pairs), config.modulus],
                "evaluation_set_sha256": identity.evaluation_set_sha256, "values": values}))
            if j == 0:
                continue
            captured = []

            def persist_neural(role, index, row):
                captured.append(writer.record(f"neural/step_{j:03d}/{role}_{index:03d}.json", {
                    "numeric_method_sha256": method_reference["sha256"], "step": j,
                    "parameter_sha256": raw_sha(reference[j, :n]), "role": role, "index": index, "bound": row}))

            result = envelope_row(reference[j, :n], train_pairs, train_labels, evaluation_pairs, evaluation_labels,
                spec, config, normalization_eps=normalization_eps, include_training=j < H,
                radius=options["outer_radius"], geometry_bits=tuple(options["geometry_precision_ladder"]),
                transport_bits=options["transport_bits"], guard=guard, persist=persist_neural)
            neural_rows.append(writer.record(f"neural/step_{j:03d}/summary.json", {"result": result, "records": captured}))
    finally:
        ctx.prec = old

    green_bits, queries = options["green_hvp_bits"], []
    powers, probes = options["green_powers"], options["green_probes"]

    def green_hvp(j, direction):
        guard()
        require(H > 1 and ctx.prec == green_bits, "invalid Green HVP context")
        query = len(queries)
        phase, offset = divmod(query, H-1)
        probe, within = divmod(phase, 2*max(powers))
        power, transpose = within//2+1, bool(within % 2)
        require(probe < probes and j == (H-1-offset if transpose else offset+1), "unexpected Green query schedule")
        encoded_direction = encode_vector(direction, precision_bits=green_bits, guard=guard)
        direction_sha = raw_sha(encoded_direction)
        del encoded_direction
        values = factored_chunked_hvp(reference[j, :n], train_pairs, train_labels, spec, config,
            normalization_eps=normalization_eps, direction=direction, chunk_size=chunk_size,
            progress=lambda start, end: guard())
        captured = capture(writer, f"green_kernels/query_{query:06d}.npy", values, green_bits, guard)
        record = writer.record(f"green_kernels/query_{query:06d}.json", {"producer": "factored_chunked_hvp",
            "numeric_method_sha256": method_reference["sha256"], "query": query, "probe": probe,
            "power": power, "transpose": transpose, "step": j, "precision_bits": green_bits,
            "parameter_sha256": raw_sha(reference[j, :n]), "encoded_direction_sha256": direction_sha,
            "examples": len(train_pairs), "parameters": n, "chunk_size": chunk_size, "hvp": captured})
        queries.append(record)
        return values

    contract = {"protocol_sha256": writer.method["bindings"]["protocol_sha256"],
                "source_manifest_sha256": writer.method["bindings"]["source_manifest_sha256"],
                "reference_sha256": identity.reference_sha256, "training_set_sha256": identity.training_set_sha256,
                "checkpoint_sha256": identity.candidate_sha256}
    green = construct_green(writer.root / "green", green_hvp, parameters=n, horizon=H,
        learning_rate=config.learning_rate, momentum=config.momentum, probes=probes, powers=powers,
        seed=probe_seed, failure_ratio=probability_ratio, precision_bits=green_bits,
        scalar_bits=options["scalar_bits"], contract=contract, guard=guard, progress=lambda row: guard(), sync=writer.sync)
    require(len(queries) == probes*2*max(powers)*(H-1), "incomplete captured Green family")
    green_info = writer.record("green_capture.json", {"result": green, "contract": contract, "kernels": queries})
    guard()
    result = {"status": "complete_numeric_evidence_constructed", "numeric_method": method_reference,
              "identity": asdict(identity), "reference": reference_info, "float_correction": correction_info,
              "response": response_info, "neural_rows": neural_rows, "point_logits": point_rows,
              "green": green_info, "event_certificate_issued": False,
              "independent_semantic_replay_required": True, "future_observation_executed": False}
    writer.record("numeric_result.json", result)
    return result
