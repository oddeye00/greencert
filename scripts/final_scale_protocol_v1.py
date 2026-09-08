"""Draft final-scale configuration and deterministic selection contract.

This module alone is not a public experiment seal. No training entry point
may run until the entire executable constructor/observer bundle is frozen.
"""
import json


SCHEMA = "final_scale_protocol_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def scientific_settings():
    return {
        "rungs": [{"index": index, "model_dim": width, "hidden_dim": 4 * width,
                   "parameters": 48 * width**2 + 90 * width, "seed": 94101}
                  for index, width in enumerate((144, 208, 320))],
        "architecture": {"modulus": 17, "depth": 4, "heads": 4,
                         "normalization": "layernorm", "dropout": 0.0,
                         "activation": "gelu", "all_parameters_trainable": True},
        "optimizer": {"name": "full_batch_momentum", "learning_rate": 0.003,
                      "momentum": 0.9, "weight_decay": 0.01, "loss": "cross_entropy"},
        "data": {"train_fraction": 0.6, "training_count": 173,
                 "trigger_count": 58, "certification_count": 58},
        "selection": {"maximum_updates": 12000, "inspection_stride": 25,
                      "trigger_minimum_correct": 29, "target_correct": 35,
                      "persistence": 5, "horizon": 64, "clock_sweeps": 4,
                      "numeric_cap": 1e6, "candidates_per_rung": 1},
        "construction": {"additional_float64_recenterings": 1, "outer_radius": 1e-8,
                         "geometry_precision_ladder": [64, 96, 192], "transport_bits": 192,
                         "response_bits": 192, "green_hvp_bits": 128, "scalar_bits": 256,
                         "green_probes": 4, "green_powers": [1, 2], "hvp_chunk_size": 8,
                         "global_failure_probability_ratio": [1, 10000000],
                         "per_rung_failure_probability_ratio": [1, 30000000],
                         "probe_seed_base": 194101},
        "execution": {"device": "cpu", "dtype": "float64", "training_threads": 4,
                      "kernel_threads": 1, "interop_threads": 1, "maximum_kernel_workers": 2,
                      "deterministic_algorithms": True, "mha_fastpath": False,
                      "minimum_free_mib": 16384, "selection_seconds_per_rung": 7200,
                      "construction_seconds_per_rung": 86400, "observation_audit_seconds_per_rung": 3600,
                      "overall_seconds": 345600, "automatic_retry_or_resume": False},
        "stopping": {"order": "ascending_width", "continue_after_no_candidate": True,
                     "continue_after_scientific_abstention": True,
                     "stop_on_resource_limit_or_execution_failure": True,
                     "stop_on_uncovered_issued_bracket": True,
                     "retune_after_outcome": False, "keep_all_rung_dispositions": True},
    }


def validate_settings(value):
    require(canonical(value) == canonical(scientific_settings()), "scientific configuration differs")
    return value


def validate_counts(counts, *, evaluation_count, expected_length):
    require(type(counts) is list and len(counts) == expected_length, "incomplete count path")
    require(all(type(v) is int and 0 <= v <= evaluation_count for v in counts), "invalid count path")
    return counts


def first_event(counts, *, target, persistence):
    require(type(target) is int and target > 0 and type(persistence) is int
            and 1 <= persistence <= len(counts), "invalid event")
    require(all(type(v) is int and v >= 0 for v in counts), "invalid count path")
    streak = 0
    for index, count in enumerate(counts):
        streak = streak + 1 if count >= target else 0
        if streak == persistence:
            return index - persistence + 1
    return None


def eligible(current, settings):
    expected = {"training", "trigger", "certification"}
    require(type(current) is dict and set(current) == expected, "current-count fields differ")
    sizes = settings["data"]
    for key, size_key in (("training", "training_count"), ("trigger", "trigger_count"),
                          ("certification", "certification_count")):
        require(type(current[key]) is int and 0 <= current[key] <= sizes[size_key], "invalid current count")
    rule = settings["selection"]
    return current["trigger"] >= rule["trigger_minimum_correct"] and current["certification"] < rule["target_correct"]


def select_without_continuation(engine, settings, *, record, guard):
    """Select the first positive clock event; return before the next true update.

The engine's clock evaluates reference states, never true continuation.
Production must provide a durable recorder, checked resource/source guard,
and an authenticated engine. Tests use deterministic synthetic engines.
"""
    rule, sizes = settings["selection"], settings["data"]
    maximum, stride = rule["maximum_updates"], rule["inspection_stride"]
    require(type(maximum) is int and maximum >= 0 and type(stride) is int and stride > 0,
            "invalid selector horizon or stride")
    for step in range(maximum + 1):
        guard()
        if step % stride == 0:
            current = engine.current_counts()
            row = {"anchor": step, "current": current, "eligible": eligible(current, settings)}
            chosen = None
            if row["eligible"]:
                counts = engine.clock_counts(rule["horizon"], rule["clock_sweeps"], rule["numeric_cap"])
                validate_counts(counts, evaluation_count=sizes["certification_count"], expected_length=rule["horizon"] + 1)
                require(counts[0] == current["certification"], "clock parameter anchor changed")
                chosen = first_event(counts, target=rule["target_correct"], persistence=rule["persistence"])
                row.update(clock_counts=counts, predicted_offset=chosen)
            record(row)
            guard()
            if chosen is not None and chosen > 0:
                return {"status": "candidate_selected", "anchor": step,
                        "predicted_offset": chosen, "future_continuation_executed": False}
        if step == maximum:
            return {"status": "no_candidate", "completed_updates": step,
                    "future_continuation_executed": False}
        engine.advance()
