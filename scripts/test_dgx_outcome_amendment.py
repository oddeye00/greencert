"""Synthetic hardware-amendment contract tests; no current-model access."""
import copy
import json
from dgx_outcome_bridge import validate_runtime_only


def audit():
    original = {
        "runtime": {"machine": "source"}, "config": {"threads": 4, "optimizer": "momentum"},
        "anchor": 3925, "horizon": 64, "target": 35, "persistence": 5,
        "evaluation_count": 58, "training_examples": [{"index": 0}],
        "evaluation_examples": [{"index": 0}], "checkpoint_sha256": "a"*64,
        "predicted_offset": 44, "minimum_free_mib": 2048,
        "future_outcome_access": False, "automatic_retry_or_resume": False,
        "automatic_outcome_join": False, "sources": {"frozen.py": "b"*64},
    }
    actual = {"runtime": {"machine": "target"}, "torch_build": "fixed",
              "torch_cuda_build": "fixed", "execution_device": "cpu"}
    effective = {**copy.deepcopy(original), "runtime": actual["runtime"]}
    amendment = {"schema": "dgx_hardware_only_outcome_amendment_v1",
        "allowed_changed_protocol_fields": ["runtime"], "runtime_capture": actual,
        "future_outcome_access": False, "automatic_retry_or_resume": False,
        "user_authorized_hardware_amendment": True}
    validate_runtime_only(original, effective, amendment, actual)
    refused = []
    for field in (set(original)-{"runtime"}):
        changed = copy.deepcopy(effective)
        changed[field] = {"unauthorized_change": True}
        try:
            validate_runtime_only(original, changed, amendment, actual)
        except ValueError:
            refused.append("changed_"+field)
        else:
            raise AssertionError("non-runtime mutation accepted")
    for label, mutate in (
        ("added_field", lambda o,e,a,r: e.update(extra=True)),
        ("removed_field", lambda o,e,a,r: e.pop("anchor")),
        ("unknown_scope", lambda o,e,a,r: a.update(allowed_changed_protocol_fields=["runtime", "config"])),
        ("different_runtime", lambda o,e,a,r: r.update(runtime={"machine":"other"})),
        ("different_build", lambda o,e,a,r: r.update(torch_build="other")),
        ("different_device", lambda o,e,a,r: r.update(execution_device="cuda")),
        ("no_user_authority", lambda o,e,a,r: a.update(user_authorized_hardware_amendment=False)),
        ("prior_future_access", lambda o,e,a,r: a.update(future_outcome_access=True)),
        ("retry_enabled", lambda o,e,a,r: a.update(automatic_retry_or_resume=True)),
    ):
        o,e,a,r = (copy.deepcopy(v) for v in (original,effective,amendment,actual))
        mutate(o,e,a,r)
        try:
            validate_runtime_only(o,e,a,r)
        except ValueError:
            refused.append(label)
        else:
            raise AssertionError("invalid amendment accepted")
    return {"status":"PASS", "synthetic_refusals":sorted(refused), "accepted_runtime_only_case":True,
            "current_candidate_model_constructed":False, "future_outcome_access":False,
            "event_certificate_issued":False}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
