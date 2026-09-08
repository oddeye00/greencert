"""Generic physical-anchor response using the established signed recurrence.

The old two-response construction propagates the particular response from
zero and the physical velocity-offset response separately, then adds them.
Here the same linear recurrence starts at the encoded physical offset.
This is a numerical refactoring, not a different nonlinear remainder theorem.
The callback must return outward enclosures of the exact same objective's
gradient and HVP at each reference parameter, in the supplied direction.
"""
import hashlib

import numpy as np
from flint import ctx

from known_anchor_response import encode_velocity_offset
from outward_green_products import sequence_upper
from outward_variational_response import signed_momentum_step


def require(value, message):
    if not value:
        raise ValueError(message)


def propagate(reference, parameter, unscaled_velocity, derivatives, *, learning_rate,
              momentum, precision_bits=192, guard, persist):
    """Stream one authenticated combined response, retaining every local error.

Returns scalar inputs for a later independent audit/assembly. It issues no
certificate. The caller authenticates the reference, physical checkpoint,
derivative callback and persistence implementation before invoking this API.
"""
    require(type(precision_bits) is int and precision_bits >= 64, "invalid response precision")
    require(isinstance(reference, np.ndarray) and reference.dtype == np.float64 and
            reference.ndim == 2 and len(reference) >= 2 and reference.shape[1] > 0
            and reference.shape[1] % 2 == 0, "invalid reference layout")
    n, H = reference.shape[1] // 2, len(reference) - 1
    for value in (parameter, unscaled_velocity):
        require(isinstance(value, np.ndarray) and value.dtype == np.float64 and
                value.shape == (n,) and np.isfinite(value).all(), "invalid physical checkpoint")
    require(np.isfinite(reference).all() and np.array_equal(reference[0, :n], parameter),
            "parameter anchor differs")
    require(callable(derivatives) and callable(guard) and callable(persist), "callbacks required")
    guard()
    old = ctx.prec
    ctx.prec = precision_bits
    try:
        encoding = encode_velocity_offset(unscaled_velocity, reference[0, n:],
                                           learning_rate=learning_rate, momentum=momentum)
        encoded = encoding.pop("encoded_velocity_offset")
        response = np.concatenate((np.zeros(n, dtype=np.float64), encoded))
        initial = {"encoding": encoding,
                   "encoded_velocity_offset_sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
                   "response_sha256": hashlib.sha256(response.tobytes()).hexdigest()}
        persist(0, response.copy(), initial)
        errors, norms = [], [0.0]
        for j in range(H):
            guard()
            gradient, hvp = derivatives(j, response[:n].copy())
            require(ctx.prec == precision_bits, "derivative callback changed response precision")
            row = signed_momentum_step(reference[j], reference[j+1], response, gradient, hvp,
                                       learning_rate=learning_rate, momentum=momentum)
            response = row.pop("next_response")
            require(np.isfinite(response).all(), "nonfinite response")
            row.update(step=j + 1, derivative_step=j,
                       response_sha256=hashlib.sha256(response.tobytes()).hexdigest())
            errors.append(row["recurrence_error_upper"])
            norms.append(row["parameter_response_norm_upper"])
            persist(j + 1, response.copy(), row)
            guard()
        ctx.prec = max(256, precision_bits)
        residual = sequence_upper(errors)
        return {"status": "physical_anchor_response_propagated", "parameters": n, "horizon": H,
                "parameter_norms": tuple(norms), "residual_upper": residual,
                "first_injection_error_upper": encoding["unrepresented_first_injection_norm_upper"],
                "parameter_anchor_matches": True, "initial": initial,
                "local_residuals": tuple(errors), "precision_bits": precision_bits,
                "event_certificate_issued": False,
                "scope": "conditional on authenticated outward derivative callback; assembly pending"}
    finally:
        ctx.prec = old
