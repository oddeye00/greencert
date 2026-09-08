"""Lossless fixed-width storage of finite Arb midpoint/radius vectors.

Each row stores the exact dyadic midpoint and the exact dyadic radius, not
a decimal rendering or rounded binary64 endpoints. Loading fails if the
chosen precision cannot reproduce both components exactly. This authenticates
an interval representation; it does not prove how a neural bound was obtained.
"""
import numpy as np
from flint import arb, ctx


SCHEMA = "arb_dyadic_vector_v1"
WORD = 2**64
SIGN = 2**63


def require(condition, message):
    if not condition:
        raise ValueError(message)


def layout(precision_bits):
    require(type(precision_bits) is int and 64 <= precision_bits <= 1024, "unsupported codec precision")
    limbs = (precision_bits+63)//64
    return {"schema": SCHEMA, "precision_bits": precision_bits, "dtype": "<i8",
            "columns": ["mid_sign", "mid_exponent", "radius_mantissa", "radius_exponent"] +
                       [f"mid_limb_{j}" for j in range(limbs)],
            "limb_encoding": "little-endian unsigned 64-bit words stored as signed two's-complement int64",
            "interval_components_preserved_exactly": True}


def checked_int64(value):
    require(type(value) is int and -SIGN <= value < SIGN, "dyadic exponent or radius exceeds int64")
    return value


def encode_vector(values, *, precision_bits, guard):
    spec = layout(precision_bits)
    require(type(values) in (list, tuple) and values, "nonempty Arb vector required")
    limbs = len(spec["columns"])-4
    result = np.zeros((len(values), 4+limbs), dtype="<i8")
    for index, value in enumerate(values):
        if index % 1024 == 0:
            guard()
        require(isinstance(value, arb) and value.is_finite(), "nonfinite or non-Arb value")
        m, e = map(int, value.mid().man_exp())
        r, re = map(int, value.rad().man_exp())
        magnitude = abs(m)
        require(magnitude.bit_length() <= precision_bits, "midpoint exceeds registered precision")
        require(r >= 0, "negative interval radius")
        result[index, :4] = (0 if m == 0 else (1 if m > 0 else -1),
                            checked_int64(e), checked_int64(r), checked_int64(re))
        for limb in range(limbs):
            word = (magnitude >> (64*limb)) & (WORD-1)
            result[index, 4+limb] = word if word < SIGN else word-WORD
    guard()
    return result


def decode_vector(encoded, *, precision_bits, guard):
    spec = layout(precision_bits)
    require(isinstance(encoded, np.ndarray) and encoded.dtype == np.dtype("<i8") and encoded.ndim == 2
            and encoded.shape[0] > 0 and encoded.shape[1] == len(spec["columns"]), "encoded vector layout differs")
    require(ctx.prec >= precision_bits, "decoding precision below registered midpoint precision")
    result = []
    for index, row in enumerate(encoded):
        if index % 1024 == 0:
            guard()
        sign, e, r, re = map(int, row[:4])
        require(sign in (-1, 0, 1) and r >= 0, "invalid sign or radius")
        magnitude = sum((int(word) % WORD) << (64*j) for j, word in enumerate(row[4:]))
        require(magnitude.bit_length() <= precision_bits and
                ((sign == 0 and magnitude == 0 and e == 0) or
                 (sign != 0 and magnitude > 0 and magnitude % 2 == 1)), "noncanonical dyadic midpoint")
        require((r == 0 and re == 0) or (r > 0 and r % 2 == 1), "noncanonical dyadic radius")
        midpoint = (sign*magnitude, e)
        radius = (r, re)
        value = arb(midpoint, radius)
        # Some FLINT builds round a full 30-bit radius upward even when the
        # input dyadic is representable. Try an interior preimage of that
        # rounding bin, then REQUIRE exact agreement with the stored radius.
        # This reconstructs a representation; it is not permission to shrink
        # a bound. No value is returned unless BOTH components match exactly.
        if r.bit_length() == 30 and tuple(map(int, value.rad().man_exp())) != radius:
            value = arb(midpoint, (2*r-1, re-1))
        require(value.is_finite() and tuple(map(int, value.mid().man_exp())) == midpoint and
                tuple(map(int, value.rad().man_exp())) == radius, "decoded interval components changed")
        result.append(value)
    guard()
    return result
