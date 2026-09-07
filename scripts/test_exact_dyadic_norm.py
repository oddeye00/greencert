"""Independent rational-oracle tests for exact dyadic squared norms."""
import json
import math

import numpy as np

from exact_dyadic_norm import MAX_CHUNK, squared_integer, encloses_norm, norm_upper


def oracle(values):
    total = 0
    for value in values:
        numerator, denominator = float(value).as_integer_ratio()
        exponent = denominator.bit_length()-1
        if denominator != 1 << exponent:
            raise AssertionError("oracle denominator is not dyadic")
        total += numerator*numerator << (2148-2*exponent)
    return total


def main():
    cases, refused = [], []
    def check(name, values, chunks=(65536,)):
        expected = oracle(values)
        for chunk in chunks:
            assert squared_integer(values, chunk_size=chunk) == expected, name
        cases.append({"name": name, "coordinates": len(values), "chunks": list(chunks)})

    check("signed_zero", np.array([0., -0.]), (1, 2, 65536))
    check("empty", np.array([], dtype=np.float64))
    check("subnormal_boundary", np.array([math.ulp(0.), -math.ulp(0.),
          np.nextafter(np.finfo(float).tiny, 0.), np.finfo(float).tiny]), (1, 3, 65536))
    check("maximum_magnitudes", np.array([np.finfo(float).max, -np.finfo(float).max, 1., 2., 3.]), (1, 4))
    mantissas = (0, 1, (1<<32)-1, 1<<32, (1<<52)-1)
    words = np.array([(e<<52)|m for e in range(2047) for m in mantissas], dtype=np.uint64)
    check("all_finite_exponents", words.view(np.float64), (7, 65536))
    rng = np.random.default_rng(7192301)
    random_words = rng.bit_generator.random_raw(8192)
    random_words[((random_words >> np.uint64(52)) & np.uint64(0x7ff)) == 0x7ff] = 0
    random_values = random_words.view(np.float64)
    check("seeded_finite_bitpatterns", random_values, (127, 65536))
    check("negative_stride", random_values[511::-3], (1, 67))
    crowded = np.full(MAX_CHUNK+3, np.nextafter(2., 0.))
    expected = oracle(crowded[:1])*len(crowded)
    assert squared_integer(crowded, chunk_size=MAX_CHUNK) == expected
    cases.append({"name": "maximum_chunk_carry_and_split", "coordinates": len(crowded), "chunks": [MAX_CHUNK]})

    exact = squared_integer(np.array([3., 4.]))
    assert encloses_norm(exact, 5.)
    assert not encloses_norm(exact, math.nextafter(5., 0.))
    assert encloses_norm(exact, 5)
    tiny = squared_integer(np.array([math.ulp(0.)]))
    assert tiny == 1 and encloses_norm(tiny, math.ulp(0.)) and not encloses_norm(tiny, 0.)
    assert encloses_norm(0, 0.) and norm_upper(np.array([0.])) == 0.
    for values in (np.array([3., 4.]), np.array([math.ulp(0.)]), rng.normal(size=1024)):
        assert encloses_norm(oracle(values), norm_upper(values))
    assert squared_integer(random_values) == squared_integer(random_values[::-1])

    def reject(name, call):
        try:
            call()
        except ValueError:
            refused.append(name)
        else:
            raise AssertionError("expected refusal: "+name)
    for name, values in (("nan", np.array([float("nan")])), ("infinity", np.array([float("inf")])),
                         ("float32", np.array([1.], dtype=np.float32)), ("list", [1.]),
                         ("matrix", np.array([[1.]]))):
        reject(name, lambda: squared_integer(values))
    for chunk in (0, -1, True, MAX_CHUNK+1):
        reject("chunk_"+str(chunk), lambda: squared_integer(np.array([1.]), chunk_size=chunk))
    for bound in (-1., float("inf"), float("nan"), True):
        reject("bound_"+str(bound), lambda: encloses_norm(1, bound))
    reject("overflowing_norm", lambda: norm_upper(np.array([np.finfo(float).max]*2)))
    reject("negative_square", lambda: encloses_norm(-1, 1.))
    print(json.dumps({"status": "PASS", "oracle_cases": cases, "refusals": refused,
        "exact_rational_oracle_agreement": True, "one_ulp_inward_bound_refused": True,
        "subnormal_square_retained": True, "synthetic_only": True,
        "neural_kernels_replayed": False, "event_certificate_issued": False}, indent=2))


if __name__ == "__main__":
    main()
