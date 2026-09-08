"""Exact component round-trip checks, including tiny/wide dyadic intervals."""
import random
import unittest

import numpy as np
from flint import arb, ctx

from final_scale_arb_codec_v1 import decode_vector, encode_vector, layout


def noop():
    pass


class CodecTests(unittest.TestCase):
    def test_exact_components_at_all_registered_precisions(self):
        rng = random.Random(8128)
        old = ctx.prec
        checked = 0
        try:
            for bits in (64, 128, 192, 256):
                ctx.prec = bits
                values = [arb(0), arb(1), arb(-1), arb(2).sqrt(), arb(1)/3,
                          arb((1, -1074)), arb((-1, -2000), (1, -2100)), arb((1, 2000), (1, 1990))]
                for _ in range(100):
                    sign = rng.choice((-1, 1))
                    midpoint = sign*(rng.getrandbits(bits-1) | 1)
                    radius = rng.getrandbits(28) | 1
                    values.append(arb((midpoint, rng.randrange(-2000, 2001)),
                                      (radius, rng.randrange(-2200, 1901))))
                data = encode_vector(values, precision_bits=bits, guard=noop)
                rebuilt = decode_vector(data, precision_bits=bits, guard=noop)
                for a, b in zip(values, rebuilt):
                    self.assertEqual(a.mid().man_exp(), b.mid().man_exp())
                    self.assertEqual(a.rad().man_exp(), b.rad().man_exp())
                    self.assertTrue(a.contains(b) and b.contains(a))
                    checked += 1
                self.assertTrue(np.array_equal(data, encode_vector(rebuilt, precision_bits=bits, guard=noop)))
            self.assertEqual(checked, 432)
        finally:
            ctx.prec = old

    def test_unsigned_high_limbs_roundtrip(self):
        old = ctx.prec
        ctx.prec = 128
        try:
            values = [arb((2**128-1, -127)), arb((-(2**128-1), -127))]
            data = encode_vector(values, precision_bits=128, guard=noop)
            self.assertTrue((data[:, 4:] == -1).all())
            rebuilt = decode_vector(data, precision_bits=128, guard=noop)
            self.assertTrue(all(a == b for a, b in zip(values, rebuilt)))
        finally:
            ctx.prec = old

    def test_full_radius_mantissas_and_rounding_boundaries(self):
        old = ctx.prec
        ctx.prec = 192
        try:
            rng = random.Random(613)
            radii = [2**29+1, 2**29+12345, 2**30-1]
            radii += [(rng.getrandbits(29) | 2**29 | 1) for _ in range(300)]
            for exponent in (-2000, -220, 0, 2000):
                data = np.zeros((len(radii), 7), dtype='<i8')
                data[:, 0], data[:, 4] = -1, 12345
                data[:, 2], data[:, 3] = radii, exponent
                values = decode_vector(data, precision_bits=192, guard=noop)
                self.assertTrue(np.array_equal(data, encode_vector(values, precision_bits=192, guard=noop)))
                self.assertTrue(all(tuple(map(int, v.rad().man_exp())) == (r, exponent)
                                    for v, r in zip(values, radii)))
        finally:
            ctx.prec = old

    def test_low_precision_load_and_oversized_midpoint_refused(self):
        old = ctx.prec
        ctx.prec = 192
        try:
            values = [arb(2).sqrt()]
            with self.assertRaisesRegex(ValueError, "midpoint exceeds"):
                encode_vector(values, precision_bits=128, guard=noop)
            data = encode_vector(values, precision_bits=192, guard=noop)
            ctx.prec = 128
            with self.assertRaisesRegex(ValueError, "below registered"):
                decode_vector(data, precision_bits=192, guard=noop)
        finally:
            ctx.prec = old

    def test_noncanonical_components_refused(self):
        old = ctx.prec
        ctx.prec = 128
        try:
            good = encode_vector([arb((1, 0), (1, -10))], precision_bits=128, guard=noop)
            for column, bad in ((0, 2), (0, 0), (2, -1), (2, 2), (4, 2)):
                changed = good.copy()
                changed[0, column] = bad
                with self.assertRaises(ValueError):
                    decode_vector(changed, precision_bits=128, guard=noop)
        finally:
            ctx.prec = old

    def test_invalid_types_shapes_and_nonfinite_values_refused(self):
        for values in ([0.1], [arb("nan")], [arb("inf")], []):
            with self.assertRaises(ValueError):
                encode_vector(values, precision_bits=128, guard=noop)
        for bad in (np.zeros((1, 6)), np.zeros((1, 5), dtype=np.int64), np.zeros((0, 6), dtype=np.int64)):
            with self.assertRaises(ValueError):
                decode_vector(bad, precision_bits=128, guard=noop)
        with self.assertRaises(ValueError):
            layout(True)

    def test_guard_failure_propagates_without_changing_precision(self):
        old = ctx.prec
        def fail():
            raise OSError("injected guard failure")
        with self.assertRaises(OSError):
            encode_vector([arb(1)], precision_bits=128, guard=fail)
        self.assertEqual(ctx.prec, old)


if __name__ == "__main__":
    unittest.main(verbosity=2)
