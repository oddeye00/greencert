"""Exact-rational regression of both packaged dot-product backends.

Tests the public artifact's code, not a private checkout copy. Includes the
underflow family that motivated the repair. This is a primitive regression,
not a neural event certificate or validation of an arbitrary BLAS binary.
"""
from fractions import Fraction
import importlib
import json
from pathlib import Path
import sys
import tempfile

from read_public_repair_archive import payload


def check(module, al, au, bl, bu):
    lo, hi = module.interval_matmul(al, au, bl, bu)
    count = 0
    for i in range(al.shape[0]):
        for j in range(bl.shape[1]):
            lower = upper = Fraction(0)
            for k in range(al.shape[1]):
                products = [Fraction(float(x))*Fraction(float(y))
                            for x in (al[i, k], au[i, k]) for y in (bl[k, j], bu[k, j])]
                lower += min(products)
                upper += max(products)
            if not Fraction(float(lo[i, j])) <= lower <= upper <= Fraction(float(hi[i, j])):
                raise AssertionError("exact interval product escaped packaged bound")
            count += 1
    return count


def main():
    import numpy as np
    _, files = payload()
    with tempfile.TemporaryDirectory(prefix="greencert-public-roundoff-") as temporary:
        root = Path(temporary)
        names = ("binary64_interval_products", "point_exact_interval_products")
        for name in names:
            with (root/(name+".py")).open("xb") as handle:
                handle.write(files["scripts/"+name+".py"])
        sys.path.insert(0, str(root))
        modules = [importlib.import_module(name) for name in names]
        if any(Path(module.__file__).resolve().parent != root.resolve() for module in modules):
            raise AssertionError("test did not import the packaged backend")
        modules[0].check_runtime_conditions()
        results = []
        for module in modules:
            entries = boxes = 0
            rng = np.random.default_rng(9072026)
            for _ in range(80):
                m, n, p = (int(v) for v in rng.integers(1, 5, size=3))
                a = rng.integers(-64, 65, size=(m, n)).astype(np.float64)/16
                b = rng.integers(-64, 65, size=(n, p)).astype(np.float64)/16
                ar = rng.integers(0, 5, size=a.shape).astype(np.float64)/64
                br = rng.integers(0, 5, size=b.shape).astype(np.float64)/64
                entries += check(module, a-ar, a+ar, b-br, b+br)
                boxes += 1
            for n in (1, 2, 4, 8, 16, 32, 64, 128):
                a = np.full((1, n), 2.**-537, dtype=np.float64)
                b = np.full((n, 1), 2.**-538, dtype=np.float64)
                entries += check(module, a, a, b, b)
                # Signs/cancellation must retain an interval rather than silently
                # clamp a small negative endpoint to zero.
                entries += check(module, -a, a, -b, b)
            refused = 0
            for value in (np.nan, np.inf, -np.inf):
                a = np.array([[value]])
                b = np.ones((1, 1))
                try:
                    module.interval_matmul(a, a, b, b)
                except (ValueError, OverflowError, FloatingPointError):
                    refused += 1
                else:
                    raise AssertionError("nonfinite primitive input accepted")
            results.append({"backend": module.__name__, "interval_boxes": boxes,
                            "underflow_lengths": 8, "signed_underflow_boxes": 8,
                            "exact_output_enclosures": entries, "nonfinite_refusals": refused})
    print(json.dumps({"status": "PASS", "packaged_backends_only": True, "results": results,
                      "neural_certificates_issued": 0}, indent=2))


if __name__ == "__main__":
    main()
