"""Outward PSD gain bounds from native matrix dot products.

The caller supplies a ball matrix containing an exact symmetric PSD Gram G.
PSD is an input contract, not inferred from the interval midpoints. The
returned number bounds sqrt(lambda_max(G)), i.e. the associated Jacobian gain.

These are classical Frobenius/trace and Wolkowicz--Styan moment bounds, not
new spectral theory. All acceptance quantities use Arb, never a float SVD.
"""
from flint import arb, arb_mat


def _nonnegative_upper(value):
    if not value.is_finite():
        raise ValueError("nonfinite moment enclosure")
    return max(arb(0), value.upper())


def frobenius_squared_enclosure(matrix):
    """Native dot includes every entry and every supplied entry radius."""
    if not isinstance(matrix, arb_mat) or min(matrix.nrows(), matrix.ncols()) < 1:
        raise ValueError("nonempty Arb matrix required")
    flat = arb_mat(1, matrix.nrows() * matrix.ncols(), matrix.entries())
    result = (flat * flat.transpose())[0, 0]
    if not result.is_finite():
        raise ValueError("nonfinite Frobenius enclosure")
    return result


def psd_moment_gain_upper(gram):
    """sqrt(min(trace, Frobenius, mean + sqrt((n-1) variance))).

    For exact PSD G, sum(G_ij**2) = tr(G**2). Interval dependencies can only
    enlarge the resulting upper bounds. In particular a slightly negative
    variance lower endpoint is harmless; the nonnegative upper endpoint is
    used before taking square roots.
    """
    if not isinstance(gram, arb_mat) or gram.nrows() != gram.ncols() or gram.nrows() < 1:
        raise ValueError("nonempty square Arb Gram required")
    n = gram.nrows()
    trace = gram.trace()
    if not trace.is_finite() or bool(trace.upper() < 0):
        raise ValueError("trace incompatible with a finite PSD Gram")
    trace_upper = _nonnegative_upper(trace)
    if n == 1:
        return trace_upper.sqrt().upper()
    squared = frobenius_squared_enclosure(gram)
    frobenius = _nonnegative_upper(squared).sqrt().upper()
    centered_squared = _nonnegative_upper(squared - trace * trace / n)
    moment = (trace / n + (centered_squared * (n - 1) / n).sqrt()).upper()
    return min(trace_upper, frobenius, _nonnegative_upper(moment)).sqrt().upper()


def psd_power_moment_gain_upper(gram, *, squarings=1):
    """Apply the same bound to G**(2**squarings), then undo the power.

    This potentially tighter variant costs native square matrix products.
    It is a benchmark option, not an unconditional faster replacement.
    """
    if type(squarings) is not int or not 0 <= squarings <= 3:
        raise ValueError("squarings must be an integer from zero through three")
    if not isinstance(gram, arb_mat) or gram.nrows() != gram.ncols() or gram.nrows() < 1:
        raise ValueError("nonempty square Arb Gram required")
    powered = gram
    for _ in range(squarings):
        powered = powered * powered
    gain = psd_moment_gain_upper(powered)
    for _ in range(squarings):
        gain = gain.sqrt().upper()
    return gain
