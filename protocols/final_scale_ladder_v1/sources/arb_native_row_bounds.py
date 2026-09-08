"""Native Arb row reduction, with the same PSD row-sum inequality.

All entries retain their ball radii through abs and the matrix-vector sum.
Only the final n row endpoints are extracted, instead of n**2 individual
endpoints. This changes arithmetic organization, not the norm inequality.
"""
from flint import arb, arb_mat


def native_absolute_row_sum_upper(matrix):
    if not isinstance(matrix, arb_mat) or min(matrix.nrows(), matrix.ncols()) < 1:
        raise ValueError("nonempty Arb matrix required")
    absolute = arb_mat(matrix.nrows(), matrix.ncols(), [abs(v) for v in matrix.entries()])
    rows = absolute * arb_mat(matrix.ncols(), 1, [1] * matrix.ncols())
    upper = arb(0)
    for value in rows.entries():
        if not value.is_finite():
            raise ValueError("nonfinite absolute row enclosure")
        upper = max(upper, value.upper())
    return upper


def psd_native_row_gain_upper(gram):
    """Caller supplies an enclosure of a true symmetric PSD Gram."""
    if not isinstance(gram, arb_mat) or gram.nrows() != gram.ncols() or gram.nrows() < 1:
        raise ValueError("nonempty square Arb Gram required")
    return native_absolute_row_sum_upper(gram).sqrt().upper()
