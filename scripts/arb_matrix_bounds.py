"""Outward matrix norms, including residual-checked numerical SVD factors.

The floating-point SVD proposes factors only. Arb encloses both their
orthogonality defects and the complete reconstruction residual. No claim
relies on the SVD implementation returning an exact factorization.
"""
import math
import numpy as np
import torch
from flint import arb,arb_mat


def as_matrix(value):
    if isinstance(value,arb_mat):
        return value
    values=np.asarray(value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else value,dtype=np.float64)
    if values.ndim!=2 or min(values.shape)<1 or not np.isfinite(values).all():
        raise ValueError("nonempty finite matrix required")
    return arb_mat(*values.shape,[arb(float(v)) for v in values.reshape(-1)])


def upper_float(value):
    if not value.is_finite():raise ValueError("nonfinite matrix bound")
    return math.nextafter(float(value.upper()),math.inf)


def frobenius_upper(matrix):
    return sum((abs(v).upper()**2 for v in matrix.entries()),arb(0)).sqrt().upper()


def absolute_row_sum_upper(matrix):
    rows=[sum((abs(matrix[i,j]).upper() for j in range(matrix.ncols())),arb(0)).upper()
          for i in range(matrix.nrows())]
    return max(rows)


def induced_bound_upper(matrix):
    return (absolute_row_sum_upper(matrix)*absolute_row_sum_upper(matrix.transpose())).sqrt().upper()


def psd_gain_upper(gram):
    """sqrt(row-sum bound), when the exact enclosed Gram is symmetric PSD."""
    if gram.nrows()!=gram.ncols():raise ValueError("square Gram required")
    return absolute_row_sum_upper(gram).sqrt().upper()


def spectral_from_factors(matrix,u,s,vt):
    """Any finite factors work; errors only enlarge the resulting enclosure."""
    matrix,u,vt=map(as_matrix,(matrix,u,vt))
    values=np.asarray(s,dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():raise ValueError("invalid diagonal")
    r=len(values)
    if (u.nrows(),u.ncols(),vt.nrows(),vt.ncols())!=(matrix.nrows(),r,r,matrix.ncols()):
        raise ValueError("factor shapes mismatch")
    sigma=arb_mat(r,r)
    for i,v in enumerate(values):sigma[i,i]=arb(float(v))
    reconstruction=u*sigma*vt
    residual=matrix-reconstruction
    u_gain=psd_gain_upper(u.transpose()*u)
    v_gain=psd_gain_upper(vt*vt.transpose())
    remainder=frobenius_upper(residual)
    factor_upper=u_gain*arb(float(abs(values).max()))*v_gain+remainder
    bound=min(factor_upper.upper(),frobenius_upper(matrix),induced_bound_upper(matrix))
    return {"upper":upper_float(bound),"svd_residual_frobenius_upper":upper_float(remainder),
            "left_factor_norm_upper":upper_float(u_gain),"right_factor_norm_upper":upper_float(v_gain),
            "numerical_svd_norm":float(abs(values).max()),
            "uses_verified_reconstruction_residual":True}


def spectral_upper(matrix):
    enclosed=as_matrix(matrix)
    midpoint=np.array([float(v.mid()) for v in enclosed.entries()]).reshape(enclosed.nrows(),enclosed.ncols())
    if not np.isfinite(midpoint).all():raise ValueError("nonfinite midpoint")
    u,s,vt=torch.linalg.svd(torch.tensor(midpoint,dtype=torch.float64),full_matrices=False)
    return spectral_from_factors(enclosed,u.numpy(),s.numpy(),vt.numpy())
