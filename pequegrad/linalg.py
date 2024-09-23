from pequegrad.tensor import Tensor
from pequegrad.ops import assign_at, fill, outer_prod


def lu_factorization(A: Tensor):
    assert A.ndim == 2
    n, m = A.shape
    assert n == m
    _A = A
    L = fill(A.shape, A.dtype, 0, A.device)
    for i in range(n):
        L = assign_at(L, fill(tuple(), A.dtype, 1, A.device), (i, i))
    for j in range(n):  # current col
        pivot = _A[j, j]
        for i in range(j + 1, n):  # current row
            el = _A[i, j]
            factor = el / pivot
            rownew = _A[i] - _A[j] * factor
            _A = assign_at(_A, rownew, i)
            L = assign_at(L, factor, (i, j))
    return L, _A


def lu_factorization_faster(A: Tensor):
    assert A.ndim == 2
    n, m = A.shape
    assert n == m
    U = A
    L = fill(A.shape, A.dtype, 0, A.device)
    for i in range(n):
        L = assign_at(L, fill(tuple(), A.dtype, 1, A.device), (i, i))
    for j in range(n - 1):  # current col
        pivot = U[j, j]
        factor = U[j + 1 :, j] / pivot  # vector representing updates for the column
        L = assign_at(L, factor, (slice(j + 1, A.shape[0]), j))
        U = assign_at(
            U,
            U[j + 1 :, :] - outer_prod(factor, U[j, :]),
            (slice(j + 1, A.shape[0]), slice(0, A.shape[1])),
        )
    return L, U


lu_factorization = lu_factorization_faster
