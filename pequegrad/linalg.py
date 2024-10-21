from pequegrad.tensor import Tensor
from pequegrad.ops import assign_at, fill, outer_prod, triu, tril, cat, eye, diag_vector
from typing import Tuple
import functools


def assert_rank(a: Tensor, rank: int, name: str):
    assert (
        a.ndim == rank
    ), f"Expected a tensor of rank {rank} for arg '{name}', got {a.ndim}"


# ======================= LU factorization =======================
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


# ======================= Solve linear system =======================


def solve(A: Tensor, b: Tensor):
    """
    Solves the linear system Ax = b for x.
    """

    assert_rank(A, 2, "A")
    assert_rank(b, 1, "b")
    # Ax = b -> LUx = b
    L, U = lu_factorization(A)

    # LUx = b -> Ly = b, solve for y
    y = fill(b.shape, b.dtype, 0, b.device)
    for i in range(A.shape[0]):
        y = y.at[i].set(b[i] - L[i, :i] @ y[:i])

    # Ux = y, solve for x
    x = fill(b.shape, b.dtype, 0, b.device)
    for i in range(A.shape[0] - 1, -1, -1):
        if i == A.shape[0] - 1:
            x = x.at[i].set(y[i] / U[i, i])
        else:
            x = x.at[i].set((y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i])
    return x


# ======================= Projection ========================
def project(v: Tensor, onto: Tensor):
    assert_rank(v, 1, "v")
    assert_rank(onto, 1, "onto")
    return ((v @ onto) / (onto @ onto)) * onto


# ======================= Gram-Schmidt  =======================


def gram_schmidt(vectors: Tuple[Tensor, ...]):
    map(lambda i, v: assert_rank(v, 1, f"vectors[{i}]"), enumerate(vectors))
    dim, dtype = vectors[0].shape[0], vectors[0].dtype
    assert len(vectors) == dim
    res = Tensor.zeros((dim, dim), dtype=dtype)
    others = []
    for i, v in enumerate(vectors):
        v_orig = v
        for o in others:
            v = v - project(v_orig, o)
        res = res.at[:, i].set(v / (v @ v) ** 0.5)
        others.append(v)
    return res


# ======================= QR factorization =======================


def qr_factorization(A: Tensor):
    m = A.shape[0]
    assert A.shape[1] == m, "QR factorization only implemented for square matrices"
    Q = gram_schmidt([A[:, i] for i in range(m)])
    mask = triu(Tensor.ones((m, m), dtype=A.dtype, device=A.device), diagonal=0)
    R = (Q.T @ A) * mask
    return Q, R


# ======================= Determinant =======================


# determinant is the product of the diagonal elements of the upper triangular matrix in the LU factorization
def det(A: Tensor) -> Tensor:
    if A.shape[0] != A.shape[1]:
        raise ValueError("Determinant is only defined for square matrices")
    L, U = lu_factorization(A)
    return functools.reduce(lambda x, y: x * y, [U[i, i] for i in range(U.shape[0])])


determinant = det


# ======================== Matrix Power ========================


def matrix_power(A: Tensor, n: int, method="eigen") -> Tensor:
    if method == "recursive":
        if n == 0:
            # eye
            ones = fill(A.shape, A.dtype, 1, A.device)
            return triu(tril(ones), diagonal=0)
        if n % 2 == 0:
            return matrix_power(A @ A, n // 2)
        else:
            return A @ matrix_power(A, n - 1)
    elif method == "eigen":
        eigvecs, eigvals = eig(A)
        eigvals__n = eigvals ** n
        return eigvecs @ diag_vector(eigvals__n) @ eigvecs.T


# ======================= Eigenvalues =======================


# use qr algorithm to find eigenvalues
def eig(A: Tensor, n_iter: int = 100) -> Tensor:
    assert_rank(A, 2, "A")
    eigvecs = eye(A.shape[0], A.shape[0], A.dtype, A.device)
    for _ in range(n_iter):
        Q, R = qr_factorization(A)
        A = R @ Q
        eigvecs = eigvecs @ Q
    eigvals = [A[i, i] for i in range(A.shape[0])]
    return eigvecs, cat([ei.unsqueeze(0) for ei in eigvals], dim=0)
