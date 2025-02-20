from pequegrad.tensor import Tensor
from pequegrad.ops import (
    assign_at,
    fill,
    outer_prod,
    triu,
    tril,
    cat,
    eye,
    diag_vector,
    exp,
)
from typing import Tuple
import functools
import pequegrad.ops as ops


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


def solve_lu(A: Tensor, b: Tensor):
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


def solve_jacobi(A: Tensor, b: Tensor, n_iter: int = 100):
    """
    Solves the linear system Ax = b for x using the Jacobi method.
    Assumes A, D and E are non-singular.
    """
    assert_rank(A, 2, "A")
    assert_rank(b, 1, "b")
    n = A.shape[0]
    x = fill(b.shape, b.dtype, 0, b.device)
    D, E = ops.extract_diag(A), ops.extract_non_diag(A)
    # Ax = b -> (D + E)x = b -> Dx = b - Ex -> x = D^-1(b - Ex)
    D_inv = 1 / D  # element-wise inverse since D is diagonal
    for _ in range(n_iter):
        x = D_inv * (b - E @ x)

    return x


def solve(A: Tensor, b: Tensor, method="lu", **kwargs):
    if method == "lu":
        return solve_lu(A, b)
    elif method == "jacobi":
        return solve_jacobi(A, b, **kwargs)
    else:
        raise ValueError("Invalid method")


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
        norm = (v @ v) ** 0.5
        res = res.at[:, i].set(v / norm)
        others.append(v)
    return res


# ======================= QR factorization =======================


# householder qr factorization
def qr_factorization_householder(A: Tensor):
    # adapted from https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
    assert A.ndim == 2
    m, n = A.shape
    Q = eye(m, m, A.dtype, A.device)
    R = A
    minus1 = fill(tuple(), A.dtype, -1, A.device)
    one = fill(tuple(), A.dtype, 1, A.device)
    for i in range(n):
        normx = ops.norm(R[i:, i])
        s = ops.where(R[i, i] > 0, minus1, one)
        u1 = R[i, i] - s * normx
        w = R[i:, i] / u1
        w = assign_at(w, fill(tuple(), A.dtype, 1, A.device), 0)
        tau = -s * u1 / normx
        R = R.at[i:, :].set(
            R[i:, :] - (tau * w).unsqueeze(0).T @ (w.unsqueeze(0) @ R[i:, :])
        )
        Q = Q.at[:, i:].set(Q[:, i:] - (Q[:, i:] @ w.unsqueeze(1)) * tau * w)
    return Q, R


def qr_factorization_householder2(A: Tensor):
    def sgn(x):
        return ops.where(
            x > 0,
            fill(tuple(), x.dtype, 1, x.device),
            fill(tuple(), x.dtype, -1, x.device),
        )

    def householder_reflection(x):
        norm_x = ops.norm(x)
        if norm_x == 0:
            return eye(len(x), len(x), x.dtype, x.device)
        u = x.at[0].set(x[0] + sgn(x[0]) * norm_x)
        u = u / ops.norm(u)
        H = ops.eye(len(x), len(x), x.dtype, x.device) - 2 * ops.outer_prod(u, u)
        return H

    assert A.ndim == 2
    m, n = A.shape
    Q = eye(m, m, A.dtype, A.device)  # Initialize Q as an identity matrix
    R = A
    for k in range(n):
        x = R[k:m, k]
        H_k = householder_reflection(x)
        R = R.at[k:m, :].set(H_k @ R[k:m, :])
        Q = Q.at[:, k:m].set(Q[:, k:m] @ H_k.T)
    return Q, R


def qr_factorization_graham_schmidt(A: Tensor):
    assert A.ndim == 2
    m, n = A.shape
    Q = gram_schmidt([A[:, i] for i in range(n)])
    R = Q.T @ A
    return Q, R


def qr_factorization(A: Tensor, method="graham_schmidt"):
    if method == "householder":
        return qr_factorization_householder2(A)
    elif method == "graham_schmidt":
        return qr_factorization_graham_schmidt(A)
    elif method == "householder2":
        return qr_factorization_householder2(A)
    else:
        raise ValueError("Invalid method")


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
        eigvals__n = eigvals**n
        return eigvecs @ diag_vector(eigvals__n) @ eigvecs.T


# ======================= Eigenvalues =======================


# use qr algorithm to find eigenvalues
def eig(A: Tensor, n_iter: int = 100, qr_method="graham_smidtch") -> Tensor:
    assert_rank(A, 2, "A")
    eigvecs = eye(A.shape[0], A.shape[0], A.dtype, A.device)
    for _ in range(n_iter):
        Q, R = qr_factorization(A, method=qr_method)
        A = R @ Q
        eigvecs = eigvecs @ Q
    eigvals = [A[i, i] for i in range(A.shape[0])]
    return eigvecs, cat([ei.unsqueeze(0) for ei in eigvals], dim=0)


# ======================= Matrix Exponential =======================


def matrix_exp(A: Tensor, diagonalizable=True) -> Tensor:
    # computes exp(A) where A is a square matrix
    if not diagonalizable:
        raise NotImplementedError("Only implemented for diagonalizable matrices")
    eigvecs, eigvals = eig(A)
    eigvals_exp = exp(eigvals)
    return eigvecs @ diag_vector(eigvals_exp) @ eigvecs.T


# ======================= Matrix Inverse =======================


def inv(A: Tensor) -> Tensor:
    # computes the inverse of a square matrix
    assert_rank(A, 2, "A")
    n = A.shape[0]
    assert n == A.shape[1], "Matrix must be square"
    I = eye(n, n, A.dtype, A.device)
    cols = [I[:, i] for i in range(n)]
    inv_cols = [solve(A, col) for col in cols]
    return cat([col.unsqueeze(1) for col in inv_cols], dim=1)


# ======================= Cholesky Decomposition =======================
def cholesky(A: Tensor) -> Tensor:
    assert_rank(A, 2, "A")
    n = A.shape[0]
    assert n == A.shape[1], "Matrix must be square"
    # assumes A is symmetric and positive definite
    # lambda greek char =
    # then, the factorization A = X @ Λ @ X^-1 = X @ Λ @ X.T = X @ (sqrt(Λ) @ sqrt(Λ)) @ X.T = (X @ sqrt(Λ)).T @ (X @ sqrt(Λ)) = C.T @ C
    x, Λ = eig(A)
    return x @ diag_vector(Λ**0.5)


# ======================= SVD =======================


def svd(A):
    # A is (m, n)
    m, n = A.shape

    if m > n:
        ATA = A.T @ A  # (n, n)
        V, eigvals_v = eig(ATA, n_iter=100)  # (n, n), (n,)
        singular_values = ops.where(
            eigvals_v > 1e-4, eigvals_v**0.5, fill((n,), A.dtype, 0, A.device)
        )  # (n,)
        S = diag_vector(singular_values)  # (n, n)
        # m > n, so we pad the singular values with zeros on the bottom
        S = cat([S, fill((m - n, n), A.dtype, 0, A.device)], dim=0)
        U = (
            A
            @ V
            @ cat(
                [
                    diag_vector(1 / singular_values),
                    fill((n, m - n), A.dtype, 0, A.device),
                ],
                dim=1,
            )
        )
        # U will have 0s on the right, so we use QR to return an orthonormal matrix
        return U, S, V.T
    elif m < n:
        # we decompose A.T instead of A, then transpose the results
        A = A.T
        m, n = A.shape
        ATA = A.T @ A  # (n, n)
        V, eigvals_v = eig(ATA, n_iter=100)  # (n, n), (n,)
        singular_values = ops.where(
            eigvals_v > 0, eigvals_v**0.5, fill((n,), A.dtype, 0, A.device)
        )  # (n,)
        S = diag_vector(singular_values)  # (n, n)
        # m > n, so we pad the singular values with zeros on the bottom
        S = cat([S, fill((m - n, n), A.dtype, 0, A.device)], dim=0)
        U = (
            A
            @ V
            @ cat(
                [
                    diag_vector(1 / singular_values),
                    fill((n, m - n), A.dtype, 0, A.device),
                ],
                dim=1,
            )
        )
        # A.T = U @ S @ V.T -> A = V @ S.T @ U.T
        # U will have 0s on the right, so we use QR to return an orthonormal matrix
        return V, S.T, U.T
    else:
        # A is square
        ATA = A.T @ A
        V, eigvals_v = eig(ATA, n_iter=100)
        singular_values = ops.where(
            eigvals_v > 0, eigvals_v**0.5, fill((n,), A.dtype, 0, A.device)
        )
        S = diag_vector(singular_values)
        U = A @ V @ diag_vector(1 / singular_values)
        return U, S, V.T
