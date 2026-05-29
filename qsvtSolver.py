import numpy as np
from qiskit.quantum_info import Statevector
from qsvt.algorithms import linear_solver


def complex2realMatrix(A):
    """Embed a complex matrix A into a real block matrix.

    Parameters
    ----------
    A : np.ndarray
        Complex coefficient matrix from the discretized GLM system.

    Returns
    -------
    np.ndarray
        Real block matrix [[Re A, -Im A], [Im A, Re A]].
    """
    A_real = np.real(A)
    A_imag = np.imag(A)
    return np.vstack([np.hstack([A_real, -A_imag]), np.hstack([A_imag, A_real])])


def complex2realVector(b):
    """Embed a complex vector b into a real vector.

    Parameters
    ----------
    b : np.ndarray
        Complex right-hand side vector.

    Returns
    -------
    np.ndarray
        Concatenated vector [Re b, Im b].
    """
    return np.concatenate([np.real(b), np.imag(b)])


def _inverse_scale(degree):
    """Return the scale factor of the precomputed QSVT inverse polynomial."""
    if degree == 2:
        return 1.666666e-02
    if degree == 3:
        return 3.333333e-03
    if degree == 4:
        return 2.5e-3
    if degree == 5:
        return 5e-4
    raise ValueError("degree must be one of 2, 3, 4, 5")


def qsvtSolverAbs(A, b, degree=3):
    """Solve A x = b with QSVT after complex-to-real conversion.

    Parameters
    ----------
    A : np.ndarray
        Complex GLM coefficient matrix.
    b : np.ndarray
        Complex GLM right-hand side.
    degree : int
        Precomputed inverse-polynomial choice used by qsvt.algorithms.linear_solver.

    Returns
    -------
    np.ndarray
        Approximate complex solution x.
    """
    c_scale = _inverse_scale(degree)
    n_qubits = int(np.log2(len(b)))

    A_real = complex2realMatrix(A)
    b_real = complex2realVector(b)

    alpha = np.linalg.norm(A_real)
    beta = np.linalg.norm(b_real)
    A_norm = A_real / alpha
    b_norm = b_real / beta

    qc = linear_solver(A_norm, rhs=b_norm, set_degree=degree)
    state = Statevector(qc)

    solution_size = 2 ** (n_qubits + 1)
    raw_segment = state.data[:solution_size]

    half_len = 2**n_qubits
    real_part = raw_segment[:half_len]
    imag_part = raw_segment[half_len:]

    x_reconstructed = real_part + 1j * imag_part
    return x_reconstructed * (beta / alpha) / c_scale


def qsvtSolverComplex(A, b, degree=3):
    """Solve a complex GLM linear system with the QSVT solver directly.

    Parameters
    ----------
    A : np.ndarray
        Complex GLM coefficient matrix.
    b : np.ndarray
        Complex GLM right-hand side.
    degree : int
        Precomputed inverse-polynomial choice used by qsvt.algorithms.linear_solver.

    Returns
    -------
    np.ndarray
        Approximate complex solution x.
    """
    c_scale = _inverse_scale(degree)

    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)
    if A.shape[0] != A.shape[1] or A.shape[0] != len(b):
        raise ValueError("A must be square and compatible with b")
    if A.shape[0] & (A.shape[0] - 1):
        raise ValueError("QSVT solver requires a power-of-two system dimension")

    alpha = np.linalg.norm(A)
    beta = np.linalg.norm(b)
    if beta == 0:
        return np.zeros_like(b)

    qc = linear_solver(A / alpha, rhs=b / beta, set_degree=degree, real_only=True)
    state = Statevector(qc)

    raw_segment = state.data[: len(b)]
    return -raw_segment * (beta / alpha) / c_scale


def qsvtSolverComplexSVDPreconditioned(A, b, degree=2, tau_rel=0.05, return_info=False):
    """Solve A x = b with direct-complex QSVT after SVD preconditioning.

    Parameters
    ----------
    A : np.ndarray
        Complex GLM coefficient matrix.
    b : np.ndarray
        Complex right-hand side.
    degree : int
        Inverse-polynomial degree preset.
    tau_rel : float
        Relative singular-value floor used in the right preconditioner.
    return_info : bool
        If True, also return conditioning diagnostics.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, dict]
        Approximate solution x, optionally with diagnostic information.
    """
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)
    if A.shape[0] != A.shape[1] or A.shape[0] != len(b):
        raise ValueError("A must be square and compatible with b")
    if tau_rel <= 0:
        raise ValueError("tau_rel must be positive")

    _, s, Vh = np.linalg.svd(A, full_matrices=False)
    tau = tau_rel * np.linalg.norm(A)
    factors = np.ones_like(s)
    lift_mask = s < tau
    factors[lift_mask] = tau / s[lift_mask]

    V = Vh.conj().T
    preconditioner = V @ np.diag(factors) @ Vh
    preconditioned_matrix = A @ preconditioner

    y = qsvtSolverComplex(preconditioned_matrix, b, degree=degree)
    x = preconditioner @ y

    if return_info:
        info = {
            "tau": tau,
            "tau_rel": tau_rel,
            "num_lifted": int(np.count_nonzero(lift_mask)),
            "cond_original": float(np.linalg.cond(A)),
            "cond_preconditioned": float(np.linalg.cond(preconditioned_matrix)),
            "min_singular_original": float(np.min(s)),
            "max_singular_original": float(np.max(s)),
        }
        return x, info
    return x
