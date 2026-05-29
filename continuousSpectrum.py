import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import block_diag, expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from tqdm.auto import tqdm


def getHamiltonian(x_idx, uList, vList, lam):
    """Return the local Z-S Hamiltonian H(x_i, lambda).

    The paper rewrites the spatial Z-S equation as i d_x psi = H psi.

    Parameters
    ----------
    x_idx : int
        Index i of the spatial grid point x_i.
    uList : np.ndarray
        Samples of q(x,0).
    vList : np.ndarray
        Samples of q*(x,0), usually np.conj(uList).
    lam : complex
        Spectral parameter lambda.

    Returns
    -------
    np.ndarray
        The 2 by 2 Hamiltonian matrix at x_i.
    """
    u = uList[x_idx]
    v = vList[x_idx]
    return np.array([[lam, 1j * u], [-1j * v, -lam]])


def _compute_single_lambda(lam, xList, uList, vList, h, l):
    """Compute scattering data for one real spectral value lambda.

    Returns transmission T(lambda) and left/right reflection coefficients
    L(lambda), R(lambda), obtained from the asymptotic coefficients a(lambda)
    and b(lambda).
    """
    qc = QuantumCircuit(1)
    qc.initialize([np.exp(-1j * lam * l), 0], 0)

    for i in range(len(xList)):
        hamiltonian = getHamiltonian(i, uList, vList, lam)
        unitary = expm(1j * hamiltonian * h)
        qc.append(Operator(unitary), [0])

    final_state = Statevector(qc)

    a = np.conj(final_state.data[0] / np.exp(1j * lam * l))
    b = np.conj(final_state.data[1] / np.exp(-1j * lam * l))

    left_reflection = -np.conj(b) / a
    right_reflection = -b / a
    transmission = 1 / a

    return transmission, left_reflection, right_reflection


def getReflectionQuantum(xList, uList, vList, lambdaList, n_jobs=-1, show_progress=True):
    """Compute continuous scattering data on a real lambda grid.

    Parameters
    ----------
    xList : np.ndarray
        Uniform spatial grid x_i.
    uList : np.ndarray
        Samples of q(x,0).
    vList : np.ndarray
        Samples of q*(x,0).
    lambdaList : np.ndarray
        Real spectral grid lambda_k for the continuous spectrum.
    n_jobs : int
        Number of joblib workers. Use 1 for deterministic serial execution.
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays T(lambda_k), L(lambda_k), and R(lambda_k).
    """
    l = xList[-1]
    h = xList[1] - xList[0]

    jobs = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_compute_single_lambda)(lam, xList, uList, vList, h, l)
        for lam in lambdaList
    )
    results = list(tqdm(jobs, total=len(lambdaList), disable=not show_progress))
    transmission, left_reflection, right_reflection = zip(*results)

    return (
        np.array(transmission),
        np.array(left_reflection),
        np.array(right_reflection),
    )


def getReflectionQuantum_parallel(xList, uList, vList, lambdaList):
    """Compute continuous scattering data in one block-diagonal simulation.

    Parameters are the same as getReflectionQuantum. The lambda grid is padded
    to a power of two and encoded in an ancillary register, matching the
    parallel direct-scattering construction in the paper.
    """
    h = xList[1] - xList[0]
    l = xList[-1]
    n_lam = len(lambdaList)

    m = int(np.ceil(np.log2(n_lam)))
    padded_size = 2**m
    qc = QuantumCircuit(m + 1)

    initial_state = np.zeros(2 ** (m + 1), dtype=complex)
    for k, lam in enumerate(lambdaList):
        initial_state[2 * k] = np.exp(-1j * lam * l)

    norm = np.linalg.norm(initial_state)
    initial_state /= norm
    qc.initialize(initial_state, range(m + 1))

    for i in range(len(xList)):
        blocks = []
        for k in range(padded_size):
            if k < n_lam:
                lam = lambdaList[k]
                hamiltonian = getHamiltonian(i, uList, vList, lam)
                unitary = expm(-1j * hamiltonian * h)
            else:
                unitary = np.eye(2, dtype=complex)
            blocks.append(unitary)

        qc.append(Operator(block_diag(*blocks)), range(m + 1))

    final_state = Statevector(qc).data * norm

    transmission, left_reflection, right_reflection = [], [], []
    for k, lam in enumerate(lambdaList):
        a = final_state[2 * k] / np.exp(1j * lam * l)
        b = final_state[2 * k + 1] / np.exp(-1j * lam * l)

        left_reflection.append(-np.conj(b) / a)
        right_reflection.append(-b / a)
        transmission.append(1 / a)

    return (
        np.array(transmission),
        np.array(left_reflection),
        np.array(right_reflection),
    )
