import numpy as np
import qsvtSolver
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def timeEvolution(lambdaList, t, scatteringData):
    """Evolve scattering data in spectral space.

    This implements the decoupled time evolution of the scattering data used in
    the paper. The real-axis continuous data receive quadratic phase factors,
    while the discrete eigenvalues lambda_j stay fixed and the norming
    constants evolve independently.

    Parameters
    ----------
    lambdaList : np.ndarray
        Real spectral grid lambda_k for the continuous spectrum.
    t : float
        Evolution time.
    scatteringData : dict
        Dictionary with keys R, L, boundStates, gammaR, and gammaL. The entries
        boundStates and gamma* correspond to lambda_j and c_j-like norming data.

    Returns
    -------
    dict
        Copy of scatteringData after spectral time evolution.
    """
    scatteringData = {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else list(value)
        for key, value in scatteringData.items()
    }

    phase_cont = 4j * (lambdaList**2) * t
    scatteringData["R"] *= np.exp(phase_cont)
    scatteringData["L"] *= np.exp(-phase_cont)

    if "boundStates" in scatteringData:
        lambdas = np.array(scatteringData["boundStates"])
        phase_disc = 4j * (lambdas**2) * t
        scatteringData["gammaL"] = np.array(scatteringData["gammaL"]) * np.exp(phase_disc)
        scatteringData["gammaR"] = np.array(scatteringData["gammaR"]) * np.exp(-phase_disc)

    return scatteringData


def calcOmegaL(xList, lambdaList, scatteringData):
    """Build the left scalar GLM kernel omega_L(alpha).

    The kernel has the form of the paper's scalar kernel omega(y): an inverse
    Fourier contribution from continuous reflection data plus a discrete sum
    over lambda_j and c_j-like norming constants.

    Parameters
    ----------
    xList : np.ndarray
        Spatial grid. It is kept for API symmetry with the reconstruction code.
    lambdaList : np.ndarray
        Uniform real spectral grid lambda_k.
    scatteringData : dict
        Evolved scattering data. Uses R(lambda_k) and gammaL.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        omega_L(alpha_p) and the corresponding alpha grid.
    """
    n_lambda = len(lambdaList)
    d_lambda = lambdaList[1] - lambdaList[0]

    r_lambda = scatteringData["R"]
    r_shifted = np.fft.ifftshift(r_lambda)
    scale = n_lambda * d_lambda / (2 * np.pi)
    omega_continuous = np.fft.ifft(r_shifted) * scale
    omega_l = np.fft.fftshift(omega_continuous)

    freqs = np.fft.fftfreq(n_lambda, d=d_lambda)
    alphaList = np.fft.fftshift(freqs) * 2 * np.pi

    if "boundStates" in scatteringData:
        for i, lamb in enumerate(scatteringData["boundStates"]):
            gamma = scatteringData["gammaL"][i]
            omega_l += gamma * np.exp(1j * lamb * alphaList)

    return omega_l, alphaList


def calcOmegaR(xList, lambdaList, scatteringData):
    """Build the right scalar GLM kernel omega_R(alpha).

    Parameters are the same as calcOmegaL. This routine uses L(lambda_k) and
    gammaR.
    """
    n_lambda = len(lambdaList)
    d_lambda = lambdaList[1] - lambdaList[0]

    l_lambda = scatteringData["L"]
    l_shifted = np.fft.ifftshift(l_lambda)
    scale = n_lambda * d_lambda / (2 * np.pi)
    omega_continuous = np.fft.ifft(l_shifted) * scale
    omega_r = np.fft.fftshift(omega_continuous)

    freqs = np.fft.fftfreq(n_lambda, d=d_lambda)
    alphaList = np.fft.fftshift(freqs) * 2 * np.pi

    if "boundStates" in scatteringData:
        for i, lamb in enumerate(scatteringData["boundStates"]):
            gamma = scatteringData["gammaR"][i]
            omega_r += gamma * np.exp(1j * lamb * alphaList)

    return omega_r, alphaList


def BuildMarchenkoSystem(omegaL, omegaR, alphaList, x, matrixSize=None):
    """Assemble the discretized GLM linear system at one spatial point x.

    The continuous GLM equation solves for B1(x,y) and B2(x,y) on y >= 0.
    After truncation and Simpson quadrature, this routine returns the matrix
    and right-hand side for one x. The recovered field is q(x,t)=2 B2(x,0).

    Parameters
    ----------
    omegaL, omegaR : np.ndarray
        Left and right scalar kernels sampled on alphaList.
    alphaList : np.ndarray
        Uniform grid for alpha = y + z + 2x.
    x : float
        Spatial point x_j where the GLM system is assembled.
    matrixSize : int or None
        Number M of quadrature/collocation points retained for y,z >= 0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Coefficient matrix and right-hand side.
    """
    d_alpha = alphaList[1] - alphaList[0]
    n_alpha = len(alphaList)
    zero_idx = np.argmin(np.abs(alphaList))

    max_points = n_alpha - zero_idx
    matrix_size = max_points if matrixSize is None else min(matrixSize, max_points)

    x_shift = int(np.round(2 * x / d_alpha))

    weights = np.ones(matrix_size)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    if matrix_size > 2:
        weights[-1] = 1
    quadrature = np.diag(weights * d_alpha / 3.0)

    row_idx, col_idx = np.meshgrid(
        np.arange(matrix_size), np.arange(matrix_size), indexing="ij"
    )

    if x > 0:
        lookup = zero_idx + row_idx + col_idx + x_shift
        valid = (lookup >= 0) & (lookup < n_alpha)
        omega_mat = np.zeros((matrix_size, matrix_size), dtype=complex)
        omega_mat[valid] = omegaL[lookup[valid]]

        rhs_idx = zero_idx + np.arange(matrix_size) + x_shift
        rhs_valid = (rhs_idx >= 0) & (rhs_idx < n_alpha)
        omega_vec = np.zeros(matrix_size, dtype=complex)
        omega_vec[rhs_valid] = omegaL[rhs_idx[rhs_valid]]

        identity = np.eye(matrix_size, dtype=complex)
        omega_d = omega_mat @ quadrature
        omega_conj_d = np.conj(omega_mat) @ quadrature

        top = np.hstack([identity, -omega_conj_d])
        bottom = np.hstack([omega_d, identity])
        matrix = np.vstack([top, bottom])
        rhs = np.concatenate([np.zeros(matrix_size, dtype=complex), -omega_vec])
    else:
        lookup = zero_idx + row_idx + col_idx - x_shift
        valid = (lookup >= 0) & (lookup < n_alpha)
        omega_mat = np.zeros((matrix_size, matrix_size), dtype=complex)
        omega_mat[valid] = omegaR[lookup[valid]]

        rhs_idx = zero_idx + np.arange(matrix_size) - x_shift
        rhs_valid = (rhs_idx >= 0) & (rhs_idx < n_alpha)
        omega_vec = np.zeros(matrix_size, dtype=complex)
        omega_vec[rhs_valid] = omegaR[rhs_idx[rhs_valid]]

        identity = np.eye(matrix_size, dtype=complex)
        omega_d = omega_mat @ quadrature
        omega_conj_d = np.conj(omega_mat) @ quadrature

        top = np.hstack([identity, omega_d])
        bottom = np.hstack([-omega_conj_d, identity])
        matrix = np.vstack([top, bottom])
        rhs = np.concatenate([np.zeros(matrix_size, dtype=complex), np.conj(omega_vec)])

    return matrix, rhs


def generate_optimized_grids(dx, N_lambda):
    """Generate a real lambda grid compatible with the spatial spacing.

    Parameters
    ----------
    dx : float
        Spatial grid spacing h.
    N_lambda : int
        Number of spectral grid points N_lambda.

    Returns
    -------
    np.ndarray
        Uniform spectral grid lambda_k.
    """
    d_alpha_target = 2 * dx
    lambda_width = 2 * np.pi / d_alpha_target
    lambda_min = -lambda_width / 2
    lambda_max = lambda_width / 2
    return np.linspace(lambda_min, lambda_max, N_lambda, endpoint=False)


def _process_single_x(x, omegaLNew, omegaRNew, alphaList, m, solver, degree, eps=1e-6):
    """Solve one GLM system and return the reconstructed q(x,t)."""
    matrix, rhs = BuildMarchenkoSystem(omegaLNew, omegaRNew, alphaList, x, m)
    norm_rhs = np.linalg.norm(rhs)

    if solver == "classic":
        result = np.linalg.solve(matrix, rhs)
    elif solver == "quantum":
        result = np.zeros_like(rhs) if norm_rhs < eps else qsvtSolver.qsvtSolverAbs(
            matrix, rhs, degree=degree
        )
    elif solver == "quantum_complex":
        result = np.zeros_like(rhs) if norm_rhs < eps else qsvtSolver.qsvtSolverComplex(
            matrix, rhs, degree=degree
        )
    elif solver == "quantum_complex_svd_precond":
        result = (
            np.zeros_like(rhs)
            if norm_rhs < eps
            else qsvtSolver.qsvtSolverComplexSVDPreconditioned(
                matrix, rhs, degree=degree, tau_rel=0.05
            )
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return 2 * result[m] if x > 0 else -2 * result[m]


def calcMarchenkoResult(
    xList,
    t,
    lambdaList,
    scatterData,
    solver="classic",
    m=16,
    degree=4,
    n_jobs=-1,
    show_progress=True,
    task_name="",
):
    """Reconstruct q(x,t) on a spatial grid from scattering data.

    Parameters
    ----------
    xList : np.ndarray
        Spatial grid x_j.
    t : float
        Evolution time.
    lambdaList : np.ndarray
        Real spectral grid lambda_k.
    scatterData : dict
        Initial scattering data. Required keys are R, L, boundStates, gammaR,
        and gammaL.
    solver : str
        Linear solver for each GLM system: classic, quantum, quantum_complex,
        or quantum_complex_svd_precond.
    m : int
        GLM truncation size M.
    degree : int
        QSVT inverse-polynomial preset, used only by quantum solvers.
    n_jobs : int
        Number of parallel workers over spatial grid points.
    show_progress : bool
        Whether to display a progress bar.
    task_name : str
        Optional label for progress output.

    Returns
    -------
    list[complex]
        Reconstructed samples q(x_j,t).
    """
    evolved_data = timeEvolution(lambdaList, t, scatterData)
    omega_l, alphaList = calcOmegaL(xList, lambdaList, evolved_data)
    omega_r, _ = calcOmegaR(xList, lambdaList, evolved_data)

    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        delayed_funcs = (
            delayed(_process_single_x)(x, omega_l, omega_r, alphaList, m, solver, degree)
            for x in xList
        )
        results = parallel(delayed_funcs)

        if show_progress:
            return [
                result
                for result in tqdm(
                    results,
                    total=len(xList),
                    desc=f"Marchenko {task_name}",
                    position=1,
                    leave=False,
                )
            ]
        return list(results)
