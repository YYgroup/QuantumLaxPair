import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, interp1d
from scipy.linalg import eigvals
from scipy.sparse.linalg import eigs


def extract_poles_dense_fd(x, q, tol=1e-3):
    """Extract discrete eigenvalues lambda_j with a dense finite-difference Z-S matrix.

    Parameters
    ----------
    x : np.ndarray
        Uniform spatial grid x_i.
    q : np.ndarray
        Samples of the initial field q(x,0).
    tol : float
        Minimum positive imaginary part used to filter the continuous spectrum.

    Returns
    -------
    np.ndarray
        Discrete eigenvalues lambda_j with Im(lambda_j)>tol.
    """
    n_grid = len(x)
    dx = x[1] - x[0]

    c1 = 8.0 / (12.0 * dx)
    c2 = -1.0 / (12.0 * dx)
    diagonals = [
        np.full(n_grid - 2, c2),
        np.full(n_grid - 1, -c1),
        np.full(n_grid - 1, c1),
        np.full(n_grid - 2, -c2),
    ]
    derivative = sp.diags(diagonals, [-2, -1, 1, 2], shape=(n_grid, n_grid), format="lil")

    derivative[0, -1] = -c1
    derivative[0, -2] = c2
    derivative[1, -1] = c2
    derivative[-1, 0] = c1
    derivative[-1, 1] = -c2
    derivative[-2, 0] = -c2
    derivative = derivative.toarray()

    q_diag = np.diag(q)
    q_conj_diag = np.diag(np.conj(q))

    top = np.hstack([1j * derivative, -1j * q_diag])
    bottom = np.hstack([-1j * q_conj_diag, -1j * derivative])
    zs_matrix = np.vstack([top, bottom])

    all_eigenvalues = eigvals(zs_matrix)
    poles = []
    for value in all_eigenvalues:
        if value.imag > tol and not any(np.abs(value - pole) < 1e-4 for pole in poles):
            poles.append(value)

    return np.array(sorted(poles, key=lambda value: value.imag, reverse=True))


def extract_poles_shift_invert(x, q, num_poles=4, sigma_guess=1.0j, tol=1e-4):
    """Extract nearby discrete eigenvalues lambda_j with sparse shift-invert.

    Parameters
    ----------
    x : np.ndarray
        Uniform spatial grid x_i.
    q : np.ndarray
        Samples of q(x,0).
    num_poles : int
        Number of eigenvalues requested from ARPACK.
    sigma_guess : complex
        Shift-invert target near the expected discrete spectrum.
    tol : float
        Minimum positive imaginary part retained.

    Returns
    -------
    np.ndarray
        Filtered discrete eigenvalues lambda_j.
    """
    n_grid = len(x)
    dx = x[1] - x[0]

    c1 = 8.0 / (12.0 * dx)
    c2 = -1.0 / (12.0 * dx)
    diagonals = [
        np.full(n_grid - 2, c2),
        np.full(n_grid - 1, -c1),
        np.full(n_grid - 1, c1),
        np.full(n_grid - 2, -c2),
    ]
    derivative = sp.diags(diagonals, [-2, -1, 1, 2], shape=(n_grid, n_grid), format="lil")

    derivative[0, -1] = -c1
    derivative[0, -2] = c2
    derivative[1, -1] = c2
    derivative[-1, 0] = c1
    derivative[-1, 1] = -c2
    derivative[-2, 0] = -c2
    derivative = derivative.tocsc()

    q_diag = sp.diags(q, 0, format="csc")
    q_conj_diag = sp.diags(np.conj(q), 0, format="csc")
    zs_matrix = sp.bmat(
        [[1j * derivative, -1j * q_diag], [-1j * q_conj_diag, -1j * derivative]],
        format="csc",
    )

    try:
        eigenvalues, _ = eigs(zs_matrix, k=num_poles, sigma=sigma_guess)
    except Exception as exc:
        print(f"eigenvalue solve failed: {exc}")
        return np.array([])

    poles = []
    for value in eigenvalues:
        if value.imag > tol and not any(np.abs(value - pole) < 1e-4 for pole in poles):
            poles.append(value)

    return np.array(sorted(poles, key=lambda value: value.imag, reverse=True))


def extract_poles_Chebyshev(xList, qList, n=200, a=0.15):
    """Extract Z-S discrete eigenvalues lambda_j with Chebyshev collocation.

    Parameters
    ----------
    xList : np.ndarray
        Spatial grid x_i.
    qList : np.ndarray
        Samples of q(x,0).
    n : int
        Number of Chebyshev nodes.
    a : float
        Tanh-map parameter for x = arctanh(chi)/a.

    Returns
    -------
    np.ndarray
        Discrete eigenvalues lambda_j in the upper half-plane.
    """
    degree = n - 1
    j = np.arange(degree + 1)
    chi = np.cos(np.pi * j / degree)

    c = np.ones(degree + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1) ** j)

    x_nodes_matrix = np.tile(chi, (n, 1)).T
    d_x = x_nodes_matrix - x_nodes_matrix.T
    derivative = (c[:, np.newaxis] / c[np.newaxis, :]) / (d_x + np.eye(n))
    derivative = derivative - np.diag(np.sum(derivative, axis=1))

    valid = (chi > -1) & (chi < 1)
    x_nodes = np.zeros(n)
    x_nodes[valid] = np.arctanh(chi[valid]) / a
    x_nodes[0] = xList[-1]
    x_nodes[-1] = xList[0]

    dHdx = a * (1 - chi**2)
    mapped_derivative = dHdx[:, np.newaxis] * derivative

    order = np.argsort(xList)
    sorted_x = xList[order]
    sorted_q = qList[order]
    interp_real = interp1d(sorted_x, np.real(sorted_q), kind="cubic", bounds_error=False, fill_value=0.0)
    interp_imag = interp1d(sorted_x, np.imag(sorted_q), kind="cubic", bounds_error=False, fill_value=0.0)
    q_nodes = interp_real(x_nodes) + 1j * interp_imag(x_nodes)

    q_diag = np.diag(q_nodes)
    q_conj_diag = np.diag(np.conj(q_nodes))

    top = np.hstack((-mapped_derivative, q_diag))
    bottom = np.hstack((q_conj_diag, mapped_derivative))
    collocation_matrix = -np.vstack((top, bottom))

    eigenvalues = np.linalg.eigvals(-1j * collocation_matrix)
    return eigenvalues[(np.imag(eigenvalues) > 0.05) & (np.abs(np.real(eigenvalues)) < 10)]


def compute_norming_constants(lambdam_array, x_grid, q_vals):
    """Compute norming constants c_j by integrating Z-S Jost solutions.

    Parameters
    ----------
    lambdam_array : complex or array-like
        Discrete eigenvalues lambda_j.
    x_grid : np.ndarray
        Spatial grid x_i.
    q_vals : np.ndarray
        Samples of q(x,0).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Left and right norming constants, gammaL and gammaR.
    """
    lambdam_array = np.atleast_1d(lambdam_array)
    L_bound = x_grid[-1]

    def q_interp(x):
        return np.interp(x, x_grid, q_vals)

    def zs_deriv(x, y, lm):
        q = q_interp(x)
        dy1 = 1j * lm * y[0] - q * y[1]
        dy2 = np.conj(q) * y[0] - 1j * lm * y[1]
        return [dy1, dy2]

    gamma_l = np.zeros(len(lambdam_array), dtype=complex)
    gamma_r = np.zeros(len(lambdam_array), dtype=complex)

    for i, lm in enumerate(lambdam_array):
        psi_initial = [np.exp(1j * lm * L_bound), 0.0 + 0j]
        sol_psi = solve_ivp(
            zs_deriv,
            t_span=(L_bound, -L_bound),
            y0=psi_initial,
            t_eval=x_grid[::-1],
            args=(lm,),
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        psi1 = sol_psi.y[0][::-1]
        psi2 = sol_psi.y[1][::-1]
        gamma_r[i] = 1j / (2 * np.trapezoid(psi1 * psi2, x=x_grid))

        phi_initial = [0.0 + 0j, np.exp(-1j * lm * (-L_bound))]
        sol_phi = solve_ivp(
            zs_deriv,
            t_span=(-L_bound, L_bound),
            y0=phi_initial,
            t_eval=x_grid,
            args=(lm,),
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        phi1 = sol_phi.y[0]
        phi2 = sol_phi.y[1]
        gamma_l[i] = 1j / (2 * np.trapezoid(phi1 * phi2, x=x_grid))

    return gamma_l, gamma_r


def compute_norming_constants_high_prec(lambdam_array, x_grid, q_vals):
    """Compute norming constants c_j with high-accuracy ODE integration.

    Compared with compute_norming_constants, this version uses a cubic spline
    for q(x,0) and augments the ODE with the integral of psi_1 psi_2.

    Parameters
    ----------
    lambdam_array : complex or array-like
        Discrete eigenvalues lambda_j.
    x_grid : np.ndarray
        Spatial grid x_i.
    q_vals : np.ndarray
        Samples of q(x,0).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Left and right norming constants, gammaL and gammaR.
    """
    lambdam_array = np.atleast_1d(lambdam_array)
    L_bound = x_grid[-1]
    q_spline = CubicSpline(x_grid, q_vals)

    def zs_deriv_extended(x, y, lm):
        q = q_spline(x)
        dy1 = 1j * lm * y[0] - q * y[1]
        dy2 = np.conj(q) * y[0] - 1j * lm * y[1]
        dy3 = y[0] * y[1]
        return [dy1, dy2, dy3]

    gamma_l = np.zeros(len(lambdam_array), dtype=complex)
    gamma_r = np.zeros(len(lambdam_array), dtype=complex)
    tol_kwargs = {"rtol": 1e-11, "atol": 1e-13}

    for i, lm in enumerate(lambdam_array):
        psi_initial = [np.exp(1j * lm * L_bound), 0.0 + 0j, 0.0 + 0j]
        sol_psi = solve_ivp(
            zs_deriv_extended,
            t_span=(L_bound, -L_bound),
            y0=psi_initial,
            args=(lm,),
            method="DOP853",
            **tol_kwargs,
        )
        integral_psi = -sol_psi.y[2][-1]
        gamma_l[i] = 1.0 / (2 * integral_psi)

        phi_initial = [0.0 + 0j, np.exp(-1j * lm * (-L_bound)), 0.0 + 0j]
        sol_phi = solve_ivp(
            zs_deriv_extended,
            t_span=(-L_bound, L_bound),
            y0=phi_initial,
            args=(lm,),
            method="DOP853",
            **tol_kwargs,
        )
        integral_phi = sol_phi.y[2][-1]
        gamma_r[i] = 1.0 / (2 * integral_phi)

    return gamma_l, gamma_r
