import numpy as np


def getExactSolitonEx1(x, t, eta=4.0, xi=0.1, x0=0.0, phi=0.0):
    """Evaluate the analytic one-soliton field q(x,t).

    Parameters
    ----------
    x : np.ndarray
        Spatial grid x_j.
    t : float
        Evolution time.
    eta : float
        Soliton amplitude parameter, Im(lambda_j).
    xi : float
        Carrier/velocity parameter, with Re(lambda_j) = -xi.
    x0 : float
        Initial position parameter in the analytic formula.
    phi : float
        Initial phase parameter.

    Returns
    -------
    np.ndarray
        Complex field q(x,t).
    """
    phase = 2 * xi * x + 4 * (eta**2 - xi**2) * t + phi
    envelope = x0 - 2 * eta * x + 8 * eta * xi * t
    return -2j * eta * np.exp(-1j * phase) / np.cosh(envelope)


def getExactN2Breather(x, t):
    """Evaluate the exact second-order breather field q(x,t).

    Parameters
    ----------
    x : np.ndarray
        Spatial grid x_j.
    t : float
        Evolution time.

    Returns
    -------
    np.ndarray
        Complex breather field q(x,t).
    """
    exp_it = np.exp(1j * t)
    exp_8it = np.exp(8j * t)
    numerator = 4 * exp_it * (np.cosh(3 * x) + 3 * exp_8it * np.cosh(x))
    denominator = np.cosh(4 * x) + 4 * np.cosh(2 * x) + 3 * np.cos(8 * t)
    return numerator / denominator


def getTwoSolitonCollisionInit(
    x,
    etas=[1.5, 1.0],
    xis=[-0.8, 0.5],
    x0s=[-8 * 1.5, 5 * 1.5],
    phis=[0, 0],
    t=0.0,
):
    """Build the two-soliton collision initial condition used by `soliton2`.

    The initial condition is an approximate superposition of two separated
    one-soliton profiles.  The `soliton2` paper experiment uses
    `xis=[0.8/2, -0.5/2]`.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid x_j.
    etas, xis, x0s, phis : list[float]
        Parameters for the two one-soliton components.
    t : float
        Time at which the two separated solitons are sampled.

    Returns
    -------
    np.ndarray
        Approximate two-soliton field q(x,t).
    """
    q = np.zeros_like(x, dtype=np.complex128)
    for eta, xi, x0, phi_i in zip(etas, xis, x0s, phis):
        q += getExactSolitonEx1(x, t, eta=eta, xi=xi, x0=x0, phi=phi_i)
    return q


def getFiniteWidthMIInit(
    x,
    A=1.0,
    width=10.0,
    eps=0.08,
    k=1.2,
    edge_power=8,
    phase=0.0,
    carrier=0.0,
):
    """Build the finite-width modulational-instability initial condition.

    The field is a broad super-Gaussian pump with a weak cosine modulation:

        q(x,0) = A exp(-(abs(x)/width)^edge_power)
                 (1 + eps cos(k x + phase)) exp(i carrier x).

    The paper experiment uses `A=1.0`, `width=10.0`, `eps=0.02`,
    `k=1.2`, and `edge_power=8`.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid x_j.
    A : float
        Pump amplitude.
    width : float
        Width of the finite quasi-continuous-wave plateau.
    eps : float
        Relative modulation depth.
    k : float
        Modulation wavenumber.
    edge_power : int
        Super-Gaussian edge power.
    phase : float
        Phase shift of the modulation.
    carrier : float
        Optional carrier wavenumber.

    Returns
    -------
    np.ndarray
        Initial field q(x,0).
    """
    if edge_power <= 0:
        raise ValueError("edge_power must be positive")

    envelope = np.exp(-((np.abs(x) / width) ** edge_power))
    modulation = 1.0 + eps * np.cos(k * x + phase)
    carrier_phase = np.exp(1j * carrier * x)
    return (A * envelope * modulation * carrier_phase).astype(np.complex128)
