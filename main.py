import argparse

import numpy as np

from cases import getExactSolitonEx1
from continuousSpectrum import getReflectionQuantum, getReflectionQuantum_parallel
from discretizeSpectrum import compute_norming_constants_high_prec, extract_poles_Chebyshev
from marchenko import calcMarchenkoResult, generate_optimized_grids


def run_soliton_pipeline(
    n_x=256,
    length=4.0,
    eta=4.0,
    xi=0.1,
    x0=0.0,
    phi=0.0,
    t=4.0,
    n_scatter=4096,
    chebyshev_nodes=1024,
    marchenko_size=16,
    marchenko_solver="classic",
    spectrum_solver="parallel",
    n_jobs=1,
    show_progress=True,
):
    """Run the full Lax-pair scattering pipeline for the soliton notebook case.

    The workflow follows the original soliton notebook:

    1. Build the initial field q(x,0) on the reconstruction grid.
    2. Extract the discrete spectrum lambda_j from a fine scattering grid.
    3. Compute the discrete norming constants.
    4. Compute the continuous scattering data on the real lambda grid.
    5. Reconstruct q(x,t) by solving the GLM equation.

    Parameters
    ----------
    n_x : int
        Number of spatial grid points for reconstruction.
    length : float
        Half-width of the physical interval, x in [-length, length].
    eta, xi, x0, phi : float
        One-soliton parameters used by getExactSolitonEx1.
    t : float
        Evolution time.
    n_scatter : int
        Number of spatial grid points used for discrete-spectrum extraction.
    chebyshev_nodes : int
        Number of Chebyshev collocation nodes used to extract lambda_j.
    marchenko_size : int
        GLM truncation size M.
    marchenko_solver : str
        Solver passed to calcMarchenkoResult.
    spectrum_solver : str
        "parallel" uses getReflectionQuantum_parallel; "serial" uses
        getReflectionQuantum.
    n_jobs : int
        Number of workers for serial continuous-spectrum and GLM solves.
    show_progress : bool
        Whether to show progress bars.

    Returns
    -------
    dict
        x grid, lambda grid, initial field, continuous data, discrete data,
        reconstructed q(x,t), exact q(x,t), and relative L2 error.
    """
    x_grid = np.linspace(-length, length, n_x, endpoint=True)
    dx = x_grid[1] - x_grid[0]
    lambda_grid = generate_optimized_grids(dx, n_x)

    q0 = getExactSolitonEx1(x_grid, t=0.0, eta=eta, xi=xi, x0=x0, phi=phi)
    q0_conj = np.conj(q0)

    x_scatter = np.linspace(-length, length, n_scatter, endpoint=True)
    q_scatter = getExactSolitonEx1(
        x_scatter, t=0.0, eta=eta, xi=xi, x0=x0, phi=phi
    )
    lambdas = extract_poles_Chebyshev(x_scatter, q_scatter, n=chebyshev_nodes)
    gammas_l, gammas_r = compute_norming_constants_high_prec(
        lambdas, x_scatter, q_scatter
    )

    if spectrum_solver == "parallel":
        transmission, left_reflection, right_reflection = getReflectionQuantum_parallel(
            x_grid, q0, q0_conj, lambda_grid
        )
    elif spectrum_solver == "serial":
        transmission, left_reflection, right_reflection = getReflectionQuantum(
            x_grid,
            q0,
            q0_conj,
            lambda_grid,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )
    else:
        raise ValueError("spectrum_solver must be 'parallel' or 'serial'")

    scattering_data = {
        "R": right_reflection,
        "L": left_reflection,
        "boundStates": lambdas,
        "gammaR": gammas_r,
        "gammaL": gammas_l,
    }

    reconstructed = np.asarray(
        calcMarchenkoResult(
            x_grid,
            t,
            lambda_grid,
            scattering_data,
            solver=marchenko_solver,
            degree=2,
            m=marchenko_size,
            n_jobs=n_jobs,
            show_progress=show_progress,
            task_name="soliton",
        )
    )
    exact = getExactSolitonEx1(x_grid, t=t, eta=eta, xi=xi, x0=x0, phi=phi)
    relative_l2_error = np.linalg.norm(np.abs(reconstructed) - np.abs(exact)) / np.linalg.norm(np.abs(exact))

    return {
        "x": x_grid,
        "lambda": lambda_grid,
        "q0": q0,
        "T": transmission,
        "L": left_reflection,
        "R": right_reflection,
        "boundStates": lambdas,
        "gammaL": gammas_l,
        "gammaR": gammas_r,
        "reconstructed": reconstructed,
        "exact": exact,
        "relative_l2_error": relative_l2_error,
    }


def main():
    """Parse command-line arguments and run the full soliton pipeline."""
    parser = argparse.ArgumentParser(
        description="Full Lax-pair scattering pipeline for the soliton example."
    )
    parser.add_argument("--n-x", type=int, default=256)
    parser.add_argument("--length", type=float, default=4.0)
    parser.add_argument("--eta", type=float, default=4.0)
    parser.add_argument("--xi", type=float, default=0.1)
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--time", type=float, default=4.0)
    parser.add_argument("--n-scatter", type=int, default=4096)
    parser.add_argument("--chebyshev-nodes", type=int, default=1024)
    parser.add_argument("--marchenko-size", type=int, default=16)
    parser.add_argument("--marchenko-solver", default="classic")
    parser.add_argument("--spectrum-solver", choices=["parallel", "serial"], default="parallel")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    result = run_soliton_pipeline(
        n_x=args.n_x,
        length=args.length,
        eta=args.eta,
        xi=args.xi,
        x0=args.x0,
        phi=args.phi,
        t=args.time,
        n_scatter=args.n_scatter,
        chebyshev_nodes=args.chebyshev_nodes,
        marchenko_size=args.marchenko_size,
        marchenko_solver=args.marchenko_solver,
        spectrum_solver=args.spectrum_solver,
        n_jobs=args.n_jobs,
        show_progress=not args.no_progress,
    )

    print(f"discrete eigenvalues lambda_j: {result['boundStates']}")
    print(f"gammaL: {result['gammaL']}")
    print(f"gammaR: {result['gammaR']}")
    print(f"relative L2 error: {result['relative_l2_error']:.3e}")


if __name__ == "__main__":
    main()
