# Quantum Lax-Pair Scattering for the NLSE

This repository contains the core code used for Lax-pair scattering based
simulation of the focusing nonlinear Schrodinger equation (NLSE),

```text
i q_t = q_xx + 2 |q|^2 q.
```

The notation follows the accompanying paper `Quantum algorithm for the nonlinear Schrodinger equation via the Lax-pair scattering`:

- `q(x,t)`: physical NLSE field.
- `lambda`: Zakharov-Shabat spectral parameter.
- `a(lambda), b(lambda)`: continuous scattering coefficients.
- `lambda_j`: discrete eigenvalues in the upper half complex plane.
- `c_j`: discrete norming constants.
- `omega(y)`: scalar GLM kernel built from evolved scattering data.
- `B1(x,y), B2(x,y)`: unknown GLM components, with `q(x,t) = 2 B2(x,0)`.

The repository has been cleaned for release: generated data, paper figures,
cache files, old notebooks, and exploratory test scripts were removed.  The only
notebook kept is `demo.ipynb`, which provides a minimal visual example.

## Quick Start

Install the dependencies in any Python environment that supports Qiskit:

```bash
pip install -r requirements.txt
python main.py
```

The command runs a small one-soliton reconstruction and prints the relative
L2 error against the analytic solution.

For a visual walkthrough:

```bash
jupyter notebook demo.ipynb
```

## Paper Experiment Settings

The default command is only a lightweight smoke test for the full pipeline.  The
larger parameter sets used for the paper experiments are intentionally more
expensive, because they were chosen to improve numerical reliability in the
direct scattering, discrete-spectrum extraction, and GLM reconstruction steps.

The original batch configuration used the following columns:

- `case`: physical example to run.
- `N`: number of physical-space grid points used for reconstruction.
- `L`: half-width of the reconstruction domain, so `x in [-L, L]`.
- `Nscatter`: number of grid points on the fine grid used to extract the
  discrete spectrum.
- `Lscatter`: half-width of that fine scattering domain.
- `n`: number of Chebyshev collocation nodes used by `extract_poles_Chebyshev`
  to find the discrete eigenvalues `lambda_j`.
- `m`: truncation size of the discretized GLM system at each spatial point.
- `spectrumSolver`: method for continuous scattering data.  `quantum` uses the
  parallel statevector construction, while `hybrid` uses the per-`lambda`
  routine with joblib parallelism.
- `MarchenkoSolver`: linear solver used in the GLM reconstruction.
- `tMax`, `numT`: final time and number of output snapshots in
  `linspace(0, tMax, numT)`.

The main paper examples use:

| case          |   N | Nscatter |  L | Lscatter |    n |  m | spectrumSolver | MarchenkoSolver                 |  tMax | numT |
| ------------- | --: | -------: | -: | -------: | ---: | -: | -------------- | ------------------------------- | ----: | ---: |
| `breather`  | 256 |     4096 | 10 |       20 | 2048 | 16 | `quantum`    | `quantum`                     |  6.28 |  101 |
| `soliton2`  | 256 |     4096 | 15 |       20 |  512 | 16 | `quantum`    | `quantum`                     |  8.00 |  101 |
| `finite_mi` | 512 |      512 | 24 |       24 |  512 | 64 | `hybrid`     | `quantum_complex_svd_precond` | 10.00 |  101 |

The `breather` case uses the exact second-order breather initial condition from
`getExactN2Breather`.  The larger Chebyshev size `n=2048` was used because the
breather reconstruction is sensitive to the accurate extraction of multiple
discrete eigenvalues.

The `soliton2` case uses `getTwoSolitonCollisionInit` with shifted carrier
parameters `xis=[0.8/2, -0.5/2]`.  The wider domains `L=15` and `Lscatter=20`
keep the separated soliton tails small at the boundaries during the collision
experiment.

The `finite_mi` case uses `getFiniteWidthMIInit` with
`A=1.0, width=10.0, eps=0.02, k=1.2, edge_power=8`.  This is a broad,
finite-width approximation to a modulated continuous wave, so it uses the
largest reconstruction domain and a larger GLM truncation `m=64`.  The
`quantum_complex_svd_precond` GLM solver was used because the corresponding GLM
systems can be more ill-conditioned than the localized soliton examples.

## File Guide

### `main.py`

Minimal command-line entry point.

- `run_soliton_pipeline(...)`: follows the full soliton workflow: builds
  `q(x,0)`, computes continuous and discrete scattering data, evolves the data,
  solves the GLM system, and returns the reconstruction diagnostics.
- `main()`: parses command-line arguments and prints the demo error.

### `cases.py`

Initial conditions and analytic helpers for the retained paper cases.

- `getExactSolitonEx1(x, t, eta, xi, x0, phi)`: exact one-soliton field, kept as
  a helper for the two-soliton collision initial condition.
- `getExactN2Breather(x, t)`: exact second-order breather used by `breather`.
- `getTwoSolitonCollisionInit(...)`: approximate two-soliton collision initial
  condition used by `soliton2`.
- `getFiniteWidthMIInit(...)`: finite-width modulated continuous-wave initial
  condition used by `finite_mi`.

### `continuousSpectrum.py`

Direct scattering routines for the continuous spectrum.

- `getHamiltonian(x_idx, uList, vList, lam)`: local Z-S Hamiltonian `H(x,lambda)`.
- `getReflectionQuantum(...)`: computes `T(lambda), L(lambda), R(lambda)` for
  each real spectral point independently.
- `getReflectionQuantum_parallel(...)`: computes the same data using a single
  block-diagonal statevector simulation over all spectral points.

### `discretizeSpectrum.py`

Discrete spectrum and norming constants.

- `extract_poles_dense_fd(...)`: dense finite-difference extraction of
  eigenvalues `lambda_j`.
- `extract_poles_shift_invert(...)`: sparse shift-invert extraction of
  eigenvalues `lambda_j`.
- `extract_poles_Chebyshev(...)`: Chebyshev-collocation extraction of
  eigenvalues `lambda_j`.
- `compute_norming_constants(...)`: ODE-based norming constants.
- `compute_norming_constants_high_prec(...)`: higher-precision norming constants
  using cubic interpolation and DOP853 integration.

### `marchenko.py`

Time evolution and inverse scattering through the GLM equation.

- `timeEvolution(lambdaList, t, scatteringData)`: applies the paper's decoupled
  spectral evolution to continuous and discrete data.
- `calcOmegaL(...)`, `calcOmegaR(...)`: build left/right scalar GLM kernels.
- `BuildMarchenkoSystem(...)`: discretizes the GLM integral equation at one
  spatial point.
- `generate_optimized_grids(dx, N_lambda)`: chooses a spectral grid compatible
  with the spatial grid.
- `calcMarchenkoResult(...)`: reconstructs `q(x,t)` over a spatial grid.

The `solver` argument accepts `"classic"`, `"quantum"`, `"quantum_complex"`, and
`"quantum_complex_svd_precond"`.

### `qsvtSolver.py`

Adapters between the GLM linear systems and the vendored QSVT implementation.

- `complex2realMatrix(A)`, `complex2realVector(b)`: real block embedding of a
  complex linear system.
- `qsvtSolverAbs(A, b, degree)`: QSVT solve after complex-to-real conversion.
- `qsvtSolverComplex(A, b, degree)`: direct complex QSVT solve.
- `qsvtSolverComplexSVDPreconditioned(...)`: dense SVD preconditioning followed
  by the direct complex QSVT solver.

### `qsvt/`

Vendored QSVT implementation.  This directory is modified from
[lbwei1016/QSVT-implementation](https://github.com/lbwei1016/QSVT-implementation).
See `qsvt/README.md` for attribution details.

## Notes

- The QSVT solver paths are experimental and require statevector simulation.
- The numerical Lax method and the discretized GLM reconstruction are mainly
  based on A. Arico, G. Rodriguez, and S. Seatzu, "Numerical solution of the
  nonlinear Schrodinger equation, starting from the scattering data", Calcolo
  48 (2011), 75-88.
