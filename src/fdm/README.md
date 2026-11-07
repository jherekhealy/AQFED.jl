# Finite Difference Methods

Finite difference solvers for one-dimensional and multi-state option problems.

## Time stepping (`solver.jl`)
- `priceTRBDF2` solves the backward problem with time-stepping schemes `TRBDF2`, `CrankNicolson`, `BDF1` and `Rannacher` (the latter damps the first steps of Crank-Nicolson).
- Boundary conditions via `Down` and `Up`; `DividendPolicy`, `NoCalibration` and `ForwardCalibration` control dividend handling and grid calibration.

## Explicit and stabilised explicit schemes
- `priceRKL2`, `priceRKG2`, `priceRKC2`: Runge-Kutta-Legendre, Gegenbauer and Chebyshev super-time-stepping, for stiff diffusion problems with large stable steps.
- `priceExplicitEuler`: baseline explicit scheme.

## Scripted products (`payoff.jl`)
- `BackwardScript` with `ScriptEvent`s (`on`, `jump!`, `setState!`) describes path-dependent products on a `ProductStateGrid`.
- Ready-made scripts: `autocallScript`, `asianScript`, `volatilitySwapScript`, `discreteVolatilitySwapScript`.

## Other
- `heston.jl`: finite differences for the Heston model; `american_dividends.jl`: American options with discrete dividends; `grid.jl`: grid construction; `rkl.jl`: RKL coefficients.
