# AQFED.jl
Julia package for the book [Applied Quantitative Finance for Equity Derivatives](https://jherekhealy.github.io)

## Installation

Requires Julia 1.x. Clone the repository, then install the package from the Julia REPL:
```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/jherekhealy/AQFED.jl")
```
Or, from a local clone, start Julia in the AQFED folder, press `]` to enter the package manager and run
```julia
(AQFED) pkg> activate .
(AQFED) pkg> instantiate
```
Then `using AQFED`. The test suite runs with `(AQFED) pkg> test`.

## Available Modules
- [Implied Volatility Solver](/src/black): fast and robust solvers to find the Black-Scholes implied volatility corresponding to a given option price.
- [Piecewise-Lognormal Model](/src/pln) (also known as spot model) for cash dividends: fast and accurate approximations.
- [American Option Pricing](/src/american): Andersen-Lake technique.
- [Basket Option Pricing](/src/basket): Curran approximations, which may also be reused to price Asian options or vanilla options in the piecewise-lognormal model.
- [Collocation](/src/collocation): fit market implied vols via stochastic collocation.
- [Bachelier](/src/bachelier): Bachelier formula and corresponding implied vol "solver".
- [Random Number Generators](/src/random): good pseudo and quasi random number generators.
- [Term Structures](/src/ts): curves, dividends, SVI and Heston term structure models.
- [Local Volatility](/src/pdde): one-step local volatility via PDDE (Andersen-Hagan and Green approximations, LVG).
- [Volatility Models](/src/vol): SVI, SSVI, SABR, Heston and LNMixture calibrations, CVI and variance swaps.
- [Finite Differences](/src/fdm): Crank-Nicolson, TR-BDF2 and Rannacher-Kulmala-Lee solvers.
- [Monte Carlo](/src/mc): Brownian bridge, Heston and local-volatility simulation.
- [Rough Heston](/src/rough): Pade-based rough Heston pricing and Monte Carlo.
- [Spread Options](/src/spread) and [Asian Options](/src/asian).
- [Fractal Models](/src/fractal) and [Minimizers](/src/minimizer) (differential evolution, simulated annealing).
