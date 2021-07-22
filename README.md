# AQFED.jl
Julia package for the book [Applied Quantitative Finance for Equity Derivatives](https://jherekhealy.github.io)

## Installation

A simple way to get started is to clone the repository, start julia from the AQFED folder, type `]` to enter the package environment and do
```julia
(v1.5) pkg> activate .
```
Then ESC or CTRL+C to exit the pkg env.

## Available Modules
- [Implied Volatility Solver](/src/black): fast and robust solvers to find the Black-Scholes implied volatility corresponding to a given option price.
- [Piecewise-Lognormal Model](/src/pln) (also known as spot model) for cash dividends: fast and accurate approximations.
- [American Option Pricing](/src/american): Andersen-Lake technique.
- [Basket Option Pricing](/src/basket): Curran approximations, which may also be reused to price Asian options or vanilla options in the piecewise-lognormal model.
- [Collocation](/src/collocation): fit market implied vols via stochastic collocation.
- [Bachelier](/src/bachelier): Bachelier formula and corresponding implied vol "solver".
- [Random Number Generators](/src/random): good pseudo and quasi random number generators.
