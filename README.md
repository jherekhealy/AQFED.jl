# AQFED.jl
Julia package for the book [Applied Quantitative Finance for Equity Derivatives](https://jherekhealy.github.io)

## Installation

A simple way to get started is to clone the repository, start julia from the AQFED folder, type `]` to enter the package environment and do
```julia
(v1.5) pkg> activate .
```
Then ESC or CTRL+C to exit the pkg env.

## Available Modules
- [Implied Volatility Solver](https://github.com/jherekhealy/AQFED.jl/tree/master/black): fast and robust solvers to find the Black-Scholes implied volatility corresponding to a given option price.
- [Piecewise Lognormal Model](https://github.com/jherekhealy/AQFED.jl/tree/master/src/pln) (also known as spot model) for cash dividends: fast and accurate approximations.
- [American Option Pricing](https://github.com/jherekhealy/AQFED.jl/tree/master/src/american): Andersen-Lake technique.
