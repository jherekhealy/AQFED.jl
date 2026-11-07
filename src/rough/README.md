# Rough Volatility

Rough Heston and rough Bergomi-type models: characteristic functions, simulation and roughness estimation.

## Pricing
- `RoughHestonParams` with `PadeCharFunc` and `AdamsCharFunc` characteristic functions. The rough Heston characteristic functions themselves live in CharFuncPricing; this module wraps them for use in AQFED.
- `logForward` gives the log-forward for a rough Heston model.

## Simulation
- `simulateHQE` and `simulateHQEPath`: hybrid quadratic-exponential scheme for rough Heston paths.
- `FlatRoughBergomi` and `FlatRoughExp`: flat-forward-variance rough models, with `covariance`, `instantVariance` and `ndims` for Monte Carlo.
- `AbsoluteBrownianSV`, `ExpOUSV`, `SVCJModel`: other stochastic volatility path generators, used as comparators.
- `IntegralDiscretization` (`LeftRectangular`, `RightRectangular`, `Trapezoidal`) selects how the fractional kernel integral is discretised.

## Roughness estimation (`estimate.jl`)
- `estimateRHanSchied` and `estimateRSHanSchied` estimate the roughness exponent from a volatility series.
- `realizedVolatility`, `parkinsonVolatility`, `realizedVolatilityCum` compute realised measures from price series.
