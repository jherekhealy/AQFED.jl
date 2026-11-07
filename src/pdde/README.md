# PDDE: Local Volatility Models

Pricing of European options in local volatility models by solving the forward (Dupire-type) PDE in a one-step (semi-analytic) form, plus the calibration routines built on them.

## Models
- `ConstantLVG`, `ConstantBlackLVG`, `LinearBlackLVG`, `EQuadraticLVG`: local volatility functions that are piecewise constant, linear or quadratic in the state, priced by `priceEuropean` and `derivativePrice`.
- `DiscreteLVG` and `DiscreteLogLVG`: discrete-time versions of the quadratic LVG, with cubic-spline or quadratic-Lagrange price interpolation (`CubicSplinePriceInterpolation`, `QuadraticLagrangePriceInterpolation`).
- `ah.jl`, `ah_bspline.jl`: Andersen-Hagan one-step local volatility, including a B-spline parameterisation.
- `green.jl` (`GreenAH` submodule): Green's function one-step local volatility with Chebyshev discretisation.

## Calibration
- `calibrateConstantLVG`, `calibrateConstantBlackLVG`, `calibrateLinearBlackLVG`, `calibrateEQuadraticLVG`, `calibrateDiscreteLogLVG` fit a model to call prices at one expiry, with weights and bounds `L`/`U`.
- `makeKnots` builds the knot placement used by the quadratic calibrations.
- `GreenAHSurface` and `calibrateGreenAHSurface` calibrate a full surface; `greenDupireLocalVolatility` evaluates the local volatility.

Related: the book chapters on local volatility use these routines. See also the [vol](../vol) module for the forward PDE term-structure calibration.
