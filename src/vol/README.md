# Volatility Surfaces and Calibration

Smile and surface models fitted to market implied volatilities or option prices.

## Surfaces
- **Fengler** (`fengler.jl`): arbitrage-aware smile surfaces built from `C2Spline` slices. `calibrateFenglerSlice`, `calibrateFenglerSurface` and `calibrateFenglerSurfaceJoint` fit them; `price`, `varianceByLogmoneyness` and `density` evaluate them.
- **Multiquadric RBF** (`cvi2d.jl`, `cvi.jl`): `Multiquadric2DSurface` with `calibrateMultiquadricSurface2D`, giving total variance, its derivatives and the implied density.
- **SVI, SSVI, SABR**: calibrations in `svi_calibration.jl`, `ssvi_calibration.jl` and `sabr_calibration.jl`.
- **Lognormal mixture** (`lnmixture.jl`): fit of a mixture of lognormals to a smile.

## Stochastic models
- `heston_calibration.jl`, `hestonts_calibration.jl`, `heston_hagan.jl`: Heston and term-structure Heston calibration, including the Hagan approximation. `calibrateDoubleHestonFromPrices` handles the double Heston model.
- `varswap.jl`: variance swap pricing and fair-variance utilities.

## American options
- `american_calibration.jl`: calibrates carry and ATM volatilities for American option sections from quotes, via `calibrateAmericanSection`, `calibrateCarryAndATMVols` and `americanImpliedVolatility`.

## Other
- `QuadraticLVForwardPDE` and `calibrateQuadraticLVTermStructure`: quadratic local volatility term-structure fit by forward PDE.

Units: volatilities are decimals (0.2 is 20%) in the APIs.
