## American Option Pricing
An implementation of Andersen-Lake technique for the integral equation of a vanilla American option is available in the [American](https://github.com/jherekhealy/AQFED.jl/tree/master/src/american) Julia module.
Example to price an American put option of maturity τ=0.75 year, and strike 100, on an asset of spot price 80, with volatility σ=30%, paying no dividend, and with interest rate r=4%. The settings nC=7 collocation points, nIter=8 iterations, nTS1=15 first quadrature points, nTS2=31 second quadrature points lead to very accurate prices and good performance in general.

```julia
using AQFED; using AQFED.American
import AQFED.TermStructure: ConstantBlackModel
strike = 100.0; spot = 80.0; σ = 0.3; τ = 0.75; q = 0.0; r = 0.04
model = ConstantBlackModel(σ, r, q)
pricer = AndersenLakeRepresentation(model, τ, 1e-8, 7, 8, 15, 31, isCall=false)
price = priceAmerican(pricer, strike, spot)
```
The output is 21.086135735070997

# Negative Rates
The implementation supports negative interest rates through the `AndersenLakeNRepresentation`, which may also be used for positive rates.

```julia
using AQFED; using AQFED.American
import AQFED.TermStructure: ConstantBlackModel
strike = 100.0; spot = 80.0; σ = 0.3; τ = 0.75; q = -0.01; r = -0.005
model = ConstantBlackModel(σ, r, q)
pricer = AndersenLakeNRepresentation(model, τ, 1e-8, 7, 8, 15, 201, isCall=false)
price = priceAmerican(pricer, strike, spot)
```
The output is 22.372978113275483

# Runge-Kutta-Legendre (RKL) finite difference scheme
An example implementation of the RKL scheme for American options is available through `makeFDMPriceInterpolation`.
It returns a piecewise polynomial based on the finite difference grid points, and may be evaluated at any point.