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
# Discrete dividends and calibration to listed option prices
`AmericanFDPricer` (in the `FDM` module) prices vanilla American options under Black-Scholes with discrete dividends
(proportional or cash) with TR-BDF2 on a log-uniform grid and a Brennan-Schwartz projection. It is a preallocated workspace
meant to be called many times, e.g. within a calibration: a price on the default 501x365 grid takes about 3ms.

```julia
using AQFED, AQFED.FDM, AQFED.TermStructure
import AQFED.American: priceAmerican
#spot, discount rate, growth rate, dividends (amount, ex-date, pay date, isProportional, isKnown)
market = DiscreteDividendMarket(93.7, 0.0114, 0.0114, [Dividend(0.0055, 0.2266, 0.2457, true, false), Dividend(0.0055, 0.4786, 0.4978, true, false)])
pricer = AmericanFDPricer(nS = 501, nT = 365)
price = priceAmerican(pricer, market, false, 95.0, 0.75, 0.3) #put of strike 95, maturity 0.75, volatility 30%
```

The `VolatilityModels` module calibrates a section of listed American options (single maturity, bid/ask quotes of calls and puts):
1. the forward from put-call parity (an approximation for American options, used as a starting point),
2. the carry (the common amount of the dividends going ex before the maturity, or the growth rate when there is none)
   jointly with the ATM volatilities, such that calls and puts of a same strike share the same volatility,
3. the American implied volatilities of the mid, bid and ask prices of each quoted strike (TOMS748 root finding), in parallel.

```julia
using AQFED.VolatilityModels
quotes = [AmericanOptionQuote(strike, callBid, callAsk, callVolume, putBid, putAsk, putVolume) for ...]
section = AmericanOptionSection(spot, τ, quotes)
cal = calibrateAmericanSection(section, DiscreteDividendMarket(spot, r, r, dividends), pricer = AmericanFDPricer())
cal.market          #calibrated market: dividend amount or growth rate
cal.vols            #AmericanImpliedVolQuote per strike: callMid, callBid, callAsk, putMid, putBid, putAsk
otmVolatility(cal.vols[1], AQFED.TermStructure.forward(cal.market, τ)) #(vol, volume) of the out-of-the-money option, e.g. for a smile fit
```
The pricer is generic: any type implementing `priceAmerican(p, market, isCall, strike, τ, σ)` and `similar(p)` may be used,
for example `JuZhongPricer()` for the carry calibration when no dividend goes ex before the maturity (`carryPricer` keyword).
A full example on AAPL option chains is in `test/american_calibration_test.jl`.
