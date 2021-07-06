# AQFED.jl
Julia package for the book [Applied Quantitative Finance for Equity Derivatives](https://jherekhealy.github.io)

## Installation

A simple way to get started is to clone the repository, start julia from the AQFED folder, type `]` to enter the package environment and do
```julia
(v1.5) pkg> activate .
```
Then ESC or CTRL+C to exit the pkg env.

## Implied Volatility Solver
### Standard Example
```julia
import AQFED.Black

f = 100.0; T=1.0; dfPayment=1.0;
vol = 0.01; strike = 150.0;
price = Black.blackScholesFormula(true, strike, f, vol * vol * T, 1.0, dfPayment)
Black.impliedVolatility(true, price, f, strike, T, dfPayment)
```

The output should read
```
julia> price = Black.blackScholesFormula(true, strike, f, vol * vol * T, 1.0, dfPayment)
9.01002030921698e-25

julia> Black.impliedVolatility(true, price, f, strike, T, dfPayment)
0.03999999999999892
```

### Multiple precision
```julia
import AQFED.Black

f = BigFloat("100.0"); T=BigFloat("1.0"); dfPayment=BigFloat("1.0");
vol = BigFloat("0.01"); strike = BigFloat("150.0");
price = Black.blackScholesFormula(true, strike, f, vol * vol * T, BigFloat(1.0), dfPayment)
Black.impliedVolatility(true, price, f, strike, T, dfPayment)
```
and the output is: 0.009999999999999999999999999999999999999999999999999999999999999999999999999787059, for a price of 3.005699303008737538090695390409788090781673843536520016899477489819371529654813e-361

### Benchmarking
```julia
import AQFED.Black
using BenchmarkTools

f = 100.0; T=1.0; dfPayment=1.0;
vol = 0.04; strike = 150.0;
price = Black.blackScholesFormula(true, strike, f, vol * vol * T, 1.0, dfPayment)

@benchmark Black.impliedVolatilitySRGuess(true, price, f, strike, T, dfPayment)
@benchmark Black.impliedVolatilitySRHalley(true, price, f, strike, T, dfPayment, 0.0, 64, Black.Householder())
@benchmark Black.impliedVolatilityLiSOR(true, price, f, strike, T, dfPayment, 0.0, 0.0, 64, Black.SORTS())
@benchmark Black.impliedVolatilityJaeckel(true, price, f, strike, T, dfPayment)
```

## Piecewise Lognormal Model Approximations
The piecewise lognormal model is used to represent the fixed stock price jump at each cash dividend ex-date.
The Julia package provides two related approximations:

* EtoreGobetPLNPricer, from [Etore and Gobet](https://hal.archives-ouvertes.fr/hal-00507787),
* LeFlochLehmanPLNPricer, the improvement from [Le Floc'h](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2698283).


As input, a vector of CapitalizedDividend objects is used. Each CapitalizedDividend contains the dividend details as well as the capitalization factor from dividend ex-date to option maturity date. Below is an example that displays the accuracy for a single cash dividend accross three strikes.
The reference values have been obtained by nearly exact integration (Haug-Haug-Lewis formula).
```julia
using AQFED; using AQFED.PLN; using AQFED.TermStructure; using Printf
spot = 100.0; vol = 0.3; discountRate = 0.0; divAmount = 7.0; tte = 1.0; ttp = tte
ttd = 0.9 * tte
isCall = true
discountFactor = exp(-discountRate * ttp)
dividends = Vector{CapitalizedDividend{Float64}}(undef, 1)
dividends[1] =
    CapitalizedDividend(Dividend{Float64}(divAmount, ttd, ttd, false, false), exp((tte - ttd) * discountRate))
ll = LeFlochLehmanPLNPricer(2)
refHHL = [43.24845580, 9.07480013, 1.06252880, 0.10473887]
for (i, strike) in enumerate(LinRange(50.0, 200.0, 4))
          priceLL = priceEuropean(ll, isCall, strike,
                  spot * exp(tte * discountRate), vol * vol * tte, tte,  discountFactor, dividends)
          @printf "%4.1f  LL %.8f %.2e\n" strike priceLL (priceLL - refHHL[i])
end
```

## American Option Pricing
An implementation of Andersen-Lake technique for the integral equation of a vanilla American option is available in the [American](https://github.com/jherekhealy/AQFED.jl/tree/master/src/american) Julia module.
Example to price an American put option of maturity τ=0.75 year, and strike 100, on an asset of spot price 80, with volatility σ=30%, paying no dividend, and with interest rate r=4%. The settings nC=7 collocation points, nIter=8 iterations, nTS1=15 first quadrature points, nTS2=31 second quadrature points lead to very accurate prices and good performance in general.

```
using AQFED; using AQFED.American
import AQFED.TermStructure: ConstantBlackModel
strike = 100.0; spot = 80.0; σ = 0.3; τ = 0.75; q = 0.0; r = 0.04
model = ConstantBlackModel(σ, r, q)
pricer = AndersenLakeRepresentation(model, τ, 1e-8, 7, 8, 15, 31, isCall=false)
price = priceAmerican(pricer, strike, spot)
```
The output is 21.086135735070997
