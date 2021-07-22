## Approximations for Vanilla Basket Options
The following approximations of vanilla basket option prices under the Black-Scholes model are available:

- Curran lower bound approximation (using Deelstra et al. formulation): this is `DeelstraLBBasketPricer`.
- Curran shifted lognormal approximation as per Deelstra et al.: this is `DeelstraBasketPricer`. There are 9 possible variations, represented by δIndex=1,2,3 and fIndex=1,2,3.

Below is an example of a call option of strike 110 and maturity one year on a basket composed of 5 assets with weights [0.05, 0.15, 0.2, 0.25, 0.35].
```julia
using AQFED; using AQFED.Basket
weight = [0.05, 0.15, 0.2, 0.25, 0.35]
spot = [100.0, 100.0, 100.0, 100.0, 100.0]
r = 0.05; rho = 0.5; τ = 1.0
sigma = [0.2, 0.2, 0.2, 0.2, 0.2]
correlation = [1.0 rho rho rho rho
	rho 1.0 rho rho rho
	rho rho 1.0 rho rho
	rho rho rho 1.0 rho
	rho rho rho rho 1.0]
totalVariance = sigma.^2 .* τ
forward = spot .* exp(r*τ)
discountFactor = exp(-r*τ)
strike = 110.0
price = priceEuropean(DeelstraBasketPricer(), true, strike,  discountFactor, spot, forward, totalVariance, weight, correlation)
```
The output is 4.396711483510768. The reference Monte-Carlo value from Ju [Pricing Asian and Basket Options via Taylor Expansions](https://www.academia.edu/download/4686930/jujcf02.pdf) is 4.3969 with a standard error of 0.0004 and our own Monte-Carlo simulation leads to a reference value of 4.3967.
With `DeelstraLBBasketPricer()`, the output is 4.392504857517001.

## Asians via Basket
Basket approximations may be reused to price an Asian option on a single asset, with a term-structure of volatilities and drifts.
The Basket module provides the functions `priceAsianFixedStrike` and `priceAsianFloatingStrike` for this purpose.

Below is an example of a weekly averaging call option with first observation at valuation time t=0 and last observation at maturity date t=3 years,
using a volatility of 50% and an interest rate of 9% as in the example of Ju (2002)

```julia
using AQFED; using AQFED.Basket; using Printf
spot = 100.0; r = 0.09; q = 0.0; σ = 0.5
τ=3.0; strikes = [95.0, 100.0, 105.0]
refPrices = [24.5526, 22.6115, 20.8241] #TS values

n = Int(τ * 52)
weights = zeros(Float64, n + 1)
tvar = zeros(Float64, n + 1)
forward = zeros(Float64, n + 1)
for i = 1:length(weights)
	weights[i] = 1.0 / (n + 1)
	ti = (i-1) / n * τ
	tvar[i] = σ^2 * ti
	forward[i] = spot * exp((r - q) * ti)
end
discountFactor = exp(-r * τ)

p = DeelstraBasketPricer(1, 3)
for (refPrice,strike) in zip(refPrices,strikes)
	price = priceAsianFixedStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
	@printf("%6.2f Deelstra    %.4f %.2e\n", strike, price, price - refPrice)
end
```

The errors are much smaller than any approximation listed in the Ju (2002) paper.

## Automatic Differentiation

The code supports Julia ForwardDiff. In particular, it has been modified as to handle the case where t=0, which may lead to divisions by 0 in the greeks calculations.

Below is an example to obtain the delta and vega at each Asian observation date (314 sensitivities).
```julia
using ForwardDiff; using AQFED; using AQFED.Basket
spot = 100.0; r = 0.09; q = 0.0; σ = 0.5
τ=3.0; strike = 100.0
n = Int(τ * 52)
weights = fill(1.0/(n+1), n + 1)
vols = fill(σ,n+1)
obsTimes = collect(0:n) .* (τ/n)
forward = spot .* exp.((r - q) .* obsTimes)
discountFactor = exp(-r * τ)

p = DeelstraBasketPricer(1, 3)
x = vcat(forward, vols)
f = function(x::AbstractArray)  
	forward =  x[1:n+1]
	vols = x[n+2:2*n+2]
	tvar = vols.^2 .* obsTimes
	priceAsianFixedStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
end
ForwardDiff.gradient(f, x)
```
29 ms for a single price, and 3s for all 314 sensitivities.
BackwardDiff is also supported (not for the lower bound approximation), but the code is not optimized for it, and it is not faster as a consequence.

## Discrete Cash Dividends via Baskets
The Basket approximation may be used to price European options under the [piecewise-lognormal model](https://github.com/jherekhealy/AQFED.jl/tree/master/src/pln) (also known as spot model) for a single underlying paying multiple dividends. Below is the example of Vellekoop and Nieuwenhuis (2006) with 7 dividends.

```julia
using AQFED.TermStructure
spot = 100.0; strike=70.0; r = 0.06; q = 0.0; σ = 0.25
τd = 0.5; τ=7.0
x = [strike, spot, τ, σ, r, τd]
p = DeelstraBasketPricer(1, 3)
f = function(x)
	τ=x[3]; r=x[5];τd=x[6]; Basket.priceEuropean(p, true, x[1], x[2]*exp(x[3]*x[5]),x[4]^2*x[3],x[3],exp(-x[3]*x[5]),[CapitalizedDividend(Dividend(6.0, τd, τd, false, false), exp((τ - τd) * r)),
		CapitalizedDividend(Dividend(6.5, τd + 1, τd + 1, false, false), exp((τ - τd - 1) * r)),
		CapitalizedDividend(Dividend(7.0, τd + 2, τd + 2, false, false), exp((τ - τd - 2) * r)),
		CapitalizedDividend(Dividend(7.5, τd + 3, τd + 3, false, false), exp((τ - τd - 3) * r)),
		CapitalizedDividend(Dividend(8.0, τd + 4, τd + 4, false, false), exp((τ - τd - 4) * r)),
		CapitalizedDividend(Dividend(8.0, τd + 5, τd + 5, false, false), exp((τ - τd - 5) * r)),
		CapitalizedDividend(Dividend(8.0, τd + 6, τd + 6, false, false), exp((τ - τd - 6) * r))])
end
ForwardDiff.gradient(f, x)
```
The output with `DeelstraBasketPricer` is 26.08099127059646  (the reference value is 26.08). It takes 2.6 ms for a single price and  3.5 ms for all 6 sensitivities.
The output with `DeelstraLBBasketPricer` is 26.069554947778418, in 0.05 ms and it takes 0.06 ms to compute the 6 sensitivities.


## References
Deeltra, G. Diallo, I and Vanmaele, M (2010) [Moment matching approximation of Asian basket option prices](https://www.sciencedirect.com/science/article/pii/S0377042709002106)

Ju, N. (2002) [Pricing Asian and basket options via Taylor expansion](https://www.academia.edu/download/4686930/jujcf02.pdf)

Healy, J. (2021) [The Pricing of Vanilla Options with Cash Dividends as a Classic Vanilla Basket Option Problem](https://arxiv.org/abs/2106.12971)

Vellekoop, M.H. and Nieuwenhuis, J.W. (2006) [Efficient Pricing of Derivatives on Assets with Discrete Dividends](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.486.9053&rep=rep1&type=pdf)
