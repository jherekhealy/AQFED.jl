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

## References
Deeltra, G. Diallo, I and Vanmaele, M (2010) [Moment matching approximation of Asian basket option prices](https://www.sciencedirect.com/science/article/pii/S0377042709002106)

Ju, N. (2002) [Pricing Asian and basket options via Taylor expansion](https://www.academia.edu/download/4686930/jujcf02.pdf)
