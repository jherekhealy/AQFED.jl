## Approximations for Vanilla Basket Options
The following approximations of vanilla basket option prices under the Black-Scholes model are available:

- Curran lower bound approximation (using Deelstra et al. formulation): this is `DeelstraLBBasketPricer`.
- Curran shifted lognormal approximation as per Deelstra et al.: this is `DeelstraBasketPricer`. There are 9 possible variations, represented by δIndex=1,2,3 and fIndex=1,2,3.

Below is an example of a call option of strike 110 and maturity one year on a basket composed of 5 assets with weights [0.05, 0.15, 0.2, 0.25, 0.35].
```julia
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


## References
Deeltra, G. Diallo, I and Vanmaele, M (2010) [Moment matching approximation of Asian basket option prices](https://www.sciencedirect.com/science/article/pii/S0377042709002106)
