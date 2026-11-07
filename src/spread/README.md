# Spread Options

Pricing of spread options (difference of two underlyings).

## Pricers
- `PearsonPricer`: spread pricer, with `oneStepGridSpread` as its one-step grid calculation. Used through `priceVanillaSpread(pricer, isCall, strike, discountFactor, spot, ...)`.
- `priceAsianSpread` prices Asian-style spread options from forwards and total variances.
- `black_integral.jl` gives numerical integration for spread options under Black-Scholes.
- `margrabe0` is the Margrabe exchange-option formula, the zero-strike case, used as a reference.
- `inner` and `oneStepGridSpread` are the internal building blocks.

![Spread option price against strike](../../resources/images/spread_price_vs_strike.png)

*Call price of the spread `F₁ - F₂ - K` for two assets with volatilities 20% and 30%, spot and forward 100, maturity 1 year, for three correlations. Price falls with the strike and rises as correlation falls.*

Inputs are the two spots, volatilities, correlation `rho`, rates and dividend yields `q1`, `q2`, plus the maturity.
