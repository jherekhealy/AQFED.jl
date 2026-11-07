# Asian Options

Arithmetic-average Asian options, fixed and floating strike, including basket and spread variants.

## Fixed strike
- `priceAsianFixedStrike`: entry point for fixed-strike Asians. `VorstAsianExpansion` gives Vorst's closed-form approximation.
- `MonteCarloByPathEngine`: path-based Monte Carlo reference, usable for any model that can simulate the average.
- `priceAsianBasketFixedStrike` and `priceAsianSpread`: basket and spread Asian variants.

## Floating strike
- `priceAsianFloatingStrike`, including the spread variant `priceAsianFloatingStrikeWithSpread` (and its `D` version).
- `priceAsianBasketSpread`: basket spread Asian.

Some variants (`basketbridge.jl`, `strikespread.jl`) build on the [basket](../basket) pricers, such as `VorstGeometricExpansion`.
