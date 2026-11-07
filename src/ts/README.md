# Term Structures

Discount and drift curves, dividends and a few model term structures used across the pricers.

## Discount curves (`curve.jl`)
- `Curve` (abstract), with `ConstantRateCurve`, `InterpolatedLogDiscountFactorCurve`, `SpreadCurve`, `OppositeCurve` and `PiecewiseConstantFunction`.
- `discountFactor`, `logDiscountFactor`, `forward` and `logForward` are defined for every curve type.

## Dividends (`dividend.jl`)
- `Dividend` and `CapitalizedDividend`, with `futureValue` to capitalise a cash dividend to a given date.
- `DividendDriftCurve` turns a dividend schedule into a drift curve.
- `DiscreteDividendMarket` bundles spot, rates and discrete dividends; `withDividendAmount` and `withGrowthRate` return modified copies, and `forward` gives the forward for a time to expiry.

## Heston term structures (`heston.jl`)
- `HaganHestonPiecewiseParams` and `makeHestonTSParams` / `makeHaganHestonPiecewiseParams` convert between Heston parameters and piecewise-constant term-structure parameters.
- `HestonModel` (single or double Heston) with discount, forward and log-forward functions.

## Other
- `fukasawa_interp.jl`: interpolation utilities.

Example
```julia
using AQFED.TermStructure
c = ConstantRateCurve(0.05)
discountFactor(c, 1.0)
```
