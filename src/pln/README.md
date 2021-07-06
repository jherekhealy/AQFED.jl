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
