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
dividends = Vector{CapitalizedDividend}(undef, 1)
dividends[1] =
    CapitalizedDividend(Dividend(divAmount, ttd, ttd, false, false), exp((tte - ttd) * discountRate))
ll = LeFlochLehmanPLNPricer(3)
de = Basket.DeelstraBasketPricer()
dl = Basket.DeelstraLBBasketPricer()

refHHL = [43.24845580, 9.07480013, 1.06252880, 0.10473887]
for (i, strike) in enumerate(LinRange(50.0, 200.0, 4))
    price = PLN.priceEuropean(ll, isCall, strike,
        spot * exp(tte * discountRate), vol * vol * tte, tte,  discountFactor, dividends)
    @printf "|%5.1f|LL          |%12.8f|%11.2e|\n" strike price (price - refHHL[i])
    price = Basket.priceEuropean(de, isCall, strike,
        spot * exp(tte * discountRate), vol * vol * tte, tte,  discountFactor, dividends)
    @printf "|%5.1f|Deelstra    |%12.8f|%11.2e|\n" strike price (price - refHHL[i])
    price = Basket.priceEuropean(dl, isCall, strike,
        spot * exp(tte * discountRate), vol * vol * tte, tte,  discountFactor, dividends)
    @printf "|%5.1f|Deelstra-LB |%12.8f|%11.2e|\n" strike price (price - refHHL[i])
end
```
The output is:

|Strike|Method     | Price      |  Error    |
|----:|:-----------|-----------:|----------:|
| 50.0|LL          | 43.24845549|  -3.12e-07|
| 50.0|Deelstra    | 43.24845582|   1.84e-08|
| 50.0|Deelstra-LB | 43.24845099|  -4.81e-06|
|100.0|LL          |  9.07480027|   1.43e-07|
|100.0|Deelstra    |  9.07480014|   5.68e-09|
|100.0|Deelstra-LB |  9.07479136|  -8.77e-06|
|150.0|LL          |  1.06252874|  -6.49e-08|
|150.0|Deelstra    |  1.06252881|   5.30e-09|
|150.0|Deelstra-LB |  1.06252614|  -2.66e-06|
|200.0|LL          |  0.10473886|  -1.35e-08|
|200.0|Deelstra    |  0.10473887|  -3.13e-09|
|200.0|Deelstra-LB |  0.10473841|  -4.58e-07|

Above, we also display the results with the [Basket approximation approach](https://github.com/jherekhealy/AQFED.jl/tree/master/src/basket), which is very competitive.

## References
Etore, P. and Gobet, E. (2012) [Stochastic expansion for the pricing of call options with discrete dividends](https://hal.archives-ouvertes.fr/hal-00507787/file/dividende_v_final.pdf)

Le Floc'h, F. (2019) [More stochastic expansions for the pricing of vanilla options with cash dividends](https://arxiv.org/pdf/2106.12051)
