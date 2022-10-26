# Monte-Carlo simulations

## Rough Bergomi Example
We reproduce Figure 7.1 of Bayer, Fritz and Gatheral (2015)

```julia
using AQFED; using Printf
model = AQFED.MonteCarlo.FlatRoughBergomi(0.235,0.1,0.4,-0.85)
expiries = LinRange(0.1, 2.5, 10)
up = 0.01
nSim = 1024 * 64 * 4
for tte in expiries     
    function priceRB(strike)
        payoff = AQFED.MonteCarlo.VanillaOption(true, strike, tte)
        specTimes = AQFED.MonteCarlo.specificTimes(payoff)
        dt = min(0.1,tte/128)
        nd = AQFED.MonteCarlo.ndims(model, specTimes, dt)
        rng =  ScrambledSobolSeq(nd, nSim, Owen(30, ScramblingRngAdapter(AQFED.Random.Blabla8())))
        #rng = AbstractRNGSeq(AQFED.Random.Well1024a(UInt32(20130129)),nd)
        price, serr = AQFED.MonteCarlo.simulate(rng, model, spot, payoff, 0, nSim, dt)
        vol = AQFED.Black.impliedVolatilitySRHalley(true, price, 1.0, strike, tte, 1.0, 0e-14, 64, AQFED.Black.Householder())
        return price, serr, vol
    end
priceUp,serrUp,volUp = priceRB(1.0+up)
priceDown,serrDown,volDown = priceRB(1.0-up)
@printf("| %.2f | %.4f | %.4f |\n", tte, (volUp-volDown)/(2*up),AQFED.MonteCarlo.atmSkew(model,tte) )
end

```
The output is:

|Expiry|MC IV Skew      |  Approx IV Skew    |
|----:|-----------:|----------:|
| 0.10 | -0.1963 | -0.1968 |
| 0.37 | -0.1154 | -0.1155 |
| 0.63 | -0.0922 | -0.0919 |
| 0.90 | -0.0799 | -0.0792 |
| 1.17 | -0.0718 | -0.0709 |
| 1.43 | -0.0659 | -0.0649 |
| 1.70 | -0.0613 | -0.0602 |
| 1.97 | -0.0577 | -0.0565 |
| 2.23 | -0.0548 | -0.0534 |
| 2.50 | -0.0522 | -0.0508 |


The example implementation is somewhat slow, but in practice, the covariance matrix square root may be cached.

## References
Bayer, C. Fritz, P., and Gatheral, J. (2015) [Pricing under rough volatility](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554754)

