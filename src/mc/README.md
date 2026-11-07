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

## Heston stochastic local volatility
`HestonLVModel` combines a Heston variance with a local volatility surface: d ln S = (r-q-L²V/2)dt + L(t,S)√V dW where the
leverage L² = σ²_LV(t,S) / E[V | S_t = S] makes the model reprice the vanilla options of the surface.
The conditional expectation is calibrated on the simulation grid by the particle method (Guyon and Henry-Labordère):
`calibrateLeverage` evolves particles with the same Andersen QE scheme (including the martingale correction, adjusted for the leverage)
as `simulateScheme`, and estimates E[V | ln S] before each step from the sorted particles with a `LocalComponentCalculator`:

- `NonConformCalculator`: piecewise linear least squares on bins of equal particle count, kept positive.
- `CompactSupportCalculator`: Nadaraya-Watson with a quartic kernel at particle quantiles, Huynh rational spline.
- `BinCalculator`, `BinSplineCalculator`: bin averages, piecewise constant or Huynh rational spline.
- `LocalLinearCalculator`: local linear kernel regression, linearly interpolated.

The local variance is the discrete Dupire formula `discreteLocalVarianceByLogmoneyness` (forward difference in time of the total
implied variance). The calibration is multithreaded (by blocks of particles, the result does not depend on the number of threads)
and `simulateSchemeParallel` prices by blocks of paths on all the threads.

We reproduce the example of Van der Stoep et al. (2014) with the SVI surface calibrated to their quotes:
```julia
using AQFED, Printf
import AQFED.MonteCarlo as MC
import AQFED.TermStructure: SVISection, VarianceSurfaceBySection, HestonModel, LocalVolatilityModel, varianceByLogmoneyness
import AQFED.Random: ScrambledSobolSeq, Owen, ScramblingRngAdapter, Blabla8
import AQFED.Black: blackScholesFormula

#SVI implied variance surface of the Van der Stoep et al. example
expiries = [0.5, 2.0, 5.0, 8.0, 10.0]
surface = VarianceSurfaceBySection([
        SVISection(0.02211047690170513, 0.08728514140099171, -0.7470846191571271, 0.01399832600143515, 0.05952343749999972, 0.5, 1.0),
        SVISection(0.007098341000109986, 0.08061891341040876, -0.43536114522531977, 0.14912190613260817, 0.07722949218749897, 2.0, 1.0),
        SVISection(6.729216331508232e-05, 0.05413804595661382, -0.3492177313652438, 0.3205809004034747, 0.1382236328125005, 5.0, 1.0),
        SVISection(0.009964200677573882, 0.03157899101219913, -0.5202476879856001, 0.2325780180852694, 0.16806298828124872, 8.0, 1.0),
        SVISection(0.0044141318643527916, 0.03190262256058419, -0.42036142370466967, 0.41444576466757904, 0.21499192810058354, 10.0, 1.0)], expiries)
heston = HestonModel(0.031329, 0.111927, 0.062762, -0.755929, 0.126743, 0.0, 0.0) #v0, κ, θ, ρ, σ, r, q
model = MC.HestonLVModel(heston, LocalVolatilityModel(surface, 0.0, 0.0))
spot, tte, nSim = 1.0, 5.0, 1024 * 64
config = MC.GridConfig(model, [tte], 1.0 / 100, MC.QE(withBB=true, withMartingale=false))
nd = MC.ndims(model, config)
#leverage calibrated once by the particle method, on its own quasi-random sequence
rngCalibration = ScrambledSobolSeq(nd, nSim, Owen(30, ScramblingRngAdapter(Blabla8(UInt64[1, 2, 3, 4]))))
leverage = MC.calibrateLeverage(rngCalibration, model, spot, config, nSim, calculator=MC.NonConformCalculator(gridSize=128))
for strike in (0.75, 1.0, 1.41)
    payoff = MC.VanillaOption(strike >= spot, strike, MC.BulletCashFlow(tte, tte, false, 0.0), 0.0)
    rng = ScrambledSobolSeq(nd, nSim, Owen(30, ScramblingRngAdapter(Blabla8())))
    price, stderr = MC.simulateSchemeParallel(rng, model, spot, payoff, 0, nSim, config, leverage=leverage)
    ref = blackScholesFormula(strike >= spot, strike, spot, varianceByLogmoneyness(surface, log(strike / spot), tte) * tte, 1.0, 1.0)
    @printf("| %.2f | %.5f | %.5f | %.5f |\n", strike, price, ref, stderr)
end
```
The output is:

| Strike | SLV price | Surface price | Standard error |
|----:|------:|------:|------:|
| 0.75 | 0.05578 | 0.05534 | 0.00057 |
| 1.00 | 0.13094 | 0.13046 | 0.00093 |
| 1.41 | 0.02015 | 0.01974 | 0.00062 |

On 65536 particles and 500 steps, the calibration takes around 3s on one thread and 2s on 14 threads,
and a price 2.4s on one thread and 0.6s on 14 threads (Intel Ultra 5 245KF).
The goquant implementation takes 13s to 15s for the calibration and a price.

### Schöbel-Zhu stochastic local volatility
`SchobelZhuLVModel` replaces the Heston variance by the Schöbel-Zhu volatility V (an Ornstein-Uhlenbeck process), with
L² = σ²_LV(t,S) / E[V² | S_t = S]. It is simulated by the `EAE` scheme: exact Ornstein-Uhlenbeck volatility, and the
log-asset with the Itô identity for ∫V dZ and trapezoidal time integrals. The calibration and pricing are the same generic
`calibrateLeverage` and `simulateSchemeParallel` as for Heston (any `StochasticLocalVolatilityModel` works), so that the example
above only changes in the model and the scheme:
```julia
import AQFED.TermStructure: SchobelZhuModel
sz = SchobelZhuModel(0.17700060676549295, 0.47004835147845592, 0.12560749278392161, -0.55573864712677212, 0.13330983125180917, 0.0, 0.0) #v0, κ, θ, ρ, σ, r, q
model = MC.SchobelZhuLVModel(sz, LocalVolatilityModel(surface, 0.0, 0.0))
config = MC.GridConfig(model, [tte], 1.0 / 100, MC.EAE(withBB=true))
```
| Strike | SLV price | Surface price | Standard error |
|----:|------:|------:|------:|
| 0.75 | 0.05544 | 0.05534 | 0.00057 |
| 1.00 | 0.13081 | 0.13046 | 0.00098 |
| 1.41 | 0.02017 | 0.01974 | 0.00069 |

The calibration takes around 5s on one thread and 2.4s on 14 threads, and a price 0.7s on 14 threads, against 12.7s for goquant.
The plain Schöbel-Zhu model is simulated by `simulateScheme(rng, sz, spot, payoff, start, nSim, config)` with the same `EAE` config.

## Discretely monitored exotics
`DiscreteBarrierOption` (up/down, in/out), `DiscreteOneTouch` (optionally smoothed by a `UniformKernel` or `EpanechnikovKernel`
over a given width, for stable finite difference Greeks) and `AutocallNote` (with an optional capital protection at maturity)
are payoffs on the asset values at their observation times, usable with any model of the vectorized engine.
The finite difference counterparts are backward scripts (`AQFED.FDM.BackwardScript`, see `autocallScript` and `asianScript`).

## References
Bayer, C. Fritz, P., and Gatheral, J. (2015) [Pricing under rough volatility](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554754)

Guyon, J. and Henry-Labordère, P. (2012) [Being particular about calibration](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032)

Van der Stoep, A. W., Grzelak, L. A., and Oosterlee, C. W. (2014) [The Heston stochastic-local volatility model: efficient Monte Carlo simulation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2278122)

