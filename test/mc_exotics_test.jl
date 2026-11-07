using AQFED, Test
import AQFED.MonteCarlo as MC
import AQFED.TermStructure: SVISection, VarianceSurfaceBySection, LocalVolatilityModel
import AQFED.Random: ScrambledSobolSeq, Owen, ScramblingRngAdapter, Blabla8
import AQFED.Black: AnalyticBarrierEngine, BarrierPayoff, calculate, calculateOneTouch, blackScholesFormula, DownIn, UpIn, DownOut, UpOut
import AQFED.Math: normcdf

#Black-Scholes as a flat local volatility model, priced by the vectorized local volatility engine
function flatBlackPricer(σ, r, q, spot; dt=1.0 / 52, n=1 << 16)
    model = LocalVolatilityModel(VarianceSurfaceBySection([SVISection(σ^2, 0.0, 0.0, 1.0, 0.0, 1.0, spot)], [1.0]), r, q)
    return payoff -> begin
        config = MC.GridConfig(model, MC.specificTimes(payoff), dt, MC.LogEuler(withBB=false))
        rng = ScrambledSobolSeq(MC.ndims(model, config), n, Owen(30, ScramblingRngAdapter(Blabla8())))
        MC.last_stat(MC.simulateScheme(rng, model, spot, payoff, 0, n, config))
    end
end

@testset "DiscreteBarrierOption" begin
    σ, r, q, spot, T, nObs = 0.25, 0.03, 0.01, 100.0, 1.0, 52
    price = flatBlackPricer(σ, r, q, spot)
    barrierTimes = collect(range(T / nObs, T, length=nObs))
    β = 0.5826 * σ * sqrt(T / nObs) #Broadie-Glasserman-Kou continuity correction of the barrier
    an = AnalyticBarrierEngine(spot, σ^2 * T, exp(-(r - q) * T), exp(-r * T), exp(-r * T))
    for (level, kinds) in ((130.0, (UpOut, UpIn)), (80.0, (DownOut, DownIn))), isCall in (true, false)
        vanilla = price(MC.VanillaOption(isCall, 100.0, MC.BulletCashFlow(T, T, false, 0.0), 0.0))[1]
        values = map(kinds) do kind
            p, stderr = price(MC.DiscreteBarrierOption(isCall, 100.0, level, kind, barrierTimes, T))
            shifted = kind in (UpOut, UpIn) ? level * exp(β) : level * exp(-β)
            #the continuity correction is an approximation of order O(Δt)
            @test abs(p - calculate(an, BarrierPayoff(isCall, 100.0, shifted, 0.0, kind))) < 3 * stderr + 0.02
            p
        end
        @test isapprox(sum(values), vanilla, rtol=1e-12) #in-out parity on each path
    end
end

@testset "DiscreteOneTouch" begin
    σ, spot, T, nObs = 0.25, 100.0, 1.0, 52
    price = flatBlackPricer(σ, 0.0, 0.0, spot) #with zero rates, the payment at maturity is equivalent to the payment at hit
    barrierTimes = collect(range(T / nObs, T, length=nObs))
    β = 0.5826 * σ * sqrt(T / nObs)
    an = AnalyticBarrierEngine(spot, σ^2 * T, 1.0, 1.0, 1.0)
    for (isDown, level) in ((true, 85.0), (false, 115.0))
        p, stderr = price(MC.DiscreteOneTouch(isDown, level, barrierTimes, 1.0))
        @test abs(p - calculateOneTouch(an, isDown, isDown ? level * exp(-β) : level * exp(β), 1.0)) < 3 * stderr
        #the smoothed one touch is close for a small width, and larger with the uniform kernel than with the Epanechnikov kernel
        pE, _ = price(MC.DiscreteOneTouch(isDown, level, barrierTimes, 1.0, width=2.0, kernel=MC.EpanechnikovKernel()))
        pU, _ = price(MC.DiscreteOneTouch(isDown, level, barrierTimes, 1.0, width=2.0))
        @test abs(pE - p) < 3 * stderr
        @test pU >= pE
    end
end

@testset "AutocallNote" begin
    σ, r, q, spot, T = 0.25, 0.03, 0.01, 100.0, 1.0
    price = flatBlackPricer(σ, r, q, spot)
    df, F = exp(-r * T), spot * exp((r - q) * T)
    #single observation with the capital protection at the autocall level: a digital call and a share below the level
    level, coupon = 1.05, 1.08
    d1 = (log(F / (level * spot)) + σ^2 * T / 2) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    p, stderr = price(MC.AutocallNote([T], [level], [coupon], [df]; reference=spot, protectionLevel=level))
    @test abs(p - df * (coupon * normcdf(d2) + F / spot * normcdf(-d1))) < 3 * stderr
    #quarterly observations: called at the first date, or never called
    times = [0.25, 0.5, 0.75, 1.0]
    coupons = [1.02, 1.04, 1.06, 1.08]
    @test isapprox(price(MC.AutocallNote(times, zeros(4), coupons, exp.(-r .* times); reference=spot))[1], coupons[1] * exp(-r * 0.25), rtol=1e-14)
    @test isapprox(price(MC.AutocallNote(times, fill(Inf, 4), coupons, exp.(-r .* times); reference=spot))[1], df, rtol=1e-14)
    #the price decreases with the autocall level
    pLow, _ = price(MC.AutocallNote(times, fill(1.0, 4), coupons, exp.(-r .* times); reference=spot, protectionLevel=0.7))
    pHigh, _ = price(MC.AutocallNote(times, fill(1.1, 4), coupons, exp.(-r .* times); reference=spot, protectionLevel=0.7))
    @test pLow > pHigh
    @test_throws DimensionMismatch MC.AutocallNote(times, fill(1.0, 3), coupons, exp.(-r .* times))
end
