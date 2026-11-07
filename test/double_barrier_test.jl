using AQFED, Test
import AQFED.Black: DoubleBarrierEngine, DoubleBarrierPayoff, calculate, calculateDoubleTouch, calculateDoubleNoTouch,
    AnalyticBarrierEngine, BarrierPayoff, calculateOneTouch, blackScholesFormula, UpOut, DownOut

@testset "DoubleBarrier" begin
    spot, K, σ, r, q, T = 100.0, 100.0, 0.25, 0.1, 0.05, 0.25
    db = DoubleBarrierEngine(spot, σ^2 * T, exp(-(r - q) * T), exp(-r * T), exp(-r * T))
    #double knock-out calls, reference values of the goquant test
    for (down, up, ref) in zip([50.0, 75, 90, 95], [200.0, 130, 115, 105], [5.5288, 4.3806, 1.1200, 4.8e-4])
        @test isapprox(calculate(db, DoubleBarrierPayoff(true, K, down, up, 0.0, false)), ref, atol=1e-4)
    end
    #single barrier limits
    sb = AnalyticBarrierEngine(spot, σ^2 * T, exp(-(r - q) * T), exp(-r * T), exp(-r * T))
    for isCall in (true, false)
        @test isapprox(calculate(db, DoubleBarrierPayoff(isCall, K, 1e-3, 120.0, 0.0, false)), calculate(sb, BarrierPayoff(isCall, K, 120.0, 0.0, UpOut)), atol=1e-12)
        @test isapprox(calculate(db, DoubleBarrierPayoff(isCall, K, 85.0, 1e5, 0.0, false)), calculate(sb, BarrierPayoff(isCall, K, 85.0, 0.0, DownOut)), atol=1e-12)
    end
    @test isapprox(calculateDoubleTouch(db, 1e-3, 120.0, 0.0, 1.0), calculateOneTouch(sb, false, 120.0, 1.0), atol=1e-12)
    @test isapprox(calculateDoubleTouch(db, 85.0, 1e5, 1.0, 0.0), calculateOneTouch(sb, true, 85.0, 1.0), atol=1e-12)
    #in-out parity and rebates: with zero rates the rebate paid at hit is the rebate paid at maturity
    db0 = DoubleBarrierEngine(spot, σ^2 * T, 1.0, 1.0, 1.0)
    for isCall in (true, false)
        vanilla = blackScholesFormula(isCall, K, spot, σ^2 * T, 1.0, 1.0)
        out = calculate(db0, DoubleBarrierPayoff(isCall, K, 85.0, 120.0, 2.0, false))
        in = calculate(db0, DoubleBarrierPayoff(isCall, K, 85.0, 120.0, 2.0, true))
        @test isapprox(out + in, vanilla + 2.0, atol=1e-12)
    end
    @test isapprox(calculateDoubleTouch(db0, 85.0, 120.0, 1.0, 1.0) + calculateDoubleNoTouch(db0, 85.0, 120.0, 1.0), 1.0, atol=1e-12)
    #Monte Carlo with the Brownian bridge exit probabilities (2000 steps, 2·10⁶ paths, stderr < 2e-3): call, put, touch up, touch down, no touch
    T = 0.5
    db = DoubleBarrierEngine(spot, σ^2 * T, exp(-(r - q) * T), exp(-r * T), exp(-r * T))
    values = [calculate(db, DoubleBarrierPayoff(true, K, 85.0, 120.0, 0.0, false)), calculate(db, DoubleBarrierPayoff(false, K, 85.0, 120.0, 0.0, false)),
        calculateDoubleTouch(db, 85.0, 120.0, 0.0, 1.0), calculateDoubleTouch(db, 85.0, 120.0, 1.0, 0.0), calculateDoubleNoTouch(db, 85.0, 120.0, 1.0)]
    mcValues = [1.29265, 0.82777, 0.30639, 0.32966, 0.33054]
    @test isapprox(values, mcValues, atol=8e-3)
end
