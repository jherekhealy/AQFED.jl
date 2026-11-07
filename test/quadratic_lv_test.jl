using AQFED, Test
import AQFED.VolatilityModels: QuadraticLVParams, NormalizedQuadraticParams, andersenQuadraticParams, impliedVolatilityAB, impliedVolatilityLPP,
    guessQuadraticParamsAB, calibrateQuadraticLV, TimeChangeAveraging, VarianceMatchAveraging, ATMOptionMatchAveraging
import AQFED.Black

quadraticVol(p, K, f, τ) = (isCall = K > f; Black.impliedVolatility(isCall, AQFED.VolatilityModels.priceEuropean(p, isCall, K, f, τ), f, K, τ, 1.0))

@testset "QuadraticLV regimes" begin
    #put prices at strikes 60, 100, 140 (80, 100, 115 for the last two) for f=100, τ=1, by Crank-Nicolson on 16000 steps with absorption at 0
    f, τ = 100.0, 1.0
    cases = [(andersenQuadraticParams(f, 0.2, 0.5, 0.0), [60.0, 100, 140], [0.0798291, 7.9755161, 40.2946647]), #γ=0
        (andersenQuadraticParams(f, 0.2, 0.5, 10.0), [60.0, 100, 140], [0.4371375, 8.2494428, 40.8330753]), #complex roots
        (QuadraticLVParams(15.0, 0.0, 0.0005), [60.0, 100, 140], [0.0910395, 7.9888241, 40.316146]), #complex roots, Z₀ = π/6
        (QuadraticLVParams(5.0, 2sqrt(5.0 * 0.001), 0.001), [60.0, 100, 140], [0.3245063, 11.6260258, 42.4101854]), #double root
        (QuadraticLVParams(10.0, 0.15, -0.0003), [60.0, 100, 140], [0.166677, 8.7641242, 40.434278]), #l < f < u
        (QuadraticLVParams(3.0, -0.21, 0.003), [80.0, 100, 115], [0.0171809, 4.7856821, 16.2279655]), #l < u < f
        (QuadraticLVParams(-72.0, 0.96, -0.003), [80.0, 100, 115], [0.0552453, 2.3879013, 15.0000001])] #f < l < u
    for (p, strikes, refs) in cases, (K, ref) in zip(strikes, refs)
        put = AQFED.VolatilityModels.priceEuropean(p, false, K, f, τ)
        @test isapprox(put, ref, atol=3e-5)
        @test isapprox(AQFED.VolatilityModels.priceEuropean(p, true, K, f, τ) - put, f - K, atol=1e-10)
    end
    @test AQFED.VolatilityModels.priceEuropean(QuadraticLVParams(15.0, 0.0, 0.0005), false, 100.0, f, τ, 0.9) ≈ 0.9 * AQFED.VolatilityModels.priceEuropean(QuadraticLVParams(15.0, 0.0, 0.0005), false, 100.0, f, τ)
end

@testset "QuadraticLV Andersen paper" begin
    f, τ = 100.0, 1.0
    @test isapprox(quadraticVol(andersenQuadraticParams(f, 0.2, 0.5, 10.0), 180.0, f, τ) * 100, 27.76, atol=1e-2)
    strikes = collect(range(50.0, 180.0, length=21))
    for (s, tolAB) in ((0.0, 1e-5), (2.0, 1e-4), (5.0, 2e-4), (10.0, 5e-4))
        p = andersenQuadraticParams(f, 0.2, 0.5, s)
        vols = [quadraticVol(p, K, f, τ) for K in strikes]
        errAB = maximum(abs.([impliedVolatilityAB(p, K, f, τ) for K in strikes] .- vols))
        errLPP3 = maximum(abs.([impliedVolatilityLPP(p, 3, K, f, τ) for K in strikes] .- vols))
        @test errAB < tolAB
        @test errAB < errLPP3
    end
    #guess from the at-the-money derivatives of the AB expansion, then least-squares calibration
    p = andersenQuadraticParams(f, 0.2, 0.5, 5.0)
    h = 0.1
    v0, vu, vd = (impliedVolatilityAB(p, K, f, τ) for K in (f, f + h, f - h))
    guess = guessQuadraticParamsAB(f, τ, v0, (vu - vd) / 2h, (vu - 2v0 + vd) / h^2)
    @test isapprox([guess.α, guess.β, guess.γ], [p.α, p.β, p.γ], rtol=0.05)
    vols = [quadraticVol(p, K, f, τ) for K in strikes]
    pc = calibrateQuadraticLV(τ, f, strikes, vols)
    @test maximum(abs.([quadraticVol(pc, K, f, τ) for K in strikes] .- vols)) < 1e-6
    #capped inverse Vega weighted prices: same minimum on an exact fit
    pc = calibrateQuadraticLV(τ, f, strikes, vols, useVol=false)
    @test maximum(abs.([quadraticVol(pc, K, f, τ) for K in strikes] .- vols)) < 1e-6
end

@testset "QuadraticLV time-averaging" begin
    #two periods, the reference implied volatilities are obtained by Crank-Nicolson with the piecewise constant parameters
    params = QuadraticLVParams.([NormalizedQuadraticParams(0.1, 0.0, 0.0), NormalizedQuadraticParams(0.15, 1.0, 10.0)])
    expiries = [1.0, 2.0]
    strikes = [0.6, 0.8, 1.0, 1.3, 1.6]
    refVols = [0.17745, 0.14167, 0.13035, 0.14259, 0.16655]
    for (method, tol) in ((TimeChangeAveraging, 1e-3), (VarianceMatchAveraging, 3e-4), (ATMOptionMatchAveraging, 3e-4))
        vols = [Black.impliedVolatility(false, AQFED.VolatilityModels.priceEuropean(expiries, params, false, K, 1.0, 2.0; method=method), 1.0, K, 2.0, 1.0) for K in strikes]
        @test maximum(abs.(vols .- refVols)) < tol
    end
end

@testset "QuadraticLV forward PDE" begin
    QLV = AQFED.VolatilityModels
    #Andersen-Hutchings two periods, against the Crank-Nicolson reference of the averaging test
    params = QuadraticLVParams.([NormalizedQuadraticParams(0.1, 0.0, 0.0), NormalizedQuadraticParams(0.15, 1.0, 10.0)])
    pde = QLV.QuadraticLVForwardPDE([1.0, 2.0], params, 1.0, 2.0)
    vols = [Black.impliedVolatility(false, QLV.priceEuropean(pde, false, K), 1.0, K, 2.0, 1.0) for K in [0.6, 0.8, 1.0, 1.3, 1.6]]
    @test maximum(abs.(vols .- [0.17745, 0.14167, 0.13035, 0.14259, 0.16655])) < 1e-4
    #Kahale: the constant parameters calibration reaches the minimum (the calibration in (α, β, γ) stopped at an RMSE of 0.00521)
    strikes = 590 .* [0.85, 0.90, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4]
    expiries = [0.175, 0.425]
    forwards = 590 .* exp.((0.06 - 0.0262) .* expiries)
    vols = [[0.19, 0.168, 0.133, 0.113, 0.102, 0.097, 0.12, 0.142, 0.169, 0.2], [0.177, 0.155, 0.138, 0.125, 0.109, 0.103, 0.1, 0.114, 0.13, 0.15]]
    p = calibrateQuadraticLV(expiries[1], forwards[1], strikes, vols[1])
    rmse = sqrt(sum(abs2, [quadraticVol(p, K, forwards[1], expiries[1]) for K in strikes] .- vols[1]) / length(strikes))
    @test rmse < 0.005140
    #single period: the PDE matches the analytic price
    pde = QLV.QuadraticLVForwardPDE(expiries[1:1], [p], forwards[1], expiries[1], highestStrike=strikes[end])
    @test maximum(abs.([QLV.priceEuropean(pde, true, K) - QLV.priceEuropean(p, true, K, forwards[1], expiries[1]) for K in strikes])) < 0.01
    #term-structure (Table tbl:qlv_kahale of the book was calibrated on a finer grid)
    ts, rmses = QLV.calibrateQuadraticLVTermStructure(expiries, forwards, fill(strikes, 2), vols)
    @test ts[1].α ≈ 4950 rtol = 0.01
    @test isapprox(rmses, [0.00514, 0.00294], atol=2e-5)
    #with the price measure, the RMSE in volatilities is slightly larger
    ts, rmses = QLV.calibrateQuadraticLVTermStructure(expiries, forwards, fill(strikes, 2), vols, useVol=false)
    @test isapprox(rmses, [0.00539, 0.00302], atol=5e-5)
end
