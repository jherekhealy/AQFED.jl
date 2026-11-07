using Test, AQFED, AQFED.VolatilityModels
isdefined(@__MODULE__, :SmileData) || include("smile_data.jl")

#bootstrap of the piecewise-constant Hagan Heston parameters on SPX500 as of October 8, 2024 (first 7 expiries, 101 quotes
#interpolated for Δ_simple ∈ [10%, 90%], κ̃ = 0.5). The objective is flat in ρ for a small α̃: BFGS from the fixed guess
#(ρ, α̃) = (-0.5, 0.5) used to stop there for the expiry T = 0.95, and the following expiries inherited a worse fit.
@testset "HestonHaganBootstrap" begin
    s = loadSurface("spx500_20241008_surface")
    n = 7
    ts, forwards = s.expiries[1:n], s.forwards[1:n]
    strikesM = ones(n) * s.strikes'
    volsF, strikesF = VolatilityModels.filterVolsBySimpleDelta(ts, forwards, strikesM, s.vols[1:n, :], deltas=collect(range(0.1, 0.9, length=101)))
    pricesF, isCallF, weightsF = VolatilityModels.convertVolsToPricesOTMWeights(ts, forwards, strikesF, volsF, vegaFloor=1e-4, truncationDev=10.0)
    settings = (κ=0.5, lower=[1e-2, -0.99, 0.05], upper=[2.0, 0.5, 20.0], minimizer="BFGS", method="Cos", isGlobal=false,
        swapCurve=VolatilityModels.makeSwapCurveReplication(ts, forwards, strikesF, volsF))
    #per expiry bounds on the RMSE in volatility, about 10% above the values obtained
    for (name, calibrate, maxRmse) in (
        ("full", VolatilityModels.calibrateHestonHaganFromPrices, [0.17, 0.44, 0.63, 0.68, 0.62, 0.54, 0.36] ./ 100),
        ("reduced", VolatilityModels.calibrateHestonHaganFromPricesVS, [0.23, 0.61, 0.91, 1.18, 1.19, 1.13, 0.90] ./ 100))
        params, _ = calibrate(ts, forwards, strikesF, pricesF, isCallF, weightsF; settings...)
        volError, _ = VolatilityModels.estimateVolError(params, ts, forwards, strikesF, volsF)
        rmse = [sqrt(sum(abs2, volError[i, :]) / size(volError, 2)) for i = 1:n]
        println(name, " RMSE in vol % ", 100 .* rmse)
        for i = 1:n
            @test !(isapprox(params.ρ[i], -0.5, atol=1e-3) && isapprox(params.σ[i], 0.5, atol=1e-3))
            @test rmse[i] < maxRmse[i]
        end
    end
end
