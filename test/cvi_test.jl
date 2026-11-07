using Test, AQFED, AQFED.VolatilityModels, Statistics
import AQFED.TermStructure: SVISection, varianceByLogmoneyness

@testset "CVI-Vogt" begin
    #raw SVI of Axel Vogt (T=1) has butterfly arbitrage near k=1: CVI and RBF fits must remove it
    vogt = SVISection(-0.0410, 0.1331, 0.3060, 0.4153, 0.3586, 1.0, 1.0)
    ys = collect(range(-1.5, 1.5, length=31))
    vols = sqrt.(varianceByLogmoneyness.(vogt, ys))
    Δσ = fill(0.005, length(ys))
    kCheck = vcat(range(-4.5, 4.5, length=9001), [-45.0, 45.0])
    for kw in ((basis=:cvi, n=15, λ=0.01), (basis=:cvi, n=15, λ=0.01, solver=:clarabel), (basis=:rbf, m=12, wingFactor=2.0, rangeBandwidth=0.2, minBandwidth=2.0))
        surface, info = VolatilityModels.calibrateConvexVolatilitySurface([1.0], [1.0], [ys], [vols], [Δσ]; kw...)
        slice = surface.slices[1]
        rmse = sqrt(mean(abs2, sqrt.(slice.(ys)) .- vols))
        minG = minimum(k -> VolatilityModels.gatheralG(slice, k), kCheck)
        println(kw.basis, " rmse ", rmse, " min g ", minG, " iterations ", info.iterations)
        @test rmse < 0.003
        @test minG > -1e-5
    end
end

@testset "CVI-Kahale-calendar" begin
    r, q, spot = 0.06, 0.0262, 590.0
    strikes = spot .* [0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4]
    ttes = [0.175, 0.425, 0.695, 0.94, 1, 1.5, 2, 3, 4, 5]
    vols = [0.19 0.168 0.133 0.113 0.102 0.097 0.12 0.142 0.169 0.2;
        0.177 0.155 0.138 0.125 0.109 0.103 0.1 0.114 0.13 0.15;
        0.172 0.157 0.144 0.133 0.118 0.104 0.1 0.101 0.108 0.124;
        0.171 0.159 0.149 0.137 0.127 0.113 0.106 0.103 0.1 0.11;
        0.171 0.159 0.15 0.138 0.128 0.115 0.107 0.103 0.099 0.108;
        0.169 0.16 0.151 0.142 0.133 0.124 0.119 0.113 0.107 0.102;
        0.169 0.161 0.153 0.145 0.137 0.13 0.126 0.119 0.115 0.111;
        0.168 0.161 0.155 0.149 0.143 0.137 0.133 0.128 0.124 0.123;
        0.168 0.162 0.157 0.152 0.148 0.143 0.139 0.135 0.13 0.128;
        0.168 0.164 0.159 0.154 0.151 0.148 0.144 0.14 0.136 0.132]
    forwards = [spot * exp((r - q) * t) for t in ttes]
    ys = [log.(strikes ./ f) for f in forwards]
    volsV = [vols[i, :] for i in eachindex(ttes)]
    Δσ = [fill(0.005, length(strikes)) for _ in ttes]
    #baseline, and the recommended set-up: feasible SQP, lean grid with exchange, Clarabel
    for kw in ((nCalendar=200,), (method=:feasible, nGrid=40, marketPoints=false, pointsPerKnot=3, exchange=10, solver=:clarabel))
        surface, info = VolatilityModels.calibrateConvexVolatilitySurface(ttes, forwards, ys, volsV, Δσ; basis=:cvi, n=8, λ=0.01, kw...)
        sl = surface.slices
        minD = minimum(minimum(k -> sl[j+1](k) * sl[j+1].tte - sl[j](k) * sl[j].tte, range(-3, 3, length=6001)) for j = 1:length(sl)-1)
        minG = minimum(minimum(k -> VolatilityModels.gatheralG(sl[j], k), range(-3, 3, length=6001)) for j = 1:length(sl))
        rmse = sqrt(mean(abs2, vcat([sqrt.(sl[j].(ys[j])) .- volsV[j] for j in eachindex(ttes)]...)))
        println("Kahale CVI rmse ", rmse, " min g ", minG, " min calendar spread ", minD)
        @test rmse < 0.001
        @test minD > -1e-6
        @test minG > -1e-6
    end
end
