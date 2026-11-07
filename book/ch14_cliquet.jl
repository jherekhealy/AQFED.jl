# Chapter 14: cliquets under the Heston model. Quasi Monte-Carlo convergence against the time-step, accuracy of the
# discretization schemes over the (ρ, σ) plane, calibration of κ. Former study in test/cliquet_mc_tests.jl
# (book/legacy/cliquet_mc_study.jl).
# The grid of the time-steps (gridpoints_uniform_vs_ramp, chapter 6) is also generated here.

using AQFED.MonteCarlo, AQFED.Random, AQFED.TermStructure, CharFuncPricing, Random123, Statistics, StatsBase, Printf
import AQFED.TermStructure: HestonModel, InterpolatedLogDiscountFactorCurve
include(joinpath(@__DIR__, "..", "test", "market_data.jl"))

#drift and discount curves from the forwards and the discount factors of the market data
function marketCurves(ts, dfs, forwards)
    logdf = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, log.(dfs)))
    logForward = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, -log.(forwards)))
    return InterpolatedLogDiscountFactorCurve(logForward), InterpolatedLogDiscountFactorCurve(logdf)
end

#5 years cliquet with yearly observations after a 3 months lag, Heston parameters with a strongly violated Feller condition
function cliquet5y()
    spot, ts, dfs, forwards, _, _ = makeMarketData20260512()
    driftCurve, discountCurve = marketCurves(ts, dfs, forwards)
    model = HestonModel(HestonParams(0.0550, 5.0, 0.0527, -0.688, 3.237), driftCurve, discountCurve)
    lag = 3.0 / 12
    payoff = MonteCarlo.CliquetOption(0.098, 0.0, 1e8, 0.0, 0.0547 * 5,
        MonteCarlo.PeriodicCashFlow([lag, 1.0 + lag, 2.0 + lag, 3.0 + lag, 4.0 + lag, 5.0 + lag], 5.0 + lag, 1, 0.0), 0.0)
    return (spot=spot, model=model, payoff=payoff, specTimes=MonteCarlo.specificTimes(payoff))
end

## randomized quasi Monte-Carlo

simulatePrice(simulate, args...) = MonteCarlo.last_stat(only(simulate(args...)))

#price and path standard error with a Sobol sequence Owen-scrambled by a Philox generator of the given seed
function rqmcRun(simulate, model, spot, payoff, start, n, config, seed::UInt64)
    rng = Philox4x((UInt32(seed % UInt32), UInt32(seed >> 32)))
    seq = ScrambledSobolSeq(MonteCarlo.ndims(model, config), n, Owen(31, ScramblingRngAdapter(rng)))
    return simulatePrice(simulate, seq, model, spot, deepcopy(payoff), start, n, config)
end

"""
    rqmcConvergence(simulate, timesteps; configFactory, mrep, n, ...)

Mean and standard error of `mrep` independent RQMC estimates with `n` paths, for each number of time-steps per year.
"""
function rqmcConvergence(simulate, timesteps; model, spot, payoff, start=0, n, configFactory, mrep=128, refPrice, masterSeed)
    master = Philox4x((UInt32(masterSeed % UInt32), UInt32(masterSeed >> 32)))
    out = DataFrame(timestep=Int[], nsteps=Int[], price=Float64[], error=Float64[], stderr=Float64[], pathstderr=Float64[])
    for timestep in timesteps
        config = configFactory(1.0 / timestep)
        seeds = rand(master, UInt64, mrep)
        results = Vector{Tuple{Float64,Float64}}(undef, mrep)
        Threads.@threads for j in 1:mrep
            results[j] = rqmcRun(simulate, model, spot, payoff, start, n, config, seeds[j])
        end
        prices = first.(results)
        price = mean(prices)
        push!(out, (timestep, length(config.genTimes) - 1, price, abs(price - refPrice), std(prices) / sqrt(mrep), mean(last.(results))))
    end
    return out
end

@compute "cliquet_5y_rqmc" begin
    c = cliquet5y()
    #reference with DVSS2X, 1M paths and daily steps
    config = MonteCarlo.GridConfig(c.model, c.specTimes, 1.0 / 365, MonteCarlo.DVSS2X(withBB=false), mindt=0.5 / 365, tstop=0.0)
    seq = ScrambledSobolSeq(MonteCarlo.ndims(c.model, config), 1024 * 1024, OwenFaureTezuka(31, ScramblingRngAdapter(AQFED.Random.Well1024a())))
    refPrice, _ = simulatePrice(MonteCarlo.simulateDVSS2XIter, seq, c.model, c.spot, c.payoff, 0, 1024 * 1024, config)
    timesteps = [4, 8, 12, 16, 24, 32, 52, 64, 96, 128]
    schemes = [("DVSS2X", MonteCarlo.simulateDVSS2XIter, MonteCarlo.DVSS2X(withBB=false), 0x1111111111111111),
        ("QE", MonteCarlo.simulateQEIter, MonteCarlo.QE(withBB=false, withMartingale=true), 0x2222222222222222)]
    grids = [("almost uniform", 0.0), ("ramp to 1Y", 1.0)]
    results = DataFrame()
    for (scheme, simulate, method, seed) in schemes, (grid, tstop) in grids
        df = rqmcConvergence(simulate, timesteps; model=c.model, spot=c.spot, payoff=c.payoff, n=16 * 1024, refPrice=refPrice, masterSeed=seed,
            configFactory=dt -> MonteCarlo.GridConfig(c.model, c.specTimes, dt, method, mindt=0.5 / 365, tstop=tstop))
        df[!, :scheme] .= scheme
        df[!, :grid] .= grid
        append!(results, df)
    end
    results[!, :refPrice] .= refPrice
    writetable("cliquet_5y_rqmc.csv", results)
end

@figure "cliquet_5y_error_vs_nsteps_dvss_qe" begin
    df = readcsv("cliquet_5y_rqmc.csv")
    fig = bookFigure(size=(800, 300))
    xs = sort(unique(df.nsteps))[1:2:end]
    ax = bookAxis(fig[1, 1], xlabel="number of steps", ylabel="absolute error", xscale=log10, yscale=log10, xticks=(xs, string.(xs)))
    markers = Dict("almost uniform" => :circle, "ramp to 1Y" => :utriangle)
    for (i, scheme) in enumerate(unique(df.scheme)), grid in unique(df.grid)
        d = sort(df[(df.scheme.==scheme).&(df.grid.==grid), :], :nsteps)
        scatterlines!(ax, d.nsteps, d.error, color=PALETTE[i], marker=markers[grid], label="$scheme, $grid")
    end
    Legend(fig[1, 2], ax, framevisible=false)
    fig
end

#grid points of the DVSS2X scheme with 6 steps per year, almost uniform or ramping up to one year (chapter 6)
@figure "gridpoints_uniform_vs_ramp" begin
    c = cliquet5y()
    method = MonteCarlo.DVSS2X(withBB=false)
    uniform = MonteCarlo.GridConfig(c.model, c.specTimes, 1.0 / 6, method, mindt=0.5 / 365, tstop=0.0).genTimes
    ramp = MonteCarlo.GridConfig(c.model, c.specTimes, 1.0 / 6, method, mindt=0.5 / 365, tstop=1.0).genTimes
    fig = bookFigure(size=(800, 200))
    ax = bookAxis(fig[1, 1], xlabel="time", title="Sampling points: uniform vs linear ramp", yticks=([1.0, 0.0], ["uniform", "linear ramp"]), ygridvisible=false)
    scatter!(ax, uniform, ones(length(uniform)), markersize=6)
    scatter!(ax, ramp, zeros(length(ramp)), markersize=6)
    ylims!(ax, -0.5, 1.5)
    fig
end

## accuracy of the schemes on a monthly cliquet over the (ρ, σ) plane

#1 year monthly cliquets for 10 local caps, against counterparty prices (market data of June 2, 2026)
function monthlyCliquets()
    spot, ts, dfs, _, forwards, strikes, vols = makeMarketData20260602()
    refValues = [0.89, 1.30, 1.77, 2.28, 2.80, 3.32, 3.83, 4.77, 5.58, 6.26] ./ 100
    driftCurve, discountCurve = marketCurves(ts, dfs, forwards)
    caps = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00, 3.50, 4.0] ./ 100
    obsTimes = collect(range(0.0, 1.0, length=13))
    payoff = MonteCarlo.ListPayoff([MonteCarlo.CliquetOption(cap, -1.0, 1e100, 0.0, 0.0, MonteCarlo.PeriodicCashFlow(obsTimes, obsTimes[end], 1, 0.0), 0.0) for cap in caps])
    reduction = VolatilityModels.makeVarianceSwapReductionExplicit(ts, forwards, strikes, vols, isLinearExtrapolation=false, tte=1.0)
    return (spot=spot, refValues=refValues, driftCurve=driftCurve, discountCurve=discountCurve, payoff=payoff,
        specTimes=MonteCarlo.specificTimes(payoff), v0=reduction.v0, θ=reduction.θ)
end

const CLIQUET_SCHEMES = Dict(
    "qe" => (dt=1 / 64, method=() -> MonteCarlo.QE(withBB=true, withMartingale=true), title="QE"),
    "dvss2x" => (dt=1 / 64, method=() -> MonteCarlo.DVSS2X(withBB=true), title="DVSS2X"),
    "alfonsi2" => (dt=1 / 64, method=() -> MonteCarlo.Alfonsi2(withBB=true), title="Alfonsi2"),
    "fulltrunc" => (dt=1 / 256, method=() -> MonteCarlo.FullTruncation(true), title="Full Truncation"))

#RMSE of the cliquet prices against the reference values with κ=7, for each (ρ, σ) of the grid
function cliquetRMSESurface(c, scheme, n; rhos, sigmas, κ=7.0)
    s = CLIQUET_SCHEMES[scheme]
    out = DataFrame(scheme=String[], n=Int[], rho=Float64[], sigma=Float64[], rmse=Float64[])
    for ρ in rhos, σ in sigmas
        model = HestonModel(HestonParams(c.v0, κ, c.θ, ρ, σ), c.driftCurve, c.discountCurve)
        config = MonteCarlo.GridConfig(model, c.specTimes, s.dt, s.method(), mindt=0.5 / 365, tstop=0.0, maxdt=s.dt)
        seq = ScrambledSobolSeq(MonteCarlo.ndims(model, config), n, Owen(31, ScramblingRngAdapter(AQFED.Random.Chacha8SIMD(UInt32))))
        values, _ = MonteCarlo.simulateIterParallel(seq, model, c.spot, c.payoff, 0, n, config)
        push!(out, (scheme, n, ρ, σ, StatsBase.rmsd(values, c.refValues)))
    end
    return out
end

@compute "cliquet_rmse_surfaces" begin
    c = monthlyCliquets()
    rhos, sigmas = range(-1.0, -0.8, length=101), range(1.0, 1.5, length=101)
    df = vcat([cliquetRMSESurface(c, scheme, n, rhos=rhos, sigmas=sigmas) for (scheme, n) in
               (("qe", 256), ("qe", 1024), ("dvss2x", 256), ("dvss2x", 1024), ("alfonsi2", 256), ("fulltrunc", 256))]...)
    writetable("cliquet_rmse_surfaces.csv", df)
end

#log10 of the RMSE as a 3D surface and as a heatmap
function rmseSurfaceFigure(scheme, n)
    df = readcsv("cliquet_rmse_surfaces.csv")
    df = df[(df.scheme.==scheme).&(df.n.==n), :]
    rhos, sigmas = sort(unique(df.rho)), sort(unique(df.sigma))
    z = [log10(df[(df.rho.==ρ).&(df.sigma.==σ), :rmse][1]) for ρ in rhos, σ in sigmas]
    fig = bookFigure(size=(1200, 500))
    surface3D!(fig[1, 1], rhos, sigmas, z, xlabel="ρ", ylabel="σ", zlabel="log10(RMSE)", title="$(CLIQUET_SCHEMES[scheme].title), n = $n")
    ax2 = bookAxis(fig[1, 2], xlabel="ρ", ylabel="σ")
    hm = heatmap!(ax2, rhos, sigmas, z, colormap=SURFMAP)
    hm.rasterize = 3
    Colorbar(fig[1, 3], hm)
    return fig
end
@figure "qe_256_surface_heatmap" rmseSurfaceFigure("qe", 256)
@figure "qe_1024_surface_heatmap" rmseSurfaceFigure("qe", 1024)
@figure "dvss2x_256_surface_heatmap" rmseSurfaceFigure("dvss2x", 256)
@figure "alfonsi2_256_surface_heatmap" rmseSurfaceFigure("alfonsi2", 256)
@figure "fulltrunc_256_surface_heatmap" rmseSurfaceFigure("fulltrunc", 256)

## calibration of κ: profile of the RMSE (the profiles were computed by the κ grid study of book/legacy/cliquet_mc_study.jl)

@figure "qe_kappa_profile_compare" begin
    nRef = 10
    series = [("qe_profile_1024.csv", "NM, n = 1024", :solid, :circle), ("qe_profile_4096.csv", "NM, n = 4096", :solid, :rect),
        ("qe_profile_512_4096.csv", "NM warm-start (512 → 4096)", :solid, :utriangle), ("qe_profile_1024_de.csv", "DE, n = 1024", :dash, :diamond),
        ("qe_profile_4096_de.csv", "DE, n = 4096", :dash, :rtriangle)]
    fig = bookFigure(size=(850, 360))
    ax = bookAxis(fig[1, 1], xlabel="κ", ylabel="RMSE", yscale=log10, xgridvisible=false, xminorticksvisible=true, yminorticksvisible=true)
    for (i, (file, label, style, marker)) in enumerate(series)
        df = readcsv(file)
        rmse = sqrt.(df.rmse ./ nRef)
        lines!(ax, df.kappa, rmse, color=PALETTE[i], linewidth=2.0, linestyle=style)
        idx = 1:max(1, length(df.kappa) ÷ 22):length(df.kappa)
        scatter!(ax, df.kappa[idx], rmse[idx], color=PALETTE[i], marker=marker, markersize=11, strokecolor=:white, strokewidth=0.8, label=label)
    end
    Legend(fig[1, 2], ax, framevisible=false)
    fig
end
