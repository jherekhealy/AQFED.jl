# Chapter 7: arbitrage-free interpolation of option prices. B-spline collocation, LVG, convex rational spline (Schaback),
# arbitrage-free cubic spline (Fengler), RBF, Andreasen-Huge, lognormal mixtures, on the market smiles of test/data/smiles.
# The calibrations of each dataset are run by `@compute "smile_<dataset>"`, which writes book/data/smile_<dataset>.csv;
# the figures only read this file.

include(joinpath(@__DIR__, "smile_methods.jl"))

function writeSmileCurves(d::SmileData, models; kwargs...)
    df = smileCurves(d, models; kwargs...)
    writetable("smile_$(d.name).csv", df)
    return df
end

## SPX500 options of 1 week, March 24, 2017 as of March 16, 2017

@compute "smile_spxw170324_170316" begin
    d = loadSmile("spxw170324_170316")
    prices, wv = Collocation.weightedPrices(true, d.strikes, d.vols, d.weights, d.forward, 1.0, d.tte, vegaFloor=1e-8)
    strikesf, pricesf = Collocation.filterConvexPrices(d.strikes, prices, wv, d.forward, tol=1e-6)
    upper = d.forward * exp(logmoneyness(d)[end] * 3)
    y = logmoneyness(d)
    rbf, rbfVariance = rbfModel(d, d.weights, knots=range(y[1], stop=y[end], length=8))
    #Fengler on the prices of the RBF interpolation
    pricesRBF, weightsRBF = Collocation.weightedPrices(true, d.strikes, sqrt.(rbfVariance.(y)), d.weights, d.forward, 1.0, d.tte)
    w1 = ones(length(prices))
    models = [
        "ExpBSpline" => expBSplineModel(d, prices, wv),
        "CubicBSpline" => cubicBSplineModel(d, prices, wv),
        "LVG-Equidistributed" => quadraticLVGModel(d, prices, wv, useVol=false, location="Equidistributed", size=10, L=d.strikes[1], U=d.strikes[end]),
        "LVG-MidXX" => quadraticLVGModel(d, prices, wv, useVol=false, location="Mid-XX", size=0, L=d.strikes[1], U=d.strikes[end]),
        "Schaback-0" => schabackModel(schabackExact(d, strikesf, pricesf, upper=upper)),
        "Schaback-1" => schabackModel(schabackFit(d, d.strikes, prices, wv, upper=upper, penalty=1.0)),
        "RBF" => rbf,
        "Fengler-RBF-200" => fenglerModel(d, pricesRBF, weightsRBF, λ=200.0),
        [string("Fengler-", λ) => fenglerModel(d, prices, wv, λ=λ) for λ in (1e-10, 100.0, 1000.0, 10000.0)]...,
        "LVG-AH" => quadraticLVGModel(d, prices, w1, otm=true, useVol=true, location="Equidistributed", optimizer="LM-ALG", L=d.strikes[1] / 2, U=d.strikes[end] * 2, isC3=true, size=10),
        "AH-101" => andreasenHugeModel(d, prices, w1, discreteSize=101),
        "AH-1000" => andreasenHugeModel(d, prices, w1, discreteSize=1000),
        "AH-101-Lagrange" => andreasenHugeLagrangeModel(d, prices, w1, discreteSize=101),
        "AH-1000-Lagrange" => andreasenHugeLagrangeModel(d, prices, w1, discreteSize=1000),
        "LVG-Uniform4" => quadraticLVGModel(d, prices, wv, useVol=false, location="Uniform-C", size=4, L=d.strikes[1], U=d.strikes[end]),
        "LVG-Equidistributed4" => quadraticLVGModel(d, prices, wv, useVol=true, location="Equidistributed", size=4, L=d.strikes[1], U=d.strikes[end]),
    ]
    writeSmileCurves(d, models)
end

const SPXW = "spxw170324_170316"
const BSPL_LABELS = Dict("ExpBSpline" => "Exp B-spline", "CubicBSpline" => "Cubic B-spline")
const FENGLER_SPXW = ["Fengler-1.0e-10", "Fengler-100.0", "Fengler-1000.0", "Fengler-10000.0"]
fenglerLabels(names) = Dict(n => "Fengler λ=" * replace(n, r"^Fengler-(RBF-)?" => "") for n in names)

@figure "spxw170324_170316_vol_bspl" smileVolFigure(SPXW, ["ExpBSpline", "CubicBSpline"], labels=BSPL_LABELS, size=(480, 320))
@figure "spxw170324_170316_dens_bspl" smileDensityFigure(SPXW, ["ExpBSpline", "CubicBSpline"], labels=BSPL_LABELS, size=(480, 320), logscale=true, legendposition=:lt)
@figure "spxw170324_170316_vol_schaback" smileVolFigure(SPXW, ["Schaback-0", "Schaback-1"], labels=Dict("Schaback-0" => "Schaback λ=0", "Schaback-1" => "Schaback λ=1"))
@figure "spxw170324_170316_dens_schaback" smileDensityFigure(SPXW, ["Schaback-0", "Schaback-1"], labels=Dict("Schaback-0" => "Schaback λ=0", "Schaback-1" => "Schaback λ=1"), logscale=true, legendposition=:lt)
@figure "spxw170324_170316_vol_lvgq" smileVolFigure(SPXW, ["LVG-Equidistributed"], labels=Dict("LVG-Equidistributed" => "Quadratic LVG"))
@figure "spxw170324_170316_dens_lvgq" smileDensityFigure(SPXW, ["LVG-Equidistributed", "LVG-MidXX"],
    labels=Dict("LVG-MidXX" => "Mid-XX n=147", "LVG-Equidistributed" => "Equidistributed n=10"), logscale=true, legendposition=:lt)
@figure "spxw170324_170316_vol_fengler_rbf" smileVolFigure(SPXW, ["RBF", "Fengler-RBF-200"], labels=fenglerLabels(["Fengler-RBF-200"]))
@figure "spxw170324_170316_dens_fengler_rbf" smileDensityFigure(SPXW, ["RBF", "Fengler-RBF-200"], labels=fenglerLabels(["Fengler-RBF-200"]), logscale=true, ylims=(1e-8, 0.02))
@figure "spxw170324_170316_vol_fengler" smileVolFigure(SPXW, FENGLER_SPXW, labels=fenglerLabels(FENGLER_SPXW), size=(800, 380), referenceAlpha=0.25)
@figure "spxw170324_170316_dens_fengler" smileDensityFigure(SPXW, FENGLER_SPXW, labels=fenglerLabels(FENGLER_SPXW), ylims=(1e-8, 0.02), legendposition=:lt)
@figure "spxw170324_170316_dens_fengler_log" smileDensityFigure(SPXW, FENGLER_SPXW, labels=fenglerLabels(FENGLER_SPXW), logscale=true, ylims=(1e-8, 0.02), legendposition=:lt)
const LVG4_LABELS = Dict("LVG-Uniform4" => "Uniform 4 points", "LVG-Equidistributed4" => "Equidistributed 4 points", "LVG-Equidistributed" => "Equidistributed 10 points")
@figure "spxw170324_170316_vol_lvgq4" smileVolFigure(SPXW, ["LVG-Uniform4", "LVG-Equidistributed4"], labels=LVG4_LABELS)
@figure "spxw170324_170316_dens_lvgq4" smileDensityFigure(SPXW, ["LVG-Uniform4", "LVG-Equidistributed4", "LVG-Equidistributed"], labels=LVG4_LABELS, logscale=true, legendposition=:lt)
const AH_LAGRANGE_LABELS = Dict("AH-101-Lagrange" => "Andreasen-Huge l=101", "AH-1000-Lagrange" => "Andreasen-Huge l=1000", "LVG-AH" => "LVG")
@figure "vol_spw_1m_050218ah_d100" smileVolFigure(SPXW, ["AH-101-Lagrange", "LVG-AH"], labels=AH_LAGRANGE_LABELS, reference=false, ylims=(5, 50), xlims=(-0.3, 0.12))
@figure "density_spw_1m_050218ah_d" smileDensityFigure(SPXW, ["AH-101-Lagrange", "LVG-AH", "AH-1000-Lagrange"], labels=AH_LAGRANGE_LABELS, logscale=true, legendposition=:lt)
const AH_LABELS = Dict("AH-101" => "Andreasen-Huge l=101", "AH-1000" => "Andreasen-Huge l=1000", "LVG-AH" => "LVG")
@figure "density_spw_1m_050218ahspl_d" smileDensityFigure(SPXW, ["AH-101", "LVG-AH", "AH-1000"], labels=AH_LABELS, logscale=true, ylims=(1e-10, 2e-2), legendposition=:lt)
@figure "vol_spw_1m_050218ahspl_d100" smileVolFigure(SPXW, ["AH-101", "LVG-AH"], labels=AH_LABELS, reference=false, ylims=(5, 50), xlims=(-0.3, 0.12))

## Jaeckel examples, cases I and II

@compute "smile_jaeckel_case_i" begin
    d = loadSmile("jaeckel_case_i")
    prices, wv = Collocation.weightedPrices(true, d.strikes, d.vols, d.weights, d.forward, 1.0, d.tte, vegaFloor=1e-8)
    pp = PPInterpolation.CubicSplineNatural(logmoneyness(d), d.vols .^ 2)
    rbf, _ = rbfModel(d, d.weights, knots=logmoneyness(d))
    models = [
        "LVG-MidXX" => quadraticLVGModel(d, prices, wv, useVol=false, location="Mid-XX", size=0, L=d.strikes[1] / 2, U=d.strikes[end] * 2),
        "Schaback" => schabackModel(schabackExact(d, d.strikes, prices, upper=50.0)),
        "RBF" => rbf,
        "CubicSplineVariance" => varianceModel(d, y -> pp(y)),
    ]
    #the Fengler quadratic program has currently no solution on this example (Goldfarb-Idnani), and COSMO does not converge
    try
        #unit weights and a vega floor of 1e-5: the Goldfarb-Idnani quadratic program has no solution with the default floor
        p1, w1 = Collocation.weightedPrices(true, d.strikes, d.vols, ones(length(d.strikes)), d.forward, 1.0, d.tte, vegaFloor=1e-5)
        push!(models, "Fengler-1.0e-13" => fenglerModel(d, p1, w1, λ=1e-13, eps=1e-13))
    catch e
        @warn "Fengler spline not calibrated on Jaeckel case I" exception = e
    end
    writeSmileCurves(d, models, n=501)
end

@figure "jaeckel_case_i_lvgq_vol" smileVolFigure("jaeckel_case_i", ["LVG-MidXX"], labels=Dict("LVG-MidXX" => "Quadratic LVG"), size=(480, 320), ylims=(21, 25), xlims=(-0.1, 3.6))
@figure "jaeckel_case_i_lvgq_dens" smileDensityFigure("jaeckel_case_i", ["LVG-MidXX"], size=(480, 320), legend=false)
@figure "jaeckel_case_i_fengler_rbf" smileVolFigure("jaeckel_case_i", ["Fengler-1.0e-13", "Schaback", "RBF"], labels=fenglerLabels(["Fengler-1.0e-13"]),
    size=(640, 320), ylims=(21, 25), xlims=(-0.1, 3.1))

@figure "jaeckel_case_i_fengler_rbf_dens" begin
    curves = readcsv("smile_jaeckel_case_i.csv")
    fig = bookFigure(size=(800, 250))
    #colors as in jaeckel_case_i_fengler_rbf (Fengler, Schaback, RBF)
    panels = [("CubicSplineVariance", "Cubic spline", 4), ("RBF", "RBF", 3), ("Fengler-1.0e-13", "Fengler", 1), ("Schaback", "Schaback", 2)]
    for (i, (model, title, color)) in enumerate(panels)
        c = rows(curves, :Model, model)
        ax = bookAxis(fig[1, i], xlabel="Forward log-moneyness", ylabel=i == 1 ? "Probability density" : "", title=title)
        lines!(ax, c.Logmoneyness, c.Density, color=PALETTE[color])
        ylims!(ax, -0.05, 1.0)
    end
    fig
end

@compute "smile_jaeckel_case_ii" begin
    d = loadSmile("jaeckel_case_ii")
    prices, wv = Collocation.weightedPrices(true, d.strikes, d.vols, d.weights, d.forward, 1.0, d.tte, vegaFloor=1e-5)
    models = [
        "ExpBSpline" => expBSplineModel(d, prices, wv, size=0, rawFit=true),
        #Levenberg-Marquardt: GaussNewton stops in a poor local minimum with spurious density spikes (RMSE 0.108% vs 0.095%)
        "CubicBSpline" => cubicBSplineModel(d, prices, wv, size=0, rawFit=true, optimizerName="LM"),
        "LVG-LinearBlack" => linearBlackLVGModel(d, prices, wv),
        "LVG-MidXX" => quadraticLVGModel(d, prices, wv, useVol=false, location="Mid-XX", size=0, L=d.strikes[1], U=d.strikes[end]),
        jaeckelCaseIIMixtures(d)...,
    ]
    writeSmileCurves(d, models)
end

#lognormal mixtures (section "Mixture of Lognormal Distributions"): hypersphere (A), softmax (B) and fixed grid (C) calibrations
function jaeckelCaseIIMixtures(d::SmileData)
    prices, wv = Collocation.weightedPrices(true, d.strikes, d.vols, d.weights, d.forward, 1.0, d.tte, vegaFloor=1e-7)
    kernels = [
        "Mixture-A4" => VolatilityModels.calibrateLognormalMixture(d.tte, d.forward, d.strikes, prices, wv, useVol=false, size=4),
        "Mixture-A6" => VolatilityModels.calibrateLognormalMixture(d.tte, d.forward, d.strikes, prices, wv, useVol=false, size=6),
        "Mixture-B2" => VolatilityModels.calibrateLognormalMixtureSoftmax(d.tte, d.forward, d.strikes, prices, wv, useVol=false, size=2),
        "Mixture-B4" => VolatilityModels.calibrateLognormalMixtureSoftmax(d.tte, d.forward, d.strikes, prices, wv, useVol=false, size=4),
        "Mixture-B6" => VolatilityModels.calibrateLognormalMixtureSoftmax(d.tte, d.forward, d.strikes, prices, wv, useVol=false, size=6),
        "Mixture-C51" => first(VolatilityModels.calibrateLognormalMixtureGrid(d.tte, d.forward, d.strikes, prices, wv, size=51)),
        "Mixture-CS" => first(VolatilityModels.calibrateLognormalMixtureGrid(d.tte, d.forward, d.strikes, prices, wv, adaptive=true, smoothing=:auto)),
    ]
    #parameters, for tbl:mln_param_jaeckel
    params = vcat([DataFrame(Model=name, Component=eachindex(k.α), Alpha=k.α, Forward=k.x, Lambda=k.σ) for (name, k) in kernels]...)
    writetable("jaeckel_case_ii_mixture_params.csv", params)
    return [name => mixtureModel(k) for (name, k) in kernels]
end

const MIXTURE_II_LABELS = Dict("Mixture-A4" => "Mixture of 4", "Mixture-A6" => "Mixture of 6", "Mixture-B2" => "Mixture of 2", "Mixture-B4" => "Mixture of 4", "Mixture-B6" => "Mixture of 6",
    "Mixture-C51" => "Grid of 51, λ = Δx", "Mixture-CS" => "Adaptive grid, λ = σ(x)√T/4, smoothed")
#Option B (softmax), published figure and table tbl:mln_param_jaeckel
@figure "jaeckel_case_ii_mixture_vol" smileVolFigure("jaeckel_case_ii", ["Mixture-B4", "Mixture-B6"], labels=MIXTURE_II_LABELS, referenceAlpha=1.0)
@figure "jaeckel_case_ii_mixture_dens" smileDensityFigure("jaeckel_case_ii", ["Mixture-B4", "Mixture-B6"], labels=MIXTURE_II_LABELS, ylims=(0, 2))
#Option B (softmax) and option C (fixed grid, without and with smoothing)
@figure "jaeckel_case_ii_mixture_softmax_vol" smileVolFigure("jaeckel_case_ii", ["Mixture-B4", "Mixture-B6"], labels=MIXTURE_II_LABELS, referenceAlpha=1.0, legendposition=:ct)
@figure "jaeckel_case_ii_mixture_softmax_dens" smileDensityFigure("jaeckel_case_ii", ["Mixture-B4", "Mixture-B6"], labels=MIXTURE_II_LABELS, ylims=(0, 2))
@figure "jaeckel_case_ii_mixture_grid_vol" smileVolFigure("jaeckel_case_ii", ["Mixture-C51", "Mixture-CS"], labels=MIXTURE_II_LABELS, referenceAlpha=1.0, legendposition=:ct)
@figure "jaeckel_case_ii_mixture_grid_dens" smileDensityFigure("jaeckel_case_ii", ["Mixture-C51", "Mixture-CS"], labels=MIXTURE_II_LABELS, ylims=(0, 2))

@figure "jaeckel_case_ii_bspl_vol" smileVolFigure("jaeckel_case_ii", ["ExpBSpline", "CubicBSpline"], labels=BSPL_LABELS, size=(480, 320), legendposition=:ct, referenceAlpha=1.0)
@figure "jaeckel_case_ii_bspl_dens" smileDensityFigure("jaeckel_case_ii", ["ExpBSpline", "CubicBSpline"], labels=BSPL_LABELS, size=(480, 320))
@figure "jaeckel_case_ii_lvgq_vol" smileVolFigure("jaeckel_case_ii", ["LVG-MidXX"], labels=Dict("LVG-MidXX" => "Quadratic LVG"), size=(480, 320))
@figure "jaeckel_case_ii_lvgq_dens_log" smileDensityFigure("jaeckel_case_ii", ["LVG-MidXX", "LVG-LinearBlack"],
    labels=Dict("LVG-LinearBlack" => "Linear-Black", "LVG-MidXX" => "Quadratic"), size=(480, 320), logscale=true)

## AAPL options of 4 days, as of October 28, 2013

@compute "smile_aapl_20131028_4d" begin
    d = loadSmile("aapl_20131028_4d")
    w = d.weights ./ sqrt(sum(d.weights .^ 2))
    prices, wv = Collocation.weightedPrices(true, d.strikes, d.vols, w, d.forward, 1.0, d.tte, vegaFloor=1e-5)
    strikesf, pricesf = Collocation.filterConvexPrices(d.strikes, prices, wv, d.forward, tol=1e-6)
    upper = d.forward * exp(logmoneyness(d)[end] * 3)
    y = logmoneyness(d)
    rbf, rbfVariance = rbfModel(d, d.weights, knots=range(y[1], stop=y[end], length=8))
    pricesRBF = @. Black.blackScholesFormula(true, d.strikes, d.forward, rbfVariance(y) * d.tte, 1.0, 1.0)
    models = [
        "ExpBSpline" => expBSplineModel(d, prices, wv),
        "CubicBSpline" => cubicBSplineModel(d, prices, wv),
        "Schaback-1e-5" => schabackModel(schabackFit(d, strikesf, pricesf, wv[in(strikesf).(d.strikes)], upper=upper, penalty=1e-5)),
        "Schaback-8" => schabackModel(schaback8Knots(d, strikesf, pricesf, upper=upper)),
        [string("Fengler-", λ) => fenglerModel(d, prices, wv, λ=λ, eps=1e-13) for λ in (1e-8, 1e-6, 1e-5)]...,
        "RBF" => rbf,
        [string("Fengler-RBF-", λ) => fenglerModel(d, pricesRBF, wv, λ=λ, eps=1e-13) for λ in (1e-8, 1e-6)]...,
        "LVG-Equidistributed" => quadraticLVGModel(d, prices, wv, useVol=false, location="Equidistributed", size=10, L=d.strikes[1], U=d.strikes[end]),
    ]
    writeSmileCurves(d, models)
end

const AAPL = "aapl_20131028_4d"
const FENGLER_AAPL = ["Fengler-1.0e-8", "Fengler-1.0e-6", "Fengler-1.0e-5"]
const FENGLER_RBF_AAPL = ["Fengler-RBF-1.0e-8", "Fengler-RBF-1.0e-6"]
const SCHABACK_AAPL = Dict("Schaback-1e-5" => "Schaback λ=1e-5", "Schaback-8" => "Schaback on 8 knots")
@figure "aapl_20131028_vol_bspl" smileVolFigure(AAPL, ["ExpBSpline", "CubicBSpline"], labels=BSPL_LABELS, size=(480, 320))
@figure "aapl_20131028_dens_bspl" smileDensityFigure(AAPL, ["ExpBSpline", "CubicBSpline"], labels=BSPL_LABELS, size=(480, 320))
@figure "aapl_20131028_schaback" smileVolFigure(AAPL, ["Schaback-1e-5", "Schaback-8"], labels=SCHABACK_AAPL)
@figure "aapl_20131028_schaback_dens" smileDensityFigure(AAPL, ["Schaback-1e-5", "Schaback-8"], labels=SCHABACK_AAPL, legendposition=:cb)
@figure "aapl_20131028_fengler" smileVolFigure(AAPL, FENGLER_AAPL, labels=fenglerLabels(FENGLER_AAPL))
@figure "aapl_20131028_fengler_dens" smileDensityFigure(AAPL, FENGLER_AAPL, labels=fenglerLabels(FENGLER_AAPL))
@figure "aapl_20131028_rbf_fengler" smileVolFigure(AAPL, [FENGLER_RBF_AAPL..., "RBF"], labels=fenglerLabels(FENGLER_RBF_AAPL))
@figure "aapl_20131028_fengler_rbf_dens" smileDensityFigure(AAPL, [FENGLER_RBF_AAPL..., "RBF"], labels=fenglerLabels(FENGLER_RBF_AAPL))
@figure "aapl_20131028_lvgq_vol" smileVolFigure(AAPL, ["LVG-Equidistributed"], labels=Dict("LVG-Equidistributed" => "Quadratic LVG"), size=(480, 320))
@figure "aapl_20131028_lvgq_dens" smileDensityFigure(AAPL, ["LVG-Equidistributed"], size=(480, 320), legend=false)

## SPX500 options of 1 month, March 7, 2018 as of February 5, 2018

@compute "smile_spx500_20180205_1m" begin
    d = loadSmile("spx500_20180205_1m")
    w = d.weights ./ sqrt(sum(d.weights .^ 2))
    prices, wv = Collocation.weightedPrices(true, d.strikes, d.vols, w, d.forward, 1.0, d.tte, vegaFloor=1e-5)
    models = [
        "LVG-Equidistributed" => quadraticLVGModel(d, prices, wv, useVol=false, location="Equidistributed", size=10, L=d.strikes[1], U=d.strikes[end]),
        [string("Mixture-", n) => mixtureModel(VolatilityModels.calibrateLognormalMixture(d.tte, d.forward, d.strikes, prices, wv, useVol=false, size=n)) for n in (2, 4, 6)]...,
    ]
    writeSmileCurves(d, models)
end

const SPX1M = "spx500_20180205_1m"
const MIXTURE_LABELS = Dict("Mixture-$n" => "Mixture of $n" for n in (2, 4, 6))
@figure "vol_spw_1m_050218_lvgq" smileVolFigure(SPX1M, ["LVG-Equidistributed"], labels=Dict("LVG-Equidistributed" => "Quadratic LVG"))
@figure "density_spw_1m_050218_lvgq" smileDensityFigure(SPX1M, ["LVG-Equidistributed"], legend=false)
@figure "vol_spw_1m_050218_mln6" smileVolFigure(SPX1M, ["Mixture-2", "Mixture-4", "Mixture-6"], labels=MIXTURE_LABELS, size=(800, 320))
@figure "density_spw_1m_050218_mln6" smileDensityFigure(SPX1M, ["Mixture-2", "Mixture-4", "Mixture-6"], labels=MIXTURE_LABELS, size=(800, 320))

## TSLA options of 1 month, July 20, 2018 as of June 15, 2018

@compute "smile_tsla_20180615_1m" begin
    d = loadSmile("tsla_20180615_1m")
    w = d.weights ./ sqrt(sum(d.weights .^ 2))
    prices, wv = Collocation.weightedPrices(true, d.strikes, d.vols, w, d.forward, 1.0, d.tte, vegaFloor=1e-5)
    strikesf, pricesf = Collocation.filterConvexPrices(d.strikes, prices, wv, d.forward, tol=1e-6)
    upper = d.forward * exp(logmoneyness(d)[end] * 3)
    models = [
        [string("Fengler-", λ) => fenglerModel(d, prices, wv, λ=λ, eps=1e-13) for λ in (50.0, 200.0, 400.0, 1600.0)]...,
        "ExpBSpline" => expBSplineModel(d, prices, wv),
        "CubicBSpline" => cubicBSplineModel(d, prices, wv, extrapolationFactor=1.1, optimizerName="LeastSquaresOptim"),
        "LVG-Equidistributed" => quadraticLVGModel(d, prices, wv, useVol=false, location="Equidistributed", size=10, L=d.strikes[1], U=d.strikes[end]),
        "Schaback-1" => schabackModel(schabackFit(d, strikesf, pricesf, wv[in(strikesf).(d.strikes)], upper=upper, penalty=1.0)),
        "Schaback-8" => schabackModel(schaback8Knots(d, strikesf, pricesf, upper=upper)),
    ]
    writeSmileCurves(d, models, n=501)
end

const TSLA = "tsla_20180615_1m"
const FENGLER_TSLA = ["Fengler-50.0", "Fengler-200.0", "Fengler-400.0", "Fengler-1600.0"]
const SCHABACK_TSLA = Dict("Schaback-1" => "Schaback λ=1", "Schaback-8" => "Schaback on 8 knots")
@figure "tsla_180615_180720_fengler" smileVolFigure(TSLA, FENGLER_TSLA, labels=fenglerLabels(FENGLER_TSLA))
@figure "tsla_180615_180720_fengler_dens" smileDensityFigure(TSLA, FENGLER_TSLA, labels=fenglerLabels(FENGLER_TSLA))
@figure "tsla_180615_180720_vol_bspl" smileVolFigure(TSLA, ["ExpBSpline", "CubicBSpline"], labels=BSPL_LABELS, size=(480, 320))
@figure "tsla_180615_180720_dens_bspl" smileDensityFigure(TSLA, ["ExpBSpline", "CubicBSpline"], labels=BSPL_LABELS, size=(480, 320), legendposition=:lt)
@figure "tsla_180615_180720_vol_lvgq" smileVolFigure(TSLA, ["LVG-Equidistributed"], labels=Dict("LVG-Equidistributed" => "Quadratic LVG"), size=(480, 320))
@figure "tsla_180615_180720_dens_lvgq" smileDensityFigure(TSLA, ["LVG-Equidistributed"], size=(480, 320), legend=false)
@figure "tsla_180615_180720_schaback" smileVolFigure(TSLA, ["Schaback-1", "Schaback-8"], labels=SCHABACK_TSLA)
@figure "tsla_180615_180720_schaback_dens" smileDensityFigure(TSLA, ["Schaback-1", "Schaback-8"], labels=SCHABACK_TSLA, logscale=true, legendposition=:lt)

## Kahale surface: arbitrage-free cubic spline (Fengler) and LVG surfaces

#Gatheral local variance denominator with central finite differences of the total variance w(y)
function gatheralDenominator(w, y; h=1e-4)
    w0, wp, wm = w(y), w(y + h), w(y - h)
    dw = (wp - wm) / (2h)
    d2w = (wp - 2w0 + wm) / h^2
    return 1 - y / w0 * dw + dw^2 * (-1 / 4 - 1 / w0 + y^2 / w0^2) / 4 + d2w / 2
end

@compute "kahale_surfaces" begin
    s = loadKahaleSurface()
    forwards = s.spot .* exp.((s.r - s.q) .* s.expiries)
    n, m = length(s.expiries), length(s.strikes)
    prices, weights, strikesM = zeros(n, m), zeros(n, m), zeros(n, m)
    for (i, t) in enumerate(s.expiries)
        prices[i, :], weights[i, :] = Collocation.weightedPrices(true, s.strikes, s.vols[i, :], ones(m), forwards[i], 1.0, t)
        strikesM[i, :] = s.strikes
    end
    fengler = VolatilityModels.calibrateFenglerSurface(s.expiries, forwards, strikesM, prices, weights, λ=1e-2, solver="GI")
    #independent slices (Mid-XX knots of each expiry), the total variance interpolated linearly in time
    lvg = PDDE.LVGSurface(s.expiries, forwards, Vector{PDDE.QuadraticLVG}([PDDE.calibrateQuadraticLVG(t, 1.0, s.strikes ./ forwards[i], prices[i, :] ./ forwards[i], weights[i, :],
        model=PDDE.Quadratic(), location="Mid-XX", optimizer="LM-ALG") for (i, t) in enumerate(s.expiries)]))
    #arbitrage checks: smallest increase of the total variance between consecutive expiries, and smallest g of Gatheral
    #at the expiries and in between, on a wide range of log-moneyness
    yc = range(-1.0, 1.0, length=2001)
    minCal = minimum(PDDE.varianceByLogmoneyness(lvg, y, s.expiries[i+1]) * s.expiries[i+1] - PDDE.varianceByLogmoneyness(lvg, y, s.expiries[i]) * s.expiries[i] for i in 1:n-1 for y in yc)
    tc = sort(vcat(s.expiries, [(s.expiries[i] + s.expiries[i+1]) / 2 for i in 1:n-1]))
    minG = minimum(gatheralDenominator(y -> PDDE.varianceByLogmoneyness(lvg, y, t) * t, y) for t in tc for y in yc)
    rmse = sqrt(sum(abs2, [1e4 * (sqrt(PDDE.varianceByLogmoneyness(lvg, log(K / forwards[i]), t)) - s.vols[i, j]) for (i, t) in enumerate(s.expiries), (j, K) in enumerate(s.strikes)]) / (n * m))
    println("Kahale LVG independent slices: RMSE ", rmse, " bp, min calendar spread in total variance ", minCal, ", min g ", minG)
    writetable("kahale_lvg_checks.csv", DataFrame(RMSE_bp=rmse, MinCalendarTotalVariance=minCal, MinG=minG))
    k = range(s.strikes[1], stop=s.strikes[end], length=101)
    #total variance and density on the expiries of the quotes
    slices = DataFrame(Expiry=Float64[], Strike=Float64[], Logmoneyness=Float64[], FenglerTotalVariance=Float64[], FenglerDensity=Float64[], LVGTotalVariance=Float64[])
    for (i, t) in enumerate(s.expiries)
        y = log.(k ./ forwards[i])
        density = [ForwardDiff.derivative(x -> ForwardDiff.derivative(z -> VolatilityModels.price(fengler, log(z / forwards[i]), t), x), ki) for ki in k]
        append!(slices, DataFrame(Expiry=t, Strike=collect(k), Logmoneyness=y, FenglerTotalVariance=AQFED.TermStructure.varianceByLogmoneyness.(fengler, y, t) .* t,
            FenglerDensity=density, LVGTotalVariance=PDDE.varianceByLogmoneyness.(lvg, y, t) .* t))
    end
    writetable("kahale_slices.csv", slices)
    #implied and local volatilities of the LVG surface on a finer grid of expiries
    grid = DataFrame(Expiry=Float64[], Strike=Float64[], ImpliedVol=Float64[], LocalVol=Float64[])
    for t in range(s.expiries[1] / 2, stop=s.expiries[end], length=25)
        f = s.spot * exp((s.r - s.q) * t)
        w = y -> PDDE.varianceByLogmoneyness(lvg, y, t) * t
        h = 1e-4
        lv = [sqrt((PDDE.varianceByLogmoneyness(lvg, log(ki / f), t + h) * (t + h) - w(log(ki / f))) / (h * gatheralDenominator(w, log(ki / f)))) for ki in k]
        append!(grid, DataFrame(Expiry=t, Strike=collect(k), ImpliedVol=sqrt.(w.(log.(k ./ f)) ./ t), LocalVol=lv))
    end
    writetable("kahale_lvg_grid.csv", grid)
end

function kahaleTotalVarianceFigure(column)
    s = loadKahaleSurface()
    slices = readcsv("kahale_slices.csv")
    forwards = s.spot .* exp.((s.r - s.q) .* s.expiries)
    fig = bookFigure(size=(480, 340))
    ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Total variance")
    colors = cgrad(:viridis, length(s.expiries), categorical=true)
    for (i, t) in enumerate(s.expiries)
        scatter!(ax, log.(s.strikes ./ forwards[i]), s.vols[i, :] .^ 2 .* t, color=(:black, 0.6), markersize=4)
        c = slices[slices.Expiry.==t, :]
        lines!(ax, c.Logmoneyness, c[!, column], label=string(t), color=colors[i])
    end
    Legend(fig[1, 2], ax, "T", framevisible=false)
    return fig
end
@figure "kahale_fengler_totalvar" kahaleTotalVarianceFigure(:FenglerTotalVariance)
@figure "kahale_lvg_totalvar" kahaleTotalVarianceFigure(:LVGTotalVariance)

function kahaleSurfaceFigure(x, y, z, zlabel; azimuth, zrange=nothing)
    fig = bookFigure(size=(600, 600))
    surface3D!(fig[1, 1], x, y, z, xlabel="Expiry", ylabel="Strike", zlabel=zlabel, azimuth=azimuth, elevation=π / 6, zrange=zrange)
    return fig
end

function gridMatrix(df, column)
    xs, ys = sort(unique(df.Expiry)), sort(unique(df.Strike))
    z = [df[(df.Expiry.==x).&(df.Strike.==y), column][1] for x in xs, y in ys]
    return xs, ys, z
end

@figure "kahale_fengler_dens" begin
    xs, ys, z = gridMatrix(readcsv("kahale_slices.csv"), :FenglerDensity)
    kahaleSurfaceFigure(xs, ys, z, "Probability density", azimuth=-0.05π)
end
@figure "kahale_lvg_iv3d" (g = gridMatrix(readcsv("kahale_lvg_grid.csv"), :ImpliedVol); kahaleSurfaceFigure(g[1], g[2], g[3] .* 100, "Implied volatility in %", azimuth=1.25π))
@figure "kahale_lvg_iv3db" (g = gridMatrix(readcsv("kahale_lvg_grid.csv"), :ImpliedVol); kahaleSurfaceFigure(g[1], g[2], g[3] .* 100, "Implied volatility in %", azimuth=1.75π))
@figure "kahale_lvg_lv3d" (g = gridMatrix(readcsv("kahale_lvg_grid.csv"), :LocalVol); kahaleSurfaceFigure(g[1], g[2], g[3] .* 100, "Local volatility in %", zrange=(0, 45), azimuth=1.25π))
@figure "kahale_lvg_lv3db" (g = gridMatrix(readcsv("kahale_lvg_grid.csv"), :LocalVol); kahaleSurfaceFigure(g[1], g[2], g[3] .* 100, "Local volatility in %", zrange=(0, 45), azimuth=1.75π))


## AUD/NZD five Delta quotes of 1w, July 9, 2014 as of July 2, 2014 (tbl:audnzd_quotes), strikes from the broker convention

const AUDNZD = "audnzd_20140702_1w"
@compute "smile_audnzd_20140702_1w" begin
    d = loadSmile(AUDNZD)
    prices, wv = Collocation.weightedPrices(true, d.strikes, d.vols, d.weights, d.forward, 1.0, d.tte, vegaFloor=1e-8)
    models = [
        "Mixture-2" => mixtureModel(VolatilityModels.calibrateLognormalMixture(d.tte, d.forward, d.strikes, prices, wv, useVol=false, size=2)),
        "Mixture-3eq" => mixtureModel(VolatilityModels.calibrateLognormalMixtureFixedWeights(d.tte, d.forward, d.strikes, prices, wv, α=fill(1 / 3, 3))),
        "ExpBSpline" => expBSplineModel(d, prices, wv, size=0, rawFit=true),
        "ExpBSpline-Equidistant" => expBSplineModel(d, prices, wv, size=10),
        "CubicBSpline" => cubicBSplineModel(d, prices, wv, size=0, rawFit=true),
        "QuadraticBSpline" => quadraticBSplineModel(d, prices, wv, size=0, rawFit=true),
        "LVG-5" => quadraticLVGModel(d, prices, wv, useVol=false, location="Mid-XX", size=0, L=d.strikes[1] / 2, U=d.strikes[end] * 2),
    ]
    writeSmileCurves(d, models, lower=d.forward * exp(-0.033), upper=d.forward * exp(0.033))
end
const AUDNZD_LABELS = Dict("Mixture-2" => "Mixture of 2", "Mixture-3eq" => "Mixture of 3 with equal α", "ExpBSpline" => "Exp B-spline",
    "ExpBSpline-Equidistant" => "Exp B-spline equidistant", "CubicBSpline" => "Cubic B-spline", "QuadraticBSpline" => "Quadratic B-spline", "LVG-5" => "Quadratic LVG")
@figure "audnzd_mixture_vols" smileVolFigure(AUDNZD, ["Mixture-2", "Mixture-3eq"], labels=AUDNZD_LABELS, referenceAlpha=1.0, legendposition=:ct)
@figure "audnzd_mixture_dens" smileDensityFigure(AUDNZD, ["Mixture-2", "Mixture-3eq"], labels=AUDNZD_LABELS, legendposition=:lt)
#same color per model in the volatility and the density figures
const AUDNZD_BSPL_COLORS = Dict("ExpBSpline" => PALETTE[1], "CubicBSpline" => PALETTE[2], "ExpBSpline-Equidistant" => PALETTE[3], "QuadraticBSpline" => PALETTE[4])
@figure "audnzd_bspl_vol" smileVolFigure(AUDNZD, ["ExpBSpline", "CubicBSpline"], labels=AUDNZD_LABELS, referenceAlpha=1.0, legendposition=:ct, colors=AUDNZD_BSPL_COLORS)
@figure "audnzd_bspl_dens" smileDensityFigure(AUDNZD, ["ExpBSpline", "CubicBSpline", "ExpBSpline-Equidistant", "QuadraticBSpline"], labels=AUDNZD_LABELS, colors=AUDNZD_BSPL_COLORS, legendposition=:lt)
@figure "audnzd_lvgq_vol" smileVolFigure(AUDNZD, ["LVG-5"], labels=AUDNZD_LABELS, referenceAlpha=1.0, legendposition=:rb)
@figure "audnzd_lvgq_dens" smileDensityFigure(AUDNZD, ["LVG-5"], labels=AUDNZD_LABELS, legend=false)

## Cubic spline on Δ or on strikes for the 1w AUD/NZD quotes (test/vol_calibration_tests.jl, testset "deltavolraw"):
## natural cubic spline of the volatility against the call forward Δ (SplineDeltaSmile), or of the total variance against
## the forward log-moneyness (SplineSmile), and the implied probability densities (second derivative of the undiscounted
## call price in strike, by central finite differences).
@compute "audnzd_delta_spline" begin
    d = loadSmile(AUDNZD)
    smiles = ["Delta" => VolatilityModels.calibrateSmile(VolatilityModels.SplineDeltaSmile, d.strikes, d.vols, d.forward, d.tte),
        "Strike" => VolatilityModels.calibrateSmile(VolatilityModels.SplineSmile, d.strikes, d.vols, d.forward, d.tte)]
    call(smile, k) = AQFED.Black.blackScholesFormula(true, k, d.forward, smile(k)^2 * d.tte, 1.0, 1.0)
    out = DataFrame(Interpolation=String[], Logmoneyness=Float64[], Delta=Float64[], Vol=Float64[], Density=Float64[])
    for (name, smile) in smiles, y in range(-0.033, 0.033, length=661)
        k = d.forward * exp(y)
        vol = smile(k)
        h = 1e-4 * k
        density = (call(smile, k + h) - 2call(smile, k) + call(smile, k - h)) / h^2
        delta = VolatilityModels.deltaForStrike(VolatilityModels.CallForwardDelta(), k, vol, d.tte, d.forward, 1.0)
        push!(out, (name, y, delta, vol, density))
    end
    writetable("audnzd_delta_spline.csv", out)
end
@figure "audnzd_delta_spline_dens" begin
    df = readcsv("audnzd_delta_spline.csv")
    d = loadSmile(AUDNZD)
    quoteDelta = [VolatilityModels.deltaForStrike(VolatilityModels.CallForwardDelta(), k, v, d.tte, d.forward, 1.0) for (k, v) in zip(d.strikes, d.vols)]
    labels = Dict("Delta" => "Cubic spline on Δ", "Strike" => "Cubic spline on strikes")
    fig = bookFigure(size=(800, 600))
    ax1 = bookAxis(fig[1, 1], xlabel="Call Δ in %", ylabel="Volatility in %", xreversed=true)
    ax2 = bookAxis(fig[1, 2], xlabel="Forward log-moneyness", ylabel="Volatility in %")
    ax3 = bookAxis(fig[2, 1:2], xlabel="Forward log-moneyness", ylabel="Probability density")
    for (i, name) in enumerate(("Delta", "Strike"))
        c = df[df.Interpolation.==name, :]
        lines!(ax1, c.Delta .* 100, c.Vol .* 100, color=PALETTE[i])
        lines!(ax2, c.Logmoneyness, c.Vol .* 100, color=PALETTE[i])
        lines!(ax3, c.Logmoneyness, c.Density, color=PALETTE[i], label=labels[name])
    end
    referenceScatter!(ax1, quoteDelta .* 100, d.vols .* 100, dense=false, markersize=8)
    referenceScatter!(ax2, logmoneyness(d), d.vols .* 100, dense=false, markersize=8)
    xlims!(ax1, 100, 0)
    ylims!(ax1, 5, 8); ylims!(ax2, 5, 8)
    bookLegend!(ax3, position=:rt)
    fig
end
