# Book: chapter 7, section "Mixture of Lognormal Distributions" (\label{sec:mln}), subsection "Spurious spikes".
# Figures and table generated here:
#   fig:jaeckel_case_ii_mixture          (Option A, hypersphere)   jaeckel_case_ii_mixture_{vol,dens}.pdf
#   fig:jaeckel_case_ii_mixture_softmax  (Option B, softmax)       jaeckel_case_ii_mixture_softmax_{vol,dens}.pdf
#   fig:jaeckel_case_ii_mixture_grid     (Option C, fixed grid QP) jaeckel_case_ii_mixture_grid_{vol,dens}.pdf
#   tbl:mln_param_jaeckel                                          data/jaeckel_case_ii_mixture_params.csv
# Data: case II of tbl:jackel_clamping_1.
#
# compute_* calibrates and writes CSV files to book/data, plot_* only reads them.
# Usage, from the test environment:
#   julia --project=test -e 'include("book/ch07_lnmixture.jl"); compute_jaeckel_case_ii_mixture(); plot_jaeckel_case_ii_mixture()'

using AQFED, AQFED.Black, AQFED.Collocation, AQFED.VolatilityModels
import AQFED.VolatilityModels: priceEuropean, density
using CSV, DataFrames, CairoMakie, StatsBase

const BOOK_DATA = joinpath(@__DIR__, "data")
const BOOK_FIGURES = expanduser("~/mypapers/eqd_book")

function jaeckelCaseII()
    strikes = [0.035123777453185, 0.049095433048156, 0.068624781300891, 0.095922580089594, 0.134078990076508, 0.18741338653678, 0.261963320525776, 0.366167980681693, 0.511823524787378, 0.715418426368358, 1.0, 1.39778339939642, 1.95379843162821, 2.73098701349666, 3.81732831143284, 5.33579814376678, 7.45829006788743, 10.4250740447762, 14.5719954372667, 20.3684933182917, 28.4707418310251]
    vols = [0.649712512502887, 0.629372247414191, 0.598339248024188, 0.560748840467284, 0.518685454812697, 0.473512707134552, 0.426434688827871, 0.378806875802102, 0.332366264644264, 0.289407658380454, 0.253751752243855, 0.235378088110653, 0.235343538571543, 0.260395028879884, 0.31735041252779, 0.368205175099723, 0.417582432865276, 0.46323707706565, 0.504386489988866, 0.539752566560924, 0.566370957381163]
    return (strikes=strikes, vols=vols, forward=1.0, tte=5.07222222222222)
end

#name => calibration function of (tte, forward, strikes, prices, weights)
function jaeckelCaseIIMixtureModels(tte, atmVol)
    λAtm = atmVol * sqrt(tte)
    return [
        "A4" => (t, f, k, p, w) -> calibrateLognormalMixture(t, f, k, p, w, useVol=false, size=4),
        "A6" => (t, f, k, p, w) -> calibrateLognormalMixture(t, f, k, p, w, useVol=false, size=6),
        "B4" => (t, f, k, p, w) -> calibrateLognormalMixtureSoftmax(t, f, k, p, w, useVol=false, size=4),
        "B6" => (t, f, k, p, w) -> calibrateLognormalMixtureSoftmax(t, f, k, p, w, useVol=false, size=6),
        #Option C as described in the book: bandwidth = grid spacing, no smoothing
        "C51" => (t, f, k, p, w) -> first(calibrateLognormalMixtureGrid(t, f, k, p, w, size=51)),
        #Option C with a bandwidth independent of the grid and a roughness penalty
        "C101S" => (t, f, k, p, w) -> first(calibrateLognormalMixtureGrid(t, f, k, p, w, size=101, bandwidth=0.25λAtm, smoothing=1e-8)),
    ]
end

function compute_jaeckel_case_ii_mixture(; dataDir=BOOK_DATA)
    d = jaeckelCaseII()
    w1 = ones(length(d.strikes))
    prices, wv = Collocation.weightedPrices(true, d.strikes, d.vols, w1, d.forward, 1.0, d.tte, vegaFloor=1e-7)
    w = w1 ./ sqrt(sum(w1 .^ 2))
    kFine = d.forward .* exp.(range(log(d.strikes[1]), stop=log(d.strikes[end]), length=1001))
    iv(kernel, k) = Black.impliedVolatility(k >= d.forward, priceEuropean(kernel, k >= d.forward, k), d.forward, k, d.tte, 1.0)
    curves = DataFrame(model=String[], logmoneyness=Float64[], vol=Float64[], density=Float64[])
    params = DataFrame(model=String[], component=Int[], alpha=Float64[], forward=Float64[], lambda=Float64[])
    summary = DataFrame(model=String[], rmse=Float64[], mean=Float64[], time=Float64[])
    for (name, calibrate) in jaeckelCaseIIMixtureModels(d.tte, d.vols[11])
        time = @elapsed kernel = calibrate(d.tte, d.forward, d.strikes, prices, wv)
        append!(curves, DataFrame(model=name, logmoneyness=log.(kFine ./ d.forward), vol=iv.(kernel, kFine), density=density.(kernel, kFine)))
        append!(params, DataFrame(model=name, component=eachindex(kernel.α), alpha=kernel.α, forward=kernel.x, lambda=kernel.σ))
        push!(summary, (name, StatsBase.L2dist(w .* iv.(kernel, d.strikes), w .* d.vols), sum(kernel.α .* kernel.x), time))
    end
    mkpath(dataDir)
    CSV.write(joinpath(dataDir, "jaeckel_case_ii_mixture_curves.csv"), curves)
    CSV.write(joinpath(dataDir, "jaeckel_case_ii_mixture_params.csv"), params)
    CSV.write(joinpath(dataDir, "jaeckel_case_ii_mixture_summary.csv"), summary)
    CSV.write(joinpath(dataDir, "jaeckel_case_ii_market.csv"), DataFrame(logmoneyness=log.(d.strikes ./ d.forward), vol=d.vols))
    return summary
end

function plotMixturePair(curves, market, models::Vector{Pair{String,String}}, prefix; outDir=BOOK_FIGURES)
    colors = Makie.wong_colors()
    figVol = Figure(size=(480, 380), fontsize=14)
    ax = Axis(figVol[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
    scatter!(ax, market.logmoneyness, market.vol .* 100, color=colors[1], markersize=7, label="Reference")
    figDens = Figure(size=(480, 380), fontsize=14)
    axd = Axis(figDens[1, 1], xlabel="Forward log-moneyness", ylabel="Probability density")
    for (i, (model, label)) in enumerate(models)
        c = curves[curves.model.==model, :]
        lines!(ax, c.logmoneyness, c.vol .* 100, color=colors[i+1], linewidth=1.5, label=label)
        lines!(axd, c.logmoneyness, c.density, color=colors[i+1], linewidth=1.5, label=label)
    end
    ylims!(axd, 0.0, 2.0)
    axislegend(ax, position=:ct, framevisible=false)
    axislegend(axd, position=:rt, framevisible=false)
    save(joinpath(outDir, prefix * "_vol.pdf"), figVol)
    save(joinpath(outDir, prefix * "_dens.pdf"), figDens)
    return figVol, figDens
end

function plot_jaeckel_case_ii_mixture(; dataDir=BOOK_DATA, outDir=BOOK_FIGURES)
    curves = CSV.read(joinpath(dataDir, "jaeckel_case_ii_mixture_curves.csv"), DataFrame)
    market = CSV.read(joinpath(dataDir, "jaeckel_case_ii_market.csv"), DataFrame)
    plotMixturePair(curves, market, ["B4" => "Mixture of 4", "B6" => "Mixture of 6"], "jaeckel_case_ii_mixture_softmax", outDir=outDir)
    plotMixturePair(curves, market, ["C51" => "Grid of 51, λ = Δx", "C101S" => "Grid of 101, smoothed"], "jaeckel_case_ii_mixture_grid", outDir=outDir)
    #fig:jaeckel_case_ii_mixture (Option A) is regenerated only on request, to keep the published figure:
    #plotMixturePair(curves, market, ["A4" => "Mixture of 4", "A6" => "Mixture of 6"], "jaeckel_case_ii_mixture", outDir=outDir)
end
