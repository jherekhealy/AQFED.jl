# Chapter 7, section "Mixture of Lognormal Distributions": comparison of the calibrations A (hypersphere), B (softmax, multi-start)
# and C (fixed grid QP) on the market smiles of test/data/smiles: accuracy, martingale error, number of modes of the density,
# calibration time. Variants: A with bandwidths boxed in [0.1, 10]σ_ATM√T, B with the bandwidth floor 0.15σ_ATM√T, C as in the book
# (51 kernels of bandwidth Δx), smoothed C (bandwidth σ_min√T/4, margin from the wing volatilities, automatic smoothing), and C on the
# volatility-adaptive grid (centers uniform in ∫dx/σ(x)√T): 51 kernels of overlap 4/3 (C51A), and λ = σ(x)√T/4 with smoothing (CSA).
# Note: ~/mypapers/lnmixture/lnmixture.tex.

using StatsBase, Random

const MIXTURE_DATASETS = ["jaeckel_case_i", "jaeckel_case_ii", "aapl_20131028_4d", "spx500_20180205_1m", "spx500_20180205_1m_arbfree", "tsla_20180615_1m", "spxw170324_170316"]

#the SPX500 1m quotes are not arbitrage-free on the right wing (the call price at 2860 is larger than at 2835):
#"<name>_arbfree" is the closest set of arbitrage-free prices, in the least squares sense with the calibration weights.
function comparisonSmile(name)
    endswith(name, "_arbfree") || return loadSmile(name)
    d = loadSmile(replace(name, "_arbfree" => ""))
    prices, wv = Collocation.weightedPrices(true, d.strikes, d.vols, d.weights, d.forward, 1.0, d.tte, vegaFloor=1e-7)
    _, filtered, _ = Collocation.filterConvexPrices(d.strikes, prices, wv, d.forward, forceConvexity=true)
    vols = [Black.impliedVolatility(true, p, d.forward, k, d.tte, 1.0) for (p, k) in zip(filtered, d.strikes)]
    return SmileData(name, d.tte, d.forward, d.strikes, vols, d.weights)
end

function mixtureCalibrations(d::SmileData, prices, wv)
    args = (d.tte, d.forward, d.strikes, prices, wv)
    return [
        "A4" => () -> VolatilityModels.calibrateLognormalMixture(args..., useVol=false, size=4),
        "A6" => () -> VolatilityModels.calibrateLognormalMixture(args..., useVol=false, size=6),
        "A6F" => () -> VolatilityModels.calibrateLognormalMixture(args..., useVol=false, size=6, transformName="closed"),
        "B2" => () -> VolatilityModels.calibrateLognormalMixtureSoftmax(args..., useVol=false, size=2),
        "B3" => () -> VolatilityModels.calibrateLognormalMixtureSoftmax(args..., useVol=false, size=3),
        "B4" => () -> VolatilityModels.calibrateLognormalMixtureSoftmax(args..., useVol=false, size=4),
        "B6" => () -> VolatilityModels.calibrateLognormalMixtureSoftmax(args..., useVol=false, size=6),
        "B8" => () -> VolatilityModels.calibrateLognormalMixtureSoftmax(args..., useVol=false, size=8),
        "B6F" => () -> VolatilityModels.calibrateLognormalMixtureSoftmax(args..., useVol=false, size=6, λMinFactor=0.15),
        "C51" => () -> first(VolatilityModels.calibrateLognormalMixtureGrid(args..., size=51, overlap=1.0)),
        "CS" => () -> first(VolatilityModels.calibrateLognormalMixtureGrid(args..., bandwidth=:auto, wingMargin=true, smoothing=:auto)),
        "C51A" => () -> first(VolatilityModels.calibrateLognormalMixtureGrid(args..., size=51, adaptive=true, overlap=4 / 3)),
        "CSA" => () -> first(VolatilityModels.calibrateLognormalMixtureGrid(args..., adaptive=true, overlap=4 / 3, smoothing=:auto)),
        #same kernels as CS and CSA, without roughness penalty
        "CS0" => () -> first(VolatilityModels.calibrateLognormalMixtureGrid(args..., bandwidth=:auto, wingMargin=true)),
        "CSA0" => () -> first(VolatilityModels.calibrateLognormalMixtureGrid(args..., adaptive=true, overlap=4 / 3)),
        #CSA with overlap=1 to illustrate the issue
        "CSA1" => () -> first(VolatilityModels.calibrateLognormalMixtureGrid(args..., adaptive=true, overlap=1.0, smoothing=:auto)),
    ]
end

#legend labels, with the names of the calibrations in the note ~/mypapers/lnmixture
const MIXTURE_COMPARISON_LABELS = Dict("A4" => "A4: 4 lognormals, hypersphere", "A6" => "A6: 6 lognormals, hypersphere",
    "B2" => "B2: 2 lognormals", "B3" => "B3: 3 lognormals", "B4" => "B4: 4 lognormals", "B6" => "B6: 6 lognormals",
    "B8" => "B8: 8 lognormals", "CSA1" => "CSA, γ = 1",
    "A6F" => "A6F: A6 with bandwidths in a box", "B6F" => "B6F: B6 with bandwidth floor 0.15",
    "C51" => "C51: 51 uniform kernels", "CS" => "CS: uniform kernels, penalty",
    "C51A" => "C51A: 51 adaptive kernels", "CSA" => "CSA: adaptive kernels, penalty",
    "CS0" => "CS0: uniform kernels, no penalty", "CSA0" => "CSA0: adaptive kernels, no penalty",
    "CSA1" => "CSA (γ=1): adaptive, penalty")

#number of local maxima of the density between the extreme strikes, number of those whose prominence exceeds 1% of the
#maximum density (the others are invisible wiggles), and minimum density
function densityModes(kernel, d::SmileData)
    x = exp.(range(log(d.strikes[1]), log(d.strikes[end]), length=2001))
    p = VolatilityModels.density.(Ref(kernel), x)
    maxima = filter(i -> p[i] > p[i-1] && p[i] >= p[i+1], 2:length(p)-1)
    function prominence(i)
        l = findlast(>(p[i]), view(p, 1:i-1))
        r = findfirst(>(p[i]), view(p, i+1:length(p)))
        return p[i] - max(minimum(view(p, something(l, 1):i)), minimum(view(p, i:(r === nothing ? length(p) : i + r))))
    end
    return length(maxima), count(i -> prominence(i) > 0.01 * maximum(p), maxima), minimum(p)
end

@compute "mixture_comparison" begin
    summary = DataFrame(Dataset=String[], Method=String[], Components=Int[], RMSE=Float64[], MartingaleError=Float64[], Modes=Int[], SignificantModes=Int[], MinDensity=Float64[], Time=Float64[])
    curves = DataFrame()
    for name in MIXTURE_DATASETS
        d = comparisonSmile(name)
        prices, wv = Collocation.weightedPrices(true, d.strikes, d.vols, d.weights, d.forward, 1.0, d.tte, vegaFloor=1e-7)
        w = d.weights ./ sqrt(sum(d.weights .^ 2))
        models = Pair{String,SmileModel}[]
        for (method, calibrate) in mixtureCalibrations(d, prices, wv)
            kernel = calibrate() #compilation
            time = minimum(@elapsed(calibrate()) for _ in 1:3)
            m = mixtureModel(kernel)
            rmse = StatsBase.L2dist(w .* impliedVol.(Ref(m), Ref(d), d.strikes), w .* d.vols)
            modes, significantModes, minDensity = densityModes(kernel, d)
            push!(summary, (name, method, count(>(1e-10), kernel.α), rmse, sum(kernel.α .* kernel.x) / d.forward - 1, modes, significantModes, minDensity, time))
            push!(models, method => m)
        end
        c = smileCurves(d, models, n=1001)
        c[!, :Dataset] .= name
        append!(curves, c)
    end
    writetable("mixture_comparison_summary.csv", summary)
    writetable("mixture_comparison_curves.csv", curves)
    summary
end

#cost and effect of the roughness penalty on the grids of CS and CSA: no penalty (one QP), the smoothing factor selected by
#smoothing=:auto when known in advance (one QP), and the automatic selection (21 QPs)
@compute "mixture_penalty" begin
    timed(f) = (f(); minimum(@elapsed(f()) for _ in 1:3))
    table = DataFrame(Dataset=String[], Grid=String[], Kernels=Int[], Smoothing=Float64[], RMSE0=Float64[], Modes0=Int[], Time0=Float64[],
        RMSE=Float64[], Modes=Int[], TimeFixed=Float64[], TimeAuto=Float64[])
    for name in MIXTURE_DATASETS
        d = comparisonSmile(name)
        prices, wv = Collocation.weightedPrices(true, d.strikes, d.vols, d.weights, d.forward, 1.0, d.tte, vegaFloor=1e-7)
        w = d.weights ./ sqrt(sum(d.weights .^ 2))
        rmse(k) = StatsBase.L2dist(w .* impliedVol.(Ref(mixtureModel(k)), Ref(d), d.strikes), w .* d.vols)
        for (grid, kw) in ("CS" => (bandwidth=:auto, wingMargin=true), "CSA" => (adaptive=true,))
            calibrate(smoothing) = VolatilityModels.calibrateLognormalMixtureGrid(d.tte, d.forward, d.strikes, prices, wv; smoothing=smoothing, kw...)
            k0, _, _ = calibrate(0.0)
            k, _, smoothing = calibrate(:auto)
            push!(table, (name, grid, length(k.x), smoothing, rmse(k0), densityModes(k0, d)[2], timed(() -> calibrate(0.0)),
                rmse(k), densityModes(k, d)[2], timed(() -> calibrate(smoothing)), timed(() -> calibrate(:auto))))
        end
    end
    writetable("mixture_penalty_summary.csv", table)
    table
end

#Number of components of option B, Levenberg-Marquardt (LeastSquaresOptim) against Gauss-Newton (GaussNewton): accuracy, density
#modes and spikes, local minima (spread of the fits from the different starting points) and stability (mean L1 distance between the
#density calibrated to the quotes and the densities calibrated to the quotes perturbed by an iid noise of 0.05 volatility point).
#The fixed-grid calibrations C51A and CSA are included for reference.
const MIXTURE_SIZES = [2, 3, 4, 5, 6, 8]
const MIXTURE_NOISE = 0.0005
const MIXTURE_NOISE_DRAWS = 10

function densityL1(k1, k2, d::SmileData)
    x = exp.(range(log(d.strikes[1]), log(d.strikes[end]), length=2001))
    p1, p2 = VolatilityModels.density.(Ref(k1), x), VolatilityModels.density.(Ref(k2), x)
    #trapezoidal rule in strike
    return sum((abs.(p1 .- p2)[1:end-1] .+ abs.(p1 .- p2)[2:end]) ./ 2 .* diff(x))
end

@compute "mixture_components" begin
    timed(f) = (f(); minimum(@elapsed(f()) for _ in 1:3))
    table = DataFrame(Dataset=String[], Method=String[], Components=Int[], Solver=String[], RMSE=Float64[], WorstStartRMSE=Float64[],
        Modes=Int[], SignificantModes=Int[], MaxDensity=Float64[], MinBandwidth=Float64[], Time=Float64[], Stability=Float64[])
    rng = Random.MersenneTwister(2026)
    noiseCurves = DataFrame()
    for name in MIXTURE_DATASETS
        d = comparisonSmile(name)
        noisy = [SmileData(name, d.tte, d.forward, d.strikes, d.vols .+ MIXTURE_NOISE .* randn(rng, length(d.vols)), d.weights) for _ in 1:MIXTURE_NOISE_DRAWS]
        quotes(dd) = Collocation.weightedPrices(true, dd.strikes, dd.vols, dd.weights, dd.forward, 1.0, dd.tte, vegaFloor=1e-7)
        w = d.weights ./ sqrt(sum(d.weights .^ 2))
        rmse(k) = StatsBase.L2dist(w .* impliedVol.(Ref(mixtureModel(k)), Ref(d), d.strikes), w .* d.vols)
        λAtm = d.vols[argmin(abs.(d.strikes .- d.forward))] * sqrt(d.tte)
        function record(method, components, solver, calibrate, starts)
            prices, wv = quotes(d)
            k = calibrate(prices, wv)
            if solver != "GN" && name in ("jaeckel_case_ii", "spx500_20180205_1m") && method in ("B4", "B8", "CSA")
                y = range(log(d.strikes[1] / d.forward), log(d.strikes[end] / d.forward), length=501)
                for (draw, kd) in enumerate(vcat([k], [calibrate(quotes(dn)...) for dn in noisy]))
                    append!(noiseCurves, DataFrame(Dataset=name, Method=method, Draw=draw - 1, Logmoneyness=collect(y),
                        Density=VolatilityModels.density.(Ref(kd), d.forward .* exp.(y))))
                end
            end
            worst = isempty(starts) ? rmse(k) : maximum(rmse(start(prices, wv)) for start in starts)
            modes, significant, _ = densityModes(k, d)
            x = exp.(range(log(d.strikes[1]), log(d.strikes[end]), length=2001))
            stability = sum(densityL1(k, calibrate(quotes(dn)...), d) for dn in noisy) / length(noisy)
            push!(table, (name, method, components, solver, rmse(k), worst, modes, significant,
                maximum(VolatilityModels.density.(Ref(k), x)) * d.forward, minimum(k.σ) / λAtm, timed(() -> calibrate(prices, wv)), stability))
        end
        for n in MIXTURE_SIZES, solver in (:LM, :GN)
            calibrate(prices, wv; kw...) = VolatilityModels.calibrateLognormalMixtureSoftmax(d.tte, d.forward, d.strikes, prices, wv, size=n, solver=solver; kw...)
            starts = [(p, v) -> calibrate(p, v, centerSpread=[s]) for s in (0.5, 1.0, 2.0)]
            record("B$n", n, string(solver), calibrate, starts)
        end
        record("C51A", 51, "GI", (p, v) -> first(VolatilityModels.calibrateLognormalMixtureGrid(d.tte, d.forward, d.strikes, p, v, size=51, adaptive=true)), [])
        record("CSA", 0, "GI", (p, v) -> first(VolatilityModels.calibrateLognormalMixtureGrid(d.tte, d.forward, d.strikes, p, v, adaptive=true, smoothing=:auto)), [])
        @info "mixture_components" name
    end
    writetable("mixture_components_summary.csv", table)
    writetable("mixture_components_noise_curves.csv", noiseCurves)
    table
end

#densities below LOG_FLOOR times the peak are not drawn in log scale: the deep tails would flatten the rest of the figure
const LOG_FLOOR = 1e-12

#ticks at whole decades between lo and hi, at most six of them, labelled 10ⁿ
function decadeTicks(lo, hi)
    e0, e1 = ceil(Int, log10(lo)), floor(Int, log10(hi))
    es = e0:max(1, ceil(Int, (e1 - e0) / 5)):e1
    return (10.0 .^ es, [rich("10", superscript(string(e))) for e in es])
end

function mixtureComparisonFigure(dataset, methods, quantity; logscale=false, labels=MIXTURE_COMPARISON_LABELS, legendposition=nothing)
    curves = readcsv("mixture_comparison_curves.csv")
    c = rows(curves[curves.Dataset.==dataset, :], :Model, methods...)
    fig = bookFigure(size=(480, 380))
    if quantity == :Vol
        d = comparisonSmile(dataset)
        ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
        referenceScatter!(ax, logmoneyness(d), d.vols .* 100)
        groupLines!(ax, c.Logmoneyness, c.Vol .* 100, c.Model, order=collect(methods), labels=labels)
    else
        if logscale
            peak = maximum(c.Density)
            c = c[c.Density.>LOG_FLOOR*peak, :]
        end
        ax = logscale ? bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Probability density", yscale=log10,
                yticks=decadeTicks(LOG_FLOOR * peak, 3peak)) :
             bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Probability density")
        groupLines!(ax, c.Logmoneyness, c.Density, c.Model, order=collect(methods), labels=labels)
        logscale && ylims!(ax, LOG_FLOOR * peak, 3peak)
    end
    position = something(legendposition, quantity == :Vol ? :ct : emptiestCorner(c.Logmoneyness, c.Density, logy=logscale))
    bookLegend!(ax, position=position)
    return fig
end

#methods retained (main text of the note) and alternatives not retained (appendix of the note, ~/mypapers/lnmixture)
const MIXTURE_MAIN = ["B6", "C51A", "CSA"]

for dataset in MIXTURE_DATASETS
    @eval @figure $("mixture_comparison_$(dataset)_vol") mixtureComparisonFigure($dataset, MIXTURE_MAIN, :Vol)
    @eval @figure $("mixture_comparison_$(dataset)_dens") mixtureComparisonFigure($dataset, MIXTURE_MAIN, :Density)
    #uniform grids of the book (51 kernels of bandwidth Δx) and of bandwidth σmin√T/4, against the adaptive grid
    @eval @figure $("mixture_comparison_$(dataset)_grid_dens") mixtureComparisonFigure($dataset, ["C51", "CS", "CSA"], :Density)
end

#the oscillations of the density in the wings of SPXW 1w are only visible in log scale
@figure "mixture_comparison_spxw170324_170316_dens_log" mixtureComparisonFigure("spxw170324_170316", MIXTURE_MAIN, :Density, logscale=true)
@figure "mixture_comparison_spxw170324_170316_grid_dens_log" mixtureComparisonFigure("spxw170324_170316", ["C51", "CS", "CSA"], :Density, logscale=true)

#kernel overlap γ=1 against γ=4/3 on the adaptive grid with penalty (CSA)
const GAMMA_LABELS = Dict("CSA1" => "CSA, γ = 1", "CSA" => "CSA, γ = 4/3")
for dataset in ["jaeckel_case_ii", "spxw170324_170316"]
    @eval @figure $("mixture_comparison_$(dataset)_gamma_vol") mixtureComparisonFigure($dataset, ["CSA1", "CSA"], :Vol, labels=GAMMA_LABELS)
    @eval @figure $("mixture_comparison_$(dataset)_gamma_dens_log") mixtureComparisonFigure($dataset, ["CSA1", "CSA"], :Density, logscale=true, labels=GAMMA_LABELS)
end

#number of components of option B
for dataset in ["jaeckel_case_ii", "spx500_20180205_1m"]
    @eval @figure $("mixture_comparison_$(dataset)_components_dens") mixtureComparisonFigure($dataset, ["B2", "B4", "B6", "B8"], :Density)
end

#stability: densities calibrated to the quotes perturbed by an iid noise of 0.05 volatility point (study "mixture_components")
function mixtureNoiseFigure(dataset, method)
    c = readcsv("mixture_components_noise_curves.csv")
    c = c[(c.Dataset.==dataset).&(c.Method.==method), :]
    fig = bookFigure(size=(480, 380))
    ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Probability density")
    for draw in sort(unique(c.Draw))
        r = c[c.Draw.==draw, :]
        lines!(ax, r.Logmoneyness, r.Density, color=draw == 0 ? :black : (PALETTE[1], 0.5), linewidth=draw == 0 ? 1.5 : 0.75,
        )
    end
    return fig
end
for dataset in ["jaeckel_case_ii", "spx500_20180205_1m"], method in ["B4", "B8", "CSA"]
    @eval @figure $("mixture_comparison_$(dataset)_noise_$(lowercase(method))") mixtureNoiseFigure($dataset, $method)
end

#effect of the roughness penalty on the adaptive grid
for dataset in ["jaeckel_case_i", "jaeckel_case_ii", "spx500_20180205_1m", "tsla_20180615_1m"]
    @eval @figure $("mixture_comparison_$(dataset)_penalty_dens") mixtureComparisonFigure($dataset, ["CSA0", "CSA"], :Density)
end

#hypersphere parameterization (option A), and bandwidths of options A and B constrained in a box
for dataset in ["jaeckel_case_ii", "spx500_20180205_1m"]
    @eval @figure $("mixture_comparison_$(dataset)_optiona_dens") mixtureComparisonFigure($dataset, ["A6", "B6", "CSA"], :Density)
    @eval @figure $("mixture_comparison_$(dataset)_floor_dens") mixtureComparisonFigure($dataset, ["A6F", "B6F", "CSA"], :Density)
end

#right wing of the SPX500 1m smile: market quotes, closest arbitrage-free quotes and the calibrations to the latter
@figure "mixture_comparison_spx500_20180205_1m_arbfree_wing" begin
    d, da = loadSmile("spx500_20180205_1m"), comparisonSmile("spx500_20180205_1m_arbfree")
    curves = readcsv("mixture_comparison_curves.csv")
    methods = MIXTURE_MAIN
    c = rows(curves[curves.Dataset.=="spx500_20180205_1m_arbfree", :], :Model, methods...)
    c = c[c.Logmoneyness.>=0.0, :]
    fig = bookFigure(size=(480, 380))
    ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
    groupLines!(ax, c.Logmoneyness, c.Vol .* 100, c.Model, order=methods, labels=MIXTURE_COMPARISON_LABELS)
    right = d.strikes .>= d.forward
    referenceScatter!(ax, logmoneyness(d)[right], d.vols[right] .* 100, label="Market", marker=:xcross, markersize=8)
    scatter!(ax, logmoneyness(da)[right], da.vols[right] .* 100, label="Arbitrage-free", color=(:black, 0.6), markersize=5)
    bookLegend!(ax, position=:ct)
    fig
end
