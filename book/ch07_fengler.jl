# Chapter 7: the arbitrage-free cubic spline in prices of Fengler (src/vol/fengler.jl). Directly on market quotes, the
# smoothing penalty does not lead to a realistic density; as an arbitrage-free
# projection of a smooth prior (least-squares multiquadric on the implied variance), it works on a wide variety of smiles.

using AQFED, AQFED.VolatilityModels, AQFED.Black, AQFED.Collocation, Printf

const FENGLER_DATASETS = ["spxw170324_170316", "spx500_20180205_1m", "tsla_20180615_1m", "aapl_20131028_4d", "jaeckel_case_i", "jaeckel_case_ii"]
const FENGLER_TITLES = Dict("spxw170324_170316" => "SPXW 1w", "spx500_20180205_1m" => "SPX500 1m", "tsla_20180615_1m" => "TSLA 1m",
    "aapl_20131028_4d" => "AAPL 4d", "jaeckel_case_i" => "Jäckel case I", "jaeckel_case_ii" => "Jäckel case II")

"""total variance w(y) extrapolated beyond [yL, yR] with the value and slope at the edge (C¹): linearly if the slope
increases outwards (below Lee's bound), otherwise with a slope decaying exponentially to zero"""
function extrapolatedTotalVariance(w, y, yL, yR)
    yL <= y <= yR && return w(y)
    ye = y < yL ? yL : yR
    he = 1e-5
    we = w(ye)
    wp = (w(ye + he) - w(ye - he)) / (2he)
    outward = y < yL ? -wp : wp
    d = abs(y - ye)
    outward >= 0 && return we + min(outward, 1.9) * d
    h = min(0.5, we / (2abs(outward)))
    return we + outward * h * (1 - exp(-d / h))
end

"""knots uniform in local standard deviation (in the variable ∫ dy/√w(y)) over [min(-nsd s₀, y_L - min(3 s_L, log 4)),
max(nsd s₀, y_R + min(3 s_R, log 4))], where s₀, s_L, s_R are the standard deviations √w at the money and at the first and
last quotes: the grid extends the quoted range by three local standard deviations, at most a factor 4 in strike"""
function fenglerKnots(w, yL, yR; npts=200, nsd=6.0, nfine=4001)
    s = sqrt(w(0.0))
    lo = min(-nsd * s, yL - min(3sqrt(w(yL)), log(4.0)))
    hi = max(nsd * s, yR + min(3sqrt(w(yR)), log(4.0)))
    yy = collect(range(lo, hi, length=nfine))
    iw = [1 / sqrt(max(extrapolatedTotalVariance(w, v, yL, yR), 1e-8)) for v in yy]
    z = cumsum(vcat(0.0, [(iw[i] + iw[i+1]) / 2 * (yy[i+1] - yy[i]) for i in 1:length(yy)-1]))
    ys = Float64[]
    for zi in range(0, z[end], length=npts)
        j = clamp(searchsortedlast(z, zi), 1, length(yy) - 1)
        push!(ys, yy[j] + (zi - z[j]) / (z[j+1] - z[j]) * (yy[j+1] - yy[j]))
    end
    return ys
end

"""Fengler spline (normalized forward, equal weights) on the prices of the total variance w at the knots of fenglerKnots. The
second derivative is only bounded by zero: a positive floor ε on c''(K) would force a price of at least ε(K_max-K)²/2 in the
right wing, about 6e-4 for Jäckel case I (strikes up to 113 times the forward) with the default ε=1e-7. The dual active set
method of Goldfarb and Idnani is exact; the interior point method of Clarabel is ten times faster but its solution is only
accurate to about 1e-6 in price, which misses the prices of the far wing of Jäckel case I. No penalty on the squared
density: λ=1e-12 is only a ridge for the positive definiteness required by Goldfarb-Idnani"""
function fenglerProjection(w, tte, yL, yR; npts=200, λ=1e-12)
    K = exp.(fenglerKnots(w, yL, yR, npts=npts))
    C = [Black.blackScholesFormula(true, k, 1.0, extrapolatedTotalVariance(w, log(k), yL, yR), 1.0, 1.0) for k in K]
    return VolatilityModels.calibrateFenglerSlice(tte, 1.0, K, C, ones(npts), λ=λ, solver="GI", eps=0.0)
end

"""least-squares multiquadric prior on the implied variance of a single smile: 16 centers equidistant over the quotes, of
bandwidth their spacing, with the roughness penalty μ ∫ (∂³v/∂x³ / σ̄²)² dx"""
fenglerPrior(d, y; m=16, μ=1e-6) = VolatilityModels.fitMultiquadricSurface2D([d.tte], [1.0], [y], [d.vols]; weights=[d.weights ./ (2 .* d.vols)],
    m=m, bandwidth=:spacing, roughness=μ)

fenglerVol(sp, y, tte) = sqrt(VolatilityModels.varianceByLogmoneyness(sp, y, 1.0, tte))
fenglerDensity(sp, y) = exp(y) * VolatilityModels.evaluateSecondDerivative(sp, exp(y))

## counter-example: Fengler directly on the SPXW 1w quotes, with a small and a larger penalty

@compute "fengler_knots" begin
    d = loadSmile("spxw170324_170316")
    y = logmoneyness(d)
    k = d.strikes ./ d.forward
    c, w = Collocation.weightedPrices(true, k, d.vols, d.weights, 1.0, 1.0, d.tte, vegaFloor=1e-8)
    yy = range(y[1], y[end], length=1001)
    out = DataFrame(Method=String[], Kind=String[], Y=Float64[], Value=Float64[])
    rmse = DataFrame(Method=String[], RMSE=Float64[])
    for (name, f) in (("Penalty λ=1e-9, all quotes as knots", () -> VolatilityModels.calibrateFenglerSlice(d.tte, 1.0, k, c, w, λ=1e-9, solver="Clarabel")),
        ("Penalty λ=1e-6, all quotes as knots", () -> VolatilityModels.calibrateFenglerSlice(d.tte, 1.0, k, c, w, λ=1e-6, solver="Clarabel")))
        sp = f()
        e = [1e4 * (fenglerVol(sp, yi, d.tte) - v) for (yi, v) in zip(y, d.vols)]
        e = filter(isfinite, e)
        push!(rmse, (name, sqrt(sum(abs2, d.weights[1:length(e)] .* e) / sum(abs2, d.weights[1:length(e)]))))
        append!(out, DataFrame(Method=name, Kind="vol", Y=collect(yy), Value=[100fenglerVol(sp, yi, d.tte) for yi in yy]))
        append!(out, DataFrame(Method=name, Kind="density", Y=collect(yy), Value=[fenglerDensity(sp, yi) for yi in yy]))
    end
    append!(out, DataFrame(Method="Market", Kind="vol", Y=y, Value=100 .* d.vols))
    println(rmse)
    writetable("fengler_knots.csv", out)
    writetable("fengler_knots_rmse.csv", rmse)
end

@figure "fengler_knots" begin
    df = readcsv("fengler_knots.csv")
    fig = bookFigure(size=(760, 330))
    ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Implied volatility in %")
    q = df[df.Method.=="Market", :]
    referenceScatter!(ax, q.Y, q.Value, label="Market")
    axd = bookAxis(fig[1, 2], xlabel="Forward log-moneyness", ylabel="Probability density", yscale=log10)
    methods = [m for m in unique(df.Method) if m != "Market"]
    for (i, m) in enumerate(methods)
        d = df[(df.Method.==m).&(df.Kind.=="vol"), :]
        lines!(ax, d.Y, d.Value, label=m, color=PALETTE[i], linewidth=1.2)
        d = df[(df.Method.==m).&(df.Kind.=="density"), :]
        lines!(axd, d.Y, max.(d.Value, 1e-8), color=PALETTE[i], linewidth=1.2)
    end
    ylims!(ax, 5, 70)
    axislegend(ax, position=:rt)
    fig
end

## what works: Fengler on a least-squares multiquadric prior, on a variety of smiles

@compute "fengler_prior_gallery" begin
    out = DataFrame(Data=String[], Method=String[], Kind=String[], Y=Float64[], Value=Float64[])
    summary = DataFrame(Data=String[], Quotes=Int[], Resolvable=Int[], PriorRMSE=Float64[], FenglerRMSE=Float64[], FenglerMax=Float64[], PriorNegative=Int[], Time_ms=Float64[])
    for name in FENGLER_DATASETS
        d = loadSmile(name)
        y = logmoneyness(d)
        mq = fenglerPrior(d, y)
        w = z -> VolatilityModels.totalVarianceDerivatives(mq, z, d.tte)[1]
        sp = fenglerProjection(w, d.tte, y[1], y[end])
        t = minimum(@elapsed((p = fenglerPrior(d, y); fenglerProjection(z -> VolatilityModels.totalVarianceDerivatives(p, z, d.tte)[1], d.tte, y[1], y[end]))) for _ in 1:3)
        #quotes whose out-of-the-money price is at least 1e-8 (counted only: all the quotes enter the error measure)
        otm = [Black.blackScholesFormula(exp(yi) >= 1, exp(yi), 1.0, v^2 * d.tte, 1.0, 1.0) for (yi, v) in zip(y, d.vols)]
        resolvable = otm .>= 1e-8
        wrmse(e) = (ok = isfinite.(e); sqrt(sum(abs2, d.weights[ok] .* e[ok]) / sum(abs2, d.weights[ok])))
        eP = [1e4 * (sqrt(max(w(yi), 0.0) / d.tte) - v) for (yi, v) in zip(y, d.vols)]
        eF = [1e4 * (fenglerVol(sp, yi, d.tte) - v) for (yi, v) in zip(y, d.vols)]
        span = y[end] - y[1]
        yy = range(y[1] - 0.05span, y[end] + 0.05span, length=801)
        dP = [densityFromTotalVariance(z -> extrapolatedTotalVariance(w, z, y[1], y[end]), yi) for yi in yy]
        push!(summary, (name, length(y), count(resolvable), wrmse(eP), wrmse(eF), maximum(abs, filter(isfinite, eF)), count(<(0), dP), 1e3t))
        @printf("%-20s prior RMSE %.1f bp, Fengler on prior %.1f bp, prior negative density at %d points, %.1f ms\n", name, wrmse(eP), wrmse(eF), count(<(0), dP), 1e3t)
        append!(out, DataFrame(Data=name, Method="Multiquadric", Kind="vol", Y=collect(yy), Value=[100sqrt(max(extrapolatedTotalVariance(w, yi, y[1], y[end]), 0.0) / d.tte) for yi in yy]))
        append!(out, DataFrame(Data=name, Method="Multiquadric", Kind="density", Y=collect(yy), Value=dP))
        append!(out, DataFrame(Data=name, Method="Fengler on the multiquadric", Kind="vol", Y=collect(yy), Value=[100fenglerVol(sp, yi, d.tte) for yi in yy]))
        append!(out, DataFrame(Data=name, Method="Fengler on the multiquadric", Kind="density", Y=collect(yy), Value=[fenglerDensity(sp, yi) for yi in yy]))
        append!(out, DataFrame(Data=name, Method="Market", Kind="vol", Y=y, Value=100 .* d.vols))
    end
    writetable("fengler_prior_gallery.csv", out)
    writetable("fengler_prior_summary.csv", summary)
end

function fenglerGalleryFigure(names)
    df = readcsv("fengler_prior_gallery.csv")
    fig = bookFigure(size=(900, 600))
    dense = all(count((df.Data .== n) .& (df.Method .== "Market")) >= 40 for n in names) #referenceScatter!: a ribbon for dense quotes
    Legend(fig[0, 1:3], [dense ? LineElement(color=(C_RIBBON, 0.5), linewidth=6) : MarkerElement(marker=:circle, color=:transparent, strokecolor=C_RING, strokewidth=1.8, markersize=10),
            LineElement(color=PALETTE[1], linestyle=:dash, linewidth=3), LineElement(color=PALETTE[2], linewidth=3)],
        ["Market", "Multiquadric", "Fengler on the multiquadric"], orientation=:horizontal, tellwidth=false, tellheight=true)
    for (c, name) in enumerate(names)
        dd = df[df.Data.==name, :]
        ax = bookAxis(fig[1, c], ylabel=c == 1 ? "Implied volatility in %" : "", title=FENGLER_TITLES[name])
        q = dd[dd.Method.=="Market", :]
        referenceScatter!(ax, q.Y, q.Value, label="Market")
        axd = bookAxis(fig[2, c], xlabel="Forward log-moneyness", ylabel=c == 1 ? "Probability density" : "")
        for (i, m) in enumerate(["Multiquadric", "Fengler on the multiquadric"])
            d = dd[(dd.Method.==m).&(dd.Kind.=="vol"), :]
            lines!(ax, d.Y, d.Value, label=m, color=PALETTE[i], linestyle=i == 1 ? :dash : :solid)
            d = dd[(dd.Method.==m).&(dd.Kind.=="density"), :]
            lines!(axd, d.Y, d.Value, label=m, color=PALETTE[i], linestyle=i == 1 ? :dash : :solid)
        end
        hlines!(axd, [0.0], color=C_GRID)
        linkxaxes!(ax, axd)
    end
    return fig
end
@figure "fengler_prior_gallery_1" fenglerGalleryFigure(FENGLER_DATASETS[1:3])
@figure "fengler_prior_gallery_2" fenglerGalleryFigure(FENGLER_DATASETS[4:6])
