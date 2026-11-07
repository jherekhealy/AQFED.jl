# Chapter 7: arbitrage-free surfaces from the B-spline Andreasen-Huge model (src/pdde/ah_bspline.jl): on the Kahale surface
# against independent quadratic LVG slices, on two SPX500 surfaces of 2025 against CVI (src/vol/cvi.jl), and the price
# maps between the grid points of the Andreasen-Huge step.
# The SPX500 quotes of 2025 come from a market data provider (preprocessed by it), cut at 3.5 at-the-money standard
# deviations; they are read from the data of the comparison note (SPX2025_QUOTES) and the computations write the quotes
# they plot, with the model curves, to book/data.

using AQFED, AQFED.PDDE, AQFED.Collocation, AQFED.Black, AQFED.VolatilityModels, Printf

const SPX2025_QUOTES = expanduser("~/mypapers/sanos_practice/compare/data/quotes_proj.csv")
const SPX2025_DATES = ["20250123", "20250409"]
const SPX2025_TITLES = Dict("20250123" => "23 January 2025", "20250409" => "9 April 2025")
#the recommended setting: 12 points per step over the quotes extended by 10% on each side, the two outer coefficients
#tied (zero slope at the outer knots, no kink of the density at the last quotes)
const AHS_KNOTS = (size=12, smoothWings=true, knotExtension=0.1)
const AHS_UNIFORM = (useVol=true, grid=:uniform, gridSize=1601, priceMap=:box, AHS_KNOTS...)

"expiries, log-moneyness and implied vols of one SPX500 surface of 2025 (forward 1 at each expiry)"
function loadSPX2025(case; variant="raw")
    rows = [split(l, ',') for l in readlines(SPX2025_QUOTES)[2:end]]
    r = [x for x in rows if x[1] == case && x[2] == variant]
    js = sort(unique(parse.(Int, getindex.(r, 3))))
    ttes = [parse(Float64, first(x for x in r if parse(Int, x[3]) == j)[4]) for j in js]
    ys = [[parse(Float64, x[5]) for x in r if parse(Int, x[3]) == j] for j in js]
    vols = [[parse(Float64, x[6]) for x in r if parse(Int, x[3]) == j] for j in js]
    return ttes, ys, vols
end
normalizedCalls(ks, vols, ttes) = [[Black.blackScholesFormula(true, k, 1.0, v^2 * T, 1.0, 1.0) for (k, v) in zip(ks[i], vols[i])] for (i, T) in enumerate(ttes)]
calibrateAHS(ttes, ks, vols; kwargs...) = PDDE.calibrateBSplineAHSurface(ttes, ones(length(ttes)), ks, normalizedCalls(ks, vols, ttes), [ones(length(k)) for k in ks]; kwargs...)

gatheralG(k, w, wp, wpp) = (1 - k * wp / (2w))^2 - wp^2 / 4 * (1 / w + 0.25) + wpp / 2
"density of log-moneyness from a total variance function of log-moneyness"
function densityFromTotalVariance(w, k)
    h = 1e-4 * max(abs(k), 0.05)
    w0, wp, wm = w(k), w(k + h), w(k - h)
    d2 = -k / sqrt(w0) - sqrt(w0) / 2
    return gatheralG(k, w0, (wp - wm) / 2h, (wp - 2w0 + wm) / h^2) * exp(-d2^2 / 2) / sqrt(2π * w0)
end
"Dupire local volatility from a total variance function w(k, t)"
function dupireLocalVol(w, k, t)
    h, dt = 1e-4 * max(abs(k), 0.05), 1e-5 * t
    w0, wp, wm = w(k, t), w(k + h, t), w(k - h, t)
    g = gatheralG(k, w0, (wp - wm) / 2h, (wp - 2w0 + wm) / h^2)
    return sqrt(max((w(k, t + dt) - w(k, t - dt)) / 2dt / g, 0.0))
end
"total variance of slices interpolated linearly in time"
function interpolatedTotalVariance(sliceW, ttes, k, t)
    j = clamp(searchsortedlast(ttes, t), 1, length(ttes) - 1)
    a = (t - ttes[j]) / (ttes[j+1] - ttes[j])
    return (1 - a) * sliceW(j, k) + a * sliceW(j + 1, k)
end
rmseBp(errors) = (e = filter(isfinite, errors); sqrt(sum(abs2, e) / length(e)))

## Kahale surface: B-spline Andreasen-Huge against independent quadratic LVG slices

@compute "kahale_bspline_ah" begin
    s = loadKahaleSurface()
    forwards = s.spot .* exp.((s.r - s.q) .* s.expiries)
    n, m = length(s.expiries), length(s.strikes)
    prices, weights = zeros(n, m), zeros(n, m)
    for (i, t) in enumerate(s.expiries)
        prices[i, :], weights[i, :] = Collocation.weightedPrices(true, s.strikes, s.vols[i, :], ones(m), forwards[i], 1.0, t)
    end
    slices = [PDDE.calibrateQuadraticLVG(t, 1.0, s.strikes ./ forwards[i], prices[i, :] ./ forwards[i], weights[i, :],
        model=PDDE.Quadratic(), location="Mid-XX", optimizer="LM-ALG") for (i, t) in enumerate(s.expiries)]
    ks = [s.strikes ./ forwards[i] for i in 1:n]
    pv = [prices[i, :] ./ forwards[i] for i in 1:n]
    wv = [weights[i, :] for i in 1:n]
    ah = PDDE.calibrateBSplineAHSurface(s.expiries, ones(n), ks, pv, wv; AHS_UNIFORM..., location="Mid-XX")
    ahNoC3 = PDDE.calibrateBSplineAHSurface(s.expiries, ones(n), ks, pv, wv; AHS_UNIFORM..., location="Mid-XX", c3=false)
    rmse = rmseBp([1e4 * (sqrt(PDDE.varianceByLogmoneyness(ah, log(ks[i][j]), t)) - s.vols[i, j]) for (i, t) in enumerate(s.expiries) for j in 1:m])
    minCal = minimum(PDDE.priceNormalized(ah, x, s.expiries[i+1]) - PDDE.priceNormalized(ah, x, s.expiries[i]) for i in 1:n-1 for x in exp.(range(-1.0, 1.0, length=2001)))
    println("Kahale B-spline AH: RMSE ", rmse, " bp, min calendar spread on prices ", minCal)
    writetable("kahale_ahs_checks.csv", DataFrame(RMSE_bp=rmse, MinCalendarPrice=minCal))
    #implied and local volatility on the grid of the LVG figures (strikes of the quotes, expiries from half the first one)
    k = range(s.strikes[1], stop=s.strikes[end], length=101)
    grid = DataFrame(Expiry=Float64[], Strike=Float64[], ImpliedVol=Float64[], LocalVol=Float64[])
    for t in range(s.expiries[1] / 2, stop=s.expiries[end], length=25)
        f = s.spot * exp((s.r - s.q) * t)
        append!(grid, DataFrame(Expiry=t, Strike=collect(k), ImpliedVol=[sqrt(PDDE.varianceByLogmoneyness(ah, log(ki / f), t)) for ki in k],
            LocalVol=[PDDE.localVolatility(ah, ki / f, t) for ki in k]))
    end
    writetable("kahale_ahs_grid.csv", grid)
    #density of the forward moneyness around the forward at the first two expiries
    dens = DataFrame(Expiry=Float64[], Moneyness=Float64[], Curve=String[], Density=Float64[])
    h = 1e-4
    for i in 1:2, x in range(0.85, 1.15, length=601)
        lvgDensity = (PDDE.priceEuropean(slices[i], true, x + h) - 2PDDE.priceEuropean(slices[i], true, x) + PDDE.priceEuropean(slices[i], true, x - h)) / h^2
        append!(dens, DataFrame(Expiry=s.expiries[i], Moneyness=x, Curve=["LVG slice", "B-spline AH, no C3 condition", "B-spline AH, C3 condition"],
            Density=[lvgDensity, PDDE.densityNormalized(ahNoC3, x, i), PDDE.densityNormalized(ah, x, i)]))
    end
    writetable("kahale_ahs_density.csv", dens)
end

@figure "kahale_ahs_lv3d" (g = gridMatrix(readcsv("kahale_ahs_grid.csv"), :LocalVol); kahaleSurfaceFigure(g[1], g[2], g[3] .* 100, "Local volatility in %", zrange=(0, 45), azimuth=1.25π))

@figure "kahale_ahs_density" begin
    df = readcsv("kahale_ahs_density.csv")
    fig = bookFigure(size=(760, 330))
    order = ["LVG slice", "B-spline AH, no C3 condition", "B-spline AH, C3 condition"]
    styles = Dict("LVG slice" => :solid, "B-spline AH, no C3 condition" => :dash, "B-spline AH, C3 condition" => :dot)
    for (c, t) in enumerate(sort(unique(df.Expiry)))
        d = df[df.Expiry.==t, :]
        ax = bookAxis(fig[1, c], xlabel="Forward moneyness", ylabel=c == 1 ? "Probability density" : "", title=@sprintf("T = %.3f", t))
        for (i, name) in enumerate(order)
            e = d[d.Curve.==name, :]
            lines!(ax, e.Moneyness, e.Density, label=name, color=PALETTE[i], linestyle=styles[name])
        end
        c == 1 && axislegend(ax, position=:lt)
    end
    fig
end

## SPX500 surfaces of 2025: B-spline Andreasen-Huge against CVI

const SPX2025_TARGETS = (0.1, 1.0, 5.0) #expiries of the smile panels

@compute "spx2025_surfaces" begin
    curves = DataFrame(Case=String[], Method=String[], Kind=String[], Expiry=Float64[], X=Float64[], Value=Float64[])
    summary = DataFrame(Case=String[], Method=String[], RMSE_bp=Float64[], MaxError_bp=Float64[], Time_s=Float64[], Quotes=Int[], Expiries=Int[])
    let (ttes, ys, vols) = loadSPX2025(SPX2025_DATES[1]) #compilation
        calibrateAHS(ttes, [exp.(y) for y in ys], vols; AHS_UNIFORM...)
        VolatilityModels.calibrateConvexVolatilitySurface(ttes, ones(length(ttes)), ys, vols, [fill(0.005, length(v)) for v in vols]; basis=:cvi, n=15, λ=0.01)
    end
    for case in SPX2025_DATES
        ttes, ys, vols = loadSPX2025(case)
        ks = [exp.(y) for y in ys]
        tAH = @elapsed ah = calibrateAHS(ttes, ks, vols; AHS_UNIFORM...)
        tCVI = @elapsed cvi = VolatilityModels.calibrateConvexVolatilitySurface(ttes, ones(length(ttes)), ys, vols,
            [fill(0.005, length(v)) for v in vols]; basis=:cvi, n=15, λ=0.01)[1]
        cviW = (j, k) -> cvi.slices[j](k) * ttes[j]
        ahW = (j, k) -> PDDE.varianceByLogmoneyness(ah, k, ttes[j]) * ttes[j]
        lvAH = (k, t) -> 100 * PDDE.localVolatility(ah, exp(k), t)
        lvCVI = (k, t) -> 100 * dupireLocalVol((kk, tt) -> interpolatedTotalVariance(cviW, ttes, kk, tt), k, t)
        for (method, sw, lv, t) in (("B-spline AH", ahW, lvAH, tAH), ("CVI", cviW, lvCVI, tCVI))
            e = [1e4 * (sqrt(max(sw(j, y), 0.0) / T) - v) for (j, T) in enumerate(ttes) for (y, v) in zip(ys[j], vols[j])]
            push!(summary, (case, method, rmseBp(e), maximum(abs, filter(isfinite, e)), t, sum(length, ys), length(ttes)))
            @printf("%s %-12s RMSE %.2f bp, max %.1f bp, %.2f s\n", case, method, rmseBp(e), maximum(abs, filter(isfinite, e)), t)
            for target in SPX2025_TARGETS
                j = argmin(abs.(ttes .- target))
                y1, y2 = ys[j][1], ys[j][end]
                span = y2 - y1
                for y in range(y1 - 0.25span, y2 + 0.25span, length=301)
                    push!(curves, (case, method, "vol", ttes[j], y, 100sqrt(max(sw(j, y), 0.0) / ttes[j])))
                    #density of log-moneyness; for the B-spline AH model, the density of its cells
                    d = method == "CVI" ? densityFromTotalVariance(k -> sw(j, k), y) : exp(y) * PDDE.densityNormalized(ah, exp(y), j)
                    push!(curves, (case, method, "density", ttes[j], y, d))
                end
            end
            for t in range(ttes[2], 5.0, length=1000)
                push!(curves, (case, method, "lvcut", t, -0.15, lv(-0.15, t)))
            end
            for k in range(-0.5, 0.2, length=61), t in range(ttes[1], 5.0, length=61)
                push!(curves, (case, method, "lv3d", t, k, lv(k, t)))
            end
        end
        for target in SPX2025_TARGETS
            j = argmin(abs.(ttes .- target))
            append!(curves, DataFrame(Case=case, Method="Market", Kind="vol", Expiry=ttes[j], X=ys[j], Value=100 .* vols[j]))
        end
    end
    writetable("spx2025_surfaces.csv", curves)
    writetable("spx2025_summary.csv", summary)
end

const SPX2025_METHODS = ["B-spline AH", "CVI"]

function spx2025SmileFigure(case)
    df = readcsv("spx2025_surfaces.csv")
    df = df[string.(df.Case).==case, :]
    fig = bookFigure(size=(900, 480))
    expiries = sort(unique(df[df.Kind.=="vol", :Expiry]))
    for (c, t) in enumerate(expiries)
        ax = bookAxis(fig[1, c], ylabel=c == 1 ? "Implied vol. in %" : "", title=@sprintf("T = %.2f", t))
        q = df[(df.Method.=="Market").&(df.Expiry.==t), :]
        referenceScatter!(ax, q.X, q.Value, label="Market")
        for (i, mth) in enumerate(SPX2025_METHODS)
            d = df[(df.Method.==mth).&(df.Kind.=="vol").&(df.Expiry.==t), :]
            lines!(ax, d.X, d.Value, label=mth, color=PALETTE[i])
        end
        c == 1 && axislegend(ax, position=:rt)
        axd = bookAxis(fig[2, c], xlabel="Forward log-moneyness", ylabel=c == 1 ? "Density" : "")
        for (i, mth) in enumerate(SPX2025_METHODS)
            d = df[(df.Method.==mth).&(df.Kind.=="density").&(df.Expiry.==t), :]
            lines!(axd, d.X, d.Value, label=mth, color=PALETTE[i])
        end
        linkxaxes!(ax, axd)
    end
    return fig
end
@figure "spx_20250123_ahs_smiles" spx2025SmileFigure("20250123")
@figure "spx_20250409_ahs_smiles" spx2025SmileFigure("20250409")

@figure "spx2025_ahs_lvcut" begin
    df = readcsv("spx2025_surfaces.csv")
    fig = bookFigure(size=(760, 330))
    for (c, case) in enumerate(SPX2025_DATES)
        ax = bookAxis(fig[1, c], xlabel="Time to maturity", ylabel=c == 1 ? "Local volatility in %" : "", title=SPX2025_TITLES[case])
        for (i, mth) in enumerate(SPX2025_METHODS)
            d = df[(string.(df.Case).==case).&(df.Method.==mth).&(df.Kind.=="lvcut"), :]
            lines!(ax, d.Expiry, d.Value, label=mth, color=PALETTE[i])
        end
        c == 1 && axislegend(ax, position=:rt)
    end
    fig
end

@figure "spx_20250409_ahs_lv3d" begin
    df = readcsv("spx2025_surfaces.csv")
    fig = bookFigure(size=(900, 390))
    for (c, mth) in enumerate(SPX2025_METHODS)
        d = df[(string.(df.Case).=="20250409").&(df.Method.==mth).&(df.Kind.=="lv3d"), :]
        ks, ts = sort(unique(d.X)), sort(unique(d.Expiry))
        Z = [d[(d.X.==k).&(d.Expiry.==t), :Value][1] for k in ks, t in ts]
        surface3D!(fig[1, c], ks, ts, Z, title=mth, xlabel="Log-moneyness", ylabel="Expiry", zlabel="Local vol. in %", zrange=(0, 70))
    end
    fig
end

## Prices between the grid points of the B-spline Andreasen-Huge model, 9 April 2025, T = 0.45

const AHS_MAPS = ("Cell density", "Cubic spline of the prices")

@compute "ahs_price_maps" begin
    ttes, ys, vols = loadSPX2025("20250409")
    ks = [exp.(y) for y in ys]
    j = argmin(abs.(ttes .- 0.45))
    xs = exp.(range(0.30, 0.50, length=1501))
    fd(f, x) = (h = 1e-4 * x; (f(x + h) - 2f(x) + f(x - h)) / h^2)
    df = DataFrame(Grid=String[], Map=String[], Expiry=Float64[], Y=Float64[], Density=Float64[])
    for (gname, gkw) in (("Quoted strikes on the grid", (grid=:strikes, gridSize=401)), ("Uniform grid", (grid=:uniform, gridSize=1601)))
        ah = calibrateAHS(ttes, ks, vols; useVol=true, priceMap=:box, AHS_KNOTS..., gkw...)
        spl = PDDE.ahNaturalSpline(ah.z, ah.prices[j])
        maps = Dict("Cell density" => x -> PDDE.priceNormalized(ah, x, ttes[j]), "Cubic spline of the prices" => x -> PDDE.ahSplineAt(spl, x))
        for mname in AHS_MAPS
            append!(df, DataFrame(Grid=gname, Map=mname, Expiry=ttes[j], Y=log.(xs), Density=[x * fd(maps[mname], x) for x in xs]))
        end
    end
    writetable("ahs_price_maps.csv", df)
end

@figure "ahs_price_maps" begin
    df = readcsv("ahs_price_maps.csv")
    fig = bookFigure(size=(760, 330))
    axs = Axis[]
    for (c, g) in enumerate(["Quoted strikes on the grid", "Uniform grid"])
        ax = bookAxis(fig[1, c], xlabel="Forward log-moneyness", ylabel=c == 1 ? "Probability density" : "", title=g)
        for (i, mname) in enumerate(AHS_MAPS)
            d = df[(df.Grid.==g).&(df.Map.==mname), :]
            nrow(d) == 0 && continue
            lines!(ax, d.Y, d.Density, label=mname, color=PALETTE[i], linewidth=1.4, linestyle=i == 2 ? :dash : :solid)
        end
        c == 1 && ylims!(ax, 0, 0.08)
        c == 2 && axislegend(ax, position=:rt)
        push!(axs, ax)
    end
    fig
end

## AAPL options of 2 years (January 19, 2018 as of February 12, 2016, Table tbl:aapl_vols_2y): natural cubic spline of the
## variance in log-moneyness on the raw quotes and on the closest arbitrage-free prices, and CVI on the raw quotes

const AAPL2Y_TTE = 707 / 365
const AAPL2Y_FORWARD = 47.5 / exp(-0.7015) #from the log-moneyness of the table

@compute "aapl_2y_filter_cvi" begin
    df = readtable("aapl_180119_160212_am_prop_vol.txt")
    strikes, vols = Float64.(df.Strike), Float64.(df.Vol)
    f, tte = AAPL2Y_FORWARD, AAPL2Y_TTE
    ys = log.(strikes ./ f)
    prices, wv = Collocation.weightedPrices(true, strikes, vols, ones(length(strikes)), f, 1.0, tte, vegaFloor=1e-8)
    println("AAPL 2y arbitrage-free: ", Collocation.isArbitrageFree(strikes, prices, f))
    _, pricesF, _ = Collocation.filterConvexPrices(strikes, prices, wv, f, forceConvexity=true)
    volsF = [Black.impliedVolatility(true, p, f, k, tte, 1.0) for (k, p) in zip(strikes, pricesF)]
    @printf("largest move of the filter: %.2f vol points\n", 100maximum(abs.(volsF .- vols)))
    splineRaw = CubicSplineNatural(ys, vols .^ 2)
    splineF = CubicSplineNatural(ys, volsF .^ 2)
    cvi = VolatilityModels.calibrateConvexVolatilitySurface([tte], [1.0], [ys], [vols], [fill(0.005, length(vols))]; basis=:cvi, n=15, λ=0.01)[1].slices[1]
    #the knots and the constraint grid of the slice, as built by the calibration (40 points uniform in asinh(z/2), 3 per knot interval)
    b = cvi.basis
    knots = b.scale .* b.knots
    kL, kR = min(knots[1], ys[1]), max(knots[end], ys[end])
    grid = vcat([2b.scale * sinh(u) for u in range(asinh(kL / (2b.scale)), asinh(kR / (2b.scale)), length=40)],
        [b.scale * (b.knots[i] + (b.knots[i+1] - b.knots[i]) * l / 4) for i in 1:length(b.knots)-1 for l in 0:3])
    sliceVar(y) = VolatilityModels.varianceByLogmoneyness(VolatilityModels.CVISurface([cvi]), y, tte)
    rmse(v) = 1e4 * sqrt(sum(abs2, sqrt.(v.(ys)) .- vols) / length(ys))
    @printf("RMSE (bp): CVI %.1f, filtered quotes vs raw %.1f\n", rmse(sliceVar), 1e4 * sqrt(sum(abs2, volsF .- vols) / length(vols)))
    out = DataFrame(Method=String[], Kind=String[], Y=Float64[], Value=Float64[])
    yy = range(ys[1] - 0.05, ys[end] + 0.05, length=1601)
    for (name, v) in (("Spline", y -> splineRaw(y)), ("Spline, filtered", y -> splineF(y)), ("CVI", sliceVar))
        append!(out, DataFrame(Method=name, Kind="vol", Y=collect(yy), Value=[100sqrt(max(v(y), 0.0)) for y in yy]))
        append!(out, DataFrame(Method=name, Kind="density", Y=collect(yy), Value=[densityFromTotalVariance(z -> v(z) * tte, y) for y in yy]))
        append!(out, DataFrame(Method=name, Kind="g", Y=collect(yy), Value=[(w = z -> v(z) * tte; h = 1e-4; w0 = w(y);
            gatheralG(y, w0, (w(y + h) - w(y - h)) / 2h, (w(y + h) - 2w0 + w(y - h)) / h^2)) for y in yy]))
    end
    append!(out, DataFrame(Method="Market", Kind="vol", Y=ys, Value=100 .* vols))
    append!(out, DataFrame(Method="Filtered", Kind="vol", Y=ys, Value=100 .* volsF))
    append!(out, DataFrame(Method="CVI", Kind="knot", Y=knots, Value=zeros(length(knots))))
    append!(out, DataFrame(Method="CVI", Kind="grid", Y=sort(unique(grid)), Value=zeros(length(unique(grid)))))
    writetable("aapl_2y_filter_cvi.csv", out)
end

@figure "aapl_2y_spline_filtered" begin
    df = readcsv("aapl_2y_filter_cvi.csv")
    fig = bookFigure(size=(760, 330))
    ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Implied volatility in %")
    q = df[(df.Method.=="Market").&(df.Kind.=="vol"), :]
    referenceScatter!(ax, q.Y, q.Value, label="Market", dense=false)
    q = df[(df.Method.=="Filtered").&(df.Kind.=="vol"), :]
    scatter!(ax, q.Y, q.Value, label="Filtered", marker=:xcross, color=C_QUOTE, markersize=7)
    axd = bookAxis(fig[1, 2], xlabel="Forward log-moneyness", ylabel="Probability density")
    for (i, m) in enumerate(["Spline", "Spline, filtered"])
        d = df[(df.Method.==m).&(df.Kind.=="vol"), :]
        lines!(ax, d.Y, d.Value, label=m, color=PALETTE[i])
        d = df[(df.Method.==m).&(df.Kind.=="density"), :]
        lines!(axd, d.Y, d.Value, label=m, color=PALETTE[i])
    end
    axislegend(ax, position=:ct)
    axislegend(axd, position=:rt)
    fig
end

@figure "aapl_2y_cvi" begin
    df = readcsv("aapl_2y_filter_cvi.csv")
    fig = bookFigure(size=(760, 330))
    ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Implied volatility in %")
    knots = df[(df.Method.=="CVI").&(df.Kind.=="knot"), :Y]
    vlines!(ax, knots, color=(C_GRID, 1.0), linestyle=:dash, linewidth=1)
    q = df[(df.Method.=="Market").&(df.Kind.=="vol"), :]
    referenceScatter!(ax, q.Y, q.Value, label="Market", dense=false)
    for (i, m) in enumerate(["Spline, filtered", "CVI"])
        d = df[(df.Method.==m).&(df.Kind.=="vol"), :]
        lines!(ax, d.Y, d.Value, label=m, color=PALETTE[i+1])
    end
    xlims!(ax, extrema(df[df.Kind.=="vol", :Y])...)
    axislegend(ax, position=:ct)
    axd = bookAxis(fig[1, 2], xlabel="Forward log-moneyness", ylabel="Probability density")
    vlines!(axd, knots, color=(C_GRID, 1.0), linestyle=:dash, linewidth=1)
    for (i, m) in enumerate(["Spline, filtered", "CVI"])
        d = df[(df.Method.==m).&(df.Kind.=="density"), :]
        lines!(axd, d.Y, d.Value, label=m, color=PALETTE[i+1])
    end
    g = df[(df.Method.=="CVI").&(df.Kind.=="grid"), :Y]
    d = df[(df.Method.=="CVI").&(df.Kind.=="density"), :]
    keep = (g .>= minimum(d.Y)) .& (g .<= maximum(d.Y))
    scatter!(axd, g[keep], fill(minimum(df[df.Kind.=="density", :Value]), count(keep)), marker=:vline, markersize=9, color=C_QUOTE, label="Constraint grid")
    xlims!(axd, extrema(d.Y)...)
    axislegend(axd, position=:rt)
    fig
end

## SPX500 surfaces of 2025: arbitrage-free cubic spline (Fengler) on top of a least-squares two-dimensional multiquadric

"""total variance of the least-squares 2D multiquadric at expiry j, extrapolated beyond the quotes with the value and the
slope at the last quote (C¹): linearly when the slope increases outwards (capped below Lee's bound), and otherwise with a
slope decaying exponentially to zero, w = w_e + s h (1 - exp(-|y - y_e|/h)), h = min(1/2, w_e/(2|s|)), which stays above w_e/2"""
function priorTotalVariance(prior, ttes, ys, j, y)
    yL, yR = ys[j][1], ys[j][end]
    yL <= y <= yR && return VolatilityModels.totalVarianceDerivatives(prior, y, ttes[j])[1]
    ye = y < yL ? yL : yR
    w, wp = VolatilityModels.totalVarianceDerivatives(prior, ye, ttes[j])[1:2]
    outward = y < yL ? -wp : wp #slope in the outward direction
    d = abs(y - ye)
    outward >= 0 && return w + min(outward, 1.9) * d
    h = min(0.5, w / (2abs(outward)))
    return w + outward * h * (1 - exp(-d / h))
end

"""Fengler surface on the prices of the prior, on 200 knots per expiry uniform in local standard deviation (fenglerKnots
of ch07_fengler.jl), equal weights, no penalty (λ=1e-12, a ridge for Goldfarb-Idnani), density bounded by zero"""
function fenglerOnPrior(prior, ttes, ys; npts=200, λ=1e-12, solver="GI")
    n = length(ttes)
    K = zeros(n, npts); C = zeros(n, npts)
    for (j, T) in enumerate(ttes)
        w = z -> VolatilityModels.totalVarianceDerivatives(prior, z, T)[1]
        K[j, :] = exp.(fenglerKnots(w, ys[j][1], ys[j][end], npts=npts, nfine=401))
        C[j, :] = [Black.blackScholesFormula(true, k, 1.0, priorTotalVariance(prior, ttes, ys, j, log(k)), 1.0, 1.0) for k in K[j, :]]
    end
    return VolatilityModels.calibrateFenglerSurface(ttes, ones(n), K, C, ones(n, npts), λ=λ, solver=solver, eps=0.0)
end

@compute "spx2025_fengler" begin
    curves = DataFrame(Case=String[], Method=String[], Kind=String[], Expiry=Float64[], X=Float64[], Value=Float64[])
    summary = DataFrame(Case=String[], Method=String[], RMSE_bp=Float64[], MaxError_bp=Float64[], Time_s=Float64[], MinCalendar=Float64[])
    for case in SPX2025_DATES
        ttes, ys, vols = loadSPX2025(case); n = length(ttes)
        fit() = (p = VolatilityModels.fitMultiquadricSurface2D(ttes, ones(n), ys, vols; m=8); (p, fenglerOnPrior(p, ttes, ys)))
        fit()
        tPrior = minimum(@elapsed(VolatilityModels.fitMultiquadricSurface2D(ttes, ones(n), ys, vols; m=8)) for _ in 1:3)
        tAll = minimum(@elapsed(fit()) for _ in 1:3)
        prior, fs = fit()
        wPrior = (j, y) -> priorTotalVariance(prior, ttes, ys, j, y)
        wF = (j, y) -> VolatilityModels.varianceByLogmoneyness(fs.sections[j], y, 1.0, ttes[j]) * ttes[j]
        #local volatility: Dupire on the total variance interpolated linearly in time (the prior: its own time dependence)
        lvPrior = (k, t) -> 100 * dupireLocalVol((kk, tt) -> VolatilityModels.totalVarianceDerivatives(prior, kk, tt)[1], k, t)
        lvF = (k, t) -> 100 * dupireLocalVol((kk, tt) -> interpolatedTotalVariance(wF, ttes, kk, tt), k, t)
        for (method, sw, lv, t) in (("2D multiquadric", wPrior, lvPrior, tPrior), ("Fengler on the 2D multiquadric", wF, lvF, tAll))
            e = filter(isfinite, [1e4 * (sqrt(max(sw(j, y), 0.0) / T) - v) for (j, T) in enumerate(ttes) for (y, v) in zip(ys[j], vols[j])])
            #smallest calendar spread on normalized prices between consecutive expiries, on 2001 strikes inside both grids
            cal = NaN
            if method != "2D multiquadric"
                xs = exp.(range(-1.5, 1.0, length=2001))
                inside(j, x) = fs.sections[j].x[1] <= x <= fs.sections[j].x[end]
                cal = minimum(fs.sections[j+1](x) - fs.sections[j](x) for j in 1:n-1 for x in xs if inside(j, x) && inside(j + 1, x))
            end
            push!(summary, (case, method, rmseBp(e), maximum(abs, e), t, cal))
            @printf("%s %-31s RMSE %.2f bp, max %.1f bp, %.3f s, calendar %.1e\n", case, method, rmseBp(e), maximum(abs, e), t, cal)
            for target in SPX2025_TARGETS
                j = argmin(abs.(ttes .- target))
                y1, y2 = ys[j][1], ys[j][end]; span = y2 - y1
                for y in range(y1 - 0.25span, y2 + 0.25span, length=301)
                    push!(curves, (case, method, "vol", ttes[j], y, 100sqrt(max(sw(j, y), 0.0) / ttes[j])))
                    d = method == "2D multiquadric" ? densityFromTotalVariance(k -> sw(j, k), y) :
                        exp(y) * VolatilityModels.evaluateSecondDerivative(fs.sections[j], exp(y))
                    push!(curves, (case, method, "density", ttes[j], y, d))
                end
            end
            for t in range(ttes[2], 5.0, length=1000)
                push!(curves, (case, method, "lvcut", t, -0.15, lv(-0.15, t)))
            end
            for k in range(-0.5, 0.2, length=41), t in range(ttes[1], 5.0, length=41)
                push!(curves, (case, method, "lv3d", t, k, lv(k, t)))
            end
        end
        for target in SPX2025_TARGETS
            j = argmin(abs.(ttes .- target))
            append!(curves, DataFrame(Case=case, Method="Market", Kind="vol", Expiry=ttes[j], X=ys[j], Value=100 .* vols[j]))
        end
    end
    writetable("spx2025_fengler.csv", curves)
    writetable("spx2025_fengler_summary.csv", summary)
end

const FENGLER_METHODS = ["2D multiquadric", "Fengler on the 2D multiquadric"]
function spx2025FenglerFigure(case)
    df = readcsv("spx2025_fengler.csv")
    df = df[string.(df.Case).==case, :]
    fig = bookFigure(size=(900, 560))
    expiries = sort(unique(df[df.Kind.=="vol", :Expiry]))
    for (c, t) in enumerate(expiries)
        ax = bookAxis(fig[1, c], ylabel=c == 1 ? "Implied volatility in %" : "", title=@sprintf("T = %.2f", t))
        q = df[(df.Method.=="Market").&(df.Expiry.==t), :]
        referenceScatter!(ax, q.X, q.Value, label="Market")
        for (i, mth) in enumerate(FENGLER_METHODS)
            d = df[(df.Method.==mth).&(df.Kind.=="vol").&(df.Expiry.==t), :]
            lines!(ax, d.X, d.Value, label=mth, color=PALETTE[i], linestyle=i == 1 ? :dash : :solid)
        end
        c == 1 && axislegend(ax, position=:rt)
        axd = bookAxis(fig[2, c], xlabel="Forward log-moneyness", ylabel=c == 1 ? "Probability density" : "")
        for (i, mth) in enumerate(FENGLER_METHODS)
            d = df[(df.Method.==mth).&(df.Kind.=="density").&(df.Expiry.==t), :]
            lines!(axd, d.X, d.Value, label=mth, color=PALETTE[i], linestyle=i == 1 ? :dash : :solid)
        end
        hlines!(axd, [0.0], color=C_GRID)
        linkxaxes!(ax, axd)
    end
    return fig
end
@figure "spx_20250123_fengler_smiles" spx2025FenglerFigure("20250123")
@figure "spx_20250409_fengler_smiles" spx2025FenglerFigure("20250409")

@figure "spx2025_fengler_lvcut" begin
    df = readcsv("spx2025_fengler.csv")
    ah = readcsv("spx2025_surfaces.csv")
    fig = bookFigure(size=(760, 330))
    for (c, case) in enumerate(SPX2025_DATES)
        ax = bookAxis(fig[1, c], xlabel="Time to maturity", ylabel=c == 1 ? "Local volatility in %" : "", title=SPX2025_TITLES[case])
        d = df[(string.(df.Case).==case).&(df.Method.=="Fengler on the 2D multiquadric").&(df.Kind.=="lvcut"), :]
        lines!(ax, d.Expiry, d.Value, label="Fengler on the 2D multiquadric", color=PALETTE[2])
        d = ah[(string.(ah.Case).==case).&(ah.Method.=="B-spline AH").&(ah.Kind.=="lvcut"), :]
        lines!(ax, d.Expiry, d.Value, label="B-spline AH", color=PALETTE[1])
        c == 1 && axislegend(ax, position=:rt)
    end
    fig
end

@figure "spx_20250409_fengler_lv3d" begin
    df = readcsv("spx2025_fengler.csv")
    fig = bookFigure(size=(900, 450))
    for (c, mth) in enumerate(FENGLER_METHODS)
        d = df[(string.(df.Case).=="20250409").&(df.Method.==mth).&(df.Kind.=="lv3d"), :]
        ks, ts = sort(unique(d.X)), sort(unique(d.Expiry))
        Z = [d[(d.X.==k).&(d.Expiry.==t), :Value][1] for k in ks, t in ts]
        surface3D!(fig[1, c], ks, ts, Z, title=mth, xlabel="Log-moneyness", ylabel="Expiry", zlabel="Local vol. in %", zrange=(0, 70))
    end
    fig
end

## SPX500 surface of 31 March 2026, with calendar arbitrage at the long end: B-spline AH, CVI, the quadratic LVG with
## increasing local variance (Falck, calibrated from the last expiry to the first) and Fengler's spline on a 2D multiquadric
## (MQ+F, from the last expiry to the first). The MQ+F curves are those of the comparison note (scripts/mqf_common.jl,
## data/mqf_curves.csv), on the same quotes and expiries.

const SPX2026_CASE = "20260331"
const MQF_CURVES = expanduser("~/mypapers/sanos_practice/compare/data/mqf_curves.csv")
const SPX2026_METHODS = ["B-spline AH", "CVI", "LVG, increasing local variance", "MQ+F"]
const SPX2026_LABELS = Dict("B-spline AH" => "B-spline AH", "CVI" => "CVI", "LVG, increasing local variance" => "LVG, increasing loc. var.", "MQ+F" => "Fengler on 2D multiquadric")

"quadratic LVG with increasing local variance from the last expiry to the first (the setting of the comparison note):
10 nested knots per expiry, flat local variance beyond the quotes, boundaries those of the B-spline AH grid"
function calibrateFalckLVG(ttes, ys, vols)
    keep = [j for j in eachindex(ttes) if count(>(0.0), ys[j]) >= 2 && count(<(0.0), ys[j]) >= 2]
    tl = ttes[keep]; ks = [exp.(ys[j]) for j in keep]
    h = max(log(6.0), 6 * vols[keep][end][argmin(abs.(ys[keep][end]))] * sqrt(tl[end]))
    lvg = PDDE.calibrateLVGSurface(collect(tl), ones(length(tl)), ks, normalizedCalls(ks, vols[keep], tl), [ones(length(k)) for k in ks];
        size=10, location="Log-Uniform", useVol=true, order=:backward, L=min(exp(-h), minimum(first.(ks)) / 2), U=max(exp(h), 2maximum(last.(ks))), flatWings=true)
    return lvg, keep
end

@compute "spx_20260331_surfaces" begin
    curves = DataFrame(Case=String[], Method=String[], Kind=String[], Expiry=Float64[], X=Float64[], Value=Float64[])
    summary = DataFrame(Method=String[], Expiry=Float64[], RMSE_bp=Float64[], Time_s=Float64[])
    ttes, ys, vols = loadSPX2025(SPX2026_CASE)
    ks = [exp.(y) for y in ys]
    calibrateAHS(ttes, ks, vols; AHS_UNIFORM...) #compilation
    tAH = @elapsed ah = calibrateAHS(ttes, ks, vols; AHS_UNIFORM...)
    tCVI = @elapsed cvi = VolatilityModels.calibrateConvexVolatilitySurface(ttes, ones(length(ttes)), ys, vols,
        [fill(0.005, length(v)) for v in vols]; basis=:cvi, n=15, λ=0.01)[1]
    tLVG = @elapsed (lvg, keep) = calibrateFalckLVG(ttes, ys, vols)
    ahW = (j, k) -> PDDE.varianceByLogmoneyness(ah, k, ttes[j]) * ttes[j]
    cviW = (j, k) -> cvi.slices[j](k) * ttes[j]
    lvgW = (j, k) -> (i = findfirst(==(j), keep); i === nothing ? NaN : PDDE.varianceByLogmoneyness(lvg.sections[i], k, 1.0, ttes[j]) * ttes[j])
    targets = [argmin(abs.(ttes .- t)) for t in SPX2025_TARGETS]
    for (method, sw, t) in (("B-spline AH", ahW, tAH), ("CVI", cviW, tCVI), ("LVG, increasing local variance", lvgW, tLVG))
        for (j, T) in enumerate(ttes)
            isnan(sw(j, 0.0)) && continue
            e = [1e4 * (sqrt(max(sw(j, y), 0.0) / T) - v) for (y, v) in zip(ys[j], vols[j])]
            push!(summary, (method, T, rmseBp(e), t))
        end
        e = [1e4 * (sqrt(max(sw(j, y), 0.0) / T) - v) for (j, T) in enumerate(ttes) if !isnan(sw(j, 0.0)) for (y, v) in zip(ys[j], vols[j])]
        push!(summary, (method, 0.0, rmseBp(e), t))
        @printf("%s %-31s RMSE %.2f bp, %.2f s\n", SPX2026_CASE, method, rmseBp(e), t)
        for j in targets
            y1, y2 = ys[j][1], ys[j][end]; span = y2 - y1
            for y in range(y1 - 0.25span, y2 + 0.25span, length=301)
                push!(curves, (SPX2026_CASE, method, "vol", ttes[j], y, 100sqrt(max(sw(j, y), 0.0) / ttes[j])))
                d = method == "B-spline AH" ? exp(y) * PDDE.densityNormalized(ah, exp(y), j) : densityFromTotalVariance(k -> sw(j, k), y)
                push!(curves, (SPX2026_CASE, method, "density", ttes[j], y, d))
            end
        end
    end
    #MQ+F: the curves of the comparison note at the same expiries (index of the expiry in column a)
    for l in readlines(MQF_CURVES)[2:end]
        r = split(l, ',')
        (r[1] == SPX2026_CASE && r[2] == "MQ+F" && r[3] in ("iv", "dens")) || continue
        j = parse(Int, r[4]); j in targets || continue
        v = parse(Float64, r[6])
        push!(curves, (SPX2026_CASE, "MQ+F", r[3] == "iv" ? "vol" : "density", ttes[j], parse(Float64, r[5]), r[3] == "iv" ? 100v : v))
    end
    for j in targets
        append!(curves, DataFrame(Case=SPX2026_CASE, Method="Market", Kind="vol", Expiry=ttes[j], X=ys[j], Value=100 .* vols[j]))
    end
    writetable("spx_20260331_surfaces.csv", curves)
    writetable("spx_20260331_summary.csv", summary)
end

@figure "spx_20260331_smiles" begin
    df = readcsv("spx_20260331_surfaces.csv")
    fig = bookFigure(size=(900, 520))
    Legend(fig[0, 1:3], vcat([LineElement(color=(C_RIBBON, 0.5), linewidth=6)], [LineElement(color=PALETTE[i], linewidth=3) for i in eachindex(SPX2026_METHODS)]),
        vcat(["Market"], [SPX2026_LABELS[m] for m in SPX2026_METHODS]), orientation=:horizontal, nbanks=2, tellwidth=false, tellheight=true)
    expiries = sort(unique(df[df.Kind.=="vol", :Expiry]))
    for (c, t) in enumerate(expiries)
        ax = bookAxis(fig[1, c], ylabel=c == 1 ? "Implied vol. in %" : "", title=@sprintf("T = %.2f", t))
        q = df[(df.Method.=="Market").&(df.Expiry.≈t), :]
        referenceScatter!(ax, q.X, q.Value, label="Market", dense=true)
        for (i, mth) in enumerate(SPX2026_METHODS)
            d = df[(df.Method.==mth).&(df.Kind.=="vol").&(df.Expiry.≈t), :]
            lines!(ax, d.X, d.Value, label=mth, color=PALETTE[i])
        end
        axd = bookAxis(fig[2, c], xlabel="Forward log-moneyness", ylabel=c == 1 ? "Density" : "")
        for (i, mth) in enumerate(SPX2026_METHODS)
            d = df[(df.Method.==mth).&(df.Kind.=="density").&(df.Expiry.≈t), :]
            lines!(axd, d.X, d.Value, label=mth, color=PALETTE[i])
        end
        linkxaxes!(ax, axd)
    end
    fig
end

## SANOS: Dupire local volatility at the first expiry, beyond the quotes, on 13 February and 10 March 2025 (data of the
## SANOS practice note, sanos_julia examples/compute_sanos_lv_extremes.jl): without penalty (S2), with the roughness
## penalty of the book (S2+P), and with an exponential envelope beyond the last quote at a quarter of the decay rate of
## the edge (S2+E/4)

const SANOS_LV_CUT = expanduser("~/mypapers/sanos_practice/data/lv_extremes_sanos_cut.csv")
const SANOS_LV_CONFIGS = ["S2" => "no penalty", "S2+P" => "roughness penalty", "S2+E/4" => "exponential envelope"]

@compute "sanos_lv_first_expiry" begin
    df = DataFrame(Case=String[], Config=String[], Expiry=Float64[], QuoteMin=Float64[], QuoteMax=Float64[], X=Float64[], Value=Float64[])
    for l in readlines(SANOS_LV_CUT)[2:end]
        r = split(l, ';')
        any(first(c) == r[2] for c in SANOS_LV_CONFIGS) || continue
        push!(df, (r[1], r[2], parse.(Float64, r[3:7])...))
    end
    writetable("sanos_lv_first_expiry.csv", df)
end

@figure "sanos_lv_first_expiry" begin
    df = readcsv("sanos_lv_first_expiry.csv")
    fig = bookFigure(size=(760, 360))
    Legend(fig[0, 1:2], [LineElement(color=PALETTE[i], linewidth=3) for i in eachindex(SANOS_LV_CONFIGS)], [last(c) for c in SANOS_LV_CONFIGS],
        orientation=:horizontal, tellwidth=false, tellheight=true)
    for (c, (case, ttl)) in enumerate((("20250213", "13 Feb. 2025"), ("20250310", "10 Mar. 2025")))
        d0 = df[string.(df.Case).==case, :]
        ax = bookAxis(fig[1, c], xlabel="Forward log-moneyness", ylabel=c == 1 ? "Local volatility in %" : "", yscale=log10,
            title=@sprintf("%s, T = %.3f", ttl, d0.Expiry[1]))
        vspan!(ax, [d0.QuoteMin[1]], [d0.QuoteMax[1]], color=(C_GRID, 0.6))
        for (i, (cfg, label)) in enumerate(SANOS_LV_CONFIGS)
            d = d0[d0.Config.==cfg, :]
            keep = isfinite.(d.Value) .& (d.Value .> 0)
            lines!(ax, d.X[keep], d.Value[keep], label=label, color=PALETTE[i])
        end
    end
    fig
end
