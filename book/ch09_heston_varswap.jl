# Chapter 9: calibration of the Heston model to variance swap prices, SPX500 as of October 8, 2024.

isdefined(@__MODULE__, :SmileData) || include(joinpath(@__DIR__, "..", "test", "smile_data.jl"))

@compute "heston_varswap_oct2024" begin
    s = loadSurface("spx500_20241008_surface")
    ts, forwards, strikes, vols = s.expiries, s.forwards, s.strikes, s.vols
    strikesM = ones(length(ts)) * strikes'
    replication(linear) = [VolatilityModels.priceVarianceSwap(VolatilityModels.FukasawaVarianceSwapReplication(linear), t, log.(strikes ./ forwards[i]), vols[i, :] .^ 2, 1.0)
                           for (i, t) in enumerate(ts)]
    points = DataFrame(Expiry=ts, Flat=sqrt.(replication(false)), Linear=sqrt.(replication(true)))
    writetable("heston_varswap_oct2024_replication.csv", points)
    #Heston fits: 3D, 1D and explicit reductions to variance swaps, and to vanillas with a truncation at 0.5 or 2 standard deviations
    vanillaFit(dev) = begin
        prices, isCall, weights = VolatilityModels.convertVolsToPricesOTMWeights(ts, forwards, strikesM, vols, vegaFloor=1e-2, truncationDev=dev)
        first(VolatilityModels.calibrateHestonFromPrices(ts, forwards, strikesM, prices, isCall, weights))
    end
    fits = [
        "3D LS" => VolatilityModels.makeVarianceSwapReduction(ts, forwards, strikes, vols, isLinearExtrapolation=true),
        "1D LS" => VolatilityModels.makeVarianceSwapReduction1D(ts, forwards, strikes, vols, isLinearExtrapolation=true),
        "Explicit" => VolatilityModels.makeVarianceSwapReductionExplicit(ts, forwards, strikesM, vols, isLinearExtrapolation=false, tte=1.0),
        "Vanillas 0.5 dev" => vanillaFit(0.5),
        "Vanillas 2.0 dev" => vanillaFit(2.0)]
    tdense = range(0.0, ts[end], length=100)
    curves = DataFrame(Fit=String[], Expiry=Float64[], Vol=Float64[], v0=Float64[], kappa=Float64[], theta=Float64[])
    for (label, p) in fits
        y = [sqrt(10000 * VolatilityModels.integratedVarianceExpectation(p.v0, p.κ, p.θ; dt=t)) for t in tdense]
        append!(curves, DataFrame(Fit=label, Expiry=collect(tdense), Vol=y, v0=p.v0, kappa=p.κ, theta=p.θ))
    end
    writetable("heston_varswap_oct2024_fits.csv", curves)
end

@figure "heston_varswap_oct2024" begin
    points = readcsv("heston_varswap_oct2024_replication.csv")
    curves = readcsv("heston_varswap_oct2024_fits.csv")
    fig = bookFigure(size=(860, 420))
    ax = bookAxis(fig[1, 1], xlabel="Expiry", ylabel="Price in volatility %", title="Variance swap prices: replication vs Heston fits")
    scatter!(ax, points.Expiry, points.Flat, color=:black, marker=:circle, markersize=11, strokecolor=:white, strokewidth=0.8, label="Flat extrapolation")
    scatter!(ax, points.Expiry, points.Linear, color=:black, marker=:utriangle, markersize=12, strokecolor=:white, strokewidth=0.8, label="Linear extrapolation")
    styles = Dict("3D LS" => :solid, "1D LS" => :dash, "Explicit" => :dot, "Vanillas 0.5 dev" => :dashdot, "Vanillas 2.0 dev" => :dashdot)
    for (i, fit) in enumerate(unique(curves.Fit))
        c = curves[curves.Fit.==fit, :]
        lines!(ax, c.Expiry, c.Vol, linewidth=2.5, linestyle=styles[fit], color=PALETTE[i], label=fit)
    end
    bookLegend!(ax, position=:rb)
    fig
end

## time-dependent Heston, SPX500 as of October 8, 2024 (commented study of test/hestonts_test.jl, book/legacy/test_plots.jl):
## full calibration (σ̃, ρ, α̃) against the reduced calibration (ρ, α̃ with σ̃ fixed from the variance swap replication on the
## full strike range or on the liquid range only), on 101 quotes interpolated for Δ_simple ∈ [10%, 90%]

const HESTONTD_LABELS = ["Reduced calibration on full range", "Reduced calibration Δ ∈ [10%, 90%]", "Full calibration"]
const HESTONTD_DELTAS = collect(range(0.1, 0.9, length=101))

#Hagan's piecewise-constant parameterization with κ̃ = 0.5, calibrated by bootstrap; its forward variance is ξ0(t) = σ̃(t)²
@compute "hestontd_varcurve" begin
    s = loadSurface("spx500_20241008_surface")
    ts, forwards, strikes, vols = s.expiries, s.forwards, s.strikes, s.vols
    strikesM = ones(length(ts)) * strikes'
    volsF, strikesF = VolatilityModels.filterVolsBySimpleDelta(ts, forwards, strikesM, vols, deltas=HESTONTD_DELTAS)
    pricesF, isCallF, weightsF = VolatilityModels.convertVolsToPricesOTMWeights(ts, forwards, strikesF, volsF, vegaFloor=1e-4, truncationDev=10.0)
    swapCurve = VolatilityModels.makeSwapCurveReplication(ts, forwards, strikesM, vols)
    swapCurveF = VolatilityModels.makeSwapCurveReplication(ts, forwards, strikesF, volsF)
    settings = (κ=0.5, lower=[1e-2, -0.99, 0.05], upper=[2.0, 0.5, 20.0], minimizer="BFGS", method="Cos", isGlobal=false)
    reduced(curve) = first(VolatilityModels.calibrateHestonHaganFromPricesVS(ts, forwards, strikesF, pricesF, isCallF, weightsF; swapCurve=curve, settings...))
    models = [reduced(swapCurve), reduced(swapCurveF),
        first(VolatilityModels.calibrateHestonHaganFromPrices(ts, forwards, strikesF, pricesF, isCallF, weightsF; swapCurve=swapCurveF, settings...))]
    #forward variance of the replications, piecewise constant between expiries
    replication = DataFrame(Start=vcat(0.0, ts[1:end-1]), Full=VolatilityModels.computeForwardVariance(swapCurve, ts),
        Liquid=VolatilityModels.computeForwardVariance(swapCurveF, ts))
    params = DataFrame(Calibration=String[], Start=Float64[], Leverage=Float64[], Rho=Float64[], Alpha=Float64[], RMSEVol=Float64[])
    smiles = DataFrame(Calibration=String[], Expiry=Float64[], Strike=Float64[], Vol=Float64[])
    for (label, p) in zip(HESTONTD_LABELS, models)
        volError, _ = VolatilityModels.estimateVolError(p, ts, forwards, strikesF, volsF)
        rmseVol = [sqrt(sum(abs2, volError[i, :]) / size(volError, 2)) for i in eachindex(ts)]
        append!(params, DataFrame(Calibration=label, Start=p.startTime, Leverage=p.leverage, Rho=p.ρ, Alpha=p.σ, RMSEVol=rmseVol))
        for (i, t) in enumerate(ts)
            tsParams, τ = AQFED.TermStructure.makeHestonTSParams(p, tte=t)
            pricer = CharFuncPricing.makeCosCharFuncPricer(DefaultCharFunc(tsParams), τ, 256, 8)
            for k in range(strikes[1], strikes[end], length=201)
                price = CharFuncPricing.priceEuropean(pricer, true, k / forwards[i], 1.0, τ, 1.0)
                vol = try Black.impliedVolatility(true, max(price, 1e-32), 1.0, k / forwards[i], t, 1.0) catch; NaN end
                push!(smiles, (label, t, k, vol))
            end
        end
    end
    writetable("hestontd_varcurve_replication.csv", replication)
    writetable("hestontd_varcurve_params.csv", params)
    writetable("hestontd_varcurve_smiles.csv", smiles)
end

const HESTONTD_COLORS = Dict(zip(HESTONTD_LABELS, PALETTE[1:3]))

#forward variance curves and the smiles of three expiries in one figure with a shared legend; the market quotes within the
#calibration range Δ_simple ∈ [10%, 90%] as a grey ribbon under the curves, the others as rings
@figure "hestontd_full_reduced" begin
    s = loadSurface("spx500_20241008_surface")
    params = readcsv("hestontd_varcurve_params.csv")
    smiles = readcsv("hestontd_varcurve_smiles.csv")
    _, strikesF = VolatilityModels.filterVolsBySimpleDelta(s.expiries, s.forwards, ones(length(s.expiries)) * s.strikes', s.vols, deltas=[0.1, 0.9])
    fig = bookFigure(size=(767, 620))
    italic(x) = rich(x, font=:italic)
    ax = bookAxis(fig[1, 1], xlabel=rich("Time ", italic("t"), " in years"), ylabel="Forward variance",
        title=rich("Forward variance ", italic("ξ"), rich("0", offset=(0, -0.3), fontsize=13), "(", italic("t"), ")"), titlealign=:left)
    for label in HESTONTD_LABELS
        c = params[params.Calibration.==label, :]
        stairs!(ax, vcat(c.Start, s.expiries[end]), vcat(c.Leverage, c.Leverage[end]) .^ 2, step=:post, color=HESTONTD_COLORS[label], linewidth=2.0)
    end
    for (k, tte) in enumerate((0.4767, 1.6931, s.expiries[end]))
        i = argmin(abs.(s.expiries .- tte))
        t = s.expiries[i]
        used = minimum(strikesF[i, :]) .<= s.strikes .<= maximum(strikesF[i, :])
        ax = bookAxis(fig[fld(k, 2)+1, mod(k, 2)+1], xlabel=k == 1 ? "" : "Strike (relative to spot)", ylabel=k < 3 ? "Implied volatility in %" : "",
            title=rich(italic("T"), " = $(round(t, digits=2)) years"), titlealign=:left)
        referenceScatter!(ax, s.strikes[used], 100 .* s.vols[i, used], label="Market, Δ ∈ [10%, 90%]", dense=true)
        referenceScatter!(ax, s.strikes[.!used], 100 .* s.vols[i, .!used], label="Market", dense=false, markersize=7)
        for label in HESTONTD_LABELS
            c = smiles[(smiles.Calibration.==label).&(abs.(smiles.Expiry .- t).<1e-8), :]
            lines!(ax, c.Strike, 100 .* c.Vol, color=HESTONTD_COLORS[label], label=label)
        end
        k == 1 && Legend(fig[3, 1:2], ax, orientation=:horizontal, nbanks=2, tellwidth=false, tellheight=true)
    end
    fig
end

## time-dependent Heston calibrations of the section on bootstrap or global calibration, on the quotes interpolated at Δ_simple ∈ {25%, 50%, 75%}
## (Exact) or on the market quotes (commented study of test/hestonts_test.jl and book/legacy/test_plots.jl). The global calibrations reproduce
## the published curves. The bootstrap uses Nelder-Mead and Gauss-Newton, which reproduces the table for κ̃ = 2.

const HESTONTD_BOUNDS = (lower=[1e-2, -0.99, 0.05], upper=[2.0, 0.5, 20.0])

function hestonTDBootstrap(ts, forwards, strikes, vols, prices, isCall, weights, κ)
    swapCurve = VolatilityModels.makeSwapCurveReplication(ts, forwards, strikes, vols)
    return first(VolatilityModels.calibrateHestonHaganFromPrices(ts, forwards, strikes, prices, isCall, weights; κ=κ, minimizer="NM", method="Cos",
        isGlobal=false, swapCurve=swapCurve, HESTONTD_BOUNDS...))
end

function hestonTDGlobal(ts, forwards, strikes, vols, prices, isCall, weights, κ; knots=fill(length(ts), 3))
    swapCurve = VolatilityModels.makeSwapCurveReplication(ts, forwards, strikes, vols)
    return first(VolatilityModels.calibrateHestonHaganFromPricesParam(ts, forwards, strikes, prices, isCall, weights; κ=κ, numberOfKnots=knots,
        minimizer="User", method="Cos", swapCurve=swapCurve, HESTONTD_BOUNDS...))
end

hestonTDRows(figure, label, ts, p) = DataFrame(Figure=figure, Calibration=label, Maturity=ts, Leverage=p.leverage, Rho=p.ρ, Alpha=p.σ)

@compute "hestontd_spx500_oct2024_25Delta" begin
    s = loadSurface("spx500_20241008_surface")
    ts, forwards, vols = s.expiries, s.forwards, s.vols
    strikes = ones(length(ts)) * s.strikes'
    volsE, strikesE = VolatilityModels.filterVolsBySimpleDelta(ts, forwards, strikes, vols, deltas=[0.25, 0.5, 0.75])
    exact = (strikesE, volsE, VolatilityModels.convertVolsToPricesOTMWeights(ts, forwards, strikesE, volsE, vegaFloor=1e-4, truncationDev=10.0)...)
    #truncationDev = Φ⁻¹(75%) keeps the quotes with Δ_simple ∈ [25%, 75%]
    range25 = (strikes, vols, VolatilityModels.convertVolsToPricesOTMWeights(ts, forwards, strikes, vols, vegaFloor=1e-4, truncationDev=0.675)...)
    all = (strikes, vols, VolatilityModels.convertVolsToPricesOTMWeights(ts, forwards, strikes, vols, vegaFloor=1e-2, truncationDev=10.0)...)
    bootstrap(q, κ) = hestonTDBootstrap(ts, forwards, q..., κ)
    calibrate(q, κ; kwargs...) = hestonTDGlobal(ts, forwards, q..., κ; kwargs...)
    globalExact = calibrate(exact, 0.5)
    params = vcat(
        hestonTDRows("kappa2", "Bootstrap", ts, bootstrap(range25, 2.0)), hestonTDRows("kappa2", "Global", ts, calibrate(range25, 2.0)),
        hestonTDRows("kappa2", "Bootstrap Exact", ts, bootstrap(exact, 2.0)), hestonTDRows("kappa2", "Global Exact", ts, calibrate(exact, 2.0)),
        hestonTDRows("kappa05", "Bootstrap", ts, bootstrap(all, 0.5)), hestonTDRows("kappa05", "Global", ts, calibrate(all, 0.5)),
        hestonTDRows("kappa05", "Bootstrap Exact", ts, bootstrap(exact, 0.5)), hestonTDRows("kappa05", "Global Exact", ts, globalExact),
        hestonTDRows("kappa05_smooth", "Global Exact", ts, globalExact),
        #M knots for σ̃, 5 for ρ and 5 for α̃
        hestonTDRows("kappa05_smooth", "Global Spline", ts, calibrate(exact, 0.5, knots=[length(ts), 5, 5])))
    writetable("hestontd_spx500_oct2024_25Delta.csv", params)
end

function hestonTDParamsFigure(figure, styles)
    df = readcsv("hestontd_spx500_oct2024_25Delta.csv")
    df = df[df.Figure.==figure, :]
    fig = bookFigure(size=(800, 300), figure_padding=TIGHT_PADDING)
    local ax
    for (position, (column, scale, label)) in enumerate(((:Leverage, 100.0, L"\tilde{\sigma}\ (\%)"), (:Alpha, 1.0, L"\tilde{\alpha}"), (:Rho, 1.0, L"\rho")))
        ax = bookAxis(fig[1, position], xlabel="Maturity in years", ylabel=label)
        for (i, (name, style)) in enumerate(styles)
            c = df[df.Calibration.==name, :]
            lines!(ax, c.Maturity, scale .* c[!, column], color=PALETTE[i], linestyle=style, label=name)
        end
    end
    Legend(fig[2, 1:3], ax, orientation=:horizontal, framevisible=false, padding=(0, 0, 0, 0), margin=(0, 0, 0, 0))
    rowgap!(fig.layout, 4)
    colgap!(fig.layout, 10)
    fig
end

const HESTONTD_ALL_STYLES = [("Bootstrap", :solid), ("Global", :solid), ("Bootstrap Exact", :dash), ("Global Exact", :dash)]
@figure "hestontd_spx500_oct2024_25Delta_kappa2" hestonTDParamsFigure("kappa2", HESTONTD_ALL_STYLES)
@figure "hestontd_spx500_oct2024_25Delta_kappa05" hestonTDParamsFigure("kappa05", HESTONTD_ALL_STYLES)
@figure "hestontd_spx500_oct2024_25Delta_kappa05_smooth" hestonTDParamsFigure("kappa05_smooth", [("Global Exact", :solid), ("Global Spline", :solid)])

## smiles of the Fourier, Benhamou-Gobet-Miri and Hagan approximations on calibrated parameters, κ̃ = 5: DJIA options of Rouah (table tbl:djia_rouah)
## and SPX500 options in 2010

const DJIA_ROUAH = (spot=129.14, r=0.0010, q=0.0068, days=[37.0, 72.0, 135.0, 226.0], strikes=collect(124.0:136.0),
    vols=[19.62 19.1 18.6 18.1 17.61 17.18 16.71 16.44 16.45 16.61 17.01 17.55 17.96
        19.47 19.05 18.61 18.12 17.74 17.43 17.06 16.71 16.41 16.25 16.02 16.1 16.57
        20.19 19.8 19.43 19.07 18.71 18.42 18.13 17.83 17.6 17.43 17.26 17.16 17.24
        21.15 20.82 20.57 20.21 20.0 19.74 19.5 19.27 18.99 18.84 18.62 18.46 18.42] ./ 100)

function hestonTDSmileRows(panel, p, tte, forward, strikes)
    tsParams, τ = AQFED.TermStructure.makeHestonTSParams(p, tte=tte)
    pricer = CharFuncPricing.makeCosCharFuncPricer(DefaultCharFunc(tsParams), τ, 256, 10)
    bgm, hagan = VolatilityModels.BGMApprox(tsParams), VolatilityModels.HaganHestonTSApprox(p, tte)
    vol(price, k) = try Black.impliedVolatility(true, max(price, 1e-16), forward, k, tte, 1.0) catch; NaN end
    return DataFrame(Panel=panel, Strike=collect(strikes),
        Fourier=[100 * vol(CharFuncPricing.priceEuropean(pricer, true, k, forward, τ, 1.0), k) for k in strikes],
        BGM=[100 * vol(VolatilityModels.priceEuropean(bgm, true, k, forward, τ, 1.0), k) for k in strikes],
        Hagan=[100 * vol(VolatilityModels.priceEuropean(hagan, true, k, forward, tte, 1.0), k) for k in strikes])
end

@compute "hestontd_spx500_djia_iv" begin
    fit(ts, forwards, strikes, vols) = hestonTDGlobal(ts, forwards, strikes, vols,
        VolatilityModels.convertVolsToPricesOTMWeights(ts, forwards, strikes, vols, vegaFloor=1e-2, truncationDev=10.0)..., 5.0)
    d = DJIA_ROUAH
    ts = d.days ./ 365
    forwards = d.spot .* exp.((d.r - d.q) .* ts)
    djia = fit(ts, forwards, ones(length(ts)) * d.strikes', d.vols)
    s = loadCalibrationSurface("spx500_2010.csv")
    forwards2010 = s.spot .* exp.((0.02 - 0.01) .* s.expiries)
    spx = fit(s.expiries, forwards2010, ones(length(s.expiries)) * s.strikes', s.vols)
    i = argmin(abs.(s.expiries .- 175 / 365))
    writetable("hestontd_spx500_djia_iv.csv", vcat(hestonTDSmileRows("DJIA - maturity 135 days", djia, ts[3], forwards[3], range(124.0, 136.0, length=61)),
        hestonTDSmileRows("SPX500 in 2010 - maturity 175 days", spx, s.expiries[i], forwards2010[i], range(s.strikes[1], s.strikes[end], length=101))))
end

@figure "hestontd_spx500_djia_iv" begin
    df = readcsv("hestontd_spx500_djia_iv.csv")
    fig = bookFigure(size=(800, 400), figure_padding=TIGHT_PADDING)
    local ax
    for (position, panel) in enumerate(unique(df.Panel))
        c = df[df.Panel.==panel, :]
        ax = bookAxis(fig[1, position], xlabel="Strike", ylabel=position == 1 ? "Implied volatility in %" : "", title=panel)
        for (i, (column, label)) in enumerate(((:Fourier, "Fourier"), (:BGM, "Benhamou, Gobet, Miri"), (:Hagan, "Hagan")))
            lines!(ax, c.Strike, c[!, column], color=PALETTE[i], label=label)
        end
    end
    Legend(fig[2, 1:2], ax, orientation=:horizontal, framevisible=false, padding=(0, 0, 0, 0), margin=(0, 0, 0, 0))
    rowgap!(fig.layout, 4)
    fig
end
