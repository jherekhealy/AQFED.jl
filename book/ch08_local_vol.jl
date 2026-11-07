# Chapter 8: local volatility. Dupire, arbitrage in SVI, discretization of the local volatility PDE and Monte-Carlo,
# quadratic local volatility, Andreasen-Huge.

using AQFED

## Dupire with the example of Labordere: model and local volatilities around the liquidator date

function liquidatorFigure(curves; logscale, xlims, ylims=nothing, ylabel="Model volatility in %")
    fig = bookFigure(size=(480, 360))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel=ylabel, xscale=logscale ? log10 : identity,
        xticks=logscale ? ([5, 10, 20, 40, 80, 160], ["5", "10", "20", "40", "80", "160"]) : Makie.automatic)
    styles = [:solid, :dash, :dot, :dashdot]
    for (i, (strike, values, label)) in enumerate(curves)
        lines!(ax, strike, sqrt.(values) .* 100, label=label, linestyle=styles[i])
    end
    xlims!(ax, xlims...)
    ylims === nothing || ylims!(ax, ylims...)
    bookLegend!(ax, title="Expiry")
    return fig
end

function liquidatorModelVols()
    df = readtable("dupire_labordere_liquidator_modelvol.txt")
    return [(df.Strike, df.Before, "3.499"), (df.Strike, df.BeforeShifted, "3.499 shifted"), (df.Strike, df.At, "3.5"), (df.Strike, df.Maturity, "5.0")]
end
function liquidatorLocalVols()
    df = readtable("dupire_labordere_liquidator_localvol100.txt")
    #columns are named by the expiry; the figure shows the jump around the dividend date 3.5
    return [(df.Strike, df[!, "3.499"], "3.499"), (df.Strike, df[!, "3.5"], "3.5"), (df.Strike, df[!, "3.501"], "3.501")]
end
@figure "dupire_labordere_liquidator_modelvol_log" liquidatorFigure(liquidatorModelVols(), logscale=true, xlims=(2, 500))
@figure "dupire_labordere_liquidator_modelvol" liquidatorFigure(liquidatorModelVols(), logscale=false, xlims=(0.5, 20), ylims=(40, 70))
@figure "dupire_labordere_liquidator_localvol_log" liquidatorFigure(liquidatorLocalVols(), logscale=true, xlims=(4, 500))
@figure "dupire_labordere_liquidator_localvol" liquidatorFigure(liquidatorLocalVols(), logscale=false, xlims=(5, 100))

## SVI: local volatility and arbitrage

#S&P500 SVI slices (variance parameters a, b, ρ, s, m) at 7y and 10y of the chapter table, for the two calibration constraints
const SVI_SP500_EXPIRIES = [7.0, 10.0]
const SVI_SP500_FORWARDS = [1727.03, 1957.28]
const SVI_SP500_PARAMS = Dict(
    :positive => [(0.0414, 0.0278, -1.0, 0.1047, 0.1263), (0.0427, 0.0248, -1.0, 0.0005, 0.0480)],
    :increasing => [(0.0406, 0.0280, -1.0, 0.1263, 0.1344), (0.0400, 0.0251, -1.0, 0.1263, 0.1808)])

#SVI surface linearly interpolated in total variance between 7y and 10y along the forward log-moneyness, and its forward curve
function sviSP500Surface(constraint)
    sections = [AQFED.TermStructure.SVISection(p..., t, f) for (p, t, f) in zip(SVI_SP500_PARAMS[constraint], SVI_SP500_EXPIRIES, SVI_SP500_FORWARDS)]
    surface = AQFED.TermStructure.VarianceSurfaceBySection(sections, SVI_SP500_EXPIRIES)
    μ = log(SVI_SP500_FORWARDS[2] / SVI_SP500_FORWARDS[1]) / (SVI_SP500_EXPIRIES[2] - SVI_SP500_EXPIRIES[1])
    forward(t) = SVI_SP500_FORWARDS[1] * exp(μ * (t - SVI_SP500_EXPIRIES[1]))
    return surface, forward
end

#Dupire local volatility of the SVI surface on a strike grid.
#The narrow trough at y = m of the s = 0.0005 slice is hit by the strike grid at some expiries only: a row of spikes.
function sviLocalVolSurfaceFigure(constraint; strikes=range(1000, 2800, length=61), times=range(7.01, 9.99, length=150), dt=1e-4)
    surface, forward = sviSP500Surface(constraint)
    Z = [sqrt(AQFED.TermStructure.localVarianceByLogmoneyness(surface, log(k / forward(t)), t, t + dt)) for k in strikes, t in times]
    fig = bookFigure(size=(480, 380))
    surface3D!(fig[1, 1], strikes, times, Z .* 100, xlabel="Strike", ylabel="Expiry", zlabel="Local vol. in %", zrange=(0, 40),
        azimuth=1.15π, elevation=0.1π, aspect=(1, 1, 0.8), protrusions=(50, 10, 45, 0))
    return fig
end
@figure "svi_localvol_sp500_positive_3d" sviLocalVolSurfaceFigure(:positive)
@figure "svi_localvol_sp500_increasing_3d" sviLocalVolSurfaceFigure(:increasing)

function totalVarianceFigure(df, y)
    fig = bookFigure(size=(480, 400))
    ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Total variance")
    groupLines!(ax, df.Logmoneyness, y, df.Expiry, linestyles=true)
    bookLegend!(ax, title="Expiry", position=:ct)
    return fig
end
@figure "svi_zeliade_arb" (df = readtable("svi_zeliade_arb.txt"); totalVarianceFigure(df, df.Variance))
@figure "svi_zeliade_arb_zero" begin
    df = rows(readtable("svi_zeliade_arb_zero.txt"), :Model, "TRBDF2-LV")
    totalVarianceFigure(df, df.Volatility .^ 2 .* df.Expiry)
end

vogtReference!(ax, x, y) = referenceScatter!(ax, x, y, label="Vogt", marker=:cross, markersize=8)

@figure "svi_vogt_arb_zero" begin
    zero = readtable("svi_vogt_arb_zero.txt")
    vogt = rows(zero, :Method, "Vogt")
    variants = [("svi_vogt_arb_zero.txt", "Floor-0"), ("svi_vogt_arb_abs.txt", "Abs"), ("svi_vogt_arb_cap10.txt", "Cap-10"),
        ("svi_vogt_arb_cap1000.txt", "Cap-1000"), ("svi_vogt_arb_search.txt", "Search")]
    #svi_vogt_arb_cap10.txt names the column Model
    variant(f) = (d = readtable(f); "Model" in names(d) ? rename!(d, :Model => :Method) : d)
    df = vcat([(d = rows(variant(f), :Method, "TRBDF2-LV"); d[!, :Method] .= label; d) for (f, label) in variants]...)
    fig = bookFigure(size=(800, 500))
    ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
    vogtReference!(ax, vogt.Logmoneyness, vogt.Volatility .* 100)
    groupLines!(ax, df.Logmoneyness, df.Volatility .* 100, df.Method, linestyles=true)
    bookLegend!(ax, title="Method")
    fig
end

#file with Parameters (method), y, total variance w and the local variance denominator g
function vogtFigure(file, parameters, quantity; size=(400, 300), legendposition=:ct)
    df = readtable(file)
    vogt = rows(df, :Parameters, "Vogt")
    #all the methods except the reference when parameters is nothing
    d = parameters === nothing ? df[df.Parameters.!="Vogt", :] : rows(df, :Parameters, parameters...)
    value(t) = quantity == :vol ? sqrt.(t.w) .* 100 : t.g
    fig = bookFigure(size=size)
    ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel=quantity == :vol ? "Volatility in %" : "Denominator")
    quantity == :g && hlines!(ax, [0.0], color=:black, linewidth=0.8)
    vogtReference!(ax, vogt.y, value(vogt))
    groupLines!(ax, d.y, value(d), d.Parameters, order=parameters === nothing ? unique(d.Parameters) : collect(parameters), sortx=false)
    bookLegend!(ax, position=legendposition)
    return fig
end
@figure "svi_vogt_arb_vol" vogtFigure("svi_vogt_arb.txt", ("Gatheral", "QE-Penalty"), :vol)
@figure "svi_vogt_arb_g" vogtFigure("svi_vogt_arb.txt", ("Gatheral", "QE-Penalty"), :g, legendposition=:lt)
@figure "svi_vogt_arb_spline10_vol" vogtFigure("svi_vogt_arb_spline10.txt", nothing, :vol)
@figure "svi_vogt_arb_spline10_g" vogtFigure("svi_vogt_arb_spline10.txt", nothing, :g, legendposition=:lt)
@figure "svi_vogt_arb_vol_cap" vogtFigure("svi_vogt_arb_rk.txt", ("Cap-10", "Cap-100"), :vol, size=(800, 500))

function vogtLocalVarianceFigure(parameter)
    df = rows(readtable("svi_vogt_arb_rk.txt"), :Parameters, parameter)
    fig = bookFigure(size=(400, 250))
    ax = bookAxis(fig[1, 1], xlabel="Log-moneyness", ylabel="Local variance")
    lines!(ax, df.y, df.w ./ df.g)
    return fig
end
@figure "svi_vogt_arb_lv_cap100" vogtLocalVarianceFigure("Cap-100")
@figure "svi_vogt_arb_lv_cap10" vogtLocalVarianceFigure("Cap-10")

## Kahale example: raw spline against de-arbitraged spline

function kahaleSplines(t=0.425)
    pick(file) = (d = readtable(file); d[d.t.==t, :])
    return pick("kahale_raw_lv.txt"), pick("kahale_raw_lv_spline.txt"), pick("kahale_fixed_lv2.txt")
end

@figure "kahale_raw_lv_spline_iv" begin
    raw, rawSpline, fixed = kahaleSplines()
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Log-moneyness", ylabel="Implied volatility in %")
    referenceScatter!(ax, raw.y, sqrt.(raw.w ./ raw.t) .* 100, marker=:cross, markersize=8)
    lines!(ax, fixed.y, sqrt.(fixed.w ./ fixed.t) .* 100, label="De-arbitraged Spline", color=PALETTE[1])
    lines!(ax, rawSpline.y, sqrt.(rawSpline.w ./ rawSpline.t) .* 100, label="Raw Spline", color=PALETTE[2])
    bookLegend!(ax, position=:ct)
    fig
end

@figure "kahale_raw_lv_spline_lv" begin
    _, rawSpline, fixed = kahaleSplines()
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Log-moneyness", ylabel="Denominator")
    hlines!(ax, [0.0], color=:black, linewidth=0.8)
    lines!(ax, fixed.y, fixed.g, label="De-arbitraged Spline", color=PALETTE[1])
    lines!(ax, rawSpline.y, rawSpline.g, label="Raw Spline", color=PALETTE[2])
    bookLegend!(ax, position=:ct)
    fig
end

## PDE discretization of the local volatility

@figure "lv_discretisation_zeliade_box" begin
    df = rows(readtable("lv_discretisation_zeliade.txt"), :Method, "RE-LV", "RE-LVD", "RE-LVFD", "TRBDF2-LV", "TRBDF2-LVD", "TRBDF2-LVFD", "TRBDF2-Black")
    relabel!(df, :Method, "RE-LV" => "RE-(D)", "RE-LVD" => "RE-(B)", "RE-LVFD" => "RE-(A)", "TRBDF2-LV" => "TRBDF2-(D)", "TRBDF2-LVD" => "TRBDF2-(B)", "TRBDF2-LVFD" => "TRBDF2-(A)")
    df = df[(df.t.>4.3).&(df.Error.!=0), :]
    methods = sort(unique(df.Method))
    fig = bookFigure(size=(900, 400))
    ax = bookAxis(fig[1, 1], ylabel="Absolute error in price", yscale=log10, xticks=(1:length(methods), methods))
    boxplot!(ax, [findfirst(==(m), methods) for m in df.Method], abs.(df.Error), color=PALETTE[7])
    fig
end

@figure "lv_discretisation_zeliade_100_500" begin
    df = rows(readtable("lv_discretisation_zeliade_100_500.txt"), :Method, "TRBDF2-LV", "TRBDF2-LVG", "TRBDF2-Black")
    relabel!(df, :Method, "TRBDF2-LV" => "Analytic", "TRBDF2-LVG" => "Numeric", "TRBDF2-Black" => "Black")
    df = df[df.Error.!=0, :]
    fig = bookFigure(size=(800, 300))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Absolute error in price", yscale=log10)
    groupLines!(ax, df.Strike, abs.(df.Error), df.Method, linestyles=true)
    bookLegend!(ax, position=:rb)
    fig
end

function localVolFlatLinearFigure(file)
    df = readtable(file)
    black = rows(df, :Method, "TRBDF2-Black")
    lv = rows(df, :Method, "TRBDF2-LocalVol")
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Log-moneyness", ylabel="Implied Black volatility in %")
    scatter!(ax, black.y, black.Volatility .* 100, label="TRBDF2 Black", color=:white, strokecolor=PALETTE[1], strokewidth=1)
    lines!(ax, lv.y, lv.Volatility .* 100, label="TRBDF2 Local vol.", color=PALETTE[2])
    bookLegend!(ax, position=:lt)
    return fig
end
@figure "localvol_flatlinear" localVolFlatLinearFigure("localvol_flatlinear.txt")
@figure "localvol_flatlinear_dirac" localVolFlatLinearFigure("localvol_flatlinear_dirac.txt")

function kahalePriceErrorFigure(file)
    df = rows(readtable(file), :Method, "TRBDF2-LV")
    df = df[df.Error.!=0, :]
    expiries = sort(unique(df.t))
    colors = cgrad(:acton10, length(expiries), categorical=true)
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Relative error in price", yscale=log10)
    for (i, t) in enumerate(expiries)
        d = df[df.t.==t, :]
        scatter!(ax, d.Strike, abs.(d.Error ./ d.Price), label=string(t), color=colors[i], markersize=5)
    end
    ylims!(ax, 1e-7, 1e-1)
    Legend(fig[1, 2], ax, "Expiry", framevisible=false)
    return fig
end
@figure "kahale_linear_price400" kahalePriceErrorFigure("kahale_linear_price400.txt")
@figure "kahale_flat_price400" kahalePriceErrorFigure("kahale_flat_price400.txt")

@figure "kahale_linear_price_positive400" begin
    pick(file) = (d = readtable(file); d[(d.Method.=="TRBDF2-LV").&(d.Strike.==826), :])
    fig = bookFigure(size=(800, 200))
    ax = bookAxis(fig[1, 1], xlabel="Expiry in ACT/365", ylabel="Relative error in price", yscale=log10)
    for (file, label) in (("kahale_linear_price_positive400.txt", "Raw"), ("kahale_linear_price400.txt", "Capped"))
        d = pick(file)
        lines!(ax, d.t, abs.(d.Error ./ d.Price), label=label)
    end
    bookLegend!(ax)
    fig
end

## AAPL 1m: linear implied volatility against flat local volatility

@figure "aapl_1m_lv_flat_black" begin
    df = rows(readtable("aapl_1m_lv_trv.txt", squeeze=true), :Method, "TRBDF2-LV", "TRBDF2-LVG", "Black")
    relabel!(df, :Method, "TRBDF2-LV" => "Linear-IV", "TRBDF2-LVG" => "Flat-LV")
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Volatility in %")
    ref = rows(df, :Method, "Black")
    referenceScatter!(ax, ref.Strike, ref.Volatility .* 100)
    models = df[df.Method.!="Black", :]
    groupLines!(ax, models.Strike, models.Volatility .* 100, models.Method, order=["Flat-LV", "Linear-IV"])
    limits!(ax, 68, 117, 28, 60)
    bookLegend!(ax)
    fig
end

@figure "aapl_1m_lv_flat" begin
    df = readtable("aapl_1m_lv_flat.txt", squeeze=true)
    df = df[df.Method.!="Black", :]
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Log-moneyness", ylabel="Local variance")
    groupLines!(ax, df.y, df.w ./ df.g, df.Method, order=["Flat-LV", "Linear-IV"])
    ylims!(ax, 0, 2)
    bookLegend!(ax, position=:ct)
    fig
end

## Quadratic local volatility

function analyticExpansionFigure(file; y=:Volatility, ylabel="Black volatility in %", reference="Analytic", x=:Strike)
    df = readtable(file)
    ref = rows(df, :Method, reference)
    models = df[df.Method.!=reference, :]
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel=ylabel)
    referenceScatter!(ax, ref[!, x], ref[!, y] .* 100, label=reference)
    groupLines!(ax, models[!, x], models[!, y] .* 100, models.Method)
    bookLegend!(ax)
    return fig
end
@figure "qlv_1m_expansion" analyticExpansionFigure("qlv_1m_expansion.txt")
@figure "aapl_1m_qlv" analyticExpansionFigure("aapl_1m_qlv.txt", y=:Vol, reference="Reference", x=:K)
@figure "aapl_2y_qlv" analyticExpansionFigure("aapl_2y_qlv.txt", y=:Vol, reference="Reference", x=:K)

@figure "qlv_2y_expansion_error" begin
    df = readtable("qlv_2y_expansion.txt")
    df = df[df.Method.!="Analytic", :]
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Error in volatility %")
    groupLines!(ax, df.Strike, df.Error .* 100, df.Method)
    bookLegend!(ax)
    fig
end

#quadratic local volatility calibrated to the Kahale surface: constant parameters per expiry, and the term-structure bootstrapped
#with the Fokker-Planck PDE (parameters of Table tbl:qlv_kahale), priced by the PDE or by parameter averaging; SVI for comparison
@compute "kahale_quadraticlv" begin
    isdefined(@__MODULE__, :SmileData) || include(joinpath(@__DIR__, "..", "test", "smile_data.jl"))
    QLV = AQFED.VolatilityModels
    s = loadKahaleSurface()
    n = length(s.expiries)
    forwards = s.spot .* exp.((s.r - s.q) .* s.expiries)
    vols = [s.vols[i, :] for i = 1:n]
    pdeSize = (nK=4801, stepsPerYear=3200)
    params, rmses = QLV.calibrateQuadraticLVTermStructure(s.expiries, forwards, fill(s.strikes, n), vols; pdeSize...)
    out = DataFrame()
    for (i, τ) in enumerate(s.expiries)
        f, df = forwards[i], exp(-s.r * τ)
        constant = QLV.calibrateQuadraticLV(τ, f, s.strikes, vols[i])
        pde = QLV.QuadraticLVForwardPDE(s.expiries[1:i], params[1:i], f, τ; highestStrike=maximum(s.strikes), pdeSize...)
        svi, _ = QLV.calibrateSVISectionFromGuess(τ, f, log.(s.strikes ./ f), vols[i], ones(length(s.strikes)), s=0.1, m=0.0, aMin=0.0, sMin=5e-4, noarbFactor=2.0)
        for (j, K) in enumerate(s.strikes)
            ref = AQFED.Black.blackScholesFormula(true, K, f, vols[i][j]^2 * τ, 1.0, df)
            prices = ("Quadratic" => QLV.priceEuropean(constant, true, K, f, τ, df), "QuadraticTS-FDM" => QLV.priceEuropean(pde, true, K, df),
                "QuadraticTS-Average" => QLV.priceEuropean(s.expiries[1:i], params[1:i], true, K, f, τ, df),
                "SVI" => AQFED.Black.blackScholesFormula(true, K, f, AQFED.TermStructure.varianceByLogmoneyness(svi, log(K / f)) * τ, 1.0, df))
            for (method, price) in prices
                vol = AQFED.Black.impliedVolatility(true, price / df, f, K, τ, 1.0)
                push!(out, (Expiry=τ, Strike=K, Method=method, Price=price, Error=price / ref - 1, Vol=vol, VolError=vol - vols[i][j]))
            end
        end
    end
    writetable("kahale_quadraticlv.csv", out)
    writetable("kahale_quadraticlv_params.csv", DataFrame(Expiry=s.expiries, Alpha=getfield.(params, :α), Beta=getfield.(params, :β), Gamma=getfield.(params, :γ), RMSE=rmses))
end

function kahaleQuadraticLV()
    df = readcsv("kahale_quadraticlv.csv")
    relabel!(df, :Method, "QuadraticTS-Average" => "Parameter averaging", "QuadraticTS-FDM" => "Fokker-Planck")
    return df[df.Expiry.==0.94, :]
end
@figure "qlv_kahale_average" begin
    df = rows(kahaleQuadraticLV(), :Method, "Parameter averaging", "Fokker-Planck")
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Black volatility in %")
    groupLines!(ax, df.Strike, df.Vol .* 100, df.Method)
    bookLegend!(ax)
    fig
end
@figure "qlv_kahale_fit" begin
    df = rows(kahaleQuadraticLV(), :Method, "Quadratic", "Fokker-Planck", "SVI")
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Error in Black volatility %")
    groupLines!(ax, df.Strike, df.VolError .* 100, df.Method)
    bookLegend!(ax)
    fig
end

## Monte-Carlo discretization of the local volatility

const MILSTEIN_LABELS = ["LV-Euler-Discrete" => "Euler-A", "LV-Euler-Backward" => "Euler-B", "LV-Euler-Mid" => "Euler-D",
    "LV-Milstein-Discrete" => "Milstein-A", "LV-Milstein-Backward" => "Milstein-B", "LV-Milstein-Mid" => "Milstein-D",
    "LV-Milstein-Clark" => "Milstein-Clark", "LV-Euler-Clark" => "Euler-Clark", "LV-Euler-F" => "Euler", "LV-Milstein-F" => "Milstein"]

#errors at N=65535 paths for a time-step file, plus the Euler scheme with half the time-step from halfFile
function milsteinErrors(file, halfFile, expiry, methods)
    df = readtable(file)
    df = relabel!(df[df.N.==65535, :], :Method, MILSTEIN_LABELS...)
    df = rows(df[df.Expiry.==expiry, :], :Method, methods...)
    halfFile === nothing && return df
    half = readtable(halfFile)
    half = half[(half.N.==65535).&(half.Method.=="LV-Euler-F").&(half.Expiry.==expiry), :]
    half[!, :Method] .= "Euler-Half"
    return vcat(df, half)
end

function milsteinFigure(df; absolute=true)
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel=absolute ? "Absolute error" : "Error in price")
    groupLines!(ax, df.Strike, absolute ? abs.(df.Error) : df.Error, df.Method)
    bookLegend!(ax)
    return fig
end
@figure "lv_milstein_zeliade_ssobolbb_016_milstein_short" milsteinFigure(milsteinErrors("lv_milstein_zeliade_ssobolbb_016.txt", "lv_milstein_zeliade_ssobolbb_008.txt", 0.16, ("Euler", "Milstein")))
@figure "lv_milstein_zeliade_ssobolbb_016_milstein" milsteinFigure(milsteinErrors("lv_milstein_zeliade_ssobolbb_016.txt", "lv_milstein_zeliade_ssobolbb_008.txt", 4.34, ("Euler-D", "Milstein-D", "Milstein-Clark")))
@figure "lv_milstein_zeliade_ssobolbb_016_euler" milsteinFigure(milsteinErrors("lv_milstein_zeliade_ssobolbb_016.txt", nothing, 4.34, ("Euler-D", "Euler-A", "Euler-B", "Euler-Clark")), absolute=false)
@figure "lv_milstein_zeliade_ssobolbb_001_milstein_short" milsteinFigure(milsteinErrors("lv_milstein_zeliade_ssobolbb_001.txt", "lv_milstein_zeliade_ssobolbb_0005.txt", 0.16, ("Euler", "Milstein")))

function mcAndreasenHugeFigure(file)
    df = readtable(file)
    df = df[df.Method.!="Analytic", :]
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Error")
    groupLines!(ax, df.Strike, df.Error, df.Method)
    bookLegend!(ax, position=:lt)
    return fig
end
@figure "mcah_zeliade_016" mcAndreasenHugeFigure("mcah_zeliade_016.txt")
@figure "mcah_zeliade_0016" mcAndreasenHugeFigure("mcah_zeliade_0016.txt")

## Andreasen-Huge with long time-steps

@figure "flat_ah_linear_400_lm" discreteLocalVolFigure("flat_ah_linear_400_lm.txt", size=(800, 400))

function longStepFigure(file; logError, withHalf)
    df = readtable(file)
    d = df[df[!, "Time-step"].==0.25, :]
    labels = string.(d.Method, " Δt=0.25")
    x, y = d.Strike, logError ? abs.(d.Error) : d.Error
    if withHalf
        h = df[(df[!, "Time-step"].==0.5).&(df.Method.=="Long-step-calibrated"), :]
        x, y, labels = vcat(x, h.Strike), vcat(y, logError ? abs.(h.Error) : h.Error), vcat(labels, fill("Long-step-calibrated Δt=0.5", nrow(h)))
    end
    fig = bookFigure(size=(400, 330))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Error in % of spot", yscale=logError ? log10 : identity)
    groupLines!(ax, x, y, labels)
    Legend(fig[2, 1], ax, framevisible=false, tellheight=true)
    return fig
end
@figure "ah_flat_calibrated_longstep" longStepFigure("ah_flat_calibrated_longstep.txt", logError=true, withHalf=true)
@figure "ah_flat_calibrated_longstep_vanilla" longStepFigure("ah_flat_calibrated_longstep_vanilla.txt", logError=true, withHalf=true)
@figure "ah_zeliade_calibrated_longstep" longStepFigure("ah_zeliade_calibrated_longstep.txt", logError=false, withHalf=false)
@figure "ah_zeliade_calibrated_longstep_vanilla" longStepFigure("ah_zeliade_calibrated_longstep_vanilla.txt", logError=false, withHalf=false)
