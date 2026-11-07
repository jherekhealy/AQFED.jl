# Chapter 11: variance and volatility swaps, quanto variance.

#legends below the axes: the curves span the whole panel
@figure "varswap_derman" begin
    F = 100.0
    K = 60.0:10.0:140.0
    x = 60.0:0.1:140.0
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Value")
    lines!(ax, x, x ./ F .- 1 .- log.(x ./ F), label=rich("Payoff ", rich("f", font=:italic)), color=:black, linestyle=:dash)
    scatterlines!(ax, K, K ./ F .- 1 .- log.(K ./ F), label="Piecewise-linear", color=PALETTE[1], markersize=8)
    Legend(fig[2, 1], ax, orientation=:horizontal, tellwidth=false, tellheight=true)
    fig
end

@figure "varswap_trap" begin
    trap = readtable("varswap_trap_8.txt", squeeze=true)
    adapt = readtable("varswap_adapt_8.txt", squeeze=true)
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Value")
    a = adapt[sortperm(adapt.K), :]
    t = trap[sortperm(trap.K), :]
    lines!(ax, a.K, a.value, label="Integrand", color=:black, linestyle=:dash)
    scatterlines!(ax, t.K, t.value, label="Trapezoidal", color=PALETTE[1], markersize=8)
    xlims!(ax, 60, 140)
    Legend(fig[2, 1], ax, orientation=:horizontal, tellwidth=false, tellheight=true)
    fig
end

## SVI (a free or a ≥ 0) and LVG interpolations and extrapolations of the market smile of Table 12.1 (test/varswap_test.jl, "SVI")
const BAD_VOL_STRIKES = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]
const BAD_VOL_VOLS = [39.8, 34.9, 30.8, 27.4, 25.9, 24.5, 23.8, 23.1, 22.3, 21.5, 20.7, 19.8, 19.0, 18.2, 16.6, 15.4, 14.3, 14.7, 15.6]

@compute "varswap_bad_vol" begin
    tte, forward = 1.0, 1.0
    strikes, volA = BAD_VOL_STRIKES, BAD_VOL_VOLS ./ 100
    logm, w = log.(strikes ./ forward), ones(length(strikes))
    svi0, _ = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logm, volA, w, aMin=0.0)
    svi, _ = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logm, volA, w, aMin=-0.2)
    prices, wp = AQFED.Collocation.weightedPrices(true, strikes, volA, w, forward, 1.0, tte, vegaFloor=1e-8)
    lvg = AQFED.PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, wp, useVol=false, model=AQFED.PDDE.Quadratic(), location="Mid-XX", size=0, L=strikes[1] / 4, U=strikes[end] * 4)
    k = collect(range(0.3, 2.0, length=341))
    sviVol(s) = 100 .* sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(s, log.(k ./ forward)))
    out = vcat(DataFrame(Model="LVG", A=NaN, Strike=k, Vol=[100 * AQFED.Black.impliedVolatility(true, AQFED.PDDE.priceEuropean(lvg, true, x), forward, x, tte, 1.0) for x in k]),
        DataFrame(Model="SVI", A=svi.a, Strike=k, Vol=sviVol(svi)), DataFrame(Model="SVI a ≥ 0", A=svi0.a, Strike=k, Vol=sviVol(svi0)))
    writetable("varswap_bad_vol.csv", out)
end

@figure "varswap_bad_vol" begin
    df = readcsv("varswap_bad_vol.csv")
    fig = bookFigure(size=(800, 400))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Implied volatility in %")
    #range of the quotes, beyond it the curves are extrapolations
    vspan!(ax, BAD_VOL_STRIKES[1], BAD_VOL_STRIKES[end], color=(:black, 0.04))
    for (j, model) in enumerate(["LVG", "SVI", "SVI a ≥ 0"])
        d = df[df.Model.==model, :]
        label = model == "LVG" ? "Quadratic LVG" : rich("SVI, ", rich("a", font=:italic), model == "SVI" ? " = " * replace(string(round(d.A[1], digits=2)), "-" => "−") : " = 0")
        lines!(ax, d.Strike, d.Vol, color=PALETTE[j], label=label)
    end
    referenceScatter!(ax, BAD_VOL_STRIKES, BAD_VOL_VOLS, label="Market", dense=false, markersize=9)
    xlims!(ax, 0.3, 2.0)
    Legend(fig[1, 2], ax, framevisible=false)
    fig
end

#Bates model: variance and volatility swap prices in volatility points against the jump mean
function batesFigure(file, x, xlabel, y, ylims; negate=false)
    df = readtable(file)
    df = df[df.Method.!="LocalVol", :]
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel=xlabel, ylabel="Price in volatility")
    groupLines!(ax, negate ? -df[!, x] : df[!, x], df[!, y], df.Method)
    ylims!(ax, ylims...)
    bookLegend!(ax)
    return fig
end
@figure "bates_schoutens_var" batesFigure("bates_schoutens.txt", :k, "jump mean k", :VarSqrt, (30.8, 33))
@figure "bates_schoutens_vol" batesFigure("bates_schoutens.txt", :k, "jump mean k", :Vol, (30.8, 33))
@figure "bates_broadie_var" batesFigure("bates_broadie.txt", :a, "jump mean α", :VarSqrt, (11.5, 15), negate=true)
@figure "bates_broadie_vol" batesFigure("bates_broadie.txt", :a, "jump mean α", :Vol, (11.5, 15), negate=true)

function quantoVarianceFigure(file)
    df = readtable(file)
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Correlation", ylabel="Price")
    groupLines!(ax, df.Correlation, df.Price, df.Method)
    bookLegend!(ax)
    return fig
end
@figure "quanto_schobelzhu_10" quantoVarianceFigure("schobelzhu_quanto_raw.txt")
@figure "quanto_schobelzhu_10c" quantoVarianceFigure("schobelzhu_quanto_calib.txt")

#Black volatility surface (log-moneyness in [-0.4, 0.4]) of the method in file, as a matrix (log-moneyness, expiry)
function volSurface(file)
    df = readtable(file)
    df = df[(df.Logmoneyness.>-0.4).&(df.Logmoneyness.<0.4), :]
    expiries = sort(unique(df.Expiry))
    ys = sort(unique(df.Logmoneyness))
    m = fill(NaN, length(ys), length(expiries))
    for r in eachrow(df)
        m[searchsortedfirst(ys, r.Logmoneyness), searchsortedfirst(expiries, r.Expiry)] = r.BlackVolatility * 100
    end
    return ys, expiries, m
end

function surfaceFigure(ys, expiries, z)
    fig = bookFigure(size=(400, 400))
    surface3D!(fig[1, 1], ys, expiries, z, xlabel="Log-moneyness", ylabel="Expiry", zlabel="Black volatility in %", azimuth=1.25π, elevation=0.17π)
    return fig
end
#Bos et al. dividend approximation, and its difference with the finite difference reference
@figure "bos_flat_interpolation_2" surfaceFigure(volSurface("Bos.txt")...)
@figure "bosfdm_flat_interpolation_2" begin
    ys, expiries, fdm = volSurface("FDM.txt")
    _, _, bos = volSurface("Bos.txt")
    surfaceFigure(ys, expiries, fdm .- bos)
end

function fourierIntegrand(v, xmax, step)
    k = 0.1
    h(z) = exp(-(4 * z * z + 1) / 8 * v) * (z * cos(z * k) - 0.5 * sin(z * k)) * 2 / sqrt(8 * z * z + 2)
    x = 0.0:step:xmax
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="x", ylabel="h(x)")
    lines!(ax, x, h.(x), linewidth=0.8)
    return fig
end
@figure "integrand_fourier_ksmall" fourierIntegrand(0.001^2, 5000.0, 0.1)
@figure "integrand_fourier_ksmall_vlarge" fourierIntegrand(0.1^2, 100.0, 0.1)

function volSwapReturnFigure(file; scale=10000)
    df = readtable(file)
    fig = bookFigure(size=(320, 320))
    ax = bookAxis(fig[1, 1], xlabel="accumulated variance ×$(scale)", ylabel="volatility swap price")
    lines!(ax, df.variance .* scale, df.price)
    return fig
end
@figure "volswap_return_s_filon" volSwapReturnFigure("volswap_return_s_filon.txt")
@figure "volswap_return_l_filon" volSwapReturnFigure("volswap_return_l_filon.txt", scale=1)
@figure "volswap_return_s_adaptsim" volSwapReturnFigure("volswap_return_s_adaptsim.txt")
