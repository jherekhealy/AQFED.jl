# Chapter 4 (finite differences for American options): cash dividends, proportional volatility, exercise boundaries and Gamma
# interpolation. Data from the original R study (figures_old/lefloch_eqd_book_colo.r), persisted in book/data.

const AM_DIV_STYLES = Dict("ConstantYield" => :dash, "DailyYield" => :dot, "Proportional" => :dashdot, "Cash" => :solid,
    "Buehler" => :dash, "Lehman" => :dot)

#American option price against the dividend ex-date, for different dividend models
function amDividendTimeFigure(methods)
    df = readtable("am_dividend_time_p.txt")
    #compact: wide panel, legend inside, in the empty area below the curves on the left
    fig = bookFigure(size=(288, 200))
    ax = bookAxis(fig[1, 1], xlabel="Dividend ex-date in days", ylabel="Price")
    for (i, m) in enumerate(methods)
        d = sort(df[df.Method.==m, :], :td)
        lines!(ax, d.td .* 365, d.Price, label=m, color=PALETTE[i], linestyle=get(AM_DIV_STYLES, m, :solid))
    end
    axislegend(ax, position=(0.02, 0.12), framevisible=false, rowgap=0, patchsize=(16, 8))
    fig
end
@figure "am_dividend_time_p" amDividendTimeFigure(["ConstantYield", "DailyYield", "Proportional", "Cash"])
@figure "am_dividend_time_hybrid_p" amDividendTimeFigure(["Buehler", "Lehman", "Proportional", "Cash"])

#implied volatility of the American option prices under a proportional dividend model
function amPropVolFigure(file)
    df = readtable(file)
    models = unique(df.Model)
    fig = bookFigure(size=(288, 324))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Volatility in %")
    styles = [:solid, :dash, :dot, :dashdot]
    for (i, m) in enumerate(models)
        d = sort(df[df.Model.==m, :], :K)
        lines!(ax, d.K, d.Volatility .* 100, label=m, color=PALETTE[i], linestyle=styles[mod1(i, 4)])
    end
    Legend(fig[2, 1], ax, orientation=:horizontal, nbanks=3, framevisible=false)
    fig
end
@figure "am_prop_vol_05" amPropVolFigure("am_prop_vol_05.txt")
@figure "am_prop_vol_bad" amPropVolFigure("am_prop_vol.txt")

#early-exercise boundary for each dividend model (one panel per model)
function boundaryFacetFigure(file; xticks, yticks=Makie.automatic)
    df = readtable(file)
    df = df[isfinite.(df.S), :]
    #compact: the six models in a single row, with a shared x label
    fig = facetLines(df, :t, :S, :Method, ncol=6, xlabel="", ylabel="Asset price", size=(720, 190), xticks=xticks, yticks=yticks)
    Label(fig[2, 1:6], "Time to maturity", tellwidth=false)
    rowgap!(fig.layout, 4)
    colgap!(fig.layout, 14)
    fig
end
@figure "exercise_boundary_call" boundaryFacetFigure("exercise_boundary_callz.txt", xticks=([0.0, 0.2], ["0", "0.2"]), yticks=[100, 130, 160])
@figure "exercise_boundary_put_meyer1" boundaryFacetFigure("exercise_boundary_put_meyer1z.txt", xticks=([0.0, 0.25, 0.5], ["0", "", "0.5"]))

#Gamma obtained by interpolating the finite difference prices, against the Gamma on the grid (reference)
@figure "american_gamma_interp" begin
    df = readtable("american_gamma_interpm.txt")
    fig = bookFigure(size=(432, 180))
    axs = Axis[]
    for (i, m) in enumerate(["C2", "Hermite", "Rational"])
        d = sort(df[(df.Method.==m).&(df.Spot.>=97.25).&(df.Spot.<=98), :], :Spot)
        ax = bookAxis(fig[1, i], title=m, xlabel="Spot", ylabel=i == 1 ? "Gamma" : "")
        lines!(ax, d.Spot, d.Gamma, color=PALETTE[1])
        referenceScatter!(ax, d.Spot, d.GammaFDM, dense=false, markersize=7)
        i > 1 && hideydecorations!(ax, grid=false, ticks=false)
        push!(axs, ax)
    end
    linkyaxes!(axs...)
    fig
end
@figure "american_gamma_interp_lagrange" begin
    m = readtable("american_gamma_interpm.txt")
    mi = sort(readtable("american_gamma_interp_c2_lagrange.txt"), :Spot)
    mi = mi[(mi.Spot.>=97.25).&(mi.Spot.<=98), :]
    ref = unique(sort(m[(m.Spot.>=97.25).&(m.Spot.<=98), [:Spot, :GammaFDM]], :Spot), :Spot)
    fig = bookFigure(size=(432, 216))
    ax = bookAxis(fig[1, 1], xlabel="Spot", ylabel="Gamma")
    lines!(ax, mi.Spot, mi.Gamma, label="C2", color=PALETTE[2])
    lines!(ax, mi.Spot, mi.GammaFDM, label="Lagrange", color=PALETTE[3])
    referenceScatter!(ax, ref.Spot, ref.GammaFDM, dense=false, markersize=7)
    Legend(fig[1, 2], ax, "Interpolation", framevisible=false)
    fig
end
