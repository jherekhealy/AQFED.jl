# Chapter 10: exotic options. Barriers, digitals, forward start, quanto.

using AQFED

@figure "barrier_termstructure_5y" begin
    df = readtable("barrier_termstructure_5y.txt", squeeze=true)
    black = rows(df, :Method, "Black")
    pde = rows(df, :Method, "TR-BDF2")
    fig = bookFigure(size=(800, 300))
    ax = bookAxis(fig[1, 1], xlabel="Barrier level", ylabel="Price")
    axr = Axis(fig[1, 1], yaxisposition=:right, ylabel="Relative error",
        ytickformat=v -> string.(round.(Int, v .* 100), "%"))
    hidespines!(axr)
    hidexdecorations!(axr)
    l1 = lines!(ax, black.Level, black.Price, color=PALETTE[1])
    l2 = lines!(axr, black.Level, black.Price ./ pde.Price .- 1, color=PALETTE[2])
    axislegend(ax, [l1, l2], ["Price", "Relative error"], position=:lb)
    fig
end

@figure "barrier_cashdividend_1y" begin
    df = readtable("barrier_cashdividend_1y.txt", squeeze=true)
    fig = bookFigure(size=(800, 300))
    ax = bookAxis(fig[1, 1], xlabel="Dividend ex-date", ylabel="Price")
    groupLines!(ax, df.ExDate, df.Price, df.Method)
    bookLegend!(ax, position=:lb)
    fig
end

function barrierSmileFigure(file)
    df = readtable(file, squeeze=true)
    df = relabel!(df[df.Method.!="LMG2", :], :Method, "TR-BDF2" => "Local vol")
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Barrier level", ylabel="Price")
    groupLines!(ax, df.Level, df.Price, df.Method)
    bookLegend!(ax)
    return fig
end
@figure "barrier_smile" barrierSmileFigure("barrier_smile.txt")
@figure "barrier_zeliade4" barrierSmileFigure("barrier_zeliade4.txt")

#undiscounted digital put price N(-d2) as a function of the Black volatility, for a strike close to the forward
@figure "digital_impliedvol" begin
    strike, forward, t = 6.0, 5.996, 1.0
    vol = range(0.001, stop=0.3, step=0.001)
    d = @. (log(forward / strike) - vol^2 * t / 2) / (vol * sqrt(t))
    fig = bookFigure(size=(800, 300))
    ax = bookAxis(fig[1, 1], xlabel="Black volatility in %", ylabel="Digital put price")
    lines!(ax, vol .* 100, AQFED.Math.normcdf.(-d))
    fig
end

@figure "lambda" begin
    r, q = -0.02 / 100, -0.22 / 100
    s = range(0.0, stop=0.12, step=0.001)
    fig = bookFigure(size=(800, 300))
    ax = bookAxis(fig[1, 1], xlabel="σ", ylabel="λ²", xticks=([0, 2.5, 5.0, 7.5, 10.0, 12.5] ./ 100, ["0.0%", "2.5%", "5.0%", "7.5%", "10.0%", "12.5%"]))
    lines!(ax, s, @.(0.25 * s^4 + (r + q) * s^2 + (r - q)^2))
    fig
end

@figure "forward_start_heston_rel" begin
    df = readtable("forward_start_hestonlv.txt")
    heston = rows(df, :Model, "Heston")
    fig = bookFigure(size=(800, 300))
    axes = Axis[]
    for (i, t) in enumerate(unique(df.T1))
        ax = bookAxis(fig[1, i], xlabel="Strike", ylabel=i == 1 ? "Relative difference" : "", title="T1=$(round(t, digits=3))")
        h = heston[heston.T1.==t, :]
        for (j, model) in enumerate(("Black", "Black-Static", "LocalVol", "LocalVol-Frozen"))
            d = df[(df.T1.==t).&(df.Model.==model), :]
            lines!(ax, d.Strike, d.Price ./ h.Price .- 1, label=model, color=PALETTE[j])
        end
        push!(axes, ax)
    end
    Legend(fig[2, :], axes[1], orientation=:horizontal, framevisible=false)
    fig
end

## quanto with a skew: displaced diffusions for the equity and the FX rate

function quantoSkew()
    spotEquity, tte, betaEquity, volEquity = 1.0, 2.0, 1.0 / 16, 28.98 / 100
    betaFX, volFX, correlation, spotFX = 1.0, 10.0 / 100, 50 / 100, 1.0
    strike = 0.05:0.05:1.6
    bs(k, f, v) = AQFED.Black.blackScholesFormula(true, k, f, v, 1.0, 1.0)
    c = correlation * volEquity * volFX * betaEquity * betaFX * tte
    shift = (1.0 / betaEquity - 1) * spotEquity
    exact = @. 1.0 / betaFX * bs(strike + shift, spotEquity / betaEquity * exp(-c), (volEquity * betaEquity)^2 * tte) -
               (1.0 / betaFX - 1) * bs(strike + shift, spotEquity / betaEquity, (volEquity * betaEquity)^2 * tte)
    blackPriceEquity = @. bs(strike + (1 - betaEquity) / betaEquity * spotEquity, spotEquity / betaEquity, (volEquity * betaEquity)^2 * tte)
    blackVolEquity = @. AQFED.Black.impliedVolatility(true, blackPriceEquity, spotEquity, strike, tte, 1.0)
    blackPriceEquityATM = bs(spotEquity + (1 - betaEquity) / betaEquity * spotEquity, spotEquity / betaEquity, (volEquity * betaEquity)^2 * tte)
    blackVolEquityATM = AQFED.Black.impliedVolatility(true, blackPriceEquityATM, spotEquity, spotEquity, tte, 1.0)
    blackPriceFX = spotFX * bs(1 / spotFX + (1 - betaFX) / betaFX * spotFX, spotFX / betaFX, (volFX * betaFX)^2 * tte)
    blackVolFX = AQFED.Black.impliedVolatility(true, blackPriceFX, spotFX, spotFX, tte, 1.0)
    cBlackATM = correlation * blackVolEquityATM * blackVolFX * tte
    black = @. bs(strike, spotEquity * exp(-cBlackATM), blackVolEquity^2 * tte)
    blackATM = @. bs(strike, spotEquity * exp(-cBlackATM), blackVolEquityATM^2 * tte)
    return strike, exact, black, blackATM
end

@figure "quanto_skew_price" begin
    strike, exact, black, blackATM = quantoSkew()
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Price")
    for (i, (y, label)) in enumerate(((exact, "Exact"), (black, "Black"), (blackATM, "BlackATM")))
        lines!(ax, strike, y, label=label, linestyle=(:solid, :dash, :dot)[i])
    end
    bookLegend!(ax)
    fig
end

@figure "quanto_skew_error" begin
    strike, exact, black, blackATM = quantoSkew()
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Relative error in price")
    lines!(ax, strike, (black .- exact) ./ exact, label="Black")
    lines!(ax, strike, (blackATM .- exact) ./ exact, label="BlackATM", linestyle=:dash)
    bookLegend!(ax)
    fig
end

function schobelZhuQuantoFigure(y, ylabel; scale=1)
    df = readtable("schobelzhu_quanto.txt")
    df = df[df.Model.!="Schobel-Zhu", :]
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel=ylabel)
    groupLines!(ax, df.Strike, df[!, y] .* scale, df.Model)
    bookLegend!(ax)
    return fig
end
@figure "schobelzhu_quanto_abs" schobelZhuQuantoFigure(:Error, "Absolute error in price")
@figure "schobelzhu_quanto_rel" schobelZhuQuantoFigure(:RelError, "Relative error in price in %", scale=100)
