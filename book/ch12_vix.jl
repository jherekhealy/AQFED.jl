# Chapter 12: VIX futures and options.

function vixSmileFigure(file)
    df = readtable(file)
    ref = rows(df, :Model, "Reference")
    models = df[(df.Model.!="Reference").&(df.Model.!="Parabola"), :]
    fig = bookFigure(size=(800, 400))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Volatility in %")
    referenceScatter!(ax, ref.Strike, ref.Volatility .* 100)
    groupLines!(ax, models.Strike, models.Volatility .* 100, models.Model)
    Legend(fig[1, 2], ax, framevisible=false)
    return fig
end
@figure "vix170419_1m" vixSmileFigure("vix170419_1m.txt")
@figure "spxw170421_1m_eq" vixSmileFigure("spxw170421_1m_EQ.txt")

function vixFutureFigure(file, x, xlabel; legendposition=:rt)
    df = readtable(file)
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel=xlabel, ylabel="Price")
    groupLines!(ax, df[!, x], df.Price, df.Method)
    bookLegend!(ax, position=legendposition)
    return fig
end
@figure "vix_future_heston" vixFutureFigure("vix_future_heston.txt", :Maturity, "Maturity", legendposition=:rb)
@figure "vix_future_heston_vv" vixFutureFigure("vix_future_heston_vv.txt", :VolOfVar, "Vol of var")
