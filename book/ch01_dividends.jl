# Chapter 1: dividend yields equivalent to cash and proportional dividends.

#yield equivalent to 4 semi-annual dividends of 0.5, cash (amount) or proportional (0.5%).
function dividendYields(spot, t)
    cashSum = 0.5 * count(d -> t >= d, (0.5, 1.0, 1.5, 2.0))
    propFactor = (1 - 0.5 / 100)^count(d -> t >= d, (0.5, 1.0, 1.5, 2.0))
    return -log(1 - cashSum / spot) / t, -log(propFactor) / t
end

function yieldFigure(index)
    t = range(0.01, stop=2.0, step=0.01)
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Time", ylabel="Dividend yield in %", yminorgridvisible=true)
    for spot in (100.0, 200.0)
        lines!(ax, t, [100 * dividendYields(spot, ti)[index] for ti in t], label=string(Int(spot)))
    end
    bookLegend!(ax, title="S(0)", position=:rb)
    return fig
end

@figure "yield_cash" yieldFigure(1)
@figure "yield_prop" yieldFigure(2)

@figure "logyield_cash_100" begin
    t = range(0.01, stop=2.0, step=0.01)
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="t", ylabel="-q t", yminorgridvisible=true)
    lines!(ax, t, [-ti * dividendYields(100.0, ti)[1] for ti in t])
    fig
end
