# Chapter 2: cash dividends. Data from the studies in ~/mypapers/cash_dividend_basket.

using AQFED

const CASH_DIVIDEND_MODELS = ["Zhang-2", "BB-LB", "LL-3", "GS", "BB", "BB-Ju"]

@figure "price_continuity_01" begin
    df = readtable("price_continuity_01.txt")
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Option maturity", ylabel="Option price", yminorgridvisible=true)
    lines!(ax, df.T, df.Price, color=PALETTE[1])
    limits!(ax, 0.45, 0.55, 8, 9)
    fig
end

@figure "hhl_greeks_calibrated_dgv" begin
    df = readtable("hhl_greeks_calibrated.txt")
    fig = bookFigure(size=(800, 800))
    for (i, greek) in enumerate((:Delta, :Gamma, :Vega))
        ax = bookAxis(fig[i, 1], xlabel=i == 3 ? "S" : "", ylabel=String(greek), yminorgridvisible=true)
        groupLines!(ax, df.S, df[!, greek], df.Model, linestyles=true)
        i == 1 && bookLegend!(ax, position=:rb)
    end
    fig
end

@figure "hhl_lehman_volerror" begin
    df = readtable("lehman_hhl.txt")
    fig = bookFigure(size=(800, 350))
    ax = bookAxis(fig[1, 1], xlabel="S(0)", ylabel="Error in Black volatility %", yminorgridvisible=true)
    groupLines!(ax, df.S, df.VolError .* 100, df.Model, linestyles=true)
    bookLegend!(ax, position=:rb)
    fig
end

# single dividend alpha=7, S=100, r=6%, sigma=30%, T=1, varying the ex-date: error in implied volatility of the basket and
# PLN approximations against HHL (setup of test/pln_european_test.jl, testset "EtoreGobetSingle").
const SINGLE_DIV_MODELS = ["Deelstra", "GS", "Ju", "LL2", "LL3", "VL2", "VL3"]

@compute "single_r6_td_basket" begin
    spot, vol, r, divAmount, tte = 100.0, 0.3, 0.06, 7.0, 1.0
    discountFactor = exp(-r * tte)
    pricers = Dict(
        "Ju" => AQFED.Basket.JuBasketPricer(),
        "Deelstra" => AQFED.Basket.DeelstraBasketPricer(1, 3, AQFED.Math.GaussLegendre(101)),
        "VL3" => AQFED.Basket.VorstLevyExpansion(3),
        "VL2" => AQFED.Basket.VorstLevyExpansion(2),
        "LL2" => AQFED.PLN.LeFlochLehmanPLNPricer(2),
        "LL3" => AQFED.PLN.LeFlochLehmanPLNPricer(3),
        "GS" => AQFED.PLN.GocseiSahelPLNPricer(),
        "HHL" => AQFED.PLN.HHLPLNPricer(AQFED.Math.GaussLegendre(101), AQFED.PLN.LIQUIDATOR))
    ttdList = collect(LinRange(0.01, 0.99, 99))
    forwardList = @. spot * exp(tte * r) - divAmount * exp((tte - ttdList) * r)
    out = DataFrame()
    for strike in (50.0, 100.0, 150.0)
        prices = Dict(name => map(ttd -> AQFED.PLN.priceEuropean(pricer, true, strike, spot * exp(tte * r), vol^2 * tte, tte, discountFactor,
                [AQFED.TermStructure.CapitalizedDividend(AQFED.TermStructure.Dividend(divAmount, ttd, ttd, false, false), exp((tte - ttd) * r))]), ttdList)
                      for (name, pricer) in pricers)
        refVol = AQFED.Black.impliedVolatility.(true, prices["HHL"], forwardList, strike, tte, discountFactor)
        for name in SINGLE_DIV_MODELS
            vols = AQFED.Black.impliedVolatility.(true, prices[name], forwardList, strike, tte, discountFactor)
            append!(out, DataFrame(Model=name, Strike=strike, Td=ttdList, VolError=abs.(vols .- refVol) .* 100))
        end
    end
    writetable("single_r6_td_basket.csv", out)
end

@figure "single_r6_strike100_td" begin
    df = readcsv("single_r6_td_basket.csv")
    fig = bookFigure()
    axes = Axis[]
    for (i, k) in enumerate((50, 100, 150))
        d = df[df.Strike.==k, :]
        ax = bookAxis(fig[1, i], xlabel=i == 2 ? "Dividend ex-date in years" : "", ylabel=i == 1 ? "Absolute error in volatility (%)" : "",
            title="Strike = $k", yscale=log10)
        groupLines!(ax, d.Td, d.VolError .+ 1e-12, d.Model, order=SINGLE_DIV_MODELS)
        ylims!(ax, 1e-8, 1e-2)
        i > 1 && hideydecorations!(ax, grid=false)
        push!(axes, ax)
    end
    linkyaxes!(axes...)
    Legend(fig[2, 1:3], axes[1], "Model", orientation=:horizontal, titleposition=:left, framevisible=false)
    fig
end

function cashDividendErrorFigure(x, y, group; xlabel="Strike", ylabel)
    fig = bookFigure(size=(700, 350))
    ax = bookAxis(fig[1, 1], xlabel=xlabel, ylabel=ylabel, yscale=log10)
    groupLines!(ax, x, y, group, order=CASH_DIVIDEND_MODELS)
    Legend(fig[1, 2], ax, "Model", framevisible=false)
    return fig
end

#basket and PLN approximations of the many dividends (Gocsei-Sahel) and extreme (Zhang) examples, as in test/pln_european_test.jl
const MULTI_DIV_MODELS = ["Deelstra", "GS", "LL2", "LL3", "VL2", "VL3"]
const TS = AQFED.TermStructure
multiDivPricers(deelstra) = Dict("Deelstra" => deelstra, "GS" => AQFED.PLN.GocseiSahelPLNPricer(),
    "LL2" => AQFED.PLN.LeFlochLehmanPLNPricer(2), "LL3" => AQFED.PLN.LeFlochLehmanPLNPricer(3),
    "VL2" => AQFED.Basket.VorstLevyExpansion(2), "VL3" => AQFED.Basket.VorstLevyExpansion(3))

@compute "gocsei_10y_basket" begin
    spot, σ, r, tte = 100.0, 0.25, 0.03, 10.0
    dividends = [TS.CapitalizedDividend(TS.Dividend(2.0, 1 / 365 + (i - 1) / 2, 1 / 365 + (i - 1) / 2, false, false), exp((tte - 1 / 365 - (i - 1) / 2) * r)) for i in 1:20]
    rawForward, df = spot * exp(r * tte), exp(-r * tte)
    f = rawForward - sum(TS.futureValue(cd) for cd in dividends)
    strikes = collect(LinRange(50.0, 200.0, 25))
    #reference prices of the test (TR-BDF2 with 4001 space steps and 1461 time steps)
    refPrices = [37.25774928130802, 34.82295687801383, 32.578012089019964, 30.507029047577678, 28.59517673673863, 26.828758256303566, 25.195214009920473, 23.6830812789917, 22.281930279673148, 20.9822890825383, 19.775564934794758, 18.65396647497763, 17.6104294093903, 16.63854701151422, 15.732506053255964, 14.887028315556972, 14.097317550062657, 13.35901160784934, 12.668139369769055, 12.02108208013933, 11.414538681002798, 10.845494757468998, 10.311194727664201, 9.809116938376736, 9.336951357363082]
    refVols = AQFED.Black.impliedVolatility.(true, refPrices, f, strikes, tte, df)
    out = DataFrame()
    for (name, pricer) in multiDivPricers(AQFED.Basket.DeelstraBasketPricer(3, 3, AQFED.Math.GaussLegendre(33)))
        prices = [AQFED.PLN.priceEuropean(pricer, true, k, rawForward, σ^2 * tte, tte, df, dividends) for k in strikes]
        vols = AQFED.Black.impliedVolatility.(true, prices, f, strikes, tte, df)
        append!(out, DataFrame(Model=name, Strike=strikes, VolError=abs.(vols .- refVols) .* 100))
    end
    writetable("gocsei_10y_basket.csv", out)
end

@compute "zhang_extreme_basket" begin
    spot, σ, r, tte = 100.0, 0.8, 0.05, 1.0
    dividends = [TS.CapitalizedDividend(TS.Dividend(25.0, t, t, false, false), exp((tte - t) * r)) for t in (0.3, 0.7)]
    strikes = collect(LinRange(50.0, 200.0, 16))
    model = TS.TSBlackModel(TS.FlatSurface(σ), TS.ConstantRateCurve(r), TS.ConstantRateCurve(r))
    refPrices = [AQFED.FDM.priceTRBDF2(AQFED.FDM.KreissSmoothDefinition(AQFED.FDM.VanillaEuropean(true, k, tte)), spot, model, dividends,
        M=4001, N=365 * 2 + 1, grid=AQFED.FDM.UniformGrid(false), solverName="TDMA", ndev=6)(spot) for k in strikes]
    out = DataFrame()
    for (name, pricer) in multiDivPricers(AQFED.Basket.DeelstraBasketPricer(1, 3, AQFED.Math.GaussLegendre(33)))
        prices = [AQFED.PLN.priceEuropean(pricer, true, k, spot * exp(tte * r), σ^2 * tte, tte, exp(-r * tte), dividends) for k in strikes]
        append!(out, DataFrame(Model=name, Strike=strikes, Error=abs.(prices .- refPrices)))
    end
    writetable("zhang_extreme_basket.csv", out)
end

function multiDivErrorFigure(x, y, group; ylabel)
    fig = bookFigure(size=(525, 262))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel=ylabel, yscale=log10)
    groupLines!(ax, x, y, group, order=MULTI_DIV_MODELS)
    Legend(fig[2, 1], ax, "Model", orientation=:horizontal, titleposition=:left, framevisible=false)
    return fig
end
@figure "gocsei_10y" begin
    df = readcsv("gocsei_10y_basket.csv")
    multiDivErrorFigure(df.Strike, df.VolError .+ 1e-12, df.Model, ylabel="Error in implied volatility %")
end
@figure "zhang_extreme" begin
    df = readcsv("zhang_extreme_basket.csv")
    multiDivErrorFigure(df.Strike, df.Error .+ 1e-12, df.Model, ylabel="Absolute error in price")
end
