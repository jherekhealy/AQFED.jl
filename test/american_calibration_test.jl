using AQFED, Test
using AQFED.FDM, AQFED.TermStructure, AQFED.American, AQFED.VolatilityModels
import AQFED.American: priceAmerican
import AQFED.FDM: Liquidator, Survivor, Shift
include("american_calibration_data.jl")

const americanDataDir = joinpath(@__DIR__, "data", "american")

@testset "AmericanFDPricer" begin
    #reference values from goquant (market/americanvol_test.go PriceAmericanPLN), same grid and scheme.
    #goquant puts are not used: it applies the call-like Brennan-Schwartz sweep to puts, which is less accurate.
    divs = aaplDividends(Date(2016, 2, 12), 93.7)
    market = DiscreteDividendMarket(93.7, 0.0114, 0.0114, divs)
    p = AmericanFDPricer()
    for (strike, goPrice) in ((80.0, 18.9999938758563), (95.0, 11.5064777493719), (110.0, 6.72049785402806))
        @test isapprox(priceAmerican(p, market, true, strike, 1.9342465753424658, 0.25), goPrice, atol = 1e-10)
    end
    #put without dividends, against Andersen-Lake
    r, σ, τ, spot = 0.0114, 0.25, 1.9342465753424658, 93.7
    for strike in (80.0, 95.0, 110.0)
        ref = priceAmerican(AndersenLakeRepresentation(ConstantBlackModel(σ, r, 0.0), τ, 1e-12, 21, 16, 41, 81, isCall = false), strike, spot)
        price = priceAmerican(p, DiscreteDividendMarket(spot, r, r), false, strike, τ, σ)
        priceFine = priceAmerican(AmericanFDPricer(nS = 2001, nT = 2001), DiscreteDividendMarket(spot, r, r), false, strike, τ, σ)
        @test isapprox(price, ref, atol = 3e-4)
        @test abs(priceFine - ref) < abs(price - ref) / 10
    end
    #cash dividend policies, call on a stock paying a large dividend
    cashMarket = DiscreteDividendMarket(100.0, 0.02, 0.02, [Dividend(5.0, 0.5, 0.5, false, false)])
    noDividendPrice = priceAmerican(p, DiscreteDividendMarket(100.0, 0.02, 0.02), true, 100.0, 1.0, 0.3)
    prices = [priceAmerican(AmericanFDPricer(dividendPolicy = policy), cashMarket, true, 100.0, 1.0, 0.3) for policy in (Liquidator, Survivor, Shift)]
    @test all(x -> 0 < x < noDividendPrice, prices)
    @test maximum(prices) - minimum(prices) < 1e-6 #the policies differ only for S < D, far from the spot
    #a pricer is a reusable workspace: same price on repeated calls and in a copy
    @test priceAmerican(p, market, false, 95.0, 1.5, 0.3) == priceAmerican(similar(p), market, false, 95.0, 1.5, 0.3)
end

@testset "AmericanImpliedVolatility" begin
    market = DiscreteDividendMarket(93.7, 0.0114, 0.0114, aaplDividends(Date(2016, 2, 12), 93.7))
    p = AmericanFDPricer()
    for (isCall, strike, σ) in ((true, 80.0, 0.35), (false, 95.0, 0.3), (false, 120.0, 0.25))
        price = priceAmerican(p, market, isCall, strike, 1.2, σ)
        @test isapprox(americanImpliedVolatility(p, market, isCall, price, strike, 1.2, guess = 0.2, xtol = 1e-10), σ, atol = 1e-8)
    end
    @test isnan(americanImpliedVolatility(p, market, false, 120.0 - 93.7, 120.0, 1.2)) #exercise value only
end

@testset "AAPL Jan 2018 dividend calibration" begin
    section, valuationDate = loadAmericanSection(joinpath(americanDataDir, "aapl_20160212_20180119.csv"))
    @test isapprox(section.τ, 1.9369863013698629, atol = 1e-15)
    market0 = DiscreteDividendMarket(section.spot, 0.0114, 0.0114, aaplDividends(valuationDate, section.spot))
    cal = calibrateAmericanSection(section, market0, pricer = AmericanFDPricer())
    @test isapprox(cal.market.dividends[1].amount, 0.0053, atol = 2e-5) #goquant: 0.005299721394 with ATM strikes 92.5,95,97.5
    @test all(isapprox.(cal.atmVols, [0.3331, 0.3285, 0.3259], atol = 1e-3))
    #implied vols for the goquant calibrated dividend against goquant (TestCalibrateCash): calls agree up to the solver tolerance.
    goMarket = DiscreteDividendMarket(section.spot, 0.0114, 0.0114, aaplDividends(valuationDate, section.spot, amount = 0.005299721394 * section.spot))
    vols = americanImpliedVolatilities(AmericanFDPricer(), section, goMarket)
    goCallVols = Dict(47.5 => (0.3570835958, NaN, 0.4107970178), 95.0 => (0.3248800714, 0.3183519695, 0.3314136979), 150.0 => (0.2966433623, 0.292807625, 0.3004138369))
    for v in vols
        if haskey(goCallVols, v.strike)
            goVols = goCallVols[v.strike]
            @test isapprox(v.callMid, goVols[1], atol = 2e-6)
            @test isnan(goVols[2]) ? isnan(v.callBid) : isapprox(v.callBid, goVols[2], atol = 2e-6)
            @test isapprox(v.callAsk, goVols[3], atol = 2e-6)
        end
    end
    f = forward(cal.market, section.τ)
    otm = [otmVolatility(v, f) for v in cal.vols]
    @test all(x -> 0.25 < x[1] < 0.45, filter(x -> isfinite(x[1]), otm))
end

@testset "AAPL Mar 2016 growth calibration" begin
    section, valuationDate = loadAmericanSection(joinpath(americanDataDir, "aapl_20160212_20160304.csv"))
    market0 = DiscreteDividendMarket(section.spot, 0.0114, 0.0114, aaplDividends(valuationDate, section.spot))
    p = AmericanFDPricer()
    cal = calibrateAmericanSection(section, market0, pricer = p) #no dividend before the maturity: the growth rate is calibrated
    calJZ = calibrateAmericanSection(section, market0, pricer = p, carryPricer = JuZhongPricer())
    @test isapprox(cal.market.μ, -0.0042, atol = 1e-4) #goquant (Ju-Zhong, ATM strikes 93.5,94,94.5): -0.004263843911
    @test isapprox(calJZ.market.μ, cal.market.μ, atol = 1e-4)
    goMarket = DiscreteDividendMarket(section.spot, 0.0114, -0.004263843911, market0.dividends)
    vols = americanImpliedVolatilities(p, section, goMarket)
    v95 = vols[findfirst(v -> v.strike == 95.0, vols)]
    @test isapprox(v95.callMid, 0.3161635784, atol = 2e-6)
    @test isapprox(v95.putMid, 0.3177170311, atol = 2e-6)
end
