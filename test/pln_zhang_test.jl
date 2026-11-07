using AQFED, Test
import AQFED.PLN: ZhangPLNPricer, LaborderePLNPricer, HHLPLNPricer, LIQUIDATOR
import AQFED.TermStructure: CapitalizedDividend, Dividend
import AQFED.Basket: VorstLevyExpansion

capitalizedDividends(r, tte, exDates, amount) = [CapitalizedDividend(Dividend(amount, t, t, false, false), exp(r * (tte - t))) for t in exDates]

@testset "Zhang paper" begin
    #Zhang (2011), call of strike 100 with σ=30% and r=5%
    r = 0.05
    dividends = capitalizedDividends(r, 1.0, [0.3, 0.7], 2.5)
    price(p) = AQFED.PLN.priceEuropean(p, true, 100.0, 100exp(r), 0.09, 1.0, exp(-r), dividends)
    @test isapprox(price(ZhangPLNPricer(1)), 11.63176, atol=1e-5)
    @test isapprox(price(ZhangPLNPricer(2)), 11.63398, atol=1e-5)
    @test abs(price(LaborderePLNPricer()) - price(VorstLevyExpansion(3))) < 2e-3
    dividends = capitalizedDividends(r, 3.0, [0.5, 1.5, 2.5], 5.0)
    price3(p) = AQFED.PLN.priceEuropean(p, true, 100.0, 100exp(3r), 0.27, 3.0, exp(-3r), dividends)
    @test isapprox(price3(ZhangPLNPricer(1)), 19.04039, atol=1e-5)
    @test abs(price3(ZhangPLNPricer(2)) - price3(VorstLevyExpansion(3))) < 1e-4
    @test abs(price3(LaborderePLNPricer()) - price3(VorstLevyExpansion(3))) < 2e-2
end

@testset "Zhang Labordere single dividend" begin
    r, tte = 0.06, 1.0
    dividends = capitalizedDividends(r, tte, [0.5], 7.0)
    forward = 100exp(r * tte)
    hhl = HHLPLNPricer(AQFED.Math.GaussLegendre(101), LIQUIDATOR)
    for strike in LinRange(50.0, 200.0, 4), isCall in (true, false)
        ref = AQFED.PLN.priceEuropean(hhl, isCall, strike, forward, 0.09, tte, exp(-r * tte), dividends)
        @test abs(AQFED.PLN.priceEuropean(ZhangPLNPricer(2), isCall, strike, forward, 0.09, tte, exp(-r * tte), dividends) - ref) < 2e-5
        @test abs(AQFED.PLN.priceEuropean(LaborderePLNPricer(), isCall, strike, forward, 0.09, tte, exp(-r * tte), dividends) - ref) < 2e-3
    end
    #put-call parity holds by construction for the first order formula and Labordère
    for p in (ZhangPLNPricer(1), LaborderePLNPricer())
        c = AQFED.PLN.priceEuropean(p, true, 100.0, forward, 0.09, tte, exp(-r * tte), dividends)
        put = AQFED.PLN.priceEuropean(p, false, 100.0, forward, 0.09, tte, exp(-r * tte), dividends)
        @test isapprox(c - put, exp(-r * tte) * (forward - 7exp(r * 0.5) - 100), atol=1e-10)
    end
end
