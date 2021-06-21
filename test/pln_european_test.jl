using AQFED, Test
import AQFED.PLN: EtoreGobetPLNPricer, LeFlochLehmanPLNPricer, priceEuropean
import AQFED.TermStructure: CapitalizedDividend, Dividend

@testset "EtoreGobetSingle" begin
    spot = 100.0
    vol = 0.3
    discountRate = 0.0
    divAmount = 7.0
    tte = 1.0
    ttd = 0.9 * tte
    ttp = tte
    isCall = true
    discountFactor = exp(-discountRate * ttp)
    dividends = Vector{CapitalizedDividend{Float64}}(undef, 1)
    #amount::T        exDate::Float64        payDate::Float64        isProportional::bool        isKnown::bool
    dividends[1] =
        CapitalizedDividend(Dividend{Float64}(divAmount, ttd, ttd, false, false), exp((tte - ttd) * discountRate))
    println("S Model Price Error")
    eg2 = EtoreGobetPLNPricer(2)
    eg3 = EtoreGobetPLNPricer(3)

    ll2 = LeFlochLehmanPLNPricer(2)
    ll3 = LeFlochLehmanPLNPricer(3)
    refHHL = [43.24845580, 9.07480013, 1.06252880, 0.10473887]
    refEg2 = [43.24846889, 9.07479039, 1.06253441, 0.10474035]
    refEg3 = [43.24845582, 9.07480026, 1.06252875, 0.10473885]
    for (i, strike) in enumerate(LinRange(50.0, 200.0, 4))
        price = priceEuropean(
            eg2,
            isCall,
            strike,
            spot * exp(tte * discountRate),
            vol * vol * tte,
            tte,
            discountFactor,
            dividends,
        )
        println(strike, " EG-2 ", price, " ", price-refHHL[i])
        @test isapprox(refEg2[i], price, atol = 1e-8)
        price = priceEuropean(
            eg3,
            isCall,
            strike,
            spot * exp(tte * discountRate),
            vol * vol * tte,
            tte,
            discountFactor,
            dividends,
        )
        println(strike, " EG-3 ", price, " ", price-refHHL[i])
        @test isapprox(refEg3[i], price, atol = 1e-8)
        price = priceEuropean(
            ll2,
            isCall,
            strike,
            spot * exp(tte * discountRate),
            vol * vol * tte,
            tte,
            discountFactor,
            dividends,
        )
        println(strike, " LL-2 ", price, " ", price-refHHL[i])
        price = priceEuropean(
            ll3,
            isCall,
            strike,
            spot * exp(tte * discountRate),
            vol * vol * tte,
            tte,
            discountFactor,
            dividends,
        )
        println(strike, " LL-3 ", price, " ", price-refHHL[i])
    end

end
