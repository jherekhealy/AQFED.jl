using AQFED, Test
import AQFED.PLN: EtoreGobetPLNPricer, LeFlochLehmanPLNPricer, priceEuropean
import AQFED.TermStructure: CapitalizedDividend, Dividend, futureValue
import AQFED.Basket: DeelstraBasketPricer, DeelstraLBBasketPricer

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
    d = DeelstraBasketPricer(1, 3)
    dlb = DeelstraLBBasketPricer(1, 3)
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
        println(strike, " EG-2 ", price, " ", price - refHHL[i])
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
        println(strike, " EG-3 ", price, " ", price - refHHL[i])
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
        println(strike, " LL-2 ", price, " ", price - refHHL[i])
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
        println(strike, " LL-3 ", price, " ", price - refHHL[i])
        price = AQFED.Basket.priceEuropean(
            d,
            isCall,
            strike,
            spot * exp(tte * discountRate),
            vol * vol * tte,
            tte,
            discountFactor,
            dividends,
        )
        println(strike, " Deelstra ", price, " ", price - refHHL[i])
        price = AQFED.Basket.priceEuropean(
            dlb,
            isCall,
            strike,
            spot * exp(tte * discountRate),
            vol * vol * tte,
            tte,
            discountFactor,
            dividends,
        )
        println(strike, " Deelstra-LB ", price, " ", price - refHHL[i])

    end

end



@testset "MultipleVellekoop" begin
    spot = 100.0
    σ = 0.25
    discountRate = 0.06
    tte = 7.0
    isCall = true
    ttd = 0.9
    dividends = Vector{CapitalizedDividend{Float64}}(undef, 7)
    dividends[1] = CapitalizedDividend(Dividend{Float64}(6.0, ttd, ttd, false, false), exp((tte - ttd) * discountRate))
    dividends[2] =
        CapitalizedDividend(Dividend{Float64}(6.5, ttd + 1, ttd + 1, false, false), exp((tte - ttd - 1) * discountRate))
    dividends[3] =
        CapitalizedDividend(Dividend{Float64}(7.0, ttd + 2, ttd + 2, false, false), exp((tte - ttd - 2) * discountRate))
    dividends[4] =
        CapitalizedDividend(Dividend{Float64}(7.5, ttd + 3, ttd + 3, false, false), exp((tte - ttd - 3) * discountRate))
    dividends[5] =
        CapitalizedDividend(Dividend{Float64}(8.0, ttd + 4, ttd + 4, false, false), exp((tte - ttd - 4) * discountRate))
    dividends[6] =
        CapitalizedDividend(Dividend{Float64}(8.0, ttd + 5, ttd + 5, false, false), exp((tte - ttd - 5) * discountRate))
    dividends[7] =
        CapitalizedDividend(Dividend{Float64}(8.0, ttd + 6, ttd + 6, false, false), exp((tte - ttd - 6) * discountRate))
    rawForward = spot * exp(discountRate * tte)
    df = exp(-discountRate * tte)
    ll3 = LeFlochLehmanPLNPricer(3)
    d = DeelstraBasketPricer(3, 3)
    dlb = DeelstraLBBasketPricer(3, 3)
    refPrices = [
        34.19664044,
        30.49396147,
        27.21393776,
        24.31297892,
        21.74885502,
        19.48228598,
        17.47762235,
        15.70300783,
        14.13025908,
        12.73460326,
        11.49435473,
    ]
    f = spot * exp(discountRate * tte) - sum(futureValue(cd) for cd in dividends)

    for (i, strike) in enumerate(LinRange(50.0, 150.0, 11))
        refVol = Black.impliedVolatility(isCall, refPrices[i], f, strike, tte, df)
        price = priceEuropean(ll3, isCall, strike, rawForward, σ^2 * tte, tte, df, dividends)
        vol = Black.impliedVolatility(isCall, price, f, strike, tte, df)
        println(strike, " LL-3 ", price, " ", price / refPrices[i] - 1, " ", vol - refVol)
        price = AQFED.Basket.priceEuropean(d, isCall, strike, rawForward, σ^2 * tte, tte, df, dividends)
        vol = Black.impliedVolatility(isCall, price, f, strike, tte, df)
        println(strike, " Deelstra ", price, " ", price / refPrices[i] - 1, " ", vol - refVol)
        price = AQFED.Basket.priceEuropean(dlb, isCall, strike, rawForward, σ^2 * tte, tte, df, dividends)
        vol = Black.impliedVolatility(isCall, price, f, strike, tte, df)
        println(strike, " Deelstra-LB ", price, " ", price / refPrices[i] - 1, " ", vol - refVol)

    end
end




@testset "MultipleGocsei" begin
    spot = 100.0
    σ = 0.25
    discountRate = 0.03
    divAmount = 2.0
    tte = 10.0
    isCall = true
    ttd0 = 1.0 / 365
    dividends = Vector{CapitalizedDividend{Float64}}(undef, 20)
    for i = 1:20
        ttd = ttd0 + (i - 1) / 2
        dividends[i] =
            CapitalizedDividend(Dividend{Float64}(divAmount, ttd, ttd, false, false), exp((tte - ttd) * discountRate))
    end
    rawForward = spot * exp(discountRate * tte)
    df = exp(-discountRate * tte)
    ll3 = LeFlochLehmanPLNPricer(3)
    d = DeelstraBasketPricer(1, 3) #1,3 minimizes relative error in price, 3,3 minimizes volatility error
    dlb = DeelstraLBBasketPricer(1, 3)
    refPrices = [37.25772649, 28.59516740, 22.28192756, 17.61042951, 14.09731856, 11.41453970, 9.33695198]
    f = spot * exp(discountRate * tte) - sum(futureValue(cd) for cd in dividends)

    for (i, strike) in enumerate(LinRange(50.0, 200.0, 7))
        refVol = Black.impliedVolatility(isCall, refPrices[i], f, strike, tte, df)
        price = priceEuropean(ll3, isCall, strike, rawForward, σ^2 * tte, tte, df, dividends)
        vol = Black.impliedVolatility(isCall, price, f, strike, tte, df)
        println(strike, " LL-3 ", price, " ", price / refPrices[i] - 1, " ", vol - refVol)
        price = AQFED.Basket.priceEuropean(d, isCall, strike, rawForward, σ^2 * tte, tte, df, dividends)
        vol = Black.impliedVolatility(isCall, price, f, strike, tte, df)
        println(strike, " Deelstra ", price, " ", price / refPrices[i] - 1, " ", vol - refVol)
        price = AQFED.Basket.priceEuropean(dlb, isCall, strike, rawForward, σ^2 * tte, tte, df, dividends)
        vol = Black.impliedVolatility(isCall, price, f, strike, tte, df)
        println(strike, " Deelstra-LB ", price, " ", price / refPrices[i] - 1, " ", vol - refVol)

    end
end
