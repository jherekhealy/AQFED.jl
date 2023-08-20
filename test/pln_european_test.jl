using AQFED, Test, ForwardDiff
import AQFED.PLN: EtoreGobetPLNPricer, LeFlochLehmanPLNPricer, priceEuropean
import AQFED.TermStructure: CapitalizedDividend, Dividend, futureValue, TSBlackModel, FlatSurface, ConstantRateCurve
import AQFED.Basket: DeelstraBasketPricer, DeelstraLBBasketPricer, GaussLegendre, GaussKronrod, DoubleExponential, TanhSinh

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
        CapitalizedDividend(Dividend(divAmount, ttd, ttd, false, false), exp((tte - ttd) * discountRate))
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
        @test isapprox(refEg2[i], price, atol=1e-8)
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
        @test isapprox(refEg3[i], price, atol=1e-8)
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
    dividends = [CapitalizedDividend(Dividend(6.0, ttd, ttd, false, false), exp((tte - ttd) * discountRate)),
        CapitalizedDividend(Dividend(6.5, ttd + 1, ttd + 1, false, false), exp((tte - ttd - 1) * discountRate)),
        CapitalizedDividend(Dividend(7.0, ttd + 2, ttd + 2, false, false), exp((tte - ttd - 2) * discountRate)),
        CapitalizedDividend(Dividend(7.5, ttd + 3, ttd + 3, false, false), exp((tte - ttd - 3) * discountRate)),
        CapitalizedDividend(Dividend(8.0, ttd + 4, ttd + 4, false, false), exp((tte - ttd - 4) * discountRate)),
        CapitalizedDividend(Dividend(8.0, ttd + 5, ttd + 5, false, false), exp((tte - ttd - 5) * discountRate)),
        CapitalizedDividend(Dividend(8.0, ttd + 6, ttd + 6, false, false), exp((tte - ttd - 6) * discountRate))]
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


@testset "ForwardDiff" begin
    spot = 100.0
    strike = 70.0
    r = 0.06
    q = 0.0
    σ = 0.25
    τd = 0.5
    τ = 7.0
    x = [strike, spot, τ, σ, r, τd, 6.0]
    p = DeelstraBasketPricer(1, 3)
    f = function (x)
        τ = x[3]
        r = x[5]
        τd = x[6]
        baseAmount = x[7]
        AQFED.Basket.priceEuropean(p, true, x[1], x[2] * exp(x[3] * x[5]), x[4]^2 * x[3], x[3], exp(-x[3] * x[5]), [CapitalizedDividend(Dividend(baseAmount, τd, τd, false, false), exp((τ - τd) * r)),
            CapitalizedDividend(Dividend(baseAmount + 0.5, τd + 1, τd + 1, false, false), exp((τ - τd - 1) * r)),
            CapitalizedDividend(Dividend(baseAmount + 1.0, τd + 2, τd + 2, false, false), exp((τ - τd - 2) * r)),
            CapitalizedDividend(Dividend(baseAmount + 1.5, τd + 3, τd + 3, false, false), exp((τ - τd - 3) * r)),
            CapitalizedDividend(Dividend(baseAmount + 2.0, τd + 4, τd + 4, false, false), exp((τ - τd - 4) * r)),
            CapitalizedDividend(Dividend(baseAmount + 2.0, τd + 5, τd + 5, false, false), exp((τ - τd - 5) * r)),
            CapitalizedDividend(Dividend(baseAmount + 2.0, τd + 6, τd + 6, false, false), exp((τ - τd - 6) * r))])
    end
    @test isapprox(26.08099127059646, f(x), atol=1e-5)
    ForwardDiff.gradient(f, x)
    pll = LeFlochLehmanPLNPricer(3)
    fll = function (x)
        τ = x[3]
        r = x[5]
        τd = x[6]
        divAmount = x[7]
        AQFED.PLN.priceEuropean(pll, true, x[1], x[2] * exp(x[3] * x[5]), x[4]^2 * x[3], x[3], exp(-x[3] * x[5]), [CapitalizedDividend(Dividend(divAmount, τd, τd, false, false), exp((τ - τd) * r)),
            CapitalizedDividend(Dividend(divAmount + 0.5, τd + 1, τd + 1, false, false), exp((τ - τd - 1) * r)),
            CapitalizedDividend(Dividend(divAmount + 1.0, τd + 2, τd + 2, false, false), exp((τ - τd - 2) * r)),
            CapitalizedDividend(Dividend(divAmount + 1.5, τd + 3, τd + 3, false, false), exp((τ - τd - 3) * r)),
            CapitalizedDividend(Dividend(divAmount + 2.0, τd + 4, τd + 4, false, false), exp((τ - τd - 4) * r)),
            CapitalizedDividend(Dividend(divAmount + 2.0, τd + 5, τd + 5, false, false), exp((τ - τd - 5) * r)),
            CapitalizedDividend(Dividend(divAmount + 2.0, τd + 6, τd + 6, false, false), exp((τ - τd - 6) * r))])
    end
    @test isapprox(26.085921596965157, fll(x), atol=1e-8)
    ForwardDiff.gradient(fll, x)
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
            CapitalizedDividend(Dividend(divAmount, ttd, ttd, false, false), exp((tte - ttd) * discountRate))
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
@testset "StrikeError" begin
    σ = 0.3
    r = 0.05
    q = 0.0
    tte = 2.0
    ttp = tte
    isCall = true
    nDividends = 10
    spot = 100.0
    dividends = Array{CapitalizedDividend{Float64}}(undef, nDividends)
    divAmount = 10.0 / length(dividends)
     divTimes = sort(rand(AQFED.Random.MRG32k3a(),length(dividends)).*tte)
    # divTimes =  0.01 .+ (tte - 0.02) .* (collect(1:length(dividends)) .- 1) ./ length(dividends) #not as interesting as single div not well positioned
    for i = 1:length(dividends)            
        t = divTimes[i]
        dividends[i] = CapitalizedDividend{Float64}(Dividend{Float64}(divAmount, t, t, false, false), exp((tte - t) * r))
    end
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)
    modeln = TSBlackModel(varianceSurface, discountCurve, driftCurve)
    
    for strike=40.0:5:160.0
        payoffV = AQFED.FDM.VanillaEuropean(true, strike, tte)
        payoffKV = AQFED.FDM.KreissSmoothDefinition(payoffV)
        ptref = AQFED.FDM.priceTRBDF2(payoffKV, spot,modeln, dividends, M=40001, N=365*10+1, grid=AQFED.FDM.UniformGrid(false), solverName="TDMA", ndev=6)(spot)     
        # ptref = AQFED.FDM.priceRKG2(payoffKV, spot,modeln, dividends, M=20001, N=365*10+1, grid=AQFED.FDM.UniformGrid(false), ndev=6)(spot)     
    pgl33 = DeelstraBasketPricer(1, 3,GaussLegendre(33))
    fgl33 = function (spot)
        AQFED.Basket.priceEuropean(pgl33, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
    end
    pts25 = DeelstraBasketPricer(1, 3,TanhSinh(16,1e-9))
    fts25 = function (spot)
        AQFED.Basket.priceEuropean(pts25, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
    end
    pll = LeFlochLehmanPLNPricer(3)
        fll = function (spot)
            priceEuropean(pll, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        priceGL33 =  fgl33(spot)
        priceTS25 =  fts25(spot)     
        priceLL =  fll(spot)
        println(nDividends, " ",strike," BB-L33 ", priceGL33," ",priceGL33-ptref)
        println(nDividends, " ",strike," BB-TS33 ", priceTS25," ",priceTS25-ptref)
        println(nDividends, " ",strike," LL-3 ", priceLL," ",priceLL-ptref)
    end
end

@testset "OneToHundred" begin
    σ = 0.3
    r = 0.05
    q = 0.0
    tte = 2.0
    ttp = tte
    isCall = true
    nDividendList = [1, 10, 100,1000]
    strikes = [50.0,100.0,150.0]
    spot = 100.0
    strike=50.0; nDividends=100
    for strike in strikes 
    for nDividends in nDividendList
        dividends = Array{CapitalizedDividend{Float64}}(undef, nDividends)
        divAmount = 10.0 / length(dividends)
         divTimes = sort(rand(AQFED.Random.MRG32k3a(),length(dividends)).*tte)
        # divTimes =  0.01 .+ (tte - 0.02) .* (collect(1:length(dividends)) .- 1) ./ length(dividends) #not as interesting as single div not well positioned
        for i = 1:length(dividends)            
            t = divTimes[i]
            dividends[i] = CapitalizedDividend{Float64}(Dividend{Float64}(divAmount, t, t, false, false), exp((tte - t) * r))
        end
        pgl128 = DeelstraBasketPricer(1, 3,GaussLegendre(32*4))
        fgl128 = function (spot)
            AQFED.Basket.priceEuropean(pgl128, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        pgl33 = DeelstraBasketPricer(1, 3,GaussLegendre(33))
        fgl33 = function (spot)
            AQFED.Basket.priceEuropean(pgl33, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        pgk = DeelstraBasketPricer(1, 3,GaussKronrod(1e-7))
        fgk = function (spot)
            AQFED.Basket.priceEuropean(pgk, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        pde = DeelstraBasketPricer(1, 3,DoubleExponential(1e-7))
        fde = function (spot)
            AQFED.Basket.priceEuropean(pde, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        pts25 = DeelstraBasketPricer(1, 3,TanhSinh(16,1e-9))
        fts25 = function (spot)
            AQFED.Basket.priceEuropean(pts25, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        pts25p = DeelstraBasketPricer(1, 3,TanhSinh(16,1e-9,true))
        fts25p = function (spot)
            AQFED.Basket.priceEuropean(pts25p, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        plb = DeelstraLBBasketPricer(1, 3)
        flb = function (spot)
            AQFED.Basket.priceEuropean(plb, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        pll = LeFlochLehmanPLNPricer(3)
        fll = function (spot)
            priceEuropean(pll, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        pll2 = LeFlochLehmanPLNPricer(2)
        fll2 = function (spot)
            priceEuropean(pll2, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        payoff = AQFED.FDM.VanillaEuropean(true,strike,tte)
        varianceSurface = FlatSurface(σ)
        discountCurve = ConstantRateCurve(r)
        driftCurve = ConstantRateCurve(r - q)
        modeln = TSBlackModel(varianceSurface, discountCurve, driftCurve)
        payoffV = AQFED.FDM.VanillaEuropean(true, strike, tte)
        payoffKV = AQFED.FDM.KreissSmoothDefinition(payoffV)
        ptr = @time AQFED.FDM.priceTRBDF2(payoffKV, spot,modeln, dividends, M=501, N=101, grid=AQFED.FDM.UniformGrid(false), solverName="TDMA")(spot)     
        ptref = @time AQFED.FDM.priceTRBDF2(payoffKV, spot,modeln, dividends, M=40001, N=365*10+1, grid=AQFED.FDM.UniformGrid(false), solverName="TDMA",ndev=6)(spot)     
    
        priceGL33 = @time fgl33(spot)
        priceGL128 = @time fgl128(spot)
        priceGK = @time fgk(spot)
        priceDE = @time fde(spot)
        priceTS25 = @time fts25(spot)     
        priceTS25p = @time fts25p(spot)     
        priceLB = @time flb(spot)
        priceLL = @time fll(spot)
        priceLL2 = @time fll2(spot)
        println(nDividends, " TRBDF2 ",ptref," ",0.0)
        println(nDividends, " TRBDF2 ",ptr, " ",ptr - ptref)
        println(nDividends, " BB-GL33 ", priceGL33," ",priceGL33 - ptref)
        println(nDividends, " BB-GL128 ", priceGL128," ",priceGL128 - ptref)
        println(nDividends, " BB-GK ", priceGK," ",priceGK - ptref)
        println(nDividends, " BB-DE ", priceDE," ",priceDE-ptref)
        println(nDividends, " BB-TS25 ", priceTS25," ",priceTS25 - ptref)
        println(nDividends, " BB-TS25p ", priceTS25p," ",priceTS25p - ptref)
        println(nDividends, " BB-LB ", priceLB," ",priceLB-ptref)
        println(nDividends, " LL-3 ", priceLL," ",priceLL-ptref)
        println(nDividends, " LL-2 ", priceLL2," ",priceLL2-ptref)
        # @test isapprox(price, priceLL, atol=1e-4)
        # @test isapprox(price, priceLB, atol=1e-2)
        # @test isapprox(price, priceLL2, atol=1e-2)
    end
end
#with 40K points very difficult to see limit of accuracy of BB/LL3 with 100 dividends. cODE NEARLY UNIFORM IN JULIA.
# ppINTERPOLATION ALLOCATES MORE THAN DIERCKX AND IS 50% SLOWER ON 10k ARRAYS. eval is also very slow on 400 points, repeated, likely due to c[2,i]. Should we favor c[i,2] instead or rework internals?
# quad interp seems more accurate in the middle (lower error in TRBDF2 with 400 points) - similar to cubic spline. C2 does not increase accuracy, but perhaps error profile is more uniform?
end

