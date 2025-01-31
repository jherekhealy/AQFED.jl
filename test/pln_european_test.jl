    using AQFED, Test, ForwardDiff, AQFED.Math
    import AQFED.PLN: EtoreGobetPLNPricer, LeFlochLehmanPLNPricer, priceEuropean
    import AQFED.TermStructure: CapitalizedDividend, Dividend, futureValue, TSBlackModel, FlatSurface, ConstantRateCurve
    using AQFED.Basket, AQFED.Black

@testset "EtoreGobetSingle" begin
    spot = 100.0
    vol = 0.3
    discountRate = 0.06
    divAmount = 7.0
    tte = 1.0
    ttd = 0.5 * tte
    ttp = tte
    isCall = true
    discountFactor = exp(-discountRate * ttp)
    dividends = Vector{CapitalizedDividend{Float64}}(undef, 1)
    #amount::T        exDate::Float64        payDate::Float64        isProportional::bool        isKnown::bool
    dividends[1] =
        CapitalizedDividend(Dividend(divAmount, ttd, ttd, false, false), exp((tte - ttd) * discountRate))
        forward = spot*exp(tte*discountRate)-futureValue(dividends[1])
        println("S Model Price Error")

            pricers = Dict(
                "Ju" => JuBasketPricer(),
                "LB" => DeelstraLBBasketPricer(1, 3),
                "VG2" => VorstGeometricExpansion(2),
                "Deelstra" => DeelstraBasketPricer(1, 3, AQFED.Math.GaussLegendre(101)),
                "VG3" => VorstGeometricExpansion(3),
                "VG1" => VorstGeometricExpansion(1),
                "VL3" => AQFED.Basket.VorstLevyExpansion(3),
                "VL2" => AQFED.Basket.VorstLevyExpansion(2),
                "VL1" => AQFED.Basket.VorstLevyExpansion(1),
                "EG2" => EtoreGobetPLNPricer(2),
                "EG3" => EtoreGobetPLNPricer(3),
                "LL2" => LeFlochLehmanPLNPricer(2),
                "LL3" => LeFlochLehmanPLNPricer(3),
                "GS" => AQFED.PLN.GocseiSahelPLNPricer(),
                "HHL" => AQFED.PLN.HHLPLNPricer(AQFED.Math.GaussLegendre(101), AQFED.PLN.LIQUIDATOR)
            )
            priceMap = Dict()
            refHHL = [46.22595216815442, 11.106242462849208, 1.4707115939143225, 0.16022461133659802]
            refEg2 = [46.22620327713481, 11.10600057587874, 1.4707576523468313, 0.16029402314579544]
            refEg3 = [46.22596698452294, 11.106237471193687, 1.4707153344991828, 0.16022404268086038]
            strikes = LinRange(50.0, 200.0, 4)
            for (pricerName, pricer) = pricers
                prices = map(strike -> AQFED.PLN.priceEuropean(
                        pricer,
                        isCall,
                        strike,
                        spot * exp(tte * discountRate),
                        vol * vol * tte,
                        tte,
                        discountFactor,
                        dividends,
                    ), strikes)
                priceMap[pricerName] = prices
            end
            for (pricerName, pricer) = pricers
        #     println(pricerName, " ", prices - refHHL)
        prices = priceMap[pricerName]
        println(pricerName, " ", prices, " ", prices-priceMap["HHL"], @.(AQFED.Black.impliedVolatility(isCall,prices,forward,strikes,tte,discountFactor)-AQFED.Black.impliedVolatility(isCall,priceMap["HHL"],forward,strikes,tte,discountFactor))*100)
            end
            @test isapprox(refEg2, priceMap["EG2"], atol=1e-8)
            @test isapprox(refEg3, priceMap["EG3"], atol=1e-8)

            ttdList = LinRange(0.01,0.99,99)
            strikeList = [50.0, 100.0, 150.0]   
            forwardList = @. spot*exp(tte*discountRate)-divAmount*exp((tte-ttdList)*discountRate)
            df = DataFrame()
            for strike = strikeList
                priceMap = Dict()
                for (pricerName, pricer) = pricers
                prices = map(ttd -> AQFED.PLN.priceEuropean(
                        pricer,
                        isCall,
                        strike,
                        spot * exp(tte * discountRate),
                        vol * vol * tte,
                        tte,
                        discountFactor,
                        [CapitalizedDividend(Dividend(divAmount, ttd, ttd, false, false), exp((tte - ttd) * discountRate))]    ,
                    ), ttdList)
                priceMap[pricerName] = prices
            end    
            volMap = Dict()
            for (pricerName, pricer) = pricers
                prices = priceMap[pricerName]       
                volMap[pricerName] =     @.(AQFED.Black.impliedVolatility(isCall,prices,forwardList,strike,tte,discountFactor)-AQFED.Black.impliedVolatility(isCall,priceMap["HHL"],forwardList,strike,tte,discountFactor))*100
            end
            for (pricerName,pricer) = pricers
                        dfi = DataFrame(Model=pricerName,Strike=strike, Td=ttdList, VolError=abs.(volMap[pricerName]))
                        df = vcat(df, dfi)
            end
        end
            
        #=
        using CairoMakie
            f = Figure();
    ax = Axis(f[1,1],xlabel="Dividend Ex-Date", ylabel="Error in Implied Volatility %", yscale=log10)
    for pricerName = ["Deelstra", "GS", "LL2", "LL3", "VL2","VL3", "Ju"]
        lines!(ax, ttdList, abs.(volMap[pricerName]), label=pricerName)
    endc
    axislegend(; position = :cb, nbanks = 3)
    ylims!(1e-8, 1e-2)
                    
    dfn = df[df.Model .=="LL2" .|| df.Model .== "LL3" .|| df.Model .== "GS" .|| df.Model .== "Deelstra" .|| df.Model .== "Ju" .|| df.Model .== "VL2" .|| df.Model .== "VL3",:]
dfn.StrikeName = string.("Strike = ",dfn.Strike)
    Td = :Td => (t -> t) => "Dividend ex-date in years"
    VolError = :VolError => (t -> t) => "Absolute Error in Volatility %"
    set_theme!(theme_light())
    plt = data(dfn) * mapping(Td, VolError) * mapping(color= :Model) * mapping(col= :StrikeName)  * visual(Lines)
    fig = draw(plt, legend=(position=:bottom,titleposition=:left), axis=(yscale=log10,limits=(nothing,(1e-8,1e-2)),width=250,height=250,))
    save("/home/niamh/mypapers/eqd_book/single_r6_strike100_td.pdf", fig)
    =#
    end

@testset "ZhangExtreme" begin
        spot = 100.0
        σ = 0.8
        r = 0.05
        q = 0.0
        isCall = true
        tte = 1.0
        dividends = [CapitalizedDividend(Dividend(25.0, 0.3, 0.3, false, false), exp((tte - 0.3) * r)),
            CapitalizedDividend(Dividend(25.0, 0.7, 0.7, false, false), exp((tte - 0.7) * r))]

        pricers = Dict(
            "Ju" => JuBasketPricer(),
            "LB" => DeelstraLBBasketPricer(1, 3),
            "VG2" => VorstGeometricExpansion(2),
            "Deelstra" => DeelstraBasketPricer(1, 3, AQFED.Math.GaussLegendre(33)),
            "VG3" => VorstGeometricExpansion(3),
            "VG1" => VorstGeometricExpansion(1),
            "VL3" => AQFED.Basket.VorstLevyExpansion(3),
            "VL2" => AQFED.Basket.VorstLevyExpansion(2),
            "EG2" => EtoreGobetPLNPricer(2),
            "EG3" => EtoreGobetPLNPricer(3),
            "LL2" => LeFlochLehmanPLNPricer(2),
            "LL3" => LeFlochLehmanPLNPricer(3),
            "GS" => AQFED.PLN.GocseiSahelPLNPricer()
        )
        priceMap = Dict()
        strikes = LinRange(50.0, 200.0, 16)
        varianceSurface = FlatSurface(σ)
        discountCurve = ConstantRateCurve(r)
        driftCurve = ConstantRateCurve(r - q)
        modeln = TSBlackModel(varianceSurface, discountCurve, driftCurve)
        refPrices = map(strike -> AQFED.FDM.priceTRBDF2(
                AQFED.FDM.KreissSmoothDefinition(AQFED.FDM.VanillaEuropean(isCall, strike, tte)), spot, modeln, dividends, M=4001, N=365 * 2 + 1, grid=AQFED.FDM.UniformGrid(false), solverName="TDMA", ndev=6)(spot), strikes)

        for (pricerName, pricer) = pricers
            prices = map(strike -> AQFED.PLN.priceEuropean(
                    pricer,
                    isCall,
                    strike,
                    spot * exp(tte * (r - q)),
                    σ^2 * tte,
                    tte,
                    exp(-r * tte),
                    dividends,
                ), strikes)
            priceMap[pricerName] = prices
            println(pricerName, " ", prices - refPrices)
        end
        #=
    p = plot(xlab="Strike",ylab="Absolute error in price")
    for name=["VL3","VL2","LL3","EG3","Deelstra"]
       plot!(p, strikes, abs.(priceMap[name]-refPrices), label=name, linestyle=:auto)
       end
    plot!(p, yscale=:log10)


 f = Figure(size=(700,350));
 ax = Axis(f[1,1],xlabel="Strike", ylabel="Absolute Error in Price", yscale=log10)
for name = ["Deelstra", "GS",  "LL2", "LL3", "VL2","VL3"]
                lines!(ax, strikes, abs.(priceMap[name]-refPrices), label=name)
              end
              legend = Legend(f[2,1], ax, "Model", position= :cb, titleposition= :left, orientation= :horizontal)
save("/home/niamh/mypapers/eqd_book/zhang_extreme.pdf", fig)

 #axislegend(; position = :cb, nbanks = 3)
save("/home/niamh/mypapers/eqd_book/gocsei_10y.pdf", f)
 
    =#
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
    strikes = LinRange(50.0, 150.0, 11)
    f = spot * exp(discountRate * tte) - sum(futureValue(cd) for cd in dividends)

    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(discountRate)
    driftCurve = ConstantRateCurve(discountRate)
    modeln = TSBlackModel(varianceSurface, discountCurve, driftCurve)
    refPrices = map(strike -> AQFED.FDM.priceTRBDF2(
            AQFED.FDM.KreissSmoothDefinition(AQFED.FDM.VanillaEuropean(isCall, strike, tte)),
            spot, modeln, dividends, M=8001, N=701, grid=AQFED.FDM.UniformGrid(false), solverName="TDMA")(spot), strikes)

    for (i, strike) in enumerate(strikes)
        refVol = Black.impliedVolatility(isCall, refPrices[i], f, strike, tte, df)
        price = AQFED.PLN.priceEuropean(ll3, isCall, strike, rawForward, σ^2 * tte, tte, df, dividends)
        vol = Black.impliedVolatility(isCall, price, f, strike, tte, df)
        println(strike, " LL-3 ", price, " ", price / refPrices[i] - 1, " ", vol - refVol)
        price = AQFED.PLN.priceEuropean(EtoreGobetPLNPricer(3), isCall, strike, rawForward, σ^2 * tte, tte, df, dividends)
        vol = Black.impliedVolatility(isCall, price, f, strike, tte, df)
        println(strike, " EG-3 ", price, " ", price / refPrices[i] - 1, " ", vol - refVol)
        price = AQFED.PLN.priceEuropean(d, isCall, strike, rawForward, σ^2 * tte, tte, df, dividends)
        vol = Black.impliedVolatility(isCall, price, f, strike, tte, df)
        println(strike, " Deelstra ", price, " ", price / refPrices[i] - 1, " ", vol - refVol)
        price = AQFED.PLN.priceEuropean(dlb, isCall, strike, rawForward, σ^2 * tte, tte, df, dividends)
        vol = Black.impliedVolatility(isCall, price, f, strike, tte, df)
        println(strike, " LB ", price, " ", price / refPrices[i] - 1, " ", vol - refVol)
        price = AQFED.PLN.priceEuropean(VorstLevyExpansion(2), isCall, strike, rawForward, σ^2 * tte, tte, df, dividends)
        vol = Black.impliedVolatility(isCall, price, f, strike, tte, df)
        println(strike, " VL2 ", price, " ", price / refPrices[i] - 1, " ", vol - refVol)
        price = AQFED.PLN.priceEuropean(VorstLevyExpansion(3), isCall, strike, rawForward, σ^2 * tte, tte, df, dividends)
        vol = Black.impliedVolatility(isCall, price, f, strike, tte, df)
        println(strike, " VL3 ", price, " ", price / refPrices[i] - 1, " ", vol - refVol)
        price = AQFED.PLN.priceEuropean(AQFED.PLN.GocseiSahelPLNPricer(), isCall, strike, rawForward, σ^2 * tte, tte, df, dividends)
        vol = Black.impliedVolatility(isCall, price, f, strike, tte, df)
        println(strike, " GS ", price, " ", price / refPrices[i] - 1, " ", vol - refVol)
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
        AQFED.PLN.priceEuropean(p, true, x[1], x[2] * exp(x[3] * x[5]), x[4]^2 * x[3], x[3], exp(-x[3] * x[5]), [CapitalizedDividend(Dividend(baseAmount, τd, τd, false, false), exp((τ - τd) * r)),
            CapitalizedDividend(Dividend(baseAmount + 0.5, τd + 1, τd + 1, false, false), exp((τ - τd - 1) * r)),
            CapitalizedDividend(Dividend(baseAmount + 1.0, τd + 2, τd + 2, false, false), exp((τ - τd - 2) * r)),
            CapitalizedDividend(Dividend(baseAmount + 1.5, τd + 3, τd + 3, false, false), exp((τ - τd - 3) * r)),
            CapitalizedDividend(Dividend(baseAmount + 2.0, τd + 4, τd + 4, false, false), exp((τ - τd - 4) * r)),
            CapitalizedDividend(Dividend(baseAmount + 2.0, τd + 5, τd + 5, false, false), exp((τ - τd - 5) * r)),
            CapitalizedDividend(Dividend(baseAmount + 2.0, τd + 6, τd + 6, false, false), exp((τ - τd - 6) * r))])
    end
    @test isapprox(26.08099127059646, f(x), atol=1e-4)
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
    f = spot * exp(discountRate * tte) - sum(futureValue(cd) for cd in dividends)
    strikes = LinRange(50.0, 200.0, 7)
    refPrices = [37.25772649, 28.59516740, 22.28192756, 17.61042951, 14.09731856, 11.41453970, 9.33695198]
    strikes = LinRange(50.0, 200.0, 6 * 4 + 1)
    refPrices = [37.25774928130802, 34.82295687801383, 32.578012089019964, 30.507029047577678, 28.59517673673863, 26.828758256303566, 25.195214009920473, 23.6830812789917, 22.281930279673148, 20.9822890825383, 19.775564934794758, 18.65396647497763, 17.6104294093903, 16.63854701151422, 15.732506053255964, 14.887028315556972, 14.097317550062657, 13.35901160784934, 12.668139369769055, 12.02108208013933, 11.414538681002798, 10.845494757468998, 10.311194727664201, 9.809116938376736, 9.336951357363082]

    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(discountRate)
    driftCurve = ConstantRateCurve(discountRate)
    modeln = TSBlackModel(varianceSurface, discountCurve, driftCurve)

    # refPrices =map(strike -> AQFED.FDM.priceTRBDF2(
    #     AQFED.FDM.KreissSmoothDefinition(AQFED.FDM.VanillaEuropean(isCall, strike, tte)),
    #      spot,modeln, dividends, M=4001, N=365*4+1, grid=AQFED.FDM.UniformGrid(false), solverName="TDMA"),strikes)
    refVols = map((strike, p) -> Black.impliedVolatility(isCall, p, f, strike, tte, df), strikes, refPrices)
    pricers = Dict(
        "LB" => DeelstraLBBasketPricer(3, 3),
        "VG2" => VorstGeometricExpansion(2),
        "Deelstra" => DeelstraBasketPricer(3, 3, AQFED.Math.GaussLegendre(33)),
        "VG3" => VorstGeometricExpansion(3),
        "VG1" => VorstGeometricExpansion(1),
        "VL3" => AQFED.Basket.VorstLevyExpansion(3),
        "VL2" => AQFED.Basket.VorstLevyExpansion(2),
        "EG3" => EtoreGobetPLNPricer(3),
        "LL2" => LeFlochLehmanPLNPricer(2),
        "LL3" => LeFlochLehmanPLNPricer(3),
        "GS" => AQFED.PLN.GocseiSahelPLNPricer()
    )
    priceMap = Dict()
    volMap = Dict()

    for (name, pricer) = pricers
        prices = map(strike -> AQFED.PLN.priceEuropean(pricer, isCall, strike, rawForward, σ^2 * tte, tte, df, dividends), strikes)
        vols = map((strike, p) -> Black.impliedVolatility(isCall, p, f, strike, tte, df), strikes, prices)
        priceMap[name] = prices
        volMap[name] = vols
    end
    @test isapprox(refVols, volMap["LL3"], atol=1e-3)
    @test isapprox(refVols, volMap["Deelstra"], atol=1e-4)
    #=
    p = plot(xlab="Strike",ylab="Absolute error in volatility %")
    for name=["VL3","VL2","LL3","GS","EG3","Deelstra"]
           plot!(p, strikes, abs.(volMap[name]-refVols).*100, label=name, linestyle=:auto)
           end
     plot!(p, yscale=:log10)
      plot!(p,yticks=([1e-4,1e-3,1e-2,1e-1]))

 f = Figure(size=(700,350));
 ax = Axis(f[1,1],xlabel="Strike", ylabel="Error in Implied Volatility %", yscale=log10)
for pricerName = ["Deelstra", "GS",  "LL2", "LL3", "VL2","VL3"]
                lines!(ax, strikes, abs.(volMap[pricerName] - refVols).*100, label=pricerName)
              end
              legend = Legend(f[2,1], ax, "Model", position= :cb, titleposition= :left, orientation= :horizontal)

 #axislegend(; position = :cb, nbanks = 3)
save("/home/niamh/mypapers/eqd_book/gocsei_10y.pdf", f)
 plt = data(dfn) * mapping(:Strike, VolErrorPct)  * mapping(color=:Model) *  visual(Lines)
fig = draw(plt, axis=(yscale=log10,width=700,height=350,))

    =#

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
    divTimes = sort(rand(AQFED.Random.MRG32k3a(), length(dividends)) .* tte)
    # divTimes =  0.01 .+ (tte - 0.02) .* (collect(1:length(dividends)) .- 1) ./ length(dividends) #not as interesting as single div not well positioned
    for i = 1:length(dividends)
        t = divTimes[i]
        dividends[i] = CapitalizedDividend{Float64}(Dividend{Float64}(divAmount, t, t, false, false), exp((tte - t) * r))
    end
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)
    modeln = TSBlackModel(varianceSurface, discountCurve, driftCurve)

    for strike = 40.0:5:160.0
        payoffV = AQFED.FDM.VanillaEuropean(true, strike, tte)
        payoffKV = AQFED.FDM.KreissSmoothDefinition(payoffV)
        ptref = AQFED.FDM.priceTRBDF2(payoffKV, spot, modeln, dividends, M=4001, N=365 * 10 + 1, grid=AQFED.FDM.UniformGrid(false), solverName="TDMA", ndev=6)(spot)
        # ptref = AQFED.FDM.priceRKG2(payoffKV, spot,modeln, dividends, M=20001, N=365*10+1, grid=AQFED.FDM.UniformGrid(false), ndev=6)(spot)     
        pgl33 = DeelstraBasketPricer(1, 3, GaussLegendre(32))
        fgl33 = function (spot)
            AQFED.PLN.priceEuropean(pgl33, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        pts25 = DeelstraBasketPricer(1, 3, TanhSinh(32, 1e-9))
        fts25 = function (spot)
            AQFED.PLN.priceEuropean(pts25, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        pcc25 = DeelstraBasketPricer(1, 3, Chebyshev{Float64,2}(32))
        fcc25 = function (spot)
            AQFED.PLN.priceEuropean(pcc25, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        pll = LeFlochLehmanPLNPricer(3)
        fll = function (spot)
            AQFED.PLN.priceEuropean(pll, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
        end
        priceGL33 = fgl33(spot)
        priceTS25 = fts25(spot)
        priceCC25 = fcc25(spot)
        priceLL = fll(spot)
        println(nDividends, " ", strike, " BB-L33 ", priceGL33, " ", priceGL33 - ptref)
        println(nDividends, " ", strike, " BB-TS33 ", priceTS25, " ", priceTS25 - ptref)
        println(nDividends, " ", strike, " BB-CC33 ", priceCC25, " ", priceCC25 - ptref)
        println(nDividends, " ", strike, " LL-3 ", priceLL, " ", priceLL - ptref)
        @test isapprox(ptref, priceGL33, atol=1e-4)
        @test isapprox(ptref, priceTS25, atol=1e-4)
        @test isapprox(ptref, priceCC25, atol=1e-4)
        @test isapprox(ptref, priceLL, atol=1e-4)
    end
end

@testset "OneToHundred" begin
    σ = 0.3
    r = 0.05
    q = 0.0
    tte = 2.0
    ttp = tte
    isCall = true
    nDividendList = [1, 10, 100, 1000]
    strikes = [50.0, 100.0, 150.0]
    spot = 100.0
    strike = 50.0
    nDividends = 100
    for strike in strikes
        for nDividends in nDividendList
            dividends = Array{CapitalizedDividend{Float64}}(undef, nDividends)
            divAmount = 10.0 / length(dividends)
            divTimes = sort(rand(AQFED.Random.MRG32k3a(), length(dividends)) .* tte)
            # divTimes =  0.01 .+ (tte - 0.02) .* (collect(1:length(dividends)) .- 1) ./ length(dividends) #not as interesting as single div not well positioned
            for i = 1:length(dividends)
                t = divTimes[i]
                dividends[i] = CapitalizedDividend{Float64}(Dividend{Float64}(divAmount, t, t, false, false), exp((tte - t) * r))
            end
            pgl128 = DeelstraBasketPricer(1, 3, GaussLegendre(32 * 4))
            fgl128 = function (spot)
                AQFED.PLN.priceEuropean(pgl128, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
            end
            pgl33 = DeelstraBasketPricer(1, 3, GaussLegendre(33))
            fgl33 = function (spot)
                AQFED.PLN.priceEuropean(pgl33, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
            end
            pgk = DeelstraBasketPricer(1, 3, GaussKronrod(1e-7))
            fgk = function (spot)
                AQFED.PLN.priceEuropean(pgk, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
            end
            pde = DeelstraBasketPricer(1, 3, DoubleExponential(1e-7))
            fde = function (spot)
                AQFED.PLN.priceEuropean(pde, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
            end
            # pts25 = DeelstraBasketPricer(1, 3,TanhSinh(32,1e-9))
            # fts25 = function (spot)
            #     AQFED.PLN.priceEuropean(pts25, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
            # end
            # pts25p = DeelstraBasketPricer(1, 3,TanhSinh(32,1e-9,true))
            # fts25p = function (spot)
            #     AQFED.PLN.priceEuropean(pts25p, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
            # end
            plb = DeelstraLBBasketPricer(1, 3)
            flb = function (spot)
                AQFED.PLN.priceEuropean(plb, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
            end
            pll = LeFlochLehmanPLNPricer(3)
            fll = function (spot)
                AQFED.PLN.priceEuropean(pll, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
            end
            pll2 = LeFlochLehmanPLNPricer(2)
            fll2 = function (spot)
                AQFED.PLN.priceEuropean(pll2, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
            end
            pvl2 = VorstLevyExpansion(2)
            fvl2 = function (spot)
                AQFED.PLN.priceEuropean(pvl2, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
            end
            pvl3 = VorstLevyExpansion(3)
            fvl3 = function (spot)
                AQFED.PLN.priceEuropean(pvl3, true, strike, spot * exp(r * tte), σ^2 * tte, tte, exp(-r * tte), dividends)
            end
            payoff = AQFED.FDM.VanillaEuropean(true, strike, tte)
            varianceSurface = FlatSurface(σ)
            discountCurve = ConstantRateCurve(r)
            driftCurve = ConstantRateCurve(r - q)
            modeln = TSBlackModel(varianceSurface, discountCurve, driftCurve)
            payoffV = AQFED.FDM.VanillaEuropean(true, strike, tte)
            payoffKV = AQFED.FDM.KreissSmoothDefinition(payoffV)
            ptr = @time AQFED.FDM.priceTRBDF2(payoffKV, spot, modeln, dividends, M=501, N=101, grid=AQFED.FDM.UniformGrid(false), solverName="TDMA")(spot)
            ptref = @time AQFED.FDM.priceTRBDF2(payoffKV, spot, modeln, dividends, M=40001, N=365 * 10 + 1, grid=AQFED.FDM.UniformGrid(false), solverName="TDMA", ndev=6)(spot)

            priceGL33 = @time fgl33(spot)
            priceGL128 = @time fgl128(spot)
            priceGK = @time fgk(spot)
            priceDE = @time fde(spot)
            # priceTS25 = @time fts25(spot)     
            # priceTS25p = @time fts25p(spot)     
            priceLB = @time flb(spot)
            priceLL = @time fll(spot)
            priceLL2 = @time fll2(spot)
            priceVL2 = @time fvl2(spot)
            priceVL3 = @time fvl3(spot)
            println(nDividends, " TRBDF2 ", ptref, " ", 0.0)
            println(nDividends, " TRBDF2 ", ptr, " ", ptr - ptref)
            println(nDividends, " BB-GL33 ", priceGL33, " ", priceGL33 - ptref)
            println(nDividends, " BB-GL128 ", priceGL128, " ", priceGL128 - ptref)
            println(nDividends, " BB-GK ", priceGK, " ", priceGK - ptref)
            println(nDividends, " BB-DE ", priceDE, " ", priceDE - ptref)
            # println(nDividends, " BB-TS25 ", priceTS25," ",priceTS25 - ptref)
            # println(nDividends, " BB-TS25p ", priceTS25p," ",priceTS25p - ptref)
            println(nDividends, " BB-LB ", priceLB, " ", priceLB - ptref)
            println(nDividends, " LL3 ", priceLL, " ", priceLL - ptref)
            println(nDividends, " LL2 ", priceLL2, " ", priceLL2 - ptref)
            println(nDividends, " VL2 ", priceVL2, " ", priceVL2 - ptref)
            println(nDividends, " VL3 ", priceVL3, " ", priceVL3 - ptref)
            # @test isapprox(price, priceLL, atol=1e-4)
            # @test isapprox(price, priceLB, atol=1e-2)
            # @test isapprox(price, priceLL2, atol=1e-2)
        end
    end
    # ppINTERPOLATION ALLOCATES MORE THAN DIERCKX AND IS 50% SLOWER ON 10k ARRAYS. eval is also very slow on 400 points, repeated, likely due to c[2,i]. Should we favor c[i,2] instead or rework internals?
    # quad interp seems more accurate in the middle (lower error in TRBDF2 with 400 points) - similar to cubic spline. C2 does not increase accuracy, but perhaps error profile is more uniform?
end

