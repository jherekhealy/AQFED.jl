using AQFED, Test, AQFED.Basket, Printf, DataFrames, StatsBase, AQFED.Random

makeCorrelationMatrix(rho; dim=4) = [
    1.0 rho rho rho
    rho 1.0 rho rho
    rho rho 1.0 rho
    rho rho rho 1.0
]
makeVarianceVector(sigma, tte; dim=4, firstSigma=sigma) = [firstSigma, sigma, sigma, sigma] .^ 2 .* tte

@testset "KornTable1" begin
    spot = 80.0
    strike = 100.0
    r = -0.005
    q = -0.05
    tte = 1.0
    vol = 0.2

    refPrices = [21.6921, 25.0293, 28.0074, 30.7427, 32.0412, 33.9187]
    weight = [0.25, 0.25, 0.25, 0.25]
    spot = [100.0, 100.0, 100.0, 100.0]
    strike = 100.0
    r = 0.0
    sigma = 0.4 #0.4 relatively large for 5y, not favorable to expansions?
    rhos = [0.1, 0.3, 0.5, 0.7, 0.8, 0.95]
    tte = 5.0
    q = 0.0
    forward = spot .* exp((r - q) * tte)
    discountFactor = exp(-r * tte)
    pricers = Dict(
        "Ju" => JuBasketPricer(),
        "LB" => DeelstraLBBasketPricer(1, 3),
        "VG2" => VorstGeometricExpansion(2),
        "Deelstra" => DeelstraBasketPricer(1, 3),
        "VG3" => VorstGeometricExpansion(3),
        "VG1" => VorstGeometricExpansion(1),
        "VL3" => AQFED.Basket.VorstLevyExpansion(3),
        "VL2" => AQFED.Basket.VorstLevyExpansion(2),
        "VL1" => AQFED.Basket.VorstLevyExpansion(1),
        "SG0" => ShiftedGeometricExpansion(0,3),
        "SG1" => ShiftedGeometricExpansion(1,3),
        "SG2" => ShiftedGeometricExpansion(2,3),
        "SG3" => ShiftedGeometricExpansion(3,3),
        "3MM" => AQFED.Basket.SLN3MMBasketPricer()
    )
    priceMap = Dict()
    for (pricerName, pricer) = pricers
        prices = map(rho -> priceEuropean(pricer, true, strike, discountFactor, spot, forward, makeVarianceVector(sigma, tte), weight, makeCorrelationMatrix(rho)), rhos)
        priceMap[pricerName] = prices
    end
    for (i, rho) = enumerate(rhos)
        @printf("%1.2f & %2.3f & %2.3f & %2.3f &  %2.3f & %2.3f & %2.3f & %2.3f & %2.3f & %2.3f\\\\\n", rho, refPrices[i], priceMap["Ju"][i], priceMap["3MM"][i], priceMap["LB"][i], priceMap["Deelstra"][i], priceMap["VG1"][i], priceMap["VG2"][i], priceMap["VG3"][i], priceMap["VL3"][i])
    end
    for (name, pricer) = pricers
        allPrices = priceMap[name]
        @printf("%s %1.3f %1.3f\n", name, rmsd(allPrices, refPrices), maximum(abs.(allPrices - refPrices)))
    end
    @test isapprox(refPrices, priceMap["Deelstra"], atol=1e-1)
    @test isapprox(rmsd(priceMap["Ju"], refPrices), 0.0, atol=0.04)
    @test isapprox(rmsd(priceMap["Deelstra"], refPrices), 0.0, atol=0.04)
    @test isapprox(rmsd(priceMap["3MM"], refPrices), 0.0, atol=0.04)


    for (i, rho) in enumerate(rhos)
        correlation = [
            1.0 rho rho rho
            rho 1.0 rho rho
            rho rho 1.0 rho
            rho rho rho 1.0
        ]
        tvar = [sigma, sigma, sigma, sigma] .^ 2 .* tte
        q = 0.0
        forward = spot .* exp((r - q) * tte)
        discountFactor = exp(-r * tte)
        p = DeelstraBasketPricer(3, 3, AQFED.Basket.Simpson(4096))
        asympPrice = priceEuropean(p, true, strike, discountFactor, spot, forward, tvar, weight, correlation, isLognormal=1)
        for n = 16:16 #4:1:64  
            p = DeelstraBasketPricer(3, 3, AQFED.Basket.Chebyshev{Float64,2}(n, false))
            #    p = DeelstraBasketPricer(3, 3,AQFED.Basket.FourierBoyd(n,1.0))

            price = priceEuropean(p, true, strike, discountFactor, spot, forward, tvar, weight, correlation, isLognormal=1)
            @printf("%d %.2f Deelstra %.4f %.2e\n", n, rho, price, price - asympPrice)
            @test isapprox(asympPrice, price, atol=1e-4)
        end
    end
end

@testset "KornTable2" begin
    refPrices = [54.3098, 47.4807, 41.5221, 36.3514, 31.8764, 28.0070, 24.6601, 21.7622, 19.2489, 17.0651, 15.1636]
    weight = [0.25, 0.25, 0.25, 0.25]
    spot = [100.0, 100.0, 100.0, 100.0]
    strikes = collect(50.0:10.0:150.0)
    r = 0.0
    sigma = 0.4
    rho = 0.5
    tte = 5.0
    discountFactor = exp(-r * tte)
    pricers = Dict(
        "Ju" => JuBasketPricer(),
        "LB" => DeelstraLBBasketPricer(3, 3),
        "VG2" => VorstGeometricExpansion(2),
        "Deelstra" => DeelstraBasketPricer(3, 3,AQFED.Math.GaussLegendre(33)),
        "VG3" => VorstGeometricExpansion(3),
        "VG1" => VorstGeometricExpansion(1),
        "3MM" => AQFED.Basket.SLN3MMBasketPricer(),
        "VL3" => AQFED.Basket.VorstLevyExpansion(3),
        "VL2" => AQFED.Basket.VorstLevyExpansion(2),
    )
    priceMap = Dict()
    for (pricerName, pricer) = pricers
        prices = map(strike -> priceEuropean(pricer, true, strike, discountFactor, spot, spot, makeVarianceVector(sigma, tte), weight, makeCorrelationMatrix(rho)), strikes)
        priceMap[pricerName] = prices
    end
    @test rmsd(priceMap["Deelstra"], refPrices) < 0.02
    for (i, rho) = enumerate(strikes)
        @printf("%1.2f & %2.3f & %2.3f & %2.3f &  %2.3f & %2.3f & %2.3f & %2.3f & %2.3f & %2.3f\\\\\n", rho, refPrices[i], priceMap["Ju"][i], priceMap["3MM"][i], priceMap["LB"][i], priceMap["Deelstra"][i], priceMap["VG1"][i], priceMap["VG2"][i], priceMap["VG3"][i], priceMap["VL3"][i])
    end
    sortedNames = ["Ju", "3MM", "LB", "Deelstra", "VG1", "VG2", "VG3", "VL2", "VL3"]
    for name = sortedNames
        @printf("& %s ", name)
    end
    @printf("\\\\\n")
    for name = sortedNames
        allPrices = priceMap[name]
        @printf("& %1.3f ", rmsd(allPrices, refPrices))
    end
    @printf("\\\\\n")
    for name = sortedNames
        allPrices = priceMap[name]
        @printf("& %1.3f ", maximum(abs.(allPrices - refPrices)))
    end
    @printf("\\\\\n")
end


@testset "KornTable4" begin
    spot = 100.0
    strike = 100.0
    r = 0.00
    q = 0.0
    tte = 1.0
    vol = 0.2

    weight = [0.25, 0.25, 0.25, 0.25]
    spot = [100.0, 100.0, 100.0, 100.0]
    strike = 100.0
    r = 0.0
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0]
    rho = 0.5
    tte = 5.0

    forward = spot .* exp((r - q) * tte)
    discountFactor = exp(-r * tte)

    rng = ScrambledSobolSeq(4, 2^29, Owen(31,ScramblingRngAdapter(Chacha8SIMD(UInt32))))
    refPrices = map(sigma -> AQFED.MonteCarlo.simulate(rng, [AQFED.TermStructure.ConstantBlackModel(sigma, 0.0, 0.0), AQFED.TermStructure.ConstantBlackModel(sigma, 0.0, 0.0), AQFED.TermStructure.ConstantBlackModel(sigma, 0.0, 0.0), AQFED.TermStructure.ConstantBlackModel(sigma, 0.0, 0.0)],
            forward, makeCorrelationMatrix(rho), AQFED.MonteCarlo.VanillaBasketOption(true, strike, weight, AQFED.MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0), 1024 * 1024), sigmas)


    pricers = Dict(
        "Ju" => JuBasketPricer(),
        "LB" => DeelstraLBBasketPricer(3, 3),
        "VG2" => VorstGeometricExpansion(2),
        "Deelstra" => DeelstraBasketPricer(2, 3,AQFED.Math.DoubleExponential()),
        "VG3" => VorstGeometricExpansion(3),
        "VG1" => VorstGeometricExpansion(1),
        "3MM" => AQFED.Basket.SLN3MMBasketPricer(),
        "VL3" => AQFED.Basket.VorstLevyExpansion(3),
        "VL2" => AQFED.Basket.VorstLevyExpansion(2),
        "SG2" =>ShiftedGeometricExpansion(2,3),
        "SG1" => ShiftedGeometricExpansion(1,3),
        "SG3" => ShiftedGeometricExpansion(3,3),
     )
    priceMap = Dict()
    for (pricerName, pricer) = pricers
        prices = map(sigma -> AQFED.Basket.priceEuropean(pricer, true, strike, discountFactor, spot, forward, makeVarianceVector(sigma, tte), weight, makeCorrelationMatrix(rho)), sigmas)
        priceMap[pricerName] = prices
    end
    for (i, rho) = enumerate(sigmas)
        @printf("%1.2f & %2.3f & %2.3f & %2.3f &  %2.3f & %2.3f & %2.3f & %2.3f & %2.3f& %2.3f\\\\\n", rho, refPrices[i], priceMap["Ju"][i], priceMap["3MM"][i], priceMap["LB"][i], priceMap["Deelstra"][i], priceMap["VG1"][i], priceMap["VG2"][i], priceMap["VG3"][i], priceMap["VL3"][i])
    end
    sortedNames = ["Ju", "3MM", "LB", "Deelstra", "VG1", "VG2", "VG3", "VL2", "VL3"]
    for name = sortedNames
        @printf("& %s ", name)
    end
    @printf("\\\\\n")
    for name = sortedNames
        allPrices = priceMap[name]
        @printf("& %1.3f ", rmsd(allPrices, refPrices))
    end
    @printf("\\\\\n")
    for name = sortedNames
        allPrices = priceMap[name]
        @printf("& %1.3f ", maximum(abs.(allPrices - refPrices)))
    end
    @printf("\\\\\n")


    @test rmsd(priceMap["Deelstra"], refPrices) < 0.2
    # p =  DeelstraBasketPricer(3, 3,AQFED.Math.DoubleExponential())
    # prices = map(sigma -> priceEuropean(p, true, strike, discountFactor, spot, forward, makeVarianceVector(sigma, tte), weight, makeCorrelationMatrix(0.5)), sigmas)
    # for (sigma, price, refPrice) = zip(sigmas, prices, refPrices)
    #     @printf("%.2f Deelstra %.4f %.2e\n", sigma, price, price - refPrice)
    # end
    # @test isapprox(refPrices, prices, atol=1e-1)
end

@testset "KornTable5" begin
    spot = 100.0
    strike = 100.0
    r = 0.00
    q = 0.0
    tte = 1.0
    vol = 0.2

    weight = [0.25, 0.25, 0.25, 0.25]
    spot = [100.0, 100.0, 100.0, 100.0]
    strike = 100.0
    r = 0.0
    refPrices = [19.450, 20.959, 22.995, 25.370, 30.593]
    sigmas = [0.05, 0.1, 0.15, 0.2, 0.3]
    sigmas = range(0.05, stop=1.0, step=0.05)
    refPrices = [19.445170149703884,
        20.953356975108733,
        22.988881461338213,
        25.36377806849989,
        27.925099795764126,
        30.586842476142913,
        33.29963515997311,
        36.03257500826625,
        38.764222499176924,
        41.47830929071817,
        44.16169414426057,
        46.8029977838724,
        49.39250376519999,
        51.921250539250465,
        54.381610887933896,
        56.76698232885734,
        59.07181543965413,
        61.29225106147898,
        63.42616635069005,
        65.4733758198407
    ]
    rho = 0.5
    tte = 5.0
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0]
    sigmaAlt = 1.0
    refPrices = [19.461554139374975, 20.962740909133707, 23.000629021915277, 25.375098901070963, 30.613805131033523, 36.04293194067231, 41.49640438947297, 46.81068084128687, 51.93742508839158, 56.7780111541194, 65.41084807287214]

    forward = spot .* exp((r - q) * tte)
    discountFactor = exp(-r * tte)
    sigmas = range(0.05,stop=1.0,step=0.05)
    refPrices = map(sigma -> AQFED.MonteCarlo.simulate(ScrambledSobolSeq(4, 2^29, FaureTezuka(ScramblingRngAdapter(Chacha8SIMD(UInt32)))),
     [AQFED.TermStructure.ConstantBlackModel(sigmaAlt, 0.0, 0.0), AQFED.TermStructure.ConstantBlackModel(sigma, 0.0, 0.0), AQFED.TermStructure.ConstantBlackModel(sigma, 0.0, 0.0), AQFED.TermStructure.ConstantBlackModel(sigma, 0.0, 0.0)],
            forward, makeCorrelationMatrix(rho), AQFED.MonteCarlo.VanillaBasketOption(true, strike, weight, AQFED.MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0), 1024 * 1024*16), sigmas)
    refPrices .*= discountFactor
    forward = spot .* exp((r - q) * tte)
    discountFactor = exp(-r * tte)
    pricers = Dict(
        "Ju" => JuBasketPricer(),
        "LB" => DeelstraLBBasketPricer(2, 3),
        "SG2" =>ShiftedGeometricExpansion(2,3),
        "SG1" => ShiftedGeometricExpansion(1,3),
        "SG3" => ShiftedGeometricExpansion(3,3),
        "VG0" => VorstGeometricExpansion(0),
        "VG2" => VorstGeometricExpansion(2),
        "Deelstra" => DeelstraBasketPricer(2, 3),
        "VG3" => VorstGeometricExpansion(3),
        "VG1" => VorstGeometricExpansion(1),
        "VL0" => AQFED.Basket.VorstLevyExpansion(0),
        "VL1" => AQFED.Basket.VorstLevyExpansion(1),
        "VL2" => AQFED.Basket.VorstLevyExpansion(2),
        "VL3" => AQFED.Basket.VorstLevyExpansion(3),
        "3MM" => AQFED.Basket.SLN3MMBasketPricer()
    )
    priceMap = Dict()
    for (pricerName, pricer) = pricers
        prices = map(sigma -> priceEuropean(pricer, true, strike, discountFactor, spot, forward, makeVarianceVector(sigma, tte, firstSigma=sigmaAlt), weight, makeCorrelationMatrix(rho)), sigmas)
        priceMap[pricerName] = prices
    end
    for (i, rho) = enumerate(sigmas)
        @printf("%1.2f & %2.3f & %2.3f & %2.3f &  %2.3f & %2.3f & %2.3f & %2.3f & %2.3f & %2.3f\\\\\n", rho, refPrices[i], priceMap["Ju"][i], priceMap["3MM"][i], priceMap["LB"][i], priceMap["Deelstra"][i], priceMap["VG1"][i], priceMap["VG2"][i], priceMap["VG3"][i], priceMap["VL3"][i])
    end
    sortedNames = ["Ju", "3MM", "LB", "Deelstra", "VG1", "VG2", "VG3", "VL2", "VL3"]
    for name = sortedNames
        @printf("& %s ", name)
    end
    @printf("\\\\\n")
    for name = sortedNames
        allPrices = priceMap[name]
        @printf("& %1.3f ", rmsd(allPrices, refPrices))
    end
    @printf("\\\\\n")
    for name = sortedNames
        allPrices = priceMap[name]
        @printf("& %1.3f ", maximum(abs.(allPrices - refPrices)))
    end
    @printf("\\\\\n")


    @test rmsd(priceMap["Deelstra"], refPrices) < 1e-1
    # prices = map(sigma -> priceEuropean(p, true, strike, discountFactor, spot, forward, makeVarianceVector(sigma, tte, firstSigma=sigmaAlt), weight, makeCorrelationMatrix(0.5)), sigmas)
    # for (sigma, price, refPrice) = zip(sigmas, prices, refPrices)
    #     @printf("%.2f Deelstra %.4f %.2e\n", sigma, price, price - refPrice)
    # end
    # @test isapprox(refPrices, prices, atol=1e-1)
end

@testset "DeelstraBeisser" begin
    spot = [42.55, 48.21, 34.30, 100.0, 66.19]
    sigma = [33.34, 31.13, 33.27, 35.12, 36.36] ./ 100
    q = [2.59, 2.63, 3.32, 0.69, 1.24] ./ 100
    w = [25, 20, 30, 10, 15] ./ 100
    correlation =
        [1.00 0.84 -0.07 0.45 0.43;
            0.84 1.00 0.08 0.62 0.57;
            -0.07 0.08 1.00 -0.54 -0.59;
            0.45 0.62 -0.54 1.00 0.86;
            0.43 0.57 -0.59 0.86 1.00]
    r = 0.06
    strikes = 25.0:1.0:100.0
    tte = 5.0
    refPrices = [26.807163190517862, 26.102335962431685, 25.404995695700364, 24.71591280923319, 24.035843107947915, 23.365481357324487, 22.705497015484035, 22.0565465121771, 21.419169663451665, 20.793810151247765, 20.18090404600158, 19.580840285433005, 18.993955760785784, 18.420510060644986, 17.860662144089368, 17.314506056018903, 16.78214360863553, 16.263641059429993, 15.75899966892851, 15.26812341279179, 14.790971965338008, 14.327420823691433, 13.877312490967213, 13.44046637462282, 13.016674601615376, 12.605718148018802, 12.207371648803168, 11.821427109216412, 11.447577583544264, 11.085526947217165, 10.735031320863111, 10.395774746438892, 10.06751788216867, 9.749941338689059, 9.442759951229084, 9.145687370702325, 8.858418042503878, 8.58063749451879, 8.312045821440892, 8.05239771548456, 7.8014034713763305, 7.558767134793637, 7.324260249487928, 7.097621912921985, 6.878604297804436, 6.666949762939299, 6.4623809054203765, 6.264661858679028, 6.073581076555277, 5.888922506213903, 5.710439997459931, 5.537911585683455, 5.3711292139736555, 5.209898959048127, 5.054041555073436, 4.903350727107419, 4.7576648847752505, 4.616817279192076, 4.480626492758557, 4.348935737678122, 4.221583256887396, 4.098407052817986, 3.9792698394329657, 3.8640307836192544, 3.7525384024477892, 3.6446556937746153, 3.540248827244975, 3.439217412374261, 3.341428145307426, 3.2467674354270644, 3.15514444097019, 3.0664533563304817, 2.9805846003493075, 2.8974469067956483, 2.816931357822913, 2.7389594262904247]
    #tte = 1.0

    t = reverse([tte, tte - 1.0 / 12, tte - 2.0 / 12, tte - 3.0 / 12, tte - 4.0 / 12])
    forward = zeros((length(spot), length(t)))
    variance = zeros((length(spot), length(t)))
    weight = zeros((length(spot), length(t)))
    atmStrike = 0.0
    for (i, si) = enumerate(spot)
        for (j, tj) = enumerate(t)
            forward[i, j] = si * exp((r - q[i]) * tj)
            variance[i, j] = sigma[i]^2 * tj
            weight[i, j] = w[i] / length(t)
            atmStrike += weight[i, j] * forward[i, j]
        end
    end
    discountFactor = exp(-r * tte)

    pricers = Dict(
        "Ju" => JuBasketPricer(),
        "LB" => DeelstraLBBasketPricer(3, 3),
        "VG2" => VorstGeometricExpansion(2),
        "Deelstra" => DeelstraBasketPricer(3, 3, AQFED.Math.GaussLegendre(33)),
        "VG3" => VorstGeometricExpansion(3),
        "VG1" => VorstGeometricExpansion(1),
        "VL3" => AQFED.Basket.VorstLevyExpansion(3),
        "VL2" => AQFED.Basket.VorstLevyExpansion(2),
        #"3MM" => AQFED.Basket.SLN3MMBasketPricer()
    )
    priceMap = Dict()
    for (pricerName, pricer) = pricers
        prices = map(strike -> AQFED.Asian.priceAsianBasketFixedStrike(
                pricer,
                true,
                strike,
                discountFactor,
                spot,
                forward, #forward for each asset F_i to each Asian observation t_j
                variance, #vol^2 * t_i
                weight,
                correlation, t #S_k S_j
            ), strikes)
        priceMap[pricerName] = prices
    end
    #= 
    refPrices= AQFED.Asian.priceAsianBasketFixedStrike(
        AQFED.Basket.MonteCarloEngine(true,1024*1024*16),
        true,
        strikes,
        discountFactor,
        spot,
        forward, #forward for each asset F_i to each Asian observation t_j
        variance, #vol^2 * t_i
        weight,
        correlation,t #S_k S_j
    )

    basketSpot = weight[:,1]' * spot*5
    50.498000000000005
    p = plot(xlab="Forward moneyness", ylab="Price error in b.p.")

     for pricerName = ["VL3","VG3","VG2","Deelstra","Ju","VL2"]
          plot!(p,strikes./atmStrike .- 1.0, (priceMap[pricerName] - refPrices)/basketSpot .* 10000, label=pricerName,linestyle=:auto)
           end

     plot(p)
    =#
end

@testset "Hagspihl" begin
    nAsset = 2
    rhoMin = -1.0
    rhoMax = 1.0
    rhoSteps = 200
    pSimp = AQFED.Basket.QuadBasketPricer(AQFED.Basket.Simpson(256 * 32))
    pLeg = AQFED.Basket.QuadBasketPricer(AQFED.Basket.GaussLegendre(64))
    # pSin = AQFED.Basket.QuadBasketPricer(AQFED.Basket.TanhSinh(127,1e-8))
    pCheb = AQFED.Basket.QuadBasketPricer(AQFED.Basket.Chebyshev{Float64,2}(64))
    pChebS = AQFED.Basket.QuadBasketPricer(AQFED.Basket.Chebyshev{Float64,2}(63))
    # pChebS = AQFED.Basket.QuadBasketPricer(AQFED.Basket.GaussLegendre(63))
    # pChebS = AQFED.Basket.QuadBasketPricer(AQFED.Basket.Simpson(31))

    pj = AQFED.Basket.JuBasketPricer()
    pd = AQFED.Basket.DeelstraBasketPricer(3, 3, AQFED.Basket.GaussLegendre(64))
    pd12 = AQFED.Basket.DeelstraBasketPricer(3, 1, AQFED.Basket.GaussLegendre(64))
    pds = AQFED.Basket.DeelstraLBBasketPricer(3, 3)
    for rho = range(rhoMin, stop=rhoMax, length=rhoSteps)
        r = 0.05
        q = 0.0
        σ = 0.2
        tte = 1.0
        strike = 110.0
        nWeights = 2
        discountFactor = exp(-r * tte)
        weights = zeros(Float64, nWeights)
        tvar = zeros(Float64, nWeights)
        forward = zeros(Float64, nWeights)
        spot = zeros(nWeights)
        for i = eachindex(weights)
            weights[i] = 1.0 / (nWeights)
            tvar[i] = σ^2 * tte
            spot[i] = 100.0
            forward[i] = spot[i] * exp((r - q) * tte)
        end

        correlation = [
            1.0 rho
            rho 1.0
        ]
        price = priceEuropean(pSimp, true, strike, discountFactor, spot, forward, tvar, weights, correlation, isSplit=true)
        priceLeg = priceEuropean(pLeg, true, strike, discountFactor, spot, forward, tvar, weights, correlation)
        priceCheb = priceEuropean(pCheb, true, strike, discountFactor, spot, forward, tvar, weights, correlation)
        priceChebS = priceEuropean(pChebS, true, strike, discountFactor, spot, forward, tvar, weights, correlation, isSplit=true)
        priceJu = priceEuropean(pj, true, strike, discountFactor, spot, forward, tvar, weights, correlation)
        priceD = priceEuropean(pd, true, strike, discountFactor, spot, forward, tvar, weights, correlation)
        priceD12 = priceEuropean(pd12, true, strike, discountFactor, spot, forward, tvar, weights, correlation)
        priceDS = priceEuropean(pds, true, strike, discountFactor, spot, forward, tvar, weights, correlation)
        @printf("%.0f %.2f %.1f %.2f %f %.2e %.2e %.2e %.2e %.2e %.2e %.2e\n", strike, r, σ, rho, price, priceLeg - price, priceCheb - price, priceChebS - price, priceJu - price, priceD - price, priceD12 - price, priceDS - price)
        # if math.Abs(priceRef-refValues[ir]) > 1e-8 {
        # 	t.Errorf("error too large at %d, expected %.8f was %.8f", ir, refValues[ir], priceRef)
        # }
    end

end

@testset "basketGreeks" begin
    #2 assets, w1=0.4, w2=0.6, vol1=0.1, vol2=0.9, s1=s2=1.0 (different s can be taken in w1,w2)
    #vary strike, correlation -0.9 -0.5 0 0.5 0.9.
    nAsset = 2
    σs = [0.1, 0.9]
    ws = [0.5, 0.5]
    tte = 1.0
    tvar = σs .^ 2 .* tte
    r = 0.0
    q = 0.0
    discountFactor = exp(-r * tte)
    spot = [1.0, 1.0]
    basketSpot = sum(ws .* spot)
    rhoMin = -0.99
    rhoMax = 0.99
    rhoSteps = 101
    pSimp = AQFED.Basket.QuadBasketPricer(AQFED.Basket.GaussLegendre(128))
    data = DataFrame(Rho=Float64[], Strike=Float64[], Delta1=Float64[], Delta2=Float64[], ScaledDelta1=Float64[], ScaledDelta2=Float64[], DeltaCashError=Float64[])
    shiftSize = 1e-4
    for rho = range(rhoMin, stop=rhoMax, length=rhoSteps)
        correlation = [
            1.0 rho
            rho 1.0
        ]
        strike = 1.0
        price = AQFED.Basket.priceEuropean(pSimp, true, strike, discountFactor, spot, spot .* exp((r - q) * tte), tvar, ws, correlation, isSplit=true)
        spotUp1 = spot .* [1.0 + shiftSize, 1.0]
        priceUp1 = AQFED.Basket.priceEuropean(pSimp, true, strike, discountFactor, spotUp1, spotUp1 .* exp((r - q) * tte), tvar, ws, correlation, isSplit=true)
        spotUp2 = spot .* [1.0, 1.0 + shiftSize]
        priceUp2 = AQFED.Basket.priceEuropean(pSimp, true, strike, discountFactor, spotUp2, spotUp2 .* exp((r - q) * tte), tvar, ws, correlation, isSplit=true)
        delta1 = (priceUp1 - price) / (spot[1] * shiftSize)
        delta2 = (priceUp2 - price) / (spot[2] * shiftSize)
        spotUp1 = spot .* [1.0 + shiftSize, 1.0 + shiftSize]
        priceUp1 = AQFED.Basket.priceEuropean(pSimp, true, strike, discountFactor, spotUp1, spotUp1 .* exp((r - q) * tte), tvar, ws, correlation, isSplit=true)
        delta = (priceUp1 - price) / (basketSpot * shiftSize)
        deltaCashBasket = delta * basketSpot
        deltaCashBasketI = delta1 * spot[1] + delta2 * spot[2]
        push!(data, [rho, strike, delta1, delta2, delta * ws[1], delta * ws[2], deltaCashBasket / deltaCashBasketI - 1])
    end

    data = DataFrame(Rho=Float64[], Strike=Float64[], Delta1=Float64[], Delta2=Float64[], ScaledDelta1=Float64[], ScaledDelta2=Float64[], DeltaCashError=Float64[])

    rho = -0.9
    correlation = [
        1.0 rho
        rho 1.0
    ]
    for strike = range(0.5, stop=2.0, length=101)
        price = AQFED.Basket.priceEuropean(pSimp, true, strike, discountFactor, spot, spot .* exp((r - q) * tte), tvar, ws, correlation, isSplit=true)
        spotUp1 = spot .* [1.0001, 1.0]
        priceUp1 = AQFED.Basket.priceEuropean(pSimp, true, strike, discountFactor, spotUp1, spotUp1 .* exp((r - q) * tte), tvar, ws, correlation, isSplit=true)
        spotUp2 = spot .* [1.0, 1.0001]
        priceUp2 = AQFED.Basket.priceEuropean(pSimp, true, strike, discountFactor, spotUp2, spotUp2 .* exp((r - q) * tte), tvar, ws, correlation, isSplit=true)
        delta1 = (priceUp1 - price) / (spot[1] * 1e-4)
        delta2 = (priceUp2 - price) / (spot[2] * 1e-4)
        spotUp1 = spot .* [1.0001, 1.0001]
        priceUp1 = AQFED.Basket.priceEuropean(pSimp, true, strike, discountFactor, spotUp1, spotUp1 .* exp((r - q) * tte), tvar, ws, correlation, isSplit=true)
        delta = (priceUp1 - price) / (basketSpot * 1e-4)
        deltaCashBasket = delta * basketSpot
        deltaCashBasketI = delta1 * spot[1] + delta2 * spot[2]
        push!(data, [rho, strike, delta1, delta2, delta * ws[1], delta * ws[2], deltaCashBasket / deltaCashBasketI - 1])
    end
    #Delta
    #= plot(data.Rho, data.Delta1, label="Δ1",xlab="Correlation",ylab="Δ value",legend=:outerright)
       plot!(data.Rho, data.ScaledDelta1, label="qΔA")
       plot!(data.Rho, data.Delta2, label="Δ2")
    plot!(size=(400,300),margins=3Plots.mm)


        plot(data.Strike, data.Delta1, label="Basket Delta1",xlab="Basket strike",ylab="Delta value",legend=:outerright)
       plot!(data.Strike, data.ScaledDelta1, label="Index Delta1")
       plot!(data.Strike, data.Delta2, label="Basket Delta2")
       plot!(data.Strike, data.ScaledDelta2, label="Index Delta2")

    =#
    #Vega

    #basket "vol"? best way to to imply vol using BS based on price and basket forward. Show that it is far off the simple wieghted sum. is it Gentle approx?
    #the basket delta cash = single shift delta cash. If 1 moves up and 2 moves down, single shift hedging will work, but not basket hedging.
    #move1=0.02;move2=-0.02;  plot(data.Rho, data.Delta1*move1 + data.Delta2*move2,xlab="Correlation",ylab="Δ hedge value",label="")
    #, label="Independent hedge")  # while basket hedge===0
    #basket hedge ok if correlation = 1, meaning move1 = move2.

end