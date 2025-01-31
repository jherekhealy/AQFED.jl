using AQFED, Test
using StatsBase
import AQFED.American:
    AndersenLakeRepresentation,
    AndersenLakeNRepresentation,
    priceAmerican,
    americanBoundaryPutQDP
import AQFED.TermStructure: ConstantBlackModel

@testset "ALNegative" begin
    spots = [80.0, 80.0, 80.0, 80.0, 10.0, 55.0, 75.0, 60.0]
    strike = 100.0
    r = -0.005
    q = -0.05
    vol = 0.2
    ttes = [1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0]
    qs = [-0.05, -0.01, -0.01, -0.05, -0.05, -0.01, -0.007, -0.01]
    refPrices = [20.030946627004035, 21.00207402916405, 27.322355750776925, 21.89483671167564, 90.13887335153295, 46.34568414232824, 31.278917798325534, 40.0]
    for (spot, tte, q, refPrice) = zip(spots, ttes, qs, refPrices)
        model = ConstantBlackModel(vol, r, q)
        pricer = AndersenLakeNRepresentation(model, tte, 1e-8, 5, 16, 21, 201)
        price = priceAmerican(pricer, strike, spot)
        println(spot, " ", tte, " ", q, " ", refPrice, " ", price, " ", price - refPrice)

        @test isapprox(refPrice, price, atol = 1e-4)
    end
end

@testset "ALQ0" begin
    strike = 100.0
    spot = 80.0
    vol = 0.3
    tte = 0.75
    q = 0.0
    r = 0.04
    model = ConstantBlackModel(vol, r, q)
    pricer = AndersenLakeRepresentation(model, tte, 1e-8, 10, 32, 101, 101)
    refPrice = priceAmerican(pricer, strike, spot)
    pricer = AndersenLakeRepresentation(model, tte, 1e-8, 7, 8, 15, 31)
    price = priceAmerican(pricer, strike, spot)
    println(price, " ", refPrice, " ", price - refPrice)
    @test isapprox(refPrice, price, atol = 1e-5)
    pricer = AndersenLakeRepresentation(model, tte, 1e-8, 16, 16, 31, 63)
    price = priceAmerican(pricer, strike, spot)
    println(price, " ", refPrice, " ", price - refPrice)
    @test isapprox(refPrice, price, atol = 1e-5)

    spot = 100.0
    vol = 0.1
    tte = 0.25
    q = 0.0
    r = 0.02
    model = ConstantBlackModel(vol, r, q)
    pricer = AndersenLakeRepresentation(model, tte, 1e-8, 10, 32, 101, 101)
    refPrice = priceAmerican(pricer, strike, spot)
    pricer = AndersenLakeRepresentation(model, tte, 1e-8, 7, 8, 15, 31)
    price = priceAmerican(pricer, strike, spot)
    println(price, " ", refPrice, " ", price - refPrice)
    @test isapprox(refPrice, price, atol = 1e-5)

end

@testset "ALInstability" begin
    spot = 100.0
    strike = 100.0
    vol = 0.1
    r = 0.1
    q = 0.001
    tte = 3.0
    isCall = false
    model = ConstantBlackModel(vol, r, q)
    pricer = AndersenLakeRepresentation(model, tte, 1e-8, 10, 32, 101, 101)
    price = priceAmerican(pricer, strike, spot)
    println(price)
    @test isapprox(1.7910185909001428, price, atol = 1e-12)
end



skipRInfQ(rInfq::Int, r::Float64, q::Float64)::Bool =
    (rInfq == 1 && r >= q) || (rInfq == -1 && r <= q)

function batchAL(threshold, rInfq, isBatch, m, n, l, p)
    rs = [0.02, 0.04, 0.06, 0.08, 0.10]
    qs = [0.0, 0.04, 0.08, 0.12]
    spots = [25.0, 50.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0, 175.0, 200.0]
    ttes = [1.0 / 12, 0.25, 0.5, 0.75, 1.0]
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    strike = 100.0

    refPrices = zeros((length(rs), length(qs), length(ttes), length(sigmas), length(spots)))
    @elapsed for (ir, r) in enumerate(rs), (iq, q) in enumerate(qs)
        if !skipRInfQ(rInfq, r, q)
            for (it, tte) in enumerate(ttes), (iv, vol) in enumerate(sigmas)
                model = ConstantBlackModel(vol, r, q)
                pricer = AndersenLakeRepresentation(model, tte, 1e-8, 16, 16, 31, 63)
                for (is, spot) in enumerate(spots)
                    refPrices[ir, iq, it, iv, is] = priceAmerican(pricer, strike, spot)
                end

            end
        end

    end
    prices = zeros(Float64,(length(rs), length(qs), length(ttes), length(sigmas), length(spots)))
    elap = @elapsed for (ir, r) in enumerate(rs), (iq, q) in enumerate(qs)
        if !skipRInfQ(rInfq, r, q)
            for (it, tte) in enumerate(ttes), (iv, vol) in enumerate(sigmas)
                model = ConstantBlackModel(vol, r, q)
                local pricer
                if isBatch
                    pricer = AndersenLakeRepresentation(model, tte, 1e-4, m, n, l, p)
                end
                for (is, spot) in enumerate(spots)
                    if refPrices[ir, iq, it, iv, is] > threshold
                        if !isBatch
                            pricer =
                                AndersenLakeRepresentation(model, tte, 1e-4, m, n, l, p)
                        end
                        prices[ir, iq, it, iv, is] = priceAmerican(pricer, strike, spot)
                        if abs(prices[ir, iq, it, iv, is] - refPrices[ir, iq, it, iv, is]) > 0.1
                            println(
                                ir,
                                " ",
                                iq,
                                " ",
                                it,
                                " ",
                                iv,
                                " ",
                                is,
                                " ",
                                refPrices[ir, iq, it, iv, is],
                                " ",
                                prices[ir, iq, it, iv, is],
                                " ",
                                spot,
                                " ",
                                vol,
                                " ",
                                tte,
                                " ",
                                q,
                                " ",
                                r,
                            )
                        end
                    end
                end

            end
        end

    end
    return prices, refPrices, elap
end
@testset "ALPositiveSet" begin
    threshold = 0.5
    rInfq = 0 #1, -1 or 0
    isBatch = false
    prices, refPrices, elap = batchAL(threshold, rInfq, isBatch, 7, 8, 15, 31);elap
    thrIndices = findall(z -> z > threshold, refPrices);
    mae = maxad(prices[thrIndices], refPrices[thrIndices])
    mre = maxad(prices[thrIndices] ./ refPrices[thrIndices], ones(length(thrIndices)))
    rmse = rmsd(prices[thrIndices], refPrices[thrIndices])
    println(
        "AL ",
        7,
        " ",
        rmse,
        " ",
        mae,
        " ",
        mre,
        " ",
        elap,
        " ",
        length(thrIndices) / elap,
    )

end




@testset "QDHalley" begin
    r = 0.02
    q = 0.04
    vol = 0.4
    T = 0.015
    K = 100.0
    Szero = 90.0
    #Szero = K * min(r / q, 1)
    T = 5.0
    model = ConstantBlackModel(vol, r, q)
    for i = 1:100
        tte = T * i / 100
        # americanBoundaryPutQDP(false, Szero, K, r, q, tte, vol, 1e-6, Halley)
        # americanBoundaryPutQDP(false, Szero, K, r, q, tte, vol, 1e-6, InverseQuadratic)
        Szero = americanBoundaryPutQDP(false, model, Szero, K, tte, 1e-6)
        # Szero = americanBoundaryPutQDP(false, Szero, K, r, q, tte, vol, 1e-6, SuperHalley)
        #        println("Szero = ", Szero)
        @test isless(Szero, 50.0)
    end
    @test isapprox(Szero, 22.665, atol = 1e-3)

end



function batchALN(threshold, rInfq, isBatch, m, n, l, p; method = "AL",factor=10,smoothing="Averaging",lambdaS=0.25,isShort=true, useSqrt=true)
    rs = [-0.0009, -0.0049, -0.0099, -0.0199, -0.0499]
    qs = [-0.001, -0.005, -0.01, -0.02, -0.05]
    spots = [50.0, 85.0, 95.0, 100.0, 105.0, 115.0, 150.0, 200.0]
    ttes = [7.0 / 365, 30.0 / 365, 91.0 / 365, 182 ./ 365, 273 ./ 365, 1.0]    
    if !isShort
         ttes = [2.0, 3.0, 5.0]
    end
    sigmas = [0.05, 0.1, 0.2, 0.5]
    # rs = [-0.005,-0.01,-0.02,-0.04]
    # sigmas = [0.1,0.2,0.3,0.4,0.5,0.6]
    strike = 100.0

    refPrices = zeros((length(rs), length(qs), length(ttes), length(sigmas), length(spots)))
    @elapsed for (ir, r) in enumerate(rs), (iq, q) in enumerate(qs)
        if !skipRInfQ(rInfq, r, q)
            for (it, tte) in enumerate(ttes), (iv, vol) in enumerate(sigmas)
                model = ConstantBlackModel(vol, r, q)
                pricer = AndersenLakeNRepresentation(model, tte, 1e-8, 21, 32, 41, 201)
                for (is, spot) in enumerate(spots)
                    refPrices[ir, iq, it, iv, is] = priceAmerican(pricer, strike, spot)
                end

            end
        end

    end
    prices = zeros((length(rs), length(qs), length(ttes), length(sigmas), length(spots)))
    elap = @elapsed for (ir, r) in enumerate(rs), (iq, q) in enumerate(qs)
        if !skipRInfQ(rInfq, r, q)
            for (it, tte) in enumerate(ttes), (iv, vol) in enumerate(sigmas)
                model = ConstantBlackModel(vol, r, q)
                if method == "RKL"
                    local pricer
                    if isBatch
                        pricer = AQFED.American.makeFDMPriceInterpolation(false, false, model, tte, strike,  m+1, factor * m + 1, useSqrt = useSqrt,smoothing=smoothing,lambdaS=lambdaS)
                    end
                    for (is, spot) in enumerate(spots)
                        if refPrices[ir, iq, it, iv, is] > threshold
                            if !isBatch
                                pricer = AQFED.American.makeFDMPriceInterpolation(false, false, model, tte, strike,  m+1, factor * m + 1, useSqrt = useSqrt,smoothing=smoothing,lambdaS=lambdaS)
                            end
                            prices[ir, iq, it, iv, is] = pricer(spot)
                        end
                    end
                elseif method == "TR-BDF2" || method=="TRBDF2"
                    payoff = VanillaAmerican(false, strike, tte)
                    local pricer
                    if isBatch
                        pricer = AQFED.FDM.priceTRBDF2(payoff, strike, model, nodividends, M=(factor*m+1), N=(m+1), grid=SmoothlyDeformedGrid(CubicGrid(lambdaS)), solverName="LUUL",useSqrt=useSqrt,varianceConditioner=AQFED.FDM.OneSidedConditioner())                
                    end
                    for (is, spot) in enumerate(spots)
                        if refPrices[ir, iq, it, iv, is] > threshold
                            if !isBatch
                                pricer = AQFED.FDM.priceTRBDF2(payoff, spot, model, nodividends, M=(factor*m+1), N=(m+1), grid=SmoothlyDeformedGrid(CubicGrid(lambdaS)), solverName="LUUL",useSqrt=useSqrt,varianceConditioner=AQFED.FDM.OneSidedConditioner())                
                           end
                            prices[ir, iq, it, iv, is] = pricer(spot)
                            if abs(prices[ir, iq, it, iv, is] - refPrices[ir, iq, it, iv, is]) > 0.1
                                println(
                                    ir,
                                    " ",
                                    iq,
                                    " ",
                                    it,
                                    " ",
                                    iv,
                                    " ",
                                    is,
                                    " ",
                                    refPrices[ir, iq, it, iv, is],
                                    " ",
                                    prices[ir, iq, it, iv, is],
                                    " ",
                                    spot,
                                    " ",
                                    vol,
                                    " ",
                                    tte,
                                    " ",
                                    q,
                                    " ",
                                    r,
                                )
                            end
                        end
                    end
                else
                    local pricer
                    if isBatch
                        pricer = AndersenLakeNRepresentation(model, tte, 1e-8, m, n, l, p)
                    end
                    for (is, spot) in enumerate(spots)
                        if refPrices[ir, iq, it, iv, is] > threshold # && abs(refPrices[ir, iq, it, iv, is] - strike + spot) > threshold
                            if !isBatch
                                pricer =
                                    AndersenLakeNRepresentation(model, tte, 1e-8, m, n, l, p)
                            end
                            prices[ir, iq, it, iv, is] = priceAmerican(pricer, strike, spot)
                            if abs(prices[ir, iq, it, iv, is] - refPrices[ir, iq, it, iv, is]) > 0.1
                                println(
                                    ir,
                                    " ",
                                    iq,
                                    " ",
                                    it,
                                    " ",
                                    iv,
                                    " ",
                                    is,
                                    " ",
                                    refPrices[ir, iq, it, iv, is],
                                    " ",
                                    prices[ir, iq, it, iv, is],
                                    " ",
                                    spot,
                                    " ",
                                    vol,
                                    " ",
                                    tte,
                                    " ",
                                    q,
                                    " ",
                                    r,
                                )
                            end
                        end
                    end
                end
            end
        end

    end
    return prices, refPrices, elap
end
@testset "ALNegativeSet" begin
    threshold = 1e-2
    rInfq = -1
    isBatch = true
    prices, refPrices, elap = batchALN(threshold, rInfq, isBatch, 7, 8, 15, 201);
    thrIndices = findall(z -> (z > threshold), refPrices);
    mae = maxad(prices[thrIndices], refPrices[thrIndices])
    mre = maxad(prices[thrIndices] ./ refPrices[thrIndices], ones(length(thrIndices)))
    rmse = rmsd(prices[thrIndices], refPrices[thrIndices])
    println(
        "AL ",
        7,
        " ",
        rmse,
        " ",
        mae,
        " ",
        mre,
        " ",
        elap,
        " ",
        length(thrIndices) / elap,
    )
    @test mae <= 0.000767
    @test mre <= 0.000214
    prices, refPrices, elap = batchALN(threshold, rInfq, isBatch, 5, 8, 15, 201, method="FDM");
    thrIndices = findall(z -> (z > threshold), refPrices);
    mae = maxad(prices[thrIndices], refPrices[thrIndices])
    mre = maxad(prices[thrIndices] ./ refPrices[thrIndices], ones(length(thrIndices)))
    rmse = rmsd(prices[thrIndices], refPrices[thrIndices])
    println(
        "AL ",
        7,
        " ",
        rmse,
        " ",
        mae,
        " ",
        mre,
        " ",
        elap,
        " ",
        length(thrIndices) / elap,
    )
    @test mae <= 0.000335
    @test mre < 0.000890
end
