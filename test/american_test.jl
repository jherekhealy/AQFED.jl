using AQFED, Test
using StatsBase
import AQFED.American:
    AndersenLakeRepresentation,
    AndersenLakeNRepresentation,
    priceAmerican,
    americanBoundaryPutQDP
import AQFED.TermStructure: ConstantBlackModel

@testset "ALNegative" begin
    spot = 80.0
    strike = 100.0
    r = -0.005
    q = -0.05
    tte = 1.0
    vol = 0.2

    model = ConstantBlackModel(vol, r, q)
    pricer = AndersenLakeNRepresentation(model, tte, 1e-8, 5, 16, 21, 201)
    price = priceAmerican(pricer, strike, spot)
    println(price)
    @test isapprox(20.030946627004035, price, atol = 1e-4)
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
    prices = zeros((length(rs), length(qs), length(ttes), length(sigmas), length(spots)))
    elap = @elapsed for (ir, r) in enumerate(rs), (iq, q) in enumerate(qs)
        if !skipRInfQ(rInfq, r, q)
            for (it, tte) in enumerate(ttes), (iv, vol) in enumerate(sigmas)
                model = ConstantBlackModel(vol, r, q)
                local pricer
                if isBatch
                    pricer = AndersenLakeRepresentation(model, tte, 1e-8, m, n, l, p)
                end
                for (is, spot) in enumerate(spots)
                    if refPrices[ir, iq, it, iv, is] > threshold
                        if !isBatch
                            pricer =
                                AndersenLakeRepresentation(model, tte, 1e-8, m, n, l, p)
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
    prices, refPrices, elap = batchAL(threshold, rInfq, isBatch, 7, 8, 15, 31)
    thrIndices = findall(z -> z > threshold, refPrices)
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
