using AQFED, Test
using AQFED.Basket
using Printf,DataFrames

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
    sigma = 0.4
    rhos = [0.1, 0.3, 0.5, 0.7, 0.8, 0.95]
    tte = 5.0
    p = DeelstraBasketPricer(3, 3)
    pl = DeelstraLBBasketPricer(3, 3)
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
        price = priceEuropean(pl, true, strike, discountFactor, spot, forward, tvar, weight, correlation)
        @printf("%.2f DeelstraLB %.4f %.2e\n", rho, price, price - refPrices[i])
        price = priceEuropean(p, true, strike, discountFactor, spot, forward, tvar, weight, correlation)
        @printf("%.2f Deelstra %.4f %.2e\n", rho, price, price - refPrices[i])
        @test isapprox(refPrices[i], price, atol=1e-1)
    end

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
    spot = 80.0
    strike = 100.0
    r = -0.005
    q = -0.05
    tte = 1.0
    vol = 0.2

    refPrices = [54.3098, 47.4807, 41.5221, 36.3514, 31.8764, 28.0070, 24.6601, 21.7622, 19.2489, 17.0651, 15.1636]
    weight = [0.25, 0.25, 0.25, 0.25]
    spot = [100.0, 100.0, 100.0, 100.0]
    strikes = collect(50.0:10.0:150.0)
    r = 0.0
    sigma = 0.4
    rho = 0.5
    tte = 5.0
    p = DeelstraBasketPricer(3, 3)
    p = DeelstraBasketPricer(1, 3, q=GaussKronrod())
    pl = DeelstraLBBasketPricer(3, 3)
    for (i, strike) in enumerate(strikes)
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
        price = priceEuropean(pl, true, strike, discountFactor, spot, forward, tvar, weight, correlation)
        @printf("%.2f DeelstraLB %.4f %.2e\n", strike, price, price - refPrices[i])
        price = priceEuropean(p, true, strike, discountFactor, spot, forward, tvar, weight, correlation)
        @printf("%.2f Deelstra %.4f %.2e\n", strike, price, price - refPrices[i])
        @test isapprox(refPrices[i], price, atol=2e-2)
    end
end
@testset "KornTable5" begin
    spot = 80.0
    strike = 100.0
    r = -0.005
    q = -0.05
    tte = 1.0
    vol = 0.2

    refPrices = [19.450, 20.959, 22.995, 25.370, 30.593]
    weight = [0.25, 0.25, 0.25, 0.25]
    spot = [100.0, 100.0, 100.0, 100.0]
    strike = 100.0
    r = 0.0
    sigmas = [0.05, 0.1, 0.15, 0.2, 0.3]
    rho = 0.5
    tte = 5.0
    p = DeelstraBasketPricer(3, 3)
    for (i, sigma) in enumerate(sigmas)
        correlation = [
            1.0 rho rho rho
            rho 1.0 rho rho
            rho rho 1.0 rho
            rho rho rho 1.0
        ]
        tvar = [1.0^2, sigma^2, sigma^2, sigma^2] .* tte
        q = 0.0
        forward = spot .* exp((r - q) * tte)
        discountFactor = exp(-r * tte)
        price = priceEuropean(p, true, strike, discountFactor, spot, forward, tvar, weight, correlation)
        @printf("%.2f Deelstra %.4f %.2e\n", sigma, price, price - refPrices[i])
        @test isapprox(refPrices[i], price, atol=1e-1)
    end
end


@testset "JuAsianTable4" begin
    #Table 4 of Ju "Pricing Asian and Basket Options via Taylor Expansions"
    spot = 100.0
    strikes = [95.0, 100.0, 105.0]
    r = 0.09
    q = 0.0
    tte = 3.0
    vol = 0.3
    nWeights = Int(tte * 52)
    weights = zeros(Float64, nWeights + 1)
    tvar = zeros(Float64, nWeights + 1)
    forward = zeros(Float64, nWeights + 1)
    for i = 1:length(weights)
        weights[i] = 1.0 / (nWeights + 1)
        ti = (i - 1) / (nWeights) * tte
        tvar[i] = vol^2 * ti
        forward[i] = spot * exp((r - q) * ti)
    end
    discountFactor = exp(-r * tte)
    refPrices = [19.0144, 16.5766, 14.3830] #TS values
    p = DeelstraBasketPricer(1, 3)
    pl = DeelstraLBBasketPricer(1, 3)  #1,3 best results = GC of Ju paper.
    for (refPrice, strike) in zip(refPrices, strikes)
        price = priceAsianFixedStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra    %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=1e-4)
        price = priceAsianFixedStrike(pl, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra-LB %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=1e-2)
    end

    vol = 0.5
    for i = 1:length(weights)
        ti = (i - 1) / (nWeights) * tte
        tvar[i] = vol^2 * ti
    end
    refPrices = [24.5526, 22.6115, 20.8241] #TS values
    for (refPrice, strike) in zip(refPrices, strikes)
        price = priceAsianFixedStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra    %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=2e-4)
        price = priceAsianFixedStrike(pl, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra-LB %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=3e-2)
    end
    #Deelstra is much more accurate than TE6, as expected with vols relatively high
end

@testset "LordAsianTable4" begin
    #Table 4 of Lord "Partially exact and bounded approximations for arithmetic Asian options"
    spot = 100.0
    r = 0.05
    q = 0.0
    tte = 5.0
    vol = 0.5
    moneyness = [0.5, 1.0, 1.5]
    strikes = [58.2370, 116.4741, 174.7111]
    nWeights = Int(tte)
    weights = zeros(Float64, nWeights)
    tvar = zeros(Float64, nWeights)
    forward = zeros(Float64, nWeights)
    for i = eachindex(weights)
        weights[i] = 1.0 / (nWeights)
        ti = (i) / (nWeights) * tte
        tvar[i] = vol^2 * ti
        forward[i] = spot * exp((r - q) * ti)
    end
    discountFactor = exp(-r * tte)
    refPrices = [49.3944, 26.5780, 15.5342]
    p = DeelstraBasketPricer(2, 3)
    pl = DeelstraLBBasketPricer(2, 3)  #2,3 matches paper
    for (refPrice, strike) in zip(refPrices, strikes)
        price = priceAsianFixedStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra    %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=5e-3)
        price = priceAsianFixedStrike(pl, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra-LB %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=1.5e-1)
    end

    tte = 30.0
    vol = 0.25
    moneyness = [0.5, 1.0, 1.5]
    strikes = [118.9819, 237.9638, 356.9457]
    nWeights = Int(tte)
    weights = zeros(Float64, nWeights)
    tvar = zeros(Float64, nWeights)
    forward = zeros(Float64, nWeights)
    for i = eachindex(weights)
        weights[i] = 1.0 / (nWeights)
        ti = (i) / (nWeights) * tte
        tvar[i] = vol^2 * ti
        forward[i] = spot * exp((r - q) * ti)
    end
    discountFactor = exp(-r * tte)
    refPrices = [30.5153, 19.1249, 13.1168]
    p = DeelstraBasketPricer(3, 3, AQFED.Basket.Chebyshev{Float64,2}(16))
    pl = DeelstraLBBasketPricer(3, 3)  #2,3 matches paper and 3,3 is better
    for (refPrice, strike) in zip(refPrices, strikes)
        price = priceAsianFixedStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra    %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=5e-3)
        price = priceAsianFixedStrike(pl, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra-LB %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=3e-1)
    end


end



@testset "LordAsianRateLikeTable4" begin
    #Same test case but with asian rate (inspired by paper but not in paper). Table 4 of Lord "Partially exact and bounded approximations for arithmetic Asian options"
    spot = 100.0
    r = 0.05
    q = 0.0
    tte = 5.0
    vol = 0.5
    moneyness = [0.5, 1.0, 1.5]
    strikes = [58.2370, 116.4741, 174.7111] ./ 100
    nWeights = Int(tte)
    weights = zeros(Float64, nWeights)
    tvar = zeros(Float64, nWeights)
    forward = zeros(Float64, nWeights)
    for i = eachindex(weights)
        weights[i] = 1.0 / (nWeights)
        ti = (i) / (nWeights) * tte
        tvar[i] = vol^2 * ti
        forward[i] = spot * exp((r - q) * ti)
    end
    discountFactor = exp(-r * tte)
    refPrices = [49.9222, 18.0721, 4.8351] # from a PDE accurate to 2e-4.
    p = DeelstraBasketPricer(1, 3)  # 1,3 is best
    pl = DeelstraLBBasketPricer(1, 3)  #2,3 matches paper 1,3 is best
    for (refPrice, strike) in zip(refPrices, strikes)
        price = priceAsianFloatingStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra    %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=5e-3)
        price = priceAsianFloatingStrike(pl, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra-LB %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=1.5e-1)
    end

    r = 0.1
    refPrices = [53.584722, 21.255485, 6.236079]
    for i = eachindex(weights)
        weights[i] = 1.0 / (nWeights)
        ti = (i) / (nWeights) * tte
        tvar[i] = vol^2 * ti
        forward[i] = spot * exp((r - q) * ti)
    end
    discountFactor = exp(-r * tte)
    p = DeelstraBasketPricer(1, 3)
    pl = DeelstraLBBasketPricer(1, 3)
    for (refPrice, strike) in zip(refPrices, strikes)
        price = priceAsianFloatingStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra    %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=5e-3)
        price = priceAsianFloatingStrike(pl, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra-LB %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=1.5e-1)
    end

    r = 0.0
    q = 0.05
    refPrices = [32.604220, 9.611437, 2.114077]
    for i = eachindex(weights)
        weights[i] = 1.0 / (nWeights)
        ti = (i) / (nWeights) * tte
        tvar[i] = vol^2 * ti
        forward[i] = spot * exp((r - q) * ti)
    end
    discountFactor = exp(-r * tte)
    p = DeelstraBasketPricer(1, 3)  #3,3 is worse  1,3 is best
    pl = DeelstraLBBasketPricer(1, 3)
    for (refPrice, strike) in zip(refPrices, strikes)
        price = priceAsianFloatingStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra    %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=5e-3)
        price = priceAsianFloatingStrike(pl, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra-LB %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol=1.5e-1)
    end
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
    ws = [0.9, 0.1]
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
    data = DataFrame(Rho=Float64[], Strike=Float64[], Delta1=Float64[], Delta2=Float64[], ScaledDelta1=Float64[],ScaledDelta2=Float64[],DeltaCashError=Float64[])
    shiftSize = 1e-4
    for rho = range(rhoMin, stop=rhoMax, length=rhoSteps)
        correlation = [
            1.0 rho
            rho 1.0
        ]
        strike = 1.0
            price = AQFED.Basket.priceEuropean(pSimp, true, strike, discountFactor, spot, spot .* exp((r - q) * tte), tvar, ws, correlation, isSplit=true)
            spotUp1 = spot .* [1.0+shiftSize, 1.0]
            priceUp1 = AQFED.Basket.priceEuropean(pSimp, true, strike, discountFactor, spotUp1, spotUp1 .* exp((r - q) * tte), tvar, ws, correlation, isSplit=true)
            spotUp2 = spot .* [1.0, 1.0+shiftSize]
            priceUp2 = AQFED.Basket.priceEuropean(pSimp, true, strike, discountFactor, spotUp2, spotUp2 .* exp((r - q) * tte), tvar, ws, correlation, isSplit=true)
            delta1 = (priceUp1 - price) / (spot[1] * shiftSize)
            delta2 = (priceUp2 - price) / (spot[2] * shiftSize)
            spotUp1 = spot .* [1.0+shiftSize, 1.0+shiftSize]
            priceUp1 = AQFED.Basket.priceEuropean(pSimp, true, strike, discountFactor, spotUp1, spotUp1 .* exp((r - q) * tte), tvar, ws, correlation, isSplit=true)
            delta = (priceUp1 - price) / (basketSpot * shiftSize)
            deltaCashBasket = delta*basketSpot
            deltaCashBasketI = delta1*spot[1] + delta2*spot[2]
            push!(data,[rho, strike, delta1, delta2, delta*ws[1],delta*ws[2], deltaCashBasket/deltaCashBasketI-1])
        end
    
    data = DataFrame(Rho=Float64[], Strike=Float64[], Delta1=Float64[], Delta2=Float64[], ScaledDelta1=Float64[],ScaledDelta2=Float64[],DeltaCashError=Float64[])

   rho = -0.9
    correlation = [
        1.0 rho
        rho 1.0
    ]
    for strike = range(0.5,stop=2.0,length=101)
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
        deltaCashBasket = delta*basketSpot
        deltaCashBasketI = delta1*spot[1] + delta2*spot[2]       
        push!(data,[rho, strike, delta1, delta2, delta*ws[1],delta*ws[2], deltaCashBasket/deltaCashBasketI-1])
    end
    #Delta
    #= plot(data.Rho, data.Delta1, label="Δ1",xlab="Correlation",ylab="Δ value",legend=:outerright)
       plot!(data.Rho, data.ScaledDelta1, label="qΔA")
       plot!(data.Rho, data.Delta2, label="Δ2")
plot!(size=(800,300),margins=4Plots.mm)


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