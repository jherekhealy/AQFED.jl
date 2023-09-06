using AQFED, Test
using AQFED.Basket
using Printf

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
        @test isapprox(refPrices[i], price, atol = 1e-1)
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
        p = DeelstraBasketPricer(3, 3,AQFED.Basket.Simpson(4096))
        asympPrice =  priceEuropean(p, true, strike, discountFactor, spot, forward, tvar, weight, correlation,isLognormal=1)
         for n=16:16 #4:1:64  
           p = DeelstraBasketPricer(3, 3,AQFED.Basket.Chebyshev{Float64,2}(n,false))
        #    p = DeelstraBasketPricer(3, 3,AQFED.Basket.FourierBoyd(n,1.0))
           
            price = priceEuropean(p, true, strike, discountFactor, spot, forward, tvar, weight, correlation,isLognormal=1)
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

    refPrices = [54.3098,47.4807,41.5221,36.3514,31.8764,28.0070,24.6601,21.7622,19.2489,17.0651,15.1636]
    weight = [0.25, 0.25, 0.25, 0.25]
    spot = [100.0, 100.0, 100.0, 100.0]
    strikes = collect(50.0:10.0:150.0)
    r = 0.0
    sigma = 0.4
    rho = 0.5
    tte = 5.0
    p = DeelstraBasketPricer(3, 3)
    p = DeelstraBasketPricer(1, 3,q=GaussKronrod())
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
        @test isapprox(refPrices[i], price, atol = 2e-2)
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
        @test isapprox(refPrices[i], price, atol = 1e-1)
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
        ti = (i-1) / (nWeights) * tte
        tvar[i] = vol^2 * ti
        forward[i] = spot * exp((r - q) * ti)
    end
    discountFactor = exp(-r * tte)
    refPrices = [19.0144, 16.5766, 14.3830] #TS values
    p = DeelstraBasketPricer(1, 3)
    pl = DeelstraLBBasketPricer(1, 3)  #1,3 best results = GC of Ju paper.
    for (refPrice,strike) in zip(refPrices,strikes)
        price = priceAsianFixedStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra    %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol = 1e-4)
        price = priceAsianFixedStrike(pl, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra-LB %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol = 1e-2)
    end

    vol = 0.5
    for i = 1:length(weights)
        ti = (i-1) / (nWeights) * tte
        tvar[i] = vol^2 * ti
    end
    refPrices = [24.5526, 22.6115, 20.8241] #TS values
    for (refPrice,strike) in zip(refPrices,strikes)
        price = priceAsianFixedStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra    %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol = 2e-4)
        price = priceAsianFixedStrike(pl, true, strike, discountFactor, spot, forward, tvar, weights)
        @printf("%6.2f Deelstra-LB %.4f %.2e\n", strike, price, price - refPrice)
        @test isapprox(refPrice, price, atol = 3e-2)
    end
    #Deelstra is much more accurate than TE6, as expected with vols relatively high
end
