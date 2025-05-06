using AQFED.VolatilityModels
using CharFuncPricing
using AQFED.Black

@testset "BGMTable3.2" begin
    r = 0.0
    q = 0.0
    spot = 100.0
    θ = 0.06
    v0 = 0.04
    κ = 3.0
    σ = 0.3
    ρ = 0.0
    hParams = HestonParams(v0, κ, θ, ρ, σ)
    cf = DefaultCharFunc(hParams)
    ts = [0.25, 0.5, 1.0, 10.0]
    strikeMatrix = [70.0 100.0 130.0
        60.0 100.0 150.0
        50.0 100.0 180.0
        10.0 100.0 730.0
    ]
    for (tIndex, t) = enumerate(ts)
        pricer = CharFuncPricing.JoshiYangCharFuncPricer(cf, t)
        forward = spot * exp((r - q) * t)
        discountDf = exp(-r * t)
        strikes = strikeMatrix[tIndex, :]
        refPrices = map(k -> CharFuncPricing.priceEuropean(pricer, false, k, forward, t, discountDf), strikes)
        refVols = map((k, price) -> impliedVolatility(false, price, forward, k, t, discountDf), strikes, refPrices)
        bgmPrices = map(k -> AQFED.VolatilityModels.priceEuropean(AQFED.VolatilityModels.BGMApprox(hParams), false, k, forward, t, discountDf), strikes)
        bgmVols = map((k, price) -> impliedVolatility(false, price, forward, k, t, discountDf), strikes, bgmPrices)
        println(t, " ", refVols, refPrices)
        println(t, " ", bgmVols, bgmPrices)
    end
end


@testset "BGMTable3.3" begin
    r = 0.0
    q = 0.0
    spot = 100.0
    θ = 0.06
    v0 = 0.04
    κ = 3.0
    σ = 0.3
    ρ = -0.2
    hParams = HestonParams(v0, κ, θ, ρ, σ)
    cf = DefaultCharFunc(hParams)
    ts = [0.25, 0.5, 1.0, 10.0]
    strikeMatrix = [70.0 100.0 130.0
        60.0 100.0 150.0
        50.0 100.0 180.0
        10.0 100.0 730.0
    ]
    volMatrix = zeros(size(strikeMatrix))
    for (tIndex, t) = enumerate(ts)
        pricer = CharFuncPricing.JoshiYangCharFuncPricer(cf, t)
        forward = spot * exp((r - q) * t)
        discountDf = exp(-r * t)
        strikes = strikeMatrix[tIndex, :]
        refPrices = map(k -> CharFuncPricing.priceEuropean(pricer, false, k, forward, t, discountDf), strikes)
        refVols = map((k, price) -> impliedVolatility(false, price, forward, k, t, discountDf), strikes, refPrices)
        bgmPrices = map(k -> AQFED.VolatilityModels.priceEuropean(AQFED.VolatilityModels.BGMApprox(hParams), false, k, forward, t, discountDf), strikes)
        bgmVols = map((k, price) -> impliedVolatility(false, price, forward, k, t, discountDf), strikes, bgmPrices)
        println(t, " ", refVols, refPrices)
        println(t, " ", bgmVols, bgmPrices)
        volMatrix[tIndex, :] = bgmVols
    end
    @test isapprox(0.2376, volMatrix[end, end], atol=5e-5) #ref vol from paper 
end

@testset "HestonTSVarianceSwap" begin
    hParams = HestonTSParams{Float64,Float64}(0.040650888000630374, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.1989236398237733, 0.1674474922909069, 0.13945802391851483, 0.11566822371776626, 0.09813276805220718, 0.0889063332853836, 0.08140203607508543, 0.0761938592456171, 0.07291668182147483, 0.0706942301374475, 0.0722919078326234, 0.0766303349529027, 0.08302760255041278, 0.09080180167728102, 0.09943620634867764], [-0.6790576812248026, -0.6790529055179677, -0.6790484490852274, -0.6790442223061139, -0.6790403375380464, -0.6790369071384444, -0.6790318588743152, -0.6790299763730819, -0.681629770944323, -0.7017521439374732, -0.7380569211070961, -0.78748098355775, -0.8467602028941545, -0.9126304507210292, -0.9831693862883615], [1.540691466390082, 1.4305718389383753, 1.3274020400589377, 1.2287269533611478, 1.1367456046973166, 1.0536570199197557, 0.9229542454326898, 0.8542108367184412, 0.8243933371873641, 0.7754829818519577, 0.7407599111460993, 0.7165439846092997, 0.7002209845535634, 0.689176693290895, 0.680644611628769], [0.0, 0.2602739726027397, 0.5095890410958904, 0.7589041095890411, 1.0082191780821919, 1.2575342465753425, 1.7561643835616438, 2.254794520547945, 2.7534246575342465, 3.76986301369863, 4.767123287671233, 5.764383561643836, 6.761643835616439, 7.758904109589041, 8.775342465753425])

    tte = 1.0

    cf = DefaultCharFunc(hParams)
    pricer = CharFuncPricing.JoshiYangCharFuncPricer(cf, tte, n=512)
    pricer = makeCosCharFuncPricer(cf, tte, 2048, 16)
    logmoneynessA = range(-3.5 * sqrt(tte), 3.5 * sqrt(tte), length=101)
    vPrices = map(k -> CharFuncPricing.priceEuropean(pricer, k >= 0, exp(k), 1.0, tte, 1.0), logmoneynessA)
    volA = map((k, price) -> impliedVolatility(k >= 0, price, 1.0, exp(k), tte, 1.0), logmoneynessA, vPrices)
    priceFuka = AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.FukasawaVarianceSwapReplication(true), tte, logmoneynessA, volA .^ 2, 1.0)

    priceTheo = AQFED.VolatilityModels.priceVarianceSwap(hParams, tte) * 10000
    @test isapprox(priceFuka, priceTheo, atol=0.5)

end