using AQFED.VolatilityModels
using CharFuncPricing
using AQFED.Black
using AQFED.Random
using Test

@testset "HestonRegularHagan" begin
    hParams = HestonTSParams(0.04, [1.0],[0.04],[-0.7],[0.4], [0.0])
    params = HestonParams(hParams.v0, hParams.κ[1], hParams.θ[1], hParams.ρ[1], hParams.σ[1])
    tte = 2.0    
    pricer = CharFuncPricing.JoshiYangCharFuncPricer(DefaultCharFunc(params), tte, n=512)
    logmoneynessA = range(- sqrt(tte),  sqrt(tte), length=51)
    vPrices = map(k -> CharFuncPricing.priceEuropean(pricer, k >= 0, exp(k), 1.0, tte, 1.0), logmoneynessA)
    volA = map((k, price) -> impliedVolatility(k >= 0, price, 1.0, exp(k), tte, 1.0), logmoneynessA, vPrices)

    hPrices = map(k->  AQFED.VolatilityModels.priceEuropean(AQFED.VolatilityModels.HaganHestonTSApprox(hParams), k >= 0, exp(k), 1.0, tte, 1.0), logmoneynessA)
    volH = map((k, price) -> impliedVolatility(k >= 0, price, 1.0, exp(k), tte, 1.0), logmoneynessA, hPrices)

    bgmPrices =  map(k->  AQFED.VolatilityModels.priceEuropean(AQFED.VolatilityModels.BGMApprox(params), k >= 0, exp(k), 1.0, tte, 1.0), logmoneynessA)
    volBGM = map((k, price) -> impliedVolatility(k >= 0, price, 1.0, exp(k), tte, 1.0), logmoneynessA, bgmPrices)

    haganParams = AQFED.VolatilityModels.HaganHestonPiecewiseParams([0.3,0.25,0.2,0.2],1.0, [-0.75,-0.5,-0.75,-0.75],[0.5,0.5,0.5,0.5],[0.0,0.1,1.0,2.0])
    hParams = AQFED.VolatilityModels.makeHestonTSParams(haganParams)
    
    tte=hParams.startTime[3]
    pricer = CharFuncPricing.JoshiYangCharFuncPricer(DefaultCharFunc(hParams), tte, n=512)
    logmoneynessA = range(- sqrt(tte),  sqrt(tte), length=51)
    vPrices = map(k -> CharFuncPricing.priceEuropean(pricer, k >= 0, exp(k), 1.0, tte, 1.0), logmoneynessA)
    volA = map((k, price) -> impliedVolatility(k >= 0, price, 1.0, exp(k), tte, 1.0), logmoneynessA, vPrices)

    payoff = AQFED.MonteCarlo.VanillaOption(true, 1.0, AQFED.MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0)
    timesteps = 1000
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    ndim = AQFED.MonteCarlo.ndims(hParams, specTimes, 1.0 / timesteps)
    seq = ScrambledSobolSeq(ndim, 1024*32, Owen(30, ScramblingRngAdapter(AQFED.Random.Blabla8())))
    value, stderrorl = AQFED.MonteCarlo.simulateFullTruncation(
        seq,
        hParams,
        1.0,
        payoff,
        1,
        1024*32,
        1.0 / timesteps,
        withBB=false,
    )
    hParams2 = CharFuncPricing.HestonTSParams(1.0, ones(length(haganParams.startTime)).* haganParams.κ, ones(length(haganParams.startTime)), haganParams.ρ, haganParams.σ, haganParams.startTime)
    tte = hParams2.startTime[3]
    payoff = AQFED.MonteCarlo.VanillaOption(true, 1.0, AQFED.MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0)
   
    timesteps = 200
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    ndim = AQFED.MonteCarlo.ndims(hParams2, specTimes, 1.0 / timesteps)

    
 
    valueHMC = map(k -> AQFED.MonteCarlo.simulateFullTruncation(
        ScrambledSobolSeq(ndim, 1024*128, Owen(30, ScramblingRngAdapter(AQFED.Random.Blabla8()))),
        hParams2,
        1.0,
        AQFED.MonteCarlo.VanillaOption(k>=0, exp(k), AQFED.MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0),
        1,
        1024*128,
        1.0 / timesteps,
        withBB=true, leverageFunction = AQFED.TermStructure.PiecewiseConstantFunction(haganParams.startTime, haganParams.leverage))[1], logmoneynessA)
    @test isapprox(vPrices, valueHMC, atol=1e-4)


    hagPrices = map(k->  AQFED.VolatilityModels.priceEuropean(AQFED.VolatilityModels.HaganHestonTSApprox(haganParams), k >= 0, exp(k), 1.0, tte, 1.0), logmoneynessA)
    
    @test isapprox(vPrices, hagPrices, atol=1.1e-4)

    volHag = map((k, price) -> impliedVolatility(k >= 0, price, 1.0, exp(k), tte, 1.0), logmoneynessA, hagPrices)

    
end