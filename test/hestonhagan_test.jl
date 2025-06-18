using AQFED.VolatilityModels
using CharFuncPricing
using AQFED.Black
using AQFED.Random
using Test

@testset "VanDerZwaard_Table5.9" begin
    t = [0.0, 0.25, 0.75]
    alpha = [4.5, 6.0, 7.0]
    sigma = [0.07, 0.09, 0.1]
    rho = [-0.3, -0.25, -0.4]
    kappa = 2.5
    tte = 1.75
    strike = 100.0
    forward = 100.0
    hparams = HaganHestonPiecewiseParams(sigma, kappa, rho, alpha, t)
    AQFED.VolatilityModels.priceEuropean(HaganHestonTSApprox(hparams,tte), true, strike, forward, tte, 1.0)
    tsparams,ttets = makeHestonTSParams(hparams, tte=tte)
    CharFuncPricing.priceEuropean(CharFuncPricing.JoshiYangCharFuncPricer(CharFuncPricing.DefaultCharFunc(tsparams), ttets, n=512), true, strike, forward, ttets, 1.0)
    CharFuncPricing.priceEuropean(pricer,true, strike, forward, ttets, 1.0)
    AQFED.VolatilityModels.priceEuropean(AQFED.VolatilityModels.BGMApprox(tsparams), true, strike, forward, ttets, 1.0)
end

@testset "VanDerZwaard_Table5.10" begin
    t = [0.0, 0.2, 0.7]
    alpha = [15.0,12.0,18.0]
    sigma = [0.05,0.06,0.08]
    rho = [-0.05, 0.1, 0.1]
    kappa = 2.5
    tte = 1.3
    strike = 100.0
    forward = 100.0
    hparams = HaganHestonPiecewiseParams(sigma, kappa, rho, alpha, t)
     AQFED.VolatilityModels.priceEuropean(HaganHestonTSApprox(hparams,tte), true, strike, forward, tte, 1.0)
    tsparams,ttets = makeHestonTSParams(hparams, tte=tte)
    pricer = CharFuncPricing.JoshiYangCharFuncPricer(CharFuncPricing.DefaultCharFunc(tsparams), ttets, n=512)
    CharFuncPricing.priceEuropean(pricer,true, strike, forward, ttets, 1.0)
    AQFED.VolatilityModels.priceEuropean(AQFED.VolatilityModels.BGMApprox(tsparams), true, strike, forward, ttets, 1.0)
end

@testset "Rouah_Table9.6" begin
    spot = 129.14
    K = Float64.(124:136);
    rf = 0.0010;
    q  = 0.0068;
    t = [0.0, 37, 72, 135] ./ 365;
    kappa =  1.7850 .* ones(4)
    v0 =    0.0341
    theta = [    0.2900, 0.2579, 0.2371,  0.2099]
    sigma = [2.2764,1.4506, 1.2833, 1.1512 ]
    rho = [
        −0.1640, −0.6297, −0.0452,  −0.3383 ]
  
        tsparams = HestonTSParams(v0, kappa, theta, rho, sigma, t)

    tte = 226/365
    bgmPrices = map( strike -> AQFED.VolatilityModels.priceEuropean(AQFED.VolatilityModels.BGMApprox(tsparams), true, strike, spot*exp((rf-q)*tte), tte, 1.0), K)
    volB = map((strike, price) -> impliedVolatility(true, price, spot*exp((rf-q)*tte), strike, tte, 1.0), K, bgmPrices)
    pricer = CharFuncPricing.JoshiYangCharFuncPricer(DefaultCharFunc(tsparams), tte, n=512)
    joshiPrices = map(strike-> CharFuncPricing.priceEuropean(pricer, true, strike, spot*exp((rf-q)*tte), tte, 1.0), K)
    volJ = map((strike, price) -> impliedVolatility(true, price, spot*exp((rf-q)*tte), strike, tte, 1.0), K, joshiPrices)

    kappa=[5.8947, 5.3900, 4.1022,3.7900]
    v0 = 0.0341
    theta = [    0.0067, 0.1736, 0.1544, 0.1760]
    sigma = [ 1.0246, 3.7798, 5.9139, 5.8142]
    rho = [−0.2686, −0.6429, −0.4599, −0.5475]
 
end

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


    hagPrices = map(k->  AQFED.VolatilityModels.priceEuropean(AQFED.VolatilityModels.HaganHestonTSApprox(haganParams,tte), k >= 0, exp(k), 1.0, tte, 1.0), logmoneynessA)
    
    @test isapprox(vPrices, hagPrices, atol=1.1e-4)

    volHag = map((k, price) -> impliedVolatility(k >= 0, price, 1.0, exp(k), tte, 1.0), logmoneynessA, hagPrices)

    
end