using AQFED, Test
using AQFED.VolatilityModels
using AQFED.TermStructure
using AQFED.Math
using AQFED.Black
using AQFED.PDDE
using StatsBase
using Printf

@testset "Constant" begin
    σ=0.2
    tte = 1.0
    section = FlatSection(σ,tte)
    df = 1.0
    driftDf = 1.0
    spot=100.0
    repl = AQFED.VolatilityModels.ContinuousVarianceSwapReplication(GaussLegendre(16))
    priceLeg =AQFED.VolatilityModels. PriceVarianceSwap(repl,spot, tte, section, driftDf, df,ndev=6)
    repl = AQFED.VolatilityModels.ContinuousVarianceSwapReplication(Chebyshev{Float64,2}(16))
    priceCheb =AQFED.VolatilityModels. PriceVarianceSwap(repl,spot, tte, section, driftDf, df,ndev=6)
    println(priceLeg, " ", sqrt(priceLeg))
  println(priceCheb, " ", sqrt(priceCheb))
  @test isapprox(20.0,sqrt(priceLeg),atol=1e-6)
  @test isapprox(20.0,sqrt(priceCheb),atol=1e-6)
end

@testset "SVI" begin
    tte = 1.0
    df = 1.0
    driftDf = 1.0
  
    tte=1.0
    spot = 1.0
    forward=spot/driftDf
    strikes = [0.5,0.6,    0.7,    0.8,
    0.85,
    0.9,
    0.925,
    0.95,
    0.975,
    1.0,
    1.025,
    1.05,
    1.075,
    1.1,
    1.15,
    1.2,
    1.3,
    1.4,
    1.5]
    
    volatility=[	 39.8,
    34.9,
    30.8,
    27.4,
    25.9,
    24.5,
    23.8,
    23.1,
    22.3,
    21.5,
     20.7,
    19.8,
    19.0,
    18.2,
    16.6,
    15.4,
    14.3,
    14.7,
    15.6]
    logmoneynessA = log.(strikes./forward)
    weightsA = ones(length(strikes))
    volA = volatility ./ 100
    svi0, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logmoneynessA, volA, weightsA, aMin=0.0)
    ivkSVI0 = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi0, logmoneynessA))
    rmseSVI0 = StatsBase.rmsd(volA, ivkSVI0)
    svi, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logmoneynessA, volA, weightsA, aMin=-0.2)
    ivkSVI = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi, logmoneynessA))
    rmseSVI = StatsBase.rmsd(volA, ivkSVI)
    prices, w = AQFED.Collocation.weightedPrices(true, strikes, volA, weightsA, forward, 1.0, tte,vegaFloor=1e-8)
    lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, w, useVol=false, model=PDDE.Quadratic(),location="Equidistributed",size=10,L=strikes[1]/4,U=strikes[end]*4)
    ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0);        rmseq = StatsBase.L2dist(weightsA .* volA, weightsA .* ivkq)
    lvgqe = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, w, useVol=false, model=PDDE.Quadratic(),location="Mid-XX",size=0,L=strikes[1]/4,U=strikes[end]*4)
    ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgqe, true, strikes), forward, strikes, tte, 1.0);
    rmseq = StatsBase.L2dist(weightsA .* volA, weightsA .* ivkq)
    repl = AQFED.VolatilityModels.ContinuousVarianceSwapReplication(GaussLegendre(64))
    priceSVI0 =AQFED.VolatilityModels. PriceVarianceSwap(repl,spot, tte, svi0, driftDf, df,ndev=6)
    priceSVI =AQFED.VolatilityModels. PriceVarianceSwap(repl,spot, tte, svi, driftDf, df,ndev=6)
    println(priceSVI, " SVI ", sqrt(priceSVI))
  println(priceSVI0, " SVI0 ", sqrt(priceSVI0))
  struct LVGVarianceSection <: AQFED.TermStructure.VarianceSection
    lvgq
    forward
    tte
  end
  function AQFED.TermStructure.varianceByLogmoneyness(section::LVGVarianceSection, y)
    strike = forward*exp(y)
    isCall = y >= 0
    price = PDDE.priceEuropean(section.lvgq, isCall, strike)
    vol = Black.impliedVolatility(isCall, price, forward, strike, tte, 1.0)
    vol^2
  end
  priceLVG10 =AQFED.VolatilityModels. PriceVarianceSwap(repl,spot, tte, LVGVarianceSection(lvgq,forward,tte), driftDf, df,ndev=6)
  priceLVG =AQFED.VolatilityModels. PriceVarianceSwap(repl,spot, tte, LVGVarianceSection(lvgqe,forward,tte), driftDf, df,ndev=6)
  priceFuka = AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.FukasawaVarianceSwapReplication(false),tte, logmoneynessA, volA.^2, df)
  priceFukaL = AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.FukasawaVarianceSwapReplication(true),tte, logmoneynessA, volA.^2, df)

#=  plot(strikes,volatility,seriestype=:scatter, label="Market",xlab="Strike",ylab="Volatility in %",ms=4,markerstrokewidth=0,markeralpha=0.8)
  plot!(k,@.(100*Black.impliedVolatility(true, PDDE.priceEuropean(lvgqe, true, k), forward, k, tte, 1.0)),label="LVG",linestyle=:auto)
 plot!(k,@.(sqrt(AQFED.TermStructure.varianceByLogmoneyness(svi,log(k/forward)))*100),label="SVI",linestyle=:auto)
 plot!(k,@.(sqrt(AQFED.TermStructure.varianceByLogmoneyness(svi0,log(k/forward)))*100),label="SVI a=0",linestyle=:auto)
 plot!(size=(800,400),margins=3Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/varswap_bad_vol.pdf")
 =#
end