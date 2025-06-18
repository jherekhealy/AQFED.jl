using Test, AQFED, AQFED.VolatilityModels, LinearAlgebra, StatsBase
using CharFuncPricing

@testset "HestonTS-SPX500-Oct2010" begin
    strikeA = [1008, 1209.6, 1411.2, 1612.8, 1814.4, 1915.2, 2016, 2116.8, 2217.6, 2419.2, 2620.8, 3024]
    ttes = [0.057534247,
        0.153424658,
        0.230136986,
        0.479452055,
        0.728767123,
        1.22739726,
        1.726027397,
        2.243835616,
        2.742465753,
        3.24109589,
        4.238356164
    ]
    vols = [0.469330877 0.414404424 0.367296885 0.327108549 0.294442259 0.281546662 0.271399379 0.264274619 0.260233568 0.260279105 0.267799208 0.291990984
        0.438031838 0.387414998 0.344186959 0.307577311 0.27820473 0.266796144 0.257960029 0.251900797 0.248624262 0.249290252 0.256555792 0.279222439
        0.417585513 0.372442867 0.334123526 0.301871891 0.275993204 0.265780032 0.257609374 0.251598649 0.247755671 0.245926111 0.249948783 0.266604403
        0.378863656 0.343692938 0.314288789 0.289927723 0.270499867 0.262699832 0.256217647 0.251063253 0.247212335 0.243118381 0.243005143 0.250232597
        0.361240173 0.331461693 0.306727965 0.286289798 0.269839268 0.263077346 0.257281885 0.252439294 0.248523708 0.243291194 0.241073648 0.243068308
        0.348498059 0.323073547 0.302069305 0.284735901 0.270667776 0.264780547 0.259625421 0.2551777 0.251409317 0.245772883 0.242388102 0.240807094
        0.336927448 0.314798449 0.296604756 0.281620193 0.269409546 0.264250663 0.259682153 0.255675656 0.252202671 0.246737402 0.243031476 0.239805477
        0.329408098 0.309509953 0.293207982 0.279800062 0.26884608 0.264190728 0.260040257 0.256365655 0.253139096 0.247920833 0.244167054 0.240189782
        0.323014691 0.304706378 0.289747629 0.27745988 0.267408265 0.263122021 0.259285776 0.255870853 0.252850352 0.247890508 0.244208985 0.239941502
        0.319115 0.301957248 0.287962586 0.276471393 0.267055619 0.263027894 0.259411063 0.256177099 0.25330008 0.248520602 0.24489072 0.240422639
        0.313394981 0.297812921 0.285123732 0.274699544 0.266128184 0.262442454 0.259115821 0.256121744 0.253436081 0.248902532 0.245354116 0.240656834]
    r = 0.02
    q = 0.01
    spot = 2016.0
    forwards = [spot * exp((r - q) * tte) for tte = ttes]

      strikes = strikeA' .* ones(length(ttes))
 
      #original calibration
      paramsf2, rmse = AQFED.VolatilityModels.calibrateHestonFromVolsDirect(ttes, forwards, strikes, vols,method="Andersen-Lake",deviations=4.0)
#better one
      paramsf2, rmse = AQFED.VolatilityModels.calibrateHestonFromVols(ttes, forwards, strikes, vols)

    prices, isCall, weights = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ttes, forwards, strikes, vols, vegaFloor=1e-2, truncationDev=3)


    hagParamsB, rmseB = AQFED.VolatilityModels.calibrateHestonHaganFromPricesParam(ttes, forwards, strikes, prices, isCall, weights, lower=[1e-2,-0.99,0.05],upper=[2.0,0.5,20.0], numberOfKnots=[length(ttes),3,3], κ= 4.0, minimizer="DE", method="BGM")

    hagParams, rmse = AQFED.VolatilityModels.calibrateHestonHaganFromPricesParam(ttes, forwards, strikes, prices, isCall, weights, lower=[1e-2,-0.99,0.05],upper=[2.0,0.5,20.0], numberOfKnots=[length(ttes),3,3], κ= 4.0, minimizer="DE", method="Cos")

#we are interested in plotting vol error, increase vegaFloor.
     prices, isCall, weightsM = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ttes, forwards, strikes, vols, vegaFloor=1e-3,truncationDev=0.675)
     prices, isCall, weightsM = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ttes, forwards, strikes, vols, vegaFloor=1e-2,truncationDev=2.0)
     prices, isCall, weightsM = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ttes, forwards, strikes, vols, vegaFloor=1e-2,truncationDev=4.0)
     
     κ = 5.0; minimizerBootstrap = "BFGS";
     @time hagParamsBootstrap, rmse = AQFED.VolatilityModels.calibrateHestonHaganFromPrices(ttes, forwards, strikes, prices, isCall, weightsM, lower=[1e-2,-0.99,0.05],upper=[2.0,0.5,20.0], κ= κ, minimizer=minimizerBootstrap, method="Cos",isGlobal=false,swapCurve=AQFED.VolatilityModels.makeSwapCurveReplication(ttes, forwards, strikes, vols))
    volErrorBootstrap, rmseVolB = AQFED.VolatilityModels.estimateVolError(hagParamsBootstrap, ttes, forwards, 
    strikes, vols)
    norm((weightsM .> 0) .* volErrorBootstrap) / norm(weightsM .> 0)

    @time hagParamsGlobal, rmse = AQFED.VolatilityModels.calibrateHestonHaganFromPricesParam(ttes, forwards, strikes, prices, isCall, weightsM, lower=[1e-2,-0.99,0.05],upper=[2.0,0.5,20.0], numberOfKnots=[length(ttes),length(ttes),length(ttes)], κ= κ, minimizer="User", method="Cos",swapCurve=AQFED.VolatilityModels.makeSwapCurveReplication(ttes, forwards, strikes, vols))
volError, rmseVol = AQFED.VolatilityModels.estimateVolError(hagParamsGlobal, ttes, forwards,  strikes, vols)
           norm((weightsM .> 0) .* volError) / norm(weightsM .> 0)

           #filtered moneyness
    volsF, strikesF = AQFED.VolatilityModels.filterVolsBySimpleDelta(ttes, forwards, strikes, vols, deltas = [0.25,0.5,0.75])
    pricesF, isCallF, weightsMF = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ttes, forwards, strikesF, volsF, vegaFloor=1e-4,truncationDev=10.0)

    @time hagParamsBootstrapF, rmse = AQFED.VolatilityModels.calibrateHestonHaganFromPrices(ttes, forwards, strikesF, pricesF, isCallF, weightsMF, lower=[1e-2,-0.99,0.05],upper=[2.0,0.5,20.0], κ= κ, minimizer=minimizerBootstrap, method="Cos",isGlobal=false, swapCurve=AQFED.VolatilityModels.makeSwapCurveReplication(ttes, forwards, strikesF, volsF))
    volErrorBootstrapF, rmseVolB = AQFED.VolatilityModels.estimateVolError(hagParamsBootstrapF, ttes, forwards, 
    strikesF, volsF)
    norm((weightsMF .> 0) .* volErrorBootstrapF) / norm(weightsMF .> 0)

    @time hagParamsGlobalF, rmse = AQFED.VolatilityModels.calibrateHestonHaganFromPricesParam(ttes, forwards, strikesF, pricesF, isCallF, weightsMF, lower=[1e-2,-0.99,0.05],upper=[2.0,0.5,20.0], numberOfKnots=[length(ttes),length(ttes),length(ttes)], κ=κ, minimizer="User", method="Cos",swapCurve=AQFED.VolatilityModels.makeSwapCurveReplication(ttes, forwards, strikesF, volsF))
volErrorF, rmseVol = AQFED.VolatilityModels.estimateVolError(hagParamsGlobalF, ttes, forwards,  strikesF, volsF)
           norm((weightsMF .> 0) .* volErrorF) / norm(weightsMF .> 0)

           @time hagParamsBootstrapFP, rmse = AQFED.VolatilityModels.calibrateHestonHaganFromPrices(ttes, forwards, strikesF, pricesF, isCallF, weightsMF, lower=[1e-2,-0.99,0.05],upper=[2.0,0.5,20.0], κ= κ, minimizer=minimizerBootstrap, method="Cos",isGlobal=false,penalty=1.5e-3,isPenaltyGradient=false, swapCurve=AQFED.VolatilityModels.makeSwapCurveReplication(ttes, forwards, strikesF, volsF))
           volErrorBootstrapFP, rmseVolB = AQFED.VolatilityModels.estimateVolError(hagParamsBootstrapFP, ttes, forwards, 
           strikesF, volsF)
           norm((weightsMF .> 0) .* volErrorBootstrapFP) / norm(weightsMF .> 0)

           @time hagParamsGlobalFS, rmse = AQFED.VolatilityModels.calibrateHestonHaganFromPricesParam(ttes, forwards, strikesF, pricesF, isCallF, weightsMF, lower=[1e-2,-0.99,0.05],upper=[2.0,0.5,20.0], numberOfKnots=[length(ttes),5,5], κ=κ, minimizer="User", method="Cos",swapCurve=AQFED.VolatilityModels.makeSwapCurveReplication(ttes, forwards, strikesF, volsF))
           volErrorFS, rmseVol = AQFED.VolatilityModels.estimateVolError(hagParamsGlobalFS, ttes, forwards,  strikesF, volsF)
                      norm((weightsMF .> 0) .* volErrorFS) / norm(weightsMF .> 0)
                     
                      
           
  #=
  set_theme!(theme_light())
  f = Figure(size=(800, 300))
  ax = Axis(f[1,3], xlabel="Maturity", ylabel="ρ")
  l1 =lines!(ax, ttes, hagParamsBootstrap.ρ)
  l2=lines!(ax, ttes, hagParamsGlobal.ρ)
  l3=lines!(ax, ttes, hagParamsBootstrapF.ρ)
  l4=lines!(ax, ttes, hagParamsGlobalF.ρ)
ax = Axis(f[1,2], xlabel="Maturity", ylabel="α")
  lines!(ax, ttes, hagParamsBootstrap.σ)
  lines!(ax, ttes, hagParamsGlobal.σ)
  lines!(ax, ttes, hagParamsBootstrapF.σ)
  lines!(ax, ttes, hagParamsGlobalF.σ)
ax = Axis(f[1,1], xlabel="Maturity", ylabel="σ")
  lines!(ax, ttes, hagParamsBootstrap.leverage)
  lines!(ax, ttes, hagParamsGlobal.leverage)
  lines!(ax, ttes, hagParamsBootstrapF.leverage)
  lines!(ax, ttes, hagParamsGlobalF.leverage)
leg = Legend(f[1, 4], [l1, l2, l3, l4], ["Bootstrap", "Global", "Bootstrap Exact", "Global Exact"], position = :left)

 f = Figure(size=(800, 300))
  ax = Axis(f[1,3], xlabel="Maturity", ylabel="ρ")
  l2=lines!(ax, ttes, hagParamsGlobalF.ρ)
  l4=lines!(ax, ttes, hagParamsGlobalFS.ρ)
ax = Axis(f[1,2], xlabel="Maturity", ylabel="α")
  lines!(ax, ttes, hagParamsGlobalF.σ)
  lines!(ax, ttes, hagParamsGlobalFS.σ)
ax = Axis(f[1,1], xlabel="Maturity", ylabel="σ")
  lines!(ax, ttes, hagParamsGlobalF.leverage)
  lines!(ax, ttes, hagParamsGlobalFS.leverage)
leg = Legend(f[1, 4], [l2, l4], ["Global Exact","Global Spline"], position = :left)
   GLMakie.surface(strikeA, ttes, abs.(volErrorB'),axis=(type=Axis3,xlabel="Strike",ylabel="Expiry",zlabel="Error",yreversed=true),colorscale=sqrt,colormap=:inferno)
   GLMakie.surface(strikeA, ttes, abs.((volErrorB .* (weights .> 0))'),axis=(type=Axis3,xlabel="Strike",ylabel="Expiry",zlabel="Error",yreversed=true),colorscale=sqrt,colormap=:inferno)
=#
tIndex = 3; tte=ttes[tIndex]; forward = forwards[tIndex];
#forward = exp((r-q)*tte)
 tsparams,ttets = makeHestonTSParams(hagParamsGlobal, tte=tte)
 bgmPrices = map( strike -> AQFED.VolatilityModels.priceEuropean(AQFED.VolatilityModels.BGMApprox(tsparams), true, strike, forward, ttets, 1.0), strikeA)
    volB = map((strike, price) -> impliedVolatility(true, max(1e-16,price), forward, strike, tte, 1.0), strikeA, bgmPrices)
    pricer = CharFuncPricing.makeCosCharFuncPricer(DefaultCharFunc(tsparams), ttets, 256,10)
    joshiPrices = map(strike-> CharFuncPricing.priceEuropean(pricer, true, strike, forward, ttets, 1.0), strikeA)
    volJ = map((strike, price) -> impliedVolatility(true, price, forward, strike, tte, 1.0), strikeA, joshiPrices)

    hPrices = map(strike->  AQFED.VolatilityModels.priceEuropean(AQFED.VolatilityModels.HaganHestonTSApprox(hagParamsGlobal,tte), true, strike, forward, tte, 1.0), strikeA)
    volH = map((strike, price) -> impliedVolatility(true, price, forward, strike, tte, 1.0), strikeA, hPrices)

end

#=
strikeA = strikes;forwards = uForwards;ttes = ts; strikes = strikesM;
strikeA = K; ttes = ts

  f = Figure(size=(800, 400))
  ax = Axis(f[1,1], xlabel="Strike", ylabel="Implied volatility in %",title="DJIA - maturity 135 days")
  l1 =lines!(ax, strikeA1, volJ1 .*100)
  l2=lines!(ax, strikeA1, volB1 .*100)
  l3=lines!(ax, strikeA1, volH1 .*100)
 ax = Axis(f[1,2], xlabel="Strike", ylabel="Implied volatility in %", title="SPX500 in 2010 - maturity 175 days")
  l1 =lines!(ax, strikeA2, volJ2 .*100)
  l2=lines!(ax, strikeA2, volB2 .*100)
  l3=lines!(ax, strikeA2, volH2 .*100)
leg = Legend(f[2, 1:2], [l1, l2,l3], ["Fourier", "Benhamou, Gobet, Miri", "Hagan"], position = :center, orientation = :horizontal)

strikeA1 = strikeA; ttes1 = ttes; forwards1 = forwards; hagParamsGlobal1 = hagParamsGlobal; volJ1 = volJ; volB1 = volB; volH1 = volH;
  strikeA2 = strikeA; ttes2 = ttes; forwards2 = forwards; hagParamsGlobal2 = hagParamsGlobal; volJ2 = volJ; volB2 = volB; volH2 = volH;

  strikeA = strikeA1; ttes = ttes1; forwards = forwards1; hagParamsGlobal = hagParamsGlobal1; volJ = volJ1; volB = volB1; volH = volH1;

for i=1:length(ttes)
             @printf("%d & %.2f & %.2f & %.2f & %.2e & %.2f & %.2f &%.2f & %.2e \\\\\n", ttes[i]*365, hagParamsBootstrapF.leverage[i] , hagParamsBootstrapF.ρ[i], hagParamsBootstrapF.σ[i], 100*norm(volErrorBootstrapF[i,:])/sqrt(length(volErrorBootstrapF[i,:])), hagParamsGlobalF.leverage[i], hagParamsGlobalF.ρ[i], hagParamsGlobalF.σ[i], 100*norm(volErrorF[i,:])/sqrt(length(volErrorF[i,:])) )
             end


=#
