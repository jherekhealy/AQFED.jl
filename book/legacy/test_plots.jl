# Plotting code formerly commented out in the unit tests of test/, moved here by test file. It relies on the variables of
# the enclosing testset (named in the header of each block) and on Plots.jl or Makie; the book figures are in book/chNN_*.jl.

# ====================================================================================================
# test/asian_test.jl
# ====================================================================================================

# test/asian_test.jl, @testset "ZhouTable1"
#=
#monthly
mcPrices= [15.299389368297263, 14.57686539496854, 13.856188816900787, 13.13843795070089, 12.425044737027624, 11.717802403042063, 11.018834827058319, 10.330543604033572, 9.655517173703966, 8.996445994327907, 8.355996699402478, 7.736731535780706, 7.1409796180179415, 6.570798661025787, 6.027882223477481, 5.513536051618198, 5.02867842765189, 4.573811533626664, 4.149064430923215, 3.754208711108056, 3.388700004655659, 3.0517309016729675, 2.742269976690799, 2.459096796180759, 2.200857764559134, 1.9661337138886739, 1.7534239467174166, 1.5612016822926376, 1.387953308518999, 1.2321972484574344, 1.092493304346985, 0.9674522072009062, 0.8557592694601095, 0.7561798673899554, 0.6675427909738332, 0.588777779856894, 0.5188896436675161, 0.4569585941402504, 0.4021539919698648, 0.35370850021848865, 0.31092000192847047, 0.2731750494345868, 0.23990870198002806, 0.21060892799999834, 0.1848254386658481, 0.16215079001685728, 0.1422234308998729, 0.12471902347628863, 0.1093466781544504, 0.09585572931635028, 0.08402164384365637]
strikes=(0.5:0.025:1.75).*spot
mcPrices = priceAsianFixedStrike(MonteCarloEngine(true,1024*1024*64),true,strikes,discountFactor,spot,forward,tvar,weights)

p = plot(xlab="Moneyness",ylab="Absolute price error in b.p.")
for pricerName = ["VG1","LB","Ju","VL2","VG2","Deelstra","VG3"  ]
	prices = priceMap[pricerName]
	plot!(p,strikes./spot,abs.(prices-mcPrices)/spot.*1e4,label=pricerName,linestyle=:auto)
end
plot(p, yscale=:log10)
plot!(yticks=[0.001,0.01,0.1,1.0])
plot!(legend=:outerright)
plot!(size=(800,400),margins=3Plots.mm)


=#

# test/asian_test.jl, @testset "LordAsianTable4"
#=
pricesMC= [60.60793606595354, 57.31247650189226, 54.18267122327183, 51.22169542916771, 48.428689318228436, 45.799892406977825, 43.32961151397672, 41.010861660960984, 38.835923024869295, 36.79673535378197, 34.88514243404879, 33.0931513698638, 31.412976582405495, 29.837175908368867, 28.358712611079447, 26.970903595775752, 25.667502001113547, 24.442724717752604, 23.291130513139986, 22.20760640041994, 21.18754008290338, 20.226566182432585, 19.32064393747319, 18.46605468275837, 17.659361603890037, 16.897395893527765, 16.177157682042477, 15.495953799532687, 14.851175540385004, 14.240563255431692, 13.661933820634692, 13.113203418622547, 12.592604197411841, 12.098355732354102, 11.628815077738501, 11.182494542243257, 10.758053795841896, 10.3541612683104, 9.96958437046821, 9.60322018356278, 9.254050823357348, 8.921077821386628, 8.603376566516136, 8.300101325136666, 8.010444289493691, 7.733695081382217, 7.469144109237979, 7.216094296562982, 6.973973736234091, 6.742237438096218, 6.520300674823362, 6.307674642221388, 6.103880670867341, 5.908473632679069, 5.721009724293115, 5.5411245269133325, 5.368424281306374, 5.202567907208277, 5.0432393259777974, 4.890112978420219, 4.742895199903845, 4.601318688629397, 4.465112302948394]
strikes=40.0:5.0:350.0
moneyness = strikes./sum(forward.*weights) .- 1.0
 #30y    
pricesMC= [37.365032448359585, 36.48660338642259, 35.63504352533947, 34.80990453125456, 34.01064801857199, 33.23664981514158, 32.487189002611856, 31.76155012362347, 31.058982053848837, 30.378701440907804, 29.719919907882154, 29.08188096873142, 28.463832175512422, 27.865034926482096, 27.28477160714669, 26.722343119154925, 26.177083085911846, 25.648315791333253, 25.135429954329346, 24.63782921911073, 24.154908631345226, 23.686137700484313, 23.230967486253554, 22.788868684629225, 22.359366178867045, 21.942002149312046, 21.536336858000634, 21.141921260158334, 20.758334949681856, 20.385198821450498, 20.022116582162624, 19.668744067551522, 19.32473033503131, 18.989747683043717, 18.663473699118867, 18.345608578060784, 18.03585585752176, 17.73394149223119, 17.439608211322614, 17.152595511776347, 16.87265730365258, 16.59955325460831, 16.33307864817904, 16.07300741174868, 15.819144312989975, 15.571279826401108, 15.329219950454616, 15.092778469243957, 14.86178626469504, 14.636068576230219, 14.415467366301085, 14.199838194878023, 13.989012712690275, 13.782834965146916, 13.581181972271818, 13.383931007232004, 13.190944726331512, 13.002096279427942, 12.817266247431398, 12.636341737254426, 12.459211232571572, 12.285758838104613, 12.115879447780353, 11.949484535029985, 11.786479801969488, 11.626759570476498, 11.47024491585437, 11.316846190164963, 11.16648065052727, 11.019078129069072, 10.87455045174692, 10.73281615962248, 10.593805714881254, 10.457443579079992, 10.323666109992091, 10.192417816791039, 10.06363312276608, 9.937246823652565, 9.813199123614485, 9.691433734921587, 9.571881947415145, 9.45449515576401, 9.33922137221768, 9.226017535594156, 9.114822452556895, 9.005590267659453, 8.898273955588136, 8.792837697641698, 8.689235586156213, 8.587412484273155, 8.487331330236309, 8.388958445475467, 8.29225624153133, 8.197175922257838, 8.10368958596068, 8.01176246449686, 7.9213598259412965, 7.8324353105234215, 7.744962190320569, 7.658917615817006, 7.574263311599525, 7.490970287329001, 7.409006235879907, 7.328338228003912, 7.248951363844231, 7.17082325881341, 7.0939244104296515, 7.018223355774483, 6.943694279458629, 6.870316523059794, 6.798066235362698, 6.726913927303102, 6.656844572019831, 6.587831142702975, 6.519848569819195, 6.452881402893441, 6.386911023928128, 6.321916755849759, 6.257875062020297, 6.194773152032169, 6.132594760620767, 6.071320118725697, 6.0109294224176235, 5.951397795289828, 5.892715985266672, 5.8348652888824795, 5.777834742845455, 5.721612997437061, 5.666180550102959, 5.611520482074068, 5.5576152586233, 5.5044554344170304, 5.452024374513652, 5.4003122558483545, 5.349305850079876, 5.298993210499051, 5.249360662869793, 5.200394858591663, 5.152081133994656, 5.104410943522941, 5.05737298639524, 5.010954869373652, 4.965149590761396, 4.9199418003102355, 4.87532023869603, 4.831285056760686]
strikes=75.0:5.0:800.0
#mcPrices = priceAsianFixedStrike(MonteCarloEngine(true,1024*1024*64),true,strikes,discountFactor,spot,forward,tvar,weights)
moneyness = strikes./sum(forward.*weights) .- 1.0
p = plot(xlab="Asian moneyness",ylab="Absolute price error in b.p.")
for pricerName = ["LB(2,3)","VG1","LB(3,3)","Ju","VG2","D(2,3)","D(3,3)","VG3","VL3"  ]
	prices = priceMap[pricerName]
	plot!(p,moneyness,abs.(prices-pricesMC)/spot.*1e4,label=pricerName,linestyle=:auto)
end
plot(p, yscale=:log10)
plot!(yticks=[0.01,0.1,1.0,10.0])
plot!(legend=:outerright)
plot!(size=(800,400),margins=3Plots.mm)
#non log
p = plot(xlab="Asian moneyness",ylab="Price error in b.p.")
for pricerName = ["VG1","VG2", "LB(2,3)","LB(3,3)", "Ju" ]
	prices = priceMap[pricerName]
	plot!(p,moneyness,(prices-pricesMC)/spot.*1e4,label=pricerName,linestyle=:auto)
end
#    plot!(p,yticks=[0.01,0.1,1.0,10.0])
#plot!(p,legend=:outerright)
plot!(p,size=(400,400),margins=3Plots.mm)

p2 = plot(xlab="Asian moneyness",ylab="Price error in b.p.")
for pricerName = ["VG3","VG2" ,"D(2,3)","D(3,3)","VL3" ]
	prices = priceMap[pricerName]
	plot!(p2,moneyness,(prices-pricesMC)/spot.*1e4,label=pricerName,linestyle=:auto)
end
#    plot!(p2, yticks=[0.01,0.1,1.0,10.0])
#    plot!(p2, legend=:outerright)
plot!(p2, size=(400,400),margins=3Plots.mm)


=#
# ====================================================================================================
# test/basket_test.jl
# ====================================================================================================

# test/basket_test.jl, @testset "DeelstraBeisser"
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

# test/basket_test.jl, @testset "basketGreeks"
#= plot(data.Rho, data.Delta1, label="Δ1",xlab="Correlation",ylab="Δ value",legend=:outerright)
   plot!(data.Rho, data.ScaledDelta1, label="qΔA")
   plot!(data.Rho, data.Delta2, label="Δ2")
plot!(size=(400,300),margins=3Plots.mm)


    plot(data.Strike, data.Delta1, label="Basket Delta1",xlab="Basket strike",ylab="Delta value",legend=:outerright)
   plot!(data.Strike, data.ScaledDelta1, label="Index Delta1")
   plot!(data.Strike, data.Delta2, label="Basket Delta2")
   plot!(data.Strike, data.ScaledDelta2, label="Index Delta2")

=#

# test/basket_test.jl, @testset "basketGreeks"
#move1=0.02;move2=-0.02;  plot(data.Rho, data.Delta1*move1 + data.Delta2*move2,xlab="Correlation",ylab="Δ hedge value",label="")
#, label="Independent hedge")  # while basket hedge===0
# ====================================================================================================
# test/collocation_tests.jl
# ====================================================================================================

# test/collocation_tests.jl
#using Plots

# test/collocation_tests.jl, @testset "deltavol"
#=    plot(k, vols.*100, seriestype= :scatter, label="Reference"); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(kFine, ivkFine3 .*100 ,label="Cubic collocation")
plot!(kFine, ivkFine5 .*100 ,label="Quintic collocation")
plot!(kFine, ivkFineexp.*100,label="Exp. B-spline collocation")
plot!(kFine, ivkFineLVG .*100 ,label="LVG")
plot!(kFine, ivkFineSVI .* 100, label="SVI")
=#

# test/collocation_tests.jl, @testset "deltavol"
#=
plot(log.(kFine./forward), pdf.(ppDeltaStrike,kFine),label="Cubic spline on Δ")
plot!(log.(kFine./forward),Collocation.density.(sol,kFine),label="Quintic collocation")
plot!(log.(kFine./forward),(AQFED.PDDE.derivativePrice.(lvg,true,kFine.+0.0001) .- AQFED.PDDE.derivativePrice.(lvg,true,kFine)).*10000, label="LVG")
plot!(log.(kFine./forward),(AQFED.PDDE.derivativePrice.(lvg,true,kFine.+0.0001) .- AQFED.PDDE.derivativePrice.(lvg,true,kFine)).*10000, label="LVG")
plot!(log.(kFine./forward),AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label="Fengler")

=#

# test/collocation_tests.jl, @testset "jaeckel2"
#plot(y,@.(gatheralDenomFinite(x->AQFED.TermStructure.varianceByLogmoneyness(slice,x)*tte,y)))
#plot(y,@.(gatheralDenomFinite(x->AQFED.TermStructure.varianceByLogmoneyness(slice,x)*tte,y)/sqrt(2*π*AQFED.TermStructure.varianceByLogmoneyness(slice,y)*tte)*exp(-0.5*(y/sqrt(AQFED.TermStructure.varianceByLogmoneyness(slice,y)*tte)+sqrt(AQFED.TermStructure.varianceByLogmoneyness(slice,y)*tte)/2)^2)  )  )
# ====================================================================================================
# test/fdm_test.jl
# ====================================================================================================

# test/fdm_test.jl, @testset "AmericanPutGamma"
#=
p1 = plot(priceCN.x,PPInterpolation.evaluateSecondDerivative.(priceCN,priceCN.x),ylims=(-0.005,0.075),yticks=[0,0.025,0.05,0.075],label="Crank-Nicolson")
p2 = plot(priceRAN.x,PPInterpolation.evaluateSecondDerivative.(priceRAN,priceRAN.x),ylims=(-0.005,0.075),yticks=([0,0.025,0.05,0.075],["","","",""]),label="Rannacher")
p3 = plot(priceTRBDF2.x,PPInterpolation.evaluateSecondDerivative.(priceTRBDF2,priceTRBDF2.x),ylims=(-0.005,0.075),yticks=([0,0.025,0.05,0.075],["","","",""]),label="TR-BDF2")
p4 = plot(priceRKL.x,PPInterpolation.evaluateSecondDerivative.(priceRKL,priceRKL.x),ylims=(-0.005,0.075),yticks=([0,0.025,0.05,0.075],["","","",""]),label="RKL")
p5 = plot(priceRKL.x,PPInterpolation.evaluateSecondDerivative.(priceRKG,priceRKG.x),ylims=(-0.005,0.075),yticks=([0,0.025,0.05,0.075],["","","",""]),label="RKG")
 plot(p1,p2,p3,p4,p5,layout=(1,5),size=(1024,200),margins=1Plots.mm)

=#

# test/fdm_test.jl, @testset "DiscretKOTavellaRandall"
# udata = data[startswith.(data.grid,"Uniform"),:]
# plot(udata.m,abs.(udata.error), group=udata.grid,yscale=:log10,ylabel="Absolute error in price", xlabel="number of space-steps",size=(400,300))
# savefig("~//mypapers/eqd_book/uniform_kreiss_time_vanilla.pdf")
# udata = data[startswith.(data.grid,"Cubic"),:]
# plot(udata.m,abs.(udata.error), group=udata.grid,yscale=:log10,ylabel="Absolute error in price", xlabel="number of space-steps",size=(400,300))
# savefig("~//mypapers/eqd_book/cubic_kreiss_time_vanilla.pdf")
#plot(udata.m,abs.(udata.error), group=udata.grid,yscale=:ln,yticks=([0.0001,0.0002,0.0004],[0.0001,0.0002,0.0004]),ylabel="Absolute error in price", xlabel="number of time-steps",mode="markers")

# test/fdm_test.jl, @testset "ExerciseBoundary"
#= plot(ms,abs.(alPrices .- refPrice), label="Andersen-Lake nₚₚ=1", xlabel="m", ylabel="Absolute error in price", yscale= :log10)
 plot!(ms,abs.(alppPrices .- refPrice), label="Andersen-Lake nₚₚ=m", xlabel="m", ylabel="Absolute error in price", yscale= :log10)
plot!(size=(300,200))

 savefig("~//mypapers/eqd_book/alpp_eur_error.pdf")
 savefig("~//mypapers/eqd_book/alpp_try_error_vol20.pdf")

 =#

# test/fdm_test.jl, @testset "ExerciseBoundary"
#=   
using PlotThemes
theme(:vibrant)
plot(payoffB.exerciseTimes, payoffB.exerciseBoundary, label="TR-BDF2", xlab="Time to maturity", ylab="Asset price")
plot!( t,AQFED.American.exerciseBoundary(al, spot, t), label="Andersen-Lake m=9 nₚₚ=1")
plot!( t,AQFED.American.exerciseBoundary(alpp, spot, t), label="Andersen-Lake m=5 nₚₚ=9")
plot!(size=(400,300))
plot( t,AQFED.American.exerciseBoundary(alg1, spot, t), label="Andersen-Lake m=9 nₚₚ=1")
plot!(payoffB.exerciseTimes, payoffB.exerciseBoundary, label="TR-BDF2", xlab="Time", ylab="Asset price", size=(450,300))
plot!( t,AQFED.American.exerciseBoundary(alg5, spot, t), label="Andersen-Lake m=5 nₚₚ=9",linestyle=:dot)
savefig("~//mypapers/eqd_book/alpp_eur_spread.pdf")

savefig("~//mypapers/eqd_book/alpp_eur_spread.pdf")
plot(t, -ForwardDiff.derivative.(eurpp,t) .* 100,xlab="time in ACT/365", ylab="Rate in %",label="")
plot!(size=(400,200))
savefig("~//mypapers/eqd_book/eur_forward_rate.pdf")
plot(t, -eurpp.(t) ./t .* 100,xlab="time in ACT/365", ylab="Rate in %",label="")
plot!(size=(400,200))
savefig("~//mypapers/eqd_book/eur_rate.pdf")
plot(t, -ForwardDiff.derivative.(trypp,t) .* 100,xlab="time in ACT/365", ylab="Rate in %",label="")
plot!(size=(400,200))
savefig("~//mypapers/eqd_book/try_forward_rate.pdf")
plot(t, -trypp.(t) ./t .* 100,xlab="time in ACT/365", ylab="Rate in %",label="")
plot!(size=(400,200))
savefig("~//mypapers/eqd_book/try_rate.pdf")

=#

# test/fdm_test.jl, @testset "ExerciseBoundary"
#=   
using PlotThemes
theme(:vibrant)
plot(payoffB.exerciseTimes, payoffB.exerciseBoundary, label="TR-BDF2", xlab="Time to maturity", ylab="Asset price")
plot!( t,AQFED.American.exerciseBoundary(al, spot, t), label="Andersen-Lake m=7 nₚₚ=1")
plot!( t,AQFED.American.exerciseBoundary(alpp, spot, t), label="Andersen-Lake m=5 nₚₚ=7")
plot!(size=(400,300))
savefig("~//mypapers/eqd_book/alpp_try_eur.pdf")
=#

# test/fdm_test.jl, @testset "ExerciseBoundary"
# plot(t,AQFED.American.exerciseBoundary(al, strike, t))

# test/fdm_test.jl, @testset "ExerciseBoundary"
# plot( t,AQFED.American.exerciseBoundary(alg, spot, t), label="Andersen-Lake m=15")
# plot!(payoffB.exerciseTimes, payoffB.exerciseBoundary, label="TR-BDF2", xlab="Time", ylab="Asset price", size=(450,300))
# savefig("~//mypapers/eqd_book/al_prop_nolog.pdf")

# test/fdm_test.jl, @testset "GridShift"
#udata = data[@.(startswith(data.grid,"Log")),:]
#julia> udata = data[@.(startswith(data.grid,"Cubic-0") || startswith(data.grid,"Uniform") || startswith(data.grid,"LogGrid")),:]
#julia> plot(udata.Strike,abs.(udata.error),group=udata.grid,size=(800,300),xlabel="Strike",ylabel="Absolute error in price",margin=3Plots.mm)



#        
#         labels[1,8] = "LogSchemeUniform Uncalibrated"
#         data[i,8]=price-blackPrice
#         println(strike, " LogSchemeUniform Uncalibrated ",price, " ",price-blackPrice)
#         price = AQFED.FDM.priceLogTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,calibration=ForwardCalibration(),grid="Shift")(spot)
#         labels[1,9] = "LogSchemeUniform Calibrated"
#         data[i,9]=price-blackPrice
# println(strike, " LogSchemeUniform Calibrated ",price, " ",price-blackPrice)
#         price = AQFED.FDM.priceLogTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,Smin=-1,calibration=NoCalibration(),grid="Cubic")(spot)
#         labels[1,10] = "LogSchemeCubic Uncalibrated"
#         data[i,10]=price-blackPrice
#         println(strike, " LogSchemeCubic Uncalibrated ",price, " ",price-blackPrice)
#             end
#         end
#julia> plot(strikes,abs.(data),label=labels,legend=:outertopleft,markershapes=[:circle],yscale=:log10)
# ====================================================================================================
# test/fdmbarrier_test.jl
# ====================================================================================================

# test/fdmbarrier_test.jl, @testset "BlackUpAndOut"
#=
plot(spots./strike, abs.(refPrice.(spots) .- anPriceA), xlab="Spot/Strike", ylab="Absolute error in price", label="Ghost");    plot!(spots./strike, abs.(refPriceK.(spots) .- anPriceA), label="Ghost Kreiss");    plot!(spots./strike, abs.(refPriceS.(spots) .- anPriceA), label="Deformed")
plot!(size=(400,300),margins=1Plots.mm)
savefig("~//mypapers/eqd_book/barrier_ko_spot_401.pdf")
=#

# test/fdmbarrier_test.jl, @testset "BlackUpAndOut"
#=
   plot(xStepsA,abs.(err),label="Ghost");plot!(xStepsA, abs.(errK),xlab="Number of space steps", ylab="Absolute error in price",label="Ghost Kreiss");plot!(xStepsA,abs.(errS),label="Deformed")
 plot!(size=(400,300),margins=1Plots.mm)
savefig("~//mypapers/eqd_book/barrier_ko_steps_100.pdf")  
=#

# test/fdmbarrier_test.jl, @testset "Peclet"
    #=
    etraceE = AQFED.FDM.ExtendedTrace()
    etrace = AQFED.FDM.ExtendedTrace()
    refPrice = AQFED.FDM.priceRKG2(payoffK, spot, model, dividends, M=101, N=101, Smin=0.0, Smax=150.0, grid=UniformGrid(false),eTrace=etraceE,varianceConditioner=PartialExponentialFittingConditioner())
    refPrice = AQFED.FDM.priceRKG2(payoffK, spot, model, dividends, M=101, N=101, Smin=0.0, Smax=150.0, grid=UniformGrid(false),eTrace=etrace)
    plot(refPrice.x,PPInterpolation.evaluateDerivative.(refPrice,refPrice.x),xlab="Underlying spot price", ylab="Option Δ",label="RKL",size=(400,300))
     plot(etrace.eigenvalues, seriestype=:scatter,label="Partial Exponential Fitting",size=(400,300),ms=3,markerstrokewidth=0,markeralpha=0.5)
plot(refPrice.x[2:end-1],etrace.peclets[2:end-1],xlab="Underlying asset price", ylab="Peclet ratio",yscale=:log10,ylims=(0.5,250),yticks=([1,10,100],[1,10,100]),label="No upwinding")
 plot!(refPrice.x[2:end-1],etraceE.peclets[2:end-1],label="Partial exponential fitting")

     =#

# test/fdmbarrier_test.jl, @testset "BlackOneTouch"
   #= 
   is = collect(n-50:n); prices = zeros(length(is))
   for (i,ni)=enumerate(is)
   refPriceR = AQFED.FDM.priceExplicitEuler(payoff, spot, model, dividends, M=101, N=ni, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
   prices[i] = refPriceR(spot)
   end
    plot(is,abs.(prices.-anPrice),yscale=:log10,xlab="Number of time-steps",ylab="Absolute error in price",legend=:none)
    #plot(is[50:end],abs.(prices[50:end].-anPrice),yscale=:log10,xlab="Number of time-steps",ylab="Absolute error in price",legend=:none)
Srange = 13590.0:1.0:13990.0
nA = zeros(length(Srange));
neffA = zeros(length(Srange));
nETruncA = zeros(length(Srange));
    for (i,Smax)=enumerate(Srange)
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
      dS = refPrice.x[2]-refPrice.x[1]
    ius = searchsorted(refPrice.x,barrierLevel)
    ium = ius.stop; iu=ius.start
   
     dtThresholdETrunc = dS^2/(σ^2*barrierLevel^2)
    nETrunc = ceil(Int,tte/dtThresholdETrunc)
 dtThresholdEFull = dS^2/(σ^2*refPrice.x[end]^2)
    nEFull = ceil(Int,tte/dtThresholdE)
    nETruncA[i] = nETrunc
    dtThreshold = 4*dS^2/(σ^2*refPrice.x[ium]^2*(3+(refPrice.x[iu]-barrierLevel)/(barrierLevel-refPrice.x[ium])))
    n = ceil(Int,tte/dtThreshold)
    errorR = 1.0
    ni = n
    while errorR > 1e-3 && ni < 30000
        ni+=1
        refPriceRBad = AQFED.FDM.priceExplicitEuler(payoff, spot, model, dividends, M=101, N=ni, Smin=Smin, Smax=Smax, grid=UniformGrid(false))
        errorR = abs(refPriceRBad(spot)-anPrice)
    end
  while errorR < 1e-3 && ni > 1
        refPriceRBad = AQFED.FDM.priceExplicitEuler(payoff, spot, model, dividends, M=101, N=ni, Smin=Smin, Smax=Smax, grid=UniformGrid(false))
        errorR = abs(refPriceRBad(spot)-anPrice)
        ni-=1
    end
    nA[i] = n
    neffA[i] = ni
    println(Smax," ",barrierLevel-refPrice.x[ium], " ",(barrierLevel-refPrice.x[ium])/dS," ",n, " ",ni," ",errorR)
end
nA2 = max.(nA, nETruncA)
plot(Srange, nA2, label="Theoretical",xlab="Grid upper boundary Sₘₐₓ",ylab="Number of time-steps for stability")
plot!(Srange,neffA,label="Actual",legendtitle="Threshold")
plot!(yscale=:log10)
plot!(size)
   =#

# test/fdmbarrier_test.jl, @testset "BlackUpAndIn"
#=
plot(spots./strike, abs.(refPrice.(spots) .- anPriceA), xlab="Spot/Strike", ylab="Absolute error in price", label="Ghost");    plot!(spots./strike, abs.(refPriceK.(spots) .- anPriceA), label="Ghost Kreiss");    plot!(spots./strike, abs.(refPriceS.(spots) .- anPriceA), label="Deformed")
plot!(size=(400,300),margins=1Plots.mm)
savefig("~//mypapers/eqd_book/barrier_ki_spot_401.pdf")
=#

# test/fdmbarrier_test.jl, @testset "BlackUpAndIn"
#=
   plot(xStepsA,abs.(err),label="Ghost");plot!(xStepsA, abs.(errK),xlab="Number of space steps", ylab="Absolute error in price",label="Ghost Kreiss");plot!(xStepsA,abs.(errS),label="Deformed")
 plot!(size=(400,300),margins=1Plots.mm)
savefig("~//mypapers/eqd_book/barrier_ko_steps_100.pdf")  
=#

# test/fdmbarrier_test.jl, @testset "BlackUpAndInThenDownAndOut"
#=
plot(spots./strike, abs.(refPrice.(spots) .- anPriceA), xlab="Spot/Strike", ylab="Absolute error in price", label="Ghost");    plot!(spots./strike, abs.(refPriceK.(spots) .- anPriceA), label="Ghost Kreiss");    plot!(spots./strike, abs.(refPriceS.(spots) .- anPriceA), label="Deformed")
plot!(size=(400,300),margins=1Plots.mm)
savefig("~//mypapers/eqd_book/barrier_kiko_spot_401_cubic01.pdf")
=#

# test/fdmbarrier_test.jl, @testset "BlackUpAndInThenDownAndOut"
#=
   plot(xStepsA,abs.(err),label="Ghost");plot!(xStepsA, abs.(errK),xlab="Number of space steps", ylab="Absolute error in price",label="Ghost Kreiss");plot!(xStepsA,abs.(errS),label="Deformed")
 plot!(size=(400,300),margins=1Plots.mm)
savefig("~//mypapers/eqd_book/barrier_ko_steps_100.pdf")  
=#

# test/fdmbarrier_test.jl, @testset "BlackDKO"
#= 
   plot(refPriceS.x,PPInterpolation.evaluateDerivative.(refPriceS,refPriceS.x),xlab="Underlying spot price", ylab="Option Δ",label="TR-BDF2")
   plot!(priceRKG.x,PPInterpolation.evaluateDerivative.(priceRKG,priceRKG.x),xlab="Underlying spot price", ylab="Option Δ",label="RKG")
   plot!(xlim=(barrierLevelDown,barrierLevelUp),size=(400,200),margin=3Plots.mm)
 savefig("~//mypapers/eqd_book/doubleonetouch_delta_trbdf2.pdf")
  plot(priceCN.x,PPInterpolation.evaluateDerivative.(priceCN,priceCN.x),xlab="Underlying spot price", ylab="Option Δ",label="")
   plot!(xlim=(barrierLevelDown,barrierLevelUp),size=(400,200),margin=3Plots.mm)
 savefig("~//mypapers/eqd_book/doubleonetouch_delta_cn.pdf")
  plot(priceRAN.x,PPInterpolation.evaluateDerivative.(priceRAN,priceRAN.x),xlab="Underlying spot price", ylab="Option Δ",label="")
   plot!(xlim=(barrierLevelDown,barrierLevelUp),size=(400,200),margin=3Plots.mm)
 savefig("~//mypapers/eqd_book/doubleonetouch_delta_ran.pdf")
plot(priceRAN4.x,PPInterpolation.evaluateDerivative.(priceRAN4,priceRAN4.x),xlab="Underlying spot price", ylab="Option Δ",label="")
   plot!(xlim=(barrierLevelDown,barrierLevelUp),size=(400,200),margin=3Plots.mm)
 savefig("~//mypapers/eqd_book/doubleonetouch_delta_ran4.pdf")

   =#

# test/fdmbarrier_test.jl, @testset "ContinousKO"
# udata = data[startswith.(data.grid,"Uniform"),:]
# plot(udata.m,abs.(udata.error), group=udata.grid,yscale=:log10,ylabel="Absolute error in price", xlabel="number of space-steps",size=(400,300))
# savefig("~//mypapers/eqd_book/uniform_kreiss_time_vanilla.pdf")
# udata = data[startswith.(data.grid,"Cubic"),:]
# plot(udata.m,abs.(udata.error), group=udata.grid,yscale=:log10,ylabel="Absolute error in price", xlabel="number of space-steps",size=(400,300))
# savefig("~//mypapers/eqd_book/cubic_kreiss_time_vanilla.pdf")
#plot(udata.m,abs.(udata.error), group=udata.grid,yscale=:ln,yticks=([0.0001,0.0002,0.0004],[0.0001,0.0002,0.0004]),ylabel="Absolute error in price", xlabel="number of time-steps",mode="markers")
# ====================================================================================================
# test/hestonts_test.jl
# ====================================================================================================

# test/hestonts_test.jl, @testset "HestonTS-SPX500-Oct2010"
  #=
  set_theme!(theme_light())
  f = Figure(size=(800, 300))
  ax = Axis(f[1,3], xlabel="Maturity", ylabel=L"$\rho$")
  l1 =lines!(ax, ttes, hagParamsBootstrap.ρ)
  l2=lines!(ax, ttes, hagParamsGlobal.ρ)
  l3=lines!(ax, ttes, hagParamsBootstrapF.ρ)
  l4=lines!(ax, ttes, hagParamsGlobalF.ρ)
ax = Axis(f[1,2], xlabel="Maturity", ylabel=L"$\alpha$")
  lines!(ax, ttes, hagParamsBootstrap.σ)
  lines!(ax, ttes, hagParamsGlobal.σ)
  lines!(ax, ttes, hagParamsBootstrapF.σ)
  lines!(ax, ttes, hagParamsGlobalF.σ)
ax = Axis(f[1,1], xlabel="Maturity", ylabel=L"$\sigma$")
  lines!(ax, ttes, hagParamsBootstrap.leverage)
  lines!(ax, ttes, hagParamsGlobal.leverage)
  lines!(ax, ttes, hagParamsBootstrapF.leverage)
  lines!(ax, ttes, hagParamsGlobalF.leverage)
leg = Legend(f[1, 4], [l1, l2, l3, l4], ["Bootstrap", "Global", "Bootstrap Exact", "Global Exact"], position = :left)

 f = Figure(size=(800, 300))
  ax = Axis(f[1,3], xlabel="Maturity", ylabel=L"$\rho$")
  l2=lines!(ax, ttes, hagParamsGlobalF.ρ)
  l4=lines!(ax, ttes, hagParamsGlobalFS.ρ)
ax = Axis(f[1,2], xlabel="Maturity", ylabel=L"$\alpha$")
  lines!(ax, ttes, hagParamsGlobalF.σ)
  lines!(ax, ttes, hagParamsGlobalFS.σ)
ax = Axis(f[1,1], xlabel="Maturity", ylabel=L"$\sigma$")
  lines!(ax, ttes, hagParamsGlobalF.leverage)
  lines!(ax, ttes, hagParamsGlobalFS.leverage)
leg = Legend(f[1, 4], [l2, l4], ["Global Exact","Global Spline"], position = :left)
   GLMakie.surface(strikeA, ttes, abs.(volErrorB'),axis=(type=Axis3,xlabel="Strike",ylabel="Expiry",zlabel="Error",yreversed=true),colorscale=sqrt,colormap=:inferno)
   GLMakie.surface(strikeA, ttes, abs.((volErrorB .* (weights .> 0))'),axis=(type=Axis3,xlabel="Strike",ylabel="Expiry",zlabel="Error",yreversed=true),colorscale=sqrt,colormap=:inferno)
=#

# test/hestonts_test.jl, @testset "HestonTS-SPX500-Oct2010"
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

swapCurve=AQFED.VolatilityModels.makeSwapCurveReplication(ttes, forwards, strikes, vols)

swapCurveForward = AQFED.TermStructure.PiecewiseConstantFunction(vcat(0.0,ttes[1:end-1]), AQFED.VolatilityModels.computeForwardVariance(swapCurve, ttes))
 tsmall = range(0.0,ttes[end],length=101)
p1 = plot(tsmall, map(t->swapCurveForward(t),tsmall),label="Replication - full range")

volsF, strikesF = AQFED.VolatilityModels.filterVolsBySimpleDelta(ttes, forwards, strikes, vols, deltas=collect(range(0.1,0.9,length=101)))
 pricesF, isCallF, weightsMF = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ttes, forwards, strikesF, volsF, vegaFloor=1e-4,truncationDev=10.0)

 hagParamsGlobalF, rmse = AQFED.VolatilityModels.calibrateHestonHaganFromPricesParam(ttes, forwards, strikesF, pricesF, isCallF, weightsMF, lower=[1e-2,-0.99,0.05],upper=[2.0,0.5,20.0], numberOfKnots=[length(ttes),length(ttes),length(ttes)], κ=κ, minimizer="User", method="Cos",swapCurve=AQFED.VolatilityModels.makeSwapCurveReplication(ttes, forwards, strikesF, volsF))


swapCurveF=AQFED.VolatilityModels.makeSwapCurveReplication(ttes, forwards, strikesF, volsF)
swapCurveForwardF = AQFED.TermStructure.PiecewiseConstantFunction(vcat(0.0,ttes[1:end-1]), AQFED.VolatilityModels.computeForwardVariance(swapCurveF, ttes))
   pricesF, isCallF, weightsMF = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ttes, forwards, strikesF, volsF, vegaFloor=1e-4,truncationDev=10.0)

plot!(p1,tsmall, map(t->swapCurveForwardF(t),tsmall),label="Replication - [10Δ, 90Δ]")


   hagParamsCurveF = AQFED.TermStructure.PiecewiseConstantFunction(vcat(0.0,ttes[1:end-1]),hagParamsGlobalF.leverage.^2)
 plot!(p1, tsmall, map(t->hagParamsCurveF(t),tsmall),label="Time-dependent Heston")
hagParamsCurveBF = AQFED.TermStructure.PiecewiseConstantFunction(vcat(0.0,ttes[1:end-1]),hagParamsBootstrapF.leverage.^2)

boot and global close to each other but different from repl. And repl 10% delta quite different at short end than repl full.

iExpiry = 3
paramsF0, lastτ = makeHestonTSParams(hagParamsBootstrapF0, tte=ttes[iExpiry])
   pricer = CharFuncPricing.makeCosCharFuncPricer(DefaultCharFunc(paramsF0),lastτ,256,8)
mprices = map(strike -> CharFuncPricing.priceEuropean(pricer, true, strike, 1.0, lastτ, 1.0), strikes[iExpiry,:] ./ forwards[iExpiry])
mvols0 = map((strike,mprice) -> Black.impliedVolatility(true, mprice, 1.0, strike, ttes[iExpiry], 1.0), strikes[iExpiry,:] ./ forwards[iExpiry], mprices)

paramsF1, lastτ = makeHestonTSParams(hagParamsBootstrapF1, tte=ttes[iExpiry])
   pricer = CharFuncPricing.makeCosCharFuncPricer(DefaultCharFunc(paramsF1),lastτ,256,8)
mprices = map(strike -> CharFuncPricing.priceEuropean(pricer, true, strike, 1.0, lastτ, 1.0), strikes[iExpiry,:] ./ forwards[iExpiry])
mvols1 = map((strike,mprice) -> Black.impliedVolatility(true, mprice, 1.0, strike, ttes[iExpiry], 1.0), strikes[iExpiry,:] ./ forwards[iExpiry], mprices)

paramsF, lastτ = makeHestonTSParams(hagParamsBootstrapF, tte=ttes[iExpiry])
   pricer = CharFuncPricing.makeCosCharFuncPricer(DefaultCharFunc(paramsF),lastτ,256,8)
mprices = map(strike -> CharFuncPricing.priceEuropean(pricer, true, strike, 1.0, lastτ, 1.0), strikes[iExpiry,:] ./ forwards[iExpiry])
mvols = map((strike,mprice) -> Black.impliedVolatility(true, mprice, 1.0, strike, ttes[iExpiry], 1.0), strikes[iExpiry,:] ./ forwards[iExpiry], mprices)

  hagParamsCurveF = AQFED.TermStructure.PiecewiseConstantFunction(vcat(0.0,ttes[1:end-1]),hagParamsBootstrapF0.leverage.^2)

  f = Figure(size=(400, 300))
  ax = Axis(f[1,1], xlabel="Time in years in Actual/365", ylabel="Forward variance")
  l1 =lines!(ax, tsmall, map(t->swapCurveForward(t),tsmall),label=L"Replication full range$$")
  l2=lines!(ax, tsmall, map(t->swapCurveForwardF(t),tsmall),label=L"Replication $\Delta \in [10%, 90%]$")
  l3=lines!(ax, tsmall,map(t->hagParamsCurveF(t),tsmall),label=L"Time-dependent Heston$$")
  axislegend(position = :rb)


  f = Figure(size=(400, 300))
  ax = Axis(f[1,1], xlabel="Strikes", ylabel="Implied volatility in %")
   l1 =Makie.scatter!(ax, strikes[iExpiry,:], vols[iExpiry,:].*100,label=L"Market$$",color=:black)
  l2=lines!(ax, strikes[iExpiry,:], mvols1.*100, label=L"Reduced calibration on full range$$")
  l3=lines!(ax, strikes[iExpiry,:], mvols.*100, label=L"Reduced calibration $\Delta \in [10%, 90%]$")  
  l4=lines!(ax, strikes[iExpiry,:], mvols0.*100, label=L"Full calibration$$")
  axislegend(position = :rt)

 ax = Axis(f[1,2], xlabel="Strike", ylabel="Implied volatility in %", title="SPX500 in 2010 - maturity 175 days")
  l1 =lines!(ax, strikeA2, volJ2 .*100)
  l2=lines!(ax, strikeA2, volB2 .*100)
  l3=lines!(ax, strikeA2, volH2 .*100)
leg = Legend(f[2, 1:2], [l1, l2,l3], ["Fourier", "Benhamou, Gobet, Miri", "Hagan"], position = :center, orientation = :horizontal)

=#
# ====================================================================================================
# test/pln_european_test.jl
# ====================================================================================================

# test/pln_european_test.jl, @testset "EtoreGobetSingle"
        #=
        using CairoMakie
            f = Figure();
    ax = Axis(f[1,1],xlabel="Dividend Ex-Date", ylabel="Error in Implied Volatility %", yscale=log10)
    for pricerName = ["Deelstra", "GS", "LL2", "LL3", "VL2","VL3", "Ju"]
        lines!(ax, ttdList, abs.(volMap[pricerName]), label=pricerName)
    endc
    axislegend(; position = :cb, nbanks = 3)
    ylims!(1e-8, 1e-2)
                    
    dfn = df[df.Model .=="LL2" .|| df.Model .== "LL3" .|| df.Model .== "GS" .|| df.Model .== "Deelstra" .|| df.Model .== "Ju" .|| df.Model .== "VL2" .|| df.Model .== "VL3",:]
dfn.StrikeName = string.("Strike = ",dfn.Strike)
    Td = :Td => (t -> t) => "Dividend ex-date in years"
    VolError = :VolError => (t -> t) => "Absolute Error in Volatility %"
    set_theme!(theme_light())
    plt = data(dfn) * mapping(Td, VolError) * mapping(color= :Model) * mapping(col= :StrikeName)  * visual(Lines)
    fig = draw(plt, legend=(position=:bottom,titleposition=:left), axis=(yscale=log10,limits=(nothing,(1e-8,1e-2)),width=250,height=250,))
    save("/home/niamh/mypapers/eqd_book/single_r6_strike100_td.pdf", fig)
    =#

# test/pln_european_test.jl, @testset "ZhangExtreme"
        #=
    p = plot(xlab="Strike",ylab="Absolute error in price")
    for name=["VL3","VL2","LL3","EG3","Deelstra"]
       plot!(p, strikes, abs.(priceMap[name]-refPrices), label=name, linestyle=:auto)
       end
    plot!(p, yscale=:log10)


 f = Figure(size=(700,350));
 ax = Axis(f[1,1],xlabel="Strike", ylabel="Absolute Error in Price", yscale=log10)
for name = ["Deelstra", "GS",  "LL2", "LL3", "VL2","VL3"]
                lines!(ax, strikes, abs.(priceMap[name]-refPrices), label=name)
              end
              legend = Legend(f[2,1], ax, "Model", position= :cb, titleposition= :left, orientation= :horizontal)
save("/home/niamh/mypapers/eqd_book/zhang_extreme.pdf", fig)

 #axislegend(; position = :cb, nbanks = 3)
save("/home/niamh/mypapers/eqd_book/gocsei_10y.pdf", f)
 
    =#

# test/pln_european_test.jl, @testset "MultipleGocsei"
    #=
    p = plot(xlab="Strike",ylab="Absolute error in volatility %")
    for name=["VL3","VL2","LL3","GS","EG3","Deelstra"]
           plot!(p, strikes, abs.(volMap[name]-refVols).*100, label=name, linestyle=:auto)
           end
     plot!(p, yscale=:log10)
      plot!(p,yticks=([1e-4,1e-3,1e-2,1e-1]))

 f = Figure(size=(700,350));
 ax = Axis(f[1,1],xlabel="Strike", ylabel="Error in Implied Volatility %", yscale=log10)
for pricerName = ["Deelstra", "GS",  "LL2", "LL3", "VL2","VL3"]
                lines!(ax, strikes, abs.(volMap[pricerName] - refVols).*100, label=pricerName)
              end
              legend = Legend(f[2,1], ax, "Model", position= :cb, titleposition= :left, orientation= :horizontal)

 #axislegend(; position = :cb, nbanks = 3)
save("/home/niamh/mypapers/eqd_book/gocsei_10y.pdf", f)
 plt = data(dfn) * mapping(:Strike, VolErrorPct)  * mapping(color=:Model) *  visual(Lines)
fig = draw(plt, axis=(yscale=log10,width=700,height=350,))

    =#
# ====================================================================================================
# test/rough_test.jl
# ====================================================================================================

# test/rough_test.jl, @testset "rHeston-Pade"
#=
plot(kk, (pricesb1k-prices1k), label="", xlab="Log-moneyness") #, ylab="Error in OTM option prices" )
plot!(size=(400,200))
savefig("~/mypapers/eqd_book/rheston_chebyshev_adams1k_price.pdf")
plot(kk, (volsb1k-vols1k).*100, label="", xlab="Log-moneyness") #, ylab="Error in implied volatility %" )
plot!(size=(400,200))
savefig("~/mypapers/eqd_book/rheston_chebyshev_adams1k_vol.pdf")
=#

# test/rough_test.jl, @testset "rHeston-Pade"
#=  data1 = data[data.T .== 1.0,:]
 data1 = data[data.T .== 1.0 .&& (data.Method .== "Pade" .|| data.Method .== "Adams-1024"),:]
plot([plot(gdata.Logmoneyness, gdata.Vol .* 100, group=gdata.Method, xlab="Log-moneyness", ylab=if gdata.H[1]==0.01 "Implied volatility in %" else "" end, legend = gdata.H[1]==0.45, title=string("H=",gdata.H[1]), ylim=(10.0,25.0)) for gdata in groupby(data1,:H)]..., layout=(1,4))
plot([plot(gdata.Logmoneyness, gdata.Vol .* 100, group=gdata.Method, xlab="Log-moneyness", ylab=if gdata.H[1]==0.01 "Implied volatility in %" else "" end, legend = gdata.H[1]==0.45, title=string("H=",gdata.H[1]), ylim=(7.5,35.0),xlim=(-0.75,0.75)) for gdata in groupby(data1,:H)]..., layout=(1,4))
plot!(size=(1024,256))

julia> savefig("~//mypapers/eqd_book/rheston_h_pade.pdf")
"~//mypapers/eqd_book/rheston_h_pade.pdf"


plot()

=#

# test/rough_test.jl, @testset "Rough-Calibration"
#= 
 kFine = forward.*exp.(range(-1*sqrt(tte),stop=1*sqrt(tte), length=201));
pricesFine = map(x -> priceEuropean(pricer, x >= forward, x, forward,tte,1.0),kFine)
pricesAFine = map(x -> priceEuropean(pricerA, x >= forward, x, forward,tte,1.0),kFine)
pricesCFine = map(x -> priceEuropean(pricerC, x >= forward, x, forward,tte,1.0),kFine)
pricesFFine = map(x -> priceEuropean(pricerF, x >= forward, x, forward,tte,1.0),kFine)
pricesHFine = map(x -> priceEuropean(pricerFullHeston, x >= forward, x, forward,tte,1.0),kFine)

 p3=plot(log.(strikeA./forward), volA.*100, seriestype= :scatter, label="Reference", markersize=4, markerstrokewidth=0,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(kFine .>= forward, max.(1e-16,pricesFine), forward, (kFine), tte, 1.0) .* 100, label="rHeston Pade")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(kFine .>= forward, max.(1e-16,pricesAFine), forward, (kFine), tte, 1.0) .* 100, label="rHeston Adams")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(kFine .>= forward, max.(1e-16,pricesHFine), forward, (kFine), tte, 1.0) .* 100, label="Heston")
#plot!(p3, log.(kFine./forward), sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi1, log.(kFine./forward))).*100, label="SVI")
plot!(p3, log.(kFine./forward), sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(sectionSPX, log.(kFine./forward))).*100, label="SABR")
plot!(xlims=(-1*sqrt(tte),1*sqrt(tte)),ylims=(5,30))

## fit is not great but only mid data, maybe ok with bid ask ? should really have bid + ask data as well!
SABR is better for short term, even when using 4 params in Rough heston.- still interesting to know.
=#

# test/rough_test.jl, @testset "Rough-SPX500-Oct2010"
#=if we linearly interpolate v(t)*t = V(t2)*t2 - v(t1)*t1. maybe flat foward interpolation => xi(ti) = varswap(ti)?    v(t1)*t1 , v(t2)*t2  
plot(t,xiLinear.(t), label="Linear")
plot!(t,xi.(t), label="Spline", xlab="Time", ylab="Forward variance")
plot!(t,  @.(theta + exp(-kappa*t)*(v0-theta)),label="Heston varswap" )
plot!(t,  @.(paramsF.θ + exp(-paramsF.κ*t)*(paramsF.v0-paramsF.θ)),label="Heston full" )
 plot!(size=(400,300),legend=:bottomright)
savefig("~/mypapers/eqd_book/rheston_forwardvarswap_spx2010.pdf")
plot!(ttes[2:end],(varswapPriceA[2:end] .* ttes[2:end] - varswapPriceA[1:end-1] .* ttes[1:end-1]) ./( ttes[2:end] - ttes[1:end-1]))

quad = AQFED.Math.Simpson(length(t)*16)
plot(ttes, varswapPriceA, seriestype=:scatter, label="Synthetic swap",xlab="Time to maturity", ylab="Variance")
plot!(t, @.(AQFED.Math.integrate(quad,xi,0.0,t))./t,label="Spline")
plot!(t, @.(theta + (1-exp(-kappa*t))/(kappa*t)*(v0-theta)),label="Heston varswap" )
 plot!(t,  @.(paramsF.θ + (1-exp(-paramsF.κ * t))/(paramsF.κ * t)*(paramsF.v0 - paramsF.θ)),label="Heston full" )
 plot!(size=(400,300),legend=:bottomright)
savefig("~/mypapers/eqd_book/rheston_varswap_spx2010.pdf")

=#

# test/rough_test.jl, @testset "Rough-SPX500-Oct2010"
#=
kFine = range(strikeA[1],stop=strikeA[end],length=201)
priceFineM = [priceEuropean(pricerA[i],strike>=forward, strike, forward, tte, 1.0) for (i,tte) = enumerate(ttes), (j,strike) = enumerate(kFine)]
priceHestonFineM = [priceEuropean(pricerHeston,strike>=forward, strike, forward, tte, 1.0) for (i,tte) = enumerate(ttes), (j,strike) = enumerate(kFine)]
priceFullHestonFineM = [priceEuropean(pricerFullHeston,strike>=forward, strike, forward, tte, 1.0) for (i,tte) = enumerate(ttes), (j,strike) = enumerate(kFine)]

ps = [plot(strikeA ./ forwardA[i], volM[i,:] .* 100, seriestype=:scatter,label="Market", title=string("T=",tte), legend=(i==length(ttes)) ) for (i,tte) = enumerate(ttes)]
for (i,tte) = enumerate(ttes)
    plot!(ps[i], kFine ./ forwardA[i], Black.impliedVolatility.(kFine .>= forward, max.(1e-16,priceFineM[i,:]), forward, kFine, tte, 1.0) .* 100, label="Rough Heston" )
end
for (i,tte) = enumerate(ttes)
    plot!(ps[i], kFine ./ forwardA[i], Black.impliedVolatility.(kFine .>= forward, max.(1e-20,priceHestonFineM[i,:]), forward, kFine, tte, 1.0) .* 100, label="Heston varswap")
end
for (i,tte) = enumerate(ttes)
    plot!(ps[i], kFine ./ forwardA[i], Black.impliedVolatility.(kFine .>= forward, max.(1e-20,priceFullHestonFineM[i,:]), forward, kFine, tte, 1.0) .* 100, label="Heston full" )
end

plot(ps...)
plot!(size=(1024,768))
savefig("~/mypapers/eqd_book/rheston_vol_spx2010.pdf")
linear : RoughHestonParams{Float64}(0.18731076406802621, -0.35050883317323744, 0.2451454051698534, var"#641#642"())
=#

# test/rough_test.jl, @testset "rough_mc"
    #=
    plot(steps, abs.(pricesHQE ./ refPriceH .- 1), label="rHeston HQE",xscale=:log10)
 plot!(steps, abs.(pricesRH ./ refPriceR .- 1), label="rBergomi hybrid",xscale=:log10)
 plot!(steps, abs.(pricesR ./ refPriceR .- 1), label="rBergomi standard",xscale=:log10)
 plot!(steps, abs.(pricesHeston ./ refPriceHeston .- 1), label="Heston DVSS2X",xscale=:log10)
plot!(yscale=:log10,xticks=(steps,steps),xlab="Time-steps",ylab="Relative error in price")
plot!(size=(800,400),margins=3Plots.mm)
 savefig("~/mypapers/eqd_book/rheston_mc_convergence.pdf")

    =#
# ====================================================================================================
# test/sabr_test.jl
# ====================================================================================================

# test/sabr_test.jl, @testset "flat-basket"
#=
x=range(0.5,stop=2.0,length=201)
 isoc,m = Collocation.makeIsotonicCollocation(strikes,prices,weights,tte,f,discountFactor,deg=3,degGuess=1)
 sol = Collocation.Polynomial(isoc)
 cpricesx = Collocation.priceEuropean.(sol, true, x, f, discountFactor)
plot(x, @.(AQFED.Black.impliedVolatility(true,Bachelier.bachelierFormula(true, x, f, sqrt(AQFED.TermStructure.normalVarianceByMoneyness(section,x-f)),tte,discountFactor),f,x,tte,discountFactor)).*100 .- 20,label="Normal SABR")
plot!(xlab="Strike",ylab="Error in Black volatility %")
plot!(x,AQFED.Black.impliedVolatility.(true,cpricesx,f,x,tte,discountFactor).*100 .-20,label="Cubic collocation")
 plot!(ylims=(-0.02,0.04))
  plot!(size=(400,300))
#savefig("black_normal_sabr_collo.pdf")
=#

# test/sabr_test.jl, @testset "flat-basket"
#=
plot(rhos, abs.(sabrPrices./refPrices.-1), yscale=:log10, xlab="Correlation", ylab="Relative error")
=#

# test/sabr_test.jl, @testset "basket-smile"
    #=
    ## fit of sabr
    x = range(strikes[1],stop=strikes[end],length=201)
     fitPricesx = [Bachelier.bachelierFormula(true,strikei,f,sqrt(AQFED.TermStructure.normalVarianceByMoneyness(sectionSPX, strikei .- f)),tte,1.0) for (i,strikei)=enumerate(x)]
 plot(strikes./f,vols.*100,seriestype=:scatter,xlab="Strike", ylab="Volatility in %",ms=4,markerstrokewidth=0,markeralpha=0.7,label="Market")
 plot!(x./f,abs.(AQFED.Black.impliedVolatility.(true, fitPricesx, f, x, tte, 1.0)).*100,xlab="Strike/Forward",label="Normal SABR")
 plot!(size=(400,300),margins=1Plots.mm)
    x = range(strikes2[1],stop=strikes2[end],length=201)
     fitPricesx = [Bachelier.bachelierFormula(true,strikei,f2,sqrt(AQFED.TermStructure.normalVarianceByMoneyness(sectionTSLA, strikei .- f2)),tte,1.0) for (i,strikei)=enumerate(x)]
 plot(strikes2./f2,vols2.*100,seriestype=:scatter,xlab="Strike", ylab="Volatility in %",ms=4,markerstrokewidth=0,markeralpha=0.7,label="Market")
 plot!(x./f2,abs.(AQFED.Black.impliedVolatility.(true, fitPricesx, f2, x, tte, 1.0)).*100,xlab="Strike/Forward",label="Normal SABR")
 plot!(size=(400,300),margins=1Plots.mm)
 savefig("normal_sabr_tsla_1m.pdf")

## relative error in OTM prices
plot(bstrikes,abs.(priceSABR./priceBlack .- 1).*100,yscale=:log10,yticks=([0.1,1,10,100],["0.1%","1%","10%","100%"]),label="SABR / Black")
plot!(bstrikes,abs.(priceSABR./priceBlackATM .- 1).*100,label="SABR / Black ATM",xlab="Basket strike", ylab="Relative difference")
plot!(bstrikes,abs.(priceSABRBlack./priceBlackATM .- 1).*100,label="SABR Flat ATM / Black ATM")
 plot!(size=(800,300),margins=3Plots.mm)
savefig("normal_sabr_basket_1m.pdf")

=#
# ====================================================================================================
# test/varswap_test.jl
# ====================================================================================================

# test/varswap_test.jl, @testset "SVI"
#=  plot(strikes,volatility,seriestype=:scatter, label="Market",xlab="Strike",ylab="Volatility in %",ms=4,markerstrokewidth=0,markeralpha=0.8)
  plot!(k,@.(100*Black.impliedVolatility(true, PDDE.priceEuropean(lvgqe, true, k), forward, k, tte, 1.0)),label="LVG",linestyle=:auto)
 plot!(k,@.(sqrt(AQFED.TermStructure.varianceByLogmoneyness(svi,log(k/forward)))*100),label="SVI",linestyle=:auto)
 plot!(k,@.(sqrt(AQFED.TermStructure.varianceByLogmoneyness(svi0,log(k/forward)))*100),label="SVI a=0",linestyle=:auto)
 plot!(size=(800,400),margins=3Plots.mm)
savefig("~//mypapers/eqd_book/varswap_bad_vol.pdf")
 =#
# ====================================================================================================
# test/vol_calibration_tests.jl
# ====================================================================================================

# test/vol_calibration_tests.jl, @testset "deltavolraw"
#=
p3=plot(log.(k./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=1.0); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(p3, log.(kFine./forward), smileSpl.(kFine) .* 100,label="Spline on log-moneyness")
plot!(p3, log.(kFine./forward), smileQ.(kFine) .* 100,label="Quartic on Δ")

p4=plot(log.(kFine./forward),map(x->pdf(smileSpl,x),kFine),label="Spline on log-moneyness");xlabel!(p4,"Forward log-moneyness"); ylabel!(p4,"Probability density")
plot!(p4,log.(kFine./forward),map(x->pdf(smileQ,x),kFine),label="Quartic on Δ")

f1 = Figure(size=(660,330))
ax = Axis(f1[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
s1=Makie.scatter!(ax, log.(k./forward), vols.*100,label="Market", marker=:cross, color=:black)
l1=lines!(ax, log.(kFine./forward), smileSplD.(kFine) .* 100,label="Spline on Δ")
l2=lines!(ax, log.(kFine./forward), smileSpl.(kFine) .* 100,label="Spline on log-moneyness")
l3=lines!(ax, log.(kFine./forward), smileSpl0.(kFine) .* 100,label="Spline on log-moneyness with flat extrapolation")
#axislegend(position = :rb)

ax = Axis(f1[1, 2], xlabel="Forward log-moneyness", ylabel="Probability Density")
lines!(ax, log.(kFine./forward), map(x->pdfd(smileSplD,x),kFine),label="Spline on Δ")
lines!(ax, log.(kFine./forward), map(x->pdfd(smileSpl,x),kFine),label="Spline on log-moneyness")
lines!(ax, log.(kFine./forward), map(x->pdfd(smileSpl0,x),kFine),label="Spline on log-moneyness flat")
#axislegend(position = :rt)
Legend(f1[2,1:2], [s1, l1, l2,l3],["Market","Spline on Δ", "Spline on log-moneyness", "Spline on log-moneyness flat"],orientation=:horizontal)
save("/home/fly/mypapers/fxo_arb_free/audnzd_spline_dens.pdf", f1)

yFine = log.(kFine./forward)
smileSplW = AQFED.VolatilityModels.makeTotalVarianceFunction(smileSpl,forward,tte) 
smileSVIW = AQFED.VolatilityModels.makeTotalVarianceFunction(smileSVI,forward,tte) 
smileSVINW = AQFED.VolatilityModels.makeTotalVarianceFunction(smileSVIN,forward,tte) 
f3 = Figure(size=(660,330))
ax = Axis(f3[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
s1=Makie.scatter!(ax, log.(k./forward), vols.*100,label="Market",marker=:cross, color=:black)
l1=lines!(ax, log.(kFine./forward), smileSVI.(kFine) .* 100,label="SVI a ≥ 0")
l2=lines!(ax, log.(kFine./forward), smileSVIN.(kFine) .* 100,label="SVI unconstrained")

ax = Axis(f3[1, 2], xlabel="Forward log-moneyness", ylabel="Local variance denominator")
lines!(ax, yFine, map(x->AQFED.TermStructure.localVarianceDenominator(smileSVIW,x),yFine),label="SVI a ≥ 0")
lines!(ax, yFine, map(x->AQFED.TermStructure.localVarianceDenominator(smileSVINW,x),yFine),label="SVI unconstrained")
Legend(f3[2,1:2], [s1, l1, l2],["Market", "SVI a ≥ 0", "SVI unconstrained"],orientation=:horizontal)
f3
CairoMakie.activate!()
CairoMakie.save("/home/fly/mypapers/fxo_arb_free/audnzd_svi_denom.pdf", f3)



=#

# test/vol_calibration_tests.jl, @testset "deltavolraw"
#=

f1 = Figure(size=(660,330))
ax = Axis(f1[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
s1=Makie.scatter!(ax, log.(k./forward), vols.*100,label="Market",color=:black, marker=:cross)
l1=lines!(ax, log.(kFine./forward), smileQ.(kFine) .* 100,label="Polynomial on forward Δ")
l2=lines!(ax, log.(kFine./forward), smileQ1.(kFine) .* 100,label="Polynomial on simple Δ")
l3=lines!(ax, log.(kFine./forward), smileSVI.(kFine) .* 100,label="SVI")
l4=lines!(ax, log.(kFine./forward), xssvi.(kFine) .* 100,label="xSSVI")

ax = Axis(f1[1, 2], xlabel="Forward log-moneyness", ylabel="Probability Density")
lines!(ax, log.(kFine./forward), map(x->pdfd(smileQ,x),kFine),label="Spline on log-moneyness")
lines!(ax, log.(kFine./forward), map(x->pdfd(smileQ1,x),kFine),label="Spline on Δ")
lines!(ax, log.(kFine./forward), map(x->pdfd(smileSVI,x),kFine),label="SVI")
lines!(ax, log.(kFine./forward), map(x->pdfd(xssvi,x),kFine),label="xSSVI")

Legend(f1[2,1:2], [s1, l1, l2, l3,l4],["Market", "Polynomial on forward Δ", "Polynomial on simple Δ", "SVI","xSSVI"],orientation=:horizontal)
save("/home/fly/mypapers/fxo_arb_free/eurczk_poly_dens.pdf", f1)


	p4=plot(log.(kFine./forward),map(x->AQFED.VolatilityModels.density.(kernel.kernel,x),kFine),label="Mixture of 2")
plot!(p4,log.(kFine./forward),map(x->AQFED.VolatilityModels.density.(kernel3.kernel,x),kFine),label="Mixture of 3")
=#

# test/vol_calibration_tests.jl, @testset "deltavolraw"
#=
dFine = range(1e-4,0.9999,length=200)
plot(dFine, @.(exp(smileQ.coeff[1]+smileQ.coeff[2]*dFine + smileQ.coeff[3]*dFine^2+smileQ.coeff[4]*dFine^3 + smileQ.coeff[5]*dFine^4)))
#f(N(log(forward/strike)/sigmat))= (sigma)
dVols =  @.(normcdf(log(forward/strike)/(exp(smileQ.coeff[1]+smileQ.coeff[2]*dFine + smileQ.coeff[3]*dFine^2+smileQ.coeff[4]*dFine^3 + smileQ.coeff[5]*dFine^4)*tte)))

deltasSimple = @.(normcdf((log(forward / k)) / (vols * sqrt(tte))))
f1 = Figure(size=(800,250))
ax = Axis(f1[1, 1], xlabel="Reduced Call Δ", ylabel="Volatility in %")
s1=Makie.scatter!(ax, deltasSimple, vols.*100,label="Market")
l1=lines!(ax, dFine, map(x-> AQFED.VolatilityModels.impliedVolatilityByDelta(smileQ,x)*100,dFine),label="Polynomial on forward Δ")

out = zeros(smileQ.nIter)
smileQ(3.7, method=:fixed, out = out)
ax = Axis(f1[1, 2], xlabel="Iteration", ylabel="Volatility in %")
lines!(ax, 1:length(out), out)
ax=  Axis(f1[1, 3], xlabel="Strike", ylabel="Volatility in %")
lines!(ax, kFine, smileQ.(kFine, method=:fixed,out=out) .* 100, label="Fixed-point")
lines!(ax, kFine, smileQ.(kFine) .* 100, label="Newton", linestyle=:dash)
axislegend(position = :ct)
save("/home/fly/mypapers/fxo_arb_free/usdaed_poly_conv.pdf", f1)

lines!(ax, log.(kFine./forward), map(x->pdf(smileQ1,x),kFine),label="Spline on Δ")
#
dev3 = 3 * vols[3] * sqrt(tte)
	kFine3 = exp.(collect(range(log(k[1]) - dev3, stop = log(k[end])+ dev3, length = 201)))
	yFine3 = @. log(kFine3/forward)
f1 = Figure(size=(660,330))
ax = Axis(f1[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
s1=Makie.scatter!(ax, log.(k ./ forward), vols.*100,label="Market",marker=:cross,color=:black)
l1=lines!(ax, yFine3, map(x-> smileSVI(x)*100,kFine3),label="SVI a >= 0")
l2=lines!(ax, yFine3, map(x-> xssvi(x)*100,kFine3),label="xSSVI")
l3=lines!(ax, yFine3, map(x-> smileQ(x)*100,kFine3),label="Polynomial on forward Δ")
l4=lines!(ax, yFine3, map(x-> smileSplD(x)*100,kFine3),label="Spline on Δ")
ax = Axis(f1[1, 2], xlabel="Forward log-moneyness", ylabel="Local variance denominator")
l1=lines!(ax, yFine3, map(x-> AQFED.TermStructure.localVarianceDenominator(smileSVIW,x),yFine3),label="SVI a >= 0")
l2=lines!(ax, yFine3, map(x-> AQFED.TermStructure.localVarianceDenominator(xssviW,x),yFine3),label="xSSVI")
l3=lines!(ax, yFine3, map(x-> AQFED.TermStructure.localVarianceDenominator(smileQW,x),yFine3),label="Polynomial on forward Δ")
l4=lines!(ax, yFine3, map(x-> AQFED.TermStructure.localVarianceDenominator(smileSplDW,x),yFine3),label="Spline on Δ")
Legend(f1[2,1:2], [s1, l1, l2, l3, l4],["Market", "SVI a ≥ 0", "xSSVI", "Polynomial Δ", "Spline Δ"],orientation=:horizontal)
save("/home/fly/mypapers/fxo_arb_free/usdaed_svi_dens.pdf", f1)

ax = Axis(f1[1, 2], xlabel="Forward log-moneyness", ylabel="Probability density")
l1=lines!(ax, yFine3, map(x-> pdfd(smileSVI,x),kFine3),label="SVI a >= 0",linestyle=:auto)
l2=lines!(ax, yFine3, map(x-> pdfd(xssvi,x),kFine3),label="xSSVI",linestyle=:auto)
l3=lines!(ax, yFine3, map(x-> pdfd(smileQ,x),kFine3),label="Polynomial on forward Δ",linestyle=:auto)
l4=lines!(ax, yFine3, map(x-> pdfd(lvg,x),kFine3),label="LVG",linestyle=:auto)

save("/home/fly/mypapers/fxo_arb_free/eurczk_poly_dens.pdf", f1)

=#

# test/vol_calibration_tests.jl, @testset "deltavolraw"
#=
dev3 = 3 * vols[3] * sqrt(tte)
kFine3 = exp.(collect(range(log(k[1]) - dev3, stop = log(k[end])+ dev3, length = 201)))
yFine3 = @. log(kFine3/forward)
f1 = Figure(size=(660,360))
ax = Axis(f1[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
s1=Makie.scatter!(ax, log.(k ./ forward), vols.*100,label="Market",marker=:cross,color=:black)
l1=lines!(ax, yFine3, map(x-> smileSVI(x)*100,kFine3),label="SVI a ≥ 0")
l2=lines!(ax, yFine3, map(x-> xssvi(x)*100,kFine3),label="xSSVI")
l3=lines!(ax, yFine3, map(x-> smileSpl(x)*100,kFine3))
l4=lines!(ax, yFine3, map(x-> smileQ(x)*100,kFine3))
l5=lines!(ax, yFine3, map(x-> quintic(x)*100,kFine3))
ax = Axis(f1[1, 2], xlabel="Forward log-moneyness", ylabel="Probability density")
Legend(f1[2,1:2], [s1, l1, l2, l3, l4,l5],
["Market", "SVI a ≥ 0", "xSSVI", "Spline on log-moneyness", "Polynomial Δ", "Quintic collocation"],orientation=:horizontal, nbanks=2)
l1=lines!(ax, yFine3, map(x-> pdfd(smileSVI,x),kFine3),label="SVI a >= 0")
l2=lines!(ax, yFine3, map(x-> pdfd(xssvi,x),kFine3),label="xSSVI")
l3=lines!(ax, yFine3, map(x-> pdfd(smileSpl,x),kFine3),label="Spline on log-moneyness")
l4=lines!(ax, yFine3, map(x-> pdfd(smileQ,x),kFine3))
l5=lines!(ax, yFine3, map(x-> pdfd(quintic,x),kFine3))
save("/home/fly/mypapers/fxo_arb_free/eurtry_svi_dens.pdf", f1)

l1=lines!(ax, yFine3, map(x-> AQFED.TermStructure.localVarianceDenominator(smileSVIW,x),yFine3),label="SVI a >= 0")
l2=lines!(ax, yFine3, map(x-> AQFED.TermStructure.localVarianceDenominator(xssviW,x),yFine3),label="xSSVI",linestyle=:dot)
l3=lines!(ax, yFine3, map(x-> AQFED.TermStructure.localVarianceDenominator(smileQW,x),yFine3),label="Polynomial on forward Δ",linestyle=:dash)
l4=lines!(ax, yFine3, map(x-> AQFED.TermStructure.localVarianceDenominator(smileSplDW,x),yFine3),label="Spline on Δ",linestyle=:dashdot)

save("/home/fly/mypapers/fxo_arb_free/eurczk_poly_dens.pdf", f1)

=#

# test/vol_calibration_tests.jl, @testset "deltavolraw"
#=
dev3 = 1.25 * vols[3] * sqrt(tte)
kFine3 = exp.(collect(range(log(k[1]) - dev3, stop = log(k[end])+ dev3, length = 201)))
yFine3 = @. log(kFine3/forward)
f1 = Figure(size=(660,330))
ax = Axis(f1[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
s1=Makie.scatter!(ax, log.(k ./ forward), vols.*100,label="Market",marker=:cross,color=:black)
l1=lines!(ax, yFine3, map(x-> smileSVI(x)*100,kFine3),label="SVI a ≥ 0")
l2=lines!(ax, yFine3, map(x-> xssvi(x)*100,kFine3),label="xSSVI")
l3=lines!(ax, yFine3, map(x-> sabr(x)*100,kFine3))
l4=lines!(ax, yFine3, map(x-> smileQ(x)*100,kFine3))
l5=lines!(ax, yFine3, map(x-> lvg(x)*100,kFine3))
ax = Axis(f1[1, 2], xlabel="Forward log-moneyness", ylabel="Probability density")
Legend(f1[2,1:2], [s1, l1, l2, l3, l4,l5],
["Market", "SVI a ≥ 0", "xSSVI", "SABR", "Polynomial Δ", "LVG"],orientation=:horizontal)
l1=lines!(ax, yFine3, map(x-> pdfd(smileSVI,x),kFine3),label="SVI a ≥ 0")
l2=lines!(ax, yFine3, map(x-> pdfd(xssvi,x),kFine3),label="xSSVI")
l3=lines!(ax, yFine3, map(x-> pdfd(sabr,x),kFine3),label="Spline on log-moneyness")
l4=lines!(ax, yFine3, map(x-> pdfd(smileQ,x),kFine3))
l5=lines!(ax, yFine3, map(x-> pdfd(lvg,x),kFine3))
save("/home/fly/mypapers/fxo_arb_free/man_svi_dens.pdf", f1)

=#

# test/vol_calibration_tests.jl, @testset "deltavolraw"
#=
dev3 = 2 * vols[3] * sqrt(tte)
kFine3 = exp.(collect(range(log(k[1]) - dev3, stop = log(k[end])+ dev3, length = 201)))
yFine3 = @. log(kFine3/forward)
f1 = Figure(size=(600,300))
ax = Axis(f1[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
s0 =Makie.scatter!(ax, log.(kFull ./ forward), fullVols.*100,label="Market (all)",marker=:cross,color=:grey)
s1=Makie.scatter!(ax, log.(k ./ forward), vols.*100,label="Market (5 points)",marker=:cross,color=:black)
l1=lines!(ax, yFine3, map(x-> smileSVI(x)*100,kFine3),label="SVI a ≥ 0")
l2=lines!(ax, yFine3, map(x-> xssvi(x)*100,kFine3),label="xSSVI")
l3=lines!(ax, yFine3, map(x-> sabr(x)*100,kFine3),label="SABR")
l4=lines!(ax, yFine3, map(x-> smileQ(x)*100,kFine3),label="Polynomial Δ")
l5=lines!(ax, yFine3, map(x-> lvg(x)*100,kFine3), label="LVG")
#axislegend(position = :ct)
Legend(f1[1,2], [s1, l1, l2, l3, l4,l5],["Market", "SVI a ≥ 0", "xSSVI", "SABR", "Polynomial Δ", "LVG"],orientation=:vertical)
save("/home/fly/mypapers/fxo_arb_free/eurusd_extra.pdf", f1)

ax = Axis(f1[1, 2], xlabel="Forward log-moneyness", ylabel="Probability density")
l1=lines!(ax, yFine3, map(x-> pdfd(smileSVI,x),kFine3),label="SVI a ≥ 0")
l2=lines!(ax, yFine3, map(x-> pdfd(xssvi,x),kFine3),label="xSSVI",linestyle=:dot)
l3=lines!(ax, yFine3, map(x-> pdfd(sabr,x),kFine3),label="Spline on log-moneyness",linestyle=:dash)
l4=lines!(ax, yFine3, map(x-> pdfd(smileQ,x),kFine3))
l5=lines!(ax, yFine3, map(x-> pdfd(lvg,x),kFine3),linestyle=:dashdot)

=#

# test/vol_calibration_tests.jl, @testset "deltavolraw"
#=
dev3 = 2 * vols[3] * sqrt(tte)
kFine3 = exp.(collect(range(log(k[1]) - dev3, stop = log(k[end])+ dev3, length = 201)))
yFine3 = @. log(kFine3/forward)
f1 = Figure(size=(600,300))
ax = Axis(f1[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
s0 =Makie.scatter!(ax, log.(kFull ./ forward), fullVols.*100,label="Market (all)",marker=:cross,color=:grey)
s1=Makie.scatter!(ax, log.(k ./ forward), vols.*100,label="Market (5 points)",marker=:cross,color=:black)
l1=lines!(ax, yFine3, map(x-> smileSVI(x)*100,kFine3),label="SVI a ≥ 0")
l2=lines!(ax, yFine3, map(x-> xssvi(x)*100,kFine3),label="xSSVI")
l3=lines!(ax, yFine3, map(x-> sabr(x)*100,kFine3),label="SABR")
l4=lines!(ax, yFine3, map(x-> smileQ(x)*100,kFine3),label="Polynomial Δ")
l5=lines!(ax, yFine3, map(x-> lvg(x)*100,kFine3), label="LVG")
#axislegend(position = :ct)
Legend(f1[1,2], [s1, l1, l2, l3, l4,l5],["Market", "SVI a ≥ 0", "xSSVI", "SABR", "Polynomial Δ", "LVG"],orientation=:vertical)
save("/home/fly/mypapers/fxo_arb_free/eurusd_extra_1y.pdf", f1)
=#

# test/vol_calibration_tests.jl, @testset "deltavolcalibrated"
 #=
    k=kRaw;
	dev3 = volAtm * sqrt(tte);
    kFine3 = forward .* exp.(collect(range(k[1] - dev3/2, stop = k[end]+ dev3/2, length = 1001)));
    yFine3 = @. log(kFine3/forward);
    f1 = Figure(size=(600,300))
    ax = Axis(f1[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
    s1=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineSmile] , volMap[AQFED.VolatilityModels.SplineSmile].*100,label="Implied quotes (Spline)",marker=:cross,color=:black)
	s2=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineDeltaSmile], volMap[AQFED.VolatilityModels.SplineDeltaSmile].*100,label="Implied quotes (SplineDelta)",marker=:cross,color=:grey70)
	s3=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.ExpPolynomialSmile], volMap[AQFED.VolatilityModels.ExpPolynomialSmile].*100,label="Implied quotes (ExpPolynomial)",marker=:cross,color=:grey50)
    #l1=lines!(ax, yFine3, map(x-> smileSVI(x)*100,kFine3),label="SVI a ≥ 0")
    #l2=lines!(ax, yFine3, map(x-> xssvi(x)*100,kFine3),label="xSSVI")
    l1=lines!(ax, yFine3, map(x-> smileSABRClark(x)*100,kFine3),label="SABR Clark")
    l2=lines!(ax, yFine3, map(x-> smileSABRATMClark(x)*100,kFine3),label="Dimension 2")
    l3=lines!(ax, yFine3, map(x-> smileMap[AQFED.VolatilityModels.SABRSmile](x)*100,kFine3))
    l4=lines!(ax, yFine3, map(x-> smileMap[AQFED.VolatilityModels.SABRATMSmile](x)*100,kFine3),label="Dimension 5")
    #axislegend(position = :ct)
    #Legend(f1[1,2], [s1, s2, s3, l1, l2, l3],["Quotes (Spline)", "Quotes (Spline Δ)", "Quotes (Polynomial Δ)", "SABR Clark", "SABR ATM Clark", "SABR Clark Extended"],orientation=:vertical)
    Legend(f1[1,2], [[s1, s2, s3], [l2,l4], [l1,l3]],[["Spline", "Spline Δ", "Polynomial Δ"],["Dimension 2", "Dimension 5"], ["Dimension 2", "Dimension 5"]], ["Implied Quotes", "SABR ATM Objective", "SABR Objective"],valign=:top)        
    save("/home/fly/mypapers/fxo_arb_free/eurhkd_sabr.pdf", f1, backend=CairoMakie)

	f2 = Figure(size=(600,300))
    ax = Axis(f2[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
     s1=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineSmile] , volMap[AQFED.VolatilityModels.SplineSmile].*100,label="Implied quotes (Spline)",marker=:cross,color=:black)
	s2=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineDeltaSmile] , volMap[AQFED.VolatilityModels.SplineDeltaSmile].*100,label="Implied quotes (SplineDelta)",marker=:cross,color=:grey70)
	s3=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.ExpPolynomialSmile] , volMap[AQFED.VolatilityModels.ExpPolynomialSmile].*100,label="Implied quotes (ExpPolynomial)",marker=:cross,color=:grey50)
    #l1=lines!(ax, yFine3, map(x-> smileSVI(x)*100,kFine3),label="SVI a ≥ 0")
    l2=lines!(ax, yFine3, map(x-> smileXSSVIClark(x)*100,kFine3),label="xSSVI")
    l3=lines!(ax, yFine3, map(x-> smileMap[AQFED.VolatilityModels.XSSVISmile](x)*100,kFine3),label="XSSVI Clark Extended")
    #axislegend(position = :ct)
    #Legend(f2[1,2], [s1, s2, s3, l2, l3],["Quotes (Spline)", "Quotes (Spline Δ)", "Quotes (Polynomial Δ)", "XSSVI Clark", "XSSVI Clark Extended"],orientation=:vertical)
    Legend(f2[1,2], [[s1, s2, s3], [l2,l3]],[["Spline", "Spline Δ", "Polynomial Δ"],["Dimension 2", "Dimension 5"]], ["Implied Quotes", "XSSVI Objective"])      
    save("/home/cian/mypapers/fxo_arb_free/eurhkd_xssvi.pdf", f2, backend=CairoMakie)
    save("/home/cian/mypapers/fxo_arb_free/eurhkd_xssvi.pdf", f2, backend=CairoMakie)
	
    f3 = Figure(size=(600,340))
    ax = Axis(f3[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")


    l1=lines!(ax, yFine3, map(x-> smileMap[AQFED.VolatilityModels.SplineSmile](x)*100,kFine3))
    l2=lines!(ax, yFine3, map(x-> smileMap[AQFED.VolatilityModels.SplineDeltaSmile](x)*100,kFine3))
    l3=lines!(ax, yFine3, map(x-> smileMap[AQFED.VolatilityModels.ExpPolynomialSmile](x)*100,kFine3))

    s1=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineSmile] , volMap[AQFED.VolatilityModels.SplineSmile].*100,marker=:circle,color=Makie.wong_colors()[1])
	s2=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineDeltaSmile], volMap[AQFED.VolatilityModels.SplineDeltaSmile].*100,label="Implied quotes (SplineDelta)",marker=:rect,color=Makie.wong_colors()[2])
	s3=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.ExpPolynomialSmile], volMap[AQFED.VolatilityModels.ExpPolynomialSmile].*100,label="Implied quotes (ExpPolynomial)",marker=:diamond,color=Makie.wong_colors()[3])
    s4=Makie.scatter!(ax, kRaw, volsRaw.*100,label="Smile quotes",marker=:cross,color=Makie.wong_colors()[4])

    #axislegend(position = :ct)
    Legend(f3[1,2], [[l1,s1], [l2,s2], [l3,s3], s4],["Spline", "Spline Δ", "Polynomial Δ","Smile convention"],orientation=:vertical)
    #Legend(f2[1,2], [[s1, s2, s3], [l1, l2,l3]],[["Spline", "Spline Δ", "Polynomial Δ"],[["Spline", "Spline Δ", "Polynomial Δ"], ["Implied Quotes", "XSSVI Objective"])     
 save("/home/cian/mypapers/fxo_arb_free/eurtry_2y_exact.pdf", f3, backend=CairoMakie)

    f4 =  Figure(size=(600,340))
    ax = Axis(f4[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
  s1=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineSmile], volMap[AQFED.VolatilityModels.SplineSmile].*100,label="Implied quotes (Spline)",marker=:cross,color=:black)
	s2=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineDeltaSmile] , volMap[AQFED.VolatilityModels.SplineDeltaSmile].*100,label="Implied quotes (SplineDelta)",marker=:cross,color=:grey70)
    l1 = lines!(ax, yFine3, map(x-> smileMap[AQFED.VolatilityModels.SABRATMSmile](x)*100,kFine3),label="SABR")
  l3=lines!(ax, yFine3, map(x-> smileMap[AQFED.VolatilityModels.XSSVISmile](x)*100,kFine3),label="XSSVI")

   f5 =  Figure(size=(600,340))
    ax = Axis(f5[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
  s1=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineSmile], volMap[AQFED.VolatilityModels.SplineSmile].*100,label="Implied quotes (Spline)",marker=:circle,color=:black)
	s2=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineDeltaSmile], volMap[AQFED.VolatilityModels.SplineDeltaSmile].*100,label="Implied quotes (SplineDelta)",marker=:rect,color=:black)
     l1 = lines!(ax, yFine3, map(x-> smileMap[AQFED.VolatilityModels.SABRATMSmile](x)*100,kFine3),label="SABR")
    l2 = lines!(ax, yFine3, map(x-> smileATMSABRVanillaStrike(x)*100,kFine3),label="SABR on Spline")
      l3 = lines!(ax, yFine3, map(x-> smileATMSABRVanillaDelta(x)*100,kFine3),label="SABR on Spline Δ")
Legend(f5[1,2], [[s1, s2, s3], [l1, l2,l3]],[["Spline", "Spline Δ"],["Market Quotes", "Vanilla Spline", "Vanilla Spline Δ"]], ["Implied Quotes", "SABR ATM Calibration"],orientation=:vertical)
 save("/home/cian/mypapers/fxo_arb_free/eurtry_2y_sabr.pdf", f5, backend=CairoMakie)

       f6 =  Figure(size=(600,340))
    ax = Axis(f6[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
  s1=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineSmile], volMap[AQFED.VolatilityModels.SplineSmile].*100,label="Implied quotes (Spline)",marker=:circle,color=:black)
	s2=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineDeltaSmile], volMap[AQFED.VolatilityModels.SplineDeltaSmile].*100,label="Implied quotes (SplineDelta)",marker=:rect,color=:black)
     l1 = lines!(ax, yFine3, map(x-> smileMap[AQFED.VolatilityModels.XSSVISmile](x)*100,kFine3),label="XSSVI")
    l2 = lines!(ax, yFine3, map(x-> smileXSSVIVanillaStrike(x)*100,kFine3),label="SABR on Spline")
      l3 = lines!(ax, yFine3, map(x-> smileXSSVIVanillaDelta(x)*100,kFine3),label="SABR on Spline Δ")     
Legend(f6[1,2], [[s1, s2, s3], [l1, l2,l3]],[["Spline", "Spline Δ"],["Market Quotes", "Vanilla Spline", "Vanilla Spline Δ"]], ["Implied Quotes", "XSSVI Calibration"],orientation=:vertical)
 save("/home/cian/mypapers/fxo_arb_free/eurtry_2y_xssvi.pdf", f6, backend=CairoMakie)


 	f7= Figure(size=(600,300))
    ax = Axis(f7[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
     s1=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineSmile] , volMap[AQFED.VolatilityModels.SplineSmile].*100,label="Implied quotes (Spline)",marker=:cross,color=:black)
	s2=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.SplineDeltaSmile] , volMap[AQFED.VolatilityModels.SplineDeltaSmile].*100,label="Implied quotes (SplineDelta)",marker=:cross,color=:grey70)
	s3=Makie.scatter!(ax, kMap[AQFED.VolatilityModels.ExpPolynomialSmile] , volMap[AQFED.VolatilityModels.ExpPolynomialSmile].*100,label="Implied quotes (ExpPolynomial)",marker=:cross,color=:grey50)
    l1=lines!(ax, yFine3, map(x-> smileMap[AQFED.VolatilityModels.QuinticCollocationSmile](x)*100,kFine3),label="Mixture")
    l2=lines!(ax, yFine3, map(x-> smileMap[AQFED.VolatilityModels.SABRATMSmile](x)*100,kFine3),label="SABRATM")
    l3=lines!(ax, yFine3, map(x-> smileMap[AQFED.VolatilityModels.XSSVISmile](x)*100,kFine3),label="XSSVI")
    #axislegend(position = :ct)
    Legend(f7[1,2], [[s1, s2, s3], [l1, l2,l3]],[["Spline", "Spline Δ", "Polynomial Δ"],["Quintic Collocation", "SABR ATM", "XSSVI"]], ["Implied Quotes", "Parameterization"])     
    save("/home/cian/mypapers/fxo_arb_free/eurhkd_xssvi.pdf", f2, backend=CairoMakie)
    save("/home/cian/mypapers/fxo_arb_free/eurhkd_xssvi.pdf", f2, backend=CairoMakie)
	

 
=#

# test/vol_calibration_tests.jl, @testset "deltavolcalibrated"
#=
plot(log.(kRaw./forward), volsRaw, seriestype= :scatter, label="Smile convention", m=:cross)
plot!(log.(kFine./forward), smileExactMap[AQFED.VolatilityModels.SABRSmile].(kFine),label="SABR Clark")
plot!(log.(kFine./forward), smileExactMap[AQFED.VolatilityModels.XSSVISmile].(kFine),label="XSSVI Clark")
plot!(log.(kFine./forward), smileExactMap[AQFED.VolatilityModels.SVISmile].(kFine),label="SVI Clark")
plot!(log.(kMapExact[AQFED.VolatilityModels.SplineSmile] ./forward), volMapExact[AQFED.VolatilityModels.SplineSmile], seriestype= :scatter, label="Broker Spline", m=:cross)
plot!(log.(kMapExact[AQFED.VolatilityModels.ExpPolynomialSmile] ./forward), volMapExact[AQFED.VolatilityModels.ExpPolynomialSmile], seriestype= :scatter, label="Broker Polynomial", m = :cross)
plot!(log.(kMapExact[AQFED.VolatilityModels.LVGSmile] ./forward), volMapExact[AQFED.VolatilityModels.LVGSmile], seriestype= :scatter, label="Broker LVG", m = :cross)


=#

# test/vol_calibration_tests.jl, @testset "deltavolcalibrated"
  #=
  f8= Figure(size=(600,300))
ax = Axis(f8[1, 1], xlabel="Forward log-moneyness", ylabel="Call forward Δ with premium",limits=(yRange[1],yRange[end],0.0,0.25))
    l1 = lines!(ax, yRange, deltaValues)
  
    f9= Figure(size=(600,300))
ax = Axis(f9[1, 1], xlabel="Forward log-moneyness", ylabel="Δsimple")
    l1 = lines!(ax, yRange, map(x->normcdf(-x/(2*smile(exp(x)*forward))),yRange))

    drange = range(0.1,0.9,length=101)
      f10= Figure(size=(600,300))
ax = Axis(f10[1, 1], xlabel="Δsimple", ylabel="Implied volatility %")
    l1 = lines!(ax, drange,map(x->AQFED.VolatilityModels.impliedVolatilityByDelta(smile,x)*100,drange)
 
    drange = range(0.0,1.0,length=101)
    f11= Figure(size=(600,300))
ax = Axis(f11[1, 1], xlabel="d", ylabel="Δsimple")
 l1 = lines!(ax, drange,map(x->normcdf(log(forward/5.0)/(sqrt(tte)*AQFED.VolatilityModels.impliedVolatilityByDelta(smile,x)))-x,drange))

 vrange = range(-5*volAtm,5*volAtm,length=101)
    f12  Figure(size=(600,300))
ax = Axis(f12[1, 1], xlabel="v", ylabel="f(v)")
 l1 = lines!(ax, vrange,map(x->AQFED.VolatilityModels.impliedVolatilityByDelta(smile,normcdf(log(forward/5.0)/(sqrt(tte)*x)))-x,vrange))

 =#

# test/vol_calibration_tests.jl, @testset "deltavolcalibrated"
# plot(deltaRange, map(x->AQFED.VolatilityModels.impliedVolatilityByDelta(ExpPolynomialSmile([0.11414222327265336, -11.854545452862252, 49.24883225877462, -84.12103007927499, 48.491914737536796], 0.32084, 39.5120054163372, 2.0, 256),x),deltaRange))

# test/vol_calibration_tests.jl, @testset "deltavolcalibrated"
#=
plot(log.(kFine./forward), pdf.(ppDeltaStrike,kFine),label="Cubic spline on Δ")
plot!(log.(kFine./forward),Collocation.density.(sol,kFine),label="Quintic collocation")
plot!(log.(kFine./forward),(AQFED.PDDE.derivativePrice.(lvg,true,kFine.+0.0001) .- AQFED.PDDE.derivativePrice.(lvg,true,kFine)).*10000, label="LVG")
plot!(log.(kFine./forward),(AQFED.PDDE.derivativePrice.(lvg,true,kFine.+0.0001) .- AQFED.PDDE.derivativePrice.(lvg,true,kFine)).*10000, label="LVG")
plot!(log.(kFine./forward),AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label="Fengler")

=#
