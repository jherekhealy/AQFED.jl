using AQFED, Test
using AQFED.Rough, AQFED.Black, CharFuncPricing
using PPInterpolation, DataFrames
import SpecialFunctions: gamma
using GaussNewton, AQFED.Math
@testset "rHeston-Pade" begin
    eta = 0.8
    H=0.05
    al = H+0.5
    τ = 1.0
    forward = 1.0
    nu = eta*sqrt(2H)*gamma(al)
    params0 = RoughHestonParams(H,-0.65,nu, s -> 0.025)
    cf = PadeCharFunc(params0)
    kk = -0.4:0.01:0.4 #log-strike
    strikes = exp.(kk) ./ forward
    pricer = CharFuncPricing.AdaptiveFilonCharFuncPricer(cf,τ,qTol=1e-8)
    prices = map(x -> priceEuropean(pricer, x >= forward, x, forward,τ,1.0),strikes)
    vols = [Black.impliedVolatility(strike >= forward,p,forward,strike,τ,1.0) for (strike,p) = zip(strikes,prices)]
    pricerCos = CharFuncPricing.makeCosCharFuncPricer(cf,τ,128*2,-6*sqrt(τ),6*sqrt(τ))
    prices = map(x -> priceEuropean(pricerCos,false, x, forward,τ,1.0),strikes)
    vols = [Black.impliedVolatility(false,p,forward,strike,τ,1.0) for (strike,p) = zip(strikes,prices)]

    cfa = AdamsCharFunc(params0,256)
    pricerCos = CharFuncPricing.makeCosCharFuncPricer(cfa,τ,256,-6*sqrt(τ),6*sqrt(τ))
    prices = map(x -> priceEuropean(pricerCos,false, x, forward,τ,1.0),strikes)
    vols = [Black.impliedVolatility(false,p,forward,strike,τ,1.0) for (strike,p) = zip(strikes,prices)]

    #adaptive threshold does not work, similarly too hgih N in cos breaks down with Adams method unless N is pushed very high.
    #truncation needs to be set as per ElEuch paper.
    #pricer = CharFuncPricing.AdaptiveFilonCharFuncPricer(cfa,τ,qTol=1e-8,myTrans=CharFuncPricing.IdentityTransformation(0.0,150.0))

    data = DataFrame(T=Float64[], H=Float64[], Logmoneyness=Float64[], Method=String[], Price=Float64[],Vol=Float64[])

    τs = [0.1, 1.0, 10.0]
    Hs = [0.01,0.05,0.25,0.45]
    upperBound1k = [120.0,150.0,120.0*3,150.0*4]
    upperBound = [60.0,70.0,40.0*3,70.0*2]
    for (Hindex, H) = enumerate(Hs)

    params0 = RoughHestonParams(H,-0.65,nu, s -> 0.025)
    cf = PadeCharFunc(params0)
    cfa = AdamsCharFunc(params0,256)
    cfa1k = AdamsCharFunc(params0,256*4)
    for τ in τs
        kk = (-0.5:0.01:0.5) .* sqrt(τ)
        strikes = exp.(kk) ./ forward
        pricer = CharFuncPricing.AdaptiveFilonCharFuncPricer(cf,τ,qTol=1e-8)
        prices = map(x -> priceEuropean(pricer, x >= forward, x, forward,τ,1.0),strikes)
        vols = [Black.impliedVolatility(strike >= forward,max(p,1e-32),forward,strike,τ,1.0) for (strike,p) = zip(strikes,prices)]
        for (k,pricek,volk) = zip(kk,prices,vols)
            push!(data,[τ, H, k, "Pade", pricek, volk])
        end
        pricerCos = CharFuncPricing.makeCosCharFuncPricer(cf,τ,128*4,-6*τ^H,6*τ^H)
        prices = map(x -> priceEuropean(pricerCos,x >= forward, x, forward,τ,1.0),strikes)
        vols = [Black.impliedVolatility(strike >= forward,max(p,1e-32),forward,strike,τ,1.0) for (strike,p) = zip(strikes,prices)]
        for (k,pricek,volk) = zip(kk,prices,vols)
            push!(data,[τ, H, k, "Pade-Cos", pricek, volk])
        end

        pricer = CharFuncPricing.AdaptiveFilonCharFuncPricer(cfa,τ,qTol=1e-8,myTrans=CharFuncPricing.IdentityTransformation(0.0,upperBound[Hindex]/sqrt(τ)))
        prices = map(x -> priceEuropean(pricer, x >= forward, x, forward,τ,1.0),strikes)
        vols = [Black.impliedVolatility(strike >= forward,max(p,1e-32),forward,strike,τ,1.0) for (strike,p) = zip(strikes,prices)]
        for (k,pricek,volk) = zip(kk,prices,vols)
            push!(data,[τ, H, k, "Adams-256", pricek, volk])
        end
        pricer = CharFuncPricing.AdaptiveFilonCharFuncPricer(cfa1k,τ,qTol=1e-8,myTrans=CharFuncPricing.IdentityTransformation(0.0,upperBound1k[Hindex]/sqrt(τ)))
        prices = map(x -> priceEuropean(pricer, x >= forward, x, forward,τ,1.0),strikes)
        vols = [Black.impliedVolatility(strike >= forward,max(p,1e-32),forward,strike,τ,1.0) for (strike,p) = zip(strikes,prices)]
        for (k,pricek,volk) = zip(kk,prices,vols)
            push!(data,[τ, H, k, "Adams-1024", pricek, volk])
        end
    end
    end

    ## Adams difficulties wirh rho close to -1. (-0.9) prices are too small in the right wing (difficulties below 10^-5)
    #The smile is always the same?!? vary varianceCurve instead.
    #=  data1 = data[data.T .== 1.0,:]
     data1 = data[data.T .== 1.0 .&& (data.Method .== "Pade" .|| data.Method .== "Adams-1024"),:]
 plot([plot(gdata.Logmoneyness, gdata.Vol .* 100, group=gdata.Method, xlab="Log-moneyness", ylab=if gdata.H[1]==0.01 "Implied volatility in %" else "" end, legend = gdata.H[1]==0.45, title=string("H=",gdata.H[1]), ylim=(10.0,25.0)) for gdata in groupby(data1,:H)]..., layout=(1,4))
plot([plot(gdata.Logmoneyness, gdata.Vol .* 100, group=gdata.Method, xlab="Log-moneyness", ylab=if gdata.H[1]==0.01 "Implied volatility in %" else "" end, legend = gdata.H[1]==0.45, title=string("H=",gdata.H[1]), ylim=(7.5,35.0),xlim=(-0.75,0.75)) for gdata in groupby(data1,:H)]..., layout=(1,4))
 plot!(size=(1024,256))

julia> savefig("/home/fabien/mypapers/eqd_book/rheston_h_pade.pdf")
"/home/fabien/mypapers/eqd_book/rheston_h_pade.pdf"


    plot()

=#

end

@testset "Rough-Calibration" begin
    #Calibration to market data?
     # single slice SPX 1w, 1m. Step1: compute var swap price, step 2, calibrate the 3 params.
     # Kahale? or other market data?


     tte = 8 / 365
     forward = 2385.099980
     logmoneynessA = [-0.6869194871171992, -0.6068767794436628, -0.532768807289941, -0.4976774874786709, -0.4637759358029896, -0.4309861129799986, -0.3992374146654184, -0.3684657559986646, -0.33861279284898355, -0.3240139934278308, -0.30962525597573115, -0.2954406209837748, -0.2814543790090349, -0.26766105687669905, -0.25405540482092054, -0.2406323844887798, -0.23796926706929608, -0.23266421483960298, -0.22738715773875914, -0.22213780185261547, -0.21691585787146372, -0.21431507617140633, -0.2117210409943598, -0.20655307083591715, -0.20141167133549853, -0.19885085047382484, -0.19629657066872805, -0.1937487985899293, -0.19120750116125684, -0.18867264555806873, -0.18614419920471004, -0.18362212977200013, -0.18110640517475293, -0.17859699356932715, -0.17609386335120858, -0.17359698315262134, -0.17110632184016958, -0.1686218485125076, -0.1661435324980405, -0.1636713433526514, -0.161205250857458, -0.1587452250165956, -0.15629123605502887, -0.1538432544163888, -0.15140125076083716, -0.148965195962956, -0.1465350611096642, -0.1441108174981579, -0.14169243663387623, -0.1392798902284923, -0.13687315019792728, -0.1344721886603892, -0.13207697793443432, -0.12968749053705286, -0.12730369918177656, -0.12492557677680924, -0.12255309642317881, -0.12018623141291263, -0.1178249552272328, -0.11546924153477382, -0.11311906418982019, -0.11077439723056604, -0.10843521487739377, -0.10610149153117368, -0.10377320177158252, -0.1014503203554428, -0.09913282221508024, -0.0968206824567008, -0.09451387635878575, -0.09221237937050648, -0.08991616711015624, -0.08762521536360049, -0.08533950008274448, -0.08305899738401923, -0.0807836835468837, -0.07851353501234465, -0.07624852838149249, -0.07398864041405494, -0.0717338480269658, -0.06948412829295042, -0.06723945843912654, -0.06499981584562174, -0.06276517804420538, -0.06053552271693634, -0.058310827694825165, -0.05609107095651222, -0.05387623062695936, -0.051666284976156614, -0.04946121241784262, -0.047260991508240154, -0.04506560094480458, -0.042875019564985926, -0.04068922634500564, -0.038508200398645415, -0.036331920976049974, -0.034160367462542106, -0.03199351937745175, -0.029831356372956492, -0.027673858232935385, -0.02552100487183431, -0.023372776333544718, -0.021229152790293252, -0.019090114541543912, -0.016955642012911255, -0.01482571575508632, -0.012700316442772813, -0.010579424873635267, -0.00846302196725765, -0.006351088764114094, -0.004243606424549351, -0.0021405562277706388, -4.191957084939753e-05, 0.00205232203226539, 0.004142186951724677, 0.006227693442746083, 0.008308859646570679, 0.010385703591409612, 0.012458243193382021, 0.01452649625744107, 0.016590480478292587, 0.018650213441303184, 0.020705712623399236, 0.02275699539395665, 0.024804079015681525, 0.026846980645481605, 0.02888571733532992, 0.030920306033117483, 0.0329507635834995, 0.03497710672873193, 0.03699935210949975, 0.03901751626573696, 0.04103161563743823, 0.043041666565462246, 0.04504768529232801, 0.047049687963001116, 0.0490476906256742, 0.051041709232538625, 0.05303175964054866, 0.055017857612178096, 0.057000018816169125, 0.05897825882827487, 0.06292303711929127, 0.06685231525918088, 0.0766084902045455, 0.08627040111628252, 0.10531859608697686, 0.12401072909912947]
     varianceA = [1.8200197664374549, 1.4370094334451218, 1.1219099566714799, 0.9859599462531226, 0.8627183857443178, 0.7511014725956007, 0.6501483352485907, 0.5590044022510807, 0.47690751535658166, 0.4390359056317844, 0.4031762929848544, 0.36925452377926, 0.3372003590784474, 0.30694725112809984, 0.27843213923947446, 0.2515952639413264, 0.29502153563617, 0.2830047657555973, 0.2712844890319138, 0.24226714853976236, 0.23183476228172648, 0.22671954041689377, 0.22167105301385384, 0.2117725713755103, 0.20213594455246672, 0.19741479334333528, 0.19275787514762352, 0.18816478940262324, 0.18363514013269241, 0.19245394381540987, 0.18774646219447227, 0.1931708733800091, 0.18836213710401709, 0.18362166477667016, 0.17894905232157457, 0.1743439006235602, 0.16980581551151402, 0.16533440774315933, 0.16092929299184514, 0.15659009183542455, 0.15231642974729748, 0.14810793708970532, 0.14396424910937627, 0.15175373848335327, 0.14742546106522328, 0.14316525056148532, 0.13897274278814326, 0.13484757886042228, 0.13078940520904156, 0.12679787360018158, 0.12698507216807917, 0.12666214036631684, 0.12264576148854993, 0.11869797410294719, 0.1148184450420629, 0.11100684703754489, 0.10726285877978288, 0.1090416962702069, 0.10525922821145543, 0.10154614385585496, 0.09790213839890792, 0.09432691374485798, 0.09492399037226473, 0.09134495876803216, 0.08597378047010444, 0.08439736588823006, 0.08426114388629118, 0.08084296929154224, 0.07749604380802587, 0.07557643823153187, 0.07357590949849342, 0.07033998972395482, 0.0693831071831427, 0.06620062034223427, 0.06309160518186421, 0.06188662882497834, 0.058843681476270004, 0.05667929073178767, 0.05449735900229306, 0.052304126192498454, 0.0501052963138917, 0.047906115908252146, 0.045711439117111015, 0.044042558111935956, 0.041353370093888606, 0.03964591852876113, 0.03748009108593441, 0.03572173670686245, 0.03358414257983117, 0.0318043449203995, 0.03000981456603729, 0.028476576574445737, 0.02665244238817238, 0.025279020460443044, 0.023638525471996946, 0.02234149404948676, 0.02096744995618417, 0.019534675716108427, 0.01843908912068194, 0.017230629061443955, 0.01613237800134248, 0.015096495307406125, 0.01424005329279472, 0.013278697405891386, 0.012223570633781915, 0.011558353962214205, 0.01084803389956623, 0.01015550946791267, 0.009439143257793953, 0.009017014431699322, 0.008459419775164121, 0.00800388501313797, 0.007667388809916037, 0.007428053991319023, 0.007291029572886738, 0.007146673645267026, 0.007028007621607059, 0.0070508200234122706, 0.007083084168032932, 0.007226153861559947, 0.007404430732963872, 0.007752077699396105, 0.008092286263668957, 0.008497349440946097, 0.008882001670948669, 0.009550658436510925, 0.009819636480828963, 0.0105822823697076, 0.010843199460539351, 0.01195258512100742, 0.012513863578351731, 0.012944164536274989, 0.01407980127355266, 0.01525521935110815, 0.0164698287915165, 0.017723063978316998, 0.019014381486472137, 0.02034325819835555, 0.020375932932072235, 0.019953139348489884, 0.021212731474714613, 0.023827799654318166, 0.023357353790689615, 0.029924305157087, 0.03715120974705468, 0.0534583441284658, 0.07205188800574357]
     weightsA = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 2, 2, 2, 2, 2, 2, 3.999999999999999, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.000000000000001, 3.000000000000001, 3.000000000000001, 3.000000000000001, 3.000000000000001, 3.500000000000001, 3.500000000000001, 6.000000000000002, 3.500000000000001, 3.999999999999999, 3.999999999999999, 3.999999999999999, 2.9999999999999996, 4.499999999999998, 4.499999999999998, 5.000000000000001, 5.000000000000001, 5.000000000000001, 5.499999999999999, 5.499999999999999, 4.000000000000001, 6.000000000000002, 4.333333333333333, 6.500000000000002, 4.66666666666667, 7.000000000000002, 7.500000000000002, 7.500000000000002, 7.999999999999993, 5.666666666666665, 5.999999999999999, 9.000000000000004, 9.500000000000002, 10.000000000000002, 7.333333333333328, 7.6666666666666705, 8.33333333333334, 12.99999999999999, 9.666666666666671, 15.499999999999988, 11.33333333333334, 12.666666666666673, 13.999999999999988, 15.666666666666679, 17.66666666666668, 12.4, 17.499999999999982, 13.333333333333341, 15.666666666666679, 18.333333333333343, 16.249999999999986, 19.000000000000025, 22.74999999999998, 21.6, 25.6, 34.000000000000114, 26.666666666666682, 20.66666666666668, 15.666666666666679, 11.666666666666675, 17.000000000000007, 9.500000000000002, 9.33333333333334, 6.999999999999999, 7.999999999999993, 4.333333333333333, 5.000000000000001, 3.999999999999999, 3.500000000000001, 5.000000000000001, 2.5, 2, 2, 3.000000000000001, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 1, 1, 1, 1, 1, 1, 1, 1]
     volA = sqrt.(varianceA)
     strikeA = exp.(logmoneynessA) .* forward
     refPrices, weights = AQFED.Collocation.weightedPrices(true, strikeA, volA, weightsA, forward, 1.0, tte,vegaFloor=1e-4)
     svi0, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logmoneynessA, volA, weightsA, aMin=0.0)
     ivkSVI0 = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi0, logmoneynessA))
     rmseSVI0 = StatsBase.rmsd(volA, ivkSVI0)
      xiFuka = AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.FukasawaVarianceSwapReplication(true),tte, logmoneynessA, varianceA, 1.0,u=1.0)
     subset = findall(x -> x > -0.4*sqrt(tte) && x < 0.4*sqrt(tte), logmoneynessA)
     svi1, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logmoneynessA[subset], volA[subset], weightsA[subset], aMin=0.0)
     volSpline = CubicSplineNatural(log.(strikeA[subset]./forward),volA[subset])
     guess = AQFED.VolatilityModels.initialGuessBlackATM(forward,tte,1.0,volSpline(0.0),PPInterpolation.evaluateDerivative(volSpline,0.0),PPInterpolation.evaluateSecondDerivative(volSpline,0.0))
     sectionSPX = AQFED.VolatilityModels.calibrateSABRSectionFromGuess(tte,forward,logmoneynessA[subset],volA[subset],weightsA[subset],guess)
 
     xiFuka = AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.FukasawaVarianceSwapReplication(true),tte, logmoneynessA[subset], varianceA[subset], 1.0,u=1.0)

     transformV = [AlgebraicTransformation(0.01, 0.5), AlgebraicTransformation(-0.99, 0.99), MQMinTransformation(0.01,1.0)] #MQMinTransformation(0.0001,1.0)]
     obj = function (fvec::Z, coeff::AbstractArray{W}) where {Z,W}
        xi = xiFuka #transformV[4](coeff[4])
        params0 = RoughHestonParams( transformV[1](coeff[1]) , transformV[2](coeff[2]),transformV[3](coeff[3]), s -> xi)
        cf = PadeCharFunc(params0)
        pricer = CharFuncPricing.AdaptiveFilonCharFuncPricer(cf,tte,qTol=1e-8)
        prices = map(x -> priceEuropean(pricer, true, x, forward, tte,1.0),strikeA[subset])
        @. fvec = weights[subset] * (prices - refPrices[subset])
    end
     x0 = [0.1, -0.65, 0.4] #H, rho, nu
     coeff0 = [AQFED.Math.inv(transformV[i],x0[i]) for i = eachindex(x0)]
     fvec = zeros(length(subset));
     objValue, fit = GaussNewton.optimize!(obj, coeff0, fvec, autodiff=:single, iscale=1, abstol=length(fvec) * eps())
     coeff = fit.minimizer
        xi = xiFuka #  transformV[4](coeff[4]) #xiFuka     
        params0 = RoughHestonParams( transformV[1](coeff[1]) , transformV[2](coeff[2]),transformV[3](coeff[3]), s-> xi )          # s -> xiFuka)
        cf = PadeCharFunc(params0)
        pricer = CharFuncPricing.AdaptiveFilonCharFuncPricer(cf,tte,qTol=1e-8)
        
#### calibrate v0 as well, estimate by Fuka may be very large: example: SPX500. Is it really var swap price? CHECK THIS
transformV = [AlgebraicTransformation(0.01, 0.5), AlgebraicTransformation(-0.99, 0.99), MQMinTransformation(0.01,1.0),MQMinTransformation(0.0001,1.0)]
obj = function (fvec::Z, coeff::AbstractArray{W}) where {Z,W}
   xi = transformV[4](coeff[4])
   params0 = RoughHestonParams( transformV[1](coeff[1]) , transformV[2](coeff[2]),transformV[3](coeff[3]), s -> xi)
   cf = PadeCharFunc(params0)
   pricer = CharFuncPricing.AdaptiveFilonCharFuncPricer(cf,tte,qTol=1e-8)
   prices = map(x -> priceEuropean(pricer, true, x, forward, tte,1.0),strikeA[subset])
   @. fvec = weights[subset] * (prices - refPrices[subset])
end
x0 = [0.1, -0.65, 0.4,xiFuka] #H, rho, nu
coeff0 = [AQFED.Math.inv(transformV[i],x0[i]) for i = eachindex(x0)]
fvec = zeros(length(subset));
objValue, fit = GaussNewton.optimize!(obj, coeff0, fvec, autodiff=:single, iscale=1, abstol=length(fvec) * eps())
coeff = fit.minimizer
xi = transformV[4](coeff[4]) #xiFuka     
params1 = RoughHestonParams( transformV[1](coeff[1]) , transformV[2](coeff[2]),transformV[3](coeff[3]), s-> xi )       
cf = PadeCharFunc(params1)
pricerF = CharFuncPricing.AdaptiveFilonCharFuncPricer(cf,tte,qTol=1e-8)
cfa = AdamsCharFunc(params0,2000)
pricerA = CharFuncPricing.AdaptiveFilonCharFuncPricer(cfa,tte,qTol=1e-8,myTrans=CharFuncPricing.IdentityTransformation(0.0,12*150.0))
cfc = AQFED.Rough.RoughHestonCVCharFunc(cfa) #with Black control variate, the price otm is better.
pricerC = CharFuncPricing.AdaptiveFilonCharFuncPricer(cfc,tte,qTol=1e-8,myTrans=CharFuncPricing.IdentityTransformation(0.0,12*150.0))


objHestonFullPrices = function (fvec::Z, coeff::AbstractArray{W}) where {Z,W}
    local params0 = HestonParams(transformV[3](coeff[1]),1.0,transformV[3](coeff[1]), transformV[2](coeff[2]),transformV[3](coeff[3]))
    local cf = DefaultCharFunc(params0)
    pricer = CharFuncPricing.ALCharFuncPricer(cf)     
    prices = map(x -> priceEuropean(pricer, true, x, forward, tte,1.0),strikeA[subset])
    @. fvec = weights[subset] * (prices - refPrices[subset])
 end
 x0 = [xiFuka, -0.65, 0.4] # rho, nu
 coeff0 =  [AQFED.Math.inv(transformV[3],x0[1]),AQFED.Math.inv(transformV[2],x0[2]),AQFED.Math.inv(transformV[3],x0[3])]
 fvec = zeros(length(subset));
 objValue, fit = GaussNewton.optimize!(objHestonFullPrices, coeff0, fvec, autodiff=:single, iscale=1, abstol=length(fvec) * eps())
 coeff = fit.minimizer
 paramsF = HestonParams(transformV[3](coeff[1]),1.0,transformV[3](coeff[1]), transformV[2](coeff[2]),transformV[3](coeff[3]))
 cf = DefaultCharFunc(paramsF)
 pricerFullHeston = CharFuncPricing.ALCharFuncPricer(cf)     


#TODO ChebyshevCharFunc(AdamsCharFunc) what about cheb poly 64 points on cos, sin and then use AdaptiveFilon/Flinn. How large is the diff?
#0.123 vs 0.114 RMSE   vs  0.213431  HESTON
     #obj1 = LsqFit.OnceDifferentiable(obj, coeff0, copy(fvec); autodiff=:finite, inplace=true)       
     #fit = LsqFit.levenberg_marquardt(obj1, x0,  lower=[0.00], upper=maxValueV)

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

 strikeA = Float64.([1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2325, 2350, 2375, 2400, 2425, 2450, 2470, 2475, 2480, 2490, 2510, 2520, 2530, 2540, 2550, 2560, 2570, 2575, 2580, 2590, 2600, 2610, 2615, 2620, 2625, 2630, 2635, 2640, 2645, 2650, 2655, 2660, 2665, 2670, 2675, 2680, 2685, 2690, 2695, 2700, 2705, 2710, 2715, 2720, 2725, 2730, 2735, 2740, 2745, 2750, 2755, 2760, 2765, 2770, 2775, 2780, 2785, 2790, 2795, 2800, 2805, 2810, 2815, 2835, 2860, 2900])
    volA = [0.684882717072609, 0.6548002174209514, 0.6279717042323061, 0.6040669049212617, 0.5769233835086068, 0.5512534351594732, 0.5260245499632258, 0.5004353919117, 0.4741366518169333, 0.46171589561249216, 0.4457089283432941, 0.4336614266663264, 0.420159764469498, 0.4074628373496824, 0.3931682390848574, 0.3814047904881801, 0.37929970817058073, 0.3771088224218263, 0.3724714977308359, 0.36029419336555424, 0.35467069448268806, 0.3505327949033959, 0.3441904382413214, 0.3392727917494692, 0.33306859556194446, 0.32820593458977093, 0.3243137942797042, 0.32204084870033645, 0.3168000315981532, 0.3109143207658998, 0.3050420836154825, 0.30241566311445206, 0.29948796266862154, 0.29609035936524486, 0.2923777072285143, 0.28951623883712746, 0.28584033838767425, 0.283342147794602, 0.2808533651372528, 0.27703523377755246, 0.27371493615870945, 0.2708906740100327, 0.2678887418986713, 0.2645328136650213, 0.26234402136468965, 0.2585977172018311, 0.25550003988953746, 0.2521896614376435, 0.2495339851370865, 0.24665927818229774, 0.24355339309186683, 0.24020198229067014, 0.23658800157061083, 0.23457380906338043, 0.23040670495884208, 0.2278656924642955, 0.22304945749920857, 0.21988751701341647, 0.2184983910827269, 0.21470247194448602, 0.21050624458263925, 0.20817463333507674, 0.20550780781621286, 0.19996741584940433, 0.19900703596491134, 0.19506177682405323, 0.19054732989021844, 0.18842657099566548, 0.18589260856179804, 0.18287835748424114, 0.17929170978590483, 0.17500063441150882, 0.18575101811296996, 0.20717302702012957, 0.22524785579801052]
    weightsA = [1.1789826122551597, 1.224744871391589, 1.284523257866513, 1.3601470508735443, 1.4317821063276355, 1.5165750888103102, 1.6124515496597098, 1.7175564037317668, 1.8395212376698413, 1.8973665961010275, 1.949358868961793, 2.024845673131659, 2.0976176963403033, 2.179449471770337, 2.258317958127243, 2.32379000772445, 2.345207879911715, 2.3664319132398464, 2.4083189157584592, 2.479919353527449, 2.5199206336708304, 2.569046515733026, 2.6076809620810595, 2.6551836094703507, 2.7625312572654126, 2.7477263328068173, 2.765863337187866, 2.7928480087537886, 2.871393034605969, 2.964704653791087, 3.0174928596261394, 3.0017001984929568, 3, 3.024896692450835, 3.120391338480345, 2.9916550603303182, 2.947349434130382, 2.8809720581775866, 2.8284271247461903, 2.8083087326973732, 2.711088342345192, 2.6685599339741506, 2.62445329583912, 2.565469285152568, 2.4899799195977463, 2.4289915602982237, 2.4279079146675357, 2.313006701244076, 2.258317958127243, 2.202271554554524, 2.1447610589527217, 2.085665361461421, 2.024845673131659, 1.97484176581315, 1.91049731745428, 1.857417562100671, 1.7888543819998317, 1.7320508075688772, 1.6881943016134133, 1.6278820596099706, 1.5652475842498528, 1.5165750888103102, 1.466287829861518, 1.3964240043768943, 1.3601470508735443, 1.3038404810405297, 1.2449899597988732, 1.2041594578792296, 1.161895003862225, 1.118033988749895, 1.0723805294763609, 1.02469507659596, 1.0099504938362078, 1.0910894511799618, 1.005037815259212]
    forward = 2629.8026715608194
    tte = 0.0821917808219178
    logmoneynessA = log.(strikeA ./ forward)
    sumw2 = sum(weightsA .^ 2)
    w = weightsA ./ sqrt(sumw2)
    
end
using PPInterpolation
@testset "Rough-SPX500-Oct2010" begin
strikeA = [1008,	1209.6,	1411.2,	1612.8,	1814.4,	1915.2,	2016,	2116.8,	2217.6,	2419.2,	2620.8,	3024]
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
volM = [    0.469330877	0.414404424	0.367296885	0.327108549	0.294442259	0.281546662	0.271399379	0.264274619	0.260233568	0.260279105	0.267799208	0.291990984;
    0.438031838	0.387414998	0.344186959	0.307577311	0.27820473	0.266796144	0.257960029	0.251900797	0.248624262	0.249290252	0.256555792	0.279222439;
    0.417585513	0.372442867	0.334123526	0.301871891	0.275993204	0.265780032	0.257609374	0.251598649	0.247755671	0.245926111	0.249948783	0.266604403;
    0.378863656	0.343692938	0.314288789	0.289927723	0.270499867	0.262699832	0.256217647	0.251063253	0.247212335	0.243118381	0.243005143	0.250232597;
    0.361240173	0.331461693	0.306727965	0.286289798	0.269839268	0.263077346	0.257281885	0.252439294	0.248523708	0.243291194	0.241073648	0.243068308;
    0.348498059	0.323073547	0.302069305	0.284735901	0.270667776	0.264780547	0.259625421	0.2551777	0.251409317	0.245772883	0.242388102	0.240807094;
    0.336927448	0.314798449	0.296604756	0.281620193	0.269409546	0.264250663	0.259682153	0.255675656	0.252202671	0.246737402	0.243031476	0.239805477;
    0.329408098	0.309509953	0.293207982	0.279800062	0.26884608	0.264190728	0.260040257	0.256365655	0.253139096	0.247920833	0.244167054	0.240189782;
    0.323014691	0.304706378	0.289747629	0.27745988	0.267408265	0.263122021	0.259285776	0.255870853	0.252850352	0.247890508	0.244208985	0.239941502;
    0.319115	0.301957248	0.287962586	0.276471393	0.267055619	0.263027894	0.259411063	0.256177099	0.25330008	0.248520602	0.24489072	0.240422639;
    0.313394981	0.297812921	0.285123732	0.274699544	0.266128184	0.262442454	0.259115821	0.256121744	0.253436081	0.248902532	0.245354116	0.240656834]
r = 0.02
q = 0.01
spot = 2016.0
forwardA = [spot * exp((r-q)*tte) for tte = ttes]
varswapPriceA = zeros(length(ttes))
#compute forward var curve
for (i, tte) = enumerate(ttes)
    logmoneynessA = log.(strikeA./forwardA[i])
    varianceA = volM[i,:].^2
    varswapPriceA[i] = AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.FukasawaVarianceSwapReplication(true),tte, logmoneynessA, varianceA, 1.0,u=1.0)
end
varswapSpline = CubicSplineNatural(vcat(0.0,ttes), vcat(0.0,varswapPriceA .* ttes))
xi = function(u)
    if u <= 0.0
        #PPInterpolation.evaluateDerivative(varswapSpline,ttes[1])
        varswapPriceA[1]
    elseif u >= ttes[end]
        PPInterpolation.evaluateDerivative(varswapSpline,ttes[end])
    else
        PPInterpolation.evaluateDerivative(varswapSpline,u)
    end
end
varswapLinear = PPInterpolation.makeLinearPP(vcat(0.0,ttes), vcat(0.0,varswapPriceA .* ttes))
xiLinear = function(u)
    if u <= 0.0
        varswapPriceA[1]
    elseif u >= ttes[end]
        PPInterpolation.evaluateDerivative(varswapLinear,ttes[end]-1e-8)
    else
        PPInterpolation.evaluateDerivative(varswapLinear,u)
    end
end

# fit heston

objHeston = function(fvec::Z, coeff::AbstractArray{W}) where {Z,W}
    v0 = coeff[1]
    theta = coeff[2]
    kappa = coeff[3]
    @. fvec = [theta + (1-exp(-kappa*t))/(kappa*t)*(v0-theta) for t in ttes] - varswapPriceA
 end
 coeff0 = [varswapPriceA[1], varswapPriceA[end],1.0]
 fvec = zeros(length(varswapPriceA))
 objValue, fit = GaussNewton.optimize!(objHeston, coeff0, fvec, autodiff=:single, iscale=1, abstol=length(fvec) * eps())
 coeff = fit.minimizer
 v0 = coeff[1]
 theta = coeff[2]
 kappa = coeff[3]
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
#xSSVI forward curve?

#calibrate rough heston
refPriceM = zeros(size(volM))
weightM = zeros(size(volM))
for (i,tte) = enumerate(ttes)
refPrices, weights = AQFED.Collocation.weightedPrices(true, strikeA, volM[i,:], ones(length(strikeA)), forwardA[i], 1.0, tte,vegaFloor=1e-4)
refPriceM[i,:] = refPrices
weightM[i,:] = weights
end
transformV = [AlgebraicTransformation(0.01, 0.5), AlgebraicTransformation(-0.99, 0.99), MQMinTransformation(0.01,1.0)] #MQMinTransformation(0.0001,1.0)]
obj = function (fvec::Z, coeff::AbstractArray{W}) where {Z,W}
   local xiLocal = xi
   local params0 = RoughHestonParams( transformV[1](coeff[1]) , transformV[2](coeff[2]),transformV[3](coeff[3]), xiLocal)
   local cf = PadeCharFunc{RoughHestonParams{Float64},Complex,3}(params0, Chebyshev{Float64,1}(64))
   for (i,tte) = enumerate(ttes)
     pricer = CharFuncPricing.AdaptiveFilonCharFuncPricer(cf,tte,qTol=1e-8)     
     for (j,x) = enumerate(strikeA)
      price = priceEuropean(pricer, true, x, forwardA[i], tte,1.0)
      fvec[j+(i-1)*length(ttes)] = weightM[i,j] * (price -refPriceM[i,j])
     end
    end
end
x0 = [0.1, -0.65, 0.4] #H, rho, nu
coeff0 = [AQFED.Math.inv(transformV[i],x0[i]) for i = eachindex(x0)]
fvec = zeros(length(refPriceM));
objValue, fit = GaussNewton.optimize!(obj, coeff0, fvec, autodiff=:single, iscale=1, abstol=length(fvec) * eps())
coeff = fit.minimizer
xiLocal = xi
params0 = RoughHestonParams( transformV[1](coeff[1]) , transformV[2](coeff[2]),transformV[3](coeff[3]), xiLocal )       
cf = PadeCharFunc{RoughHestonParams{Float64},Complex,3}(params0, Chebyshev{Float64,1}(64))
pricerA = [CharFuncPricing.AdaptiveFilonCharFuncPricer(cf,tte,qTol=1e-8) for tte = ttes]
priceM = [priceEuropean(pricerA[i],true, strike, forwardA[i], tte, 1.0) for (i,tte) = enumerate(ttes), (j,strike) = enumerate(strikeA)]


objHestonPrices = function (fvec::Z, coeff::AbstractArray{W}) where {Z,W}
   local params0 = HestonParams(v0,kappa,theta, transformV[2](coeff[1]),transformV[3](coeff[2]))
   local cf = DefaultCharFunc(params0)
   pricer = CharFuncPricing.ALCharFuncPricer(cf)     
   for (i,tte) = enumerate(ttes)
      for (j,x) = enumerate(strikeA)
      price = priceEuropean(pricer, true, x, forwardA[i], tte,1.0)
      fvec[j+(i-1)*length(ttes)] = weightM[i,j] * (price -refPriceM[i,j])
     end
    end
end
x0 = [-0.65, 0.4] # rho, nu
coeff0 =  [AQFED.Math.inv(transformV[2],x0[1]),AQFED.Math.inv(transformV[3],x0[2])]
fvec = zeros(length(refPriceM));
objValue, fit = GaussNewton.optimize!(objHestonPrices, coeff0, fvec, autodiff=:single, iscale=1, abstol=length(fvec) * eps())
coeff = fit.minimizer
params0 = HestonParams(v0,kappa,theta, transformV[2](coeff[1]),transformV[3](coeff[2]))
cf = DefaultCharFunc(params0)
pricerHeston = CharFuncPricing.ALCharFuncPricer(cf)     
   
objHestonFullPrices = function (fvec::Z, coeff::AbstractArray{W}) where {Z,W}
    local params0 = HestonParams(transformV[3](coeff[1]),transformV[3](coeff[2]),transformV[3](coeff[3]), transformV[2](coeff[4]),transformV[3](coeff[5]))
    local cf = DefaultCharFunc(params0)
    pricer = CharFuncPricing.ALCharFuncPricer(cf)     
    for (i,tte) = enumerate(ttes)
       for (j,x) = enumerate(strikeA)
       price = priceEuropean(pricer, true, x, forwardA[i], tte,1.0)
       fvec[j+(i-1)*length(ttes)] = weightM[i,j] * (price -refPriceM[i,j])
      end
     end
 end
 x0 = [v0, kappa, theta, -0.65, 0.4] # rho, nu
 coeff0 =  [AQFED.Math.inv(transformV[3],x0[1]),AQFED.Math.inv(transformV[3],x0[2]),AQFED.Math.inv(transformV[3],x0[3]),AQFED.Math.inv(transformV[2],x0[4]),AQFED.Math.inv(transformV[3],x0[5])]
 fvec = zeros(length(refPriceM));
 objValue, fit = GaussNewton.optimize!(objHestonFullPrices, coeff0, fvec, autodiff=:single, iscale=1, abstol=length(fvec) * eps())
 coeff = fit.minimizer
 paramsF = HestonParams(transformV[3](coeff[1]),transformV[3](coeff[2]),transformV[3](coeff[3]), transformV[2](coeff[4]),transformV[3](coeff[5]))
 cf = DefaultCharFunc(paramsF)
 pricerFullHeston = CharFuncPricing.ALCharFuncPricer(cf)     
 
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

end