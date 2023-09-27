using AQFED, Test
using StatsBase
using AQFED.Black
using AQFED.Collocation
import Polynomials: coeffs
using AQFED.Math
using AQFED.PDDE
#using Plots


@testset "svi-bad" begin
    tte=1.0
    forward=1.0
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
    prices, weights = Collocation.weightedPrices(true, strikes, volA, weightsA, forward, 1.0, tte,vegaFloor=1e-8)
    lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, model=PDDE.Quadratic(),location="Equidistributed",size=10,L=strikes[1],U=strikes[end])
    ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0);        rmseq = StatsBase.L2dist(weights .* volA, weights .* ivkq)
    lvgqe = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, model=PDDE.Quadratic(),location="Mid-XX",size=0,L=strikes[1],U=strikes[end])
    ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgqe, true, strikes), forward, strikes, tte, 1.0);
    rmseq = StatsBase.L2dist(weights .* volA, weights .* ivkq)

end
@testset "spxw170324_170316" begin
    tte = 8 / 365
    forward = 2385.099980
    logmoneynessA = [-0.6869194871171992, -0.6068767794436628, -0.532768807289941, -0.4976774874786709, -0.4637759358029896, -0.4309861129799986, -0.3992374146654184, -0.3684657559986646, -0.33861279284898355, -0.3240139934278308, -0.30962525597573115, -0.2954406209837748, -0.2814543790090349, -0.26766105687669905, -0.25405540482092054, -0.2406323844887798, -0.23796926706929608, -0.23266421483960298, -0.22738715773875914, -0.22213780185261547, -0.21691585787146372, -0.21431507617140633, -0.2117210409943598, -0.20655307083591715, -0.20141167133549853, -0.19885085047382484, -0.19629657066872805, -0.1937487985899293, -0.19120750116125684, -0.18867264555806873, -0.18614419920471004, -0.18362212977200013, -0.18110640517475293, -0.17859699356932715, -0.17609386335120858, -0.17359698315262134, -0.17110632184016958, -0.1686218485125076, -0.1661435324980405, -0.1636713433526514, -0.161205250857458, -0.1587452250165956, -0.15629123605502887, -0.1538432544163888, -0.15140125076083716, -0.148965195962956, -0.1465350611096642, -0.1441108174981579, -0.14169243663387623, -0.1392798902284923, -0.13687315019792728, -0.1344721886603892, -0.13207697793443432, -0.12968749053705286, -0.12730369918177656, -0.12492557677680924, -0.12255309642317881, -0.12018623141291263, -0.1178249552272328, -0.11546924153477382, -0.11311906418982019, -0.11077439723056604, -0.10843521487739377, -0.10610149153117368, -0.10377320177158252, -0.1014503203554428, -0.09913282221508024, -0.0968206824567008, -0.09451387635878575, -0.09221237937050648, -0.08991616711015624, -0.08762521536360049, -0.08533950008274448, -0.08305899738401923, -0.0807836835468837, -0.07851353501234465, -0.07624852838149249, -0.07398864041405494, -0.0717338480269658, -0.06948412829295042, -0.06723945843912654, -0.06499981584562174, -0.06276517804420538, -0.06053552271693634, -0.058310827694825165, -0.05609107095651222, -0.05387623062695936, -0.051666284976156614, -0.04946121241784262, -0.047260991508240154, -0.04506560094480458, -0.042875019564985926, -0.04068922634500564, -0.038508200398645415, -0.036331920976049974, -0.034160367462542106, -0.03199351937745175, -0.029831356372956492, -0.027673858232935385, -0.02552100487183431, -0.023372776333544718, -0.021229152790293252, -0.019090114541543912, -0.016955642012911255, -0.01482571575508632, -0.012700316442772813, -0.010579424873635267, -0.00846302196725765, -0.006351088764114094, -0.004243606424549351, -0.0021405562277706388, -4.191957084939753e-05, 0.00205232203226539, 0.004142186951724677, 0.006227693442746083, 0.008308859646570679, 0.010385703591409612, 0.012458243193382021, 0.01452649625744107, 0.016590480478292587, 0.018650213441303184, 0.020705712623399236, 0.02275699539395665, 0.024804079015681525, 0.026846980645481605, 0.02888571733532992, 0.030920306033117483, 0.0329507635834995, 0.03497710672873193, 0.03699935210949975, 0.03901751626573696, 0.04103161563743823, 0.043041666565462246, 0.04504768529232801, 0.047049687963001116, 0.0490476906256742, 0.051041709232538625, 0.05303175964054866, 0.055017857612178096, 0.057000018816169125, 0.05897825882827487, 0.06292303711929127, 0.06685231525918088, 0.0766084902045455, 0.08627040111628252, 0.10531859608697686, 0.12401072909912947]
    varianceA = [1.8200197664374549, 1.4370094334451218, 1.1219099566714799, 0.9859599462531226, 0.8627183857443178, 0.7511014725956007, 0.6501483352485907, 0.5590044022510807, 0.47690751535658166, 0.4390359056317844, 0.4031762929848544, 0.36925452377926, 0.3372003590784474, 0.30694725112809984, 0.27843213923947446, 0.2515952639413264, 0.29502153563617, 0.2830047657555973, 0.2712844890319138, 0.24226714853976236, 0.23183476228172648, 0.22671954041689377, 0.22167105301385384, 0.2117725713755103, 0.20213594455246672, 0.19741479334333528, 0.19275787514762352, 0.18816478940262324, 0.18363514013269241, 0.19245394381540987, 0.18774646219447227, 0.1931708733800091, 0.18836213710401709, 0.18362166477667016, 0.17894905232157457, 0.1743439006235602, 0.16980581551151402, 0.16533440774315933, 0.16092929299184514, 0.15659009183542455, 0.15231642974729748, 0.14810793708970532, 0.14396424910937627, 0.15175373848335327, 0.14742546106522328, 0.14316525056148532, 0.13897274278814326, 0.13484757886042228, 0.13078940520904156, 0.12679787360018158, 0.12698507216807917, 0.12666214036631684, 0.12264576148854993, 0.11869797410294719, 0.1148184450420629, 0.11100684703754489, 0.10726285877978288, 0.1090416962702069, 0.10525922821145543, 0.10154614385585496, 0.09790213839890792, 0.09432691374485798, 0.09492399037226473, 0.09134495876803216, 0.08597378047010444, 0.08439736588823006, 0.08426114388629118, 0.08084296929154224, 0.07749604380802587, 0.07557643823153187, 0.07357590949849342, 0.07033998972395482, 0.0693831071831427, 0.06620062034223427, 0.06309160518186421, 0.06188662882497834, 0.058843681476270004, 0.05667929073178767, 0.05449735900229306, 0.052304126192498454, 0.0501052963138917, 0.047906115908252146, 0.045711439117111015, 0.044042558111935956, 0.041353370093888606, 0.03964591852876113, 0.03748009108593441, 0.03572173670686245, 0.03358414257983117, 0.0318043449203995, 0.03000981456603729, 0.028476576574445737, 0.02665244238817238, 0.025279020460443044, 0.023638525471996946, 0.02234149404948676, 0.02096744995618417, 0.019534675716108427, 0.01843908912068194, 0.017230629061443955, 0.01613237800134248, 0.015096495307406125, 0.01424005329279472, 0.013278697405891386, 0.012223570633781915, 0.011558353962214205, 0.01084803389956623, 0.01015550946791267, 0.009439143257793953, 0.009017014431699322, 0.008459419775164121, 0.00800388501313797, 0.007667388809916037, 0.007428053991319023, 0.007291029572886738, 0.007146673645267026, 0.007028007621607059, 0.0070508200234122706, 0.007083084168032932, 0.007226153861559947, 0.007404430732963872, 0.007752077699396105, 0.008092286263668957, 0.008497349440946097, 0.008882001670948669, 0.009550658436510925, 0.009819636480828963, 0.0105822823697076, 0.010843199460539351, 0.01195258512100742, 0.012513863578351731, 0.012944164536274989, 0.01407980127355266, 0.01525521935110815, 0.0164698287915165, 0.017723063978316998, 0.019014381486472137, 0.02034325819835555, 0.020375932932072235, 0.019953139348489884, 0.021212731474714613, 0.023827799654318166, 0.023357353790689615, 0.029924305157087, 0.03715120974705468, 0.0534583441284658, 0.07205188800574357]
    weightsA = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 2, 2, 2, 2, 2, 2, 3.999999999999999, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.000000000000001, 3.000000000000001, 3.000000000000001, 3.000000000000001, 3.000000000000001, 3.500000000000001, 3.500000000000001, 6.000000000000002, 3.500000000000001, 3.999999999999999, 3.999999999999999, 3.999999999999999, 2.9999999999999996, 4.499999999999998, 4.499999999999998, 5.000000000000001, 5.000000000000001, 5.000000000000001, 5.499999999999999, 5.499999999999999, 4.000000000000001, 6.000000000000002, 4.333333333333333, 6.500000000000002, 4.66666666666667, 7.000000000000002, 7.500000000000002, 7.500000000000002, 7.999999999999993, 5.666666666666665, 5.999999999999999, 9.000000000000004, 9.500000000000002, 10.000000000000002, 7.333333333333328, 7.6666666666666705, 8.33333333333334, 12.99999999999999, 9.666666666666671, 15.499999999999988, 11.33333333333334, 12.666666666666673, 13.999999999999988, 15.666666666666679, 17.66666666666668, 12.4, 17.499999999999982, 13.333333333333341, 15.666666666666679, 18.333333333333343, 16.249999999999986, 19.000000000000025, 22.74999999999998, 21.6, 25.6, 34.000000000000114, 26.666666666666682, 20.66666666666668, 15.666666666666679, 11.666666666666675, 17.000000000000007, 9.500000000000002, 9.33333333333334, 6.999999999999999, 7.999999999999993, 4.333333333333333, 5.000000000000001, 3.999999999999999, 3.500000000000001, 5.000000000000001, 2.5, 2, 2, 3.000000000000001, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    volA = sqrt.(varianceA)
    svi0, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logmoneynessA, volA, weightsA, aMin=0.0)
    ivkSVI0 = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi0, logmoneynessA))
    rmseSVI0 = StatsBase.rmsd(volA, ivkSVI0)
    svi, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logmoneynessA, volA, weightsA, aMin=-2 * maximum(volA)^2)
    ivkSVI = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi, logmoneynessA))
    rmseSVI = StatsBase.rmsd(volA, ivkSVI)
    xssvi = AQFED.VolatilityModels.calibrateXSSVISection(tte, forward, logmoneynessA, vols, w)
    ivkXSSVI = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(xssvi, logmoneynessA))
    rmseXSSVI = StatsBase.rmsd(volA, ivkXSSVI)
    strikeA = forward .* exp.(logmoneynessA)
    sumw2 = sum(weightsA .^ 2);w = weightsA ./ sqrt(sumw2);vols=volA;strikes=strikeA;
    prices, weights = Collocation.weightedPrices(true, strikeA, volA, weightsA, forward, 1.0, tte,vegaFloor=1e-8)
    strikesf, pricesf = AQFED.Collocation.filterConvexPrices(strikeA, prices, weights, forward,tol=1e-6)
    isoc, m = Collocation.makeIsotonicCollocation(strikesf, pricesf, weights, tte, forward, 1.0, deg=3, degGuess=3)
    sol3 = Collocation.Polynomial(isoc)
    ivk3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol3, true, strikeA, forward, 1.0), forward, strikeA, tte, 1.0);
   rmse3 = StatsBase.L2dist(w .* volA, w .* ivk3)
    isoc, m = Collocation.makeIsotonicCollocation(strikesf, pricesf, weights, tte, forward, 1.0, deg=5, degGuess=3, minSlope=1e-5)
    sol5 = Collocation.Polynomial(isoc)
    ivk5 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol5, true, strikeA, forward, 1.0), forward, strikeA, tte, 1.0);
    rmse5 = StatsBase.L2dist(w .* volA, w .* ivk5)
    isoc, m = Collocation.makeIsotonicCollocation(strikesf, pricesf, weights, tte, forward, 1.0, deg=9, degGuess=3, minSlope=1e-5)
    sol9 = Collocation.Polynomial(isoc)
    ivk9 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol9, true, strikeA, forward, 1.0), forward, strikeA, tte, 1.0);
    rmse9 = StatsBase.L2dist(w .* volA, w .* ivk9)
    bsple, m = Collocation.makeExpBSplineCollocation(
               strikes,
               prices,
               weights,
               tte,
               forward,
               1.0,
               penalty=0e-2,
               size=10,
               minSlope=1e-8,
               rawFit=false
           )
           ivkbe = @. Black.impliedVolatility(true, Collocation.priceEuropean(bsple, true, strikeA, forward, 1.0), forward, strikeA, tte, 1.0);
           rmsebe = StatsBase.L2dist(w .* volA, w .* ivkbe)
           bspl3, m = Collocation.makeBSplineCollocation(
            strikes,
            prices,
            weights,
            tte,
            forward,
            1.0,
            penalty=0e-2,
            size=10,
            minSlope=1e-8,
            rawFit=false,N=3
        )
        ivkb3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(bspl3, true, strikeA, forward, 1.0), forward, strikeA, tte, 1.0);
        rmseb3 = StatsBase.L2dist(w .* volA, w .* ivkb3)
        lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, model=PDDE.Quadratic(),location="Equidistributed",size=10,L=strikes[1],U=strikes[end])
        ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikeA), forward, strikeA, tte, 1.0);        rmseq = StatsBase.L2dist(w .* volA, w .* ivkq)
        lvgqe = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, model=PDDE.Quadratic(),location="Mid-XX",size=0,L=strikes[1],U=strikes[end])
        ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgqe, true, strikeA), forward, strikeA, tte, 1.0);
        rmseq = StatsBase.L2dist(w .* volA, w .* ivkq)
        solvers=["LM","LM-Curve","GN-LOG","GN-ALG","GN-MQ","LM-LOG","LM-ALG","LM-MQ"]
        for guess = ["Constant","Spline"]
        for solverName = solvers
            lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, model=PDDE.Quadratic(),location="Equidistributed",size=10,L=strikes[1],U=strikes[end],optimizer=solverName,guess=guess)
            ivkq = Black.impliedVolatility.(forward.<strikes, abs.(PDDE.priceEuropean.(lvgq, forward.<strikes, strikeA)), forward, strikeA, tte, 1.0);        rmseq = StatsBase.L2dist(w .* volA, w .* ivkq)
            elapsed = @belapsed PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, model=PDDE.Quadratic(),location="Equidistributed",size=10,L=strikes[1],U=strikes[end],optimizer=$solverName, guess=$guess)
            println(guess, " ",solverName, " ",rmseq," ",elapsed)
        end
    end
            #=   

     kFine = forward.*exp.(range(logmoneynessA[1],stop=logmoneynessA[end], length=1001));
  p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
 plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bsple,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Exp B-Spline")
 plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bspl3,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Cubic B-spline")
plot!(p3, size=(480,320))
savefig("/home/fabien/mypapers/eqd_book/spxw170324_170316_vol_bspl.pdf")
p4=plot(log.(kFine./forward), Collocation.density.(bsple,kFine),label="Exp B-spline",xlab="Log-moneyness",ylab="Probability density",color=2)
plot!(p4,log.(kFine./forward), Collocation.density.(bspl3,kFine),label="Cubic B-spline",color=3)
plot!(p4,yscale=:log10, legend=:topleft)
plot!(p4, size=(480,320))
savefig("/home/fabien/mypapers/eqd_book/spxw170324_170316_dens_bspl.pdf")
   
     p3=plot(logmoneynessA, volA.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, Collocation.priceEuropean.(sol3,true,kFine,forward,1.0), forward, (kFine), tte, 1.0) .* 100, label="Cubic collocation")
    plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, Collocation.priceEuropean.(sol5,true,kFine,forward,1.0), forward, kFine, tte, 1.0) .* 100, label="Quintic collocation")
    plot!(p3,size=(480,380))
    savefig(p3,"/home/fabien/mypapers/eqd_book/spxw170324_170316_vol_schaback.pdf")
    
    plot(log.(kFine./forward), Collocation.density.(sol3,kFine),label="Cubic collocation", yscale=:log10, xlab="Forward log-moneyness",ylab="Probability density")
    plot!(log.(kFine./forward), Collocation.density.(sol5,kFine),label="Quintic collocation")
    plot!(ylim=(1e-16,0.0),legend=:topleft)
plot!(size=(480,380))
    savefig("/home/fabien/mypapers/eqd_book/spxw170324_170316_dens_schaback.pdf")

      p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
 plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,PDDE.priceEuropean.(lvgq,true,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Quadratic LVG")
plot!(p3, size=(480,380))
savefig(p3,"/home/fabien/mypapers/eqd_book/spxw170324_170316_vol_lvgq.pdf")
p2 = plot(kFine,(PDDE.derivativePrice.(lvgqe,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgqe,true,kFine)).*10000, label="Mid-XX n=147",yscale=:log10, xlab="Forward log-moneyness",ylab="Probability density")
plot!(p2,kFine,(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label="Equidistributed n=10",yscale=:log10, xlab="Forward log-moneyness",ylab="Probability density")
plot!(p2,legend=:topleft)
plot!(p2, size=(480,380))
    savefig(p2,"/home/fabien/mypapers/eqd_book/spxw170324_170316_dens_lvgq.pdf")
   =#
   
    allStrikes = vcat(0.0, strikesf, forward*exp(logmoneynessA[end]*3))
    allPrices = vcat(forward, pricesf, 0.0)
    allWeights = vcat(sum(weights),weights,sum(weights))
    leftB = Math.FirstDerivativeBoundary(-1.0)
    rightB = Math.FirstDerivativeBoundary(0.0)
    cs = Math.makeConvexSchabackRationalSpline(allStrikes, allPrices, leftB, rightB, iterations=128)
    ivkScha = @. Black.impliedVolatility(
        true,
        cs(strikeA),
        forward,
        strikeA,
        tte,
        1.0,
    );
    rmseScha = StatsBase.rmsd(volA, ivkScha)
    allStrikes = vcat(0.0, strikes, forward*exp(logmoneynessA[end]*3))
    allPrices = vcat(forward, prices, 0.0)
    println("scha ", rmseScha)
    csf, rmseSchaFit = fitConvexSchabackRationalSpline(
        allStrikes, allPrices,
        allWeights,
        leftB,
        rightB;
        penalty = 1.0
    )
    ivkSchaFit = @. Black.impliedVolatility(
        true,
        csf(strikeA),
        forward,
        strikeA,
        tte,
        1.0,
    );
    rmseSchaFit = StatsBase.rmsd(ivkSchaFit, volA)
    println("schafit ", rmseSchaFit)
    
    rbf = AQFED.VolatilityModels.calibrateMultiquadric(tte, forward, logmoneynessA, volA, weightsA)
    rbf = AQFED.VolatilityModels.calibrateMultiquadric(tte, forward, logmoneynessA, volA, weightsA, knots=range(logmoneynessA[1], stop=logmoneynessA[end], length=8))
    ivkRBF = sqrt.(rbf.(logmoneynessA))
    rmseRBF = StatsBase.rmsd(volA, ivkRBF)
    pricesB, weightsB = Collocation.weightedPrices(true, strikeA, ivkRBF, weightsA, forward, 1.0, tte)
    fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikeA, pricesB, weightsB, λ=200.0, solver="GI")
    ivkFengler = @. Black.impliedVolatility(true, max.(fengler.(strikeA), 1e-16), forward, strikeA, tte, 1.0)
    rmseFengler = StatsBase.rmsd(volA, ivkFengler)
    #=
    p3=plot(logmoneynessA, volA.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(p3, kFine,sqrt.(rbf.(kFine)) .* 100, label="RBF")
    plot!(p3, kFine, Black.impliedVolatility.(true, max.(fengler.(forward.*exp.(kFine)),1e-16), forward, forward.*exp.(kFine), tte, 1.0) .* 100, label=string("Fengler λ=",200))
    plot!(p3,size=(480,380))
    savefig(p3,"/home/fabien/mypapers/eqd_book/spxw170324_170316_vol_fengler_rbf.pdf")
    p3 = plot(kFine, AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,forward.*exp.(kFine)),label="string("Fengler λ=",200))

    pdf(pp,z) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.Black.blackScholesFormula(true, y, forward, pp(log(y/forward))*tte,1.0,1.0),x),z)
    p3 =plot(kFine,@.(pdf(rbf, forward*exp(kFine))),label="RBF")

     plot!(kFine, AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,forward.*exp.(kFine)),label=string("Fengler λ=",200),xlab="Forward log-moneyness",ylab="Probability density")
    plot!(ylim=(1e-8,0.02),yscale=:log10)
    plot!(p3,size=(480,380))
    savefig(p3,"/home/fabien/mypapers/eqd_book/spxw170324_170316_dens_fengler_rbf.pdf")
##Schaback
   dev = 1.0
    kFine = range(logmoneynessA[1]*dev,stop=logmoneynessA[end]*dev, length=1001);
 
    p3=plot(logmoneynessA, volA.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(p3, kFine, Black.impliedVolatility.(true, max.(cs.(forward.*exp.(kFine)),1e-16), forward, forward.*exp.(kFine), tte, 1.0) .* 100, label=string("Schaback λ=",0))
    plot!(p3, kFine, Black.impliedVolatility.(true, max.(csf.(forward.*exp.(kFine)),1e-16), forward, forward.*exp.(kFine), tte, 1.0) .* 100, label=string("Schaback λ=",1))
    plot!(p3,size=(480,380))
    savefig(p3,"/home/fabien/mypapers/eqd_book/spxw170324_170316_vol_schaback.pdf")
    
    plot(logmoneynessA, AQFED.Math.evaluateSecondDerivative.(cs,strikeA),label="Schaback λ=0", yscale=:log10, xlab="Forward log-moneyness",ylab="Probability density")
    plot!(logmoneynessA, AQFED.Math.evaluateSecondDerivative.(csf,strikeA),label="Schaback λ=1", yscale=:log10, xlab="Forward log-moneyness",ylab="Probability density")
    plot!(ylim=(1e-16,0.0),legend=:topleft)
plot!(size=(480,380))
    savefig("/home/fabien/mypapers/eqd_book/spxw170324_170316_dens_schaback.pdf")

    =#
    λs = [1e-10, 100.0, 1000.0, 10000.0]
    results = []
    for λ in λs
        fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikeA, prices, weights, λ=λ, solver="GI")
        ivkFengler = @. Black.impliedVolatility(true, max.(fengler.(strikeA), 1e-16), forward, strikeA, tte, 1.0)
        rmseFengler = StatsBase.rmsd(volA, ivkFengler)
        println(λ, " ", rmseFengler)
        ivkFineFengler = @. Black.impliedVolatility(true, max.(fengler.(forward .* exp.(kFine)), 1e-16), forward, forward .* exp.(kFine), tte, 1.0)
        push!(results, (λ, fengler, ivkFengler, rmseFengler, ivkFineFengler))
    end
    #1e-10, 1.0, 10.0,100.0

    #=
    p3=plot(logmoneynessA, volA.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(logmoneynessA, sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi,logmoneynessA)) .* 100, label="SVI")
    plot!(logmoneynessA, sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi0,logmoneynessA)) .* 100, label="SVI a=0")
    plot!(logmoneynessA, ivkFengler .* 100, label="Fengler λ=1e-10")
    plot!(logmoneynessA, sqrt.(rbf.(logmoneynessA)).*100, label="RBF")
    dev = exp(volA[1]*sqrt(tte))
    dev = 1.0
    kFine = range(logmoneynessA[1]*dev,stop=logmoneynessA[end]*dev, length=1001);
    p3=plot(logmoneynessA, volA.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1, markeralpha=0.25); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    for (λ,fengler, ivkFengler, rmseFengler, ivkFineFengler) in results
        plot!(p3, kFine, Black.impliedVolatility.(true, max.(fengler.(forward.*exp.(kFine)),1e-16), forward, forward.*exp.(kFine), tte, 1.0) .* 100, label=string("Fengler λ=",λ))
    end
    plot(p3,margin=3Plots.mm,size=(800,380))
    savefig(p3,"/home/fabien/mypapers/eqd_book/spxw170324_170316_vol_fengler.pdf")

    ###density
    p3 = plot(kFine, AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,forward.*exp.(kFine)),label="Fengler")
    plot!(log.(kFine./forward),AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label="Fengler")
    plot!(log.(kFine./forward),Collocation.density.(sol,kFine),label="Quintic collocation")
    plot!(log.(kFine./forward),(AQFED.PDDE.derivativePrice.(lvg,true,kFine.+0.0001) .- AQFED.PDDE.derivativePrice.(lvg,true,kFine)).*10000, label="LVG")

    p3 = plot(xlabel="Forward log-moneyness", ylabel="Probability density")
    for (λ,fengler, ivkFengler, rmseFengler, ivkFineFengler) in results
       plot!(p3,kFine, AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,forward.*exp.(kFine)),label=string("Fengler λ=",λ))
    end
    plot!(p3, ylim=(1e-8,0.02),yscale=:continuous,legend=:topleft)
    plot!(p3,size=(480,380))
    savefig(p3, "/home/fabien/mypapers/eqd_book/spxw170324_170316_dens_fengler.pdf")
    plot!(p3, ylim=(1e-8,0.02),yscale=:log10,legend=:topleft)
    plot!(p3,size=(480,380))
    savefig(p3, "/home/fabien/mypapers/eqd_book/spxw170324_170316_dens_fengler_log.pdf")

    =#
kernel2 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=2)
    ivkMLN2= @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel2, strikes >= forward, strikes),
    forward,
                strikes,
                tte,
                1.0,
            );
           rmseMLN2 = StatsBase.L2dist(w .* ivkMLN2, w .* vols)
           kernel3 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=3)
           ivkMLN3= @. AQFED.Black.impliedVolatility(
                          strikes >= forward,
                          AQFED.VolatilityModels.priceEuropean.(kernel3, strikes >= forward, strikes),
                          forward,
                          strikes,
                          tte,
                          1.0,
                      );
                      rmseMLN3 = StatsBase.L2dist(w .* ivkMLN3, w .* vols)
           
kernel4 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=4)
ivkMLN4= @. AQFED.Black.impliedVolatility(
               true,
               AQFED.VolatilityModels.priceEuropean.(kernel4, true, strikes),
               forward,
               strikes,
               tte,
               1.0,
           );
           rmseMLN4 = StatsBase.L2dist(w .* ivkMLN4, w .* vols)
           kernel6 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=6)
ivkMLN6= @. AQFED.Black.impliedVolatility(
               true,
               AQFED.VolatilityModels.priceEuropean.(kernel6, true, strikes),
               forward,
               strikes,
               tte,
               1.0,
           );
           rmseMLN6 = StatsBase.L2dist(w .* ivkMLN6, w .* vols)

   #=
    plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel2,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 2")
    plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel4,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 4")
    plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel6,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 6")
 plot!(size=(800,320),margin=3Plots.mm)
  savefig("/home/fabien/mypapers/eqd_book/vol_spw_1m_050218_mln6.pdf")

plot(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel2,kFine),label="Mixture of 2", color=2,xlab="Forward log-moneyness",ylab="Probability density")
 plot!(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel4,kFine),label="Mixture of 4",color=3)
 plot!(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel6,kFine),label="Mixture of 6",color=4)
plot!(size=(800,320),margin=3Plots.mm)
    savefig("/home/fabien/mypapers/eqd_book/density_spw_1m_050218_mln6.pdf")
=  =#

##LVG vs AH
w1 = ones(length(prices));
ah = PDDE.calibrateDiscreteLogLVG(tte, forward, strikes, prices, w1, useVol=true, model=PDDE.Quadratic(),location="Equidistributed",optimizer="GN-MQ",L=strikes[1]/2,U=strikes[end]*2,isC3=true,size=10,discreteSize=1000)
lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, w1, useVol=true, model=PDDE.Quadratic(),location="Equidistributed",optimizer="LM-ALG",L=strikes[1]/2,U=strikes[end]*2,isC3=true,size=10)
ah100 = PDDE.calibrateDiscreteLogLVG(tte, forward, strikes, prices, w1, useVol=true, model=PDDE.Quadratic(),location="Equidistributed",optimizer="GN-MQ",L=strikes[1]/2,U=strikes[end]*2,isC3=true,size=10,discreteSize=101)
ah1000 = PDDE.calibrateDiscreteLogLVG(tte, forward, strikes, prices, w1, useVol=true, model=PDDE.Quadratic(),location="Equidistributed",optimizer="GN-MQ",L=strikes[1]/2,U=strikes[end]*2,isC3=true,size=10,discreteSize=1000)

#=
d2price(strike) = ForwardDiff.derivative(y -> ForwardDiff.derivative(x -> PDDE.priceEuropean(ah,true,x),y),strike)

plot(log.(kFine./forward),d2price.(kFine), label="AH-1000 LinearBlack", xlab="Forward log-moneyness",ylab="Probability density",ylims=(1e-10,2e-2),yscale=:log10)
plot!(log.(kFine./forward),(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label="LVG LinearBlack", xlab="Forward log-moneyness",ylab="Probability density",yscale=:log10,z_order=:back)

ah = ah100
plot(log.(kFine./forward),d2price.(kFine), label="Andreasen-Huge l=101", xlab="Forward log-moneyness",ylab="Probability density",ylims=(1e-10,1.0),yscale=:log10,z_order=:back)
plot!(log.(kFine./forward),(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label="LVG", xlab="Forward log-moneyness",ylab="Probability density",yscale=:log10,z_order=:back)
ah = ah1000
plot!(log.(kFine./forward),d2price.(kFine), label="Andreasen-Huge l=1000", xlab="Forward log-moneyness",ylab="Probability density",ylims=(1e-10,2e-2),yscale=:log10)
plot!(legend=:topleft)
 plot!(size=(480,380))
 savefig("/home/fabien/mypapers/eqd_book/density_spw_1m_050218ahspl_d.pdf")
 p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5)
plot!(p3,log.(kFine./forward), Black.impliedVolatility.(forward.<kFine, max.(1e-32,PDDE.priceEuropean.(lvgq,forward.<kFine,kFine)), forward, (kFine), tte, 1.0) .* 100, label="LVG")
 plot!(p3,log.(kFine./forward), Black.impliedVolatility.(forward.<kFine, max.(1e-32,PDDE.priceEuropean.(ah100,forward.<kFine,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Andreasen-Huge l=101")
plot!(p3,log.(kFine./forward), Black.impliedVolatility.(forward.<kFine, max.(1e-32,PDDE.priceEuropean.(ah1000,forward.<kFine,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Andreasen-Huge l=1000")
 plot!(p3,size=(480,380))
 savefig("/home/fabien/mypapers/eqd_book/vol_spw_1m_050218ahspl.pdf")

 plot(log.(kFine./forward), Black.impliedVolatility.(forward.<kFine, max.(1e-32,PDDE.priceEuropean.(ah100,forward.<kFine,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Andreasen-Huge l=101")
 plot!(log.(kFine./forward), Black.impliedVolatility.(forward.<kFine, max.(1e-32,PDDE.priceEuropean.(lvgq,forward.<kFine,kFine)), forward, (kFine), tte, 1.0) .* 100, label="LVG",xlab="Forward log-moneyness", ylab="Implied volatility in %",z_order=:back,linestyle=:dot)
 plot!(ylims=(5,50),xlim=(-0.3,0.12))
plot!(size=(480,380))
savefig("/home/fabien/mypapers/eqd_book/vol_spw_1m_050218ahspl_d100.pdf")

=#
end

@testset "deltavol" begin
    #AUDNZD 2014/07/02
    #exp 20140709
    tte = 7.0 / 365
    spot = 1.0784
    forward = 1.07845
    df = 0.999712587139
    rr10 = 0.35 / 100
    rr25 = 0.4 / 100
    volAtm = 5.14 / 100
    bf25 = 0.25 / 100
    bf10 = 1.175 / 100
    vol10Call = volAtm + bf10 + rr10 / 2
    vol10Put = volAtm + bf10 - rr10 / 2
    vol25Call = volAtm + bf25 + rr25 / 2
    vol25Put = volAtm + bf25 - rr25 / 2

    deltas = [0.1, 0.25, 0.5, 0.75, 0.9]
    vols = [vol10Call, vol25Call, volAtm, vol25Put, vol10Put]
# delta =Phi((log F/K +1/2 v^2)/v) == v*Phiinv(delta) = logF-logK + 1/2v^2

    # deltas = [0.05,0.25,0.5,0.75,0.95]    
    #  vols = [3.715,2.765,2.83,3.34,4.38]./100
    #vols = [4.0,3.03,3.06,3.65,4.790]./100
    # deltas = [0.1,0.25,0.5,0.75,0.9]
    # vols = [8.510, 8.210,8.257,8.835,9.589]./100
    #     0.09935431799838582, 9.589285714285715
    # 0.25044390637610986, 8.835714285714285
    # 0.5003228410008074, 8.257142857142856
    # 0.7495560936238905, 8.210714285714285
    # 0.9000000000000004, 8.510714285714286

    # forward = 1.0
    # tte = 31.0/365
    ppDelta = PPInterpolation.CubicSplineNatural(deltas, vols)
  
    #convert fwd deltas to moneyness
    k = zeros(Float64, length(deltas))
    for (i, delta) in enumerate(deltas)
        k[i] = forward * exp(-vols[i] * sqrt(tte) * norminv(delta) + vols[i]^2 * tte / 2)
    end
    reverse!(k)
    reverse!(vols)
    w1 = ones(length(k))
    prices, weights = Collocation.weightedPrices(true, k, vols, w1, forward, 1.0, tte)
    isoc, m = Collocation.makeIsotonicCollocation(k, prices, weights, tte, forward, 1.0, deg=3, degGuess=1)
    sol = Collocation.Polynomial(isoc)
    dev = exp(3 * vols[3] * sqrt(tte))
    ivk3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol, true, k, forward, 1.0), forward, k, tte, 1.0)
    kFine = collect(range(k[1] / dev, stop=k[end] * dev, length=201))
    ivkFine3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol, true, kFine, forward, 1.0), forward, kFine, tte, 1.0)
    rmse3 = StatsBase.rmsd(vols, ivk3)
    #    isoc, m = Collocation.makeIsotonicCollocation(k, prices, weights, tte, forward, 1.0, deg = 5, degGuess = 1) #strangely minSlope has big influence
    isoc, m = Collocation.makeIsotonicCollocation(k, prices, weights, tte, forward, 1.0, deg=5, degGuess=3, minSlope=1e-5)
    sol = Collocation.Polynomial(isoc)
    ivk5 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol, true, k, forward, 1.0), forward, k, tte, 1.0)
    rmse5 = StatsBase.rmsd(vols, ivk5)
    ivkFine5 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol, true, kFine, forward, 1.0), forward, kFine, tte, 1.0)
    bsple, m = Collocation.makeExpBSplineCollocation(
        k,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=0,
        minSlope=1e-8,
        rawFit=true,
    )
    ivkexp = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(bsple, true, k, forward, 1.0),
        forward,
        k,
        tte,
        1.0,
    )
    rmseexpe = StatsBase.rmsd(ivkexp, vols)

    bsplee, m = Collocation.makeExpBSplineCollocation(
        k,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=length(k),
        minSlope=1e-8,
        rawFit=true,
    )
    ivstrikesbe = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(bsplee, true, k, forward, 1.0),
        forward,
        k,
        tte,
        1.0,
    )
    rmsebe = StatsBase.rmsd(ivstrikesbe, vols)
    println("bsple ", rmsebe)
    bspl3, m = Collocation.makeBSplineCollocation(
        k,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=0,
        minSlope=1e-8,
        rawFit=true,N=3
    )
    ivstrikesb3 = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(bspl3, true, k, forward, 1.0),
        forward,
        k,
        tte,
        1.0,
    )
    rmseb3 = StatsBase.rmsd(ivstrikesb3, vols)
    println("bspl3 ", rmseb3)
    bspl3e, m = Collocation.makeBSplineCollocation(
        k,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=length(k),
        minSlope=1e-8,
        rawFit=true,N=3
    )
    ivstrikesb3 = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(bspl3e, true, k, forward, 1.0),
        forward,
        k,
        tte,
        1.0,
    )
    rmseb3 = StatsBase.rmsd(ivstrikesb3, vols)
    println("bspl3e ", rmseb3)
    #===
    p3=plot(log.(k./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=1.0); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
 plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bsple,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Exp B-spline")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bspl3,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Cubic B-spline")
plot!(p3, legend=:top,size=(480,320))
savefig("/home/fabien/mypapers/eqd_book/audnzd_bspl_vol.pdf")
 p4=plot(log.(kFine), Collocation.density.(bsplee,kFine),label="Exp B-spline equidistant",xlab="Log-moneyness",ylab="Probability density")
 plot!(p4,log.(kFine), Collocation.density.(bsple,kFine),label="Exp B-spline")
 plot!(p4,log.(kFine), Collocation.density.(bspl3,kFine),label="Cubic B-spline")
plot!(p4,log.(kFine), Collocation.density.(bspl2,kFine),label="Quadratic B-spline")
 plot!(p4, size=(480,320))
savefig("/home/fabien/mypapers/eqd_book/audnzd_bspl_dens.pdf")
==#
lvgq = PDDE.calibrateQuadraticLVG(tte, forward, k, prices, weights, useVol=false, model=PDDE.Quadratic(),location="Mid-XX",size=0,L=k[1]/2,U=k[end]*2)
ivkLVG = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, k), forward, k, tte, 1.0)
rmseLVG = StatsBase.rmsd(vols, ivkLVG)

    #=

p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,PDDE.priceEuropean.(lvgq,true,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Quadratic LVG")
plot!(p3, size=(480,320))
 savefig("/home/fabien/mypapers/eqd_book/audnzd_lvgq_vol.pdf")
p2 = plot(log.(kFine./forward),(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label=:none, xlab="Forward log-moneyness",ylab="Probability density")
 plot!(p2, size=(480,320))
 savefig("/home/fabien/mypapers/eqd_book/audnzd_lvgq_dens.pdf")
=#

    pp = PPInterpolation.CubicSplineNatural(log.(k ./ forward), vols .^ 2)
    pp = PPInterpolation.CubicSplineNatural(k, vols)
    deltaFine = reverse(range(0.001, stop=0.999, length=1001))
    kDeltaFine = @. forward * exp(-ppDelta(deltaFine) * sqrt(tte) * norminv(deltaFine) + ppDelta(deltaFine)^2 * tte / 2)
    ppDeltaStrike = PPInterpolation.CubicSplineNatural(kDeltaFine, ppDelta.(deltaFine))
    lvg = PDDE.calibrateLinearBlackLVG(tte, forward, k, prices, weights, useVol=false, L=k[1] / 2, U=k[end] * 2)
    ivkLVG = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvg, true, k), forward, k, tte, 1.0)
    rmseLVG = StatsBase.rmsd(vols, ivkLVG)
    ivkFineLVG = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvg, true, kFine), forward, kFine, tte, 1.0)
    fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, k, prices, weights, λ=1e-10, solver="GI")
    ivkFengler = @. Black.impliedVolatility(true, max.(fengler.(k), 1e-16), forward, k, tte, 1.0)
    rmseFengler = StatsBase.rmsd(vols, ivkFengler)
    ivkFineFengler = @. Black.impliedVolatility(true, max.(fengler.(kFine), 1e-16), forward, kFine, tte, 1.0)
    svi, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, log.(k ./ forward), vols, ones(length(vols)), aMin=-2 * maximum(vols)^2)
    ivkSVI = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi, log.(k ./ forward)))
    rmseSVI = StatsBase.rmsd(vols, ivkSVI)
    ivkFineSVI = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi, log.(kFine ./ forward)))
    svi0, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, log.(k ./ forward), vols, ones(length(vols)), aMin=0.0)
    ivkSVI0 = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi0, log.(k ./ forward)))
    rmseSVI0 = StatsBase.rmsd(vols, ivkSVI0)
    ivkFineSVI0 = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi0, log.(kFine ./ forward)))
 
    ivkXSSVI = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(xssvi, log.(k ./ forward)))
    rmseXSSVI = StatsBase.rmsd(vols, ivkXSSVI)
   
    weightsA = ones(length(vols));strikes=k;
    sumw2 = sum(weightsA .^ 2);w = weightsA ./ sqrt(sumw2)


    
kernel2 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=2)
    ivkMLN2= @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel2, strikes >= forward, strikes),
    forward,
                strikes,
                tte,
                1.0,
            );
           rmseMLN2 = StatsBase.L2dist(w .* ivkMLN2, w .* vols)
           kernel3 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=3)
           ivkMLN3= @. AQFED.Black.impliedVolatility(
                          strikes >= forward,
                          AQFED.VolatilityModels.priceEuropean.(kernel3, strikes >= forward, strikes),
                          forward,
                          strikes,
                          tte,
                          1.0,
                      );
                      rmseMLN3 = StatsBase.L2dist(w .* ivkMLN3, w .* vols)
                     kernel3f = AQFED.VolatilityModels.calibrateLognormalMixtureFixedWeights(tte, forward, strikes, prices, weights, useVol=false,α=ones(3)/3)
                    ivkMLN3f= @. AQFED.Black.impliedVolatility(
                                                    strikes >= forward,
                                                    AQFED.VolatilityModels.priceEuropean.(kernel3f, strikes >= forward, strikes),
                                                    forward,
                                                    strikes,
                                                    tte,
                                                    1.0,
                   rmseMLN3f = StatsBase.L2dist(w .* ivkMLN3f, w .* vols)
                       #=
 plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel2,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 2")
plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel3f,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 3 with equal α")
plot!(size=(480,380),legend=:top)
    savefig("/home/fabien/mypapers/eqd_book/audnzd_mixture_vols.pdf")
 plot(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel2,kFine),label="Mixture of 2",color=2,xlabel="Forward log-moneyness",ylabel="Probability density")
 plot!(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel3,kFine),label="Mixture of 3 with equal α",color=3,xlabel="Forward log-moneyness",ylabel="Probability density")
 plot!(size=(480,380))
savefig("/home/fabien/mypapers/eqd_book/audnzd_mixture_dens.pdf")
 

           =#


    #Quintic
    #pqDeltas = BSplineKit.interpolate(deltas, vols, BSplineOrder(6))
    #pqDeltaStrike = PPInterpolation.CubicSplineNatural(kDeltaFine, pqDelta.(deltaFine))
  
    #TODO cubic spline in delta? meaning we first sample and then compute eqv k,
    #=  
    p1 = plot(deltas.*100, reverse(vols).*100,xlab="Call Δ in %", seriestype= :scatter, ylab="Volatility in %", label="",xflip=true)# label="Market quote")
    plot!(p1, deltaFine.*100, ppDeltaStrike.(kDeltaFine).*100, label="") #,label="Cubic spline on Δ")
    plot!(p1, deltaFine.*100, pp.(kDeltaFine).*100,label="") #, label="cubic spline on strike"
    plot!(p1,ylims=(5,8))

    p2 = plot(log.(k./forward), vols.*100, seriestype= :scatter, label="",xlabel="Forward log-moneyness",ylabel="Volatility in %")
    plot!(p2, log.(kFine./forward), ppDeltaStrike.(kFine).* 100,label="") #,label="Cubic spline on Δ")
    plot!(p2, log.(kFine./forward), pp.(kFine).* 100, label="") #,label="Cubic spline on strike")
plot!(p2,ylims=(5,8))

    p3= plot(log.(kFine./forward), pdf.(ppDeltaStrike,kFine),label="Cubic spline on Δ",color=2)
 plot!(p3, log.(kFine./forward), pdf.(pp,kFine),label="Cubic spline on strikes",color=3)
plot!(p3, xlab="Forward log-moneyness", ylab="Probability density")
l=@layout [a b ; c]
plot(p1,p2,p3,layout=l,size=(800,600))
savefig("/home/fabien/mypapers/eqd_book/audnzd_delta_spline_dens.pdf")

plot(log.(k./forward), vols.*100, seriestype= :scatter, label="Reference"); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(kFine, ivkFine3 .*100 ,label="Cubic collocation")
    plot!(kFine, ivkFine5 .*100 ,label="Quintic collocation")
    plot!(kFine, ivkFineexp.*100,label="Exp. B-spline collocation")
    plot!(kFine, ivkFineLVG .*100 ,label="LVG")
    plot!(kFine, ivkFineSVI .* 100, label="SVI")


    =#


    #pdf(z) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.Black.blackScholesFormula(true, y, forward, pp(log(y/forward))*tte,1.0,1.0),x),z)
    pdf(pp, z) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.Black.blackScholesFormula(true, y, forward, pp(y)^2 * tte, 1.0, 1.0), x), z)
    pdfsvi(svi, z) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.Black.blackScholesFormula(true, y, forward, AQFED.TermStructure.varianceByLogmoneyness(svi, log(y / forward)) * tte, 1.0, 1.0), x), z)
    #=
    plot(log.(kFine./forward), pdf.(ppDeltaStrike,kFine),label="Cubic spline on Δ")
    plot!(log.(kFine./forward),Collocation.density.(sol,kFine),label="Quintic collocation")
    plot!(log.(kFine./forward),(AQFED.PDDE.derivativePrice.(lvg,true,kFine.+0.0001) .- AQFED.PDDE.derivativePrice.(lvg,true,kFine)).*10000, label="LVG")
    plot!(log.(kFine./forward),(AQFED.PDDE.derivativePrice.(lvg,true,kFine.+0.0001) .- AQFED.PDDE.derivativePrice.(lvg,true,kFine)).*10000, label="LVG")
    plot!(log.(kFine./forward),AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label="Fengler")

    =#
    allStrikes = vcat(0.0, k, k[end]*2)
    allPrices = vcat(forward, prices, 0.0)
    leftB = Math.FirstDerivativeBoundary(-1.0)
    rightB = Math.FirstDerivativeBoundary(0.0)
    cs = Math.makeConvexSchabackRationalSpline(allStrikes, allPrices, leftB, rightB, iterations=128)
    ivstrikes = @. Black.impliedVolatility(
        true,
        cs(k),
        forward,
        k,
        tte,
        1.0,
    )
    rmse = StatsBase.rmsd(ivstrikes, vols)
end
@testset "jaeckel1" begin
    strikes = [
        0.035123777453185,
        0.049095433048156,
        0.068624781300891,
        0.095922580089594,
        0.134078990076508,
        0.18741338653678,
        0.261963320525776,
        0.366167980681693,
        0.511823524787378,
        0.715418426368358,
        1.0,
        1.39778339939642,
        1.95379843162821,
        2.73098701349666,
        3.81732831143284,
        5.33579814376678,
        7.45829006788743,
        10.4250740447762,
        14.5719954372667,
        20.3684933182917,
        28.4707418310251,
    ]
    vols = [
        0.642412798191439,
        0.621682849924325,
        0.590577891369241,
        0.553137221952525,
        0.511398042127817,
        0.466699250819768,
        0.420225808661573,
        0.373296313420122,
        0.327557513727855,
        0.285106482185545,
        0.249328882881654,
        0.228967051575314,
        0.220857187809035,
        0.218762825294675,
        0.218742183617652,
        0.218432406892364,
        0.217198426268117,
        0.21573928902421,
        0.214619929462215,
        0.2141074555437,
        0.21457985392644,
    ]
    forward = 1.0
    tte = 5.07222222222222
    w1 = ones(length(strikes))
    prices, weights = Collocation.weightedPrices(true, strikes, vols, w1, forward, 1.0, tte, vegaFloor=1e-8)
    fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (weights), λ=60, eps=1e-13, solver="GI")
    ivstrikesFengler = @. Black.impliedVolatility(
        true,
        fengler(strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseFengler = StatsBase.rmsd(ivstrikesFengler, vols)
    println("Fengler ", rmseFengler)
    kFine = exp.(range(log(strikes[1]), stop=log(strikes[end]), length=501))

    allStrikes = vcat(0.0, strikes, 50.0)
    allPrices = vcat(forward, prices, 0.0)
    leftB = Math.FirstDerivativeBoundary(-1.0)
    rightB = Math.FirstDerivativeBoundary(0.0)
    cs = Math.makeConvexSchabackRationalSpline(allStrikes, allPrices, leftB, rightB, iterations=128)
    ivstrikes = @. Black.impliedVolatility(
        true,
        cs(strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmse = StatsBase.rmsd(ivstrikes, vols)
    println("scha ", rmse)
    pp = PPInterpolation.CubicSplineNatural(log.(strikes), vols .^ 2)
    rbf = AQFED.VolatilityModels.calibrateMultiquadric(tte, forward, log.(strikes), vols, w1, knots=log.(strikes))

    lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, weights, useVol=true, L=strikes[1] / 2, U=strikes[end] * 2)
    ivstrikes = @. Black.impliedVolatility(
        true,
        PDDE.priceEuropean(lvg, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmse = StatsBase.rmsd(ivstrikes, vols)
    println("LVG-Black ", rmse)

#for linear, use isC3=false as forward is part of the strikes and there is no extra param.
lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, model=PDDE.Quadratic(),location="Mid-XX",size=0,L=strikes[1]/2,U=strikes[end]*2)
ivkLVG = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0)
rmseLVG = StatsBase.rmsd(vols, ivkLVG)

    #=

p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,PDDE.priceEuropean.(lvgq,true,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Quadratic LVG")
plot!(p3, ylim=(21,25),xlim=(-0.1,3.6), size=(480,320))
 savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_i_lvgq_vol.pdf")
p2 = plot(log.(kFine./forward),(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label=:none, xlab="Forward log-moneyness",ylab="Probability density")
 plot!(p2, size=(480,320))
 savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_i_lvgq_dens.pdf")
=#


    #= 
    pdf(pp,z) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.Black.blackScholesFormula(true, y, forward, pp(log(y/forward))*tte,1.0,1.0),x),z)

    p1=plot(log.(kFine./forward),AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label="Fengler",color=2, ylim=(-0.05,1.0))
    p2 = plot(log.(kFine./forward),AQFED.Math.evaluateSecondDerivative.(cs,kFine),label="Schaback",color=3,ylim=(-0.05,1.0))
    p3 = plot(log.(kFine./forward),pdf.(pp,kFine),label="Cubic spline on implied variances",color=1,ylim=(-0.05,1.0))
    p4 = plot(log.(kFine./forward),pdf.(rbf,kFine),label="RBF",color=4,ylim=(-0.05,1.0))
    plot(p3,p4,p1,p2, layout=(1,4),legend=false,titles=["Cubic spline" "RBF" "Fengler" "Schaback"],size=(800,250))
    savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_i_fengler_rbf_dens.pdf")

    plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, max.(fengler.(kFine),1e-16), forward, kFine, tte, 1.0)*100),label="Fengler λ=1e-13")
    plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, max.(cs.(kFine),1e-16), forward, kFine, tte, 1.0)*100),label="Schaback")
    plot!(log.(kFine./forward), sqrt.(rbf.(log.(kFine./forward))).*100,label="RBF")
    #plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, PDDE.priceEuropean.(lvg,true,kFine), forward, kFine, tte, 1.0)*100),label="LVG Linear Black")
    plot!(ylim=(21,25),xlim=(-0.1,3.1),size=(640,320),margin=2Plots.mm)
    savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_i_fengler_rbf.pdf")

    =#

    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, weights, tte, forward, 1.0, deg=7, degGuess=1)
    sol = Collocation.Polynomial(isoc)
    ivstrikes = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(sol, true, strikes, forward, 1.0),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmse = StatsBase.rmsd(ivstrikes, vols)
    println("poly ", rmse)
    bsple, m = Collocation.makeExpBSplineCollocation(
        strikes,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=0,
        minSlope=1e-8,
        rawFit=true,
    )
    ivstrikesbe = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(bsple, true, strikes, forward, 1.0),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmsebe = StatsBase.rmsd(ivstrikesbe, vols)
    println("bsple ", rmsebe)
    bspl2, m = Collocation.makeBSplineCollocation(
        strikes,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=0,
        minSlope=1e-8,
        rawFit=true,
    )
    ivstrikesb2 = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(bspl2, true, strikes, forward, 1.0),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseb2 = StatsBase.rmsd(ivstrikesb2, vols)
    println("bspl2 ", rmseb2)
    bspl3, m = Collocation.makeBSplineCollocation(
        strikes,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=0,
        minSlope=1e-8,
        rawFit=true,N=3
    )
    ivstrikesb3 = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(bspl3, true, strikes, forward, 1.0),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseb3 = StatsBase.rmsd(ivstrikesb3, vols)
    println("bspl3 ", rmseb3)
    #===
    p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=1.0); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
 plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bsple,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Exponential B-spline")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bspl3,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Cubic B-spline")
plot!(p3, legend=:top,size=(400,200))
savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_ii_bspl_vol.pdf")
 p4=plot(log.(kFine), Collocation.density.(bsple,kFine),label="Exponential B-spline",xlab="Log-moneyness",ylab="Probability density",color=2)
 plot!(p4,log.(kFine), Collocation.density.(bspl3,kFine),label="Cubic B-spline",color=3)
plot!(p4, size=(400,200))
savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_ii_bspl_dens.pdf")
==#
  

    allStrikes = vcat(0.0, strikes, 50.0)
    allPrices = vcat(forward, prices, 0.0)
    leftB = Math.FirstDerivativeBoundary(-1.0)
    rightB = Math.FirstDerivativeBoundary(0.0)
    cs = Math.makeConvexSchabackRationalSpline(allStrikes, allPrices, leftB, rightB, iterations=128)
    ivstrikes = @. Black.impliedVolatility(
        true,
        cs(strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmse = StatsBase.rmsd(ivstrikes, vols)
    println("scha ", rmse)

    prices, weights = Collocation.weightedPrices(true, strikes, vols, w1, forward, 1.0, tte, vegaFloor=1e-8)
    sumw2 = sum(w1 .^ 2);w = w1 ./ sqrt(sumw2);
kernel2 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=2)
    ivkMLN2= @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel2, strikes >= forward, strikes),
    forward,
                strikes,
                tte,
                1.0,
            );
           rmseMLN2 = StatsBase.L2dist(w .* ivkMLN2, w .* vols)
           kernel3 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=3)
           ivkMLN3= @. AQFED.Black.impliedVolatility(
                          strikes >= forward,
                          AQFED.VolatilityModels.priceEuropean.(kernel3, strikes >= forward, strikes),
                          forward,
                          strikes,
                          tte,
                          1.0,
                      );
                      rmseMLN3 = StatsBase.L2dist(w .* ivkMLN3, w .* vols)
           
kernel4 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=4)
ivkMLN4= @. AQFED.Black.impliedVolatility(
    strikes >= forward,
               AQFED.VolatilityModels.priceEuropean.(kernel4, strikes >= forward, strikes),
               forward,
               strikes,
               tte,
               1.0,
           );
           rmseMLN4 = StatsBase.L2dist(w .* ivkMLN4, w .* vols)
           kernel4v = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=true,size=4)
           ivkMLN4v= @. AQFED.Black.impliedVolatility(
            strikes >= forward,
                          AQFED.VolatilityModels.priceEuropean.(kernel4v, strikes >= forward, strikes),
                          forward,
                          strikes,
                          tte,
                          1.0,
                      );
                      rmseMLN4v = StatsBase.L2dist(w .* ivkMLN4v, w .* vols)
           
           kernel6 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=6)
ivkMLN6= @. AQFED.Black.impliedVolatility(
    strikes >= forward,
               AQFED.VolatilityModels.priceEuropean.(kernel6, strikes >= forward, strikes),
               forward,
               strikes,
               tte,
               1.0,
           );
           rmseMLN6 = StatsBase.L2dist(w .* ivkMLN6, w .* vols)
        

           #=
 plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel4,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 4 price")
plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel4v,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 4 vol")


    #plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, PDDE.priceEuropean.(lvg,true,kFine), forward, kFine, tte, 1.0)*100),label="LVG Linear Black")
    plot!(ylim=(21,25),xlim=(-0.1,3.1),size=(640,320),margin=2Plots.mm)
    savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_i_fengler_rbf.pdf")

           =#
end

@testset "jaeckel2" begin
    strikes = [
        0.035123777453185,
        0.049095433048156,
        0.068624781300891,
        0.095922580089594,
        0.134078990076508,
        0.18741338653678,
        0.261963320525776,
        0.366167980681693,
        0.511823524787378,
        0.715418426368358,
        1.0,
        1.39778339939642,
        1.95379843162821,
        2.73098701349666,
        3.81732831143284,
        5.33579814376678,
        7.45829006788743,
        10.4250740447762,
        14.5719954372667,
        20.3684933182917,
        28.4707418310251,
    ]
    vols = [0.649712512502887,
        0.629372247414191,
        0.598339248024188,
        0.560748840467284,
        0.518685454812697,
        0.473512707134552,
        0.426434688827871,
        0.378806875802102,
        0.332366264644264,
        0.289407658380454,
        0.253751752243855,
        0.235378088110653,
        0.235343538571543,
        0.260395028879884,
        0.31735041252779,
        0.368205175099723,
        0.417582432865276,
        0.46323707706565,
        0.504386489988866,
        0.539752566560924,
        0.566370957381163]

    forward = 1.0
    tte = 5.07222222222222
    w1 = ones(length(strikes))
    prices, weights = Collocation.weightedPrices(true, strikes, vols, w1, forward, 1.0, tte, vegaFloor=1e-5)

    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, sqrt.(weights), tte, forward, 1.0, deg=7, degGuess=1)
    sol = Collocation.Polynomial(isoc)
    ivstrikes = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(sol, true, strikes, forward, 1.0),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmse = StatsBase.rmsd(ivstrikes, vols)
    println("poly ", rmse)
    bsple, m = Collocation.makeExpBSplineCollocation(
        strikes,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=0,
        minSlope=1e-8,
        rawFit=true,
    )
    ivstrikesbe = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(bsple, true, strikes, forward, 1.0),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmsebe = StatsBase.rmsd(ivstrikesbe, vols)
    println("bsple ", rmsebe)
    bspl2, m = Collocation.makeBSplineCollocation(
        strikes,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=0,
        minSlope=1e-8,
        rawFit=true,
    )
    ivstrikesb2 = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(bspl2, true, strikes, forward, 1.0),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseb2 = StatsBase.rmsd(ivstrikesb2, vols)
    println("bspl2 ", rmseb2)
    bspl3, m = Collocation.makeBSplineCollocation(
        strikes,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=0,
        minSlope=1e-8,
        rawFit=true,N=3
    )
    ivstrikesb3 = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(bspl3, true, strikes, forward, 1.0),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseb3 = StatsBase.rmsd(ivstrikesb3, vols)
    println("bspl3 ", rmseb3)
    #===
    p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=1.0); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
 plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bsple,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Exp B-spline")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bspl3,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Cubic B-spline")
plot!(p3, legend=:top,size=(480,320))
savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_ii_bspl_vol.pdf")
 p4=plot(log.(kFine), Collocation.density.(bsple,kFine),label="Exp B-spline",xlab="Log-moneyness",ylab="Probability density",color=2)
 plot!(p4,log.(kFine), Collocation.density.(bspl3,kFine),label="Cubic B-spline",color=3)
plot!(p4, size=(480,320))
savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_ii_bspl_dens.pdf")
==#
    allStrikes = vcat(0.0, strikes, 50.0)
    allPrices = vcat(forward, prices, 0.0)
    leftB = Math.FirstDerivativeBoundary(-1.0)
    rightB = Math.FirstDerivativeBoundary(0.0)
    cs = Math.makeConvexSchabackRationalSpline(allStrikes, allPrices, leftB, rightB, iterations=128)
    ivstrikes = @. Black.impliedVolatility(
        true,
        cs(strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmse = StatsBase.rmsd(ivstrikes, vols)
    println("scha ", rmse)
    lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, weights, useVol=true, L=strikes[1] / 2, U=strikes[end] * 2)
    ivstrikes = @. Black.impliedVolatility(
        true,
        PDDE.priceEuropean(lvg, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmse = StatsBase.rmsd(ivstrikes, vols)
    println("LVG-Black ", rmse)

lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, model=PDDE.Quadratic(),location="Mid-XX",size=0,L=k[1],U=k[end])
ivkLVG = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0);rmseLVG = StatsBase.rmsd(vols, ivkLVG)

    #=

p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,PDDE.priceEuropean.(lvgq,true,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Quadratic LVG")
plot!(p3, size=(480,320))
 savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_ii_lvgq_vol.pdf")
p2 = plot(log.(kFine./forward),(PDDE.derivativePrice.(lvg,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvg,true,kFine)).*10000, label="Linear-Black", xlab="Forward log-moneyness",ylab="Probability density")
plot!(log.(kFine./forward),(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label="Quadratic", xlab="Forward log-moneyness",ylab="Probability density")
 plot!(p2, size=(480,320),scale=:log10)
 savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_ii_lvgq_dens_log.pdf")
=#
    prices, weights = Collocation.weightedPrices(true, strikes, vols, w1, forward, 1.0, tte, vegaFloor=1e-7)
    sumw2 = sum(w1 .^ 2);w = w1 ./ sqrt(sumw2);
kernel2 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=2)
    ivkMLN2= @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel2, strikes >= forward, strikes),
    forward,
                strikes,
                tte,
                1.0,
            );
           rmseMLN2 = StatsBase.L2dist(w .* ivkMLN2, w .* vols)
           kernel3 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=3)
           ivkMLN3= @. AQFED.Black.impliedVolatility(
                          strikes >= forward,
                          AQFED.VolatilityModels.priceEuropean.(kernel3, strikes >= forward, strikes),
                          forward,
                          strikes,
                          tte,
                          1.0,
                      );
                      rmseMLN3 = StatsBase.L2dist(w .* ivkMLN3, w .* vols)
           
kernel4 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=4)
ivkMLN4= @. AQFED.Black.impliedVolatility(
    strikes >= forward,
               AQFED.VolatilityModels.priceEuropean.(kernel4, strikes >= forward, strikes),
               forward,
               strikes,
               tte,
               1.0,
           );
           rmseMLN4 = StatsBase.L2dist(w .* ivkMLN4, w .* vols)
           kernel4v = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=true,size=4)
           ivkMLN4v= @. AQFED.Black.impliedVolatility(
            strikes >= forward,
                          AQFED.VolatilityModels.priceEuropean.(kernel4v, strikes >= forward, strikes),
                          forward,
                          strikes,
                          tte,
                          1.0,
                      );
                      rmseMLN4v = StatsBase.L2dist(w .* ivkMLN4v, w .* vols)
           
           kernel6 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=6)
ivkMLN6= @. AQFED.Black.impliedVolatility(
    strikes >= forward,
               AQFED.VolatilityModels.priceEuropean.(kernel6, strikes >= forward, strikes),
               forward,
               strikes,
               tte,
               1.0,
           );
           rmseMLN6 = StatsBase.L2dist(w .* ivkMLN6, w .* vols)
           #=
 plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel4,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 4")
plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel6,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 6")
plot!(size=(480,380))
    savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_ii_mixture_vol.pdf")


    #plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, PDDE.priceEuropean.(lvg,true,kFine), forward, kFine, tte, 1.0)*100),label="LVG Linear Black")
    plot!(ylim=(21,25),xlim=(-0.1,3.1),size=(640,320),margin=2Plots.mm)
    plot(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel4,kFine),label="Mixture of 4",color=2,xlabel="Forward log-moneyness",ylabel="Probability density")
 plot!(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel6,kFine),label="Mixture of 6",color=3,xlabel="Forward log-moneyness",ylabel="Probability density")
 plot!(ylims=(0.0,2.0))
plot!(size=(480,380))
    savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_ii_mixture_dens.pdf")

           =#
        

end

@testset "AAPL4days" begin
    forward = 1.0
    tte = 4 / 365
    strikes = [0.8489536621823619,
        0.8674389636273045,
        0.877204783258595,
        0.8873193821624314,
        0.8963876432486297,
        0.9061534628799202,
        0.9159192825112106,
        0.9246387643248628,
        0.9337070254110611,
        0.9434728450423515,
        0.9528898854010959,
        0.9623069257598402,
        0.9720727453911306,
        0.9811410064773288,
        0.9909068261086194,
        0.9999750871948175,
        1.009392127553562,
        1.0188091679123064,
        1.0282262082710507,
        1.037992027902341,
        1.0467115097159931,
        1.0568261086198296,
        1.0655455904334818,
        1.0756601893373183,
        1.0850772296960627,
        1.094494270054807,
        1.1039113104135514,
        1.1129795714997497,
        1.1230941704035864,
        1.1328599900348766,
        1.1415794718485288,
        1.1509965122072732,
        1.170179372197308] #moneyness
    vols = [0.7431893004115226,
        0.7154115226337449,
        0.7123251028806584,
        0.6993621399176955,
        0.686604938271605,
        0.6833127572016462,
        0.6761111111111112,
        0.6754938271604939, 0.6787860082304528, 0.67940329218107, 0.6872222222222223, 0.6919547325102882, 0.6993621399176955, 0.7028600823045268, 0.7040946502057613, 0.6995679012345679, 0.6931893004115227, 0.6802263374485598, 0.6664403292181071, 0.6610905349794239, 0.6485390946502059, 0.6421604938271606, 0.639485596707819, 0.6417489711934158, 0.6456584362139919, 0.6425720164609054, 0.6485390946502059, 0.6555349794238684, 0.6639711934156379, 0.6783744855967079, 0.6870164609053498, 0.6903086419753087,
        0.7080041152263374]
    weightsV = ones(length(vols))
    logmoneynessA = log.(strikes ./ forward)
    sumw2 = sum(weightsV .^ 2)
    w = weightsV ./ sqrt(sumw2)
    svi0, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logmoneynessA, vols, w, aMin=0.0)
    ivkSVI0 = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi0, logmoneynessA))
    rmseSVI0 = StatsBase.L2dist(w .* vols, w .* ivkSVI0)
    xssvi = AQFED.VolatilityModels.calibrateXSSVISection(tte, forward, logmoneynessA, vols, w)
    ivkXSSVI = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(xssvi, logmoneynessA))
    rmseXSSVI = StatsBase.L2dist(w .* vols, w .* ivkSVI0)
    prices, weights = Collocation.weightedPrices(true, strikes, vols, w, forward, 1.0, tte, vegaFloor=1e-5)
   
    bsple, m = Collocation.makeExpBSplineCollocation(
        strikes,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=10,
        minSlope=1e-8,
        rawFit=false
    )
    ivkbe = @. Black.impliedVolatility(true, Collocation.priceEuropean(bsple, true, strikes, forward, 1.0), forward, strikes, tte, 1.0);
    rmsebe = StatsBase.L2dist(w .* vols, w .* ivkbe)
    bspl3, m = Collocation.makeBSplineCollocation(
     strikes,
     prices,
     weights,
     tte,
     forward,
     1.0,
     penalty=0e-2,
     size=10,
     minSlope=1e-8,
     rawFit=false,N=3
 )
 ivkb3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(bspl3, true, strikes, forward, 1.0), forward, strikes, tte, 1.0);
 rmseb3 = StatsBase.L2dist(w .* vols, w .* ivkb3)

#== 
kFine = forward.*exp.(range(logmoneynessA[1],stop=logmoneynessA[end], length=1001));
p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bsple,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Exp B-Spline")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bspl3,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Cubic B-spline")
plot!(p3, size=(480,320))
savefig("/home/fabien/mypapers/eqd_book/aapl_20131028_vol_bspl.pdf")
p4=plot(log.(kFine./forward), Collocation.density.(bsple,kFine),label="Exp B-spline",xlab="Log-moneyness",ylab="Probability density",color=2)
plot!(p4,log.(kFine./forward), Collocation.density.(bspl3,kFine),label="Cubic B-spline",color=3)
#plot!(p4,yscale=:log10)
plot!(p4, size=(480,320))
savefig("/home/fabien/mypapers/eqd_book/aapl_20131028_dens_bspl.pdf")
==#
    strikesf, pricesf = AQFED.Collocation.filterConvexPrices(strikes, prices, weights, forward,tol=1e-6)
    allStrikes = vcat(0.0, strikesf, forward*exp(logmoneynessA[end]*3))
    allPrices = vcat(forward, pricesf, 0.0)
    allWeights = vcat(sum(weights),weights,sum(weights))
    leftB = Math.FirstDerivativeBoundary(-1.0)
    rightB = Math.FirstDerivativeBoundary(0.0)
    cs = Math.makeConvexSchabackRationalSpline(allStrikes, allPrices, leftB, rightB, iterations=128)
    ivkScha = @. Black.impliedVolatility(
        true,
        cs(strikes),
        forward,
        strikes,
        tte,
        1.0,
    );
    rmseScha = StatsBase.L2dist(w .* ivkScha, w .* vols)
    println("scha ", rmseScha)
   
    csf, rmseSchaFit = fitConvexSchabackRationalSpline(
        allStrikes, allPrices,
        allWeights,
        leftB,
        rightB;
        penalty = 1e-5
    )
    ivkSchaFit = @. Black.impliedVolatility(
        true,
        csf(strikes),
        forward,
        strikes,
        tte,
        1.0,
    );
    rmseSchaFit = StatsBase.L2dist(w .* ivkSchaFit, w .* vols)
    println("schafit ", rmseSchaFit)
    step = floor(Int, length(strikesf)/7)
    subset = collect(1:step:length(strikesf))
    if subset[end] != length(strikesf)
        append!(subset,length(strikesf))
    end
    allStrikest = vcat(0.0, strikesf[subset], forward*exp(logmoneynessA[end]*3))
    allPricest = vcat(forward, pricesf[subset], 0.0)
    
    cs8 = Math.makeConvexSchabackRationalSpline(allStrikest, allPricest, leftB, rightB, iterations=128)
    ivkScha8 = @. Black.impliedVolatility(
        true,
        cs8(strikes),
        forward,
        strikes,
        tte,
        1.0,
    );
    rmseScha8 = StatsBase.L2dist(w .* ivkScha8, w .* vols)
    println("scha8 ", rmseScha8)
   #=
    plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(kFine./forward),Black.impliedVolatility.(true, max.(csf.(kFine),1e-16), forward, kFine, tte, 1.0) .* 100, label=string("Schaback λ=",1e-5))
    plot!(log.(kFine./forward),Black.impliedVolatility.(true, max.(cs8.(kFine),1e-16), forward, kFine, tte, 1.0) .* 100, label=string("Schaback on 8 knots"))
  plot!(size=(480,380))
 savefig("/home/fabien/mypapers/eqd_book/aapl_20131028_schaback.pdf")

plot(log.(kFine./forward), AQFED.Math.evaluateSecondDerivative.(cs8,kFine),label="Schaback on 8 knots", xlab="Forward log-moneyness",ylab="Probability density")
 plot!(log.(kFine./forward), AQFED.Math.evaluateSecondDerivative.(csf,kFine),label=string("Schaback λ=",1e-5))
plot!(legend=:bottom)
  plot!(size=(480,380))
    savefig("/home/fabien/mypapers/eqd_book/aapl_20131028_schaback_dens.pdf")
=#
   
    λs = [1e-8, 1e-6, 1e-5]
    for λ in λs
        fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (weights), λ=λ, eps=1e-8, solver="GI")
        ivstrikesFengler = @. Black.impliedVolatility(
            true,
            fengler(strikes),
            forward,
            strikes,
            tte,
            1.0,
        )
        rmseFengler = StatsBase.L2dist(w .* ivstrikesFengler, w .* vols)
        println(λ, " Fengler ", rmseFengler)
    end
    kFine = exp.(range(log(strikes[1]), stop=log(strikes[end]), length=501))

    #=p3 = plot(xlabel="Forward log-moneyness", ylabel="Probability density")
    for λ in λs
    fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (weights),λ=λ,eps=1e-13,solver="GI")
    plot!(p3,log.(kFine./forward), AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label=string("Fengler λ=",λ))
    end
    plot(p3,margin=3Plots.mm,size=(480,380))
    savefig(p3,"/home/fabien/mypapers/eqd_book/aapl_20131028_fengler_dens.pdf")
    p4 = plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    for λ in λs
    fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (weights),λ=λ,eps=1e-13,solver="GI")
    plot!(p4,log.(kFine./forward), @.(Black.impliedVolatility(true, max.(fengler.(kFine),1e-16), forward, kFine, tte, 1.0)*100),label=string("Fengler λ=",λ))
    end
    plot(p4,margin=3Plots.mm,size=(480,380))
    savefig(p4,"/home/fabien/mypapers/eqd_book/aapl_20131028_fengler.pdf")

    =#
    rbf = AQFED.VolatilityModels.calibrateMultiquadric(tte, forward, log.(strikes), vols, weightsV, knots=range(log(strikes[1]), stop=log(strikes[end]), length=8))
    rmseRBF = StatsBase.L2dist(w .* sqrt.(rbf.(log.(strikes))), w .* vols)
    pricesRBF = @. blackScholesFormula(true, strikes, forward, rbf(log(strikes / forward)) * tte, 1.0, 1.0)
    λs = [1e-8, 1e-6]
    for λ in λs
        fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, pricesRBF, (weights), λ=λ, eps=1e-8, solver="GI")
        ivstrikesFengler = @. Black.impliedVolatility(
            true,
            fengler(strikes),
            forward,
            strikes,
            tte,
            1.0,
        )
        rmseFengler = StatsBase.L2dist(w .* ivstrikesFengler, w .* vols)
        println(λ, " Fengler ", rmseFengler)

        bspl, m = Collocation.makeExpBSplineCollocation(
            strikes,
            prices,
            weights,
            tte,
            forward,
            1.0,
            penalty=0e-2,
            size=8,
            minSlope=1e-8,
            rawFit=true,
        )
        ivkexp = @. Black.impliedVolatility(
            true,
            Collocation.priceEuropean(bspl, true, strikes, forward, 1.0),
            forward,
            strikes,
            tte,
            1.0,
        )
        rmseexp = StatsBase.L2dist(w .* ivkexp, w .* vols)

    end
    #=
    pdfp = z-> ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.Black.blackScholesFormula(true, y, 1.0, rbf(log(y))*tte,1.0,1.0),x),z)

    p3 = plot(xlabel="Forward log-moneyness", ylabel="Probability density")
    plot!(p3,log.(kFine./forward),pdfp.(kFine),label="RBF")
    for λ in λs
        fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, pricesRBF, (weights),λ=λ,eps=1e-13,solver="GI")
        plot!(p3,log.(kFine./forward), AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label=string("Fengler λ=",λ))
     end
    plot(p3,margin=2Plots.mm,size=(480,380))
    savefig(p3,"/home/fabien/mypapers/eqd_book/aapl_20131028_fengler_rbf_dens.pdf")
     p4 = plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
     for λ in λs
        fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, pricesRBF, (weights),λ=λ,eps=1e-13,solver="GI")
    plot!(p4,log.(kFine./forward), @.(Black.impliedVolatility(true, max.(fengler.(kFine),1e-16), forward, kFine, tte, 1.0)*100),label=string("Fengler λ=",λ))
     end
    plot!(p4,log.(kFine./forward),sqrt.(rbf.(log.(kFine)./forward)) .*100 ,label="RBF")
    plot(p4,margin=2Plots.mm,size=(480,380))
    savefig(p4,"/home/fabien/mypapers/eqd_book/aapl_20131028_rbf_fengler.pdf")

     =#

    lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, weights, useVol=true, L=strikes[1] / 2, U=strikes[end] * 2)
    ivstrikes = @. Black.impliedVolatility(
        true,
        PDDE.priceEuropean(lvg, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmse = StatsBase.L2dist(w .* ivstrikes, w .* vols)
    println("LVG-Black ", rmse)
    lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, weights, useVol=true, L=strikes[1] / 2, U=strikes[end] * 2, penalty=1e-6)
    ivstrikes = @. Black.impliedVolatility(
        true,
        PDDE.priceEuropean(lvg, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmse = StatsBase.L2dist(w .* ivstrikes, w .* vols)
    println("LVG-Black ", rmse)
    lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, model=PDDE.Quadratic(),location="Equidistributed",size=10,L=strikes[1],U=strikes[end])
    ivstrikes = @. Black.impliedVolatility(
        true,
        PDDE.priceEuropean(lvgq, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmse = StatsBase.L2dist(w .* ivstrikes, w .* vols)

    #plot(kFine,(AQFED.PDDE.derivativePrice.(lvg,true,kFine.+0.0001) .- AQFED.PDDE.derivativePrice.(lvg,true,kFine)).*10000, label="LVG")

    #=

p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,PDDE.priceEuropean.(lvgq,true,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Quadratic LVG")
plot!(p3, size=(480,320))
 savefig("/home/fabien/mypapers/eqd_book/aapl_20131028_lvgq_vol.pdf")
p2 = plot(log.(kFine./forward),(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label=:none, xlab="Forward log-moneyness",ylab="Probability density")
 plot!(p2, size=(480,320))
 savefig("/home/fabien/mypapers/eqd_book/aapl_20131028_lvgq_dens.pdf")
=#
   kernel2 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=2)
    ivkMLN2= @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel2, strikes >= forward, strikes),
    forward,
                strikes,
                tte,
                1.0,
            );
           rmseMLN2 = StatsBase.L2dist(w .* ivkMLN2, w .* vols)
           kernel3 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=3)
           ivkMLN3= @. AQFED.Black.impliedVolatility(
                          strikes >= forward,
                          AQFED.VolatilityModels.priceEuropean.(kernel3, strikes >= forward, strikes),
                          forward,
                          strikes,
                          tte,
                          1.0,
                      );
                      rmseMLN3 = StatsBase.L2dist(w .* ivkMLN3, w .* vols)
           
kernel4 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=4)
ivkMLN4= @. AQFED.Black.impliedVolatility(
               true,
               AQFED.VolatilityModels.priceEuropean.(kernel4, true, strikes),
               forward,
               strikes,
               tte,
               1.0,
           );
           rmseMLN4 = StatsBase.L2dist(w .* ivkMLN4, w .* vols)
           kernel6 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=6)
ivkMLN6= @. AQFED.Black.impliedVolatility(
               true,
               AQFED.VolatilityModels.priceEuropean.(kernel6, true, strikes),
               forward,
               strikes,
               tte,
               1.0,
           );
           rmseMLN6 = StatsBase.L2dist(w .* ivkMLN6, w .* vols)

end

@testset "SPX1m" begin
    strikes = Float64.([1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2325, 2350, 2375, 2400, 2425, 2450, 2470, 2475, 2480, 2490, 2510, 2520, 2530, 2540, 2550, 2560, 2570, 2575, 2580, 2590, 2600, 2610, 2615, 2620, 2625, 2630, 2635, 2640, 2645, 2650, 2655, 2660, 2665, 2670, 2675, 2680, 2685, 2690, 2695, 2700, 2705, 2710, 2715, 2720, 2725, 2730, 2735, 2740, 2745, 2750, 2755, 2760, 2765, 2770, 2775, 2780, 2785, 2790, 2795, 2800, 2805, 2810, 2815, 2835, 2860, 2900])
    vols = [0.684882717072609, 0.6548002174209514, 0.6279717042323061, 0.6040669049212617, 0.5769233835086068, 0.5512534351594732, 0.5260245499632258, 0.5004353919117, 0.4741366518169333, 0.46171589561249216, 0.4457089283432941, 0.4336614266663264, 0.420159764469498, 0.4074628373496824, 0.3931682390848574, 0.3814047904881801, 0.37929970817058073, 0.3771088224218263, 0.3724714977308359, 0.36029419336555424, 0.35467069448268806, 0.3505327949033959, 0.3441904382413214, 0.3392727917494692, 0.33306859556194446, 0.32820593458977093, 0.3243137942797042, 0.32204084870033645, 0.3168000315981532, 0.3109143207658998, 0.3050420836154825, 0.30241566311445206, 0.29948796266862154, 0.29609035936524486, 0.2923777072285143, 0.28951623883712746, 0.28584033838767425, 0.283342147794602, 0.2808533651372528, 0.27703523377755246, 0.27371493615870945, 0.2708906740100327, 0.2678887418986713, 0.2645328136650213, 0.26234402136468965, 0.2585977172018311, 0.25550003988953746, 0.2521896614376435, 0.2495339851370865, 0.24665927818229774, 0.24355339309186683, 0.24020198229067014, 0.23658800157061083, 0.23457380906338043, 0.23040670495884208, 0.2278656924642955, 0.22304945749920857, 0.21988751701341647, 0.2184983910827269, 0.21470247194448602, 0.21050624458263925, 0.20817463333507674, 0.20550780781621286, 0.19996741584940433, 0.19900703596491134, 0.19506177682405323, 0.19054732989021844, 0.18842657099566548, 0.18589260856179804, 0.18287835748424114, 0.17929170978590483, 0.17500063441150882, 0.18575101811296996, 0.20717302702012957, 0.22524785579801052]
    weightsV = [1.1789826122551597, 1.224744871391589, 1.284523257866513, 1.3601470508735443, 1.4317821063276355, 1.5165750888103102, 1.6124515496597098, 1.7175564037317668, 1.8395212376698413, 1.8973665961010275, 1.949358868961793, 2.024845673131659, 2.0976176963403033, 2.179449471770337, 2.258317958127243, 2.32379000772445, 2.345207879911715, 2.3664319132398464, 2.4083189157584592, 2.479919353527449, 2.5199206336708304, 2.569046515733026, 2.6076809620810595, 2.6551836094703507, 2.7625312572654126, 2.7477263328068173, 2.765863337187866, 2.7928480087537886, 2.871393034605969, 2.964704653791087, 3.0174928596261394, 3.0017001984929568, 3, 3.024896692450835, 3.120391338480345, 2.9916550603303182, 2.947349434130382, 2.8809720581775866, 2.8284271247461903, 2.8083087326973732, 2.711088342345192, 2.6685599339741506, 2.62445329583912, 2.565469285152568, 2.4899799195977463, 2.4289915602982237, 2.4279079146675357, 2.313006701244076, 2.258317958127243, 2.202271554554524, 2.1447610589527217, 2.085665361461421, 2.024845673131659, 1.97484176581315, 1.91049731745428, 1.857417562100671, 1.7888543819998317, 1.7320508075688772, 1.6881943016134133, 1.6278820596099706, 1.5652475842498528, 1.5165750888103102, 1.466287829861518, 1.3964240043768943, 1.3601470508735443, 1.3038404810405297, 1.2449899597988732, 1.2041594578792296, 1.161895003862225, 1.118033988749895, 1.0723805294763609, 1.02469507659596, 1.0099504938362078, 1.0910894511799618, 1.005037815259212]
    forward = 2629.8026715608194
    tte = 0.0821917808219178
    logmoneynessA = log.(strikes ./ forward)
    sumw2 = sum(weightsV .^ 2)
    w = weightsV ./ sqrt(sumw2)
    svi0, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logmoneynessA, vols, w, aMin=0.0)
    ivkSVI0 = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi0, logmoneynessA))
    rmseSVI0 = StatsBase.L2dist(w .* vols, w .* ivkSVI0)
    xssvi = AQFED.VolatilityModels.calibrateXSSVISection(tte, forward, logmoneynessA, vols, w)
    ivkXSSVI = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(xssvi, logmoneynessA))
    rmseXSSVI = StatsBase.L2dist(w .* vols, w .* ivkXSSVI)
    #plot!(logmoneynessA, sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(xssvi,logmoneynessA)).*100,label="XSSVI")

    prices, weights = Collocation.weightedPrices(true, strikes, vols, w, forward, 1.0, tte, vegaFloor=1e-5)
    
    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, weights, tte, forward, 1.0, deg=3, degGuess=3)
    sol3 = Collocation.Polynomial(isoc)
    ivk3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol3, true, strikes, forward, 1.0), forward, strikes, tte, 1.0);
   rmse3 = StatsBase.L2dist(w .* vols, w .* ivk3)
    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, weights, tte, forward, 1.0, deg=5, degGuess=3, minSlope=1e-5)
    sol5 = Collocation.Polynomial(isoc)
    ivk5 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol5, true, strikes, forward, 1.0), forward, strikes, tte, 1.0);
    rmse5 = StatsBase.L2dist(w .* vols, w .* ivk5)
    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, weights, tte, forward, 1.0, deg=9, degGuess=3, minSlope=1e-5)
    sol9 = Collocation.Polynomial(isoc)
    ivk9 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol9, true, strikes, forward, 1.0), forward, strikes, tte, 1.0);
    rmse9 = StatsBase.L2dist(w .* vols, w .* ivk9)
    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, weights, tte, forward, 1.0, deg=11, degGuess=3, minSlope=1e-5)
    sol11 = Collocation.Polynomial(isoc)
    ivk11 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol11, true, strikes, forward, 1.0), forward, strikes, tte, 1.0);
    rmse11 = StatsBase.L2dist(w .* vols, w .* ivk11)

    bsple, m = Collocation.makeExpBSplineCollocation(
        strikes,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=10,
        minSlope=1e-8,
        rawFit=false
    )
    ivkbe = @. Black.impliedVolatility(true, Collocation.priceEuropean(bsple, true, strikes, forward, 1.0), forward, strikes, tte, 1.0);
    rmsebe = StatsBase.L2dist(w .* vols, w .* ivkbe)
    bspl3, m = Collocation.makeBSplineCollocation(
     strikes,
     prices,
     weights,
     tte,
     forward,
     1.0,
     penalty=0e-2,
     size=10,
     minSlope=1e-8,
     rawFit=false,N=3
 )
 ivkb3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(bspl3, true, strikes, forward, 1.0), forward, strikes, tte, 1.0);
 rmseb3 = StatsBase.L2dist(w .* vols, w .* ivkb3)
 lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, model=PDDE.Quadratic(),location="Equidistributed",size=10, L=strikes[1], U=strikes[end])
 ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0);
 rmseq = StatsBase.L2dist(w .* vols, w .* ivkq)
#== 
kFine = forward.*exp.(range(logmoneynessA[1],stop=logmoneynessA[end], length=1001));
p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bsple,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Exp B-Spline")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bspl3,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Cubic B-spline")
plot!(p3, size=(480,320))
savefig("/home/fabien/mypapers/eqd_book/aapl_20131028_vol_bspl.pdf")
p4=plot(log.(kFine./forward), Collocation.density.(bsple,kFine),label="Exp B-spline",xlab="Log-moneyness",ylab="Probability density",color=2)
plot!(p4,log.(kFine./forward), Collocation.density.(bspl3,kFine),label="Cubic B-spline",color=3)
#plot!(p4,yscale=:log10)
plot!(p4, size=(480,320))
savefig("/home/fabien/mypapers/eqd_book/aapl_20131028_dens_bspl.pdf")
==#   
    #=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, Collocation.priceEuropean.(sol5,true,kFine), forward, kFine, tte, 1.0)*100),label="Collocation")
    
p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,PDDE.priceEuropean.(lvgq,true,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Quadratic LVG")
savefig("/home/fabien/mypapers/eqd_book/vol_spw_1m_050218_lvgq.pdf")
p2 = plot(log.(kFine./forward),(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label=:none, xlab="Forward log-moneyness",ylab="Probability density")
 plot!(p2, size=(480,380))
 savefig("/home/fabien/mypapers/eqd_book/density_spw_1m_050218_lvgq.pdf")
   =#
    
    λs = [1.6e4, 3.2e4, 6.4e4]
    for λ in λs
        fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (weights), λ=λ, eps=1e-13, solver="GI")
        ivstrikesFengler = @. Black.impliedVolatility(
            true,
            fengler(strikes),
            forward,
            strikes,
            tte,
            1.0,
        )
        rmseFengler = StatsBase.L2dist(w .* ivstrikesFengler, w .* vols)
        println(λ, " Fengler ", rmseFengler)
    end
    kFine = exp.(range(log(strikes[1]), stop=log(strikes[end]), length=501))
    lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, weights, useVol=true, L=strikes[1] / 2, U=strikes[end] * 2, penalty=1.0)
    ivstrikes = @. Black.impliedVolatility(
        true,
        PDDE.priceEuropean(lvg, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmse = StatsBase.L2dist(w .* ivstrikes, w .* vols)
    println("LVG-Black ", rmse)
    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, weights, tte, forward, 1.0, deg=3, degGuess=1)
    sol = Collocation.Polynomial(isoc)
    dev = exp(3 * vols[3] * sqrt(tte))
    ivk3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol, true, strikes, forward, 1.0), forward, strikes, tte, 1.0)
    # kFine = collect(range(k[1]/dev,stop=k[end]*dev,length=201))
    ivkFine3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol, true, kFine, forward, 1.0), forward, kFine, tte, 1.0)
    rmse3 = StatsBase.L2dist(w .* ivk3, w .* vols)
    #    isoc, m = Collocation.makeIsotonicCollocation(k, prices, weights, tte, forward, 1.0, deg = 5, degGuess = 1) #strangely minSlope has big influence
    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, weights, tte, forward, 1.0, deg=5, degGuess=3, minSlope=1e-5)
    sol = Collocation.Polynomial(isoc)
    ivk5 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol, true, strikes, forward, 1.0), forward, strikes, tte, 1.0)
    rmse5 = StatsBase.L2dist(w .* ivk5, w .* vols)
    ivkFine5 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol, true, kFine, forward, 1.0), forward, kFine, tte, 1.0)
    #=p4 = plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, PDDE.priceEuropean.(lvg,true,kFine), forward, kFine, tte, 1.0)*100),label="LVG Linear Black")
    plot!(log.(kFine./forward), 100 .* ivkFine3, label="Cubic collocation")
    plot!(log.(kFine./forward), 100 .* ivkFine5, label="Quintic collocation")

    #density is not good with RBF, we see the knots locations.
    plot(log.(kFine./forward),pdfp.(kFine),label="RBF")
    plot!(log.(kFine./forward),(AQFED.PDDE.derivativePrice.(lvg,true,kFine.+0.0001) .- AQFED.PDDE.derivativePrice.(lvg,true,kFine)).*10000, label="LVG")
    =#

    ##the exp b spline collo is bad on this example. Would it be better to use less knots and no regul? and faster.
    bspl, m = Collocation.makeExpBSplineCollocation(
        strikes,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=8,
        minSlope1e-8,
        rawFit=true,
    )
    ivkexp = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(bspl, true, strikes, forward, 1.0),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseexp = StatsBase.L2dist(w .* ivkexp, w .* vols)
    strikesf, pricesf = AQFED.Collocation.filterConvexPrices(strikes, prices, weights, forward,tol=1e-6)
    allStrikes = vcat(0.0, strikesf, forward*exp(logmoneynessA[end]*3))
    allPrices = vcat(forward, pricesf, 0.0)
    allWeights = vcat(sum(weights),weights,sum(weights))
    leftB = Math.FirstDerivativeBoundary(-1.0)
    rightB = Math.FirstDerivativeBoundary(0.0)
    cs = Math.makeConvexSchabackRationalSpline(allStrikes, allPrices, leftB, rightB, iterations=128)
    ivkScha = @. Black.impliedVolatility(
        true,
        cs(strikes),
        forward,
        strikes,
        tte,
        1.0,
    );
    rmseScha = StatsBase.L2dist(w .* ivkScha, w .* vols)
    println("scha ", rmseScha)
   
    csf, rmseSchaFit = fitConvexSchabackRationalSpline(
        allStrikes, allPrices,
        allWeights,
        leftB,
        rightB;
        penalty = 0.1
    )
    ivkSchaFit = @. Black.impliedVolatility(
        true,
        csf(strikes),
        forward,
        strikes,
        tte,
        1.0,
    );
    rmseSchaFit = StatsBase.L2dist(w .* ivkSchaFit, w .* vols)
    println("schafit ", rmseSchaFit)
    step = floor(Int, length(strikesf)/7)
    subset = collect(1:step:length(strikesf))
    if subset[end] != length(strikesf)
        append!(subset,length(strikesf))
    end
    allStrikest = vcat(0.0, strikesf[subset], forward*exp(logmoneynessA[end]*3))
    allPricest = vcat(forward, pricesf[subset], 0.0)
    
    cs8 = Math.makeConvexSchabackRationalSpline(allStrikest, allPricest, leftB, rightB, iterations=128)
    ivkScha8 = @. Black.impliedVolatility(
        true,
        cs8(strikes),
        forward,
        strikes,
        tte,
        1.0,
    );
    rmseScha8 = StatsBase.L2dist(w .* ivkScha8, w .* vols)
    println("scha8 ", rmseScha8)
   #=
    plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(kFine./forward),Black.impliedVolatility.(true, max.(csf.(kFine),1e-16), forward, kFine, tte, 1.0) .* 100, label=string("Schaback λ=",1e-5))
    plot!(log.(kFine./forward),Black.impliedVolatility.(true, max.(cs8.(kFine),1e-16), forward, kFine, tte, 1.0) .* 100, label=string("Schaback on 8 knots"))
  plot!(size=(480,380))
 savefig("/home/fabien/mypapers/eqd_book/aapl_20131028_schaback.pdf")

plot(log.(kFine./forward), AQFED.Math.evaluateSecondDerivative.(cs8,kFine),label="Schaback on 8 knots", xlab="Forward log-moneyness",ylab="Probability density")
 plot!(log.(kFine./forward), AQFED.Math.evaluateSecondDerivative.(csf,kFine),label=string("Schaback λ=",1e-5))
plot!(legend=:bottom)
  plot!(size=(480,380))
    savefig("/home/fabien/mypapers/eqd_book/aapl_20131028_schaback_dens.pdf")
=#
kernel2 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=2)
ivkMLN2= @. AQFED.Black.impliedVolatility(
               true,
               AQFED.VolatilityModels.priceEuropean.(kernel2, true, strikes),
               forward,
               strikes,
               tte,
               1.0,
           );
           rmseMLN2 = StatsBase.L2dist(w .* ivkMLN2, w .* vols)
           kernel3 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=3)
           ivkMLN3= @. AQFED.Black.impliedVolatility(
                          true,
                          AQFED.VolatilityModels.priceEuropean.(kernel3, true, strikes),
                          forward,
                          strikes,
                          tte,
                          1.0,
                      );
                      rmseMLN3 = StatsBase.L2dist(w .* ivkMLN3, w .* vols)
           
kernel4 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=4)
ivkMLN4= @. AQFED.Black.impliedVolatility(
               true,
               AQFED.VolatilityModels.priceEuropean.(kernel4, true, strikes),
               forward,
               strikes,
               tte,
               1.0,
           );
           rmseMLN4 = StatsBase.L2dist(w .* ivkMLN4, w .* vols)
           kernel6 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=6)
ivkMLN6= @. AQFED.Black.impliedVolatility(
               true,
               AQFED.VolatilityModels.priceEuropean.(kernel6, true, strikes),
               forward,
               strikes,
               tte,
               1.0,
           );
           rmseMLN6 = StatsBase.L2dist(w .* ivkMLN6, w .* vols)

   #=
    plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(true, AQFED.VolatilityModels.priceEuropean.(kernel2,true,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 2")
    plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(true, AQFED.VolatilityModels.priceEuropean.(kernel4,true,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 4")
    plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(true, AQFED.VolatilityModels.priceEuropean.(kernel6,true,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 6")
 plot!(size=(800,320),margin=3Plots.mm)
  savefig("/home/fabien/mypapers/eqd_book/vol_spw_1m_050218_mln6.pdf")

plot(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel2,kFine),label="Mixture of 2", color=2,xlab="Forward log-moneyness",ylab="Probability density")
 plot!(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel4,kFine),label="Mixture of 4",color=3)
 plot!(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel6,kFine),label="Mixture of 6",color=4)
plot!(size=(800,320),margin=3Plots.mm)
    savefig("/home/fabien/mypapers/eqd_book/density_spw_1m_050218_mln6.pdf")
=  =#
end


@testset "TSLA1m" begin
    tte = 0.095890
    forward = 357.755926
    strikes = Float64.([150, 155, 160, 165, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 500, 520, 540, 560, 580])
    vols = [1.027152094560499, 0.9905195749900226, 0.9657262376591365, 0.9405597986379826, 0.9181603362313814, 0.9019382314978117, 0.8846745842549402, 0.865754243981787, 0.8456155492434201, 0.8245634579529838, 0.8028174604214972, 0.78053958851195, 0.7636802684802435, 0.7454192306685303, 0.7260651215584285, 0.7058414693439228, 0.6849143304434797, 0.663409356115238, 0.6462309799739909, 0.6301291739261891, 0.6130540004186168, 0.5946923076348443, 0.5811921286363728, 0.5687314890047378, 0.5539815904720001, 0.5422671292669776, 0.5338887990387771, 0.5234154661207794, 0.5168510552270313, 0.5072806473672073, 0.4997973159961656, 0.4896563997378466, 0.4823975850368014, 0.47936818364069134, 0.48000585384055006, 0.4757525564073338, 0.4711478482467228, 0.46788352167691083, 0.46562175169660713, 0.46299652559206567, 0.45939930288424485, 0.458565105643866, 0.45790487479637937, 0.45521398441321903, 0.453447302139774, 0.4504013827012644, 0.448004721643358, 0.4491995553643971, 0.4478840707248649, 0.45006593113797866, 0.4517530880150887, 0.4499007489879635, 0.448814967685824, 0.45160477568536983, 0.4563938928347205, 0.4600222064217672, 0.46102443173801966, 0.46406468170261544, 0.4709795491400157, 0.4762595045128011, 0.4810009989573377, 0.4855906965577297, 0.4906446878461756, 0.4960612773473766, 0.5011170526132832, 0.5059204240563133, 0.5159102206249263, 0.5505625146941026, 0.5783881966646062, 0.599260903580561, 0.6259792014943735]
    weightsV = [1.7320508075688772, 1, 1.224744871391589, 1, 2.738612787525831, 1.558387444947959, 1.9999999999999998, 1.2602520756252087, 1.3301243435223526, 2.273030282830976, 1.3944333775567928, 1.2089410496539776, 1.9999999999999998, 2.0976176963403033, 3.500000000000001, 3.286335345030995, 2.6692695630078282, 2.7838821814150116, 3.1622776601683804, 3.605551275463988, 3.3541019662496834, 3, 2.9742484506432634, 3.6469165057620923, 3.8729833462074152, 4.183300132670376, 3.7505555144093887, 4.1918287860346295, 3.7670248460125917, 4.795831523312714, 4.527692569068711, 3.482097069296032, 3.2333489534143167, 3.687817782917155, 6.3245553203367555, 6.837397165588683, 7.365459931328131, 7.0992957397195395, 7.628892449104261, 7.461009761866454, 8.706319543871567, 8.78635305459552, 7.000000000000021, 7.745966692414834, 8.093207028119338, 6.16441400296897, 4.974937185533098, 4.650268809434567, 4.315669125408015, 4.636809247747854, 4.732863826479693, 3.1144823004794873, 2.8809720581775857, 2.8284271247461894, 2.7718093060793882, 4.092676385936223, 2.7041634565979926, 2.652259934210953, 3.710691413905333, 3.777926319123662, 3.929942040850535, 3.921096785339529, 3.70809924354783, 3.517811819867573, 3.3354160160315844, 3.1622776601683777, 1.3483997249264843, 1.8929694486000912, 1.914854215512676, 1.699673171197595, 1.8708286933869707]
    #note the weights are not good they are in rel price
    logmoneynessA = log.(strikes ./ forward)
    sumw2 = sum(weightsV .^ 2)
    w = weightsV ./ sqrt(sumw2)
    svi0, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logmoneynessA, vols, w, aMin=0.0)
    ivkSVI0 = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi0, logmoneynessA))
    rmseSVI0 = StatsBase.L2dist(w .* vols, w .* ivkSVI0)
    xssvi = AQFED.VolatilityModels.calibrateXSSVISection(tte, forward, logmoneynessA, vols, w)
    ivkXSSVI = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(xssvi, logmoneynessA))
    rmseXSSVI = StatsBase.L2dist(w .* vols, w .* ivkXSSVI)

    prices, weights = Collocation.weightedPrices(true, strikes, vols, w, forward, 1.0, tte, vegaFloor=1e-5)
    λs = [50.0, 200.0, 400.0, 1600.0]
    for λ in λs
        fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (weights), λ=λ, eps=1e-13, solver="GI")
        ivstrikesFengler = @. Black.impliedVolatility(
            true,
            fengler(strikes),
            forward,
            strikes,
            tte,
            1.0,
        )
        rmseFengler = StatsBase.L2dist(w .* ivstrikesFengler, w .* vols)
        println(λ, " Fengler ", rmseFengler)
    end
    kFine = exp.(range(log(strikes[1]), stop=log(strikes[end]), length=501));

    #=p3 = plot(xlabel="Forward log-moneyness", ylabel="Probability density")
    for λ in λs
    fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (weights),λ=λ,eps=1e-13,solver="GI")
    plot!(p3,log.(kFine./forward), AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label=string("Fengler λ=",λ))
    end
    plot(p3,margin=3Plots.mm,size=(480,380))
    savefig(p3,"/home/fabien/mypapers/eqd_book/tsla_180615_180720_fengler_dens.pdf")
    p4 = plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    for λ in λs
    fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (weights),λ=λ,eps=1e-13,solver="GI")
    plot!(p4,log.(kFine./forward), @.(Black.impliedVolatility(true, max.(fengler.(kFine),1e-16), forward, kFine, tte, 1.0)*100),label=string("Fengler λ=",λ))
    end
    plot(p4,margin=3Plots.mm,size=(480,380))
    savefig(p4,"/home/fabien/mypapers/eqd_book/tsla_180615_180720_fengler.pdf")

    =#

    strikesf, pricesf = AQFED.Collocation.filterConvexPrices(strikes, prices, weights, forward,tol=1e-6)
    allStrikes = vcat(0.0, strikesf, forward*exp(logmoneynessA[end]*3))
    allPrices = vcat(forward, pricesf, 0.0)
    allWeights = vcat(sum(weights),weights,sum(weights))
    leftB = Math.FirstDerivativeBoundary(-1.0)
    rightB = Math.FirstDerivativeBoundary(0.0)
    cs = Math.makeConvexSchabackRationalSpline(allStrikes, allPrices, leftB, rightB, iterations=128)
    ivkScha = @. Black.impliedVolatility(
        true,
        cs(strikes),
        forward,
        strikes,
        tte,
        1.0,
    );
    rmseScha = StatsBase.L2dist(w .* ivkScha, w .* vols)
    println("scha ", rmseScha)
   
    csf, rmseSchaFit = fitConvexSchabackRationalSpline(
        allStrikes, allPrices,
        allWeights,
        leftB,
        rightB;
        penalty = 1.0
    )
    ivkSchaFit = @. Black.impliedVolatility(
        true,
        csf(strikes),
        forward,
        strikes,
        tte,
        1.0,
    );
    rmseSchaFit = StatsBase.L2dist(w .* ivkSchaFit, w .* vols)
    println("schafit ", rmseSchaFit)
    step = floor(Int, length(strikesf)/7)
    subset = collect(1:step:length(strikesf))
    if subset[end] != length(strikesf)
        append!(subset,length(strikesf))
    end
    allStrikest = vcat(0.0, strikesf[subset], forward*exp(logmoneynessA[end]*3))
    allPricest = vcat(forward, pricesf[subset], 0.0)
    
    cs8 = Math.makeConvexSchabackRationalSpline(allStrikest, allPricest, leftB, rightB, iterations=128)
    ivkScha8 = @. Black.impliedVolatility(
        true,
        cs8(strikes),
        forward,
        strikes,
        tte,
        1.0,
    );
    rmseScha8 = StatsBase.L2dist(w .* ivkScha8, w .* vols)
    println("scha8 ", rmseScha8)
   
  
    pp = PPInterpolation.CubicSplineNatural(log.(strikes), vols .^ 2)
    rbf = AQFED.VolatilityModels.calibrateMultiquadric(tte, forward, log.(strikes), vols, w1, knots=log.(strikes))

    lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, weights, useVol=true, L=strikes[1] / 2, U=strikes[end] * 2)
    ivstrikes = @. Black.impliedVolatility(
        true,
        PDDE.priceEuropean(lvg, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmse = StatsBase.L2dist(w .* ivstrikes, w .* vols)
    println("LVG-Black ", rmse)
    lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, model=PDDE.Quadratic(),location="Equidistributed",size=10,L=strikes[1],U=strikes[end])
    ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0);
    rmseq = StatsBase.L2dist(w .* vols, w .* ivkq)


    bspl, m = Collocation.makeExpBSplineCollocation(
        strikes,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=8,
        minSlope=1e-8,
        rawFit=true,
    )
    ivkexp = @. Black.impliedVolatility(
        true,
        Collocation.priceEuropean(bspl, true, strikes, forward, 1.0),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseexp = StatsBase.L2dist(w .* ivkexp, w .* vols)

    bsple, m = Collocation.makeExpBSplineCollocation(
        strikes,
        prices,
        weights,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=10,
        minSlope=1e-8,
        rawFit=false
    )
    ivkbe = @. Black.impliedVolatility(true, Collocation.priceEuropean(bsple, true, strikes, forward, 1.0), forward, strikes, tte, 1.0);
    rmsebe = StatsBase.L2dist(w .* vols, w .* ivkbe)
    bspl3, m = Collocation.makeBSplineCollocation(
     strikes,
     prices,
     weights,
     tte,
     forward,
     1.0,
     penalty=0e-2,
     size=10,
     minSlope=1e-8,
     rawFit=false,N=3,extrapolationFactor=1.1, optimizerName="LeastSquaresOptim"
 )
 
 ivkb3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(bspl3, true, strikes, forward, 1.0), forward, strikes, tte, 1.0);
 rmseb3 = StatsBase.L2dist(w .* vols, w .* ivkb3)
 #==
 p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
 plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bsple,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Exp B-Spline")
 plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bspl3,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Cubic B-spline")
 plot!(p3, size=(480,320))
 savefig("/home/fabien/mypapers/eqd_book/tsla_180615_180720_vol_bspl.pdf")
 p4=plot(log.(kFine./forward), Collocation.density.(bsple,kFine),label="Exp B-spline",xlab="Log-moneyness",ylab="Probability density",color=2)
 plot!(p4,log.(kFine./forward), Collocation.density.(bspl3,kFine),label="Cubic B-spline",color=3)
 #plot!(p4,yscale=:log10)
 plot!(p4, size=(480,320),legend=:topleft)
 savefig("/home/fabien/mypapers/eqd_book/tsla_180615_180720_dens_bspl.pdf")

   
p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,PDDE.priceEuropean.(lvgq,true,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Quadratic LVG")
plot!(p3, size=(480,320))
 savefig("/home/fabien/mypapers/eqd_book/tsla_180615_180720_vol_lvgq.pdf")
p2 = plot(log.(kFine./forward),(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label=:none, xlab="Forward log-moneyness",ylab="Probability density")
 plot!(p2, size=(480,320))
 savefig("/home/fabien/mypapers/eqd_book/tsla_180615_180720_dens_lvgq.pdf")

 ==#   
 

    weightsA = ones(length(vols));strikes=k;
    sumw2 = sum(weightsA .^ 2);w = weightsA ./ sqrt(sumw2)


kernel2 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=2)
    ivkMLN2= @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel2, strikes >= forward, strikes),
    forward,
                strikes,
                tte,
                1.0,
            );
           rmseMLN2 = StatsBase.L2dist(w .* ivkMLN2, w .* vols)
           kernel3 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=3)
           ivkMLN3= @. AQFED.Black.impliedVolatility(
                          strikes >= forward,
                          AQFED.VolatilityModels.priceEuropean.(kernel3, strikes >= forward, strikes),
                          forward,
                          strikes,
                          tte,
                          1.0,
                      );
                      rmseMLN3 = StatsBase.L2dist(w .* ivkMLN3, w .* vols)
           
kernel4 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, weights, useVol=false,size=4)
ivkMLN4= @. AQFED.Black.impliedVolatility(
               true,
               AQFED.VolatilityModels.priceEuropean.(kernel4, true, strikes),
               forward,
               strikes,
               tte,
               1.0,
           );
           rmseMLN4 = StatsBase.L2dist(w .* ivkMLN4, w .* vols)
 
    #=
    plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(kFine./forward),Black.impliedVolatility.(true, max.(csf.(kFine),1e-16), forward, kFine, tte, 1.0) .* 100, label=string("Schaback λ=",1))
    plot!(log.(kFine./forward),Black.impliedVolatility.(true, max.(cs8.(kFine),1e-16), forward, kFine, tte, 1.0) .* 100, label=string("Schaback on 8 knots"))
  plot!(size=(480,380))
 savefig("/home/fabien/mypapers/eqd_book/tsla_180615_180720_schaback.pdf")

plot(log.(kFine./forward), AQFED.Math.evaluateSecondDerivative.(cs8,kFine),label="Schaback on 8 knots ", yscale=:log10, xlab="Forward log-moneyness",ylab="Probability density")
 plot!(log.(kFine./forward), AQFED.Math.evaluateSecondDerivative.(csf,kFine),label="Schaback λ=1")
plot!(legend=:topleft)
  plot!(size=(480,380))
    savefig("/home/fabien/mypapers/eqd_book/tsla_180615_180720_schaback_dens.pdf")
   
    plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(strikes./forward),ivkSVI0.*100,label="SVI")
    plot!(log.(strikes./forward),ivkXSSVI.*100,label="XSSVI")
    plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, max.(fengler.(kFine),1e-16), forward, kFine, tte, 1.0)*100),label="Fengler λ=1e-13")

    #savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_i_fengler_rbf_dens.pdf")

    plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, PDDE.priceEuropean.(lvg,true,kFine), forward, kFine, tte, 1.0)*100),label="LVG Linear Black")
    #savefig("/home/fabien/mypapers/eqd_book/jaeckel_case_i_fengler_rbf.pdf")

    pdf(pp,z) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.Black.blackScholesFormula(true, y, forward, pp(log(y/forward))*tte,1.0,1.0),x),z)

    p1=plot(log.(kFine./forward),AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label="Fengler",color=2, ylim=(-0.05,1.0))
    p2 = plot(log.(kFine./forward),AQFED.Math.evaluateSecondDerivative.(cs,kFine),label="Schaback",color=3,ylim=(-0.05,1.0))
    p3 = plot(log.(kFine./forward),pdf.(pp,kFine),label="Cubic spline on implied variances",color=1,ylim=(-0.05,1.0))
    p4 = plot(log.(kFine./forward),pdf.(rbf,kFine),label="RBF",color=4,ylim=(-0.05,1.0))
    plot(p3,p4,p1,p2, layout=(1,4),legend=false,titles=["Cubic spline" "RBF" "Fengler" "Schaback"],size=(800,250))

     p1=plot(log.(kFine./forward),AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label="Fengler",color=2,ylim=(1e-8,0.008),yscale=:log10)

    =#
    #SV
    tte = 1.591781
    forward = 356.730632
    strikes = Float64.([20, 25, 50, 55, 75, 100, 120, 125, 140, 150, 160, 175, 180, 195, 200, 210, 230, 240, 250, 255, 260, 270, 275, 280, 285, 290, 300, 310, 315, 320, 325, 330, 335, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 550, 580, 590, 600, 650, 670, 680, 690, 700])
    vols = [1.2174498333432284, 1.1529735541872161, 1.004004022581606, 0.9836878310419587, 0.9059774069943589, 0.8197362843642744, 0.7716707471169627, 0.7596970666840304, 0.7255349986293692, 0.7036962946028745, 0.6872956097489304, 0.6628579700139684, 0.6544686699577009, 0.6310048431894972, 0.623904988806911, 0.613032792224091, 0.5887489086881706, 0.5772059051701408, 0.5649684953388189, 0.561418253373758, 0.5576041889980867, 0.549252193194508, 0.5453501604969959, 0.5412198514376074, 0.536873384908355, 0.5323213596815215, 0.5226363012219939, 0.5185316686163134, 0.5160935996651511, 0.5134141336907875, 0.5105033610267267, 0.5073701448945144, 0.504289607300568, 0.5013959697030378, 0.4961897259221572, 0.49144782371138285, 0.48645638955352194, 0.48167998000739076, 0.47609375436090395, 0.4697217879783784, 0.4680183980319167, 0.46552636426393684, 0.46227241039855843, 0.45969798571592974, 0.4567098331715741, 0.4539007045757515, 0.45039656985941307, 0.44619726030207824, 0.44495717278530045, 0.44309517634427675, 0.44061210739821544, 0.4398554529535865, 0.4342208537081089, 0.4299834814231043, 0.42752901586299163, 0.42452445491007096, 0.4222672729031062, 0.4203436892852212, 0.4195681468757861, 0.4188408351329149, 0.4177462959039442]
    weights = [7.7136243102707525, 3.8729833462074152, 6.082762530298218, 1.9639610121239313, 2.9154759474226504, 13.527749258468587, 2.444949442785633, 2.5260547066428276, 3.0797481203579755, 6.866065623255938, 3.496542070619614, 5.535599077142163, 5.8638396451996595, 3.9657625656704085, 5.594114554338226, 3.5878044017939312, 5.588136784946543, 9.637167633698208, 5.006246098625194, 3.8156679669355786, 3.762808812407317, 3.9349550147037164, 4.217311262607291, 4.0473389258811014, 4.58043580425081, 4.227406739590535, 10.737783756436894, 5.041825066382212, 4.936635531449573, 4.723502451553941, 4.491261356392942, 4.533133107978179, 4.638349460129725, 4.647281164532899, 4.730579074318318, 4.135214625627067, 4.024922359499621, 3.924283374069717, 3.82099463490856, 4.141485088006137, 3.636237371545237, 3.53767600483022, 3.605551275463989, 3.570311812262566, 3.433149698736217, 3.4995590551163587, 3.445782599282636, 4.071476768559913, 3.596873642484542, 3.5514649408105448, 3.8544964466377296, 3.9829180714190437, 3.3598305818093763, 3.0684025713101346, 3.0127932350101747, 3.505098327538656, 2.5895718474182514, 2.5248762345905194, 2.506513254633252, 2.4022558842332624, 3.1008683647302107]
    #    weightsV = [16.25288045106638, 8.160911900438244, 12.619088815272084, 3.675840734509003, 5.776554424173832, 27.512440009161907, 4.638043773388409, 4.883079553130767, 5.94131943234476, 13.473174307161502, 6.680481079152589, 10.538920929379609, 11.131996465169651, 7.350499176975279, 10.404413260007315, 6.461526063169936, 9.975723639859746, 16.974674230580774, 8.684906846576, 6.471547708410667, 6.3305947378547245, 6.5136123938783905, 6.92043607456792, 6.585207524684852, 7.384412661775439, 6.754704320859774, 17.091168363110985, 7.7529771845242506, 7.51038911665913, 7.111728791431016, 6.678533051977762, 6.668891459306908, 6.752119237968658, 6.690994656273483, 6.659451607392078, 5.772855007876042, 5.810926108790698, 5.841240932662621, 5.865481045309425, 6.621016050951635, 5.924276744071027, 5.929376447556495, 6.217841480543634, 6.317299897492222, 6.232948658379371, 6.515095778640602, 6.572366611401739, 8.057812002697537, 7.227514274402218, 7.298089942473972, 8.148515888009042, 8.548424627243886, 7.6509577692757835, 7.3483668093512335, 7.3353593339631376, 8.809531159992671, 6.865235860006001, 6.878549795031439, 6.92325311817954, 6.686900221418176, 8.93734577371638]
    logmoneynessA = log.(strikes ./ forward)
    sumw2 = sum(weights .^ 2)
    w = weights ./ sqrt(sumw2)
    svi0, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logmoneynessA, vols, weights, aMin=0.0)
    ivkSVI0 = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi0, logmoneynessA))
    rmseSVI0 = StatsBase.L2dist(w .* vols, w .* ivkSVI0)
    #SVI is good enough on this example

end
function gatheralDenomFinite(w, y)
    dwdy = FiniteDifferences.central_fdm(3, 1)(w, y)
    d2wdy2 = FiniteDifferences.central_fdm(3, 2)(w, y)
    return 1 - y / w(y) * dwdy + (dwdy)^2 * (-1 / 4 - 1 / w(y) + y^2 / w(y)^2) / 4 + d2wdy2 / 2
end
#plot(y,@.(gatheralDenomFinite(x->AQFED.TermStructure.varianceByLogmoneyness(slice,x)*tte,y)))
#plot(y,@.(gatheralDenomFinite(x->AQFED.TermStructure.varianceByLogmoneyness(slice,x)*tte,y)/sqrt(2*π*AQFED.TermStructure.varianceByLogmoneyness(slice,y)*tte)*exp(-0.5*(y/sqrt(AQFED.TermStructure.varianceByLogmoneyness(slice,y)*tte)+sqrt(AQFED.TermStructure.varianceByLogmoneyness(slice,y)*tte)/2)^2)  )  )

@testset "Kahale" begin
    r = 0.06
	q = 0.0262
	
relStrikes = [0.85,	0.9,0.95,1,	1.05,1.1,1.15,1.2,1.3,1.4]
spot = 590.0
strikes = spot .* relStrikes
expiries = [0.175,
0.425,
0.695,
0.94,
1,
1.5,
2,
3,
4,
5]
vols = [0.19	0.168	0.133	0.113	0.102	0.097	0.12	0.142	0.169	0.2;
0.177	0.155	0.138	0.125	0.109	0.103	0.1	0.114	0.13	0.15;
0.172	0.157	0.144	0.133	0.118	0.104	0.1	0.101	0.108	0.124;
0.171	0.159	0.149	0.137	0.127	0.113	0.106	0.103	0.1	0.11;
0.171	0.159	0.15	0.138	0.128	0.115	0.107	0.103	0.099	0.108;
0.169	0.16	0.151	0.142	0.133	0.124	0.119	0.113	0.107	0.102;
0.169	0.161	0.153	0.145	0.137	0.13	0.126	0.119	0.115	0.111;
0.168	0.161	0.155	0.149	0.143	0.137	0.133	0.128	0.124	0.123;
0.168	0.162	0.157	0.152	0.148	0.143	0.139	0.135	0.13	0.128;
0.168	0.164	0.159	0.154	0.151	0.148	0.144	0.14	0.136	0.132]
prices = zeros(Float64,size(vols))
strikesM = similar(prices)
forwards = zeros(Float64,length(expiries))
weightsM = similar(prices)
ys = similar(prices)
for (i,expiry) in enumerate(expiries)
    forwards[i] = spot*exp((r-q)*expiry)
    pricesi, weights = Collocation.weightedPrices(true, strikes, vols[i,:], ones(length(strikes)), forwards[i], 1.0, expiry)    
    for (j,strike) in enumerate(strikes)
        strikesM[i,j] = strike    
        ys[i,j] = log(strike/forwards[i])
        weightsM[i,j] = weights[j]
        prices[i,j]=pricesi[j]
    end
end
surface = AQFED.VolatilityModels.calibrateFenglerSurface(expiries,forwards,strikesM,prices, weightsM, λ=1e-2,solver="GI")

#=
k = range(strikes[1],stop=strikes[end],length=101)
 rbf = AQFED.VolatilityModels.calibrateMultiquadric(expiries, forward, ys,vols)
#create fengler from k, sqrt.(tte,rbf(log(k/f))).
p = plot(xlab="Forward Log-moneyness", ylab="Total variance")
 for (i,tte) in enumerate(expiries)
                plot!(p, log.(strikes./forwards[i]), vols[i,:].^2 .*tte, seriestype=:scatter,  markersize=3, markerstrokewidth=-1,label="",color=1)
                plot!(p, log.(k./forwards[i]), @.((AQFED.TermStructure.varianceByLogmoneyness(surface, log(k/forwards[i]),tte))*tte), label=string(tte))
                end
plot(p,legendtitle="T")
plot!(p,size=(480,340))
savefig("/home/fabien/mypapers/eqd_book/kahale_fengler_totalvar.pdf")

pdfsurfacep(svi,z,forward,tte) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.VolatilityModels.price(svi,log(y/forward),tte),x),z)

pdfvalues = zeros(length(expiries),length(k))
for (i,tte) in enumerate(expiries)
          pdfvalues[i,:] = @.(pdfsurfacep(surface, k,forwards[i],tte))
      end
plot(expiries, k, pdfvalues',st=:surface,camera=(80,30),ylab="Strike",xlab="Expiry",zlab="Probability Density",colorbar=false)
plot!(size=(480,340))
savefig("/home/fabien/mypapers/eqd_book/kahale_fengler_dens.pdf")

#PDF bof
#LV 3D?

#Prior RBF?
=#

### LVG
# forward*(1-strike/forward)
surfaceLVG = PDDE.calibrateLVGSurface(expiries,forwards,strikesM,prices,weightsM)
for (i,expiry) in enumerate(expiries)
    for (j,strike) in enumerate(strikes)
        y = log(strike/forwards[i])
        volij = sqrt(PDDE.varianceByLogmoneyness(surfaceLVG, y, expiry))
        println(expiry," ",strike," ",volij, " ",vols[i,j]," ",volij-vols[i,j])
    end    
end
#=
using Plots
gr()
k = range(strikes[1],stop=strikes[end],length=101)
p = plot(xlab="Forward Log-moneyness", ylab="Total variance")
 for (i,tte) in enumerate(expiries)
                plot!(p, log.(strikes./forwards[i]), vols[i,:].^2 .*tte, seriestype=:scatter,  markersize=3, markerstrokewidth=-1,label="",color=1)
                plot!(p, log.(k./forwards[i]), @.((PDDE.varianceByLogmoneyness(surfaceLVG, log(k/forwards[i]),tte))*tte), label=string(tte))
                end
plot(p,legendtitle="T")
savefig("/home/fabien/mypapers/eqd_book/kahale_lvg_totalvar.pdf")
p2 = plot(xlab="Forward log-moneyness",ylab="Probability density")
 for (i,tte) in enumerate(expiries)
    lvgq = surfaceLVG.sections[i]
                        plot!(p2, log.(k./forwards[i]), (PDDE.derivativePrice.(lvgq,true,k./forwards[i].+0.0001) .- PDDE.derivativePrice.(lvgq,true,k./forwards[i])).*10000, label=string(tte))
                end
plot(p2,legendtitle="T")



pyplot()
t = range(expiries[1]/2,stop=expiries[end],length=25)
ivMatrix = zeros(length(t),length(k))
for (i,tte)= enumerate(t)
    fi = spot*exp((r-q)*tte)
    ivMatrix[i,:] = @. sqrt(PDDE.varianceByLogmoneyness(surfaceLVG, log(k/fi),tte))
end
plot(t, k, ivMatrix'.*100,st=:surface,camera=(-45,30),ylab="Strike",xlab="Expiry",zlab="Implied volatility in %", legend=:none, zguidefontrotation=90,margin=0Plots.mm,size=(600,600))
savefig("/home/fabien/mypapers/eqd_book/kahale_lvg_iv3d.pdf")
plot(t, k, ivMatrix'.*100,st=:surface,camera=(45,30),ylab="Strike",xlab="Expiry",zlab="Implied volatility in %", legend=:none,  zguidefontrotation=90,margin=0Plots.mm,size=(600,600))
savefig("/home/fabien/mypapers/eqd_book/kahale_lvg_iv3db.pdf")

lvMatrix = zeros(length(t),length(k))
eps=1e-4
for (i,tte)= enumerate(t)
    fi = spot*exp((r-q)*tte)
    w(y) = PDDE.varianceByLogmoneyness(surfaceLVG, y,tte)*tte
    lvMatrix[i,:] = @. sqrt((PDDE.varianceByLogmoneyness(surfaceLVG, log(k/fi),tte+eps)*(tte+eps)-PDDE.varianceByLogmoneyness(surfaceLVG, log(k/fi),tte)*tte)/(eps*gatheralDenomFinite(w, log(k/fi))))
end
plot(t, k, lvMatrix'.*100,st=:surface,camera=(-45,30),ylab="Strike",xlab="Expiry",zlab="Local volatility in %", legend=:none, zguidefontrotation=90,margin=0Plots.mm,size=(600,600))
savefig("/home/fabien/mypapers/eqd_book/kahale_lvg_lv3d.pdf")
plot(t, k, lvMatrix'.*100,st=:surface,camera=(45,30),ylab="Strike",xlab="Expiry",zlab="Local volatility in %", legend=:none,  zguidefontrotation=90,margin=0Plots.mm,size=(600,600))
savefig("/home/fabien/mypapers/eqd_book/kahale_lvg_lv3db.pdf")

=#

