using AQFED, Test
using StatsBase
using AQFED.Black
using AQFED.Collocation
import Polynomials: coeffs
using AQFED.Math,AQFED.PDDE,AQFED.VolatilityModels
using PPInterpolation
using CharFuncPricing
using LinearAlgebra #norm
#using Plots


@testset "svi-bad" begin
    tte = 1.0
    forward = 1.0
    strikes = [0.5, 0.6, 0.7, 0.8,
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

    volatility = [39.8,
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
    logmoneynessA = log.(strikes ./ forward)
    weightsA = ones(length(strikes))
    volA = volatility ./ 100
    svi0, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logmoneynessA, volA, weightsA, aMin=0.0)
    ivkSVI0 = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi0, logmoneynessA))
    rmseSVI0 = StatsBase.rmsd(volA, ivkSVI0)
    svi, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logmoneynessA, volA, weightsA, aMin=-0.2)
    ivkSVI = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi, logmoneynessA))
    rmseSVI = StatsBase.rmsd(volA, ivkSVI)
    prices, wv = Collocation.weightedPrices(true, strikes, volA, weightsA, forward, 1.0, tte, vegaFloor=1e-8)
    lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, wv, useVol=false, model=PDDE.Quadratic(), location="Equidistributed", size=10, L=strikes[1], U=strikes[end])
    ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0)
    rmseq = StatsBase.L2dist(wv .* volA, wv .* ivkq)
    lvgqe = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, wv, useVol=false, model=PDDE.Quadratic(), location="Mid-XX", size=0, L=strikes[1], U=strikes[end])
    ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgqe, true, strikes), forward, strikes, tte, 1.0)
    rmseq = StatsBase.L2dist(wv .* volA, wv .* ivkq)

end
@testset "spxw170324_170316" begin
    tte = 8 / 365
    forward = 2385.099980
    logmoneynessA = [
        -0.6869194871171992,
        -0.6068767794436628,
        -0.532768807289941,
        -0.4976774874786709,
        -0.4637759358029896,
        -0.4309861129799986,
        -0.3992374146654184,
        -0.3684657559986646,
        -0.33861279284898355,
        -0.3240139934278308,
        -0.30962525597573115,
        -0.2954406209837748,
        -0.2814543790090349,
        -0.26766105687669905,
        -0.25405540482092054,
        -0.2406323844887798,
        -0.23796926706929608,
        -0.23266421483960298,
        -0.22738715773875914,
        -0.22213780185261547,
        -0.21691585787146372,
        -0.21431507617140633,
        -0.2117210409943598,
        -0.20655307083591715,
        -0.20141167133549853,
        -0.19885085047382484,
        -0.19629657066872805,
        -0.1937487985899293,
        -0.19120750116125684,
        -0.18867264555806873,
        -0.18614419920471004,
        -0.18362212977200013,
        -0.18110640517475293,
        -0.17859699356932715,
        -0.17609386335120858,
        -0.17359698315262134,
        -0.17110632184016958,
        -0.1686218485125076,
        -0.1661435324980405,
        -0.1636713433526514,
        -0.161205250857458,
        -0.1587452250165956,
        -0.15629123605502887,
        -0.1538432544163888,
        -0.15140125076083716,
        -0.148965195962956,
        -0.1465350611096642,
        -0.1441108174981579,
        -0.14169243663387623,
        -0.1392798902284923,
        -0.13687315019792728,
        -0.1344721886603892,
        -0.13207697793443432,
        -0.12968749053705286,
        -0.12730369918177656,
        -0.12492557677680924,
        -0.12255309642317881,
        -0.12018623141291263,
        -0.1178249552272328,
        -0.11546924153477382,
        -0.11311906418982019,
        -0.11077439723056604,
        -0.10843521487739377,
        -0.10610149153117368,
        -0.10377320177158252,
        -0.1014503203554428,
        -0.09913282221508024,
        -0.0968206824567008,
        -0.09451387635878575,
        -0.09221237937050648,
        -0.08991616711015624,
        -0.08762521536360049,
        -0.08533950008274448,
        -0.08305899738401923,
        -0.0807836835468837,
        -0.07851353501234465,
        -0.07624852838149249,
        -0.07398864041405494,
        -0.0717338480269658,
        -0.06948412829295042,
        -0.06723945843912654,
        -0.06499981584562174,
        -0.06276517804420538,
        -0.06053552271693634,
        -0.058310827694825165,
        -0.05609107095651222,
        -0.05387623062695936,
        -0.051666284976156614,
        -0.04946121241784262,
        -0.047260991508240154,
        -0.04506560094480458,
        -0.042875019564985926,
        -0.04068922634500564,
        -0.038508200398645415,
        -0.036331920976049974,
        -0.034160367462542106,
        -0.03199351937745175,
        -0.029831356372956492,
        -0.027673858232935385,
        -0.02552100487183431,
        -0.023372776333544718,
        -0.021229152790293252,
        -0.019090114541543912,
        -0.016955642012911255,
        -0.01482571575508632,
        -0.012700316442772813,
        -0.010579424873635267,
        -0.00846302196725765,
        -0.006351088764114094,
        -0.004243606424549351,
        -0.0021405562277706388,
        -4.191957084939753e-05,
        0.00205232203226539,
        0.004142186951724677,
        0.006227693442746083,
        0.008308859646570679,
        0.010385703591409612,
        0.012458243193382021,
        0.01452649625744107,
        0.016590480478292587,
        0.018650213441303184,
        0.020705712623399236,
        0.02275699539395665,
        0.024804079015681525,
        0.026846980645481605,
        0.02888571733532992,
        0.030920306033117483,
        0.0329507635834995,
        0.03497710672873193,
        0.03699935210949975,
        0.03901751626573696,
        0.04103161563743823,
        0.043041666565462246,
        0.04504768529232801,
        0.047049687963001116,
        0.0490476906256742,
        0.051041709232538625,
        0.05303175964054866,
        0.055017857612178096,
        0.057000018816169125,
        0.05897825882827487,
        0.06292303711929127,
        0.06685231525918088,
        0.0766084902045455,
        0.08627040111628252,
        0.10531859608697686,
        0.12401072909912947,
    ]
    varianceA = [
        1.8200197664374549,
        1.4370094334451218,
        1.1219099566714799,
        0.9859599462531226,
        0.8627183857443178,
        0.7511014725956007,
        0.6501483352485907,
        0.5590044022510807,
        0.47690751535658166,
        0.4390359056317844,
        0.4031762929848544,
        0.36925452377926,
        0.3372003590784474,
        0.30694725112809984,
        0.27843213923947446,
        0.2515952639413264,
        0.29502153563617,
        0.2830047657555973,
        0.2712844890319138,
        0.24226714853976236,
        0.23183476228172648,
        0.22671954041689377,
        0.22167105301385384,
        0.2117725713755103,
        0.20213594455246672,
        0.19741479334333528,
        0.19275787514762352,
        0.18816478940262324,
        0.18363514013269241,
        0.19245394381540987,
        0.18774646219447227,
        0.1931708733800091,
        0.18836213710401709,
        0.18362166477667016,
        0.17894905232157457,
        0.1743439006235602,
        0.16980581551151402,
        0.16533440774315933,
        0.16092929299184514,
        0.15659009183542455,
        0.15231642974729748,
        0.14810793708970532,
        0.14396424910937627,
        0.15175373848335327,
        0.14742546106522328,
        0.14316525056148532,
        0.13897274278814326,
        0.13484757886042228,
        0.13078940520904156,
        0.12679787360018158,
        0.12698507216807917,
        0.12666214036631684,
        0.12264576148854993,
        0.11869797410294719,
        0.1148184450420629,
        0.11100684703754489,
        0.10726285877978288,
        0.1090416962702069,
        0.10525922821145543,
        0.10154614385585496,
        0.09790213839890792,
        0.09432691374485798,
        0.09492399037226473,
        0.09134495876803216,
        0.08597378047010444,
        0.08439736588823006,
        0.08426114388629118,
        0.08084296929154224,
        0.07749604380802587,
        0.07557643823153187,
        0.07357590949849342,
        0.07033998972395482,
        0.0693831071831427,
        0.06620062034223427,
        0.06309160518186421,
        0.06188662882497834,
        0.058843681476270004,
        0.05667929073178767,
        0.05449735900229306,
        0.052304126192498454,
        0.0501052963138917,
        0.047906115908252146,
        0.045711439117111015,
        0.044042558111935956,
        0.041353370093888606,
        0.03964591852876113,
        0.03748009108593441,
        0.03572173670686245,
        0.03358414257983117,
        0.0318043449203995,
        0.03000981456603729,
        0.028476576574445737,
        0.02665244238817238,
        0.025279020460443044,
        0.023638525471996946,
        0.02234149404948676,
        0.02096744995618417,
        0.019534675716108427,
        0.01843908912068194,
        0.017230629061443955,
        0.01613237800134248,
        0.015096495307406125,
        0.01424005329279472,
        0.013278697405891386,
        0.012223570633781915,
        0.011558353962214205,
        0.01084803389956623,
        0.01015550946791267,
        0.009439143257793953,
        0.009017014431699322,
        0.008459419775164121,
        0.00800388501313797,
        0.007667388809916037,
        0.007428053991319023,
        0.007291029572886738,
        0.007146673645267026,
        0.007028007621607059,
        0.0070508200234122706,
        0.007083084168032932,
        0.007226153861559947,
        0.007404430732963872,
        0.007752077699396105,
        0.008092286263668957,
        0.008497349440946097,
        0.008882001670948669,
        0.009550658436510925,
        0.009819636480828963,
        0.0105822823697076,
        0.010843199460539351,
        0.01195258512100742,
        0.012513863578351731,
        0.012944164536274989,
        0.01407980127355266,
        0.01525521935110815,
        0.0164698287915165,
        0.017723063978316998,
        0.019014381486472137,
        0.02034325819835555,
        0.020375932932072235,
        0.019953139348489884,
        0.021212731474714613,
        0.023827799654318166,
        0.023357353790689615,
        0.029924305157087,
        0.03715120974705468,
        0.0534583441284658,
        0.07205188800574357,
    ]
    weightsA = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3.999999999999999,
        2.5,
        2.5,
        2.5,
        2.5,
        2.5,
        2.5,
        3.000000000000001,
        3.000000000000001,
        3.000000000000001,
        3.000000000000001,
        3.000000000000001,
        3.500000000000001,
        3.500000000000001,
        6.000000000000002,
        3.500000000000001,
        3.999999999999999,
        3.999999999999999,
        3.999999999999999,
        2.9999999999999996,
        4.499999999999998,
        4.499999999999998,
        5.000000000000001,
        5.000000000000001,
        5.000000000000001,
        5.499999999999999,
        5.499999999999999,
        4.000000000000001,
        6.000000000000002,
        4.333333333333333,
        6.500000000000002,
        4.66666666666667,
        7.000000000000002,
        7.500000000000002,
        7.500000000000002,
        7.999999999999993,
        5.666666666666665,
        5.999999999999999,
        9.000000000000004,
        9.500000000000002,
        10.000000000000002,
        7.333333333333328,
        7.6666666666666705,
        8.33333333333334,
        12.99999999999999,
        9.666666666666671,
        15.499999999999988,
        11.33333333333334,
        12.666666666666673,
        13.999999999999988,
        15.666666666666679,
        17.66666666666668,
        12.4,
        17.499999999999982,
        13.333333333333341,
        15.666666666666679,
        18.333333333333343,
        16.249999999999986,
        19.000000000000025,
        22.74999999999998,
        21.6,
        25.6,
        34.000000000000114,
        26.666666666666682,
        20.66666666666668,
        15.666666666666679,
        11.666666666666675,
        17.000000000000007,
        9.500000000000002,
        9.33333333333334,
        6.999999999999999,
        7.999999999999993,
        4.333333333333333,
        5.000000000000001,
        3.999999999999999,
        3.500000000000001,
        5.000000000000001,
        2.5,
        2,
        2,
        3.000000000000001,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
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
    sumw2 = sum(weightsA .^ 2)
    w = weightsA ./ sqrt(sumw2)
    vols = volA
    strikes = strikeA
    prices, wv = Collocation.weightedPrices(true, strikeA, volA, weightsA, forward, 1.0, tte, vegaFloor=1e-8)
    strikesf, pricesf = AQFED.Collocation.filterConvexPrices(strikeA, prices, wv, forward, tol=1e-6)
    isoc, m = Collocation.makeIsotonicCollocation(strikesf, pricesf, wv, tte, forward, 1.0, deg=3, degGuess=3)
    sol3 = Collocation.Polynomial(isoc)
    ivk3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol3, true, strikeA, forward, 1.0), forward, strikeA, tte, 1.0)
    rmse3 = StatsBase.L2dist(w .* volA, w .* ivk3)
    isoc, m = Collocation.makeIsotonicCollocation(strikesf, pricesf, wv, tte, forward, 1.0, deg=5, degGuess=3, minSlope=1e-5)
    sol5 = Collocation.Polynomial(isoc)
    ivk5 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol5, true, strikeA, forward, 1.0), forward, strikeA, tte, 1.0)
    rmse5 = StatsBase.L2dist(w .* volA, w .* ivk5)
    isoc, m = Collocation.makeIsotonicCollocation(strikesf, pricesf, wv, tte, forward, 1.0, deg=9, degGuess=3, minSlope=1e-5)
    sol9 = Collocation.Polynomial(isoc)
    ivk9 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol9, true, strikeA, forward, 1.0), forward, strikeA, tte, 1.0)
    rmse9 = StatsBase.L2dist(w .* volA, w .* ivk9)
    bsple, m = Collocation.makeExpBSplineCollocation(
        strikes,
        prices,
        wv,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=10,
        minSlope=1e-8,
        rawFit=false,
    )
    ivkbe = @. Black.impliedVolatility(true, Collocation.priceEuropean(bsple, true, strikeA, forward, 1.0), forward, strikeA, tte, 1.0)
    rmsebe = StatsBase.L2dist(w .* volA, w .* ivkbe)
    bspl3, m = Collocation.makeBSplineCollocation(
        strikes,
        prices,
        wv,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=10,
        minSlope=1e-8,
        rawFit=false, N=3,
    )
    ivkb3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(bspl3, true, strikeA, forward, 1.0), forward, strikeA, tte, 1.0)
    rmseb3 = StatsBase.L2dist(w .* volA, w .* ivkb3)
    lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, wv, useVol=false, model=PDDE.Quadratic(), location="Equidistributed", size=10, L=strikes[1], U=strikes[end])
    ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikeA), forward, strikeA, tte, 1.0)
    rmseq = StatsBase.L2dist(w .* volA, w .* ivkq)
    lvgqe = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, wv, useVol=false, model=PDDE.Quadratic(), location="Mid-XX", size=0, L=strikes[1], U=strikes[end])
    ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgqe, true, strikeA), forward, strikeA, tte, 1.0)
    rmseq = StatsBase.L2dist(w .* volA, w .* ivkq)
    solvers = ["LM", "LM-Curve", "GN-LOG", "GN-ALG", "GN-MQ", "LM-LOG", "LM-ALG", "LM-MQ"]
    for guess ∈ ["Constant", "Spline"]
        for solverName ∈ solvers
            lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, wv, useVol=false, model=PDDE.Quadratic(), location="Equidistributed", size=10, L=strikes[1], U=strikes[end], optimizer=solverName, guess=guess)
            ivkq = Black.impliedVolatility.(forward .< strikes, abs.(PDDE.priceEuropean.(lvgq, forward .< strikes, strikeA)), forward, strikeA, tte, 1.0)
            rmseq = StatsBase.L2dist(w .* volA, w .* ivkq)
            elapsed = @belapsed PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, wv, useVol=false, model=PDDE.Quadratic(), location="Equidistributed", size=10, L=strikes[1], U=strikes[end], optimizer=$solverName, guess=$guess)
            println(guess, " ", solverName, " ", rmseq, " ", elapsed)
        end
    end
    #=   

    	kFine = forward.*exp.(range(logmoneynessA[1],stop=logmoneynessA[end], length=1001));
    	p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    	plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bsple,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Exp B-Spline")
    	plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bspl3,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Cubic B-spline")
    	plot!(p3, size=(480,320))
    	savefig("~//mypapers/eqd_book/spxw170324_170316_vol_bspl.pdf")
    	p4=plot(log.(kFine./forward), Collocation.density.(bsple,kFine),label="Exp B-spline",xlab="Log-moneyness",ylab="Probability density",color=2)
    	plot!(p4,log.(kFine./forward), Collocation.density.(bspl3,kFine),label="Cubic B-spline",color=3)
    	plot!(p4,yscale=:log10, legend=:topleft)
    	plot!(p4, size=(480,320))
    	savefig("~//mypapers/eqd_book/spxw170324_170316_dens_bspl.pdf")

    	p3=plot(logmoneynessA, volA.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    	plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, Collocation.priceEuropean.(sol3,true,kFine,forward,1.0), forward, (kFine), tte, 1.0) .* 100, label="Cubic collocation")
    	plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, Collocation.priceEuropean.(sol5,true,kFine,forward,1.0), forward, kFine, tte, 1.0) .* 100, label="Quintic collocation")
    	plot!(p3,size=(480,380))
    	savefig(p3,"~//mypapers/eqd_book/spxw170324_170316_vol_schaback.pdf")

    	plot(log.(kFine./forward), Collocation.density.(sol3,kFine),label="Cubic collocation", yscale=:log10, xlab="Forward log-moneyness",ylab="Probability density")
    	plot!(log.(kFine./forward), Collocation.density.(sol5,kFine),label="Quintic collocation")
    	plot!(ylim=(1e-16,0.0),legend=:topleft)
    	plot!(size=(480,380))
    	savefig("~//mypapers/eqd_book/spxw170324_170316_dens_schaback.pdf")

    	p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    	plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,PDDE.priceEuropean.(lvgq,true,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Quadratic LVG")
    	plot!(p3, size=(480,380))
    	savefig(p3,"~//mypapers/eqd_book/spxw170324_170316_vol_lvgq.pdf")
    	p2 = plot(kFine,(PDDE.derivativePrice.(lvgqe,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgqe,true,kFine)).*10000, label="Mid-XX n=147",yscale=:log10, xlab="Forward log-moneyness",ylab="Probability density")
    	plot!(p2,kFine,(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label="Equidistributed n=10",yscale=:log10, xlab="Forward log-moneyness",ylab="Probability density")
    	plot!(p2,legend=:topleft)
    	plot!(p2, size=(480,380))
    	savefig(p2,"~//mypapers/eqd_book/spxw170324_170316_dens_lvgq.pdf")
    	=#

    allStrikes = vcat(0.0, strikesf, forward * exp(logmoneynessA[end] * 3))
    allPrices = vcat(forward, pricesf, 0.0)
    allWeights = vcat(sum(wv), wv, sum(wv))
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
    )
    rmseScha = StatsBase.rmsd(volA, ivkScha)
    allStrikes = vcat(0.0, strikes, forward * exp(logmoneynessA[end] * 3))
    allPrices = vcat(forward, prices, 0.0)
    println("scha ", rmseScha)
    csf, rmseSchaFit = fitConvexSchabackRationalSpline(
        allStrikes, allPrices,
        allWeights,
        leftB,
        rightB;
        penalty=1.0,
    )
    ivkSchaFit = @. Black.impliedVolatility(
        true,
        csf(strikeA),
        forward,
        strikeA,
        tte,
        1.0,
    )
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
    savefig(p3,"~//mypapers/eqd_book/spxw170324_170316_vol_fengler_rbf.pdf")
    p3 = plot(kFine, AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,forward.*exp.(kFine)),label="string("Fengler λ=",200))

    pdf(pp,z) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.Black.blackScholesFormula(true, y, forward, pp(log(y/forward))*tte,1.0,1.0),x),z)
    p3 =plot(kFine,@.(pdf(rbf, forward*exp(kFine))),label="RBF")

     plot!(kFine, AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,forward.*exp.(kFine)),label=string("Fengler λ=",200),xlab="Forward log-moneyness",ylab="Probability density")
    plot!(ylim=(1e-8,0.02),yscale=:log10)
    plot!(p3,size=(480,380))
    savefig(p3,"~//mypapers/eqd_book/spxw170324_170316_dens_fengler_rbf.pdf")
    ##Schaback
    dev = 1.0
    kFine = range(logmoneynessA[1]*dev,stop=logmoneynessA[end]*dev, length=1001);

    p3=plot(logmoneynessA, volA.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(p3, kFine, Black.impliedVolatility.(true, max.(cs.(forward.*exp.(kFine)),1e-16), forward, forward.*exp.(kFine), tte, 1.0) .* 100, label=string("Schaback λ=",0))
    plot!(p3, kFine, Black.impliedVolatility.(true, max.(csf.(forward.*exp.(kFine)),1e-16), forward, forward.*exp.(kFine), tte, 1.0) .* 100, label=string("Schaback λ=",1))
    plot!(p3,size=(480,380))
    savefig(p3,"~//mypapers/eqd_book/spxw170324_170316_vol_schaback.pdf")

    plot(logmoneynessA, AQFED.Math.evaluateSecondDerivative.(cs,strikeA),label="Schaback λ=0", yscale=:log10, xlab="Forward log-moneyness",ylab="Probability density")
    plot!(logmoneynessA, AQFED.Math.evaluateSecondDerivative.(csf,strikeA),label="Schaback λ=1", yscale=:log10, xlab="Forward log-moneyness",ylab="Probability density")
    plot!(ylim=(1e-16,0.0),legend=:topleft)
    plot!(size=(480,380))
    savefig("~//mypapers/eqd_book/spxw170324_170316_dens_schaback.pdf")

    =#
    λs = [1e-10, 100.0, 1000.0, 10000.0]
    results = []
    for λ in λs
        fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikeA, prices, wv, λ=λ, solver="GI")
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
    savefig(p3,"~//mypapers/eqd_book/spxw170324_170316_vol_fengler.pdf")

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
    savefig(p3, "~//mypapers/eqd_book/spxw170324_170316_dens_fengler.pdf")
    plot!(p3, ylim=(1e-8,0.02),yscale=:log10,legend=:topleft)
    plot!(p3,size=(480,380))
    savefig(p3, "~//mypapers/eqd_book/spxw170324_170316_dens_fengler_log.pdf")

    =#
    kernel2 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=2)
    ivkMLN2 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel2, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN2 = StatsBase.L2dist(w .* ivkMLN2, w .* vols)
    kernel3 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=3)
    ivkMLN3 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel3, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN3 = StatsBase.L2dist(w .* ivkMLN3, w .* vols)

    kernel4 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=4)
    ivkMLN4 = @. AQFED.Black.impliedVolatility(
        true,
        AQFED.VolatilityModels.priceEuropean.(kernel4, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN4 = StatsBase.L2dist(w .* ivkMLN4, w .* vols)
    kernel6 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=6)
    ivkMLN6 = @. AQFED.Black.impliedVolatility(
        true,
        AQFED.VolatilityModels.priceEuropean.(kernel6, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN6 = StatsBase.L2dist(w .* ivkMLN6, w .* vols)

    #=
    	 plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    	 plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel2,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 2")
    	 plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel4,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 4")
    	 plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel6,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 6")
    	plot!(size=(800,320),margin=3Plots.mm)
    	savefig("~//mypapers/eqd_book/vol_spw_1m_050218_mln6.pdf")

    	plot(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel2,kFine),label="Mixture of 2", color=2,xlab="Forward log-moneyness",ylab="Probability density")
    	plot!(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel4,kFine),label="Mixture of 4",color=3)
    	plot!(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel6,kFine),label="Mixture of 6",color=4)
    	plot!(size=(800,320),margin=3Plots.mm)
    	 savefig("~//mypapers/eqd_book/density_spw_1m_050218_mln6.pdf")
    	=  =#

    ##LVG vs AH
    w1 = ones(length(prices))
    ah = PDDE.calibrateDiscreteLogLVG(tte, forward, strikes, prices, w1, useVol=true, model=PDDE.Quadratic(), location="Equidistributed", optimizer="GN-MQ", L=strikes[1] / 2, U=strikes[end] * 2, isC3=true, size=10, discreteSize=1000)
    lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, w1, useVol=true, model=PDDE.Quadratic(), location="Equidistributed", optimizer="LM-ALG", L=strikes[1] / 2, U=strikes[end] * 2, isC3=true, size=10)
    ah100 = PDDE.calibrateDiscreteLogLVG(tte, forward, strikes, prices, w1, useVol=true, model=PDDE.Quadratic(), location="Equidistributed", optimizer="GN-MQ", L=strikes[1] / 2, U=strikes[end] * 2, isC3=true, size=10, discreteSize=101)
    ah1000 = PDDE.calibrateDiscreteLogLVG(tte, forward, strikes, prices, w1, useVol=true, model=PDDE.Quadratic(), location="Equidistributed", optimizer="GN-MQ", L=strikes[1] / 2, U=strikes[end] * 2, isC3=true, size=10, discreteSize=1000)

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
     savefig("~//mypapers/eqd_book/density_spw_1m_050218ahspl_d.pdf")
     p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5)
    plot!(p3,log.(kFine./forward), Black.impliedVolatility.(forward.<kFine, max.(1e-32,PDDE.priceEuropean.(lvgq,forward.<kFine,kFine)), forward, (kFine), tte, 1.0) .* 100, label="LVG")
     plot!(p3,log.(kFine./forward), Black.impliedVolatility.(forward.<kFine, max.(1e-32,PDDE.priceEuropean.(ah100,forward.<kFine,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Andreasen-Huge l=101")
    plot!(p3,log.(kFine./forward), Black.impliedVolatility.(forward.<kFine, max.(1e-32,PDDE.priceEuropean.(ah1000,forward.<kFine,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Andreasen-Huge l=1000")
     plot!(p3,size=(480,380))
     savefig("~//mypapers/eqd_book/vol_spw_1m_050218ahspl.pdf")

     plot(log.(kFine./forward), Black.impliedVolatility.(forward.<kFine, max.(1e-32,PDDE.priceEuropean.(ah100,forward.<kFine,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Andreasen-Huge l=101")
     plot!(log.(kFine./forward), Black.impliedVolatility.(forward.<kFine, max.(1e-32,PDDE.priceEuropean.(lvgq,forward.<kFine,kFine)), forward, (kFine), tte, 1.0) .* 100, label="LVG",xlab="Forward log-moneyness", ylab="Implied volatility in %",z_order=:back,linestyle=:dot)
     plot!(ylims=(5,50),xlim=(-0.3,0.12))
    plot!(size=(480,380))
    savefi("~//mypapers/eqd_book/vol_spw_1m_050218ahspl_d100.pdf")

    =#
end

using AQFED.Black
@testset "clarkgbpusd" begin
    gbpRate = 0.48 / 100
    usdRate = 0.38 / 100
    gbpRate = 0.59 / 100
    usdRate = 0.61 / 100
    tte = (31 + 30 + 31) / 365.0
    S0 = 1.4153
    SL = 1.3159
    PL = 27.4 / 100
    sigmaR = 9.5 / 100
    sigmaL = 21.9 / 100
    forward = S0 * exp((usdRate - gbpRate) * tte)
    muL = log(SL / S0) / tte
    muR = log((forward - PL * SL) / (S0 * (1 - PL))) / tte
    function mixPrice(strike)
        return (1 - PL) * blackScholesFormula(true, strike, S0 * exp(muR * tte), sigmaR^2 * tte, 1.0, exp(-usdRate * tte)) + PL * blackScholesFormula(true, strike, S0 * exp(muL * tte), sigmaL^2 * tte, 1.0, exp(-usdRate * tte))
    end
    atmVol = impliedVolatility(true, mixPrice(forward), forward, forward, tte, exp(-usdRate * tte))
    dev = exp(5 * vols[3] * sqrt(tte))
    kFine = range(forward / dev, forward * dev, length=100)
    pdf(z) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> mixPrice(y), x), z)

    tte = 31 / 365.0
    S0 = 1.4636
    SL = 1.4464
    PL = 42.7 / 100
    sigmaR = 8.2 / 100
    sigmaL = 14.5 / 100
    tte = 7.0 / 365.0
    S0 = 1.4358
    SL = 1.2900
    PL = 15.3 / 100
    sigmaR = 35.3 / 100
    sigmaL = 49.7 / 100
    tte = 8.0 / 365.0
    S0 = 1.4203
    SL = 1.3914
    PL = 39.1 / 100
    sigmaR = 18.3 / 100
    sigmaL = 37.5 / 100
    tte = 16 / 365.0
    S0 = 1.4504
    SL = 1.3865
    PL = 27.5 / 100
    sigmaR = 13.2 / 100
    sigmaL = 22.4 / 100
end

@testset "deltaeurusdbi" begin
    tte = 61.0 / 365
    spot = 1.1500
    forward = spot * exp((0.0375 - 0.01737) * tte)
    quotes = [0.1124, 0.0164, 0.0374, 0.0576, 0.1264]
    vols = [0.2130047, 0.1576571, 0.1124, 0.1000571, 0.0865747]
    deltas = [0.1, 0.25, 0.5, 0.75, 0.9]
    k = zeros(Float64, length(deltas))
    for (i, delta) in enumerate(deltas)
        # phinv(delta) = (ln f - lnk + 0.5*vsqrt^2 )/ vsqrt
        k[i] = forward * exp(-vols[i] * sqrt(tte) * norminv(delta) + vols[i]^2 * tte / 2)
    end
    reverse!(k)
    volsk = reverse(vols)
    #atm dns
    #k[3] = forward*exp(vol[3]^2/tte)

    nIter = 1
    function expq(k, coeff; nIter=32)
        v = vols[3]
        for i ∈ 1:nIter
            x = normcdf(log(forward / k) / (v * sqrt(tte)))
            v = exp(coeff[1] + coeff[2] * x + coeff[3] * x^2 + coeff[4] * x^3 + coeff[5] * x^4)
        end
        return v
    end

    function obj!(fvec::Z, c::AbstractArray{W})::Z where {Z,W}
        for (i, strike) ∈ enumerate(k)
            voli = expq(strike, c, nIter=nIter)
            fvec[i] = (voli - volsk[i])
        end
        fvec
    end
    xv = [log(vols[3]), 0.0, 0.0, 0.0, 0.0]
    fit = LeastSquaresOptim.optimize!(
        LeastSquaresProblem(x=xv, (f!)=obj!, autodiff=:forward, #:forward is 4x faster than :central
            output_length=length(vols)),
        LevenbergMarquardt();
        iterations=1000,
    )
    pdf(z) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.Black.blackScholesFormula(true, y, forward, expq(y, xv, nIter=nIter)^2 * tte, 1.0, 1.0), x), z)



    #plot(log.(kFine ./ forward), pdf.(kFine))
    #pdf is good for clark(nIter=32), lvgq, bad for clark(nIter=1) interp is bad for collo (flat poly => kink). RBF goes negative!
    #alternative: RBF on delta, log(vol), apply exp on top to guarantee > 0. Not necessarily better except for more points.
    rbf = AQFED.VolatilityModels.calibrateMultiquadric(tte, forward, deltas, (vols), w1, knots=deltas)
    function rbfStrike(k; nIter=32)
        v = sqrt(rbf(0.0))
        for i ∈ 1:nIter
            delta = normcdf((log(forward / k) + v^2 * tte / 2) / (v * sqrt(tte)))
            v = sqrt(rbf(delta))
        end
        return v
    end

    #clark data EURUSD 1Y
    tte = 1.0
    k = [1.2034, 1.2050, 1.3620, 1.5410, 1.5449]
    vols = [19.50, 19.48, 18.25, 18.90, 18.92] ./ 100
    deltas = reverse([0.1, 0.25, 0.5, 0.75, 0.9])
    kDNSpips = 1.3620
    q = 3.46 / 100
    r = 2.94 / 100
    forward = 1.3465 * exp((r - q) * tte)
end

@testset "deltavol-generated" begin
    deltas = [0.1, 0.25, 0.5, 0.75, 0.9]
    ttes = [1.0]
    vols = zeros(length(deltas))
    strikes = zeros(length(deltas))
    forward = 1.0
    df = 1.0
    size = 2
    α = zeros(size)
    kx = zeros(size)
    xv = zeros(Float64, size * 3 - 2 + size) #guess of minim
    for tte = ttes
        for i = 1:1
            rand!(xv)
            AQFED.VolatilityModels.toPositiveHypersphere!(α, 1.0, 4 * xv[1:size-1] .- 2.0)
            AQFED.VolatilityModels.toPositiveHypersphere!(kx, sqrt(forward), 4 * xv[size:2size-2] .- 2.0)
            σ = exp.(xv[end-size+1:end]) ./ 30 * sqrt(tte)
            println("σ=", σ)
            xg = @. (ifelse(α == 0.0, kx / 1e-8, kx / α))

            kernel = AQFED.VolatilityModels.LognormalKernel(xg, σ, α)
            strike = forward

            mPrice = AQFED.VolatilityModels.priceEuropean(kernel, true, strike)
            atmVol = AQFED.Black.impliedVolatilitySRHalley(true, mPrice, forward, strike, tte, 1.0, 0e-14, 64, AQFED.Black.Householder())
            dev = 6 * atmVol * sqrt(tte)
            println("atmVol ", atmVol)
            for (k, delta) = enumerate(deltas)
                objDelta = function (strike)
                    mPrice = AQFED.VolatilityModels.priceEuropean(kernel, strike > forward, strike)
                    v = AQFED.Black.impliedVolatility(strike > forward, mPrice, forward, strike, tte, 1.0)
                    return normcdf((log(forward / strike) + v^2 * tte / 2) / (v * sqrt(tte))) - delta
                end
                der = x -> ForwardDiff.derivative(objDelta, x)

                strike = find_zero(objDelta, (forward * exp(-dev), forward * exp(dev)), Roots.A42())
                mPrice = AQFED.VolatilityModels.priceEuropean(kernel, strike > forward, strike)
                strikes[k] = strike
                vols[k] = AQFED.Black.impliedVolatilitySRHalley(strike > forward, mPrice, forward, strike, tte, 1.0, 0e-14, 64, AQFED.Black.Householder())

            end
            function expq(k, coeff; nIter=32)
                v = vols[3]
                for i ∈ 1:nIter
                    x = normcdf(log(forward / k) / (v * sqrt(tte)))
                    v = (coeff[1] + coeff[2] * x + coeff[3] * x^2 + coeff[4] * x^3 + coeff[5] * x^4)
                end
                return v
            end
            nIter = 32

            function obj!(fvec::Z, c::AbstractArray{W})::Z where {Z,W}
                for (i, strike) ∈ enumerate(strikes)
                    voli = expq(strike, c, nIter=nIter)
                    fvec[i] = (voli - vols[i])
                end
                fvec
            end
            xvv = [(vols[3]), 0.0, 0.0, 0.0, 0.0]
            fit = LeastSquaresOptim.optimize!(
                LeastSquaresProblem(x=xvv, (f!)=obj!, autodiff=:forward, #:forward is 4x faster than :central
                    output_length=length(vols)),
                LevenbergMarquardt();
                iterations=1000,
            )
            pdf(z) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.Black.blackScholesFormula(true, y, forward, expq(y, xvv, nIter=nIter)^2 * tte, 1.0, 1.0), x), z)
            logm = range(-dev, dev, length=50)
            pdfm = pdf.(forward .* exp.(logm))
            if minimum(pdfm) < 0
                println(tte, " Arbitrage ", findmin(pdfm), " ", pdfm, " vols=", vols, " kernel=", kernel)
            end
        end
    end
    #vols=[0.13883768868175392, 0.15398939744604903, 0.19316926261451353, 0.3456903708470549, 0.6801677129291079]; tte = 12/365
    #vols=[0.23446892718815637, 0.243420206256986, 0.26281527721814973, 0.31389302486908144, 0.48275162267904237] tte= 1.0 vols=[0.9980599635234404, 0.9846107749610908, 0.47324211303240765, 0.26415403785544234, 0.21034435763444098] vols=[0.2081716217819625, 0.12184296438342047, 0.07085040105862439, 0.05812732426105612, 0.053168066870935594] vols=[0.17443747778078533, 0.22604050568250061, 0.3461878618986742, 0.5151816576568277, 0.5663597569275649] vols=[0.058025135362002836, 0.06826897785970193, 0.09568203590768795, 0.16941789029441737, 0.23311395611512492] kernel = AQFED.VolatilityModels.LognormalKernel{Float64, Float64, Float64}([0.018201583907926547, 1.0482945492113935], [0.0602820917562265, 0.064496893459696], [0.04688368024838003, 0.9531163197516199])
    #last item: pdf is bad: increases close to zero! low strikes. negative, oscillates, fit is not exact. same problem without exp, with quadratic. Smile is extreme: large variation in vol between 10 delta and 50 delta. We may generate ignore errors and points with too large vols to find better example?
    # LVGQ works, BSPL works, but EBSPL does not, Collo5 not so great either. LVGQ needs L<1e-4 (very small - does it hurt to always set it to almost 0?) is it a flat like extrapolation? what if lambda0=2lambda1? How to do linear wings?
    # what do we want: possibility of nearly exact interpolation, even if this usually means overfitting the quotes. Ability to control the wings. No arbitrage. Smooth density (Cubic spline is not smooth - produces fake modes)
    #1.0 Arbitrage (-24.368462282944535, 24) [0.00043685500072030044, 0.0008988714073874769, 0.0017899746674895876, 0.0034497312881712067, 0.006434469029068635, 0.01161526777245317, 0.020292464828409684, 0.03431068919660496, 0.05614526628032357, 0.08891722783041311, 0.13628490219423828, 0.202161589532859, 0.29022752928662054, 0.40324373618417475, 0.542232718464918, 0.7056557313316594, 0.8887693950939282, 1.083363933751233, 5.866226772731409e-10, 2.4018574616132887e-7, 1.9437301732080277e-9, 5.571915015374621, 13.719050255358198, -24.368462282944535, -0.7116563117474008, 0.44307425079684926, 0.7100329460958648, 0.7676938563962281, 0.7386537524151113, 0.6637233730676062, 0.5652354985496171, 0.45919853489642826, 0.35724544067165676, 0.2668343843992478, 0.19168184028940782, 0.1325831656760723, 0.08837014630387052, 0.05679164326310615, 0.035207866109240704, 0.021065836479196354, 0.012170562527245854, 0.006792687268721856, 0.003664065675221616, 0.0019109173309337438, 0.0009638563387205489, 0.0004703017769567561, 0.00022202794094690293, 0.00010142717377331243, 4.483817535453293e-5, 1.9182596246420367e-5] vols=[0.2649617370889747, 0.27488788986899443, 0.18486963688066518, 0.1275390002811324, 0.10683347541763455] kernel=LognormalKernel{Float64, Float64, Float64}([1.54140989144187, 0.9237384043983021], [0.049936695609994135, 0.06102359064494718], [0.12346627163691423, 0.8765337283630857])

end

@testset "deltavolraw" begin
    #AUDNZD 2014/07/02
    tte = 7.0 / 365
    spot = 1.0784
    forward = 1.07845
    df = 0.999712587139
    rr10 = 0.35 / 100
    rr25 = 0.4 / 100
    volAtm = 5.14 / 100
    bf25 = 0.25 / 100
    bf10 = 1.175 / 100
    vols = AQFED.VolatilityModels.convertQuotesToDeltaVols(AQFED.VolatilityModels.SmileConvention(), volAtm, bf25, rr25, bf10, rr10)
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, true, false) #prem currency is aud
    k = strikeForDelta(vols, conv)
    smileSpl = calibrateSmile(AQFED.VolatilityModels.SplineSmile, k, vols, forward, tte)
    @test rmsd(smileSpl.(k), vols) < 1e-12
    smileSplD = calibrateSmile(AQFED.VolatilityModels.SplineDeltaSmile, k, vols, forward, tte)
    @test rmsd(smileSplD.(k), vols) < 1e-12
    smileSVI = calibrateSmile(AQFED.VolatilityModels.SVISmile, k, vols, forward, tte)
    @test rmsd(smileSVI.(k), vols) < 2e-4
    smileSpl0 = calibrateSmile(AQFED.VolatilityModels.SplineSmile, k, vols, forward, tte, isFlat=true)
    dev = exp(4 * vols[3] * sqrt(tte))
    kFine = collect(range(k[1] / dev, stop=k[end] * dev, length=201))
    pdf(pol, z) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.Black.blackScholesFormula(true, y, forward, pol(y)^2 * tte, 1.0, 1.0), x), z)
    @test minimum(map(x -> pdfd(smileSpl0, x), kFine)) < 0
    @test minimum(map(x -> pdfd(smileSpl, x), kFine)) >= 0
    smileQ = calibrateSmile(AQFED.VolatilityModels.ExpPolynomialSmile, k, vols, forward, tte)
    rmsd(smileQ.(k), vols)
    @test minimum(map(x -> pdfd(smileQ, x), kFine)) >= 0

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
    #volsc = AQFED.VolatilityModels.convertQuotesToDeltaVols(conv, volAtm, bf25, rr25, bf10, rr10) => no signif change

    forward = 25.508
    df = 1.0
    tte = 32 / 365 #EURCZK  12/16/2019
    deltas = [-0.05, -0.25, 0.5, 0.25, 0.05]
    vols = [3.715, 2.765, 2.83, 3.34, 4.38] ./ 100
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, true, true) #prem 


    tte = 60 / 365
    vols = [4.0, 3.03, 3.06, 3.65, 4.790] ./ 100
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, true, true) #prem currency is eur
    k = strikeForDelta(vols, conv, deltas=deltas)
    kernel = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.LognormalMixtureSmile, k, vols, forward, tte)
    kernel3 = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.LognormalMixture3Smile, k, vols, forward, tte)
    dev = exp(4 * vols[3] * sqrt(tte))
    kFine = collect(range(k[1] / dev, stop=k[end] * dev, length=201))
    #callDelta = @.(normcdf((log(forward / k)-vols^2*tte/2) / (vols * sqrt(tte))))	
    smileSplD0 = calibrateSmile(AQFED.VolatilityModels.SplineDeltaSmile, k, vols, forward, tte, isFlat=true)
    sabr = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.SABRSmile, k, vols, forward, tte)
    xssvi = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.XSSVISmile, k, vols, forward, tte)
    smileSVI = calibrateSmile(AQFED.VolatilityModels.SVISmile, k, vols, forward, tte)
    smileQ = calibrateSmile(AQFED.VolatilityModels.ExpPolynomialSmile, k, vols, forward, tte)
    smileQ1 = calibrateSmile(AQFED.VolatilityModels.ExpPolynomialSmile, k, vols, forward, tte, nIter=1)
    xssvi = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.XSSVISmile, k, vols, forward, tte)
    sabr = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.SABRSmile, k, vols, forward, tte)
    @test rmsd(smileQ.(k), vols) < 1e-7
    @test rmsd(smileQ1.(k), vols) < 1e-7
    @test minimum(map(x -> pdfd(smileQ, x), kFine)) >= 0

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

    # #USDAED 1/24/2023
    spot = 3.67
    tte = 7 / 365
    df = exp(-3.255 / 100 * tte)
    forward = spot * exp(-3.18 / 100 * tte) / df
    volAtm = 0.34 / 100
    rr25 = 0.169 / 100
    rr10 = 0.445 / 100
    bf25 = 0.098 / 100
    bf10 = 0.433 / 100
    #USDAED 9m
    tte = 9 * 30 / 360
    df = exp(-3.255 / 100 * tte)
    forward = spot * exp(-3.18 / 100 * tte) / df
    volAtm = 0.32 / 100
    rr25 = 0.152 / 100
    rr10 = 0.412 / 100
    bf25 = 0.084 / 100
    bf10 = 0.392 / 100
    vols = AQFED.VolatilityModels.convertQuotesToDeltaVols(AQFED.VolatilityModels.SmileConvention(), volAtm, bf25, rr25, bf10, rr10)
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, true, true) #prem currency is aud
    k = strikeForDelta(vols, conv)
    smileSpl = calibrateSmile(AQFED.VolatilityModels.SplineSmile, k, vols, forward, tte)
    @test rmsd(smileSpl.(k), vols) < 1e-12
    smileSplD = calibrateSmile(AQFED.VolatilityModels.SplineDeltaSmile, k, vols, forward, tte)
    @test rmsd(smileSplD.(k), vols) < 1e-8
    smileSVI = calibrateSmile(AQFED.VolatilityModels.SVISmile, k, vols, forward, tte)
    @test rmsd(smileSVI.(k), vols) < 2e-4
    smileSpl0 = calibrateSmile(AQFED.VolatilityModels.SplineSmile, k, vols, forward, tte, isFlat=true)
    dev = exp(4 * vols[3] * sqrt(tte))
    kFine = exp.(collect(range(log(k[1] / dev), stop=log(k[end] * dev), length=201)))
    yFine = log.(kFine ./ forward)
    smileQ = calibrateSmile(AQFED.VolatilityModels.ExpPolynomialSmile, k, vols, forward, tte, nIter=32)
    smileSVI = calibrateSmile(AQFED.VolatilityModels.SVISmile, k, vols, forward, tte)
    pdfd(pol, z) = FiniteDifferences.central_fdm(5, 2)(y -> AQFED.Black.blackScholesFormula(true, y, forward, pol(y)^2 * tte, 1.0, 1.0), z)
    xssvi = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.XSSVISmile, k, vols, forward, tte)
    lvg = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.LVGSmile, k, vols, forward, tte, L=k[1] / 1.25)
    bspl = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.BSplineCollocationSmile, k, vols, forward, tte)
    quintic = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.QuinticCollocationSmile, k, vols, forward, tte)
    pdfd2(pol, z) = -(10000 / z)^2 * (2 * AQFED.Black.blackScholesFormula(true, z, forward, pol(z)^2 * tte, 1.0, 1.0) - AQFED.Black.blackScholesFormula(true, z * 0.9999, forward, pol(z * 0.9999)^2 * tte, 1.0, 1.0) - AQFED.Black.blackScholesFormula(true, z * 1.0001, forward, pol(z * 1.0001)^2 * tte, 1.0, 1.0))
    smileSplDW = AQFED.VolatilityModels.makeTotalVarianceFunction(smileSplD, forward, tte)
    smileSVIW = AQFED.VolatilityModels.makeTotalVarianceFunction(smileSVI, forward, tte)
    smileQW = AQFED.VolatilityModels.makeTotalVarianceFunction(smileQ, forward, tte)
    xssviW = AQFED.VolatilityModels.makeTotalVarianceFunction(xssvi, forward, tte)
    lvgW = AQFED.VolatilityModels.makeTotalVarianceFunction(lvg, forward, tte)
    bsplW = AQFED.VolatilityModels.makeTotalVarianceFunction(bspl, forward, tte)
    sabr = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.SABRSmile, k, vols, forward, tte)

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
    #EURTRY 29/11/2022 for=eur
    spot = 19.3483
    tte = 184 / 365 #start of breakdown/bimodality
    df = exp(-1.16702 / 100 * tte) #foreign df
    forward = spot * exp(36.77 / 100 * tte) * df
    vols = [38.44209, 29.79629, 22.12, 20.41129, 17.29409] ./ 100
    volAtm = 22.12 / 100
    bf25 = 2.187 / 100
    rr25 = 9.385 / 100
    bf10 = 7.633 / 100
    rr10 = 21.148 / 100
    tte = 1.0
    vols = [50.37981, 40.71189, 31.13, 29.14389, 23.25981] ./ 100
    vols = [0.24077330183795542, 0.28640409729845756, 0.31129999999999997, 0.40208409729845757, 0.5119733018379554]
    df = exp(-1.784 / 100 * tte)
    forward = spot * exp(37.73 / 100 * tte) * df
    volAtm = 31.13 / 100
    bf25 = 2.931 / 100
    rr25 = 11.568 / 100
    bf10 = 9.307 / 100
    rr10 = 27.12 / 100
    tte = 2.0
    df = exp(-1.95 / 100 * tte)
    forward = spot * exp(37.65 / 100 * tte) * df
    vols = [52.12681, 41.48931, 32.08400, 30.50831, 24.78681] ./ 100
    rr10 = 27.34 / 100
    rr25 = 10.981 / 100
    volAtm = 32.084 / 100
    bf25 = 3.46 / 100
    bf10 = 10.866 / 100
    vols = AQFED.VolatilityModels.convertQuotesToDeltaVols(AQFED.VolatilityModels.SmileConvention(), volAtm, bf25, rr25, bf10, rr10)
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, true, true) #prem currency is aud
    k = strikeForDelta(vols, conv)
    smileSpl = calibrateSmile(AQFED.VolatilityModels.SplineSmile, k, vols, forward, tte)
    @test rmsd(smileSpl.(k), vols) < 1e-12
    smileSplD = calibrateSmile(AQFED.VolatilityModels.SplineDeltaSmile, k, vols, forward, tte)
    @test rmsd(smileSplD.(k), vols) < 1e-8
    smileSVI = calibrateSmile(AQFED.VolatilityModels.SVISmile, k, vols, forward, tte)
    @test rmsd(smileSVI.(k), vols) < 7e-3
    smileSpl0 = calibrateSmile(AQFED.VolatilityModels.SplineSmile, k, vols, forward, tte, isFlat=true)
    dev = exp(4 * vols[3] * sqrt(tte))
    kFine = exp.(collect(range(log(k[1] / dev), stop=log(k[end] * dev), length=201)))
    yFine = log.(kFine ./ forward)
    smileQ = calibrateSmile(AQFED.VolatilityModels.ExpPolynomialSmile, k, vols, forward, tte, nIter=32)
    smileSVI = calibrateSmile(AQFED.VolatilityModels.SVISmile, k, vols, forward, tte)
    pdfd(pol, z) = FiniteDifferences.central_fdm(5, 2)(y -> AQFED.Black.blackScholesFormula(true, y, forward, pol(y)^2 * tte, 1.0, 1.0), z)
    xssvi = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.XSSVISmile, k, vols, forward, tte)
    lvg = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.LVGSmile, k, vols, forward, tte)
    bspl = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.BSplineCollocationSmile, k, vols, forward, tte)
    quintic = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.QuinticCollocationSmile, k, vols, forward, tte)
    pdfd2(pol, z) = -(10000 / z)^2 * (2 * AQFED.Black.blackScholesFormula(true, z, forward, pol(z)^2 * tte, 1.0, 1.0) - AQFED.Black.blackScholesFormula(true, z * 0.9999, forward, pol(z * 0.9999)^2 * tte, 1.0, 1.0) - AQFED.Black.blackScholesFormula(true, z * 1.0001, forward, pol(z * 1.0001)^2 * tte, 1.0, 1.0))
    smileSplDW = AQFED.VolatilityModels.makeTotalVarianceFunction(smileSplD, forward, tte)
    smileSVIW = AQFED.VolatilityModels.makeTotalVarianceFunction(smileSVI, forward, tte)
    smileQW = AQFED.VolatilityModels.makeTotalVarianceFunction(smileQ, forward, tte)
    xssviW = AQFED.VolatilityModels.makeTotalVarianceFunction(xssvi, forward, tte)
    lvgW = AQFED.VolatilityModels.makeTotalVarianceFunction(lvg, forward, tte)
    bsplW = AQFED.VolatilityModels.makeTotalVarianceFunction(bspl, forward, tte)
    sabr = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.SABRSmile, k, vols, forward, tte)
    @test rmsd(sabr.(k), vols) < 6e-3

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

    #manufactured looks like there is an outlier, quintic takes it too well and neg density, sabr svi xssvi less odd, except for extrap.
    #convention = normcdf((log(forward / strike) + v^2 * tte / 2) / (v * sqrt(tte)))
    tte = 1.0
    df = 1.0
    spot = 1.0
    forward = 1.0
    deltas = [-0.1, -0.25, 0.5, 0.25, 0.1]
    #deltas=[0.1,0.25,0.5,0.75,0.9]
    #vols=reverse([0.058025135362002836, 0.06826897785970193, 0.09568203590768795, 0.16941789029441737, 0.23311395611512492])
    vols = [0.260, 0.260, 0.195, 0.127, 0.127]

    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, false, true) #prem currency 
    #k = @.(AQFED.VolatilityModels.strikeForDelta(CallForwardDelta(), deltas, vols, tte, forward, df))
    k = strikeForDelta(vols, conv)
    prices, wv = Collocation.weightedPrices(true, k, vols, w1, forward, 1.0, tte)
    @test AQFED.Collocation.isArbitrageFree(k, prices, forward)[1]
    smileSpl = calibrateSmile(AQFED.VolatilityModels.SplineSmile, k, vols, forward, tte)
    @test rmsd(smileSpl.(k), vols) < 1e-12
    smileSplD = calibrateSmile(AQFED.VolatilityModels.SplineDeltaSmile, k, vols, forward, tte)
    @test rmsd(smileSplD.(k), vols) < 1e-8
    smileSVI = calibrateSmile(AQFED.VolatilityModels.SVISmile, k, vols, forward, tte)
    @test rmsd(smileSVI.(k), vols) < 2e-2
    smileSpl0 = calibrateSmile(AQFED.VolatilityModels.SplineSmile, k, vols, forward, tte, isFlat=true)
    dev = exp(4 * vols[3] * sqrt(tte))
    kFine = exp.(collect(range(log(k[1] / dev), stop=log(k[end] * dev), length=201)))
    yFine = log.(kFine ./ forward)
    smileQ = calibrateSmile(AQFED.VolatilityModels.ExpPolynomialSmile, k, vols, forward, tte, nIter=32)
    smileSVI = calibrateSmile(AQFED.VolatilityModels.SVISmile, k, vols, forward, tte)
    pdfd(pol, z) = FiniteDifferences.central_fdm(5, 2)(y -> AQFED.Black.blackScholesFormula(true, y, forward, pol(y)^2 * tte, 1.0, 1.0), z)
    xssvi = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.XSSVISmile, k, vols, forward, tte)
    lvg = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.LVGSmile, k, vols, forward, tte, L=1e-6)
    bspl = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.BSplineCollocationSmile, k, vols, forward, tte)
    quintic = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.QuinticCollocationSmile, k, vols, forward, tte)
    pdfd2(pol, z) = -(10000 / z)^2 * (2 * AQFED.Black.blackScholesFormula(true, z, forward, pol(z)^2 * tte, 1.0, 1.0) - AQFED.Black.blackScholesFormula(true, z * 0.9999, forward, pol(z * 0.9999)^2 * tte, 1.0, 1.0) - AQFED.Black.blackScholesFormula(true, z * 1.0001, forward, pol(z * 1.0001)^2 * tte, 1.0, 1.0))
    smileSplDW = AQFED.VolatilityModels.makeTotalVarianceFunction(smileSplD, forward, tte)
    smileSVIW = AQFED.VolatilityModels.makeTotalVarianceFunction(smileSVI, forward, tte)
    smileQW = AQFED.VolatilityModels.makeTotalVarianceFunction(smileQ, forward, tte)
    xssviW = AQFED.VolatilityModels.makeTotalVarianceFunction(xssvi, forward, tte)
    lvgW = AQFED.VolatilityModels.makeTotalVarianceFunction(lvg, forward, tte)
    bsplW = AQFED.VolatilityModels.makeTotalVarianceFunction(bspl, forward, tte)
    sabr = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.SABRSmile, k, vols, forward, tte)
    @test minimum(map(x -> AQFED.TermStructure.localVarianceDenominator(smileQW, x), yFine)) < -1e-3
    @test minimum(map(x -> pdfd(smileQ, x), kFine)) < -0.5

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

    #EURUSD 11/03/2022
    spot = 0.9759
    forward = 0.975848
    tte = 30 / 365
    dfFor = exp(-0.419914 / 100 * tte)
    dfDom = exp(-0.3518 / 100 * tte) #USD
    deltas = reverse([0.10, 0.25, 0.5, 0.75, 0.90]) #call delta forward no prem.
    vols = reverse([10.51, 10.68, 11.02, 11.73, 12.47] ./ 100)
    delta = 0.01
    vol = 11.11 / 100  #remarkably on LVG curve.
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, dfFor, false, true)
    # k = @.(AQFED.VolatilityModels.strikeForDelta(CallForwardDelta(), deltas, vols, tte, forward, df))
    k = strikeForDelta(vols, conv) #equivalent!

    prices, wv = Collocation.weightedPrices(true, k, vols, w1, forward, 1.0, tte)
    @test AQFED.Collocation.isArbitrageFree(k, prices, forward)[1]
    smileSpl = calibrateSmile(AQFED.VolatilityModels.SplineSmile, k, vols, forward, tte)
    @test rmsd(smileSpl.(k), vols) < 1e-12
    smileSplD = calibrateSmile(AQFED.VolatilityModels.SplineDeltaSmile, k, vols, forward, tte)
    @test rmsd(smileSplD.(k), vols) < 1e-8
    smileSVI = calibrateSmile(AQFED.VolatilityModels.SVISmile, k, vols, forward, tte)
    @test rmsd(smileSVI.(k), vols) < 2e-3
    smileSpl0 = calibrateSmile(AQFED.VolatilityModels.SplineSmile, k, vols, forward, tte, isFlat=true)
    dev = exp(4 * vols[3] * sqrt(tte))
    kFine = exp.(collect(range(log(k[1] / dev), stop=log(k[end] * dev), length=201)))
    yFine = log.(kFine ./ forward)
    smileQ = calibrateSmile(AQFED.VolatilityModels.ExpPolynomialSmile, k, vols, forward, tte, nIter=32)
    smileSVI = calibrateSmile(AQFED.VolatilityModels.SVISmile, k, vols, forward, tte)
    xssvi = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.XSSVISmile, k, vols, forward, tte)
    lvg = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.LVGSmile, k, vols, forward, tte)
    sabr = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.SABRSmile, k, vols, forward, tte)
    @test minimum(map(x -> pdfd(sabr, x), kFine)) >= 0
    @test minimum(map(x -> pdfd(smileSVI, x), kFine)) >= 0
    @test minimum(map(x -> pdfd(smileQ, x), kFine)) >= 0
    fullVols = reverse([11.11, 10.45, 10.51, 10.58, 10.63, 10.68, 10.73, 10.78, 10.85, 11.02, 11.25, 11.38, 11.54, 11.73, 11.93, 12.16, 12.47, 13.01, 14.04] ./ 100)
    fullDeltas = reverse([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.40, 0.35, 0.30, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01])
    kFull = AQFED.VolatilityModels.strikeForDelta(fullVols, conv, deltas=fullDeltas)
    #		  kFull = @.(AQFED.VolatilityModels.strikeForDelta(CallForwardDelta(), fullDeltas, fullVols, tte, forward, df))
    for smile = [smileQ, smileSVI, xssvi, sabr, lvg]
        println(typeof(smile), " ", AQFED.VolatilityModels.priceAutoQuanto(false, k[1], forward, tte, smile, dfDom), " ", AQFED.VolatilityModels.priceAutoQuanto(true, k[end], forward, tte, smile, df))
    end



    swapModelFreeLVG = sqrt(AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.FukasawaVarianceSwapReplication(false), tte, y, lvg.(k) .^ 2, dfDom))
    swapModelFreeQ = sqrt(AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.FukasawaVarianceSwapReplication(false), tte, y, smileQ.(k) .^ 2, dfDom))
    @test isapprox(swapModelFreeQ, swapModelFreeLVG, atol=1e-4)
    for smile = [smileQ, smileSVI, xssvi, sabr, lvg]
        println(typeof(smile), " ", sqrt(AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.ContinuousVarianceSwapReplication(AQFED.Math.GaussKronrod()), forward, tte, AQFED.VolatilityModels.FXVarianceSection(smile, forward), 1.0, dfDom)))
    end

    strike = k[1]
    for smile = [smileQ, smileSVI, xssvi, sabr, lvg]
        price(z) = AQFED.Black.blackScholesFormula(false, z, forward, smile(strike)^2 * tte, 1.0, dfDom)
        dprice = FiniteDifferences.central_fdm(2, 1)(price, strike)
        dpriceMan = (price(strike * 1.01) - price(strike * 0.99)) / (strike * 0.02)
        println(typeof(smile), " ", price(k[1]), " ", dprice, " ", dpriceMan)
    end

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
    spot = 0.9759
    tte = 1.0
    dfFor = 1 / (1 + 2.686 / 100 * tte)
    dfDom = 1 / (1 + 3.83 / 100 * tte) #USD
    forward = spot / df * dfEur

    deltas = reverse([0.10, 0.25, 0.5, 0.75, 0.90]) #call delta forward no prem.
    vols = reverse([9.21, 9.48, 10.19, 11.74, 13.46] ./ 100)
    delta = 0.01
    vol = 11.11 / 100  #remarkably on LVG curve.
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, dfDom, false, true)
    # k = @.(AQFED.VolatilityModels.strikeForDelta(CallForwardDelta(), deltas, vols, tte, forward, df))
    k = strikeForDelta(vols, conv) #equivalent!
    fullDeltas = reverse([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.40, 0.35, 0.30, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01])
    fullVols = reverse([8.87, 9.02, 9.21, 9.31, 9.40, 9.48, 9.57, 9.68, 9.84, 10.19, 10.65, 10.96, 11.32, 11.74, 12.19, 12.73, 13.46, 14.57, 16.0] ./ 100)
    kFull = AQFED.VolatilityModels.strikeForDelta(fullVols, conv, deltas=fullDeltas)

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
end

@testset "autoquanto" begin
    tte = 1.0
    dfDom = 0.9585801
    dfFor = 0.9785056
    spot = 1.2050
    forward = spot * dfFor / dfDom
    vols = [9.65, 9.40, 9.43] ./ 100
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, dfFor, false, false)
    k = AQFED.VolatilityModels.strikeForDelta(vols, conv, deltas=[-0.25, 0.5, 0.25])
    xssvi = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.XSSVISmile, k, vols, forward, tte)
    notional = 85.81 #100.0/spot
    for strike = [1.175, 1.2050, 1.2350]
        priceP = notional * AQFED.VolatilityModels.priceAutoQuanto(false, strike, forward, tte, xssvi, dfDom)
        priceC = notional * AQFED.VolatilityModels.priceAutoQuanto(true, strike, forward, tte, xssvi, dfDom)
        println(strike, " ", priceC, " ", priceP)
    end
end
@testset "deltavolcalibrated" begin
  
          smilesUseMap = Dict(
            AQFED.VolatilityModels.SplineSmile => false,
        AQFED.VolatilityModels.SplineDeltaSmile=>false,
        AQFED.VolatilityModels.ExpPolynomialSmile=>false,
        AQFED.VolatilityModels.SVISmile=>false, 
        AQFED.VolatilityModels.SABRSmile=>true,
            AQFED.VolatilityModels.SABRATMSmile=>true, 
            AQFED.VolatilityModels.XSSVISmile=>true,
            AQFED.VolatilityModels.QuinticCollocationSmile=>false)

            smilesUseMap = Dict(
                AQFED.VolatilityModels.SplineSmile => false,
            AQFED.VolatilityModels.SplineDeltaSmile=>false,
            AQFED.VolatilityModels.ExpPolynomialSmile=>false,
            AQFED.VolatilityModels.SVISmile=>false, 
            AQFED.VolatilityModels.SABRSmile=>false,
                AQFED.VolatilityModels.SABRATMSmile=>false, 
                AQFED.VolatilityModels.XSSVISmile=>false,
                AQFED.VolatilityModels.QuinticCollocationSmile=>false)
    
            volMap = Dict()
    kMap = Dict()
    smileMap = Dict()
    # volMapExact = Dict()
    # kMapExact = Dict()
    # smileExactMap = Dict()
    datasets=[
Dict("name" =>"AUDNZD 2014/07/02 exp 20140709",
    "tte" => 7.0 / 365,
    "spot" => 1.0784,
    "forward" => 1.07845,
    "df" => 0.999712587139,
    "rr10" => 0.35 / 100,
    "rr25" => 0.4 / 100,
    "volAtm" => 5.14 / 100,
    "bf25" => 0.25 / 100,
    "bf10" => 1.175 / 100,
    "isForward" => false,
    "isPremium" => true),
    Dict("name" =>"EURHKD jan 25 2024",
    "spot" =>8.510111,
"tte" =>147/365,
"df" =>0.9848102,
"dfHKD" =>0.9860398,
"forward" => 8.510111-96.07/10000,# similar to dfEUR/dfHKD
"volAtm" =>6.575/100,
"rr25" =>-0.647/100,
"rr10" =>-1.2/100,
"bf25" =>0.202/100,
"bf10" =>0.57/100,
"isForward" => false,
"isPremium" => true),
Dict("name" =>"USDAED 7d",
"spot" => 3.67,
    "tte" => 7 / 365,
    "df" => exp(-3.18 / 100 * 7/365),
    "forward" =>  3.67 * exp((3.255-3.18) / 100 * 7/365) ,
    "volAtm" => 0.34 / 100,
    "rr25" => 0.169 / 100,
    "rr10" => 0.445 / 100,
    "bf25" => 0.098 / 100,
    "bf10" => 0.433 / 100,
    "isForward" => true,
"isPremium" => true),
# Dict("name" =>"USDAED 9m",
#     "tte" => 9 * 30 / 360,
#     "df" => exp(-3.18 / 100 *  9 * 30 / 360),
#     "forward" => 3.67 * exp((3.255-3.18 )/ 100 *  9 * 30 / 360),
#     "volAtm" => 0.32 / 100,
#     "rr25" => 0.152 / 100,
#     "rr10" => 0.412 / 100,
#     "bf25" => 0.084 / 100,
#     "bf10" => 0.392 / 100,
#     "isForward" => true,
#     "isPremium" => true),
    Dict("name" =>"EURTRY 29/11/2022 for=eur 6m",
      "spot" => 19.3483,
      "tte" => 184 / 365, #start of breakdown/bimodality
      "df" => exp(-1.16702 / 100 * 184 / 365 ) ,#foreign df
      "forward" =>  19.3483 * exp((36.77-1.16702) / 100 *  184 / 365) ,
      #vols=[38.44209,	29.79629,	22.12,	20.41129,	17.29409]./100
      "volAtm" => 22.12 / 100,
      "bf25" => 2.187 / 100,
      "rr25" => 9.385 / 100,
      "bf10" => 7.633 / 100,
      "rr10" => 21.148 / 100,
      "isForward" => true,
      "isPremium" => true),
      Dict("name" =>"EURTRY 29/11/2022 for=eur 1y",
      "spot" => 19.3483,
    "tte" => 1.0,
      "df" => exp(-1.784 / 100 ),
      "forward" => 19.3483 * exp((37.73-1.784 ) / 100 ),
      "volAtm" => 31.13 / 100,
      "bf25" => 2.931 / 100,
      "rr25" => 11.568 / 100,
      "bf10" => 9.307 / 100,
      "rr10" => 27.12 / 100,
      "isForward" => true,
      "isPremium" => true),
      Dict("name" =>"EURTRY 29/11/2022 for=eur 2y",
      "spot" => 19.3483,
      "tte" => 2.0,
      "df" => exp(-1.95 / 100 * 2),
      "forward" => 19.3483 * exp((37.65-1.95) / 100 * 2),
      "rr10" => 27.34 / 100,
      "rr25" => 10.981 / 100,
      "volAtm" => 32.084 / 100,
      "bf25" => 3.46 / 100,
      "bf10" => 10.866 / 100,
      "isForward" => true,
      "isPremium" => true),
    #   Dict("name" =>"EURTRY 29/11/2022 for=eur 2y counter",
    #   "spot" => 19.3483,
    #   "tte" => 2.0,
    #   "df" => exp(-1.95 / 100 * 2),
    #   "forward" => 19.3483 * exp((37.65-1.95) / 100 * 2),
    #   "rr10" => -27.34 / 100,
    #   "rr25" => -12.981 / 100,
    #   "volAtm" => 32.084 / 100,
    #   "bf25" => 3.46 / 100,
    #   "bf10" => 10.866 / 100,
    #   "isForward" => true,
    #   "isPremium" => true),
      Dict("name" =>"USDJPY Clark 1y",
      "spot" =>90.72,
      "tte" => 1.0,
      "volAtm" => 15.95 / 100,
      "bf25" => 0.175 / 100,
      "bf10" => 5.726 / 100,
      "rr25" => -9.55 / 100,
      "rr10" => -18.855 / 100,
      "rf" => 2.94 / 100,
      "rd" => 1.71 / 100,
      "df" => exp(-2.94 / 100),
      "forward" => 90.72 * exp( (1.71-2.94) / 100),
      "isForward" => true,
      "isPremium" => true),
      Dict("name" =>"USDJPY 1m",
      "spot" =>91.88,
      "tte" => 29 / 365,
      "volAtm" => 11.435 / 100,
      "rr25" => -0.585 / 100,
      "bf25" => 0.287 / 100,
      "rr10" => -1.1 / 100,
      "bf10" => 0.907 / 100,
      "forward" => 91.87,
      "df" => exp(-0.229 / 100 * 29 / 365),
      "dfDom" => exp(-0.052 / 100 * 29 / 365),
     "isForward" => true,
      "isPremium" => true),
      Dict("name" =>"EURUSD Clark 1y",
      "spot" =>1.3465,
      "tte" => 1.0,
      "volAtm" => 18.25 / 100,
      "rr25" => -0.6 / 100,
      "bf25" => 0.95 / 100,
      "rr10" => -1.359/ 100,
      "bf10" => 3.806 / 100,
      "forward" => 1.3465 * 0.966001/0.971049,
      "df" => 0.966001,
      "dfDom" => 0.971049,
     "isForward" => false,
      "isPremium" => false)
      ]
results = Dict()

set = datasets[end]
volAtm= set["volAtm"]
     bf25 = set["bf25"]
     rr25 = set["rr25"]
     bf10 = set["bf10"]
      rr10 = set["rr10"]
      tte = set["tte"]
      name = set["name"]
      isForward =  set["isForward"]
      isPremium = set["isPremium"]
      forward = set["forward"]
      df=set["df"]
      conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, isPremium, isForward) #prem currency is aud	
      vols = AQFED.VolatilityModels.convertQuotesToDeltaVols(conv, volAtm, bf25, rr25, bf10, rr10)
      k = logmoneynessForDelta(vols, conv)
      strikes = exp.(k) * forward
 @test isapprox(1.3620,strikes[3],atol=1e-4) 
 @test isapprox(1.2050,AQFED.VolatilityModels.strikeForDelta(PutDelta(), -0.25, 19.20/100,tte,forward,df),atol=1e-4) 
 @test isapprox(1.5449,AQFED.VolatilityModels.strikeForDelta(CallDelta(), 0.25, 19.20/100,tte,forward,df),atol=1e-4) 
 for set = datasets
    volAtm= set["volAtm"]
     bf25 = set["bf25"]
     rr25 = set["rr25"]
     bf10 = set["bf10"]
      rr10 = set["rr10"]
      tte = set["tte"]
      name = set["name"]
      isForward =  set["isForward"]
      isPremium = set["isPremium"]
      forward = set["forward"]
      df=set["df"]
      println("**** processing **** ",set["name"], " ",volAtm, " ",bf25, " ", rr25, " ",bf10," ",rr10, " ",tte," ",forward)
    
      volsRaw = AQFED.VolatilityModels.convertQuotesToDeltaVols(AQFED.VolatilityModels.SmileConvention(), volAtm, bf25, rr25, bf10, rr10)
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, isPremium, isForward) #prem currency is aud	
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, isPremium, isForward) #prem currency is aud	
    kRaw = logmoneynessForDelta(volsRaw, conv)
    vols = AQFED.VolatilityModels.convertQuotesToDeltaVols(conv, volAtm, bf25, rr25, bf10, rr10)
    
    for (smileType,useDelta) = smilesUseMap
        vols = AQFED.VolatilityModels.convertQuotesToDeltaVols(conv, volAtm, bf25, rr25, bf10, rr10, smileType=smileType, isVegaWeighted=true,isGlobal=false,useDelta=useDelta)
        k = logmoneynessForDelta(vols, conv)
        volMap[smileType] = vols
        kMap[smileType] = k
        smile = AQFED.VolatilityModels.calibrateSmileToVanillas(smileType, conv, vols, useDelta=useDelta)
        smileMap[smileType] = smile
        # kExact = strikeForDelta(smile.(k), conv)
        # volsExact = smile.(kExact)
        # smileExact = AQFED.VolatilityModels.calibrateSmile(smileType, kExact, volsExact, forward, tte)
        # volsExact = smileExact.(kExact)
        # volMapExact[smileType] = volsExact
        # kMapExact[smileType] = kExact
        # smileExactMap[smileType] = smileExact
    end
	smileSABRClark = smileMap[AQFED.VolatilityModels.SABRSmile]
	smileSABRATMClark = smileMap[AQFED.VolatilityModels.SABRATMSmile]
	smileXSSVIClark = smileMap[AQFED.VolatilityModels.XSSVISmile]
    smileColloClark = smileMap[AQFED.VolatilityModels.QuinticCollocationSmile]

	for (smileType,useDelta) = smilesUseMap
        vols = AQFED.VolatilityModels.convertQuotesToDeltaVols(conv, volAtm, bf25, rr25, bf10, rr10, smileType=smileType, isVegaWeighted=true,isGlobal=true,useDelta=useDelta)
        k = logmoneynessForDelta(vols, conv)
        volMap[smileType] = vols
        kMap[smileType] = k
        smile = AQFED.VolatilityModels.calibrateSmileToVanillas(smileType, conv, vols, useDelta=useDelta)
        smileMap[smileType] = smile
    end
	smileATMSABRVanillaStrike = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.SABRATMSmile, fill(AQFED.VolatilityModels.LogmoneynessAxisTransformation(),5), kMap[AQFED.VolatilityModels.SplineSmile] , volMap[AQFED.VolatilityModels.SplineSmile] , forward, tte)
    smileATMSABRVanillaDelta = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.SABRATMSmile, fill(AQFED.VolatilityModels.LogmoneynessAxisTransformation(),5), kMap[AQFED.VolatilityModels.SplineDeltaSmile] , volMap[AQFED.VolatilityModels.SplineDeltaSmile] , forward, tte)
    smileXSSVIVanillaStrike = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.XSSVISmile,fill(AQFED.VolatilityModels.LogmoneynessAxisTransformation(),5),  kMap[AQFED.VolatilityModels.SplineSmile] , volMap[AQFED.VolatilityModels.SplineSmile] , forward, tte)
    smileXSSVIVanillaDelta = AQFED.VolatilityModels.calibrateSmile(AQFED.VolatilityModels.XSSVISmile,fill(AQFED.VolatilityModels.LogmoneynessAxisTransformation(),5),  kMap[AQFED.VolatilityModels.SplineDeltaSmile] , volMap[AQFED.VolatilityModels.SplineDeltaSmile] , forward, tte)

    l2map = Dict()
    for (namel,smile) = Dict(
        "VanillaDelta" => smileATMSABRVanillaDelta,
        "VanillaStrike" => smileATMSABRVanillaStrike,
        "Clark" => smileSABRATMClark,
        "Extended" => smileMap[AQFED.VolatilityModels.SABRATMSmile])
    l2n = norm(AQFED.VolatilityModels.evaluateSmileOnQuotes(smile, conv,  volAtm,bf25, rr25, bf10, rr10))
    l2map[string("ATMSABR ",namel)] = l2n
    end
    l2map[string("SABR Extended")]= norm(AQFED.VolatilityModels.evaluateSmileOnQuotes(smileMap[AQFED.VolatilityModels.SABRSmile], conv,  volAtm,bf25, rr25, bf10, rr10))

    for (namel,smile) = Dict(
        "VanillaDelta" => smileXSSVIVanillaDelta,
        "VanillaStrike" => smileXSSVIVanillaStrike,
        "Clark" => smileXSSVIClark,
        "Extended" => smileMap[AQFED.VolatilityModels.XSSVISmile])
    l2n = norm(AQFED.VolatilityModels.evaluateSmileOnQuotes(smile, conv,  volAtm,bf25, rr25, bf10, rr10))
    l2map[string("XSSVI ",namel)] = l2n
    end
    results[name]=  l2map
end
for (k,v) = results
    for (k2,v2)=v
    println(k," ",k2," ",v2)
    end
    end
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

    #prices of call at market deltas in vols. We want to compare those, as well as smile vs logm
    # 

#EURHKD jan 25 2024
spot=8.510111
tte=147/365
dfEUR=0.9848102
dfHKD=0.9860398
forward = spot-96.07/10000 # similar to dfEUR/dfHKD
volAtm=6.575/100
rr25=-0.647/100
rr10=-1.2/100
bf25=0.202/100
bf10=0.57/100
volsRaw = AQFED.VolatilityModels.convertQuotesToDeltaVols(AQFED.VolatilityModels.SmileConvention(), volAtm, bf25, rr25, bf10, rr10)
conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, dfEUR, true, false) 
kRaw = logmoneynessForDelta(volsRaw, conv)
vols = AQFED.VolatilityModels.convertQuotesToDeltaVols(conv, volAtm, bf25, rr25, bf10, rr10)


# calibrating SABR to delta directly does improve fit vs strike + delta after.
# SSVI 2 Delta errror >> SSVI 5 Delta error.
# delta ain SSVI/SABR slows down calibration somewhat significantly.

    spot = 3.67
    tte = 7 / 365
    df = exp(-3.255 / 100 * tte)
    forward = spot * exp(-3.18 / 100 * tte) / df
    volAtm = 0.34 / 100
    rr25 = 0.169 / 100
    rr10 = 0.445 / 100
    bf25 = 0.098 / 100
    bf10 = 0.433 / 100
    #USDAED 9m
    tte = 9 * 30 / 360
    df = exp(-3.255 / 100 * tte)
    forward = spot * exp(-3.18 / 100 * tte) / df
    volAtm = 0.32 / 100
    rr25 = 0.152 / 100
    rr10 = 0.412 / 100
    bf25 = 0.084 / 100
    bf10 = 0.392 / 100

    volsRaw = AQFED.VolatilityModels.convertQuotesToDeltaVols(AQFED.VolatilityModels.SmileConvention(), volAtm, bf25, rr25, bf10, rr10)
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, true, true) 
    kRaw = logmoneynessForDelta(volsRaw, conv)


    #=
    plot(log.(kRaw./forward), volsRaw, seriestype= :scatter, label="Smile convention", m=:cross)
    plot!(log.(kFine./forward), smileExactMap[AQFED.VolatilityModels.SABRSmile].(kFine),label="SABR Clark")
    plot!(log.(kFine./forward), smileExactMap[AQFED.VolatilityModels.XSSVISmile].(kFine),label="XSSVI Clark")
    plot!(log.(kFine./forward), smileExactMap[AQFED.VolatilityModels.SVISmile].(kFine),label="SVI Clark")
    plot!(log.(kMapExact[AQFED.VolatilityModels.SplineSmile] ./forward), volMapExact[AQFED.VolatilityModels.SplineSmile], seriestype= :scatter, label="Broker Spline", m=:cross)
    plot!(log.(kMapExact[AQFED.VolatilityModels.ExpPolynomialSmile] ./forward), volMapExact[AQFED.VolatilityModels.ExpPolynomialSmile], seriestype= :scatter, label="Broker Polynomial", m = :cross)
    plot!(log.(kMapExact[AQFED.VolatilityModels.LVGSmile] ./forward), volMapExact[AQFED.VolatilityModels.LVGSmile], seriestype= :scatter, label="Broker LVG", m = :cross)


    =#
    #EURTRY 29/11/2022 for=eur
    spot = 19.3483
    tte = 184 / 365 #start of breakdown/bimodality
    df = exp(-1.16702 / 100 * tte) #foreign df
    forward = spot * exp(36.77 / 100 * tte) * df
    #vols=[38.44209,	29.79629,	22.12,	20.41129,	17.29409]./100
    volAtm = 22.12 / 100
    bf25 = 2.187 / 100
    rr25 = 9.385 / 100
    bf10 = 7.633 / 100
    rr10 = 21.148 / 100
    volsRaw = AQFED.VolatilityModels.convertQuotesToDeltaVols(AQFED.VolatilityModels.SmileConvention(), volAtm, bf25, rr25, bf10, rr10)
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, true, true) #prem currency is aud
    kRaw = logmoneynessForDelta(volsRaw, conv)
  tte = 1.0
    #vols = [50.37981	,40.71189,	31.13,	29.14389	,23.25981]./100
    #vols=[0.24077330183795542, 0.28640409729845756, 0.31129999999999997, 0.40208409729845757, 0.5119733018379554]
    df = exp(-1.784 / 100 * tte)
    forward = spot * exp(37.73 / 100 * tte) * df
    volAtm = 31.13 / 100
    bf25 = 2.931 / 100
    rr25 = 11.568 / 100
    bf10 = 9.307 / 100
    rr10 = 27.12 / 100
    volsRaw = AQFED.VolatilityModels.convertQuotesToDeltaVols(AQFED.VolatilityModels.SmileConvention(), volAtm, bf25, rr25, bf10, rr10)
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, true, true) #prem currency is aud
    kRaw = logmoneynessForDelta(volsRaw, conv)
    tte = 2.0
    df = exp(-1.95 / 100 * tte)
    forward = spot * exp(37.65 / 100 * tte) * df
    #vols = [52.12681, 41.48931, 32.08400, 30.50831, 24.78681]./100
    rr10 = 27.34 / 100
    rr25 = 10.981 / 100
    volAtm = 32.084 / 100
    bf25 = 3.46 / 100
    bf10 = 10.866 / 100
    volsRaw = AQFED.VolatilityModels.convertQuotesToDeltaVols(AQFED.VolatilityModels.SmileConvention(), volAtm, bf25, rr25, bf10, rr10)
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, true, true) #prem currency is aud
    kRaw = logmoneynessForDelta(volsRaw, conv)

    yRange = range(-6*volAtm*sqrt(tte),8*volAtm*sqrt(tte),length=101)
    deltaValues = map(y -> deltaForLogmoneyness(CallPremiumForwardDelta(),y,1.25,tte,forward,df),yRange)

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

    #A counter example of convergence of newton's method based on error in vol.
    
    tte = 2.0
    df = exp(-1.95 / 100 * tte)
    forward = spot * exp(37.65 / 100 * tte) * df
    rr10 = -27.34 / 100
    rr25 = -12.981 / 100
    volAtm = 32.084 / 100
    bf25 = 3.46 / 100
    bf10 = 10.866 / 100
    volsRaw = AQFED.VolatilityModels.convertQuotesToDeltaVols(AQFED.VolatilityModels.SmileConvention(), volAtm, bf25, rr25, bf10, rr10)
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, true, true) #prem currency is aud
    kRaw = logmoneynessForDelta(volsRaw, conv)

 
    smile = ExpPolynomialSmile([0.11414222327265336, -11.854545452862252, 49.24883225877462, -84.12103007927499, 48.491914737536796], 0.32084, 39.5120054163372, 2.0, 256)
    smile(forward/5) #works with newton's on delta.

    #are output vanilla quotes arb free?
for (smileType,useDelta) = smilesUseMap
    vols = volMap[smileType]
    ks = kMap[smileType]
    strikes = exp.(ks) .* forward
    volsExact = smile.(strikes)
    ucalls =map((strike,vol) -> AQFED.Black.blackScholesFormula(true, strike,forward,vol^2*tte,1.0,1.0),strikes,volsExact)
    println(smileType, " ",ucalls, " ", AQFED.Collocation.isArbitrageFree(strikes, ucalls,forward))
end
   # plot(deltaRange, map(x->AQFED.VolatilityModels.impliedVolatilityByDelta(ExpPolynomialSmile([0.11414222327265336, -11.854545452862252, 49.24883225877462, -84.12103007927499, 48.491914737536796], 0.32084, 39.5120054163372, 2.0, 256),x),deltaRange))

    #For 1Y, SplineDelta leads to very similar quotes as poly delta. very different from spline strike. LVG is in between.
    # vols = [16.35306, 13.26229,11.28, 10.75929, 11.20506]./100
    # tte = 7/365
    # df=exp(-0.20/100*tte)
    # forward = spot*exp(11.72/100*tte)*df
    # tte=91/365 #SVI fit exact!
    # df=exp(-0.84/100*tte)
    # forward = spot*exp(33.70/100*tte)*df
    # vols = [30.86238,	22.00304,	16.43,	15.10004,	15.32738	]./100


    #USDJPY clark
    tte = 30 / 365
    spot = 90.72
    volAtm = 21.5 / 100
    bf25 = 0.35 / 100
    bf10 = 3.704 / 100
    rr25 = -8.35 / 100
    rr10 = -15.855 / 100
    tte = 1.0
    volAtm = 15.95 / 100
    bf25 = 0.175 / 100
    bf10 = 5.726 / 100
    rr25 = -9.55 / 100
    rr10 = -18.855 / 100
    rf = 2.94 / 100
    rd = 1.71 / 100
    df = exp(-rf * tte)
    forward = spot * exp(rd * tte) * df
    volsRaw = AQFED.VolatilityModels.convertQuotesToDeltaVols(AQFED.VolatilityModels.SmileConvention(), volAtm, bf25, rr25, bf10, rr10)
    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, true, true) #prem currency is aud
    kRaw = logmoneynessForDelta(volsRaw, conv)
    tte = 29 / 365
    volAtm = 11.435 / 100
    rr25 = -0.585 / 100
    bf25 = 0.287 / 100
    rr10 = -1.1 / 100
    bf10 = 0.907 / 100
    spot = 91.88
    forward = 91.87
    df = exp(-0.229 / 100 * tte)
    dfDom = exp(-0.052 / 100 * tte)


    conv = AQFED.VolatilityModels.BrokerConvention(tte, forward, df, true, true)
    volsRaw = AQFED.VolatilityModels.convertQuotesToDeltaVols(conv, volAtm, bf25, rr25, bf10, rr10)
    kRaw = strikeForDelta(volsRaw, conv)


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
    prices, wv = Collocation.weightedPrices(true, strikes, vols, w1, forward, 1.0, tte, vegaFloor=1e-8)
    fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (wv), λ=60, eps=1e-13, solver="GI")
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

    lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, wv, useVol=true, L=strikes[1] / 2, U=strikes[end] * 2)
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
    lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, wv, useVol=false, model=PDDE.Quadratic(), location="Mid-XX", size=0, L=strikes[1] / 2, U=strikes[end] * 2)
    ivkLVG = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0)
    rmseLVG = StatsBase.rmsd(vols, ivkLVG)

    #=

    	p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    	plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,PDDE.priceEuropean.(lvgq,true,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Quadratic LVG")
    	plot!(p3, ylim=(21,25),xlim=(-0.1,3.6), size=(480,320))
    	savefig("~//mypapers/eqd_book/jaeckel_case_i_lvgq_vol.pdf")
    	p2 = plot(log.(kFine./forward),(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label=:none, xlab="Forward log-moneyness",ylab="Probability density")
    	plot!(p2, size=(480,320))
    	savefig("~//mypapers/eqd_book/jaeckel_case_i_lvgq_dens.pdf")
    	=#


    #= 
    pdf(pp,z) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.Black.blackScholesFormula(true, y, forward, pp(log(y/forward))*tte,1.0,1.0),x),z)

    p1=plot(log.(kFine./forward),AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label="Fengler",color=2, ylim=(-0.05,1.0))
    p2 = plot(log.(kFine./forward),AQFED.Math.evaluateSecondDerivative.(cs,kFine),label="Schaback",color=3,ylim=(-0.05,1.0))
    p3 = plot(log.(kFine./forward),pdf.(pp,kFine),label="Cubic spline on implied variances",color=1,ylim=(-0.05,1.0))
    p4 = plot(log.(kFine./forward),pdf.(rbf,kFine),label="RBF",color=4,ylim=(-0.05,1.0))
    plot(p3,p4,p1,p2, layout=(1,4),legend=false,titles=["Cubic spline" "RBF" "Fengler" "Schaback"],size=(800,250))
    savefig("~//mypapers/eqd_book/jaeckel_case_i_fengler_rbf_dens.pdf")

    plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, max.(fengler.(kFine),1e-16), forward, kFine, tte, 1.0)*100),label="Fengler λ=1e-13")
    plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, max.(cs.(kFine),1e-16), forward, kFine, tte, 1.0)*100),label="Schaback")
    plot!(log.(kFine./forward), sqrt.(rbf.(log.(kFine./forward))).*100,label="RBF")
    #plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, PDDE.priceEuropean.(lvg,true,kFine), forward, kFine, tte, 1.0)*100),label="LVG Linear Black")
    plot!(ylim=(21,25),xlim=(-0.1,3.1),size=(640,320),margin=2Plots.mm)
    savefig("~//mypapers/eqd_book/jaeckel_case_i_fengler_rbf.pdf")

    =#

    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, wv, tte, forward, 1.0, deg=7, degGuess=1)
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
        wv,
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
        wv,
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
        wv,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=0,
        minSlope=1e-8,
        rawFit=true, N=3,
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
    savefig("~//mypapers/eqd_book/jaeckel_case_ii_bspl_vol.pdf")
    p4=plot(log.(kFine), Collocation.density.(bsple,kFine),label="Exponential B-spline",xlab="Log-moneyness",ylab="Probability density",color=2)
    plot!(p4,log.(kFine), Collocation.density.(bspl3,kFine),label="Cubic B-spline",color=3)
    plot!(p4, size=(400,200))
    savefig("~//mypapers/eqd_book/jaeckel_case_ii_bspl_dens.pdf")
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

    prices, wv = Collocation.weightedPrices(true, strikes, vols, w1, forward, 1.0, tte, vegaFloor=1e-8)
    sumw2 = sum(w1 .^ 2)
    w = w1 ./ sqrt(sumw2)
    kernel2 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=2)
    ivkMLN2 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel2, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN2 = StatsBase.L2dist(w .* ivkMLN2, w .* vols)
    kernel3 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=3)
    ivkMLN3 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel3, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN3 = StatsBase.L2dist(w .* ivkMLN3, w .* vols)

    kernel4 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=4)
    ivkMLN4 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel4, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN4 = StatsBase.L2dist(w .* ivkMLN4, w .* vols)
    kernel4v = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=true, size=4)
    ivkMLN4v = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel4v, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN4v = StatsBase.L2dist(w .* ivkMLN4v, w .* vols)

    kernel6 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=6)
    ivkMLN6 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel6, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN6 = StatsBase.L2dist(w .* ivkMLN6, w .* vols)


    #=
    plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel4,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 4 price")
    plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel4v,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 4 vol")


    #plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, PDDE.priceEuropean.(lvg,true,kFine), forward, kFine, tte, 1.0)*100),label="LVG Linear Black")
    plot!(ylim=(21,25),xlim=(-0.1,3.1),size=(640,320),margin=2Plots.mm)
    savefig("~//mypapers/eqd_book/jaeckel_case_i_fengler_rbf.pdf")

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
    prices, wv = Collocation.weightedPrices(true, strikes, vols, w1, forward, 1.0, tte, vegaFloor=1e-5)

    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, sqrt.(wv), tte, forward, 1.0, deg=7, degGuess=1)
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
        wv,
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
        wv,
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
        wv,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=0,
        minSlope=1e-8,
        rawFit=true, N=3,
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
    savefig("~//mypapers/eqd_book/jaeckel_case_ii_bspl_vol.pdf")
    p4=plot(log.(kFine), Collocation.density.(bsple,kFine),label="Exp B-spline",xlab="Log-moneyness",ylab="Probability density",color=2)
    plot!(p4,log.(kFine), Collocation.density.(bspl3,kFine),label="Cubic B-spline",color=3)
    plot!(p4, size=(480,320))
    savefig("~//mypapers/eqd_book/jaeckel_case_ii_bspl_dens.pdf")
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
    lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, wv, useVol=true, L=strikes[1] / 2, U=strikes[end] * 2)
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

    lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, wv, useVol=false, model=PDDE.Quadratic(), location="Mid-XX", size=0, L=k[1], U=k[end])
    ivkLVG = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0)
    rmseLVG = StatsBase.rmsd(vols, ivkLVG)

    #=

    	p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    	plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,PDDE.priceEuropean.(lvgq,true,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Quadratic LVG")
    	plot!(p3, size=(480,320))
    	savefig("~//mypapers/eqd_book/jaeckel_case_ii_lvgq_vol.pdf")
    	p2 = plot(log.(kFine./forward),(PDDE.derivativePrice.(lvg,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvg,true,kFine)).*10000, label="Linear-Black", xlab="Forward log-moneyness",ylab="Probability density")
    	plot!(log.(kFine./forward),(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label="Quadratic", xlab="Forward log-moneyness",ylab="Probability density")
    	plot!(p2, size=(480,320),scale=:log10)
    	savefig("~//mypapers/eqd_book/jaeckel_case_ii_lvgq_dens_log.pdf")
    	=#
    prices, wv = Collocation.weightedPrices(true, strikes, vols, w1, forward, 1.0, tte, vegaFloor=1e-7)
    sumw2 = sum(w1 .^ 2)
    w = w1 ./ sqrt(sumw2)
    kernel2 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=2)
    ivkMLN2 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel2, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN2 = StatsBase.L2dist(w .* ivkMLN2, w .* vols)
    kernel3 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=3)
    ivkMLN3 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel3, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN3 = StatsBase.L2dist(w .* ivkMLN3, w .* vols)

    kernel4 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=4)
    ivkMLN4 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel4, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN4 = StatsBase.L2dist(w .* ivkMLN4, w .* vols)
    kernel4v = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=true, size=4)
    ivkMLN4v = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel4v, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN4v = StatsBase.L2dist(w .* ivkMLN4v, w .* vols)

    kernel6 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=6)
    ivkMLN6 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel6, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN6 = StatsBase.L2dist(w .* ivkMLN6, w .* vols)
    #=
    plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel4,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 4")
    plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(kFine .>= forward, AQFED.VolatilityModels.priceEuropean.(kernel6,kFine .>= forward,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 6")
    plot!(size=(480,380))
    savefig("~//mypapers/eqd_book/jaeckel_case_ii_mixture_vol.pdf")


    #plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, PDDE.priceEuropean.(lvg,true,kFine), forward, kFine, tte, 1.0)*100),label="LVG Linear Black")
    plot!(ylim=(21,25),xlim=(-0.1,3.1),size=(640,320),margin=2Plots.mm)
    plot(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel4,kFine),label="Mixture of 4",color=2,xlabel="Forward log-moneyness",ylabel="Probability density")
    plot!(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel6,kFine),label="Mixture of 6",color=3,xlabel="Forward log-moneyness",ylabel="Probability density")
    plot!(ylims=(0.0,2.0))
    plot!(size=(480,380))
    savefig("~//mypapers/eqd_book/jaeckel_case_ii_mixture_dens.pdf")

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
        0.6754938271604939, 0.6787860082304528, 0.67940329218107, 0.6872222222222223, 0.6919547325102882, 0.6993621399176955, 0.7028600823045268, 0.7040946502057613, 0.6995679012345679, 0.6931893004115227, 0.6802263374485598, 0.6664403292181071,
        0.6610905349794239, 0.6485390946502059, 0.6421604938271606, 0.639485596707819, 0.6417489711934158, 0.6456584362139919, 0.6425720164609054, 0.6485390946502059, 0.6555349794238684, 0.6639711934156379, 0.6783744855967079, 0.6870164609053498,
        0.6903086419753087,
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
    prices, wv = Collocation.weightedPrices(true, strikes, vols, w, forward, 1.0, tte, vegaFloor=1e-5)

    bsple, m = Collocation.makeExpBSplineCollocation(
        strikes,
        prices,
        wv,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=10,
        minSlope=1e-8,
        rawFit=false,
    )
    ivkbe = @. Black.impliedVolatility(true, Collocation.priceEuropean(bsple, true, strikes, forward, 1.0), forward, strikes, tte, 1.0)
    rmsebe = StatsBase.L2dist(w .* vols, w .* ivkbe)
    bspl3, m = Collocation.makeBSplineCollocation(
        strikes,
        prices,
        wv,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=10,
        minSlope=1e-8,
        rawFit=false, N=3,
    )
    ivkb3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(bspl3, true, strikes, forward, 1.0), forward, strikes, tte, 1.0)
    rmseb3 = StatsBase.L2dist(w .* vols, w .* ivkb3)

    #== 
    kFine = forward.*exp.(range(logmoneynessA[1],stop=logmoneynessA[end], length=1001));
    p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bsple,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Exp B-Spline")
    plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bspl3,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Cubic B-spline")
    plot!(p3, size=(480,320))
    savefig("~//mypapers/eqd_book/aapl_20131028_vol_bspl.pdf")
    p4=plot(log.(kFine./forward), Collocation.density.(bsple,kFine),label="Exp B-spline",xlab="Log-moneyness",ylab="Probability density",color=2)
    plot!(p4,log.(kFine./forward), Collocation.density.(bspl3,kFine),label="Cubic B-spline",color=3)
    #plot!(p4,yscale=:log10)
    plot!(p4, size=(480,320))
    savefig("~//mypapers/eqd_book/aapl_20131028_dens_bspl.pdf")
    ==#
    strikesf, pricesf = AQFED.Collocation.filterConvexPrices(strikes, prices, wv, forward, tol=1e-6)
    allStrikes = vcat(0.0, strikesf, forward * exp(logmoneynessA[end] * 3))
    allPrices = vcat(forward, pricesf, 0.0)
    allWeights = vcat(sum(wv), wv, sum(wv))
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
    )
    rmseScha = StatsBase.L2dist(w .* ivkScha, w .* vols)
    println("scha ", rmseScha)

    csf, rmseSchaFit = fitConvexSchabackRationalSpline(
        allStrikes, allPrices,
        allWeights,
        leftB,
        rightB;
        penalty=1e-5,
    )
    ivkSchaFit = @. Black.impliedVolatility(
        true,
        csf(strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseSchaFit = StatsBase.L2dist(w .* ivkSchaFit, w .* vols)
    println("schafit ", rmseSchaFit)
    step = floor(Int, length(strikesf) / 7)
    subset = collect(1:step:length(strikesf))
    if subset[end] != length(strikesf)
        append!(subset, length(strikesf))
    end
    allStrikest = vcat(0.0, strikesf[subset], forward * exp(logmoneynessA[end] * 3))
    allPricest = vcat(forward, pricesf[subset], 0.0)

    cs8 = Math.makeConvexSchabackRationalSpline(allStrikest, allPricest, leftB, rightB, iterations=128)
    ivkScha8 = @. Black.impliedVolatility(
        true,
        cs8(strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseScha8 = StatsBase.L2dist(w .* ivkScha8, w .* vols)
    println("scha8 ", rmseScha8)
    #=
    	 plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    	 plot!(log.(kFine./forward),Black.impliedVolatility.(true, max.(csf.(kFine),1e-16), forward, kFine, tte, 1.0) .* 100, label=string("Schaback λ=",1e-5))
    	 plot!(log.(kFine./forward),Black.impliedVolatility.(true, max.(cs8.(kFine),1e-16), forward, kFine, tte, 1.0) .* 100, label=string("Schaback on 8 knots"))
    	plot!(size=(480,380))
    	savefig("~//mypapers/eqd_book/aapl_20131028_schaback.pdf")

    	plot(log.(kFine./forward), AQFED.Math.evaluateSecondDerivative.(cs8,kFine),label="Schaback on 8 knots", xlab="Forward log-moneyness",ylab="Probability density")
    	plot!(log.(kFine./forward), AQFED.Math.evaluateSecondDerivative.(csf,kFine),label=string("Schaback λ=",1e-5))
    	plot!(legend=:bottom)
    	plot!(size=(480,380))
    	 savefig("~//mypapers/eqd_book/aapl_20131028_schaback_dens.pdf")
    	=#

    λs = [1e-8, 1e-6, 1e-5]
    for λ in λs
        fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (wv), λ=λ, eps=1e-8, solver="GI")
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
    fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (wv),λ=λ,eps=1e-13,solver="GI")
    plot!(p3,log.(kFine./forward), AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label=string("Fengler λ=",λ))
    end
    plot(p3,margin=3Plots.mm,size=(480,380))
    savefig(p3,"~//mypapers/eqd_book/aapl_20131028_fengler_dens.pdf")
    p4 = plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    for λ in λs
    fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (wv),λ=λ,eps=1e-13,solver="GI")
    plot!(p4,log.(kFine./forward), @.(Black.impliedVolatility(true, max.(fengler.(kFine),1e-16), forward, kFine, tte, 1.0)*100),label=string("Fengler λ=",λ))
    end
    plot(p4,margin=3Plots.mm,size=(480,380))
    savefig(p4,"~//mypapers/eqd_book/aapl_20131028_fengler.pdf")

    =#
    rbf = AQFED.VolatilityModels.calibrateMultiquadric(tte, forward, log.(strikes), vols, weightsV, knots=range(log(strikes[1]), stop=log(strikes[end]), length=8))
    rmseRBF = StatsBase.L2dist(w .* sqrt.(rbf.(log.(strikes))), w .* vols)
    pricesRBF = @. blackScholesFormula(true, strikes, forward, rbf(log(strikes / forward)) * tte, 1.0, 1.0)
    λs = [1e-8, 1e-6]
    for λ in λs
        fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, pricesRBF, (wv), λ=λ, eps=1e-8, solver="GI")
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
            wv,
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
    	fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, pricesRBF, (wv),λ=λ,eps=1e-13,solver="GI")
    	plot!(p3,log.(kFine./forward), AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label=string("Fengler λ=",λ))
     end
    plot(p3,margin=2Plots.mm,size=(480,380))
    savefig(p3,"~//mypapers/eqd_book/aapl_20131028_fengler_rbf_dens.pdf")
     p4 = plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
     for λ in λs
    	fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, pricesRBF, (wv),λ=λ,eps=1e-13,solver="GI")
    plot!(p4,log.(kFine./forward), @.(Black.impliedVolatility(true, max.(fengler.(kFine),1e-16), forward, kFine, tte, 1.0)*100),label=string("Fengler λ=",λ))
     end
    plot!(p4,log.(kFine./forward),sqrt.(rbf.(log.(kFine)./forward)) .*100 ,label="RBF")
    plot(p4,margin=2Plots.mm,size=(480,380))
    savefig(p4,"~//mypapers/eqd_book/aapl_20131028_rbf_fengler.pdf")

     =#

    lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, wv, useVol=true, L=strikes[1] / 2, U=strikes[end] * 2)
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
    lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, wv, useVol=true, L=strikes[1] / 2, U=strikes[end] * 2, penalty=1e-6)
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
    lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, wv, useVol=false, model=PDDE.Quadratic(), location="Equidistributed", size=10, L=strikes[1], U=strikes[end])
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
    	savefig("~//mypapers/eqd_book/aapl_20131028_lvgq_vol.pdf")
    	p2 = plot(log.(kFine./forward),(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label=:none, xlab="Forward log-moneyness",ylab="Probability density")
    	plot!(p2, size=(480,320))
    	savefig("~//mypapers/eqd_book/aapl_20131028_lvgq_dens.pdf")
    	=#
    kernel2 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=2)
    ivkMLN2 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel2, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN2 = StatsBase.L2dist(w .* ivkMLN2, w .* vols)
    kernel3 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=3)
    ivkMLN3 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel3, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN3 = StatsBase.L2dist(w .* ivkMLN3, w .* vols)

    kernel4 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=4)
    ivkMLN4 = @. AQFED.Black.impliedVolatility(
        true,
        AQFED.VolatilityModels.priceEuropean.(kernel4, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN4 = StatsBase.L2dist(w .* ivkMLN4, w .* vols)
    kernel6 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=6)
    ivkMLN6 = @. AQFED.Black.impliedVolatility(
        true,
        AQFED.VolatilityModels.priceEuropean.(kernel6, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN6 = StatsBase.L2dist(w .* ivkMLN6, w .* vols)

end

@testset "SPX1m" begin
    strikes =
        Float64.([
            1900,
            1950,
            2000,
            2050,
            2100,
            2150,
            2200,
            2250,
            2300,
            2325,
            2350,
            2375,
            2400,
            2425,
            2450,
            2470,
            2475,
            2480,
            2490,
            2510,
            2520,
            2530,
            2540,
            2550,
            2560,
            2570,
            2575,
            2580,
            2590,
            2600,
            2610,
            2615,
            2620,
            2625,
            2630,
            2635,
            2640,
            2645,
            2650,
            2655,
            2660,
            2665,
            2670,
            2675,
            2680,
            2685,
            2690,
            2695,
            2700,
            2705,
            2710,
            2715,
            2720,
            2725,
            2730,
            2735,
            2740,
            2745,
            2750,
            2755,
            2760,
            2765,
            2770,
            2775,
            2780,
            2785,
            2790,
            2795,
            2800,
            2805,
            2810,
            2815,
            2835,
            2860,
            2900,
        ])
    vols = [
        0.684882717072609,
        0.6548002174209514,
        0.6279717042323061,
        0.6040669049212617,
        0.5769233835086068,
        0.5512534351594732,
        0.5260245499632258,
        0.5004353919117,
        0.4741366518169333,
        0.46171589561249216,
        0.4457089283432941,
        0.4336614266663264,
        0.420159764469498,
        0.4074628373496824,
        0.3931682390848574,
        0.3814047904881801,
        0.37929970817058073,
        0.3771088224218263,
        0.3724714977308359,
        0.36029419336555424,
        0.35467069448268806,
        0.3505327949033959,
        0.3441904382413214,
        0.3392727917494692,
        0.33306859556194446,
        0.32820593458977093,
        0.3243137942797042,
        0.32204084870033645,
        0.3168000315981532,
        0.3109143207658998,
        0.3050420836154825,
        0.30241566311445206,
        0.29948796266862154,
        0.29609035936524486,
        0.2923777072285143,
        0.28951623883712746,
        0.28584033838767425,
        0.283342147794602,
        0.2808533651372528,
        0.27703523377755246,
        0.27371493615870945,
        0.2708906740100327,
        0.2678887418986713,
        0.2645328136650213,
        0.26234402136468965,
        0.2585977172018311,
        0.25550003988953746,
        0.2521896614376435,
        0.2495339851370865,
        0.24665927818229774,
        0.24355339309186683,
        0.24020198229067014,
        0.23658800157061083,
        0.23457380906338043,
        0.23040670495884208,
        0.2278656924642955,
        0.22304945749920857,
        0.21988751701341647,
        0.2184983910827269,
        0.21470247194448602,
        0.21050624458263925,
        0.20817463333507674,
        0.20550780781621286,
        0.19996741584940433,
        0.19900703596491134,
        0.19506177682405323,
        0.19054732989021844,
        0.18842657099566548,
        0.18589260856179804,
        0.18287835748424114,
        0.17929170978590483,
        0.17500063441150882,
        0.18575101811296996,
        0.20717302702012957,
        0.22524785579801052,
    ]
    weightsV = [
        1.1789826122551597,
        1.224744871391589,
        1.284523257866513,
        1.3601470508735443,
        1.4317821063276355,
        1.5165750888103102,
        1.6124515496597098,
        1.7175564037317668,
        1.8395212376698413,
        1.8973665961010275,
        1.949358868961793,
        2.024845673131659,
        2.0976176963403033,
        2.179449471770337,
        2.258317958127243,
        2.32379000772445,
        2.345207879911715,
        2.3664319132398464,
        2.4083189157584592,
        2.479919353527449,
        2.5199206336708304,
        2.569046515733026,
        2.6076809620810595,
        2.6551836094703507,
        2.7625312572654126,
        2.7477263328068173,
        2.765863337187866,
        2.7928480087537886,
        2.871393034605969,
        2.964704653791087,
        3.0174928596261394,
        3.0017001984929568,
        3,
        3.024896692450835,
        3.120391338480345,
        2.9916550603303182,
        2.947349434130382,
        2.8809720581775866,
        2.8284271247461903,
        2.8083087326973732,
        2.711088342345192,
        2.6685599339741506,
        2.62445329583912,
        2.565469285152568,
        2.4899799195977463,
        2.4289915602982237,
        2.4279079146675357,
        2.313006701244076,
        2.258317958127243,
        2.202271554554524,
        2.1447610589527217,
        2.085665361461421,
        2.024845673131659,
        1.97484176581315,
        1.91049731745428,
        1.857417562100671,
        1.7888543819998317,
        1.7320508075688772,
        1.6881943016134133,
        1.6278820596099706,
        1.5652475842498528,
        1.5165750888103102,
        1.466287829861518,
        1.3964240043768943,
        1.3601470508735443,
        1.3038404810405297,
        1.2449899597988732,
        1.2041594578792296,
        1.161895003862225,
        1.118033988749895,
        1.0723805294763609,
        1.02469507659596,
        1.0099504938362078,
        1.0910894511799618,
        1.005037815259212,
    ]
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
    #plocalibrateXSSVISectiont!(logmoneynessA, sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(xssvi,logmoneynessA)).*100,label="XSSVI")

    prices, wv = Collocation.weightedPrices(true, strikes, vols, w, forward, 1.0, tte, vegaFloor=1e-5)

    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, wv, tte, forward, 1.0, deg=3, degGuess=3)
    sol3 = Collocation.Polynomial(isoc)
    ivk3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol3, true, strikes, forward, 1.0), forward, strikes, tte, 1.0)
    rmse3 = StatsBase.L2dist(w .* vols, w .* ivk3)
    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, wv, tte, forward, 1.0, deg=5, degGuess=3, minSlope=1e-5)
    sol5 = Collocation.Polynomial(isoc)
    ivk5 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol5, true, strikes, forward, 1.0), forward, strikes, tte, 1.0)
    rmse5 = StatsBase.L2dist(w .* vols, w .* ivk5)
    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, wv, tte, forward, 1.0, deg=9, degGuess=3, minSlope=1e-5)
    sol9 = Collocation.Polynomial(isoc)
    ivk9 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol9, true, strikes, forward, 1.0), forward, strikes, tte, 1.0)
    rmse9 = StatsBase.L2dist(w .* vols, w .* ivk9)
    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, wv, tte, forward, 1.0, deg=11, degGuess=3, minSlope=1e-5)
    sol11 = Collocation.Polynomial(isoc)
    ivk11 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol11, true, strikes, forward, 1.0), forward, strikes, tte, 1.0)
    rmse11 = StatsBase.L2dist(w .* vols, w .* ivk11)

    bsple, m = Collocation.makeExpBSplineCollocation(
        strikes,
        prices,
        wv,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=10,
        minSlope=1e-8,
        rawFit=false,
    )
    ivkbe = @. Black.impliedVolatility(true, Collocation.priceEuropean(bsple, true, strikes, forward, 1.0), forward, strikes, tte, 1.0)
    rmsebe = StatsBase.L2dist(w .* vols, w .* ivkbe)
    bspl3, m = Collocation.makeBSplineCollocation(
        strikes,
        prices,
        wv,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=10,
        minSlope=1e-8,
        rawFit=false, N=3,
    )
    ivkb3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(bspl3, true, strikes, forward, 1.0), forward, strikes, tte, 1.0)
    rmseb3 = StatsBase.L2dist(w .* vols, w .* ivkb3)
    lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, wv, useVol=false, model=PDDE.Quadratic(), location="Equidistributed", size=10, L=strikes[1], U=strikes[end])
    ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0)
    rmseq = StatsBase.L2dist(w .* vols, w .* ivkq)
    #== 
    kFine = forward.*exp.(range(logmoneynessA[1],stop=logmoneynessA[end], length=1001));
    p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bsple,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Exp B-Spline")
    plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bspl3,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Cubic B-spline")
    plot!(p3, size=(480,320))
    savefig("~//mypapers/eqd_book/aapl_20131028_vol_bspl.pdf")
    p4=plot(log.(kFine./forward), Collocation.density.(bsple,kFine),label="Exp B-spline",xlab="Log-moneyness",ylab="Probability density",color=2)
    plot!(p4,log.(kFine./forward), Collocation.density.(bspl3,kFine),label="Cubic B-spline",color=3)
    #plot!(p4,yscale=:log10)
    plot!(p4, size=(480,320))
    savefig("~//mypapers/eqd_book/aapl_20131028_dens_bspl.pdf")
    ==#
    #=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, Collocation.priceEuropean.(sol5,true,kFine), forward, kFine, tte, 1.0)*100),label="Collocation")

    p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,PDDE.priceEuropean.(lvgq,true,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Quadratic LVG")
    savefig("~//mypapers/eqd_book/vol_spw_1m_050218_lvgq.pdf")
    p2 = plot(log.(kFine./forward),(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label=:none, xlab="Forward log-moneyness",ylab="Probability density")
    plot!(p2, size=(480,380))
    savefig("~//mypapers/eqd_book/density_spw_1m_050218_lvgq.pdf")
    =#

    λs = [1.6e4, 3.2e4, 6.4e4]
    for λ in λs
        fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (wv), λ=λ, eps=1e-13, solver="GI")
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
    lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, wv, useVol=true, L=strikes[1] / 2, U=strikes[end] * 2, penalty=1.0)
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
    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, wv, tte, forward, 1.0, deg=3, degGuess=1)
    sol = Collocation.Polynomial(isoc)
    dev = exp(3 * vols[3] * sqrt(tte))
    ivk3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol, true, strikes, forward, 1.0), forward, strikes, tte, 1.0)
    # kFine = collect(range(k[1]/dev,stop=k[end]*dev,length=201))
    ivkFine3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol, true, kFine, forward, 1.0), forward, kFine, tte, 1.0)
    rmse3 = StatsBase.L2dist(w .* ivk3, w .* vols)
    #    isoc, m = Collocation.makeIsotonicCollocation(k, prices, wv, tte, forward, 1.0, deg = 5, degGuess = 1) #strangely minSlope has big influence
    isoc, m = Collocation.makeIsotonicCollocation(strikes, prices, wv, tte, forward, 1.0, deg=5, degGuess=3, minSlope=1e-5)
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
        wv,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=8,
        minSlope1e - 8,
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
    strikesf, pricesf = AQFED.Collocation.filterConvexPrices(strikes, prices, wv, forward, tol=1e-6)
    allStrikes = vcat(0.0, strikesf, forward * exp(logmoneynessA[end] * 3))
    allPrices = vcat(forward, pricesf, 0.0)
    allWeights = vcat(sum(wv), wv, sum(wv))
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
    )
    rmseScha = StatsBase.L2dist(w .* ivkScha, w .* vols)
    println("scha ", rmseScha)

    csf, rmseSchaFit = fitConvexSchabackRationalSpline(
        allStrikes, allPrices,
        allWeights,
        leftB,
        rightB;
        penalty=0.1,
    )
    ivkSchaFit = @. Black.impliedVolatility(
        true,
        csf(strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseSchaFit = StatsBase.L2dist(w .* ivkSchaFit, w .* vols)
    println("schafit ", rmseSchaFit)
    step = floor(Int, length(strikesf) / 7)
    subset = collect(1:step:length(strikesf))
    if subset[end] != length(strikesf)
        append!(subset, length(strikesf))
    end
    allStrikest = vcat(0.0, strikesf[subset], forward * exp(logmoneynessA[end] * 3))
    allPricest = vcat(forward, pricesf[subset], 0.0)

    cs8 = Math.makeConvexSchabackRationalSpline(allStrikest, allPricest, leftB, rightB, iterations=128)
    ivkScha8 = @. Black.impliedVolatility(
        true,
        cs8(strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseScha8 = StatsBase.L2dist(w .* ivkScha8, w .* vols)
    println("scha8 ", rmseScha8)
    #=
    	 plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    	 plot!(log.(kFine./forward),Black.impliedVolatility.(true, max.(csf.(kFine),1e-16), forward, kFine, tte, 1.0) .* 100, label=string("Schaback λ=",1e-5))
    	 plot!(log.(kFine./forward),Black.impliedVolatility.(true, max.(cs8.(kFine),1e-16), forward, kFine, tte, 1.0) .* 100, label=string("Schaback on 8 knots"))
    	plot!(size=(480,380))
    	savefig("~//mypapers/eqd_book/aapl_20131028_schaback.pdf")

    	plot(log.(kFine./forward), AQFED.Math.evaluateSecondDerivative.(cs8,kFine),label="Schaback on 8 knots", xlab="Forward log-moneyness",ylab="Probability density")
    	plot!(log.(kFine./forward), AQFED.Math.evaluateSecondDerivative.(csf,kFine),label=string("Schaback λ=",1e-5))
    	plot!(legend=:bottom)
    	plot!(size=(480,380))
    	 savefig("~//mypapers/eqd_book/aapl_20131028_schaback_dens.pdf")
    	=#
    kernel2 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=2)
    ivkMLN2 = @. AQFED.Black.impliedVolatility(
        true,
        AQFED.VolatilityModels.priceEuropean.(kernel2, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN2 = StatsBase.L2dist(w .* ivkMLN2, w .* vols)
    kernel3 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=3)
    ivkMLN3 = @. AQFED.Black.impliedVolatility(
        true,
        AQFED.VolatilityModels.priceEuropean.(kernel3, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN3 = StatsBase.L2dist(w .* ivkMLN3, w .* vols)

    kernel4 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=4)
    ivkMLN4 = @. AQFED.Black.impliedVolatility(
        true,
        AQFED.VolatilityModels.priceEuropean.(kernel4, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN4 = StatsBase.L2dist(w .* ivkMLN4, w .* vols)
    kernel6 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=6)
    ivkMLN6 = @. AQFED.Black.impliedVolatility(
        true,
        AQFED.VolatilityModels.priceEuropean.(kernel6, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN6 = StatsBase.L2dist(w .* ivkMLN6, w .* vols)

    #=
    	 plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    	 plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(true, AQFED.VolatilityModels.priceEuropean.(kernel2,true,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 2")
    	 plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(true, AQFED.VolatilityModels.priceEuropean.(kernel4,true,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 4")
    	 plot!(log.(kFine./forward),AQFED.Black.impliedVolatility.(true, AQFED.VolatilityModels.priceEuropean.(kernel6,true,kFine), forward, kFine, tte, 1.0) .* 100, label="Mixture of 6")
    	plot!(size=(800,320),margin=3Plots.mm)
    	savefig("~//mypapers/eqd_book/vol_spw_1m_050218_mln6.pdf")

    	plot(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel2,kFine),label="Mixture of 2", color=2,xlab="Forward log-moneyness",ylab="Probability density")
    	plot!(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel4,kFine),label="Mixture of 4",color=3)
    	plot!(log.(kFine./forward), AQFED.VolatilityModels.density.(kernel6,kFine),label="Mixture of 6",color=4)
    	plot!(size=(800,320),margin=3Plots.mm)
    	 savefig("~//mypapers/eqd_book/density_spw_1m_050218_mln6.pdf")
    	=  =#
end


@testset "TSLA1m" begin
    tte = 0.095890
    forward = 357.755926
    strikes =
        Float64.([
            150,
            155,
            160,
            165,
            175,
            180,
            185,
            190,
            195,
            200,
            205,
            210,
            215,
            220,
            225,
            230,
            235,
            240,
            245,
            250,
            255,
            260,
            265,
            270,
            275,
            280,
            285,
            290,
            295,
            300,
            305,
            310,
            315,
            320,
            325,
            330,
            335,
            340,
            345,
            350,
            355,
            360,
            365,
            370,
            375,
            380,
            385,
            390,
            395,
            400,
            405,
            410,
            415,
            420,
            425,
            430,
            435,
            440,
            445,
            450,
            455,
            460,
            465,
            470,
            475,
            480,
            500,
            520,
            540,
            560,
            580,
        ])
    vols = [
        1.027152094560499,
        0.9905195749900226,
        0.9657262376591365,
        0.9405597986379826,
        0.9181603362313814,
        0.9019382314978117,
        0.8846745842549402,
        0.865754243981787,
        0.8456155492434201,
        0.8245634579529838,
        0.8028174604214972,
        0.78053958851195,
        0.7636802684802435,
        0.7454192306685303,
        0.7260651215584285,
        0.7058414693439228,
        0.6849143304434797,
        0.663409356115238,
        0.6462309799739909,
        0.6301291739261891,
        0.6130540004186168,
        0.5946923076348443,
        0.5811921286363728,
        0.5687314890047378,
        0.5539815904720001,
        0.5422671292669776,
        0.5338887990387771,
        0.5234154661207794,
        0.5168510552270313,
        0.5072806473672073,
        0.4997973159961656,
        0.4896563997378466,
        0.4823975850368014,
        0.47936818364069134,
        0.48000585384055006,
        0.4757525564073338,
        0.4711478482467228,
        0.46788352167691083,
        0.46562175169660713,
        0.46299652559206567,
        0.45939930288424485,
        0.458565105643866,
        0.45790487479637937,
        0.45521398441321903,
        0.453447302139774,
        0.4504013827012644,
        0.448004721643358,
        0.4491995553643971,
        0.4478840707248649,
        0.45006593113797866,
        0.4517530880150887,
        0.4499007489879635,
        0.448814967685824,
        0.45160477568536983,
        0.4563938928347205,
        0.4600222064217672,
        0.46102443173801966,
        0.46406468170261544,
        0.4709795491400157,
        0.4762595045128011,
        0.4810009989573377,
        0.4855906965577297,
        0.4906446878461756,
        0.4960612773473766,
        0.5011170526132832,
        0.5059204240563133,
        0.5159102206249263,
        0.5505625146941026,
        0.5783881966646062,
        0.599260903580561,
        0.6259792014943735,
    ]
    weightsV = [
        1.7320508075688772,
        1,
        1.224744871391589,
        1,
        2.738612787525831,
        1.558387444947959,
        1.9999999999999998,
        1.2602520756252087,
        1.3301243435223526,
        2.273030282830976,
        1.3944333775567928,
        1.2089410496539776,
        1.9999999999999998,
        2.0976176963403033,
        3.500000000000001,
        3.286335345030995,
        2.6692695630078282,
        2.7838821814150116,
        3.1622776601683804,
        3.605551275463988,
        3.3541019662496834,
        3,
        2.9742484506432634,
        3.6469165057620923,
        3.8729833462074152,
        4.183300132670376,
        3.7505555144093887,
        4.1918287860346295,
        3.7670248460125917,
        4.795831523312714,
        4.527692569068711,
        3.482097069296032,
        3.2333489534143167,
        3.687817782917155,
        6.3245553203367555,
        6.837397165588683,
        7.365459931328131,
        7.0992957397195395,
        7.628892449104261,
        7.461009761866454,
        8.706319543871567,
        8.78635305459552,
        7.000000000000021,
        7.745966692414834,
        8.093207028119338,
        6.16441400296897,
        4.974937185533098,
        4.650268809434567,
        4.315669125408015,
        4.636809247747854,
        4.732863826479693,
        3.1144823004794873,
        2.8809720581775857,
        2.8284271247461894,
        2.7718093060793882,
        4.092676385936223,
        2.7041634565979926,
        2.652259934210953,
        3.710691413905333,
        3.777926319123662,
        3.929942040850535,
        3.921096785339529,
        3.70809924354783,
        3.517811819867573,
        3.3354160160315844,
        3.1622776601683777,
        1.3483997249264843,
        1.8929694486000912,
        1.914854215512676,
        1.699673171197595,
        1.8708286933869707,
    ]
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

    prices, wv = Collocation.weightedPrices(true, strikes, vols, w, forward, 1.0, tte, vegaFloor=1e-5)
    λs = [50.0, 200.0, 400.0, 1600.0]
    for λ in λs
        fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (wv), λ=λ, eps=1e-13, solver="GI")
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
    fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (wv),λ=λ,eps=1e-13,solver="GI")
    plot!(p3,log.(kFine./forward), AQFED.VolatilityModels.evaluateSecondDerivative.(fengler,kFine),label=string("Fengler λ=",λ))
    end
    plot(p3,margin=3Plots.mm,size=(480,380))
    savefig(p3,"~//mypapers/eqd_book/tsla_180615_180720_fengler_dens.pdf")
    p4 = plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    for λ in λs
    fengler = AQFED.VolatilityModels.calibrateFenglerSlice(tte, forward, strikes, prices, (wv),λ=λ,eps=1e-13,solver="GI")
    plot!(p4,log.(kFine./forward), @.(Black.impliedVolatility(true, max.(fengler.(kFine),1e-16), forward, kFine, tte, 1.0)*100),label=string("Fengler λ=",λ))
    end
    plot(p4,margin=3Plots.mm,size=(480,380))
    savefig(p4,"~//mypapers/eqd_book/tsla_180615_180720_fengler.pdf")

    =#

    strikesf, pricesf = AQFED.Collocation.filterConvexPrices(strikes, prices, wv, forward, tol=1e-6)
    allStrikes = vcat(0.0, strikesf, forward * exp(logmoneynessA[end] * 3))
    allPrices = vcat(forward, pricesf, 0.0)
    allWeights = vcat(sum(wv), wv, sum(wv))
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
    )
    rmseScha = StatsBase.L2dist(w .* ivkScha, w .* vols)
    println("scha ", rmseScha)

    csf, rmseSchaFit = fitConvexSchabackRationalSpline(
        allStrikes, allPrices,
        allWeights,
        leftB,
        rightB;
        penalty=1.0,
    )
    ivkSchaFit = @. Black.impliedVolatility(
        true,
        csf(strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseSchaFit = StatsBase.L2dist(w .* ivkSchaFit, w .* vols)
    println("schafit ", rmseSchaFit)
    step = floor(Int, length(strikesf) / 7)
    subset = collect(1:step:length(strikesf))
    if subset[end] != length(strikesf)
        append!(subset, length(strikesf))
    end
    allStrikest = vcat(0.0, strikesf[subset], forward * exp(logmoneynessA[end] * 3))
    allPricest = vcat(forward, pricesf[subset], 0.0)

    cs8 = Math.makeConvexSchabackRationalSpline(allStrikest, allPricest, leftB, rightB, iterations=128)
    ivkScha8 = @. Black.impliedVolatility(
        true,
        cs8(strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseScha8 = StatsBase.L2dist(w .* ivkScha8, w .* vols)
    println("scha8 ", rmseScha8)


    pp = PPInterpolation.CubicSplineNatural(log.(strikes), vols .^ 2)
    rbf = AQFED.VolatilityModels.calibrateMultiquadric(tte, forward, log.(strikes), vols, w1, knots=log.(strikes))

    lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, wv, useVol=true, L=strikes[1] / 2, U=strikes[end] * 2)
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
    lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, wv, useVol=false, model=PDDE.Quadratic(), location="Equidistributed", size=10, L=strikes[1], U=strikes[end])
    ivkq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0)
    rmseq = StatsBase.L2dist(w .* vols, w .* ivkq)


    bspl, m = Collocation.makeExpBSplineCollocation(
        strikes,
        prices,
        wv,
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
        wv,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=10,
        minSlope=1e-8,
        rawFit=false,
    )
    ivkbe = @. Black.impliedVolatility(true, Collocation.priceEuropean(bsple, true, strikes, forward, 1.0), forward, strikes, tte, 1.0)
    rmsebe = StatsBase.L2dist(w .* vols, w .* ivkbe)
    bspl3, m = Collocation.makeBSplineCollocation(
        strikes,
        prices,
        wv,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=10,
        minSlope=1e-8,
        rawFit=false, N=3, extrapolationFactor=1.1, optimizerName="LeastSquaresOptim",
    )

    ivkb3 = @. Black.impliedVolatility(true, Collocation.priceEuropean(bspl3, true, strikes, forward, 1.0), forward, strikes, tte, 1.0)
    rmseb3 = StatsBase.L2dist(w .* vols, w .* ivkb3)
    #==
    p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bsple,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Exp B-Spline")
    plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,Collocation.priceEuropean.(bspl3,true,kFine,forward,1.0)), forward, (kFine), tte, 1.0) .* 100, label="Cubic B-spline")
    plot!(p3, size=(480,320))
    savefig("~//mypapers/eqd_book/tsla_180615_180720_vol_bspl.pdf")
    p4=plot(log.(kFine./forward), Collocation.density.(bsple,kFine),label="Exp B-spline",xlab="Log-moneyness",ylab="Probability density",color=2)
    plot!(p4,log.(kFine./forward), Collocation.density.(bspl3,kFine),label="Cubic B-spline",color=3)
    #plot!(p4,yscale=:log10)
    plot!(p4, size=(480,320),legend=:topleft)
    savefig("~//mypapers/eqd_book/tsla_180615_180720_dens_bspl.pdf")


    p3=plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference", markersize=3, markerstrokewidth=-1,markeralpha=0.5); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(p3, log.(kFine./forward), Black.impliedVolatility.(true, max.(1e-32,PDDE.priceEuropean.(lvgq,true,kFine)), forward, (kFine), tte, 1.0) .* 100, label="Quadratic LVG")
    plot!(p3, size=(480,320))
    savefig("~//mypapers/eqd_book/tsla_180615_180720_vol_lvgq.pdf")
    p2 = plot(log.(kFine./forward),(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label=:none, xlab="Forward log-moneyness",ylab="Probability density")
    plot!(p2, size=(480,320))
    savefig("~//mypapers/eqd_book/tsla_180615_180720_dens_lvgq.pdf")

    ==#


    weightsA = ones(length(vols))
    strikes = k
    sumw2 = sum(weightsA .^ 2)
    w = weightsA ./ sqrt(sumw2)


    kernel2 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=2)
    ivkMLN2 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel2, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN2 = StatsBase.L2dist(w .* ivkMLN2, w .* vols)
    kernel3 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=3)
    ivkMLN3 = @. AQFED.Black.impliedVolatility(
        strikes >= forward,
        AQFED.VolatilityModels.priceEuropean.(kernel3, strikes >= forward, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN3 = StatsBase.L2dist(w .* ivkMLN3, w .* vols)

    kernel4 = AQFED.VolatilityModels.calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol=false, size=4)
    ivkMLN4 = @. AQFED.Black.impliedVolatility(
        true,
        AQFED.VolatilityModels.priceEuropean.(kernel4, true, strikes),
        forward,
        strikes,
        tte,
        1.0,
    )
    rmseMLN4 = StatsBase.L2dist(w .* ivkMLN4, w .* vols)

    #=
    plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(kFine./forward),Black.impliedVolatility.(true, max.(csf.(kFine),1e-16), forward, kFine, tte, 1.0) .* 100, label=string("Schaback λ=",1))
    plot!(log.(kFine./forward),Black.impliedVolatility.(true, max.(cs8.(kFine),1e-16), forward, kFine, tte, 1.0) .* 100, label=string("Schaback on 8 knots"))
    plot!(size=(480,380))
    savefig("~//mypapers/eqd_book/tsla_180615_180720_schaback.pdf")

    plot(log.(kFine./forward), AQFED.Math.evaluateSecondDerivative.(cs8,kFine),label="Schaback on 8 knots ", yscale=:log10, xlab="Forward log-moneyness",ylab="Probability density")
    plot!(log.(kFine./forward), AQFED.Math.evaluateSecondDerivative.(csf,kFine),label="Schaback λ=1")
    plot!(legend=:topleft)
    plot!(size=(480,380))
    savefig("~//mypapers/eqd_book/tsla_180615_180720_schaback_dens.pdf")

    plot(log.(strikes./forward), vols.*100, seriestype= :scatter, label="Reference",markersize=3,markerstrokewidth=-1); xlabel!("Forward log-moneyness"); ylabel!("Volatility in %")
    plot!(log.(strikes./forward),ivkSVI0.*100,label="SVI")
    plot!(log.(strikes./forward),ivkXSSVI.*100,label="XSSVI")
    plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, max.(fengler.(kFine),1e-16), forward, kFine, tte, 1.0)*100),label="Fengler λ=1e-13")

    #savefig("~//mypapers/eqd_book/jaeckel_case_i_fengler_rbf_dens.pdf")

    plot!(log.(kFine./forward), @.(Black.impliedVolatility(true, PDDE.priceEuropean.(lvg,true,kFine), forward, kFine, tte, 1.0)*100),label="LVG Linear Black")
    #savefig("~//mypapers/eqd_book/jaeckel_case_i_fengler_rbf.pdf")

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
    strikes =
        Float64.([
            20,
            25,
            50,
            55,
            75,
            100,
            120,
            125,
            140,
            150,
            160,
            175,
            180,
            195,
            200,
            210,
            230,
            240,
            250,
            255,
            260,
            270,
            275,
            280,
            285,
            290,
            300,
            310,
            315,
            320,
            325,
            330,
            335,
            340,
            350,
            360,
            370,
            380,
            390,
            400,
            410,
            420,
            430,
            440,
            450,
            460,
            470,
            480,
            490,
            500,
            510,
            520,
            550,
            580,
            590,
            600,
            650,
            670,
            680,
            690,
            700,
        ])
    vols = [
        1.2174498333432284,
        1.1529735541872161,
        1.004004022581606,
        0.9836878310419587,
        0.9059774069943589,
        0.8197362843642744,
        0.7716707471169627,
        0.7596970666840304,
        0.7255349986293692,
        0.7036962946028745,
        0.6872956097489304,
        0.6628579700139684,
        0.6544686699577009,
        0.6310048431894972,
        0.623904988806911,
        0.613032792224091,
        0.5887489086881706,
        0.5772059051701408,
        0.5649684953388189,
        0.561418253373758,
        0.5576041889980867,
        0.549252193194508,
        0.5453501604969959,
        0.5412198514376074,
        0.536873384908355,
        0.5323213596815215,
        0.5226363012219939,
        0.5185316686163134,
        0.5160935996651511,
        0.5134141336907875,
        0.5105033610267267,
        0.5073701448945144,
        0.504289607300568,
        0.5013959697030378,
        0.4961897259221572,
        0.49144782371138285,
        0.48645638955352194,
        0.48167998000739076,
        0.47609375436090395,
        0.4697217879783784,
        0.4680183980319167,
        0.46552636426393684,
        0.46227241039855843,
        0.45969798571592974,
        0.4567098331715741,
        0.4539007045757515,
        0.45039656985941307,
        0.44619726030207824,
        0.44495717278530045,
        0.44309517634427675,
        0.44061210739821544,
        0.4398554529535865,
        0.4342208537081089,
        0.4299834814231043,
        0.42752901586299163,
        0.42452445491007096,
        0.4222672729031062,
        0.4203436892852212,
        0.4195681468757861,
        0.4188408351329149,
        0.4177462959039442,
    ]
    wv = [
        7.7136243102707525,
        3.8729833462074152,
        6.082762530298218,
        1.9639610121239313,
        2.9154759474226504,
        13.527749258468587,
        2.444949442785633,
        2.5260547066428276,
        3.0797481203579755,
        6.866065623255938,
        3.496542070619614,
        5.535599077142163,
        5.8638396451996595,
        3.9657625656704085,
        5.594114554338226,
        3.5878044017939312,
        5.588136784946543,
        9.637167633698208,
        5.006246098625194,
        3.8156679669355786,
        3.762808812407317,
        3.9349550147037164,
        4.217311262607291,
        4.0473389258811014,
        4.58043580425081,
        4.227406739590535,
        10.737783756436894,
        5.041825066382212,
        4.936635531449573,
        4.723502451553941,
        4.491261356392942,
        4.533133107978179,
        4.638349460129725,
        4.647281164532899,
        4.730579074318318,
        4.135214625627067,
        4.024922359499621,
        3.924283374069717,
        3.82099463490856,
        4.141485088006137,
        3.636237371545237,
        3.53767600483022,
        3.605551275463989,
        3.570311812262566,
        3.433149698736217,
        3.4995590551163587,
        3.445782599282636,
        4.071476768559913,
        3.596873642484542,
        3.5514649408105448,
        3.8544964466377296,
        3.9829180714190437,
        3.3598305818093763,
        3.0684025713101346,
        3.0127932350101747,
        3.505098327538656,
        2.5895718474182514,
        2.5248762345905194,
        2.506513254633252,
        2.4022558842332624,
        3.1008683647302107,
    ]
    #    weightsV = [16.25288045106638, 8.160911900438244, 12.619088815272084, 3.675840734509003, 5.776554424173832, 27.512440009161907, 4.638043773388409, 4.883079553130767, 5.94131943234476, 13.473174307161502, 6.680481079152589, 10.538920929379609, 11.131996465169651, 7.350499176975279, 10.404413260007315, 6.461526063169936, 9.975723639859746, 16.974674230580774, 8.684906846576, 6.471547708410667, 6.3305947378547245, 6.5136123938783905, 6.92043607456792, 6.585207524684852, 7.384412661775439, 6.754704320859774, 17.091168363110985, 7.7529771845242506, 7.51038911665913, 7.111728791431016, 6.678533051977762, 6.668891459306908, 6.752119237968658, 6.690994656273483, 6.659451607392078, 5.772855007876042, 5.810926108790698, 5.841240932662621, 5.865481045309425, 6.621016050951635, 5.924276744071027, 5.929376447556495, 6.217841480543634, 6.317299897492222, 6.232948658379371, 6.515095778640602, 6.572366611401739, 8.057812002697537, 7.227514274402218, 7.298089942473972, 8.148515888009042, 8.548424627243886, 7.6509577692757835, 7.3483668093512335, 7.3353593339631376, 8.809531159992671, 6.865235860006001, 6.878549795031439, 6.92325311817954, 6.686900221418176, 8.93734577371638]
    logmoneynessA = log.(strikes ./ forward)
    sumw2 = sum(wv .^ 2)
    w = wv ./ sqrt(sumw2)
    svi0, rmsesvi = AQFED.VolatilityModels.calibrateSVISection(tte, forward, logmoneynessA, vols, wv, aMin=0.0)
    ivkSVI0 = sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi0, logmoneynessA))
    rmseSVI0 = StatsBase.L2dist(w .* vols, w .* ivkSVI0)
    #SVI is good enough on this example

end
#plot(y,@.(gatheralDenomFinite(x->AQFED.TermStructure.varianceByLogmoneyness(slice,x)*tte,y)))
#plot(y,@.(gatheralDenomFinite(x->AQFED.TermStructure.varianceByLogmoneyness(slice,x)*tte,y)/sqrt(2*π*AQFED.TermStructure.varianceByLogmoneyness(slice,y)*tte)*exp(-0.5*(y/sqrt(AQFED.TermStructure.varianceByLogmoneyness(slice,y)*tte)+sqrt(AQFED.TermStructure.varianceByLogmoneyness(slice,y)*tte)/2)^2)  )  )

@testset "Kahale" begin
    r = 0.06
    q = 0.0262

    relStrikes = [0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4]
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
    vols = [0.19 0.168 0.133 0.113 0.102 0.097 0.12 0.142 0.169 0.2;
        0.177 0.155 0.138 0.125 0.109 0.103 0.1 0.114 0.13 0.15;
        0.172 0.157 0.144 0.133 0.118 0.104 0.1 0.101 0.108 0.124;
        0.171 0.159 0.149 0.137 0.127 0.113 0.106 0.103 0.1 0.11;
        0.171 0.159 0.15 0.138 0.128 0.115 0.107 0.103 0.099 0.108;
        0.169 0.16 0.151 0.142 0.133 0.124 0.119 0.113 0.107 0.102;
        0.169 0.161 0.153 0.145 0.137 0.13 0.126 0.119 0.115 0.111;
        0.168 0.161 0.155 0.149 0.143 0.137 0.133 0.128 0.124 0.123;
        0.168 0.162 0.157 0.152 0.148 0.143 0.139 0.135 0.13 0.128;
        0.168 0.164 0.159 0.154 0.151 0.148 0.144 0.14 0.136 0.132]
    prices = zeros(Float64, LinearAlgebra.size(vols))
    strikesM = similar(prices)
    forwards = zeros(Float64, length(expiries))
    weightsM = similar(prices)
    ys = similar(prices)
    for (i, expiry) in enumerate(expiries)
        forwards[i] = spot * exp((r - q) * expiry)
        pricesi, wv = Collocation.weightedPrices(true, strikes, vols[i, :], ones(length(strikes)), forwards[i], 1.0, expiry)
        for (j, strike) in enumerate(strikes)
            strikesM[i, j] = strike
            ys[i, j] = log(strike / forwards[i])
            weightsM[i, j] = wv[j]
            prices[i, j] = pricesi[j]
        end
    end
    surface = AQFED.VolatilityModels.calibrateFenglerSurface(expiries, forwards, strikesM, prices, weightsM, λ=1e-2, solver="GI")

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
    savefig("~//mypapers/eqd_book/kahale_fengler_totalvar.pdf")

    pdfsurfacep(svi,z,forward,tte) = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> AQFED.VolatilityModels.price(svi,log(y/forward),tte),x),z)

    pdfvalues = zeros(length(expiries),length(k))
    for (i,tte) in enumerate(expiries)
    		  pdfvalues[i,:] = @.(pdfsurfacep(surface, k,forwards[i],tte))
    	  end
    plot(expiries, k, pdfvalues',st=:surface,camera=(80,30),ylab="Strike",xlab="Expiry",zlab="Probability Density",colorbar=false)
    plot!(size=(480,340))
    savefig("~//mypapers/eqd_book/kahale_fengler_dens.pdf")

    #PDF bof
    #LV 3D?

    #Prior RBF?
    =#

    ### LVG
    # forward*(1-strike/forward)
    surfaceLVG = PDDE.calibrateLVGSurface(expiries, forwards, strikesM, prices, weightsM)
    for (i, expiry) in enumerate(expiries)
        for (j, strike) in enumerate(strikes)
            y = log(strike / forwards[i])
            volij = sqrt(PDDE.varianceByLogmoneyness(surfaceLVG, y, expiry))
            println(expiry, " ", strike, " ", volij, " ", vols[i, j], " ", volij - vols[i, j])
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
    savefig("~//mypapers/eqd_book/kahale_lvg_totalvar.pdf")
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
    savefig("~//mypapers/eqd_book/kahale_lvg_iv3d.pdf")
    plot(t, k, ivMatrix'.*100,st=:surface,camera=(45,30),ylab="Strike",xlab="Expiry",zlab="Implied volatility in %", legend=:none,  zguidefontrotation=90,margin=0Plots.mm,size=(600,600))
    savefig("~//mypapers/eqd_book/kahale_lvg_iv3db.pdf")

    lvMatrix = zeros(length(t),length(k))
    eps=1e-4
    for (i,tte)= enumerate(t)
    	fi = spot*exp((r-q)*tte)
    	w(y) = PDDE.varianceByLogmoneyness(surfaceLVG, y,tte)*tte
    	lvMatrix[i,:] = @. sqrt((PDDE.varianceByLogmoneyness(surfaceLVG, log(k/fi),tte+eps)*(tte+eps)-PDDE.varianceByLogmoneyness(surfaceLVG, log(k/fi),tte)*tte)/(eps*gatheralDenomFinite(w, log(k/fi))))
    end
    plot(t, k, lvMatrix'.*100,st=:surface,camera=(-45,30),ylab="Strike",xlab="Expiry",zlab="Local volatility in %", legend=:none, zguidefontrotation=90,margin=0Plots.mm,size=(600,600))
    savefig("~//mypapers/eqd_book/kahale_lvg_lv3d.pdf")
    plot(t, k, lvMatrix'.*100,st=:surface,camera=(45,30),ylab="Strike",xlab="Expiry",zlab="Local volatility in %", legend=:none,  zguidefontrotation=90,margin=0Plots.mm,size=(600,600))
    savefig("~//mypapers/eqd_book/kahale_lvg_lv3db.pdf")

    =#
end
using PPInterpolation
@testset "Heston15March2021" begin
    ts = [0.00274, 0.08333, 0.16667, 0.25000, 0.33333, 0.41667,
        0.50000,
        0.75000,
        1.00000,
        1.50000,
        2.00000,
        3.00000,
        4.00000,
        5.00000,
        7.00000,
        10.00000,
        12.00000,
        15.00000,
        20.00000,
        25.00000,
        30.00000,
        40.00000,
        50.00000]
    rates = [0.0098,
        0.0271,
        0.0295,
        0.0319,
        0.0367,
        0.0435,
        0.0456,
        0.0525,
        0.0588,
        0.0707,
        0.1016,
        0.2350,
        0.3977,
        0.5322,
        0.7180,
        0.8626,
        0.9194,
        0.9693,
        1.0063,
        1.0199,
        1.0259,
        1.0062,
        0.9774] ./ 100
    logdfDiscountCurve = PPInterpolation.CubicSplineNatural(ts, -rates .* ts)
    ts = [0.0110,
        0.0877,
        0.1836,
        0.2603,
        0.5096,
        0.7589,
        1.0082,
        1.2575,
        1.7562,
        2.2548,
        2.7534,
        3.7699,
        4.7671,
        5.7644,
        6.7616,
        7.7589]
    yields = [5.18, 1.22, 1.13, 1.17, 1.12, 1.12, 1.08, 1.09, 1.12, 1.2, 1.15, 1.1, 1.05, 1.0, 0.99, 0.96] .* (360 / 365 / 100)
    divyieldCurve = PPInterpolation.CubicSplineNatural(ts, yields)
    ts = [
        0.010958904109589041,
        0.08767123287671233,
        0.18356164383561643,
        0.2602739726027397,
        0.5095890410958904,
        0.7589041095890411,
        1.0082191780821919,
        1.2575342465753425,
        1.7561643835616438,
        2.254794520547945,
        2.7534246575342465,
        3.76986301369863,
        4.767123287671233,
        5.764383561643836,
        6.761643835616439,
        7.758904109589041,
        8.775342465753425,
        9.772602739726027,
    ]
    vols = [
        148.8 132.34 111.74 90.8 79.71 67.78 62.48 56.95 51.37 45.74 39.91 36.89 33.87 30.92 28.13 25.6 23.37 21.47 18.84 16.53 14.75 13.98 13.97 14.66 15.66 16.91 18.27 19.64 20.98 22.27 23.5 25.79 27.89 29.83 31.64 33.32 37.1 40.42 46.05 50.74 54.76 58.29 61.44 64.27 66.85 69.19 71.13 72.72 74.04 75.12 76.02 76.75 77.36 77.86 78.28;
        84.27 74.84 62.94 50.92 44.69 38.3 35.75 33.28 30.96 28.76 26.56 25.42 24.26 23.11 21.97 20.86 19.78 18.76 17.49 16.43 15.49 14.74 14.21 13.86 13.73 13.67 13.7 13.84 14.08 14.41 14.84 15.89 17.08 18.32 19.55 20.74 23.49 25.95 30.14 33.61 36.58 39.17 41.47 43.54 45.42 47.11 48.46 49.54 50.42 51.12 51.68 52.13 52.49 52.78 53.01;
        70.4 62.64 52.96 43.42 38.67 34.08 32.36 30.69 29.03 27.35 25.64 24.77 23.9 23.03 22.17 21.31 20.45 19.62 18.74 17.96 17.24 16.59 16.03 15.58 15.27 15.02 14.84 14.72 14.66 14.67 14.73 15.0 15.43 15.95 16.53 17.15 18.73 20.25 23.03 25.43 27.53 29.38 31.04 32.54 33.9 35.08 35.95 36.61 37.09 37.45 37.72 37.92 38.07 38.18 38.27;
        65.62 58.49 49.7 41.14 36.95 32.98 31.47 29.99 28.51 27.02 25.51 24.75 24.0 23.24 22.49 21.75 21.01 20.3 19.56 18.87 18.22 17.61 17.07 16.57 16.21 15.9 15.64 15.45 15.31 15.22 15.18 15.22 15.4 15.66 16.0 16.38 17.41 18.49 20.54 22.39 24.04 25.52 26.85 28.06 29.17 30.11 30.78 31.25 31.58 31.82 31.99 32.11 32.2 32.26 32.3;
        55.99 50.14 43.12 36.55 33.48 30.7 29.63 28.56 27.46 26.33 25.21 24.65 24.08 23.51 22.94 22.37 21.8 21.24 20.73 20.2 19.69 19.18 18.7 18.25 17.86 17.5 17.18 16.9 16.66 16.44 16.26 15.99 15.82 15.72 15.7 15.73 15.97 16.37 17.34 18.37 19.37 20.31 21.18 22.0 22.75 23.36 23.75 24.0 24.16 24.27 24.33 24.38 24.4 24.42 24.43;
        49.8 44.82 39.01 33.77 31.42 29.36 28.51 27.63 26.75 25.86 24.96 24.5 24.04 23.58 23.11 22.64 22.17 21.69 21.22 20.75 20.29 19.84 19.39 18.99 18.58 18.19 17.83 17.52 17.24 17.01 16.82 16.54 16.36 16.25 16.18 16.14 16.11 16.13 16.24 16.39 16.53 16.68 16.82 16.96 17.09 17.19 17.24 17.27 17.29 17.3 17.31 17.31 17.31 17.31 17.31;
        45.92 41.6 36.73 32.33 30.32 28.47 27.72 26.96 26.18 25.4 24.63 24.25 23.86 23.47 23.08 22.69 22.3 21.92 21.47 21.07 20.67 20.27 19.87 19.27 19.0 18.74 18.49 18.26 18.03 17.82 17.62 17.25 16.93 16.66 16.41 16.21 15.84 15.64 15.58 15.79 16.13 16.54 16.97 17.41 17.84 18.17 18.36 18.46 18.53 18.56 18.58 18.59 18.6 18.6 18.6;
        43.41 39.49 35.16 31.27 29.5 27.84 27.16 26.48 25.79 25.09 24.4 24.05 23.7 23.35 23.01 22.66 22.31 21.96 21.59 21.24 20.9 20.56 20.23 19.59 19.36 19.14 18.93 18.73 18.53 18.34 18.16 17.82 17.52 17.25 17.0 16.78 16.34 16.04 15.73 15.69 15.82 16.04 16.31 16.61 16.92 17.15 17.28 17.36 17.4 17.42 17.43 17.44 17.44 17.44 17.44;
        39.49 36.24 32.79 29.66 28.19 26.8 26.24 25.68 25.1 24.51 23.93 23.64 23.34 23.05 22.77 22.48 22.21 22.0 21.61 21.33 21.04 20.77 20.5 20.28 19.96 19.66 19.38 19.12 18.88 18.66 18.46 18.12 17.84 17.63 17.47 17.34 17.14 17.05 17.05 17.16 17.31 17.48 17.65 17.83 18.0 18.13 18.21 18.25 18.27 18.29 18.29 18.3 18.3 18.3 18.3;
        37.12 34.26 31.29 28.67 27.48 26.41 25.98 25.52 24.98 24.37 23.74 23.44 23.13 22.84 22.54 22.25 21.96 21.7 21.56 21.32 21.09 20.86 20.63 20.16 19.96 19.77 19.58 19.41 19.24 19.09 18.95 18.68 18.45 18.24 18.06 17.91 17.58 17.33 17.0 16.8 16.69 16.66 16.67 16.71 16.77 16.82 16.85 16.87 16.88 16.88 16.89 16.89 16.89 16.89 16.89;
        35.53 32.93 30.29 28.02 27.02 26.15 25.81 25.42 24.9 24.27 23.62 23.31 23.0 22.7 22.4 22.1 21.8 21.51 21.53 21.32 21.12 20.92 20.72 20.09 19.96 19.84 19.71 19.59 19.47 19.36 19.25 19.03 18.82 18.62 18.44 18.26 17.86 17.51 16.96 16.57 16.29 16.12 16.01 15.95 15.94 15.93 15.93 15.93 15.93 15.93 15.92 15.92 15.92 15.92 15.92;
        32.62 30.52 28.47 26.66 25.79 24.93 24.59 24.25 23.89 23.52 23.15 22.98 22.8 22.62 22.44 22.27 22.08 21.91 21.76 21.6 21.45 21.29 21.14 20.94 20.79 20.65 20.51 20.38 20.24 20.11 19.98 19.73 19.48 19.24 19.02 18.81 18.35 17.97 17.45 17.14 16.92 16.81 16.75 16.71 16.7 16.7 16.69 16.69 16.69 16.69 16.69 16.69 16.69 16.69 16.69;
        31.06 29.22 27.47 25.92 25.18 24.47 24.19 23.91 23.62 23.33 23.02 22.87 22.73 22.58 22.44 22.3 22.15 22.01 21.91 21.78 21.65 21.52 21.4 21.22 21.11 21.0 20.89 20.78 20.67 20.56 20.45 20.23 20.02 19.81 19.62 19.42 18.97 18.58 17.94 17.46 17.13 16.98 16.9 16.85 16.83 16.81 16.8 16.8 16.8 16.8 16.8 16.79 16.79 16.79 16.79;
        30.1 28.49 26.98 25.6 24.96 24.34 24.1 23.86 23.6 23.33 23.07 22.94 22.81 22.68 22.56 22.43 22.31 22.17 22.09 21.98 21.87 21.76 21.65 21.49 21.39 21.3 21.2 21.1 21.01 20.91 20.82 20.64 20.44 20.25 20.07 19.89 19.47 19.07 18.39 17.83 17.46 17.27 17.17 17.11 17.08 17.06 17.05 17.04 17.04 17.03 17.03 17.03 17.03 17.03 17.03;
        29.57 28.09 26.72 25.46 24.88 24.32 24.1 23.88 23.65 23.4 23.16 23.04 22.92 22.8 22.68 22.57 22.45 22.33 22.26 22.16 22.06 21.96 21.86 21.7 21.62 21.53 21.44 21.36 21.27 21.18 21.1 20.93 20.76 20.58 20.41 20.24 19.84 19.45 18.77 18.19 17.79 17.59 17.48 17.42 17.38 17.35 17.33 17.32 17.32 17.32 17.31 17.31 17.31 17.31 17.31;
        29.55 28.12 26.8 25.6 25.05 24.53 24.33 24.12 23.9 23.66 23.43 23.32 23.21 23.1 22.99 22.88 22.77 22.66 22.6 22.5 22.41 22.31 22.22 22.07 21.99 21.91 21.83 21.76 21.68 21.6 21.52 21.36 21.2 21.03 20.87 20.71 20.33 19.96 19.29 18.7 18.29 18.08 17.96 17.89 17.85 17.82 17.81 17.8 17.79 17.79 17.79 17.78 17.78 17.78 17.78;
        29.66 28.29 27.02 25.9 25.38 24.89 24.7 24.51 24.31 24.09 23.86 23.75 23.65 23.55 23.44 23.34 23.24 23.13 23.08 22.99 22.9 22.82 22.73 22.58 22.51 22.44 22.36 22.29 22.21 22.14 22.07 21.92 21.76 21.61 21.46 21.32 20.96 20.6 19.96 19.37 18.96 18.74 18.62 18.55 18.51 18.48 18.47 18.46 18.45 18.45 18.44 18.44 18.44 18.44 18.44;
        29.5 28.2 27.02 25.97 25.49 25.04 24.86 24.68 24.48 24.27 24.06 23.96 23.86 23.77 23.67 23.58 23.48 23.38 23.33 23.24 23.17 23.08 23.0 22.86 22.8 22.73 22.66 22.6 22.53 22.45 22.38 22.24 22.1 21.96 21.82 21.68 21.34 21.01 20.4 19.84 19.44 19.23 19.1 19.03 18.99 18.96 18.94 18.93 18.92 18.92 18.91 18.91 18.91 18.91 18.91
    ]
    vols ./= 100
    strikes = [40, 50, 60, 70, 75, 80, 82, 84, 86, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 114, 116, 118, 120, 125, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300] ./ 100

    spot = 5751.13
    uForwards = @. exp(-logdfDiscountCurve(ts) - divyieldCurve(ts) * ts)
    forecastCurve(t) = exp(logdfForecastCurve(t) + divyieldCurve(t) * t) #curve of discount factors
    discountCurve(t) = 1.0

    uPrices, isCall, uWeights = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ts, uForwards, strikes, vols, vegaFloor=1e-2)
    uPrices, isCall, uWeights = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ts, uForwards, strikes, vols, vegaFloor=10.0)
    uWeights = ones(length(ts), length(strikes))
    params, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, uWeights)
    params, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, uWeights, reduction=AQFED.VolatilityModels.SigmaKappaReduction(0.0, 0.75))


end
@testset "Heston08October2024" begin
    #forecast
    ts = [
        0.0273972602739726,
        0.10410958904109589,
        0.2,
        0.44931506849315067,
        0.6986301369863014,
        0.947945205479452,
        1.1972602739726028,
        1.6931506849315068,
        2.1945205479452055,
        2.6904109589041094,
        3.191780821917808,
        4.189041095890411,
        5.205479452054795,
        6.2027397260273975,
        7.2,
        8.197260273972603,
        9.194520547945206,
    ]
    dfs = [0.998656, 0.994938, 0.9905, 0.979911, 0.970367, 0.961542, 0.953184, 0.937349, 0.921779, 0.906637, 0.891596, 0.862471, 0.833291, 0.805002, 0.777273, 0.749989, 0.723404]
    logdfForecastCurve = PPInterpolation.CubicSplineNatural(ts, log.(dfs))

    #sofrcurve
    ts = [
        0.0027397260273972603,
        0.019178082191780823,
        0.038356164383561646,
        0.057534246575342465,
        0.08493150684931507,
        0.16986301369863013,
        0.25205479452054796,
        0.3424657534246575,
        0.4191780821917808,
        0.4986301369863014,
        0.5808219178082191,
        0.6684931506849315,
        0.7479452054794521,
        0.8328767123287671,
        0.9178082191780822,
        1.0,
        1.4986301369863013,
        2.0,
        3.0,
        4.008219178082192,
        5.005479452054795,
        6.002739726027397,
        7.002739726027397,
        8.005479452054795,
        9.013698630136986,
        10.01095890410959,
        12.008219178082191,
        15.016438356164384,
        20.02191780821918,
        25.016438356164382,
        30.019178082191782,
        40.02739726027397,
        50.032876712328765,
    ]
    dfs = [
        0.9998655446052758,
        0.9990592658141242,
        0.9981190247355057,
        0.9971796270019692,
        0.995860536384228,
        0.9918793842928559,
        0.9882065552108679,
        0.9843057953051059,
        0.9811316126737677,
        0.977982293434981,
        0.9748020683896175,
        0.971473610055389,
        0.9685778806837968,
        0.9655473398205242,
        0.9625539466969002,
        0.959724969259051,
        0.9433435353852817,
        0.92757331917794,
        0.89692187664775,
        0.8671686254532832,
        0.8383494708781539,
        0.8098040487339867,
        0.7817611882623016,
        0.7541291703258558,
        0.7269915238856094,
        0.7007504513368383,
        0.6498808462271521,
        0.5794733401114949,
        0.48281081406533616,
        0.4118876765643804,
        0.35722809277675743,
        0.28545316076925675,
        0.24132218426777505,
    ]
    logdfDiscountCurve = PPInterpolation.CubicSplineNatural(ts, log.(dfs))

    #divcurve
    ts = [
        0.0273972602739726,
        0.10410958904109589,
        0.2,
        0.44931506849315067,
        0.6986301369863014,
        0.947945205479452,
        1.1972602739726028,
        1.6931506849315068,
        2.1945205479452055,
        2.6904109589041094,
        3.191780821917808,
        4.189041095890411,
        5.205479452054795,
        6.2027397260273975,
        7.2,
        8.197260273972603,
        9.194520547945206,
    ]
    mids = [0.003913, 0.0039242, 0.0056579, 0.0047029, 0.0051079, 0.0051398, 0.0051334, 0.0048594, 0.0048837, 0.0047432, 0.0046435, 0.004394, 0.0040593, 0.0038011, 0.0036346, 0.0034687, 0.003293]
    divyieldCurve = PPInterpolation.CubicSplineNatural(ts, mids)

    #vol
    ts = [
        0.10410958904109589,
        0.2,
        0.4767123287671233,
        0.6986301369863014,
        0.947945205479452,
        1.1972602739726028,
        1.6931506849315068,
        2.1945205479452055,
        2.6904109589041094,
        3.191780821917808,
        4.189041095890411,
        5.205479452054795,
        6.2027397260273975,
        7.2,
        8.197260273972603,
        9.194520547945206,
    ]
    strikes = [
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.905,
        0.91,
        0.915,
        0.92,
        0.925,
        0.93,
        0.935,
        0.94,
        0.945,
        0.95,
        0.955,
        0.96,
        0.965,
        0.97,
        0.975,
        0.98,
        0.985,
        0.99,
        0.995,
        1.0,
        1.005,
        1.01,
        1.015,
        1.02,
        1.025,
        1.03,
        1.035,
        1.04,
        1.045,
        1.05,
        1.055,
        1.06,
        1.065,
        1.07,
        1.075,
        1.08,
        1.085,
        1.09,
        1.095,
        1.1,
        1.15,
        1.2,
        1.25,
        1.3,
        1.35,
        1.4,
        1.45,
        1.5,
    ]
    vols =
        [
            76.6045 70.59 64.2418 57.8388 51.4678 44.9348 38.5177 32.345 26.7839 26.2666 25.7528 25.2422 24.7346 24.2299 23.7281 23.229 22.7323 22.2373 21.7429 21.2468 20.7465 20.2402 19.7273 19.2088 18.6866 18.163 17.6401 17.1204 16.6064 16.1007 15.6066 15.128 14.67 14.238 13.8381 13.4758 13.1556 12.8808 12.6543 12.4797 12.3598 12.2946 12.2807 12.3124 12.3833 12.487 12.618 12.7719 12.9448 15.2002 17.7193 20.1723 22.4814 24.6466 26.6894 28.6293 30.4794;
            64.1937 58.6376 53.2796 48.025 42.8406 37.7493 32.8381 28.3442 24.134 23.7293 23.3264 22.9253 22.5261 22.1293 21.7348 21.3425 20.9517 20.5613 20.1701 19.776 19.3776 18.9741 18.5661 18.1548 17.742 17.3293 16.9185 16.5109 16.1083 15.712 15.3237 14.9453 14.5791 14.2276 13.8938 13.5804 13.2894 13.0217 12.778 12.5596 12.3686 12.2068 12.0741 11.9682 11.8858 11.8238 11.782 11.7631 11.7701 12.8875 14.6584 16.5232 18.3383 20.054 21.6666 23.1914 24.6434;
            50.2098 46.0004 41.9003 38.0821 34.4658 31.0808 27.9027 24.8961 21.9817 21.69 21.398 21.1053 20.8119 20.5177 20.2224 19.9263 19.6294 19.3318 19.0336 18.7353 18.4372 18.1398 17.8434 17.5484 17.255 16.9633 16.6737 16.3866 16.1025 15.8219 15.5452 15.2729 15.0055 14.7436 14.4883 14.2403 14.0006 13.7701 13.5497 13.3407 13.1444 12.9613 12.7906 12.6299 12.4772 12.3315 12.1935 12.0645 11.9459 11.4195 11.7722 12.683 13.7568 14.8547 15.9252 16.9512 17.933;
            43.6288 40.0654 36.7041 33.5602 30.6967 28.0537 25.6223 23.3567 21.16 20.9381 20.7149 20.4902 20.264 20.0366 19.8079 19.578 19.3465 19.1132 18.8777 18.64 18.4007 18.1604 17.92 17.6798 17.44 17.2004 16.961 16.7219 16.4836 16.2466 16.0114 15.7784 15.5479 15.3198 15.0943 14.8719 14.6532 14.4393 14.2311 14.029 13.8331 13.6431 13.4588 13.2803 13.108 12.9424 12.7845 12.6349 12.4939 11.4978 11.2209 11.5075 12.0742 12.7447 13.4517 14.1715 14.8903;
            39.5856 36.5499 33.7399 31.1555 28.8138 26.6755 24.6773 22.7909 20.947 20.7582 20.5687 20.3787 20.1881 19.9966 19.8041 19.6102 19.4148 19.2173 19.0179 18.8165 18.6138 18.4104 18.2068 18.0033 17.7998 17.5961 17.3919 17.1873 16.9825 16.7781 16.5744 16.3716 16.1697 15.9682 15.7667 15.565 15.3635 15.163 14.9648 14.7703 14.5799 14.3939 14.2123 14.0351 13.8625 13.6946 13.5316 13.3736 13.2207 11.999 11.3752 11.244 11.5256 11.9195 12.3931 12.9126 13.4603;
            36.9391 34.281 31.8415 29.6147 27.5898 25.7329 23.9884 22.3451 20.7346 20.5704 20.4053 20.2394 20.0729 19.9058 19.7382 19.5699 19.401 19.2311 19.0602 18.8883 18.7156 18.5425 18.3691 18.1957 18.0222 17.8484 17.6743 17.4999 17.3253 17.1508 16.9767 16.8031 16.6298 16.4565 16.2827 16.108 15.9323 15.7565 15.5817 15.4088 15.2384 15.0707 14.9055 14.7428 14.5825 14.4249 14.2701 14.1184 13.9701 12.6927 11.8415 11.4397 11.4406 11.7352 12.0505 12.4019 12.7906;
            33.6397 31.4781 29.5154 27.7548 26.137 24.5766 23.1837 21.8179 20.4796 20.344 20.2084 20.0729 19.9374 19.8018 19.6659 19.5295 19.3925 19.2547 19.1163 18.9773 18.838 18.6988 18.5598 18.4211 18.2826 18.1441 18.0056 17.867 17.7283 17.5898 17.4516 17.3137 17.1761 17.0387 16.9011 16.7631 16.6245 16.4856 16.3468 16.2086 16.0714 15.9351 15.7998 15.6656 15.5325 15.4006 15.2702 15.1411 15.0136 13.8103 12.7718 12.1131 11.6693 11.505 11.5409 11.6949 11.9168;
            31.2369 29.4357 27.7795 26.2737 24.8539 23.5574 22.5286 21.4173 20.278 20.1811 20.0821 19.9775 19.8648 19.7442 19.6192 19.4943 19.3751 19.2641 19.1618 19.0652 18.9696 18.8717 18.7688 18.6617 18.5506 18.437 18.3219 18.2057 18.0898 17.9735 17.8576 17.7416 17.6253 17.5086 17.391 17.2728 17.1536 17.0341 16.9148 16.7963 16.6793 16.5636 16.4492 16.3357 16.2229 16.1106 15.9985 15.8868 15.7752 14.6849 13.7033 12.7251 12.3162 11.9161 11.6199 11.4919 11.487;
            30.4777 28.9108 27.4503 26.0792 24.7839 23.554 22.3812 21.2588 20.1819 20.0766 19.9717 19.8671 19.763 19.6593 19.556 19.4532 19.3507 19.2486 19.1469 19.0456 18.9447 18.8442 18.7441 18.6444 18.545 18.4461 18.3476 18.2494 18.1517 18.0543 17.9573 17.8607 17.7646 17.6688 17.5734 17.4784 17.3838 17.2896 17.1958 17.1024 17.0095 16.9169 16.8248 16.7331 16.6418 16.5509 16.4605 16.3705 16.281 15.4114 14.5945 13.8415 13.1673 12.5885 12.1198 11.7684 11.5312;
            29.1933 27.6658 26.2467 25.0002 23.9279 22.9794 22.1153 21.2528 20.4 20.3254 20.2466 20.1636 20.0766 19.9857 19.8925 19.8006 19.7098 19.6208 19.5341 19.4485 19.362 19.2739 19.186 19.0968 19.0066 18.9169 18.8282 18.7395 18.6508 18.5634 18.4751 18.3858 18.2963 18.2066 18.1159 18.0245 17.9335 17.8415 17.7489 17.6567 17.565 17.4733 17.3819 17.2914 17.201 17.1108 17.0211 16.9316 16.8422 15.9661 15.1162 14.3682 13.6982 13.0907 12.5727 12.1685 11.8491;
            28.0574 26.9321 25.8858 24.9058 23.9822 23.1073 22.2746 21.4791 20.7163 20.6417 20.5673 20.4933 20.4195 20.346 20.2728 20.1999 20.1272 20.0548 19.9827 19.9108 19.8392 19.7678 19.6967 19.6259 19.5553 19.4849 19.4148 19.3449 19.2753 19.2059 19.1367 19.0678 18.9991 18.9306 18.8624 18.7943 18.7265 18.659 18.5916 18.5245 18.4575 18.3908 18.3243 18.258 18.1919 18.126 18.0603 17.9949 17.9296 17.2875 16.6642 16.0586 15.4703 14.8992 14.3455 13.8103 13.2952;
            27.5021 26.5522 25.6691 24.842 24.0626 23.3241 22.621 21.9489 21.3038 21.2407 21.1778 21.1151 21.0527 20.9905 20.9285 20.8667 20.8052 20.7438 20.6827 20.6218 20.5611 20.5006 20.4403 20.3802 20.3203 20.2606 20.2011 20.1418 20.0827 20.0238 19.965 19.9064 19.848 19.7898 19.7318 19.6739 19.6163 19.5587 19.5014 19.4442 19.3872 19.3303 19.2736 19.2171 19.1607 19.1045 19.0484 18.9925 18.9367 18.3867 17.8498 17.3244 16.8091 16.3025 15.8034 15.3107 14.8231;
            27.3812 26.5356 25.7508 25.0171 24.327 23.6743 23.0543 22.4628 21.8966 21.8413 21.7861 21.7312 21.6765 21.6221 21.5678 21.5137 21.4599 21.4062 21.3528 21.2995 21.2465 21.1936 21.1409 21.0884 21.0362 20.984 20.9321 20.8804 20.8288 20.7774 20.7262 20.6752 20.6244 20.5737 20.5232 20.4728 20.4226 20.3726 20.3227 20.273 20.2235 20.1741 20.1249 20.0758 20.0269 19.9781 19.9295 19.881 19.8326 19.3569 18.8942 18.4432 18.0028 17.5719 17.1496 16.735 16.3273;
            27.4044 26.6413 25.9341 25.2738 24.6538 24.0684 23.5132 22.9845 22.4794 22.43 22.3809 22.332 22.2833 22.2348 22.1864 22.1383 22.0904 22.0426 21.9951 21.9477 21.9005 21.8535 21.8067 21.7601 21.7136 21.6673 21.6212 21.5753 21.5295 21.4839 21.4385 21.3932 21.3481 21.3032 21.2584 21.2137 21.1693 21.1249 21.0808 21.0368 20.9929 20.9492 20.9056 20.8622 20.8189 20.7758 20.7328 20.6899 20.6472 20.2274 19.8202 19.4243 19.0388 18.6628 18.2955 17.9362 17.5842;
            27.4195 26.7073 26.0479 25.4329 24.8559 24.3117 23.7961 23.3057 22.8377 22.792 22.7465 22.7012 22.6561 22.6112 22.5665 22.522 22.4776 22.4335 22.3895 22.3457 22.3021 22.2586 22.2153 22.1722 22.1293 22.0865 22.0439 22.0014 21.9592 21.917 21.8751 21.8333 21.7916 21.7501 21.7088 21.6676 21.6266 21.5857 21.5449 21.5043 21.4639 21.4235 21.3834 21.3433 21.3034 21.2637 21.2241 21.1846 21.1452 20.7588 20.3844 20.021 19.6677 19.3237 18.9883 18.6608 18.3405;
            27.5101 26.8389 26.218 25.6394 25.097 24.5859 24.102 23.6423 23.2038 23.1611 23.1185 23.0761 23.0339 22.9919 22.9501 22.9084 22.8669 22.8256 22.7845 22.7435 22.7027 22.6621 22.6216 22.5813 22.5411 22.5012 22.4613 22.4217 22.3822 22.3428 22.3036 22.2646 22.2257 22.1869 22.1483 22.1098 22.0715 22.0333 21.9953 21.9574 21.9197 21.882 21.8446 21.8072 21.77 21.7329 21.696 21.6591 21.6224 21.2623 20.9138 20.576 20.2479 19.9289 19.6183 19.3153 19.0195
        ] ./ 100

    spot = 5751.13
    uForwards = @. exp(-logdfForecastCurve(ts) - divyieldCurve(ts) * ts)
    uPrices, isCall, uWeights = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ts, uForwards, strikes, vols, vegaFloor=1e2)
    paramsf0, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, uWeights)

    uPrices, isCall, uWeights = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ts, uForwards, strikes, vols, vegaFloor=1e-2)

    reduction = AQFED.VolatilityModels.makeVarianceSwapReduction(ts, uForwards, strikes, vols, isLinearExtrapolation=true)
    paramsf1, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, uWeights, reduction=reduction)

    paramsf2, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, uWeights)


    reduction = AQFED.VolatilityModels.makeDoubleHestonVarianceSwapReduction(ts, uForwards, strikes, vols, isLinearExtrapolation=true)
    paramsd, rmse = AQFED.VolatilityModels.calibrateDoubleHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, uWeights, reduction=reduction)

    #if kappa is not limited, it will go to zero and theta will explode on this example.

    params, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, uWeights)
    params, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, uWeights, reduction=AQFED.VolatilityModels.SigmaKappaReduction(0.4, 0.0))

    #plot(strikes,vols',seriestype=:scatter,ms=2,ma=0.5,markerstrokewidth=0)
    # plot!(strikes, vols' + volError')

    #todo: calibrate with kappa or volvol exogenous. exclude opt with small vega from calibration/rmse.
    forecastCurve(t) = exp(logdfForecastCurve(t) + divyieldCurve(t) * t) #curve of discount factors
    discountCurve(t) = 1.0

    uPrices, isCall, uWeights = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ts, uForwards, strikes, vols, vegaFloor=1e-2)
    uPrices, isCall, uWeights = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ts, uForwards, strikes, vols, vegaFloor=10.0)
    uWeights = ones(length(ts), length(strikes))
    params, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, uWeights)
    params, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, uWeights, reduction=AQFED.VolatilityModels.SigmaKappaReduction(0.0, 0.75))

    methods = ["Andersen-Lake", "Joshi-Yang", "Cos-128", "Cos", "Cos-Junike", "Flinn", "Flinn-Transformed", "Swift"]
    results = Dict()
    for method ∈ methods
        elapsedTime = @elapsed params, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, uWeights, method=method)
        @printf("%s %.4f %.4f %.4f %.4f %.4f %.4f %.1f\n", method, params.v0, params.κ, params.θ, params.ρ, params.σ, rmse, elapsedTime)
        results[method] = [params.v0, params.κ, params.θ, params.ρ, params.σ, rmse, elapsedTime]
    end
    for method ∈ methods
        result = results[method]
        @printf("%s %.4f %.4f %.4f %.4f %.4f %.4f %.1f\n", method, result[1], result[2], result[3], result[4], result[5], result[6])
    end
    #Cos-256,16 fails on vegaFloor=10 while Cos-128-12 not so accurate
    #Cos-256,16 fails even more strongly on vegaFloor=10
    #Cos-128 really bad on uWeights=one. Cos-Junike slower than AL.
    #Joshi-64 surprisingly good. AdaptiveFlinn has bug. Flinn not accurate on vega=1e-2.
    payoff = AQFED.FDM.VanillaFDM2DPayoff(1.0, [1.0, 1.5], [true, true], [zeros(100, 50), zeros(100, 50)])
    refPrice = CharFuncPricing.priceEuropean(ALCharFuncPricer(DefaultCharFunc(params), n=128), true, 1.5, 1.0 / forecastCurve(1.0), 1.0, 1.0)
    priceGrid = AQFED.FDM.priceFDM2D(payoff, params, 1.0, forecastCurve, discountCurve, 100, method="EU", sDisc="Exp", exponentialFitting="None")
    spl = Spline2D(priceGrid[1], priceGrid[2], priceGrid[3][2])
    Dierckx.evaluate(spl, 1.0, params.v0)
    priceGrid = AQFED.FDM.priceFDM2D(payoff, params, 1.0, forecastCurve, discountCurve, 100, method="RKG2", sDisc="Exp", exponentialFitting="Partial Exponential Fitting")
    spl = Spline2D(priceGrid[1], priceGrid[2], priceGrid[3][2])
    Dierckx.evaluate(spl, 1.0, params.v0)
    #RKG is much more accurate and faster
end
