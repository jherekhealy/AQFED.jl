# Unit tests of the volatility calibrations. The studies behind the book figures are in book/ch07_*.jl
# (the former content of this file is kept in book/legacy/vol_calibration_study.jl).
using AQFED, Test
using StatsBase, Printf
import LeastSquaresOptim
import LeastSquaresOptim: LeastSquaresProblem, LevenbergMarquardt
import Roots
import Roots: find_zero
import Random: rand!
using AQFED.Black
using AQFED.Collocation
import Polynomials: coeffs
using AQFED.Math,AQFED.PDDE,AQFED.VolatilityModels
using PPInterpolation
using CharFuncPricing
using LinearAlgebra #norm
import FiniteDifferences
include("smile_data.jl")



@testset "deltavolraw" begin
    #probability density from the second derivative of the call price of the smile pol (uses the current forward and tte)
    pdfd(pol, z) = FiniteDifferences.central_fdm(5, 2)(y -> AQFED.Black.blackScholesFormula(true, y, forward, pol(y)^2 * tte, 1.0, 1.0), z)
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
    prices, wv = Collocation.weightedPrices(true, k, vols, ones(length(vols)), forward, 1.0, tte)
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

    prices, wv = Collocation.weightedPrices(true, k, vols, ones(length(vols)), forward, 1.0, tte)
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



    y = log.(k ./ forward)
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

    spot = 0.9759
    tte = 1.0
    dfFor = 1 / (1 + 2.686 / 100 * tte)
    dfDom = 1 / (1 + 3.83 / 100 * tte) #USD
    forward = spot * dfFor / dfDom

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
end

#lognormal mixtures on case II of Jaeckel: the calibrated mixtures are martingales and fit the smile
@testset "LognormalMixtureJaeckelCaseII" begin
    d = loadSmile("jaeckel_case_ii")
    prices, wv = Collocation.weightedPrices(true, d.strikes, d.vols, d.weights, d.forward, 1.0, d.tte, vegaFloor=1e-7)
    w = d.weights ./ sqrt(sum(d.weights .^ 2))
    vol(kernel, k) = Black.impliedVolatility(k >= d.forward, VolatilityModels.priceEuropean(kernel, k >= d.forward, k), d.forward, k, d.tte, 1.0)
    rmse(kernel) = StatsBase.L2dist(w .* vol.(Ref(kernel), d.strikes), w .* d.vols)
    mean(kernel) = sum(kernel.α .* kernel.x)
    kernelA6 = VolatilityModels.calibrateLognormalMixture(d.tte, d.forward, d.strikes, prices, wv, useVol=false, size=6)
    @test isapprox(mean(kernelA6), d.forward, atol=1e-12)
    kernelB6 = VolatilityModels.calibrateLognormalMixtureSoftmax(d.tte, d.forward, d.strikes, prices, wv, useVol=false, size=6)
    @test isapprox(mean(kernelB6), d.forward, atol=1e-12)
    @test rmse(kernelB6) < 0.005
    kernelC51, _ = VolatilityModels.calibrateLognormalMixtureGrid(d.tte, d.forward, d.strikes, prices, wv, size=51)
    @test isapprox(mean(kernelC51), d.forward, atol=1e-8)
    @test isapprox(sum(kernelC51.α), 1.0, atol=1e-12)
    @test rmse(kernelC51) < 0.001
    λAtm = d.vols[argmin(abs.(d.strikes .- d.forward))] * sqrt(d.tte)
    kernelS, _ = VolatilityModels.calibrateLognormalMixtureGrid(d.tte, d.forward, d.strikes, prices, wv, size=101, bandwidth=0.25λAtm, smoothing=1e-8)
    @test rmse(kernelS) < 0.001
    @test all(kernelS.α .>= 0)
    #volatility-adaptive grid: 51 kernels without smoothing, and automatic bandwidth with automatic smoothing
    kernelC51A, _ = VolatilityModels.calibrateLognormalMixtureGrid(d.tte, d.forward, d.strikes, prices, wv, size=51, adaptive=true, overlap=4 / 3)
    @test length(kernelC51A.x) == 51
    @test isapprox(mean(kernelC51A), d.forward, rtol=1e-12)
    @test rmse(kernelC51A) < 0.001
    kernelCSA, _ = VolatilityModels.calibrateLognormalMixtureGrid(d.tte, d.forward, d.strikes, prices, wv, adaptive=true, smoothing=:auto)
    @test isapprox(mean(kernelCSA), d.forward, rtol=1e-12)
    @test rmse(kernelCSA) < 0.001
    #wider kernels in the wings, where the implied volatility is larger
    @test kernelCSA.σ[1] > 2 * minimum(kernelCSA.σ)
end
