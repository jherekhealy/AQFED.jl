import AQFED.Black: blackScholesFormula, blackScholesVega, impliedVolatility
import AQFED.TermStructure: calibrateSVISection, SVISection, varianceByLogmoneyness
import AQFED.Collocation: weightedPrices, makeIsotonicCollocation, Polynomial, BSplineCollocation, makeBSplineCollocation
import AQFED.Collocation: priceEuropean as priceEuropeanCollocation
import Polynomials: AbstractPolynomial
import AQFED.PDDE: QuadraticLVG, calibrateQuadraticLVG, Quadratic, EQuadraticLVG, calibrateEQuadraticLVG
import AQFED.PDDE: priceEuropean as priceEuropeanPDDE
import AQFED.Math: normcdf, norminv, inv, ClosedTransformation, FitResult, MQMinTransformation, Quadrature, GaussKronrod
using GaussNewton
using PPInterpolation
export ExpPolynomialSmile, SplineSmile, calibrateSmile, convertQuotesToDeltaVols



abstract type SmileFunction end
struct SplineSmile <: SmileFunction
    pp::PPInterpolation.PP
    forward::Float64
end
function makeTotalVarianceFunction(smile::SmileFunction, forward, tte)
    return y -> smile(forward * exp(y))^2 * tte
end


function calibrateSmile(f::SmileFunction, ys::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; kwargs...) where {T}
    calibrateSmile(f, zeros(LogmoneynessAxisTransformation, length(ys)), ys, vols, forward, tte; kwargs...)
end

function calibrateSmile(::Type{SplineSmile}, axisTransforms::Vector{U}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; isFlat=false) where {T,U<:AxisTransformation}
    ys = map((trans, x, vol) -> convertToLogmoneyness(trans, x, vol), axisTransforms, xs, vols)
    pp = if !isFlat
        PPInterpolation.CubicSplineNatural(ys, vols .^ 2)
    else
        PPInterpolation.makeCubicPP(ys, vols .^ 2, PPInterpolation.FIRST_DERIVATIVE, 0.0, PPInterpolation.FIRST_DERIVATIVE, 0.0, C2())
    end
    return SplineSmile(pp, forward)
end
(spl::SplineSmile)(strike) = begin
    x = log(strike / spl.forward)
    # if x < spl.pp.x[1]
    # 	sqrt(spl.pp.a[1])
    # elseif x > spl.pp.x[end]
    # 	sqrt(spl.pp.a[end])
    # else
    sqrt(abs(PPInterpolation.evaluate(spl.pp, x)))
    # end
end

struct SVISmile <: SmileFunction
    svi::SVISection
end
function calibrateSmile(::Type{SVISmile}, axisTransforms::Vector{U}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; aMin=0.0, sMin=one(T) / 10000, noarbFactor=2 * one(T)) where {T,U<:AxisTransformation}
    ys = map((trans, x, vol) -> convertToLogmoneyness(trans, x, vol), axisTransforms, xs, vols)
    #TODO calibration including conversion
    svi0, rmsesvi = calibrateSVISection(tte, forward, ys, vols, ones(length(vols)), aMin=aMin, sMin=sMin, noarbFactor=noarbFactor)
    return SVISmile(svi0)
end
(spl::SVISmile)(strike) = sqrt(varianceByLogmoneyness(spl.svi, log(strike / spl.svi.f)))

struct XSSVISmile <: SmileFunction
    svi::XSSVISection
end
function calibrateSmile(::Type{XSSVISmile}, axisTransforms::Vector{U}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T,U<:AxisTransformation}
    svi0 = calibrateXSSVISection(tte, forward, axisTransforms, xs, vols, ones(length(vols)))
    return XSSVISmile(svi0)
end
(spl::XSSVISmile)(strike) = sqrt(varianceByLogmoneyness(spl.svi, log(strike / spl.svi.f)))

struct SABRSmile <: SmileFunction
    sabr::SABRSection
end
function calibrateSmile(::Type{SABRSmile}, axisTransforms::Vector{U}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T,U<:AxisTransformation}
    ys = map((trans, x, vol) -> convertToLogmoneyness(trans, x, vol), axisTransforms, xs, vols)
    guess = initialGuessBlackATM(forward, tte, 1.0, ys, vols)
    sabr = calibrateSABRSectionFromGuess(tte, forward, axisTransforms, xs, vols, ones(length(vols)), guess)
    return SABRSmile(sabr)
end
(spl::SABRSmile)(strike) = sqrt(varianceByLogmoneyness(spl.sabr, log(strike / spl.sabr.f)))

struct SABRATMSmile <: SmileFunction
    sabr::SABRSection
end
function calibrateSmile(::Type{SABRATMSmile}, axisTransforms::Vector{U}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T,U<:AxisTransformation}
    sabr = calibrateSABRSectionATM(tte, forward, axisTransforms, xs, vols, ones(length(vols)), 1.0)
    return SABRSmile(sabr)
end
(spl::SABRATMSmile)(strike) = sqrt(varianceByLogmoneyness(spl.sabr, log(strike / spl.sabr.f)))

struct LVGSmile <: SmileFunction
    lvg::QuadraticLVG
end
function calibrateSmile(::Type{LVGSmile}, axisTransforms::Vector{U}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; L=1e-4) where {T,U<:AxisTransformation}
    strikes = map((trans, x, vol) -> forward * exp(convertToLogmoneyness(trans, x, vol)), axisTransforms, xs, vols)
    #TODO calibration including conversion
    prices, wv = weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor=1e-8)
    lvg = calibrateQuadraticLVG(tte, forward, strikes, prices, wv, useVol=false, model=Quadratic(), location="Mid-XX", size=0, L=L, U=max(strikes[end] * 1.01, forward * exp(12 * vols[5] * sqrt(tte))))
    return LVGSmile(lvg)
end
(spl::LVGSmile)(strike) = impliedVolatility(
    strike > spl.lvg.forward,
    priceEuropeanPDDE(spl.lvg, strike > spl.lvg.forward, strike),
    spl.lvg.forward,
    strike,
    spl.lvg.tte,
    1.0)


struct ELVGSmile <: SmileFunction
    lvg::EQuadraticLVG
end
function calibrateSmile(::Type{ELVGSmile}, axisTransforms::Vector{V}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; L=1e-4, U=max(strikes[end] * 1.01, forward * exp(8 * vols[5] * sqrt(tte)))) where {T,V<:AxisTransformation}
    strikes = map((trans, x, vol) -> forward * exp(convertToLogmoneyness(trans, x, vol)), axisTransforms, xs, vols)
    #TODO calibration including conversion
    prices, wv = weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor=1e-8)
    lvg = calibrateEQuadraticLVG(tte, forward, strikes, prices, wv, useVol=false, model="Quadratic", nRefine=5, location="Mid-XX", size=0, L=L, U=U)
    return ELVGSmile(lvg)
end
(spl::ELVGSmile)(strike) = impliedVolatility(
    strike > spl.lvg.forward,
    priceEuropeanPDDE(spl.lvg, strike > spl.lvg.forward, strike),
    spl.lvg.forward,
    strike,
    spl.lvg.tte,
    1.0)

struct QuinticCollocationSmile <: SmileFunction
    collocation::AbstractPolynomial
    forward::Float64
    tte::Float64
end
function calibrateSmile(::Type{QuinticCollocationSmile}, axisTransforms::Vector{U}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; minSlope=1e-4) where {T,U<:AxisTransformation}
    strikes = map((trans, x, vol) -> forward * exp(convertToLogmoneyness(trans, x, vol)), axisTransforms, xs, vols)
    #TODO calibration including conversion
    prices, wv = weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor=1e-10)
    isoc, m = makeIsotonicCollocation(strikes, prices, wv, tte, forward, 1.0, deg=5, degGuess=1, minSlope=minSlope)
    sol = Polynomial(isoc)
    return QuinticCollocationSmile(sol, forward, tte)
end
(spl::QuinticCollocationSmile)(strike) = impliedVolatility(
    strike > spl.forward,
    abs(priceEuropeanCollocation(spl.collocation, strike > spl.forward, strike, spl.forward, 1.0)),
    spl.forward,
    strike,
    spl.tte,
    1.0)

struct BSplineCollocationSmile <: SmileFunction
    collocation::BSplineCollocation
    forward::Float64
    tte::Float64
end
function calibrateSmile(::Type{BSplineCollocationSmile}, axisTransforms::Vector{U}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T,U<:AxisTransformation}
    strikes = map((trans, x, vol) -> forward * exp(convertToLogmoneyness(trans, x, vol)), axisTransforms, xs, vols)
    #TODO calibration including conversion
    prices, wv = weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor=1e-4)
    bspl, m = makeBSplineCollocation(
        strikes,
        prices,
        wv,
        tte,
        forward,
        1.0,
        penalty=0e-2,
        size=0,
        minSlope=1e-7,
        rawFit=true, N=3,
    )
    return BSplineCollocationSmile(bspl, forward, tte)
end
(spl::BSplineCollocationSmile)(strike) = impliedVolatility(
    strike > spl.forward,
    priceEuropeanCollocation(spl.collocation, strike > spl.forward, strike, spl.forward, 1.0),
    spl.forward,
    strike,
    spl.tte,
    1.0)

struct LognormalMixtureSmile <: SmileFunction
    kernel::LognormalKernel
    forward::Float64
    tte::Float64
end
function calibrateSmile(::Type{LognormalMixtureSmile}, axisTransforms::Vector{U}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T,U<:AxisTransformation}
    strikes = map((trans, x, vol) -> forward * exp(convertToLogmoneyness(trans, x, vol)), axisTransforms, xs, vols)
    kernel3 = calibrateLognormalMixture(tte, forward, axisTransforms, xs, vols, ones(length(vols)), useVol=true, size=2)
    return LognormalMixtureSmile(kernel3, forward, tte)
end
(spl::LognormalMixtureSmile)(strike) = impliedVolatility(
    strike > spl.forward,
    priceEuropean(spl.kernel, strike > spl.forward, strike),
    spl.forward,
    strike,
    spl.tte,
    1.0)

struct LognormalMixture3Smile <: SmileFunction
    kernel::LognormalKernel
    forward::Float64
    tte::Float64
end
function calibrateSmile(::Type{LognormalMixture3Smile}, axisTransforms::Vector{U}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T,U<:AxisTransformation}
    strikes = map((trans, x, vol) -> forward * exp(convertToLogmoneyness(trans, x, vol)), axisTransforms, xs, vols)
    #TODO calibration including conversion
    prices, wv = weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor=1e-8)
    kernel3 = calibrateLognormalMixtureFX(tte, forward, strikes, vols)
    return LognormalMixtureSmile(kernel3, forward, tte)
end
(spl::LognormalMixture3Smile)(strike) = impliedVolatility(
    strike > spl.forward,
    priceEuropean(spl.kernel, strike > spl.forward, strike),
    spl.forward,
    strike,
    spl.tte,
    1.0)

struct ExpPolynomialSmile <: SmileFunction
    coeff::Vector
    atmVol::Float64
    forward::Float64
    tte::Float64
    nIter::Int
end
function calibrateSmile(::Type{ExpPolynomialSmile}, axisTransforms::Vector{U}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; nIter=256) where {T,U<:AxisTransformation}
    ys = map((trans, x, vol) -> convertToLogmoneyness(trans, x, vol), axisTransforms, xs, vols)
    midIndex = div((length(vols) + 1) , 2)
    #could compute fwd deltas from strikes, vols. and log(vol) = quartic(delta)
    simpleDeltas = if nIter == 1
        @.(normcdf(-ys / (vols[midIndex] * sqrt(tte))))
    else
        @.(normcdf(-ys / (vols * sqrt(tte))))
    end
    lvols = log.(vols)

    sum1 = length(simpleDeltas)
    sumx = sum(simpleDeltas)
    sumx2 = sum(simpleDeltas .^ 2)
    sumx3 = sum(simpleDeltas .^ 3)
    sumx4 = sum(simpleDeltas .^ 4)
    sumx5 = sum(simpleDeltas .^ 5)
    sumx6 = sum(simpleDeltas .^ 6)
    sumx7 = sum(simpleDeltas .^ 7)
    sumx8 = sum(simpleDeltas .^ 8)
    sumy = sum(lvols)
    sumxy = sum(lvols .* simpleDeltas)
    sumx2y = sum(lvols .* simpleDeltas .^ 2)
    sumx3y = sum(lvols .* simpleDeltas .^ 3)
    sumx4y = sum(lvols .* simpleDeltas .^ 4)

    m = [sum1 sumx sumx2 sumx3 sumx4;
        sumx sumx2 sumx3 sumx4 sumx5;
        sumx2 sumx3 sumx4 sumx5 sumx6;
        sumx3 sumx4 sumx5 sumx6 sumx7;
        sumx4 sumx5 sumx6 sumx7 sumx8]
    rhs = [sumy, sumxy, sumx2y, sumx3y, sumx4y]
    coeff = m \ rhs
    return ExpPolynomialSmile(coeff, vols[midIndex], forward, tte, nIter)
end

function impliedVolatilityByDelta(spl::ExpPolynomialSmile, x)
    return exp(spl.coeff[1] + spl.coeff[2] * x + spl.coeff[3] * x^2 + spl.coeff[4] * x^3 + spl.coeff[5] * x^4)
end

(spl::ExpPolynomialSmile)(strike; method=:newton, useTrace=false, out=Vector{Float64}()) = begin
    coeff = spl.coeff

    if method == :newton && spl.nIter > 1
        # v0 = spl.atmVol
        # obj = function (v0)
        #     x = normcdf((log(spl.forward / strike)) / (v0 * sqrt(spl.tte)))
        #     v = impliedVolatilityByDelta(spl, x)
        #     println("newt step ",v, " ",v0," ",x, " ",spl)
        #     return v - v0
        # end
        # #objDer = x -> ForwardDiff.derivative(obj, x)
        # # Roots.find_zero((obj,objDer), v0, Roots.Newton())
        # return Roots.find_zero(obj,
        #  (0.0, max(impliedVolatilityByDelta(spl, 0.0), impliedVolatilityByDelta(spl, 1.0))), 
        #  Roots.A42())
        
        #  v0 = spl.atmVol
        #  transformation = ExpTransformation(0.0)
        #  i=0
        #  obj = function (iv0)
        #      v0 = transformation(iv0)
        #      x = normcdf((log(spl.forward / strike)) / (v0 * sqrt(spl.tte)))
        #      v = impliedVolatilityByDelta(spl, x)
        #      i+=1; println("newt step ",v-v0," ",x, " ",spl, " ",i)
        #      return v - v0
        #  end
        # objDer = x -> ForwardDiff.derivative(obj, x)
        #  iv = Roots.find_zero((obj, objDer), inv(transformation,v0), Roots.Newton())
        #  return exp(iv)
        # return Roots.find_zero(obj, (0.0,10*v0),Roots.A42())
        # transformation = TanhTransformation(0.0,1.0)
        # i=0
        # obj = function(idelta)
        #     delta = transformation(idelta)
        #     v = impliedVolatilityByDelta(spl, delta)
        #     delta1 = normcdf((log(spl.forward / strike)) / (v * sqrt(spl.tte)))
        #     # strike1 = invcdf(delta)*(v * sqrt(spl.tte)))
        #     i+=1; println(delta, " ",delta1," ",v, i)
        #     return delta1-delta
        # end
        # objDer = x -> ForwardDiff.derivative(obj, x)
        # #idelta = Roots.find_zero(obj, inv(transformation,0.5), Roots.Secant(),atol=1e-8)
        # idelta = Roots.find_zero((obj, objDer), inv(transformation,0.5), Roots.Newton())
        # delta = transformation(idelta)
       
        # i=0
        obj = function(delta)
            v = impliedVolatilityByDelta(spl, delta)
            delta1 = normcdf((log(spl.forward / strike)) / (v * sqrt(spl.tte)))
            # strike1 = invcdf(delta)*(v * sqrt(spl.tte)))
            # i+=1; println(delta, " ",delta1," ",v," ",i)
            return delta1-delta
        end
        delta = Roots.find_zero(obj, (0.0,1.0), Roots.A42(),atol=1e-8)
       
         return impliedVolatilityByDelta(spl, delta)
        
    else #fixed point
        v = spl.atmVol
        v0 = 0.0
        i = 1
        while i <= spl.nIter && abs(v - v0) > 10 * eps(v0)
            v0 = v
            x0 = normcdf((log(spl.forward / strike)) / (v * sqrt(spl.tte)))
            v = impliedVolatilityByDelta(spl, x0)
            if useTrace
                out[i] = x0
            end
            i += 1
        end
        # println("found ",v," ",v0, " ",i, " for strike ",strike)
        v
    end
end

struct FlatSmile <: SmileFunction
    vol::Float64
end
function calibrateSmile(::Type{FlatSmile}, axisTransforms::Vector{U}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T,U<:AxisTransformation}
    return FlatSmile(vols[div(length(vols) + 1, 2)])
end
(spl::FlatSmile)(strike) = spl.vol

struct SplineDeltaSmile <: SmileFunction
    pp::PPInterpolation.PP
    forward::Float64
    tte::Float64
    atmVol::Float64
end
function calibrateSmile(::Type{SplineDeltaSmile}, axisTransforms::Vector{U}, xs::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; isFlat=false) where {T,U<:AxisTransformation}
    ys = map((trans, x, vol) -> convertToLogmoneyness(trans, x, vol), axisTransforms, xs, vols)
    callDelta = @. deltaForLogmoneyness(CallForwardDelta(), ys, vols, tte, forward, 1.0)
    pp = if isFlat
        PPInterpolation.makeCubicPP(reverse(callDelta), reverse(vols), PPInterpolation.FIRST_DERIVATIVE, 0.0, PPInterpolation.FIRST_DERIVATIVE, 0.0, C2())
    else
        PPInterpolation.CubicSplineNatural(reverse(callDelta), reverse(vols))
    end
    return SplineDeltaSmile(pp, forward, tte, vols[3])
end

impliedVolatilityByDelta(spl::SplineDeltaSmile, x) = spl.pp(x)

(spl::SplineDeltaSmile)(strike; isFlat=true) = begin
    v0 = spl.atmVol    
    obj = function (v0)
        x = deltaForStrike(CallForwardDelta(), strike, v0, spl.tte, spl.forward, 1.0)
        v = impliedVolatilityByDelta(spl, x)
        return v - v0
    end
    objDer = x -> ForwardDiff.derivative(obj, x)
    return Roots.find_zero((obj, objDer), v0, Roots.Newton())
end

function impliedVolatilityByDelta(spl::SmileFunction, delta, isCall::Bool, convention::BrokerConvention)
    withPremium = convention.withPremium
    isForward = convention.isForward
    forward = convention.forward
    tte = convention.tte
    dfForeign = convention.dfForeign
    logm = solveLogmoneynessForDelta(makeStrikeConvention(isCall, withPremium, isForward), delta, y -> spl(forward * exp(y))^2, tte, forward, dfForeign)
    return spl(forward * exp(logm))
end


function evaluateSmileOnQuotes(smileFunction, convention::BrokerConvention, volAtm, bf25, rr25, bf10, rr10; isVegaWeighted=true, isGlobal=true) where {SF<:SmileFunction}
    strikeAtm = strikeForDelta(if convention.withPremium
            ATMPremiumStraddle()
        else
            ATMStraddle()
        end, volAtm, convention.tte, convention.forward, convention.dfForeign)
    volStr25 = (volAtm + bf25)
    strike25StraddleCall = strikeForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), 0.25, volStr25, convention.tte, convention.forward, convention.dfForeign)
    strike25StraddlePut = strikeForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), -0.25, volStr25, convention.tte, convention.forward, convention.dfForeign)
    straddle25Value = blackScholesFormula(true, strike25StraddleCall, convention.forward, volStr25^2 * convention.tte, 1.0, 1.0) + blackScholesFormula(false, strike25StraddlePut, convention.forward, volStr25^2 * convention.tte, 1.0, 1.0)
    volStr10 = (volAtm + bf10)
    strike10StraddleCall = strikeForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), 0.10, volStr10, convention.tte, convention.forward, convention.dfForeign)
    strike10StraddlePut = strikeForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), -0.10, volStr10, convention.tte, convention.forward, convention.dfForeign)
    straddle10Value = blackScholesFormula(true, strike10StraddleCall, convention.forward, volStr10^2 * convention.tte, 1.0, 1.0) + blackScholesFormula(false, strike10StraddlePut, convention.forward, volStr10^2 * convention.tte, 1.0, 1.0)
    vega10 = if isVegaWeighted
        blackScholesVega(strike10StraddlePut, convention.forward, volStr10^2 * convention.tte, 1.0, 1.0, convention.tte) + blackScholesVega(strike10StraddleCall, convention.forward, volStr10^2 * convention.tte, 1.0, 1.0, convention.tte)
    else
        1.0
    end
    vega25 = if isVegaWeighted
        blackScholesVega(strike25StraddlePut, convention.forward, volStr25^2 * convention.tte, 1.0, 1.0, convention.tte) + blackScholesVega(strike25StraddleCall, convention.forward, volStr25^2 * convention.tte, 1.0, 1.0, convention.tte)
    else
        1.0
    end
    function obj!(fvec)

        vol25Call = impliedVolatilityByDelta(smileFunction, 0.25, true, convention)
        vol25Put = impliedVolatilityByDelta(smileFunction, -0.25, false, convention)
        vol10Call = impliedVolatilityByDelta(smileFunction, 0.10, true, convention)
        vol10Put = impliedVolatilityByDelta(smileFunction, -0.10, false, convention)

        # vol25Put = 2*volStr25-vol25Call
        # vol10Put = 2*volStr10 - vol10Call
        strike25Call = strikeForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), 0.25, vol25Call, convention.tte, convention.forward, convention.dfForeign)
        strike10Call = strikeForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), 0.10, vol10Call, convention.tte, convention.forward, convention.dfForeign)
        strike25Put = strikeForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), -0.25, vol25Put, convention.tte, convention.forward, convention.dfForeign)
        strike10Put = strikeForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), -0.10, vol10Put, convention.tte, convention.forward, convention.dfForeign)
        #		println(ct0, " objective calibration ",[strike10Put, strike25Put, strikeAtm, strike25Call, strike10Call], " ",[vol10Put, vol25Put, volAtm, vol25Call, vol10Call])   
        vol25StraddleCall = smileFunction(strike25StraddleCall)
        vol25StraddlePut = smileFunction(strike25StraddlePut)
        vol10StraddleCall = smileFunction(strike10StraddleCall)
        vol10StraddlePut = smileFunction(strike10StraddlePut)

        #blackScholesFormula(isCall::Bool, strike::Number, spot::Number, totalVariance::Number, driftDf::Number, discountDf::Number)
        fvec[1] =
            (straddle25Value -
             blackScholesFormula(true, strike25StraddleCall, convention.forward, vol25StraddleCall^2 * convention.tte, 1.0, 1.0) - blackScholesFormula(false, strike25StraddlePut, convention.forward, vol25StraddlePut^2 * convention.tte, 1.0, 1.0)) / vega25

        fvec[2] =
            (straddle10Value - blackScholesFormula(true, strike10StraddleCall, convention.forward, vol10StraddleCall^2 * convention.tte, 1.0, 1.0) -
             blackScholesFormula(false, strike10StraddlePut, convention.forward, vol10StraddlePut^2 * convention.tte, 1.0, 1.0)) / vega10
        if isGlobal
            fvec[3] = (smileFunction(strike25Call) - smileFunction(strike25Put) - rr25)
            fvec[4] = (smileFunction(strike10Call) - smileFunction(strike10Put) - rr10)
            fvec[5] = smileFunction(strikeAtm) - volAtm
        end
        return fvec
    end
    return obj!(zeros(5))
end

# Assumes .... 25Put 10Put ATM 10Call 25Call
function calibrateSmileToVanillas(smileType, convention::BrokerConvention, vanillaVols; useDelta=true, deltas=[-0.10,-0.25,0.5,0.25,0.10])
    atmIndex = div(length(vanillaVols) + 1, 2)
    volAtm = vanillaVols[atmIndex]
    yAtm = logmoneynessForDelta(if convention.withPremium
            ATMPremiumStraddle()
        else
            ATMStraddle()
        end, volAtm, convention.tte, convention.forward, convention.dfForeign)
    atmTrans = LogmoneynessAxisTransformation()
    return if useDelta
        deltaTrans = DeltaAxisTransformation(convention)
        calibrateSmile(smileType, [deltaTrans, deltaTrans, atmTrans, deltaTrans, deltaTrans], [deltas[atmIndex-2], deltas[atmIndex-1], yAtm, deltas[atmIndex+1], deltas[atmIndex+2]], vanillaVols, convention.forward, convention.tte)
    else
        yTrans = LogmoneynessAxisTransformation()
        vol25Call = vanillaVols[atmIndex+1]
        vol10Call = vanillaVols[atmIndex+2]
        vol25Put = vanillaVols[atmIndex-1]
        vol10Put = vanillaVols[atmIndex-2]
        y25Call = logmoneynessForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), deltas[atmIndex+1], vol25Call, convention.tte, convention.forward, convention.dfForeign)
        y10Call = logmoneynessForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), deltas[atmIndex+2], vol10Call, convention.tte, convention.forward, convention.dfForeign)
        y25Put = logmoneynessForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), deltas[atmIndex-1], vol25Put, convention.tte, convention.forward, convention.dfForeign)
        y10Put = logmoneynessForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), deltas[atmIndex-2], vol10Put, convention.tte, convention.forward, convention.dfForeign)
        calibrateSmile(smileType, [yTrans, yTrans, yTrans, yTrans, yTrans], [y10Put, y25Put, yAtm, y25Call, y10Call], vanillaVols, convention.forward, convention.tte)
    end
end

function convertQuotesToDeltaVols(convention::BrokerConvention, volAtm, bf25, rr25, bf10, rr10; smileType::Type{SF}=SplineSmile, isVegaWeighted=false, isGlobal=false, useDelta=true) where {SF<:SmileFunction}
    yAtm = logmoneynessForDelta(if convention.withPremium
            ATMPremiumStraddle()
        else
            ATMStraddle()
        end, volAtm, convention.tte, convention.forward, convention.dfForeign)
    volStr25 = (volAtm + bf25)
    strike25StraddleCall = strikeForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), 0.25, volStr25, convention.tte, convention.forward, convention.dfForeign)
    strike25StraddlePut = strikeForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), -0.25, volStr25, convention.tte, convention.forward, convention.dfForeign)
    straddle25Value = blackScholesFormula(true, strike25StraddleCall, convention.forward, volStr25^2 * convention.tte, 1.0, 1.0) + blackScholesFormula(false, strike25StraddlePut, convention.forward, volStr25^2 * convention.tte, 1.0, 1.0)
    volStr10 = (volAtm + bf10)
    strike10StraddleCall = strikeForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), 0.10, volStr10, convention.tte, convention.forward, convention.dfForeign)
    strike10StraddlePut = strikeForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), -0.10, volStr10, convention.tte, convention.forward, convention.dfForeign)
    straddle10Value = blackScholesFormula(true, strike10StraddleCall, convention.forward, volStr10^2 * convention.tte, 1.0, 1.0) + blackScholesFormula(false, strike10StraddlePut, convention.forward, volStr10^2 * convention.tte, 1.0, 1.0)
    vega10 = if isVegaWeighted
        blackScholesVega(strike10StraddlePut, convention.forward, volStr10^2 * convention.tte, 1.0, 1.0, convention.tte) + blackScholesVega(strike10StraddleCall, convention.forward, volStr10^2 * convention.tte, 1.0, 1.0, convention.tte)
    else
        1.0
    end
    vega25 = if isVegaWeighted
        blackScholesVega(strike25StraddlePut, convention.forward, volStr25^2 * convention.tte, 1.0, 1.0, convention.tte) + blackScholesVega(strike25StraddleCall, convention.forward, volStr25^2 * convention.tte, 1.0, 1.0, convention.tte)
    else
        1.0
    end
    #v-v0 * vega(v0) + price(v0)= price(v) => v-v0 = price(v) - price(v0)  / vega(v0).
    #other possibility iv of straddle.
    transforms = [MQMinTransformation(-volAtm + abs(rr25) / 2, 1.0), MQMinTransformation(-volAtm + abs(rr10) / 2, 1.0)]
    deltaTrans = DeltaAxisTransformation(convention)
    atmTrans = LogmoneynessAxisTransformation()
    yTrans = LogmoneynessAxisTransformation()
    function obj!(fvec, ct0::AbstractArray{TC}) where {TC}
        vol25SS = transforms[1](ct0[1])
        vol10SS = transforms[2](ct0[2])

        vol25Call = (vol25SS + volAtm) + rr25 / 2
        vol25Put = (vol25SS + volAtm) - rr25 / 2
        vol10Call = (vol10SS + volAtm) + rr10 / 2
        vol10Put = (vol10SS + volAtm) - rr10 / 2

        # vol25Put = 2*volStr25-vol25Call
        # vol10Put = 2*volStr10 - vol10Call
        #		println(ct0, " objective calibration ",[strike10Put, strike25Put, strikeAtm, strike25Call, strike10Call], " ",[vol10Put, vol25Put, volAtm, vol25Call, vol10Call])
        smileFunction = if useDelta
           # println("vols ",[vol10Put, vol25Put, volAtm, vol25Call, vol10Call])
            calibrateSmile(smileType, [deltaTrans, deltaTrans, atmTrans, deltaTrans, deltaTrans], [-0.10, -0.25, yAtm, 0.25, 0.10], [vol10Put, vol25Put, volAtm, vol25Call, vol10Call], convention.forward, convention.tte)
        else
            y25Call = logmoneynessForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), 0.25, vol25Call, convention.tte, convention.forward, convention.dfForeign)
            y10Call = logmoneynessForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), 0.10, vol10Call, convention.tte, convention.forward, convention.dfForeign)
            y25Put = logmoneynessForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), -0.25, vol25Put, convention.tte, convention.forward, convention.dfForeign)
            y10Put = logmoneynessForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), -0.10, vol10Put, convention.tte, convention.forward, convention.dfForeign)
            calibrateSmile(smileType, [yTrans, yTrans, yTrans, yTrans, yTrans], [y10Put, y25Put, yAtm, y25Call, y10Call], [vol10Put, vol25Put, volAtm, vol25Call, vol10Call], convention.forward, convention.tte)
        end
        vol25StraddleCall = smileFunction(strike25StraddleCall)
        vol25StraddlePut = smileFunction(strike25StraddlePut)
        vol10StraddleCall = smileFunction(strike10StraddleCall)
        vol10StraddlePut = smileFunction(strike10StraddlePut)
        #blackScholesFormula(isCall::Bool, strike::Number, spot::Number, totalVariance::Number, driftDf::Number, discountDf::Number)
        fvec[1] =
            (straddle25Value -
             blackScholesFormula(true, strike25StraddleCall, convention.forward, vol25StraddleCall^2 * convention.tte, 1.0, 1.0) - blackScholesFormula(false, strike25StraddlePut, convention.forward, vol25StraddlePut^2 * convention.tte, 1.0, 1.0)) / vega25

        fvec[2] =
            (straddle10Value - blackScholesFormula(true, strike10StraddleCall, convention.forward, vol10StraddleCall^2 * convention.tte, 1.0, 1.0) -
             blackScholesFormula(false, strike10StraddlePut, convention.forward, vol10StraddlePut^2 * convention.tte, 1.0, 1.0)) / vega10
        if isGlobal
            vol25Call = impliedVolatilityByDelta(smileFunction, 0.25, true, convention)
            vol25Put = impliedVolatilityByDelta(smileFunction, -0.25, false, convention)
            vol10Call = impliedVolatilityByDelta(smileFunction, 0.10, true, convention)
            vol10Put = impliedVolatilityByDelta(smileFunction, -0.10, false, convention)
            fvec[3] = (vol25Call - vol25Put - rr25)
            fvec[4] = (vol10Call - vol10Put - rr10)
            # fvec[3] = (smileFunction(strike25Call) - smileFunction(strike25Put) - rr25)
            # fvec[4] = (smileFunction(strike10Call) - smileFunction(strike10Put) - rr10)
            fvec[5] = smileFunction(exp(yAtm) * convention.forward) - volAtm
        end
        return fvec
    end
    fvec = zeros(Float64, if isGlobal
        5
    else
        2
    end)
    ct = [inv(transforms[1], bf25), inv(transforms[2], bf10)]
    measure, result = GaussNewton.optimize!(obj!, ct, fvec, reltol=1e-10)
    println(smileType, " measure ", measure, " fvec ", obj!(fvec, result.minimizer), " ", result)
    isGlobal = true
    fvec = zeros(Float64, 5)
    println("Full fvec ", obj!(fvec, result.minimizer))
    ct = result.minimizer
    vol25SS = transforms[1](ct[1])
    vol10SS = transforms[2](ct[2])

    vol25Call = (vol25SS + volAtm) + rr25 / 2
    vol25Put = (vol25SS + volAtm) - rr25 / 2
    vol10Call = (vol10SS + volAtm) + rr10 / 2
    vol10Put = (vol10SS + volAtm) - rr10 / 2
    #	println("ct ",ct," vols ",[vol10Put, vol25Put, volAtm, vol25Call, vol10Call])
    return [vol10Put, vol25Put, volAtm, vol25Call, vol10Call]

end


#max(eta*(S-K),0)*S in dom currency. To convert to spot in for currency.

function priceAutoQuanto(isCall::Bool, strike, forward, tte, smile::SmileFunction, discountDf::Number; ndev=6, q::Quadrature=GaussKronrod())
    integrand = function (k)
        totalVariance = smile(k)^2 * tte
        blackScholesFormula(isCall, k, forward, totalVariance, 1.0, discountDf)
    end
    atmSqrtv = smile(forward) * sqrt(tte)
    price = if isCall
        kmax = forward * exp(ndev * atmSqrtv)
        integral = integrate(q, integrand, strike, kmax)
        2 * integral + strike * blackScholesFormula(isCall, strike, forward, smile(strike)^2 * tte, 1.0, discountDf)
    else
        kmin = forward * exp(-ndev * atmSqrtv)
        integral = integrate(q, integrand, kmin, strike)
        -2 * integral + strike * blackScholesFormula(isCall, strike, forward, smile(strike)^2 * tte, 1.0, discountDf)
    end

    price
end


function priceAutoQuantoBlack(isCall::Bool, strike, spot, driftDf, discountDf, tte, smile::SmileFunction)
    dfFor = discountDf / driftDf
    tv = smile(strike)^2 * tte
    spot * blackScholesFormula(isCall, strike, spot, tv, driftDf / exp(tv), dfFor)
end
struct FXVarianceSection{T} <: VarianceSection
    smile::SmileFunction
    forward::T
end

function varianceByLogmoneyness(varianceSection::FXVarianceSection{T}, k) where {T}
    strike = varianceSection.forward * exp(k)
    varianceSection.smile(strike)^2
end
