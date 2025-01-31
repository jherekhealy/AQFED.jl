using Roots
import AQFED.Math: normcdf, norminv, inv, ClosedTransformation, FitResult, MQMinTransformation, Quadrature, GaussKronrod
import AQFED.Black: blackScholesFormula, blackScholesVega, impliedVolatility
import AQFED.TermStructure: calibrateSVISection, SVISection, varianceByLogmoneyness
import AQFED.Collocation: weightedPrices, makeIsotonicCollocation, Polynomial, BSplineCollocation, makeBSplineCollocation
import AQFED.Collocation: priceEuropean as priceEuropeanCollocation
import Polynomials: AbstractPolynomial
import AQFED.PDDE: QuadraticLVG, calibrateQuadraticLVG, Quadratic, EQuadraticLVG, calibrateEQuadraticLVG
import AQFED.PDDE: priceEuropean as priceEuropeanPDDE

using ForwardDiff
using PPInterpolation
using GaussNewton
export SmileConvention, BrokerConvention, CallDelta, PutDelta, ATMStraddle, ATMPremiumStraddle, CallPremiumDelta, PutPremiumDelta, CallForwardDelta, PutForwardDelta, CallPremiumForwardDelta, PutPremiumForwardDelta, strikeForDelta, makeStrikeConvention
export ExpPolynomialSmile, SplineSmile, calibrateSmile, convertQuotesToDeltaVols
#conversion from quotes to call deltas
abstract type FXConvention end
struct SmileConvention <: FXConvention
end
function convertQuotesToDeltaVols(convention::SmileConvention, volAtm, bf25, rr25, bf10, rr10)
	vol10Call = volAtm + bf10 + rr10 / 2
	vol10Put = volAtm + bf10 - rr10 / 2
	vol25Call = volAtm + bf25 + rr25 / 2
	vol25Put = volAtm + bf25 - rr25 / 2
	return [vol10Put, vol25Put, volAtm, vol25Call, vol10Call]
end

abstract type StrikeConvention end
struct CallDelta <: StrikeConvention
end
struct PutDelta <: StrikeConvention
end
struct ATMStraddle <: StrikeConvention
end
struct ATMPremiumStraddle <: StrikeConvention
end
struct CallPremiumDelta <: StrikeConvention
end
struct PutPremiumDelta <: StrikeConvention
end
struct CallForwardDelta <: StrikeConvention
end
struct PutForwardDelta <: StrikeConvention
end
struct CallPremiumForwardDelta <: StrikeConvention
end
struct PutPremiumForwardDelta <: StrikeConvention
end
function strikeForDelta(convention::ATMStraddle, vol, tte, forward, df)
	return exp(vol^2 * tte / 2) * forward
end
function strikeForDelta(convention::ATMPremiumStraddle, vol, tte, forward, df)
	return exp(-vol^2 * tte / 2) * forward
end
function strikeForDelta(convention::CallDelta, delta, vol, tte, forward, df)
	return forward * exp(-vol * sqrt(tte) * norminv(abs(delta) / df) + vol^2 * tte / 2)
end

function deltaForStrike(convention::CallForwardDelta, strike, vol, tte, forward, df)
	return normcdf((log(forward / strike) + vol^2 * tte / 2) / (vol * sqrt(tte)))
end

function strikeForDelta(convention::PutDelta, delta, vol, tte, forward, df)
	return forward * exp(+vol * sqrt(tte) * norminv(abs(delta) / df) + vol^2 * tte / 2)
end

function strikeForDelta(convention::CallForwardDelta, delta, vol, tte, forward, df)
	return forward * exp(-vol * sqrt(tte) * norminv(abs(delta)) + vol^2 * tte / 2)
end

function strikeForDelta(convention::PutForwardDelta, delta, vol, tte, forward, df)
	#-N(-d1) = delta => d1 = -ninv(-delta)
	return forward * exp(+vol * sqrt(tte) * norminv(abs(delta)) + vol^2 * tte / 2)
end

function strikeForDelta(convention::CallPremiumDelta, delta, vol, tte, forward, df)
	objCall = function (strike)
		delta - strike / forward * df * normcdf((log(forward / strike) - vol^2 * tte / 2) / (vol * sqrt(tte)))
	end
	objCallDer = x -> ForwardDiff.derivative(objCall, x)
	k = strikeForDelta(CallDelta(), delta, vol, tte, forward, df)
	return Roots.find_zero((objCall, objCallDer), k, Roots.Newton())
end

function strikeForDelta(convention::PutPremiumDelta, delta, vol, tte, forward, df)
	objPut = function (strike)
		delta + strike / forward * df * normcdf(-(log(forward / strike) - vol^2 * tte / 2) / (vol * sqrt(tte)))
	end
	objPutDer = x -> ForwardDiff.derivative(objPut, x)
	k = strikeForDelta(PutDelta(), delta, vol, tte, forward, df)
	return Roots.find_zero((objPut, objPutDer), k, Roots.Newton())
end

function deltaForStrike(convention::CallPremiumForwardDelta, strike, vol, tte, forward, df)
	return strike / forward * normcdf((log(forward / strike) - vol^2 * tte / 2) / (vol * sqrt(tte)))
end
function strikeForDelta(convention::CallPremiumForwardDelta, delta, vol, tte, forward, df)
	objCall = function (strike)
		delta - strike / forward * normcdf((log(forward / strike) - vol^2 * tte / 2) / (vol * sqrt(tte)))
	end
	objCallDer = x -> ForwardDiff.derivative(objCall, x)
	k = strikeForDelta(CallForwardDelta(), delta, vol, tte, forward, df)
	return Roots.find_zero((objCall, objCallDer), k, Roots.Newton())
end

function strikeForDelta(convention::PutPremiumForwardDelta, delta, vol, tte, forward, df)
	objPut = function (strike)
		delta + strike / forward * normcdf(-(log(forward / strike) - vol^2 * tte / 2) / (vol * sqrt(tte)))
	end
	objPutDer = x -> ForwardDiff.derivative(objPut, x)
	k = strikeForDelta(PutForwardDelta(), delta, vol, tte, forward, df)
	return Roots.find_zero((objPut, objPutDer), k, Roots.Newton())
end
function strikeForCallDelta(delta, vol, tte, df, withPremium::Bool)
	if withPremium
		objPut = function (strike)
			(delta - 1) + strike / forward * df * normcdf(-(log(forward / strike) - vol^2 * tte / 2) / (vol * sqrt(tte)))
		end
		objCall = function (strike)
			delta - strike / forward * df * normcdf((log(forward / strike) - vol^2 * tte / 2) / (vol * sqrt(tte)))
		end
		if delta == 0.5
			#   k = exp(0.5*vol^2 * tte)*forward
		elseif delta < 0.5
			objCallDer = x -> ForwardDiff.derivative(objCall, x)
			k = Roots.find_zero((objCall, objCallDer), k, Roots.Newton())
		else
			objPutDer = x -> ForwardDiff.derivative(objPut, x)
			k = Roots.find_zero((objPut, objPutDer), k, Roots.Newton())
		end
	end
	return k
end



struct BrokerConvention{T} <: FXConvention
	tte::T
	forward::T
	dfForeign::T
	withPremium::Bool
	isForward::Bool
end

makeStrikeConvention(isCall::Bool, withPremium::Bool, isForward::Bool) =
	if isCall
		if withPremium
			if isForward
				CallPremiumForwardDelta()
			else
				CallPremiumDelta()
			end
		else
			if isForward
				CallForwardDelta()
			else
				CallDelta()
			end
		end
	else
		if withPremium
			if isForward
				PutPremiumForwardDelta()
			else
				PutPremiumDelta()
			end
		else
			if isForward
				PutForwardDelta()
			else
				PutDelta()
			end
		end
	end

abstract type SmileFunction end
struct SplineSmile <: SmileFunction
	pp::PPInterpolation.PP
	forward::Float64
end
function makeTotalVarianceFunction(smile::SmileFunction, forward, tte)
	return y -> smile(forward * exp(y))^2 * tte
end

function calibrateSmile(::Type{SplineSmile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; isFlat = false) where {T}
	pp = if !isFlat
		PPInterpolation.CubicSplineNatural(log.(strikes ./ forward), vols .^ 2)
	else
		PPInterpolation.makeCubicPP(log.(strikes ./ forward), vols .^ 2, PPInterpolation.FIRST_DERIVATIVE, 0.0, PPInterpolation.FIRST_DERIVATIVE, 0.0, C2())
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
function calibrateSmile(::Type{SVISmile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; aMin = 0.0, sMin = one(T) / 10000, noarbFactor = 2 * one(T)) where {T}
	svi0, rmsesvi = calibrateSVISection(tte, forward, log.(strikes ./ forward), vols, ones(length(vols)), aMin = aMin, sMin = sMin, noarbFactor = noarbFactor)
	return SVISmile(svi0)
end
(spl::SVISmile)(strike) = sqrt(varianceByLogmoneyness(spl.svi, log(strike / spl.svi.f)))

struct XSSVISmile <: SmileFunction
	svi::XSSVISection
end
function calibrateSmile(::Type{XSSVISmile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T}
	svi0 = calibrateXSSVISection(tte, forward, log.(strikes ./ forward), vols, ones(length(vols)))
	return XSSVISmile(svi0)
end
(spl::XSSVISmile)(strike) = sqrt(varianceByLogmoneyness(spl.svi, log(strike / spl.svi.f)))

struct SABRSmile <: SmileFunction
	sabr::SABRSection
end
function calibrateSmile(::Type{SABRSmile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T}	
	ys = @.  log(strikes / forward)
	guess = initialGuessBlackATM(forward, tte, 1.0, ys, vols)
	sabr = calibrateSABRSectionFromGuess(tte, forward, ys, vols, ones(length(vols)), guess)
	return SABRSmile(sabr)
end
(spl::SABRSmile)(strike) = sqrt(varianceByLogmoneyness(spl.sabr, log(strike / spl.sabr.f)))

struct SABRATMSmile <: SmileFunction
	sabr::SABRSection
end
function calibrateSmile(::Type{SABRATMSmile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T}	
	ys = @.  log(strikes / forward)
	sabr = calibrateSABRSectionATM(tte, forward, ys, vols, ones(length(vols)), 1.0)
	return SABRSmile(sabr)
end
(spl::SABRATMSmile)(strike) = sqrt(varianceByLogmoneyness(spl.sabr, log(strike / spl.sabr.f)))

struct LVGSmile <: SmileFunction
	lvg::QuadraticLVG
end
function calibrateSmile(::Type{LVGSmile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; L = 1e-4) where {T}
	prices, wv = weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor = 1e-8)
	lvg = calibrateQuadraticLVG(tte, forward, strikes, prices, wv, useVol = false, model = Quadratic(), location = "Mid-XX", size = 0, L = L, U = max(strikes[end] * 1.01, forward * exp(12 * vols[5] * sqrt(tte))))
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
function calibrateSmile(::Type{ELVGSmile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; L = 1e-4, U = max(strikes[end] * 1.01, forward * exp(8 * vols[5] * sqrt(tte)))) where {T}
	prices, wv = weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor = 1e-8)
	lvg = calibrateEQuadraticLVG(tte, forward, strikes, prices, wv, useVol = false, model = "Quadratic", nRefine = 5, location = "Mid-XX", size = 0, L = L, U = U)
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
function calibrateSmile(::Type{QuinticCollocationSmile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; minSlope = 1e-4) where {T}
	prices, wv = weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor = 1e-10)
	isoc, m = makeIsotonicCollocation(strikes, prices, wv, tte, forward, 1.0, deg = 5, degGuess = 1, minSlope = minSlope)
	sol = Polynomial(isoc)
	return QuinticCollocationSmile(sol, forward, tte)
end
(spl::QuinticCollocationSmile)(strike) = impliedVolatility(
	strike > spl.forward,
	priceEuropeanCollocation(spl.collocation, strike > spl.forward, strike, spl.forward, 1.0),
	spl.forward,
	strike,
	spl.tte,
	1.0)

struct BSplineCollocationSmile <: SmileFunction
	collocation::BSplineCollocation
	forward::Float64
	tte::Float64
end
function calibrateSmile(::Type{BSplineCollocationSmile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T}
	prices, wv = weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor = 1e-4)
	bspl, m = makeBSplineCollocation(
		strikes,
		prices,
		wv,
		tte,
		forward,
		1.0,
		penalty = 0e-2,
		size = 0,
		minSlope = 1e-7,
		rawFit = true, N = 3,
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
function calibrateSmile(::Type{LognormalMixtureSmile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T}
	prices, wv = weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor = 1e-8)
	kernel3 = calibrateLognormalMixture(tte, forward, strikes, prices, wv, useVol = false, size = 2)
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
function calibrateSmile(::Type{LognormalMixture3Smile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T}
	prices, wv = weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor = 1e-8)
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
function calibrateSmile(::Type{ExpPolynomialSmile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; nIter = 256) where {T}
	#could compute fwd deltas from strikes, vols. and log(vol) = quartic(delta)
	simpleDeltas = if nIter == 1
		@.(normcdf((log(forward / strikes)) / (vols[3] * sqrt(tte))))
	else
		@.(normcdf((log(forward / strikes)) / (vols * sqrt(tte))))
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

	m = [         sum1 sumx sumx2 sumx3 sumx4;
		sumx sumx2 sumx3 sumx4 sumx5;
		sumx2 sumx3 sumx4 sumx5 sumx6;
		sumx3 sumx4 sumx5 sumx6 sumx7;
		sumx4 sumx5 sumx6 sumx7 sumx8]
	rhs = [sumy, sumxy, sumx2y, sumx3y, sumx4y]
	coeff = m \ rhs
	return ExpPolynomialSmile(coeff, vols[3], forward, tte, nIter)
end

function impliedVolatilityByDelta(spl::ExpPolynomialSmile, x)
	return exp(spl.coeff[1] + spl.coeff[2] * x + spl.coeff[3] * x^2 + spl.coeff[4] * x^3 + spl.coeff[5] * x^4)
end

(spl::ExpPolynomialSmile)(strike; method = :newton, useTrace = false, out = Vector{Float64}()) = begin
	coeff = spl.coeff

	if method == :newton && spl.nIter > 1
		v0 = spl.atmVol
		obj = function (v0)
			x = normcdf((log(spl.forward / strike)) / (v0 * sqrt(spl.tte)))
			v = impliedVolatilityByDelta(spl, x)
			return v - v0
		end
		objDer = x -> ForwardDiff.derivative(obj, x)
		return Roots.find_zero((obj, objDer), v0, Roots.Newton())
	else #fixed point
		v = spl.atmVol
		v0 = 0.0
		i = 1
		while i <= spl.nIter && abs(v - v0) > 10 * eps(v0)
			v0 = v
			x0 = normcdf((log(spl.forward / strike)) / (v * sqrt(spl.tte)))
			v = exp(coeff[1] + coeff[2] * x0 + coeff[3] * x0^2 + coeff[4] * x0^3 + coeff[5] * x0^4)
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
function calibrateSmile(::Type{FlatSmile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64) where {T}
	return FlatSmile(vols[div(length(vols)+1,2)])
end
(spl::FlatSmile)(strike) = spl.vol

struct SplineDeltaSmile <: SmileFunction
	pp::PPInterpolation.PP
	forward::Float64
	tte::Float64
	atmVol::Float64
end
function calibrateSmile(::Type{SplineDeltaSmile}, strikes::Vector{T}, vols::Vector{T}, forward::Float64, tte::Float64; isFlat = false) where {T}
	callDelta = @. deltaForStrike(CallForwardDelta(), strikes, vols, tte, forward, 1.0)
	pp = if isFlat
		PPInterpolation.makeCubicPP(reverse(callDelta), reverse(vols), PPInterpolation.FIRST_DERIVATIVE, 0.0, PPInterpolation.FIRST_DERIVATIVE, 0.0, C2())
	else
		PPInterpolation.CubicSplineNatural(reverse(callDelta), reverse(vols))
	end
	return SplineDeltaSmile(pp, forward, tte, vols[3])
end

impliedVolatilityByDelta(spl::SplineDeltaSmile, x) = spl.pp(x)

(spl::SplineDeltaSmile)(strike; isFlat = true) = begin
	v0 = spl.atmVol
	obj = function (v0)
		x = deltaForStrike(CallForwardDelta(), strike, v0, spl.tte, spl.forward, 1.0)
		v = impliedVolatilityByDelta(spl, x)
		return v - v0
	end
	objDer = x -> ForwardDiff.derivative(obj, x)
	return Roots.find_zero((obj, objDer), v0, Roots.Newton())
end

function strikeForDelta(deltaVols::Vector{Float64}, convention::BrokerConvention; deltas = [-0.10, -0.25, 0.5, 0.25, 0.10])
	withPremium = convention.withPremium
	isForward = convention.isForward
	forward = convention.forward
	tte = convention.tte
	dfForeign = convention.dfForeign
	atmIndex = div(length(deltaVols) + 1, 2)
	atmVol = deltaVols[atmIndex]
	strikes = zeros(length(deltas))
	for i ∈ 1:atmIndex-1
		strikes[i] = strikeForDelta(makeStrikeConvention(false, withPremium, isForward), deltas[i], deltaVols[i], tte, forward, dfForeign)
	end
	strikes[atmIndex] = strikeForDelta(if withPremium
			ATMPremiumStraddle()
		else
			ATMStraddle()
		end, atmVol, tte, forward, dfForeign)
	for i ∈ atmIndex+1:length(deltas)
		strikes[i] = strikeForDelta(makeStrikeConvention(true, withPremium, isForward), deltas[i], deltaVols[i], tte, forward, dfForeign)
	end
	strikes
end

function convertQuotesToDeltaVols(convention::BrokerConvention, volAtm, bf25, rr25, bf10, rr10; smileType::Type{SF} = SplineSmile) where {SF <: SmileFunction}
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
	guess = convertQuotesToDeltaVols(SmileConvention(), volAtm, bf25, rr25, bf10, rr10)
	vega10 = blackScholesVega(strike10StraddlePut, convention.forward, volStr10^2*convention.tte, 1.0, 1.0, convention.tte)
	vega25 = blackScholesVega(strike25StraddlePut, convention.forward, volStr25^2*convention.tte, 1.0, 1.0, convention.tte)
#v-v0 * vega(v0) + price(v0)= price(v) => v-v0 = price(v) - price(v0)  / vega(v0).
#other possibility iv of straddle.
	
	function obj!(fvec, ct0::AbstractArray{TC}) where {TC}
		vol25Call = abs(ct0[1]) #smilestrangle vol?
		vol10Call = abs(ct0[2])
		vol25Put = vol25Call - rr25 #may be interesting to set as output if fit is far from exact
		vol10Put = vol10Call - rr10		
		# vol25Put = 2*volStr25-vol25Call
		# vol10Put = 2*volStr10 - vol10Call
		strike25Call = strikeForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), 0.25, vol25Call, convention.tte, convention.forward, convention.dfForeign)
		strike10Call = strikeForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), 0.10, vol10Call, convention.tte, convention.forward, convention.dfForeign)
		strike25Put = strikeForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), -0.25, vol25Put, convention.tte, convention.forward, convention.dfForeign)
		strike10Put = strikeForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), -0.10, vol10Put, convention.tte, convention.forward, convention.dfForeign)
		#println(ct0, " objective calibration ",[strike10Put, strike25Put, strikeAtm, strike25Call, strike10Call], " ",[vol10Put, vol25Put, volAtm, vol25Call, vol10Call])
		smileFunction = calibrateSmile(smileType, [strike10Put, strike25Put, strikeAtm, strike25Call, strike10Call], [vol10Put, vol25Put, volAtm, vol25Call, vol10Call], convention.forward, convention.tte)
		vol25StraddleCall = smileFunction(strike25StraddleCall)
		vol25StraddlePut = smileFunction(strike25StraddlePut)
		vol10StraddleCall = smileFunction(strike10StraddleCall)
		vol10StraddlePut = smileFunction(strike10StraddlePut)

		#blackScholesFormula(isCall::Bool, strike::Number, spot::Number, totalVariance::Number, driftDf::Number, discountDf::Number)
		fvec[1] =
			(straddle25Value -
			blackScholesFormula(true, strike25StraddleCall, convention.forward, vol25StraddleCall^2 * convention.tte, 1.0, 1.0) - blackScholesFormula(false, strike25StraddlePut, convention.forward, vol25StraddlePut^2 * convention.tte, 1.0, 1.0))/2vega25

		fvec[2] =
			(straddle10Value - blackScholesFormula(true, strike10StraddleCall, convention.forward, vol10StraddleCall^2 * convention.tte, 1.0, 1.0) -
			blackScholesFormula(false, strike10StraddlePut, convention.forward, vol10StraddlePut^2 * convention.tte, 1.0, 1.0))/2vega10
		fvec[3] = (smileFunction(strike25Call) - smileFunction(strike25Put) - rr25)
		fvec[4] = (smileFunction(strike10Call) - smileFunction(strike10Put) - rr10)
		fvec[5] = smileFunction(strikeAtm) - volAtm
		return fvec
	end
	fvec = zeros(Float64, 5)
	ct = guess[4:5]
	measure = GaussNewton.optimize!(obj!, ct, fvec, reltol = 1e-10)
	obj!(fvec,ct)
	println("measure ",measure," fvec=",fvec)
	vol25Call = abs(ct[1]) #smilestrangle vol?
	vol10Call = abs(ct[2])
	vol25Put = vol25Call - rr25
	vol10Put = vol10Call - rr10

	return [vol10Put, vol25Put, volAtm, vol25Call, vol10Call]

end


function convertQuotesToDeltaVolsClark(convention::BrokerConvention, volAtm, bf25, rr25, bf10, rr10; smileType::Type{SF} = SplineSmile, isVegaWeighted = false) where {SF <: SmileFunction}
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
	guess = convertQuotesToDeltaVols(SmileConvention(), volAtm, bf25, rr25, bf10, rr10)
	vega10 = if isVegaWeighted
		2*blackScholesVega(strike10StraddlePut, convention.forward, volStr10^2*convention.tte, 1.0, 1.0, convention.tte)
	else 
		1.0
	end
	vega25 = if isVegaWeighted 
		2*blackScholesVega(strike25StraddlePut, convention.forward, volStr25^2*convention.tte, 1.0, 1.0, convention.tte)
	else 
		1.0
	end
#v-v0 * vega(v0) + price(v0)= price(v) => v-v0 = price(v) - price(v0)  / vega(v0).
#other possibility iv of straddle.
	
	function obj!(fvec, ct0::AbstractArray{TC}) where {TC}
		vol25SS = abs(ct0[1])
		vol10SS = abs(ct0[2])

		vol25Call = (vol25SS + volAtm) + rr25/2
		vol25Put = (vol25SS + volAtm) - rr25/2
		vol10Call = (vol10SS + volAtm) + rr10/2
		vol10Put = (vol10SS + volAtm) - rr10/2

		# vol25Put = 2*volStr25-vol25Call
		# vol10Put = 2*volStr10 - vol10Call
		strike25Call = strikeForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), 0.25, vol25Call, convention.tte, convention.forward, convention.dfForeign)
		strike10Call = strikeForDelta(makeStrikeConvention(true, convention.withPremium, convention.isForward), 0.10, vol10Call, convention.tte, convention.forward, convention.dfForeign)
		strike25Put = strikeForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), -0.25, vol25Put, convention.tte, convention.forward, convention.dfForeign)
		strike10Put = strikeForDelta(makeStrikeConvention(false, convention.withPremium, convention.isForward), -0.10, vol10Put, convention.tte, convention.forward, convention.dfForeign)
#		println(ct0, " objective calibration ",[strike10Put, strike25Put, strikeAtm, strike25Call, strike10Call], " ",[vol10Put, vol25Put, volAtm, vol25Call, vol10Call])
		smileFunction = calibrateSmile(smileType, [strike10Put, strike25Put, strikeAtm, strike25Call, strike10Call], [vol10Put, vol25Put, volAtm, vol25Call, vol10Call], convention.forward, convention.tte)
		vol25StraddleCall = smileFunction(strike25StraddleCall)
		vol25StraddlePut = smileFunction(strike25StraddlePut)
		vol10StraddleCall = smileFunction(strike10StraddleCall)
		vol10StraddlePut = smileFunction(strike10StraddlePut)

		#blackScholesFormula(isCall::Bool, strike::Number, spot::Number, totalVariance::Number, driftDf::Number, discountDf::Number)
		fvec[1] =
			(straddle25Value -
			blackScholesFormula(true, strike25StraddleCall, convention.forward, vol25StraddleCall^2 * convention.tte, 1.0, 1.0) - blackScholesFormula(false, strike25StraddlePut, convention.forward, vol25StraddlePut^2 * convention.tte, 1.0, 1.0))/vega25

		fvec[2] =
			(straddle10Value - blackScholesFormula(true, strike10StraddleCall, convention.forward, vol10StraddleCall^2 * convention.tte, 1.0, 1.0) -
			blackScholesFormula(false, strike10StraddlePut, convention.forward, vol10StraddlePut^2 * convention.tte, 1.0, 1.0))/vega10
		#fvec[3] = (smileFunction(strike25Call) - smileFunction(strike25Put) - rr25)
		#fvec[4] = (smileFunction(strike10Call) - smileFunction(strike10Put) - rr10)
		return fvec
	end
	fvec = zeros(Float64, 2)
	ct = [bf25, bf10]
	measure, result = GaussNewton.optimize!(obj!, ct, fvec, reltol = 1e-10)
	println("measure ",measure," ",result)
	ct = result.minimizer
	vol25SS = abs(ct[1])
	vol10SS = abs(ct[2])

	vol25Call = (vol25SS + volAtm) + rr25/2
	vol25Put = (vol25SS + volAtm) - rr25/2
	vol10Call = (vol10SS + volAtm) + rr10/2
	vol10Put = (vol10SS + volAtm) - rr10/2
#	println("ct ",ct," vols ",[vol10Put, vol25Put, volAtm, vol25Call, vol10Call])
	return [vol10Put, vol25Put, volAtm, vol25Call, vol10Call]

end


#max(eta*(S-K),0)*S in dom currency. To convert to spot in for currency.

function priceAutoQuanto(isCall::Bool, strike, forward, tte, smile::SmileFunction, discountDf::Number; ndev = 6, q::Quadrature = GaussKronrod())
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
	dfFor = discountDf/ driftDf
	tv = smile(strike)^2 * tte
	spot*blackScholesFormula(isCall, strike, spot, tv, driftDf/exp(tv), dfFor)	
end
struct FXVarianceSection{T} <: VarianceSection
	smile::SmileFunction
	forward::T
end

function varianceByLogmoneyness(varianceSection::FXVarianceSection{T}, k) where {T}
	strike = varianceSection.forward * exp(k)
	varianceSection.smile(strike)^2
end
