using PPInterpolation

import AQFED.Math: LogisticTransformation, TanhTransformation, AlgebraicTransformation
using Roots
import AQFED.Math: quadRootsReal, normcdf, normpdf, norminv, inv, ClosedTransformation, FitResult
import AQFED.Black: blackScholesFormula, blackScholesVega, impliedVolatility
using LeastSquaresOptim
using SpecialFunctions
using Optim
#using JSOSolvers, ADNLPModels

struct ExpSplineCollocation{T, U}
	g::QuadraticPP{T, U}
	forward::U
	cLeft::T #left extrapolation curvature coeff in (-1/2, 0]
	cRight::T #right extrapolation curvature coeff in [0, 1/2)
end

Base.length(p::ExpSplineCollocation) = Base.length(p.g.x)
Base.broadcastable(p::ExpSplineCollocation) = Ref(p)


function solvePositiveStrike(c::ExpSplineCollocation{U, T}, expStrike)::Tuple{U, Int} where {U, T <: Number}
	#Note: specifying types above makes a drastic difference in performance
	pp = c.g
	n = length(pp.x)
	strike = log(expStrike)
	# println("logstrike ",strike, " ",pp.a)
	if strike <= pp.a[1] # is zero
		#println(strike," ",pp.x[1], " ",pp.a[1], " ",pp.b[1], " ",pp.c[1])
		x0 = pp.x[1]
		#aa*x^2 + bb*x + cc = 0
		#extrapolation is cLeft( x-x0)^2 + b*(x-x0) + a = strike
		#we have b = pp.b[1] and a = pp.a[1] because function is C1 at x0.
		cc = pp.a[1] + x0 * (-pp.b[1] + x0 * c.cLeft) - strike
		bb = pp.b[1] - 2 * x0 * c.cLeft
		aa = c.cLeft
		allck = quadRootsReal(aa, bb, cc)
		for cki in allck
			if cki <= pp.x[1] + sqrt(eps(strike))
				return cki, 0
			end
		end
		throw(DomainError(strike, "strike not found"))
	elseif strike >= pp.a[n]
		x0 = pp.x[n]
		#aa*x^2 + bb*x + cc = 0
		#extrapolation is cRight( x-x0)^2 + b*(x-x0) + a = strike
		#we have b = valueDerivative(xn) and a = pp.a[n] because function is C1 at x0.
		b = evaluateDerivativePiece(pp, n - 1, pp.x[n])
		cc = pp.a[n] + x0 * (-b + x0 * c.cRight) - strike
		bb = b - 2 * x0 * c.cRight
		aa = c.cRight
		allck = quadRootsReal(aa, bb, cc)
		for cki in allck
			if cki >= pp.x[n] - sqrt(eps(strike))
				return cki, n
			end
		end
		throw(DomainError(strike, "strike not found"))
	end
	i = searchsortedfirst(pp.a, strike)  # x[i-1]<z<=x[i]
	if strike == pp.a[i]
		return pp.x[i], i
	end
	if i > 1
		i -= 1
	end
	return solveStrikeAt(pp, i, strike)
end


function priceEuropean(
	c::ExpSplineCollocation{T, U},
	isCall::Bool,
	strike,
	forward,
	discountDf,
)::T where {T, U}
	ck, ckIndex = solvePositiveStrike(c, strike)
	local callPrice::T
	local putPrice::T
	if ckIndex == 0
		#left extrap
		integral = firstMomentExtrapolationBounded(c, 1, -300.0, ck)
		putPrice = strike * normcdf(ck) - integral
		#println("put ",putPrice,  " ",integral," ck ",ck, " strike ", strike)
		callPrice = forward - strike + putPrice
	else
		# println("strike ", strike, " ",ck, " ",ckIndex)
		valuef = hermiteIntegralBounded(c, ck, ckIndex)
		valuek = normcdf(-ck)
		callPrice = valuef - strike * valuek
		putPrice = -(forward - strike) + callPrice
	end
	if isCall
		return callPrice * discountDf
	else
		return putPrice * discountDf
	end
end


function hermiteIntegral(p::ExpSplineCollocation{T, U})::T where {T, U}
	return hermiteIntegralBounded(p, -300.0, 0)
end


function hermiteIntegralBounded(p::ExpSplineCollocation{T, U}, ck, ckIndex::Int)::T where {T, U}
	pp = p.g
	n = length(pp.x)
	i = ckIndex
	if i > length(pp.x)
		i -= 1
	end
	integral = zero(ck)
	if ck <= pp.x[1]
		integral += firstMomentExtrapolationBounded(p, 1, ck, pp.x[1])
		i = 0
	elseif ck < pp.x[n] && i < n
		#include logck to x[i] with i-1 coeffs
		integral += firstMomentBounded(p, i, ck, pp.x[i+1])
	end
	for j ∈ i+1:n-1
		integral += firstMomentBounded(p, j, pp.x[j], pp.x[j+1])
			#  @assert _finiteD(integral)
  end

	integral += firstMomentExtrapolationBounded(p, n, max(pp.x[n], ck), 300.0)
	#		  @assert _finiteD(integral)
  if isnan(integral) || isinf(integral)
		println(ck, " ", ckIndex, " inf integral ", p.g)
		throw(DomainError("infinite integral"))
	end
	return integral
end


function firstMomentExtrapolationBounded(p::ExpSplineCollocation{T, U}, i::Int, x0t, x1t)::T where {T, U}
	pp = p.g
	# i = length(pp.x) or i = 1
	if i != length(pp.x) && i != 1
		throw(DomainError(string("i must be 1 or n but was ", i)))
	end
	x0 = pp.x[i]
	slope = if i == 1
		#println(" pp ",pp.x, " ",x0)
		pp.b[1]
	else
		evaluateDerivativePiece(pp, length(pp.x) - 1, x0)
	end
	a2 = if i == 1
		p.cLeft
	else
		p.cRight
	end
	a0 = pp.a[i] + x0 * (-slope + x0 * a2)
	a1 = slope - 2 * x0 * a2
	#   @assert _finiteD(pp.a[i])
  
	#println("a ",a0, " ",a1," ",a2," x ",x0," ",slope)
	return firstMomentExpQuadraticBounded(x0t, x1t, a0, a1, a2)
end

function firstMomentBounded(p::ExpSplineCollocation{T, U}, i::Int, a::Union{U, T}, b::Union{U, T})::T where {T, U}
	pp = p.g
	x0 = pp.x[i]
	x1 = pp.x[i+1]
	x0t = max(a, x0)
	x1t = min(b, x1)
	c0 = pp.c[i, 1]
	a0 = pp.a[i] + x0 * (-pp.b[i] + x0 * c0)
	a1 = pp.b[i] - 2 * x0 * c0
	a2 = c0
	return firstMomentExpQuadraticBounded(x0t, x1t, a0, a1, a2)
end



function density(c::ExpSplineCollocation{T, U}, strike::U)::U where {T, U <: Number}
	ck, ckIndex = solvePositiveStrike(c, strike)
	dp = if ckIndex <= 0
		pp = c.g
		pp.b[1] + 2 * c.cLeft * (ck - pp.x[1])
	elseif ckIndex >= length(c.g.x)
		pp = c.g
		n = length(pp.x)
		slope = evaluateDerivativePiece(pp, n - 1, pp.x[n])
		slope + 2 * c.cRight * (ck - pp.x[n])
	else
		evaluateDerivative(c.g, ck)
	end
	an = normpdf(ck) / (dp * strike)
	return an
end



function densityDenominator(c::ExpSplineCollocation{T, U}, strike)::U where {T, U <: Number}
	ck, ckIndex = solvePositiveStrike(c, strike)
	if ckIndex <= 0 || ckIndex >= length(c.g.x)
		return zero(ck)
	end

	dp = 1 / (evaluateDerivative(c.g, ck))

	return dp
end


function collocationFunction(c::ExpSplineCollocation{T, U}, strike::U)::U where {T, U <: Number}
	ck, ckIndex = solvePositiveStrike(c, strike)
	return if ckIndex <= 0
		pp = c.g
		exp(pp.a[1] + pp.b[1] * (ck - pp.x[1]) + c.cLeft * (ck - pp.x[1])^2)
	elseif ckIndex >= length(c.g.x)
		pp = c.g
		n = length(pp.x)
		slope = evaluateDerivativePiece(pp, n - 1, pp.x[n])
		exp(pp.a[n] + slope * (ck - pp.x[n]) + c.cRight * (ck - pp.x[n])^2)
	else
		exp(PPInterpolation.evaluate(c.g, ck))
	end
end


function adjustForward(lsc::ExpSplineCollocation)
	theoForward = hermiteIntegral(lsc)
	if (theoForward < zero(theoForward))
		println("inf forward ", theoForward, " ", lsc.g)
	end
	lnTheo = log(theoForward)
	if isnan(lnTheo) || isinf(lnTheo)
		println("inf forward ", theoForward, " ", lsc.g)
	else
		#    println("for ",theoForward)
		lnf = log(lsc.forward)
		lsc.g.a .+= lnf - lnTheo
	end
end

function makeExpSplineCollocation(
	strikes::Vector{T},
	callPrices::Vector{T},
	weights::Vector{T},
	τ::T,
	forward::T,
	discountDf::T;
	minSlope = 1e-4,
	penalty = 0.0,
	size = 0,
	rawFit = false,
	minSlopeKnots = minSlope, isFlatGuess = false, extrapolationFactor = 1.25,
	isBSpline = false, trace = nothing, minimizer = "LM", isMarketStrike = false,
	cLeft = -0.0, cRight = 0.0, withForward = false,
	autodiff = :forward,
)::Tuple{ExpSplineCollocation, FitResult} where {T} #return collocation and error measure

	strikesf, pricesf, weightsf = filterConvexPrices(
		strikes,
		callPrices ./ discountDf,
		weights,
		forward,
		tol = minSlope)
	isoc = makeExpSplineCollocationGuess(strikesf, pricesf, weightsf, τ, forward, 1.0, size = size, isFlat = isFlatGuess, isMarketStrike = isMarketStrike, extrapolationFactor = extrapolationFactor, cLeft = cLeft, cRight = cRight)
	isoc, m = if isBSpline
		rawFit ? fitBSpline(isoc, strikes, callPrices, weights, forward, discountDf, minSlope = minSlope, penalty = penalty, minimizer = minimizer, trace = trace, withForward = withForward, autodiff = autodiff) :
		fitBSpline(isoc, strikesf, pricesf, weightsf, forward, discountDf, minSlope = minSlope, penalty = penalty, minimizer = minimizer, trace = trace, withForward = withForward, autodiff = autodiff)
	else
		rawFit ? fitSpline(isoc, strikes, callPrices, weights, forward, discountDf, minSlope = minSlope, penalty = penalty, trace = trace, minimizer = minimizer, withForward = withForward, autodiff = autodiff) :
		fitSpline(isoc, strikesf, pricesf, weightsf, forward, discountDf, minSlope = minSlope, penalty = penalty, minimizer = minimizer, trace = trace, withForward = withForward, autodiff = autodiff)
	end
	return isoc, m
end

function makeExpSplineCollocationGuess(strikes, prices, weights, τ, forward, discountDf; size::Int = 0, slopeTolerance::Float64 = sqrt(eps(Float64)), isFlat = false, isMarketStrike = true, extrapolationFactor = 1.25, cLeft = -1 / 8, cRight = 1 / 8)
	if !isMarketStrike
		isob = fitExpBSplineBachelier(strikes, prices, weights, τ, forward, discountDf, size = size, isFlat = isFlat, slopeTolerance = slopeTolerance, extrapolationFactor = extrapolationFactor)
		isoc = ExpSplineCollocation(isob.g, isob.forward, cLeft, cRight)
		adjustForward(isoc)
		return isoc
	end
	a = if size == 0
		log.(strikes)
	else
		n = max(size, 3) + 1
		collect(range(log(strikes[1]), log(strikes[end]), length = n))
	end
	n = length(a)
	σ = if isFlat
		m = length(strikes)
		i = findfirst(x -> x > forward, strikes)
		if i === nothing
			i = m
		elseif i == 1
			i = 2
		end
		price = (prices[i] * (forward - strikes[i-1]) + prices[i-1] * (strikes[i] - forward)) / (strikes[i] - strikes[i-1])
		strike = forward
		#bvol = Bachelier.bachelierImpliedVolatility(price, true, strike, τ, forward, discountDf)
		# isoc = IsotonicCollocation(Polynomials.Polynomial([sqrt(bvol * sqrt(τ))]), Polynomials.Polynomial([0.0]), forward)
		vol = impliedVolatility(true, price, forward, strike, τ, discountDf)
		vol * sqrt(τ) * ones(Float64, n)
	else
		map((k, price) -> impliedVolatility(true, price, forward, exp(k), τ, discountDf) * sqrt(τ), a, prices)
	end

	x = @. ((a + σ^2 / 2 - log(forward)) / σ)
	b = copy(σ)
	c = zeros(Float64, (n - 1, 1))
	pp = QuadraticPP{Float64, Float64}(a, b, c, x)
	isoc = ExpSplineCollocation(pp, forward, cLeft, cRight)
	adjustForward(isoc)
	return isoc
end

# function computeExtrapolationCurvature(type, pp::QuadraticPP)
# 	return (-one(type) * 1/ 8, one(type) * 1 / 8)
# 	#  m = length(pp.x)
# 	#  cLeft = min(zero(pp.c[1]),max(pp.c[1], -one(pp.c[1]) * 0.4))
# 	#  cRight = max(zero(pp.c[m-1]),min(evaluateSecondDerivativePiece(pp, m - 1, pp.x[end])/2, one(pp.c[m-1]) * 0.4))
# 	#  return (cLeft, cRight)
# end
#fit try various optims, transforms.
#penalty on density function instead of pure spline
function fitSpline(isoc::ExpSplineCollocation{TZ, TX}, strikes, prices, weights, forward, discountDf; minSlope = 1e-8, penalty = 0.0, mindx = 1e-4, xmax = 10.0, withForward = false, trace = Nothing, minimizer = "LM", autodiff = :forward) where {TZ, TX}
	#we optimize the xs with fixed ys
	iter = 0
	transform1 = LogisticTransformation(-xmax, xmax)
	transform = LogisticTransformation(mindx, xmax)
	xt = zeros(TX, length(isoc.g.x))
	xt[1] = inv(transform1, max(-xmax, min(isoc.g.x[1], xmax)))
	for i ∈ 1:length(xt)-1
		xt[i+1] = inv(transform, min(max(isoc.g.x[i+1] - isoc.g.x[i], mindx)))
	end
	logstrikes = log.(strikes)
	cLeft, cRight = isoc.cLeft, isoc.cRight

	function obj!(fvec, ct0::AbstractArray{T}) where {T}
		m = length(ct0)
		ct = zeros(T, m)
		ct[1] = transform1(ct0[1])
		for i ∈ 1:m-1
			ct[i+1] = transform(ct0[i+1]) + ct[i]
		end
		pp = SchumakerQuadraticPP(ct, logstrikes, LamDerivative{T}())
		#pp = SchumakerQuadraticPP(ct, logstrikes)
		#println("pp ",pp)
		lsc = ExpSplineCollocation(pp, T(forward), T(cLeft), T(cRight))
		theoForward = hermiteIntegral(lsc)

		adjustForward(lsc)
		if !isnothing(trace) && T == Float64
			trace = push!(trace, pp.a)
		end
		iter += 1
		n = length(strikes)
		for i ∈ 1:n
			try
				fvec[i] = weights[i] * (priceEuropean(lsc, true, strikes[i], forward, discountDf) - prices[i])
				if isnan(fvec[i])
					throw(DomainError("fvec is NaN"))
				end
			catch exception
				#println(exception)
				fvec[i] = -weights[i] * prices[i]
				throw(exception)
			end
		end
		if withForward
			fvec[n+1] = 1e-2 * (theoForward - forward)
		end
		if penalty > 0
			nStart = if withForward
				n + 1
			else
				n
			end

			for i ∈ 1:length(strikes)-1
				fvec[nStart+i] = penalty * (densityDenominator(lsc, strikes[i]) - densityDenominator(lsc, strikes[i+1]))
			end
		end
		#println("fvec ",fvec)
		fvec
	end
	outlen = length(strikes)
	if withForward
		outlen += 1
	end
	if penalty > 0
		outlen += length(logstrikes) - 1
	end
	fvec = zeros(Float64, outlen)
	ct0, ssr, fit = if minimizer == "GN"
		_, fit0 = GaussNewton.optimize!(obj!, xt, fvec, autodiff = autodiff, iscale = 1, NewtStep1st = false)
		(fit0.minimizer, fit0.ssr, fit0)
	elseif minimizer == "LBFGS"
		obj1 = function (x)
			fvec = zeros(typeof(x[1]), outlen)
			return sum(obj!(fvec, x) .^ 2)
		end
		# nlp = ADNLPModel(obj1, xt, #gradient_backend =  ADNLPModels.ReverseDiffADGradient, hessian_backend = ADNLPModels.ReverseDiffADHessian
		# )
		# stats = lbfgs(nlp)
		# (stats.solution, stats.objective, stats)
		td = Optim.OnceDifferentiable(obj1, xt; autodiff = :forward)

		res = Optim.optimize(td, xt, method = Optim.BFGS())
		(Optim.minimizer(res), Optim.minimum(res), res)
	elseif minimizer == "TRUNK"
		nlp = ADNLPModel(x -> begin
				fvec = zeros(typeof(x[1]), outlen)
				return obj!(fvec, x)
			end, xt,
		)
		stats = trunk(nlp)
		(stats.solution, stats.objective, stats)
	else
		fit0 = LeastSquaresOptim.optimize!(
			LeastSquaresProblem(x = xt, f! = obj!, autodiff = autodiff,
				output_length = outlen),
			LeastSquaresOptim.Dogleg(LeastSquaresOptim.QR());
			iterations = 1024,
		)
		obj!(fvec, fit0.minimizer)
		(fit0.minimizer, fit0.ssr, fit0)
	end
	#println(iter, " fit ", fit,  #obj(fit.x))  #fit.f
	ct = copy(ct0)
	ct[1] = transform1(ct0[1])
	for i ∈ 1:length(xt)-1
		ct[i+1] = transform(ct0[i+1]) + ct[i]
	end
	#println("spline with x=", ct, "; y=", logstrikes)
	pp = SchumakerQuadraticPP(ct, logstrikes, LamDerivative{Float64}())
	#pp = SchumakerQuadraticPP(ct, logstrikes)
	lsc = ExpSplineCollocation(pp, forward, cLeft, cRight)
	adjustForward(lsc)
	return lsc, FitResult(ssr, iter, ct0, fvec, fit)
end


function fitBSpline(isoc::ExpSplineCollocation{TZ, TX}, strikes, prices, weights, forward, discountDf; minSlope = 1e-8, penalty = 0.0, trace = nothing, minimizer = "LM", withForward = false, useCurvature = false, autodiff = :forward) where {TZ, TX}
	iter = 0
	basis = BSplineBasis(3, isoc.g.x)
	t = BSplines.knots(basis)
	c = zeros(Float64, length(basis)) # = length x + 1
	spl = convert(BSplines.Spline, isoc.g)
	#println("coeff ",spl.coeffs)
	ct = zeros(Float64, length(spl.coeffs) - 1)
	alpha1 = spl.coeffs[1]
	minValue = max(1e-8 * abs(spl.coeffs[end] - spl.coeffs[1]), minSlope)
	maxValue = 2* abs(spl.coeffs[end] - spl.coeffs[1])
	# println("initial coeffs ",spl.coeffs)
	#transform = ExpMinTransformation(minValue)
	transform = LogisticTransformation(minValue, maxValue)
	for i ∈ 1:length(ct)
		ct[i] = inv(transform, min(max(spl.coeffs[i+1] - spl.coeffs[i], minValue*1.01), maxValue*0.99))
		# println(i, " ", spl.coeffs[i+1] - spl.coeffs[i]," ct ",ct[i])
	end
	#println("ct ",ct)
	cLeft, cRight = isoc.cLeft, isoc.cRight
	function obj!(fvec, ct0::AbstractArray{T}) where {T}

		ct = @. transform(ct0)
		α = zeros(T, length(basis)) # = length x + 1
		α[1] = -sum(ct) #balance out theoretical forward such that it does not explode
		for i ∈ 2:length(α)
			α[i] = ct[i-1] + α[i-1]
		end
		# println("iteration ",iter," ", ForwardDiff.value.(ct0))
		spl = BSplines.Spline(basis, α)
		pp = convert(QuadraticPP{T, TX}, spl)
		lsc = ExpSplineCollocation(pp, forward, T(cLeft), T(cRight))
	#	println("lsc ",α)
		theoForward = hermiteIntegral(lsc)
				
	 #@assert _finiteD(theoForward)
			
		adjustForward(lsc)

		iter += 1
		n = length(strikes)
		#@. fvec[1:n] = weights * (priceEuropean(lsc, true, strikes, forward, discountDf) - prices)
		for i ∈ 1:n
			fvec[i] = weights[i] * (priceEuropean(lsc, true, strikes[i], forward, discountDf) - prices[i])
			if isnan(fvec[i])
				println("NaN, ct0=", ct0, " ct=", ct, " lsc=", lsc)
				throw(DomainError("NaN fvec"))
			end
		end
		if !isnothing(trace) && T == Float64
			trace = push!(trace, vcat(pp.a, sum(fvec[1:n] .^ 2)))
		end
		if withForward
			fvec[n+1] = (theoForward - forward) * 1e-2
		end
		if penalty > 0
			nStart = if withForward
				n + 2
			else
				n + 1
			end
			if !useCurvature
				@. fvec[nStart:end] = ((1 / lsc.g.b[2:end] - 1 / lsc.g.b[1:end-1]) * penalty)
			else
				@. fvec[nStart:end] = lsc.g.c[1:end] * penalty
			end
		end
		#println("fvec ",fvec)
		fvec
	end
	outlen = length(strikes)
	if withForward
		outlen += 1
	end
	if penalty > 0
		outlen += length(isoc.g.b) - 1
	end

	fvec = zeros(Float64, outlen)
	ct0, ssr, fit = if minimizer == "LM"
		cfg = ForwardDiff.JacobianConfig(obj!, fvec, ct)
		function jac!(jvec, x)
			jvec[:] = ForwardDiff.jacobian(obj!, fvec, x, cfg)
			#println("jac ",ForwardDiff.value.(fvec))
			if isnan(ForwardDiff.value.(jvec[1]))
				println("NaN in fvec for x ", x)
			end
			jvec
		end

		fit0 = LeastSquaresOptim.optimize!(
			LeastSquaresProblem(x = ct, f! = obj!,
				g! = jac!, #useful to debug issue with ForwardDiff NaNs
				autodiff = autodiff,
				output_length = outlen),
			 LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.QR());
			#LeastSquaresOptim.Dogleg(LeastSquaresOptim.QR());
			iterations = 1024,
		)
		obj!(fvec, fit0.minimizer)
		(fit0.minimizer, fit0.ssr, fit0)
	elseif minimizer == "LBFGS"
		obj1 = function (x)
			fvec = zeros(typeof(x[1]), outlen)
			return sum(obj!(fvec, x) .^ 2)
		end
			cfg = ForwardDiff.GradientConfig(obj1, ct)
		function jac1!(jvec, x)
			jvec[:] = ForwardDiff.gradient(obj1, x, cfg)
			#println("jac ",jvec)
			jvec
		end
		#td = Optim.OnceDifferentiable(obj1, ct; autodiff = :forward)

		res = Optim.optimize(obj1, jac1!, ct, method = Optim.BFGS())
		(Optim.minimizer(res), Optim.minimum(res), res)
	else
		_, fit0 = GaussNewton.optimize!(obj!, ct, fvec, autodiff = autodiff, iscale = 1, NewtStep1st = false)
		(fit0.minimizer, fit0.ssr, fit0)
	end
	# fit = fsolve(obj!, jac!, ct, outlen; show_trace=false, method=:lm, tol=1e-8) #fit.x
	#println(iter, " fit ", fit,  #obj(fit.x))  #fit.f

	ct = @. transform(ct0)
	c[1] = -sum(ct)
	for i ∈ 2:length(c)
		c[i] = ct[i-1] + c[i-1]
	end
	spl = BSplines.Spline(basis, c)
	pp = convert(QuadraticPP{TZ, TX}, spl)
	lsc = ExpSplineCollocation(pp, forward, cLeft, cRight)
	adjustForward(lsc)
	return lsc, FitResult(ssr, iter, ct0, fvec, fit)
end

insert_and_dedup!(v::AbstractArray, x) = (splice!(v, searchsorted(v, x), [x]); v)
