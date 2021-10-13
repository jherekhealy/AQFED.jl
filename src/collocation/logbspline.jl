#Stochastic collocation towards a quadratic bspline and the lognormal density
using Roots
import AQFED.Math: normcdf, normpdf, norminv
import AQFED.Black: blackScholesFormula, blackScholesVega, impliedVolatility
using LeastSquaresOptim
using BSplines

struct QuadraticPP{T<:Number,U<:Real}
    a::Vector{T}
    b::Vector{T}
    c::Vector{T}
    x::Vector{U}
    #logx::Vector{T}
    QuadraticPP(T, U, n::Int) = new{T,U}(zeros(T, n), zeros(T, n), zeros(T, n - 1), zeros(U, n))
    QuadraticPP(a::Vector{T}, b::Vector{T}, c::Vector{T}, x::Vector{U}) where {T<:Real,U<:Real} = new{T,U}(a, b, c, x)
end

(p::QuadraticPP{T})(x::T) where {T} = evaluate(p, x)
Base.length(p::QuadraticPP) = length(pp.a)

function evaluate(self::QuadraticPP, z::T) where {T}
    if z <= self.x[1]
        return self.b[1] * (z - self.x[1]) + self.a[1]
    elseif z >= self.x[end]
        rightSlope = self.b[end]
        return rightSlope * (z - self.x[end]) + self.a[end]
    end
    i = searchsortedfirst(self.x, z)  # x[i-1]<z<=x[i]
    if z == self.x[i]
        return self.a[i]
    end
    if i > 1
        i -= 1
    end
    h = z - self.x[i]
    return self.a[i] + h * (self.b[i] + h * (self.c[i]))
end

function evaluateDerivative(self::QuadraticPP, z::T) where {T}
    if z <= self.x[1]
        return self.b[1]
    elseif z >= self.x[end]
        rightSlope = self.b[end]
        return rightSlope
    end
    i = searchsortedfirst(self.x, z)  # x[i-1]<z<=x[i]
    if z == self.x[i]
        return self.a[i]
    end
    if i > 1
        i -= 1
    end
    h = z - self.x[i]
    return self.b[i] + 2h * self.c[i]
end
function convert(::Type{QuadraticPP}, spl::BSplines.Spline)
    t = BSplines.knots(spl.basis)
    n = length(spl.basis) - 1
    T = typeof(spl.coeffs[1])
    a = zeros(T, n)
    b = zeros(T, n)
    c = zeros(T, n - 1)
    x = spl.basis.breakpoints
    α = spl.coeffs
    for i = 1:n
        a[i] = (t[i+2] - t[i+1]) / (t[i+3] - t[i+1]) * (α[i+1] - α[i]) + α[i]
        b[i] = 2 * (α[i+1] - α[i]) / (t[i+3] - t[i+1])
    end
    for i = 1:n-1
        c[i] = ((α[i+2] - α[i+1]) / (t[i+4] - t[i+2]) + (α[i] - α[i+1]) / (t[i+3] - t[i+1])) / (t[i+3] - t[i+2])
    end
    return QuadraticPP(a, b, c, x)
end

function convert(::Type{BSplines.Spline}, pp::QuadraticPP)
    basis = BSplines.BSplineBasis(3, pp.x)
    t = BSplines.knots(basis)
    n = length(pp.x)
    T = typeof(pp.x[1])
    α = zeros(T, n + 1)
    for i = 1:n
        α[i] = pp.a[i] - pp.b[i] / 2 * (t[i+2] - t[i+1])
    end
    α[n+1] = pp.b[n] / 2 * (t[n+3] - t[n+1]) + α[n]
    return BSplines.Spline(basis, α)
end
struct LogBSplineCollocation
    g::QuadraticPP #a bspline, can I express price directly in terms of bspline basis?
    σ::Number
    forward::Number
end

function solvePositiveStrike(c::LogBSplineCollocation, strike::Number)
    pp = c.g
    n = length(pp.x)
    if strike <= pp.a[1] # is zero
        return pp.x[1], 1
    elseif strike > pp.a[n]
        rightSlope = pp.b[n] #linear extrapolation a + b x-xn = strike
        return (strike - pp.a[n]) / rightSlope + pp.x[n], n
    end
    i = searchsortedfirst(pp.a, strike)  # x[i-1]<z<=x[i]
    if strike == pp.a[i]
        return pp.x[i], i
    end
    if i > 1
        i -= 1
    end
    x0 = pp.x[i]
    c = pp.c[i]
    b = pp.b[i]
    a = pp.a[i]
    cc = a + x0 * (-b + x0 * c) - strike
    bb = b - 2 * x0 * c
    aa = c
    if abs(aa) < sqrt(eps(strike))
        return -cc / bb, i
    else
        local ck::Number
        sqrtΔ = sqrt(max(bb^2 - 4 * aa * cc, 0.0))
        allck = ((-bb - sqrtΔ) / (2aa), (-bb + sqrtΔ) / (2aa))
        for cki in allck
            if cki > pp.x[i] - sqrt(eps(strike)) && cki <= pp.x[i+1] + sqrt(eps(strike))
                return cki, i
            end
        end
        println(allck, " ", pp.x[i], " ", pp.x[i+1], " ", aa, " strike ", strike)
        throw(DomainError("strike not found"))
    end
end

function priceEuropean(
    c::LogBSplineCollocation,
    isCall::Bool,
    strike::Number,
    forward::Number,
    discountDf::Number,
)::Number
    ck, ckIndex = solvePositiveStrike(c, strike)
    if ck < 0
        throw(DomainError(ck, "expected a positive value corresponding to strike"))
    end
    logck = log(ck) / c.σ
    valuef = hermiteIntegralBounded(c, ck, logck)
    valuek = normcdf(-logck)
    callPrice = valuef - strike * valuek
    putPrice = -(forward - strike) + callPrice
    if isCall
        return callPrice * discountDf
    else
        return putPrice * discountDf
    end
end


function density(c::LogBSplineCollocation, strike::Number)::Number
    ck, ckIndex = solvePositiveStrike(c, strike)
    if ck < 0
        throw(DomainError(ck, "expected a positive value corresponding to strike"))
    end
    logck = log(ck) / c.σ
    dp = evaluateDerivative(c.g, ck)

    an = normpdf(logck) / (dp * ck * c.σ)
    num =
        (
            -2 * priceEuropean(c, true, strike, 0.0, 1.0) +
            priceEuropean(c, true, strike + strike * 1e-3, 0.0, 1.0) +
            priceEuropean(c, true, strike - strike * 1e-3, 0.0, 1.0)
        ) / (strike * 1e-3)^2
    # println(strike," ", ck, " ", logck," ", dp, " ", an, " ",num)
    return an
end


function adjustForward(lsc::LogBSplineCollocation)
    theoForward = hermiteIntegral(lsc)
    ratio = lsc.forward / theoForward
    lsc.g.a .*= ratio
    lsc.g.b .*= ratio
    lsc.g.c .*= ratio
end

function hermiteIntegral(p::LogBSplineCollocation)::Number
    return hermiteIntegralBounded(p, 0.0, -300.0)
end

function hermiteIntegralBounded(p::LogBSplineCollocation, ck::Number, logck::Number)::Number
    pp = p.g
    n = length(pp.x)
    i = searchsortedfirst(pp.x, ck)  # x[i-1]<z<=x[i]
    if i > length(pp.x)
        i -= 1
    end
    lx = @. log(pp.x) / p.σ
    lx[1] = -300.0
    e1 = exp(p.σ^2 / 2)
    e2 = exp(p.σ^2 * 2)
    integral = 0.0
    if ck == pp.x[i]
    elseif ck < pp.x[n]
        #include logck to x[i] with i-1 coeffs
        x0 = pp.x[i-1]
        lx1 = lx[i]
        x1 = pp.x[i]
        a0 = pp.a[i-1] + x0 * (-pp.b[i-1] + x0 * pp.c[i-1])
        a1 = pp.b[i-1] - 2 * x0 * pp.c[i-1]
        a2 = pp.c[i-1]
        integral +=
            a0 * (normcdf(-logck) - normcdf(-lx1)) +
            a1 * e1 * (normcdf(p.σ - logck) - normcdf(p.σ - lx1)) +
            a2 * e2 * (normcdf(2 * p.σ - logck) - normcdf(2 * p.σ - lx1))
    end
    for j = i+1:n
        x0 = pp.x[j-1]
        lx0 = lx[j-1]
        x1 = pp.x[j]
        lx1 = lx[j]
        a0 = pp.a[j-1] + x0 * (-pp.b[j-1] + x0 * pp.c[j-1])
        a1 = pp.b[j-1] - 2 * x0 * pp.c[j-1]
        a2 = pp.c[j-1]
        integral +=
            a0 * (normcdf(-lx0) - normcdf(-lx1)) +
            a1 * e1 * (normcdf(p.σ - lx0) - normcdf(p.σ - lx1)) +
            a2 * e2 * (normcdf(2 * p.σ - lx0) - normcdf(2 * p.σ - lx1))
    end
    #linear extrapolation
    lx0 = max(logck, lx[n])
    a0 = pp.a[n] - pp.x[n] * pp.b[n] # x-xn * bn + an
    a1 = pp.b[n]
    integral += a0 * normcdf(-lx0) + a1 * e1 * normcdf(p.σ - lx0)
    return integral
end


function makeLogBSplineCollocationGuess(
    strikes::Vector{T},
    callPrices::Vector{T},
    weights::Vector{T},
    τ::T,
    forward::T,
    discountDf::T;
    size = 0,
) where {T}
    b = fitLogBSplineBlack(strikes, callPrices, weights, τ, forward, discountDf, size = size)
end

function makeLogBSplineCollocation(
    strikes::Vector{T},
    callPrices::Vector{T},
    weights::Vector{T},
    τ::T,
    forward::T,
    discountDf::T;
    minSlope = 1e-4,
    penalty = 0.0,
    size = 0,
)::Tuple{LogBSplineCollocation,Number} where {T} #return collocation and error measure
    strikesf, pricesf, weightsf = filterConvexPrices(strikes, callPrices ./ discountDf, weights, forward, tol=minSlope+sqrt(eps(one(minSlope))))
    isoc = makeLogBSplineCollocationGuess(strikesf, pricesf, weightsf, τ, forward, 1.0, size = size)
    #optimize towards actual prices
    isoc, m = fit(isoc, strikes, callPrices, weights, forward, discountDf, minSlope = minSlope, penalty = penalty)
    return isoc, m
end

function fitLogBSplineBlack(strikes, prices, weights, τ, forward, discountDf; size::Int = 0)
    m = length(strikes)
    i = findfirst(x -> x > forward, strikes)
    if i == nothing
        i = m
    elseif i == 1
        i = 2
    end
    price = (prices[i] * (forward - strikes[i-1]) + prices[i-1] * (strikes[i] - forward)) / (strikes[i] - strikes[i-1])
    strike = forward
    vol = impliedVolatility(true, price, forward, strike, τ, discountDf)
    σ = vol * sqrt(τ)
    strikesf, pif, xf = makeLogXFromUndiscountedPrices(strikes, prices, σ)
    if size == 0
        n = length(xf) + 1
        x = zeros(Float64, n)
        for i = 2:n-1
            # x[i] = xf[i-1]
            x[i] = xf[i-1] + (xf[i] - xf[i-1]) / 2
        end
        x[n] =max(xf[end] *1.2,3.0)
        x[1] = 0.0
    else
        n = max(size, 3) + 1
        x = collect(range(0.0, stop = max(xf[end] * 1.2, 3.0), length = n))
    end
    b = forward * exp(-σ^2 / 2) .* ones(Float64, n)

    a = @. b * x
    c = zeros(Float64, n)
    pp = QuadraticPP(a, b, c, x)
    isoc = LogBSplineCollocation(pp, σ, forward)
    return isoc
end

Base.length(p::LogBSplineCollocation) = Base.length(p.g.x)
Base.broadcastable(p::LogBSplineCollocation) = Ref(p)

positiveTransform(x::T) where {T} =  exp(x) # x^2
positiveTransformInverse(x::T) where {T} = log(x) #sqrt(x)
function fit(isoc::LogBSplineCollocation, strikes, prices, weights, forward, discountDf; minSlope = 1e-6, penalty = 0.0)
    iter = 0
    basis = BSplineBasis(3, isoc.g.x)
    t = BSplines.knots(basis)
    c = zeros(Float64, length(basis)) # = length x + 1
    function obj(ct::AbstractArray{T}) where {T}
        c = zeros(T, length(basis)) # = length x + 1
        c[1] = -(t[3] - t[2]) / (t[4] - t[2]) * positiveTransform(ct[1]) # a1 = 0
        for i = 2:length(c)
            c[i] = positiveTransform(ct[i-1]) + c[i-1]
        end
        #spl(0) = 0 and forward scaling
        spl = BSplines.Spline(basis, c)
        pp = convert(QuadraticPP, spl)
        lsc = LogBSplineCollocation(pp, isoc.σ, forward)
        adjustForward(lsc)
        iter += 1
        verr = @. weights * (priceEuropean(lsc, true, strikes, forward, discountDf) - prices)
        if penalty > 0
          vpen = @. (lsc.g.c[1:end] * penalty / lsc.forward)
            # vpen = @. ((1 / lsc.g.b[2:end] - 1 / lsc.g.b[1:end-1]) * penalty*forward)
            return vcat(verr, vpen)
        else
            return verr
        end
    end
    spl = convert(BSplines.Spline, isoc.g)

    ct = zeros(Float64, length(spl.coeffs) - 1)
    for i = 1:length(ct)
        ct[i] = positiveTransformInverse(spl.coeffs[i+1] - spl.coeffs[i])
    end
    fit = optimize(obj, ct, LevenbergMarquardt(); autodiff = :forward, show_trace = false, iterations = 1024)
    println(iter, " fit ", fit, obj(fit.minimizer))
    ct = fit.minimizer
    c[1] = -(t[3] - t[2]) / (t[4] - t[2]) * positiveTransform(ct[1]) # a1 = 0
    for i = 2:length(c)
        c[i] = positiveTransform(ct[i-1]) + c[i-1]
    end
    spl = BSplines.Spline(basis, c)
    pp = convert(QuadraticPP, spl)
    measure = fit.ssr
    lsc = LogBSplineCollocation(pp, isoc.σ, forward)
    adjustForward(lsc)
    return lsc, measure
end
