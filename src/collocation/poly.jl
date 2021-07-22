#Stochastic collocation towards a polynomial
using Polynomials
using Roots
import AQFED.Math: normcdf, normpdf, norminv, InverseQuadraticMethod
import AQFED.Black: blackScholesFormula, blackScholesVega
using AQFED.Bachelier
#using MINPACK #slower
using LeastSquaresOptim
#using SCS #slower
export Polynomial, IsotonicCollocation, solveStrike, priceEuropean, density, makeIsotonicCollocation, weightedPrices

struct IsotonicCollocation
    p1::AbstractPolynomial
    p2::AbstractPolynomial
    forward::Number
end


function Polynomial(iso::IsotonicCollocation; minSlope = 1e-4)
    p = integrate(iso.p1^2 + iso.p2^2 + minSlope * iso.forward) #add a minimum slope such that density is not too spiky and collocation not too flat
    theoForward = hermiteIntegral(p)
    p[0] = iso.forward - theoForward
    #fixing the forward is important to avoid the case where C(K=0) < 0.
    return p
end

function solveStrike(p::AbstractPolynomial, strike::Number; useHalley = true)::Number
    if useHalley
        pd = derivative(p)
        pd2 = derivative(p, 2)
        guess = 0.0
        if abs(p[1]) > eps(strike)
            guess = (strike - p[0]) / p[1]
        end
        return find_zero(
            x -> (p(x) - strike, (p(x) - strike) / pd(x), pd2(x) == 0 ? Inf : pd(x) / pd2(x)),
            guess,
            InverseQuadraticMethod(),
            atol = 100 * eps(strike),
            maxevals = 1024,
        )
    else
        pk = p - strike
        r = roots(pk)
        for ri in r
            if abs(imag(ri)) < eps(strike)
                return real(ri)
            end
        end
        throw(DomainError(strike, "no real roots at the given strike"))
    end


end

function priceEuropean(p::AbstractPolynomial, isCall::Bool, strike::Number, forward::Number, discountDf::Number)::Number
    ck = solveStrike(p, strike)
    valuef = hermiteIntegralBounded(p, ck)
    valuek = normcdf(-ck)
    callPrice = valuef - strike * valuek
    putPrice = -(forward - strike) + callPrice
    if isCall
        return callPrice * discountDf
    else
        return putPrice * discountDf
    end
end


function density(p::AbstractPolynomial, strike::Number)::Number
    ck = solveStrike(p, strike)
    return normpdf(ck) / derivative(p)(ck)
end

function rawMoment(p::AbstractPolynomial, moment::Int)::Number
    if moment == 0
        return 1.0
    end
    q = p
    for i = 2:moment
        q *= p
    end
    return hermiteIntegral(q)
end

#return mean, standard dev, skew, kurtosis of the collocation
function stats(p::AbstractPolynomial)
    μ = hermiteIntegral(p)
    μ2 = hermiteIntegral((p - μ)^2)
    μ3 = hermiteIntegral((p - μ)^3)
    μ4 = hermiteIntegral((p - μ)^4)

    skew = μ3 / μ2^1.5
    kurtosis = μ4 / μ2^2
    return μ, sqrt(μ2), skew, kurtosis
end
function hermiteIntegral(p::AbstractPolynomial)::Number
    m0 = 1.0
    nx0 = 0.0
    sum = 0.0
    sum += p[0] * m0
    m1 = nx0
    sum += p[1] * m1
    for i = 2:degree(p)
        m2 = m0 * (i - 1)
        sum += p[i] * m2
        m0 = m1
        m1 = m2
    end
    return sum
end


function hermiteIntegralBounded(p::AbstractPolynomial, ck::Number)::Number
    x0 = ck
    m0 = normcdf(-x0)
    nx0 = normpdf(x0)
    sum = 0.0
    sum += p[0] * m0
    m1 = nx0
    sum += p[1] * m1
    for i = 2:degree(p)
        m2 = m0 * (i - 1) + nx0 * x0^(i - 1)
        sum += p[i] * m2
        m0 = m1
        m1 = m2
    end
    return sum
end


function weightedPrices(
    isCall::Bool,
    strikes::Vector{T},
    vols::Vector{T},
    weights::Vector{T},
    forward::T,
    discountDf::T,
    tte::T;
    vegaFloor = T(1e-3), #vega floored at 1e-3*forward
)::Tuple{Vector{T},Vector{T}} where {T}
    prices = @. blackScholesFormula(isCall, strikes, forward, vols^2 * tte, 1.0, discountDf)
    w = @. weights / max(vegaFloor * forward, blackScholesVega(strikes, forward, vols^2 * tte, 1.0, discountDf, tte))
    return prices, w
end

function makeIsotonicCollocationGuess(
    strikes::Vector{T},
    callPrices::Vector{T},
    weights::Vector{T},
    τ::T,
    forward::T,
    discountDf::T;
    deg = 3, #1 = Bachelier (trivial, smoother if least squares iterations small). otherwise 3 is good.
) where{T}
    if deg >= 3
        strikesf, pricesf, weightsf = filterConvexPrices(strikes, callPrices ./ discountDf, weights, forward)
        strikesf, pif, xif = makeXFromUndiscountedPrices(strikesf, pricesf)
        cubic = Polynomials.fit(xif, strikesf, 3; weights = weightsf)
        if !isCubicMonotone(cubic)
            cubic = LeastSquaresCubicMurrayForward(xif, strikesf, weightsf, forward)
        end
        if deg == 3
            return IsotonicCollocation(cubic, forward)
        else
            return fitMonotonic(xif, strikesf, weightsf, forward, cubic, deg = deg)
        end
    else
        return fitBachelier(strikes, callPrices, weights, τ, forward, discountDf)
    end
end

function makeIsotonicCollocation(
    strikes::Vector{T},
    callPrices::Vector{T},
    weights::Vector{T},
    τ::T,
    forward::T,
    discountDf::T;
    deg = 3, #degree of collocation. 5 is usually best from a stability perspective.
    degGuess = 3, #1 = Bachelier (trivial, smoother if least squares iterations small). otherwise 3 is good.
)::Tuple{IsotonicCollocation,Number} where {T} #return collocation and error measure
    isoc = makeIsotonicCollocationGuess(strikes, callPrices, weights, τ, forward, discountDf, deg = degGuess)
    #optimize towards actual prices
    isoc, m = fit(isoc, strikes, callPrices, weights, forward, discountDf, deg = deg)
    return isoc, m
end

function fitBachelier(strikes, prices, weights, τ, forward, discountDf)
    m = length(strikes)
    i = findfirst(x -> x > forward, strikes)
    if i == nothing
        i = m
    elseif i == 1
        i = 2
    end
    price = (prices[i] * (forward - strikes[i-1]) + prices[i-1] * (strikes[i] - forward)) / (strikes[i] - strikes[i-1])
    strike = forward
    bvol = Bachelier.bachelierImpliedVolatility(price, true, strike, τ, forward, discountDf)
    isoc = IsotonicCollocation(Polynomials.Polynomial([sqrt(bvol * sqrt(τ))]), Polynomials.Polynomial([0.0]), forward)
    return isoc
end

function fit(isoc::IsotonicCollocation, strikes, prices, weights, forward, discountDf; deg = 3)
    q = trunc(Int, (deg + 1) / 2)
    iter = 0
    function obj(c)
        p1 = Polynomials.Polynomial(c[1:q])
        p2 = Polynomials.Polynomial(c[q+1:2*q-1])
        isoc = IsotonicCollocation(p1, p2, forward)
        p = Polynomial(isoc)
        iter += 1
        return @. weights * (priceEuropean(p, true, strikes, forward, discountDf) - prices)
    end
    c0 = zeros(Float64, 2 * q - 1)
    c1 = coeffs(isoc.p1)
    for i = 1:min(q, length(c1))
        c0[i] = c1[i]
    end
    c2 = coeffs(isoc.p2)
    for i = 1:min(q - 1, length(c2))
        c0[q+i] = c2[i]
    end
    for i = 1:length(c0)
        if c0[i] == 0
            c0[i] = eps(forward)  #zero would break automatic differentiation
        end
    end
    fit = optimize(obj, c0, LevenbergMarquardt(); show_trace = false, autodiff = :forward, iterations = deg * 300) #autodiff breaks without Halley. Would need custom
    # function obj!(fvec, x)
    #     fvec[:] = obj(x)
    #     fvec
    # end
    # fit = fsolve(obj!, c0, length(strikes); show_trace=true, method=:lm, tol=1e-10) #fit.x
    #println(iter, " fit ", fit, obj(fit.minimizer))
    c0 = fit.minimizer
    measure = fit.ssr
    return IsotonicCollocation(Polynomials.Polynomial(c0[1:q]), Polynomials.Polynomial(c0[q+1:2*q-1]), forward), measure
end

function IsotonicCollocation(cubic::AbstractPolynomial, forward::Number)
    if degree(cubic) > 3
        throw(DomainError(degree(cubic), "expected a cubic"))
    end
    e = (cubic[1] - cubic[2]^2 / (3 * cubic[3]))
    e = max(e, forward * 1e-6)
    p2 = Polynomials.Polynomial([sqrt(e)])
    p1 = Polynomials.Polynomial([cubic[2] / sqrt(3 * cubic[3]), sqrt(3 * cubic[3])])
    return IsotonicCollocation(p1, p2, forward)
end

function fitMonotonic(xif, strikesf, w1, forward, cubic; deg = 3)
    xmin = -3.0
    if deg > 3 && xif[1] > xmin && strikesf[1] > 0
        #stabilize fit
        xif = [xmin; xif]
        strikesf = [0.0; strikesf]
        w1 = [w1[1]; w1]
    end
    q = trunc(Int, (deg + 1) / 2)
    iter = 0
    function obj(c)
        p1 = Polynomials.Polynomial(c[1:q])
        p2 = Polynomials.Polynomial(c[q+1:2*q-1])
        isoc = IsotonicCollocation(p1, p2, forward)
        p = Polynomial(isoc)
        iter += 1
        return @. w1 * (p(xif) - strikesf)
    end

    isocubic = IsotonicCollocation(cubic, forward)
    c1 = coeffs(isocubic.p1)
    c2 = coeffs(isocubic.p2)
    c0 = zeros(Float64, 2 * q - 1)
    c0[1:min(q, length(c1))] = c1
    c0[q+1:min(2 * q - 1, q + length(c2))] = c2
    xamax = max(abs(xif[1]), abs(xif[end]))
    for i = 1:length(c0)
        if c0[i] == 0
            c0[i] = eps(forward) / (xamax)^(i - 1)  #zero would break automatic differentiation
        end
    end
    fit = optimize(obj, c0, LevenbergMarquardt(); show_trace = false, autodiff = :forward)
    c0 = fit.minimizer
    isoc = IsotonicCollocation(Polynomials.Polynomial(c0[1:q]), Polynomials.Polynomial(c0[q+1:2*q-1]), forward)
    #println(iter, " fitMonotonic ", Polynomial(isoc), " ", fit)
    return isoc
end

function isCubicMonotone(cubic::AbstractPolynomial)::Bool
    c = coeffs(cubic)
    if c[4] == 0 && c[3] == 0
        return c[2] > 0
    else
        return (c[3]^2 - 3 * c[2] * c[4] < 0) && (c[4] > 0)
    end
end


#LeastSquaresMurray retuns cubic coefficients a_0 + a_1 x + a_3 x^3 (a_2=0) of the least squares cubic going through x with weights w
function LeastSquaresCubicMurrayForward(x::Vector{T}, y::Vector{T}, w::Vector{T}, forward::T) where {T}
    rhs = [sum(@. w * x * y), sum(@. w * x^3 * y)]
    m = [sum(@. w * x^2) sum(@. w * x^4); sum(@. w * x^4) sum(@. w * x^6)]
    result = m \ rhs #solve linear system
    return Polynomials.Polynomial([forward, abs(result[1]), zero(T), abs(result[2])])
end
