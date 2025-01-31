#Stochastic collocation towards a polynomial and the lognormal density
using Polynomials
using Roots
import AQFED.Math: normcdf, normpdf, norminv
import AQFED.Black: blackScholesFormula, blackScholesVega, impliedVolatility
using AQFED.Bachelier
using LeastSquaresOptim
#phi((x-f)/sig) / sig*(1-Phi(-f/sig))
struct PositivePolynomial{T,U}
    p::AbstractPolynomial{T}
    σ::U
    μ::U
    z::U
end
struct IsotonicPositiveCollocation{T,U}
    p1::AbstractPolynomial{T}
    p2::AbstractPolynomial{T}
    σ::U
    forward::u
end


function PositivePolynomial(iso::IsotonicPositiveCollocation; minSlope = 0.0)
    q = degree(iso.p1) + degree(iso.p2)
    p = integrate(iso.p1^2 + Polynomials.Polynomial([0.0, 1.0])*iso.p2^2 + minSlope * iso.forward)
    z = 1-normcdf(-iso.forward/iso.σ)
    theoForward = hermiteIntegral(PositivePolynomial(p,iso.σ,iso.forward,z))
    c =  iso.forward / theoForward
    p = p*c
    return PositivePolynomial(p,iso.σ,iso.forward,z)
end

(p::PositivePolynomial)(x) = p.p(x)

function solveStrike(p::PositivePolynomial, strike::T; useNumerical=true)::T where {T}
    if useNumerical

        return find_zero(
            x -> p(x) - strike,
            (1e-7,1e5),
            Bisection()
        )
    else
        r = roots(p.p)
        for ri in r
            if (abs(imag(ri)) < eps(strike)) && real(ri) > 0
                return real(ri)
            end
        end
        throw(DomainError(strike, "no real roots at the given strike"))
    end
end

function cdf(p::PositivePolynomial,x) 
    #z = 1-normcdf(-p.f/p.σ)
    (normcdf((x-p.μ)/p.σ)-(1-p.z))/p.z
end
function pdf(p::PositivePolynomial,x) 
    normpdf((x-p.μ)/p.σ)/(p.σ*p.z)
end

function priceEuropean(p::PositivePolynomial, isCall::Bool, strike, forward, discountDf)
    ck = solveStrike(p, strike)
    if ck < 0
        throw(DomainError(ck, "expected a positive value corresponding to strike"))
    end
    valuef = hermiteIntegralBounded(p, ck)
    valuek = cdf(p,-ck)
    callPrice = valuef - strike * valuek
    putPrice = -(forward - strike) + callPrice
    if isCall
        return callPrice * discountDf
    else
        return putPrice * discountDf
    end
end


function density(p::PositivePolynomial, strike)

    ck = solveStrike(p, strike)
    if ck < 0
        throw(DomainError(ck, "expected a positive value corresponding to strike"))
    end
     dp = derivative(p.p)(ck)
    an = pdf(p,ck) *  dp
    #num = (-2*priceEuropean(p, true, strike, 0.0, 1.0) +priceEuropean(p, true, strike+1e-4, 0.0, 1.0) +priceEuropean(p, true, strike-1e-4, 0.0, 1.0) )/(1e-8)
    return an
end


#return mean, standard dev, skew, kurtosis of the collocation
# function stats(p::LogPolynomial)
#     μ = hermiteIntegral(p)
#     μ2 = hermiteIntegral(LogPolynomial((p.p - μ)^2,p.σ))
#     μ3 = hermiteIntegral(LogPolynomial((p.p - μ)^3,p.σ))
#     μ4 = hermiteIntegral(LogPolynomial((p.p - μ)^4,p.σ))

#     skew = μ3 / μ2^1.5
#     kurtosis = μ4 / μ2^2
#     return μ, sqrt(μ2), skew, kurtosis
# end

function hermiteIntegralBounded(p::PositivePolynomial{T}, ck::U)::T where {T,U}
    x0 = ck
    m0 = cdf(p,-x0)
    nx0 = pdf(p,x0)
    sum = 0.0
    sum += p[0] * m0
    m1 = p.σ^2 * nx0 + p.μ * m0
    sum += p[1] * m1
    for i = 2:degree(p)
        m2 =  p.σ^2 *  (m0 * (i - 1) - nx0 * x0^(i - 1)) + p.μ * m1
        sum += p[i] * m2
        m0 = m1
        m1 = m2
    end
    return sum
end


function hermiteIntegral(p::PositivePolynomial{T}) where {T} 
return hermiteIntegralBounded(p,zero(T))
end

##### TODO complete below
function makeLogXFromUndiscountedPrices(strikesf::Vector{T}, pricesf::Vector{T}, σ::T) where {T}
    return makeXFromUndiscountedPrices(strikesf, pricesf, s -> exp(σ*norminv(1+s)))
end

function makeIsotonicLogCollocationGuess(
    strikes::Vector{T},
    callPrices::Vector{T},
    weights::Vector{T},
    τ::T,
    forward::T,
    discountDf::T;
    deg = 3, #1 = Black (trivial, smoother if least squares iterations small). otherwise 3 is good.
) where{T}
    b = fitLogBlack(strikes, callPrices, weights, τ, forward, discountDf)
    if deg >= 3
        strikesf, pricesf, weightsf = filterConvexPrices(strikes, callPrices ./ discountDf, weights, forward)
        strikesf, pif, xif = makeLogXFromUndiscountedPrices(strikesf, pricesf, b.σ)
        # cubic = Polynomials.fit(xif, strikesf, 3; weights = weightsf) #FIXME origin must be = 0
        # if !isCubicMonotone(cubic)
        #     cubic = LeastSquaresCubicMurrayForward(xif, strikesf, weightsf, forward)
        # end
        # if deg == 3
        #     return IsotonicLogCollocation(cubic, b.σ, forward)
        # else
        #     return fitLogMonotonic(xif, strikesf, weightsf, forward, cubic, deg = deg)
        # end
        return fitLogMonotonic(xif, strikesf, weightsf, forward, b, deg = deg)
    else
        return b
    end
end

function makeIsotonicLogCollocation(
    strikes::Vector{T},
    callPrices::Vector{T},
    weights::Vector{T},
    τ::T,
    forward::T,
    discountDf::T;
    deg = 3, #degree of collocation. 5 is usually best from a stability perspective.
    degGuess = 3, #1 = Bachelier (trivial, smoother if least squares iterations small). otherwise 3 is good.
    minSlope = 1e-4, penalty=0.0
)::Tuple{IsotonicLogCollocation,Number} where {T} #return collocation and error measure
    isoc = makeIsotonicLogCollocationGuess(strikes, callPrices, weights, τ, forward, discountDf, deg = degGuess)
    #optimize towards actual prices
    isoc, m = fit(isoc, strikes, callPrices, weights, forward, discountDf, deg = deg, minSlope=minSlope,penalty=penalty)
    return isoc, m
end

function fitLogBlack(strikes, prices, weights, τ, forward, discountDf)
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
    σ = vol*sqrt(τ)
    isoc = IsotonicLogCollocation(Polynomials.Polynomial([sqrt(forward*exp(-σ^2 / 2))]), Polynomials.Polynomial([0.0]), σ, forward)
    return isoc
end

Base.length(p::LogPolynomial) = Base.length(p.p)
Base.iterate(p::LogPolynomial) = Base.iterate(p.p)
Base.iterate(p::LogPolynomial, state) = Base.iterate(p.p, state)
Base.broadcastable(p::LogPolynomial) = Ref(p)
Polynomials.degree(p::LogPolynomial) = Polynomials.degree(p.p)

isotonicDegree(deg::Int) = isodd(deg) ? trunc(Int, (deg-1)/2) : trunc(Int, deg / 2)

function fit(isoc::IsotonicLogCollocation, strikes, prices, weights, forward, discountDf; deg = 3, minSlope = 1e-6, penalty=0.0)
    q = isotonicDegree(deg)
    c0length = 2*q
    if isodd(deg)
        q+=1
        c0length = 2*q-1
    end
    iter = 0
    function obj(ct)
        c = ct
        p1 = Polynomials.Polynomial(c[1:q])
        p2 = Polynomials.Polynomial(c[q+1:c0length])
        isoc = IsotonicLogCollocation(p1, p2, isoc.σ, forward)
        p = LogPolynomial(isoc)
        iter += 1
        if penalty > 0
            ip = hermiteIntegral(LogPolynomial(derivative(p.p,2)^2,p.σ)) #does not seem appropriate here
            pvalue = penalty * ip
            return @. weights * sqrt((priceEuropean(p, true, strikes, forward, discountDf) - prices)^2 + pvalue^2)
        else
            return @. weights * (priceEuropean(p, true, strikes, forward, discountDf) - prices)
        end
    end
    c0 = zeros(Float64, c0length)
    c1 = coeffs(isoc.p1)
    for i = 1:min(q, length(c1))
        c0[i] = c1[i]
    end
    c2 = coeffs(isoc.p2)
    for i = 1:min(c0length-q, length(c2))
        c0[q+i] = c2[i]
    end
    for i = 1:length(c0)
        if c0[i] == 0
            c0[i] = eps(forward)  #zero would break automatic differentiation
        end
    end
    fit = optimize(obj, c0, LevenbergMarquardt(); show_trace = false, autodiff = :forward,iterations = deg * 300) #autodiff breaks without Halley. Would need custom

    println(iter, " fit ", fit, obj(fit.minimizer))
    c0 = fit.minimizer
    measure = fit.ssr
    return IsotonicLogCollocation(Polynomials.Polynomial(c0[1:q]), Polynomials.Polynomial(c0[q+1:c0length]), isoc.σ, forward), measure
end

function IsotonicLogCollocation(cguess::LogPolynomial, forward::Number)
    cubic = cguess.p
    if degree(cubic) == 1
        return IsotonicLogCollocation(Polynomials.Polynomial([sqrt(cguess.p[1])]), Polynomials.Polynomial([0.0]), cguess.σ, forward)
    elseif degree(cubic) > 3
        throw(DomainError(degree(cubic), "expected a cubic"))
    end
    e = (cubic[1] - cubic[2]^2 / (3 * cubic[3]))
    e = max(e, forward * 1e-6)
    p2 = Polynomials.Polynomial([sqrt(e)])
    p1 = Polynomials.Polynomial([cubic[2] / sqrt(3 * cubic[3]), sqrt(3 * cubic[3])])
    return IsotonicLogCollocation(p1, p2, cguess.σ, forward)
end




function fitLogMonotonic(xif, strikesf, w1, forward, cubic; deg = 3)
    q = isotonicDegree(deg)
    c0length = 2*q
    if isodd(deg)
        q+=1
        c0length = 2*q-1
    end
    iter = 0
    function obj(c)
        #c = vcat(0.0, ct)
        p1 = Polynomials.Polynomial(c[1:q])
        p2 = Polynomials.Polynomial(c[q+1:c0length])
        isoc = IsotonicLogCollocation(p1, p2, cubic.σ, forward)
        p = LogPolynomial(isoc)
        iter += 1
        v = @. w1 * (p(xif) - strikesf)
        #println(iter, " ", c, " ", p.p, sum(v.^2))
        return v
    end

    isocubic = typeof(cubic) == IsotonicLogCollocation ? cubic : IsotonicLogCollocation(cubic, forward)
    c1 = coeffs(isocubic.p1)
    c2 = coeffs(isocubic.p2)
    c0 = zeros(Float64,c0length)
    c0[1:min(q, length(c1))] = c1
    c0[q+1:min(c0length, q + length(c2))] = c2
    # println("c0 ",c0," isocub ",isocubic)
    xamax = xif[end]
    for i = 1:length(c0)
        if c0[i] == 0
            c0[i] = eps(forward) / (xamax)^(i - 1)  #zero would break automatic differentiation
        end
    end
    fit = optimize(obj, c0, LevenbergMarquardt(); show_trace = false, autodiff = :forward)
    c0 = fit.minimizer
    isoc = IsotonicLogCollocation(Polynomials.Polynomial(c0[1:q]), Polynomials.Polynomial(c0[q+1:c0length]), cubic.σ, forward)
    println(iter, " fitMonotonic ", LogPolynomial(isoc), " ", fit)
    return isoc
end
