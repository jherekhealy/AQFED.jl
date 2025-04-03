#4 moments matching with collocation towards a cubic polynomial
using Polynomials
using GaussNewton
using AQFED.Collocation
import AQFED.Black:blackScholesFormula

struct Collocation4MMBasketPricer
end
function priceEuropean(
    p::Collocation4MMBasketPricer,
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV}
)::T where {T,TV} #not good with positive only weights
    n = length(forward)
    y = log(strike)
    sbar = @. forward * weight
    abar = zeros(T, n)
    rhobar = zeros(T, (n, n))
    sqrtVar = sqrt.(totalVariance)
    for j = 1:size(correlation, 2)
        for i = 1:size(correlation, 1)
            rhobar[i, j] = exp(correlation[i, j] * sqrtVar[i] * sqrtVar[j])
        end
    end
    abar = rhobar' * sbar
    basketForward = one(T)
    sbar /= basketForward


    u1 = zero(T)
    u2 = zero(T)
    u3 = zero(T)
    u4 = zero(T)
    for (i, si) = enumerate(sbar)
        u1 += si
        for (j, sj) = enumerate(sbar)
            temp = si * sj * rhobar[i, j]
            u2 += temp
            for (k, sk) = enumerate(sbar)
                temp3 = temp * sk * rhobar[i, k] * rhobar[j, k]
                u3 += temp3
                for (l, sl) = enumerate(sbar)
                    u4 += temp3 * sl * rhobar[i, l] * rhobar[j, l] * rhobar[k, l]
                end
            end
        end
    end
    u3n = u3 - 3u2 * u1 + 2u1^3
    u4n = u4 - 4u1 * u3 + 6u2 * u1^2 - 3u1^4
    sig = sqrt(u2 - u1^2)
    println(u1, " ", sig, " u3n ", u3n / sig^3, " u4n ", u4n / sig^4)
    #now estimate monotonic cubic with same moments.
    function obj!(fvec::Z, c::AbstractArray{W})::Z where {Z,W}
        b2 = c[1]
        b3 = c[2]
        #b0 = u1 - b2*b3
        #a0 = b0 + b2*b3
        a2 = b2 * b3
        a3 = b3^2 / 3
        a1sq = u2 - u1^2 - 2a2^2 - 6a3^2
        a1 = sqrt(max(zero(W), a1sq))
        #b1sq = a1sq - b3^2 - b2^2

        m3 = a2 * (36 * a1 * a3 + 6a1^2 + 108a3^2 + 8a2^2)
        m4 = 12a3 * (279a3^3 + 108a1 * a3^2 + 186a2^2 * a3 + 21 * a1^2 * a3 + 48 * a1 * a2^2 + 2 * a1^3) + 60a2^2 * (a1^2 + a2^2) + 3a1^4

        fvec[1] = (m3 - u3n) / sig^3
        fvec[2] = (m4 - u4n) / sig^4
        fvec
    end
    fvec = zeros(Float64, 2)
    c0 = [zero(T), zero(T)]
    measure = GaussNewton.optimize!(obj!, c0, fvec)
    println(measure)
    b2 = c0[1]
    b3 = c0[2]
    #  b2 = 0.0; b3=0.0
    a2 = b2 * b3
    a3 = b3^2 / 3
    a1sq = u2 - u1^2 - 2a2^2 - 6a3^2
    a1 = sqrt(max(zero(T), a1sq))

    b1sq = a1 - b3^2 - b2^2
    b0 = u1 - b2 * b3

    p = Polynomials.Polynomial([b0, b1sq + b2^2, b2 * b3, b3^2 / 3])
    println(a1, " ", p, " ", Collocation.stats(p))
    #now collocate to find vanilla option price. 
    #realF = Collocation.priceEuropean(p, true, 0.0, u1, discountFactor)    
    return Collocation.priceEuropean(p, isCall, strike, realF, discountFactor)
end

struct LogCollocation4MMBasketPricer
end
function priceEuropean(
    p::LogCollocation4MMBasketPricer,
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV}
)::T where {T,TV} #not good with positive only weights
    n = length(forward)
    y = log(strike)
    sbar = @. forward * weight
    abar = zeros(T, n)
    rhobar = zeros(T, (n, n))
    sqrtVar = sqrt.(totalVariance)
    for j = 1:size(correlation, 2)
        for i = 1:size(correlation, 1)
            rhobar[i, j] = exp(correlation[i, j] * sqrtVar[i] * sqrtVar[j])
        end
    end
    abar = rhobar' * sbar
    basketForward = one(T)
    sbar /= basketForward


    u1 = zero(T)
    u2 = zero(T)
    u3 = zero(T)
    u4 = zero(T)
    for (i, si) = enumerate(sbar)
        u1 += si
        for (j, sj) = enumerate(sbar)
            temp = si * sj * rhobar[i, j]
            u2 += temp
            for (k, sk) = enumerate(sbar)
                temp3 = temp * sk * rhobar[i, k] * rhobar[j, k]
                u3 += temp3
                for (l, sl) = enumerate(sbar)
                    u4 += temp3 * sl * rhobar[i, l] * rhobar[j, l] * rhobar[k, l]
                end
            end
        end
    end
    u3n = u3 - 3u2 * u1 + 2u1^3
    u4n = u4 - 4u1 * u3 + 6u2 * u1^2 - 3u1^4
    sig = sqrt(u2 - u1^2)
    v2 = log(u2) - 2 * log(u1)

    println(u1, " ", sig, " u3n ", u3n / sig^3, " u4n ", u4n / sig^4)
    esig = exp(v2 / 2)
    #now estimate monotonic cubic with same moments.
    function obj!(fvec::Z, c::AbstractArray{W})::Z where {Z,W}
        #sum a_i e(0.5i^2*sig^2) = m1 ==> a1 = 1 and scale a0=0
        #sum a_i a_j e(0.5(i+j)^2*sig^2) = m2 ==> a2 solution of quadratic a[3]^2*%e^(18.0*s^2)+2*a[2]*a[3]*%e^(12.5*s^2)+2*a[1]*a[3]*%e^(8.0*s^2)+a[2]^2*%e^(8.0*s^2)+2*a[1]*a[2]*%e^(4.5*s^2)+a[1]^2*%e^(2.0*s^2)
        ## isotonic degree 3:   int_0^x (b3*t+b2)^2 + (b1)^2 dt = (b3x+b2)^3 / 3b3 + b1^2 x - b2^3 /3b3 ==> 
        b1 = one(W) * sqrt(u1 / esig)
        b2 = c[1]
        b3 = c[2]
        a = [b1^2 + b2^2, b3 * b2, b3^2 / 3] #a0=0
        m1 = zero(W)
        m2 = zero(W)
        m3 = zero(W)
        for i = 1:3
            m1 += a[i] * esig^(i^2)
            for j = 1:3
                m2 += a[i] * a[j] * esig^((i + j)^2)
                for k = 1:3
                    m3 += a[i] * a[j] * a[k] * esig^((i + j + k)^2)
                end
            end
        end

        #@. a *= u1/m1
        m2 *= (u1 / m1)^2
        m3 *= (u1 / m1)^3
        println("obj m ", m1, " ", m2, " ", m3)
        fvec[1] = (m2 - u2) / sig^2
        fvec[2] = (m3 - u3) / sig^3
        fvec
    end
    #TODO explicit solution for a2 based on u1, u2. What if desn't exist? penalty / add to fvec[2] error in u2?
    # if there are two takes the one with smalles m3 error. Moment matching seems to be often exact.
    #    then imply b1,b2, b3 from a1,a2, a3.
    # would be more stable but the results where GN is successful with 2D method are not to encouraging: better than LN but wrse than SLN, which is much simpler. 
    # => fixed v2 is too much of a constraint, like with smile. possibly add v2 + m4.

    fvec = zeros(Float64, 2)
    c0 = [zero(T), zero(T)]
    measure = GaussNewton.optimize!(obj!, c0, fvec)
    # c0[1] = 0; c0[2]=0
    println(measure)
    b1 = one(T) * sqrt(u1 / esig)
    b2 = c0[1]
    b3 = c0[2]
    a = [b1^2 + b2^2, b3 * b2, b3^2 / 3]
    m1 = a[1] * esig + a[2] * esig^4 + a[3] * esig^9
    @. a *= u1 / m1
    p = Collocation.LogPolynomial(Polynomials.Polynomial([zero(T), a[1], a[2], a[3]]), sqrt(v2))
    println(p, " ", Collocation.stats(p))
    #now collocate to find vanilla option price. 
    #realF = Collocation.priceEuropean(p, true, 0.0, u1, discountFactor)    
    return Collocation.priceEuropean(p, isCall, strike, u1, discountFactor)
end


struct Levy2MMBasketPricer <: BasketPricer
end
function priceEuropean(
    p::Levy2MMBasketPricer,
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV}
)::T where {T,TV}
    n = length(forward)
    y = log(strike)
    sbar = @. forward * weight
    abar = zeros(T, n)
    rhobar = zeros(T, (n, n))
    sqrtVar = sqrt.(totalVariance)
    for j = 1:size(correlation, 2)
        for i = 1:size(correlation, 1)
            rhobar[i, j] = exp(correlation[i, j] * sqrtVar[i] * sqrtVar[j])
        end
    end
    abar = rhobar' * sbar
    basketForward = one(T)
    sbar /= basketForward


    u1 = zero(T)
    u2 = zero(T)

    for (i, si) = enumerate(sbar)
        u1 += si
        for (j, sj) = enumerate(sbar)
            temp = si * sj * rhobar[i, j]
            u2 += temp
        end
    end
    #  println(u1, " ",sig," u3n ",u3n/sig^3," u4n ",u4n/sig^4)
    #now estimate monotonic cubic with same moments.
    m = 2 * log(u1) - log(u2) / 2
    v2 = log(u2) - 2 * log(u1)
    # println(m," ",v2)
    return blackScholesFormula(isCall, strike, exp(m + 0.5 * v2), v2, one(T), discountFactor)
end


struct SLN3MMBasketPricer <: BasketPricer
end
function priceEuropean(
    p::SLN3MMBasketPricer,
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV}
)::T where {T,TV}
    signc = if isCall
        1
    else
        -1
    end
    n = length(forward)
    y = log(strike)
    sbar = @. forward * weight
    abar = zeros(T, n)
    rhobar = zeros(T, (n, n))
    sqrtVar = sqrt.(totalVariance)
    for j = 1:size(correlation, 2)
        for i = 1:size(correlation, 1)
            rhobar[i, j] = exp(correlation[i, j] * sqrtVar[i] * sqrtVar[j])
        end
    end
    abar = rhobar' * sbar
    basketForward = one(T)
    sbar /= basketForward


    u1 = zero(T)
    u2 = zero(T)
    u3 = zero(T)
    for (i, si) = enumerate(sbar)
        u1 += si
        for (j, sj) = enumerate(sbar)
            temp = si * sj * rhobar[i, j]
            u2 += temp
            for (k, sk) = enumerate(sbar)
                temp3 = temp * sk * rhobar[i, k] * rhobar[j, k]
                u3 += temp3
            end
        end
    end
    u3n = u3 - 3u2 * u1 + 2u1^3
    ς = sqrt(u2 - u1^2)
    η = u3n / ς^3
    sqrt3 = (8 + 4η^2 + 4 * sqrt(4η^2 + η^4))^(1 / 3)
    w = sqrt3 / 2 + 2 / sqrt3 - 1
    b = 1 / sqrt(log(w))
    a = b / 2 * log((w) * (w - 1) / ς^2)
    d = sign(η)
    c = d * u1 - exp((1 / (2b) - a) / b)
    if (strike <= c)
        return discountFactor * max(signc * (u1 - strike), 0.0)
    end
    Q = a + b * log((strike - c) / d)
    #ok if shift a is negative (distrib shifted right) but could be problematic otherwise
    #println("a=",a,"; b=",b,"; c=",c,"; d=",d,"; u1=",u1,";") # int_... c+d*exp(z-a / b) * normpdf(z) dz... normpdf(a+b log(X-c)/d) /(Xd) dX

    # putPrice = (strike-c)*normcdf(Q)- d*exp((1-2a*b)/(2b^2))*normcdf(Q-1/b)
    #callPrice = u1 - strike + putPrice
    #  return discountFactor*ifelse(isCall, callPrice, putPrice)
    price = signc * ((u1 - c) * normcdf(-signc * (Q - 1 / b)) - (strike - c) * normcdf(-signc * Q))
    return discountFactor * price
end