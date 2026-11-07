export DoubleBarrierPayoff, DoubleBarrierEngine, calculateDoubleTouch, calculateDoubleNoTouch

"""
    DoubleBarrierPayoff(isCall, strike, down, up, rebate, isIn)

Vanilla option knocked out (or in, if `isIn`) when the asset leaves (down, up), continuously monitored.
The `rebate` is paid at the hitting time for a knock-out, at the payment date for a knock-in that is never activated.
"""
struct DoubleBarrierPayoff{T}
    isCall::Bool
    strike::T
    down::T
    up::T
    rebate::T
    isIn::Bool
end

"""
    DoubleBarrierEngine(spot, variance, driftDf, discountDf, discountDfPayment; nTerms=32)

Black-Scholes double barrier options by the method of images (Kunitomo and Ikeda 1992, Sidenius 1998): the density of the
log-asset killed outside (a, b) is a series of Gaussian densities started at x₀+2nw and 2a-x₀+2nw, w = b-a.
"""
struct DoubleBarrierEngine{T}
    spot::T
    variance::T
    driftDf::T
    discountDf::T
    discountDfPayment::T
    nTerms::Int
end
DoubleBarrierEngine(spot::T, variance::T, driftDf::T, discountDf::T, discountDfPayment::T; nTerms::Int=32) where {T} =
    DoubleBarrierEngine(spot, variance, driftDf, discountDf, discountDfPayment, nTerms)

#drift of the log-asset over the whole period
_logDrift(an::DoubleBarrierEngine) = -log(an.driftDf) - an.variance / 2

#sum over the images c, with sign ±, of exp(μ(c-x₀)/v) f(c), where f(c) is an expectation for the log-asset started at c.
function _sumImages(f, an::DoubleBarrierEngine{T}, a, b) where {T}
    x0 = log(an.spot)
    w = b - a
    m = _logDrift(an) / an.variance
    image(c) = (fc = f(c); iszero(fc) ? fc : exp(m * (c - x0)) * fc) #far images: the weight may overflow while f(c) = 0
    value = image(x0) - image(2a - x0)
    for n = 1:an.nTerms
        value += image(x0 + 2n * w) + image(x0 - 2n * w) - image(2a - x0 + 2n * w) - image(2a - x0 - 2n * w)
    end
    return value
end

#E[S 1{lo<ln S<hi}] and E[1{lo<ln S<hi}] for ln S ~ N(c+μT, v)
function _rangeExpectations(an::DoubleBarrierEngine{T}, c, lo, hi) where {T}
    lo >= hi && return zero(T), zero(T)
    sd = sqrt(an.variance)
    μT = _logDrift(an)
    mean = c + μT
    probability = normcdf((mean + an.variance - lo) / sd) - normcdf((mean + an.variance - hi) / sd)
    #the forward of a far image may overflow while the probability is 0
    es = iszero(probability) ? probability : exp(c + μT + an.variance / 2) * probability
    return es, normcdf((mean - lo) / sd) - normcdf((mean - hi) / sd)
end

function calculate(an::DoubleBarrierEngine{T}, payoff::DoubleBarrierPayoff{T}) where {T}
    a, b = log(payoff.down), log(payoff.up)
    logK = log(payoff.strike)
    lo, hi = payoff.isCall ? (max(logK, a), b) : (a, min(logK, b))
    sign = payoff.isCall ? one(T) : -one(T)
    out = an.discountDfPayment * _sumImages(an, a, b) do c
        es, e1 = _rangeExpectations(an, c, lo, hi)
        sign * (es - payoff.strike * e1)
    end
    if payoff.isIn
        vanilla = blackScholesFormula(payoff.isCall, payoff.strike, an.spot, an.variance, an.driftDf, an.discountDfPayment)
        value = vanilla - out
        if payoff.rebate != 0
            value += calculateDoubleNoTouch(an, payoff.down, payoff.up, payoff.rebate)
        end
        return value
    end
    payoff.rebate == 0 && return out
    return out + calculateDoubleTouch(an, payoff.down, payoff.up, payoff.rebate, payoff.rebate)
end

"""
    calculateDoubleNoTouch(an, down, up, payout)

Value of `payout` paid at the payment date if the asset stays within (down, up).
"""
calculateDoubleNoTouch(an::DoubleBarrierEngine{T}, down, up, payout) where {T} =
    payout * an.discountDfPayment * _sumImages(c -> _rangeExpectations(an, c, log(down), log(up))[2], an, log(down), log(up))

"""
    calculateDoubleTouch(an, down, up, rebateDown, rebateUp)

Value of `rebateDown` (respectively `rebateUp`) paid at the time the asset first hits `down` (respectively `up`), when it
happens before the maturity, discounted at the constant rate implied by `discountDf`.
"""
function calculateDoubleTouch(an::DoubleBarrierEngine{T}, down, up, rebateDown, rebateUp) where {T}
    a, b = log(down), log(up)
    x0 = log(an.spot)
    v = an.variance
    sd = sqrt(v)
    w = b - a
    m = _logDrift(an) / v
    r = -log(an.discountDf) / v
    λ = sqrt(m^2 + 2r)
    #∫₀ᵛ exp(-λ²t/2) d/(2πt³)^½ exp(-d²/2t) dt, the discounted density of the first passage at distance d > 0
    _product(e, n) = iszero(n) ? n : e * n
    passage(d) = _product(exp(-λ * d), normcdf((λ * v - d) / sd)) + _product(exp(λ * d), normcdf((-λ * v - d) / sd))
    #exit through the barrier at distance δ from the spot, before the other one: images at distances 2nw+δ and 2nw-δ
    function exitValue(δ)
        value = passage(δ)
        for n = 1:an.nTerms
            value += passage(2n * w + δ) - passage(2n * w - δ)
        end
        return value
    end
    return rebateUp * _product(exp(m * (b - x0)), exitValue(b - x0)) + rebateDown * _product(exp(m * (a - x0)), exitValue(x0 - a))
end
