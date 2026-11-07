import AQFED.Black: blackScholesFormula
import AQFED.TermStructure: CapitalizedDividend, futureValue
import AQFED.Math: normcdf
export ZhangPLNPricer, LaborderePLNPricer, priceEuropean

"""
    ZhangPLNPricer(order=2)

First or second order approximation of Zhang (2011) "Fast Valuation for European options with cash dividends", Wilmott.
"""
struct ZhangPLNPricer
    order::Int
end
ZhangPLNPricer() = ZhangPLNPricer(2)

function priceEuropean(
    p::ZhangPLNPricer,
    isCall::Bool,
    strike::T,
    rawForward::T, #The raw forward to τ (without cash dividends)
    variance::T,
    τ::T,
    discountDf::T, #discount factor to payment date
    dividends::AbstractArray{CapitalizedDividend{T}},
) where {T}
    divs = [(cd.dividend.exDate / τ, futureValue(cd)) for cd in dividends if cd.dividend.exDate <= τ]
    if isempty(divs) || variance <= zero(T)
        forward = rawForward - sum((d[2] for d in divs), init=zero(T))
        return blackScholesFormula(isCall, strike, forward, variance, one(T), discountDf)
    end
    p.order == 1 && return _zhang(isCall, strike, rawForward, variance, discountDf, divs)
    #second order: average of two first order prices with dividends scaled by u and 1/u
    sqrtVar = sqrt(variance)
    d2 = (log(rawForward / strike) - variance / 2) / sqrtVar
    dHat = [d * exp(-sqrtVar * (1 - frac) * d2 - variance * (1 - frac)^2 / 2) for (frac, d) in divs]
    sumD = sum(dHat)
    R = zero(T)
    for i in eachindex(divs), j = 1:i
        a = dHat[i] * dHat[j] * exp(variance * divs[j][1] * (1 - divs[i][1]))
        R += i == j ? a : 2a
    end
    R = sumD != 0 ? R / sumD^2 : one(T)
    u = (1 + R + sqrt((1 + R)^2 - 4)) / 2
    price1 = _zhang(isCall, strike, rawForward, variance, discountDf, [(frac, d * u) for (frac, d) in divs])
    price2 = _zhang(isCall, strike, rawForward, variance, discountDf, [(frac, d / u) for (frac, d) in divs])
    return (price1 + u * price2) / (1 + u)
end

#first order price for the dividends (exDate/τ, future value)
function _zhang(isCall::Bool, strike::T, forward::T, variance::T, discountDf::T, divs) where {T}
    sqrtVar = sqrt(variance)
    s = isCall ? one(T) : -one(T)
    d1 = _zhangD1(strike, forward, variance, divs)
    d2 = d1 - sqrtVar
    sumD = sum(d * normcdf(s * (d1 - sqrtVar * frac)) for (frac, d) in divs)
    return s * discountDf * (forward * normcdf(s * d1) - strike * normcdf(s * d2) - sumD)
end

#solves F - Σ Dᵢ exp(σ√T fᵢ x - σ²T fᵢ²/2) - K exp(σ√T x - σ²T/2) = 0 by Halley's method (decreasing and concave in x).
function _zhangD1(strike::T, forward::T, variance::T, divs; maxIterations::Int=100) where {T}
    sqrtVar = sqrt(variance)
    x = (log(forward / strike) + variance / 2) / sqrtVar
    for (frac, d) in divs #start where each dividend term is below the forward
        x = min(x, (log(forward / d) + variance * frac^2 / 2) / (sqrtVar * frac))
    end
    tol = sqrt(eps(T))
    for _ = 1:maxIterations
        strikeValue = strike * exp(sqrtVar * x - variance / 2)
        f = forward - strikeValue
        df = -sqrtVar * strikeValue
        d2f = -variance * strikeValue
        for (frac, d) in divs
            a = d * exp(sqrtVar * frac * x - variance * frac^2 / 2)
            f -= a
            df -= a * sqrtVar * frac
            d2f -= a * variance * frac^2
        end
        dx = 2f * df / (2df^2 - f * d2f)
        x -= dx
        abs(dx) <= tol * max(one(T), abs(x)) && break
    end
    return x
end

"""
    LaborderePLNPricer()

Approximation of Henry-Labordère (2009), "Calibration of local stochastic volatility models to market smiles", by a
displaced lognormal model whose shift and volatility follow the proportion of cash dividends γ(t) = 1 - Σ_{tᵢ≤t} Dᵢ/F.
Proportional dividends (`isProportional`) are supported, and are then excluded from the raw forward.
"""
struct LaborderePLNPricer end

function priceEuropean(
    p::LaborderePLNPricer,
    isCall::Bool,
    strike::T,
    rawForward::T, #The raw forward to τ (without any dividend)
    variance::T,
    τ::T,
    discountDf::T, #discount factor to payment date
    dividends::AbstractArray{CapitalizedDividend{T}},
) where {T}
    divs = sort([cd for cd in dividends if cd.dividend.exDate <= τ], by=cd -> cd.dividend.exDate)
    if isempty(divs) || variance <= zero(T)
        return blackScholesFormula(isCall, strike, rawForward, variance, one(T), discountDf)
    end
    #in units of the raw forward: A is the (negative) sum of the cash dividends adjusted by the later proportional ones,
    #B the product of the proportional factors, γ = 1 + A/F is constant between the ex-dates and 1 before the first one.
    A = zero(T)
    B = one(T)
    proportionalFactor = one(T)
    γ = one(T)
    tPrev = zero(T)
    G = zero(T) #∫₀ᵗ γ² du
    γG = zero(T) #∫₀ᵗ γ(u) G(u) du
    for cd in divs
        t = cd.dividend.exDate
        dt = t - tPrev
        γG += γ * (G * dt + γ^2 * dt^2 / 2)
        G += γ^2 * dt
        if cd.dividend.isProportional
            B *= 1 - cd.dividend.amount
            proportionalFactor /= 1 - cd.dividend.amount
        else
            A -= futureValue(cd) * proportionalFactor
            γ = 1 + A / rawForward
        end
        tPrev = t
    end
    dt = τ - tPrev
    γG += γ * (G * dt + γ^2 * dt^2 / 2)
    G += γ^2 * dt
    A *= B
    γBar = 2γG / G^2
    strikeBar = (strike - A) / B * γBar + (1 - γBar) * rawForward
    varianceBar = variance * γBar^2 * G / τ
    return blackScholesFormula(isCall, strikeBar, rawForward, varianceBar, one(T), discountDf) / γBar
end
