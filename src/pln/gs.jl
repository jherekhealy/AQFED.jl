import AQFED.Black: blackScholesFormula
import AQFED.TermStructure: CapitalizedDividend, futureValue
import AQFED.Math: OneOverSqrt2Pi, normpdf, normcdf
export GocseiSahelPLNPricer, priceEuropean

struct GocseiSahelPLNPricer
end


function priceEuropean(
    p::GocseiSahelPLNPricer,
    isCall::Bool,
    strike::T,
    rawForward::T, #The raw forward to τ (without cash dividends)
    variance::T,
    τ::T,
    discountDf::T, #discount factor to payment date
    dividends::AbstractArray{CapitalizedDividend{T}},
) where {T}
    if length(dividends) == 0
        return blackScholesFormula(isCall, strike, rawForward, variance, one(T), discountDf)
    end
    if length(dividends) == 1 && dividends[1].dividend.exDate < sqrt(eps(T))
        forward = rawForward - futureValue(dividends[1])
        return blackScholesFormula(isCall, strike, forward, variance, one(T), discountDf)
    end
    s = one(T)
    if !isCall
        s = -s
    end
    sqrtVar = sqrt(variance)
    sumDiv = sum(futureValue(cd) for cd in dividends)
    forward = rawForward
    logfk = log(forward / strike)

    d1 = logfk / sqrtVar + sqrtVar / 2
    d2 = d1 - sqrtVar
    Nd1 = normcdf(s * d1)
    Nd2 = normcdf(s * d2)
    nd1 = normpdf(d1)
    nd2 = normpdf(d2)
    mu0 = sqrtVar * forward * nd1 * (Nd1 - Nd2)^3
    mu1 = -(Nd2 * nd1 - Nd1 * nd2) * (Nd2 * nd1 - Nd1 * nd2)
    mu2 = (nd1 - nd2) * (Nd2 * nd1 - Nd1 * nd2)
    mu3 = -(nd1 - nd2) * (nd1 - nd2)
    mu4 = nd1 * (Nd1 - Nd2) * (Nd1 - Nd2)

    K0 = zero(T)
    K1 = mu1 / (2 * mu0) * sumDiv * sumDiv
    K2 = zero(T)
    K3 = zero(T)
    K4 = zero(T)
    for (i, cd) in enumerate(dividends)
        divPVi = futureValue(cd)
        di = d1 - sqrtVar * cd.dividend.exDate / τ
        Ndi = normcdf(s * di)
        K0 += divPVi * (Nd1 - Ndi) / (Nd1 - Nd2)
        K2 += divPVi * Ndi
        for j = 1:i
            divPVj = futureValue(dividends[j])
            tij = min(cd.dividend.exDate, dividends[j].dividend.exDate) / τ
            dij = d1 - sqrtVar * (cd.dividend.exDate + dividends[j].dividend.exDate) / τ
            a = divPVi * divPVj * exp(variance * tij - dij^2 / 2) / sqrt(2pi)
            K4 += a
            if i != j
                K4 += a
            end
        end
    end
    K3 = mu3 / (2 * mu0) * K2 * K2
    K2 *= mu2 / mu0 * sumDiv
    K4 *= mu4 / (2 * mu0)
    if mu0 == zero(T) && mu1 == zero(T)
        K0 = zero(T)
        K1 = zero(T)
        K2 = zero(T)
        K3 = zero(T)
        K4 = zero(T)
    end
    strikeStar = strike + K0 + K1 + K2 + K3 + K4
    forwardStar = forward + (strikeStar - strike) - sumDiv
    if forwardStar <= zero(T)
        if isCall
            return zero(T)
        else
            return strike
        end
    end
    logfk = log(forwardStar / strikeStar)
    d1 = logfk / sqrtVar + sqrtVar / 2
    d2 = d1 - sqrtVar
    Nd1 = normcdf(s * d1)
    Nd2 = normcdf(s * d2)
    price = s * (forwardStar * Nd1 - strikeStar * Nd2)
    return price * discountDf
end