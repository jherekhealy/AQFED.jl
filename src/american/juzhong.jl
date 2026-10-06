using AQFED.TermStructure
import AQFED.Math: normcdf, normpdf
import AQFED.Black: blackScholesFormula

export JuZhongRepresentation

#Ju and Zhong (1999) "An approximate formula for pricing American options", J. Derivatives:
#the quadratic approximation of MacMillan (1986) and Barone-Adesi-Whaley (1987) with a second-order correction (Exhibit 1).
#The approximation is not symmetric under the put-call transformation: calls use φ=1 directly, as in the paper.
struct JuZhongRepresentation
    isCall::Bool
    model::ConstantBlackModel
    tauMax::Float64
end

JuZhongRepresentation(model::ConstantBlackModel, tauMax::Float64; isCall::Bool=false) = JuZhongRepresentation(isCall, model, tauMax)

function priceAmerican(p::JuZhongRepresentation, K::Float64, S::Float64)::Float64
    return americanJuZhong(p.isCall, S, K, p.tauMax, p.model.r, p.model.q, p.model.vol)
end

europeanBS(isCall, S, K, T, r, q, σ) = blackScholesFormula(isCall, K, S * exp((r - q) * T), σ^2 * T, 1.0, exp(-r * T))

function americanJuZhong(isCall::Bool, S, K, T, r, q, σ; atol=1e-10)
    φ = isCall ? 1 : -1
    if (isCall && (q <= 0 || r < 0)) || (!isCall && r <= 0)
        return europeanBS(isCall, S, K, T, r, q, σ) #no early exercise (single boundary cases only)
    end
    σ2 = σ^2
    sqrtv = σ * sqrt(T)
    β = 2(r - q) / σ2
    isZeroRate = r == 0 #only possible for a call: limit r → 0 of Exhibit 1, with α/h = 2/(σ²T).
    h = -expm1(-r * T)
    αOverh = isZeroRate ? 2 / (σ2 * T) : 2r / (σ2 * h)
    disc = (β - 1)^2 + 4αOverh
    sqrtDisc = sqrt(disc)
    λ = (-(β - 1) + φ * sqrtDisc) / 2
    dfq = exp(-q * T)

    #critical price: φ = φe^{-qT}N(φd1) + λ(φ(S*-K) - VE(S*))/S*, solved by Newton from the Barone-Adesi-Whaley seed.
    λInf = (-(β - 1) + φ * sqrt((β - 1)^2 + 8r / σ2)) / 2
    if isCall
        bound = max(K, r * K / q)
        sStar = if isZeroRate
            bound * (1 + sqrtv)
        else
            sInf = K * λInf / (λInf - 1)
            max(bound, K + (sInf - K) * (-expm1(-((r - q) * T + 2sqrtv) * K / (sInf - K))))
        end
    else
        bound = q > 0 ? min(K, r * K / q) : K
        sInf = K * λInf / (λInf - 1)
        sStar = min(bound, sInf + (K - sInf) * exp(((r - q) * T - 2sqrtv) * K / (K - sInf)))
    end
    for _ = 1:100
        d1 = (log(sStar / K) + (r - q + σ2 / 2) * T) / sqrtv
        dfN = dfq * normcdf(φ * d1)
        premium = φ * (sStar - K) - europeanBS(isCall, sStar, K, T, r, q, σ)
        f = φ * dfN + λ * premium / sStar - φ
        df = dfq * normpdf(d1) / (sStar * sqrtv) + λ * ((φ - φ * dfN) / sStar - premium / sStar^2)
        step = f / df
        sNew = sStar - step
        #stay on the exercise side of the strike
        sStar = φ * (sNew - K) <= 0 ? (sStar + K) / 2 : (sNew <= 0 ? sStar / 2 : sNew)
        abs(step) < atol * K && break
    end
    if φ * (sStar - S) <= 0
        return φ * (S - K)
    end
    d1 = (log(sStar / K) + (r - q + σ2 / 2) * T) / sqrtv
    d2 = d1 - sqrtv
    hA = φ * (sStar - K) - europeanBS(isCall, sStar, K, T, r, q, σ)
    denom = 2λ + β - 1
    b, c = if isZeroRate
        b0 = -2 / (σ2^2 * T^2 * disc)
        c0 = -φ / sqrtDisc * (sStar * normpdf(d1) * dfq / (hA * sqrtv) - φ * 2q * sStar * normcdf(φ * d1) * dfq / (hA * σ2) +
                              2 / (σ2 * T) - 4 / (σ2^2 * T^2 * disc))
        (b0, c0)
    else
        α = 2r / σ2
        λh = -φ * α / (h^2 * sqrtDisc) #λ'(h)
        #∂VE(S*,h)/∂h = ∂VE/∂τ ∂τ/∂h
        dVEdh = sStar * normpdf(d1) * σ * exp((r - q) * T) / (2r * sqrt(T)) - φ * q * sStar * normcdf(φ * d1) * exp((r - q) * T) / r + φ * K * normcdf(φ * d2)
        ((1 - h) * α * λh / (2denom), -(1 - h) * α / denom * (dVEdh / hA + 1 / h + λh / denom))
    end
    lnS = log(S / sStar)
    X = b * lnS^2 + c * lnS
    return europeanBS(isCall, S, K, T, r, q, σ) + hA * (S / sStar)^λ / (1 - X)
end
