using AQFED.TermStructure
import AQFED.Math: normcdf, BivariateNormcdf
import AQFED.Black: blackScholesFormula

export BjerksundStensland2002Representation, americanCallBS2002, americanCallBS2002Flat

#Bjerksund and Stensland (2002) "Closed form valuation of American options".
#method = :twoStep (two flat boundaries over (0,t) and (t,T) with t = (√5-1)T/2, lower bound),
#         :flat (one flat boundary X_T, lower bound),
#         :proxy (2 twoStep - flat, the extrapolation suggested in the paper, not a bound anymore).
#flatBoundary = :bs2002 uses Eq. (11) of the 2002 paper, :bs1993 uses the original Bjerksund-Stensland (1993) boundary.
struct BjerksundStensland2002Representation
    isCall::Bool
    model::ConstantBlackModel
    tauMax::Float64
    method::Symbol
    flatBoundary::Symbol
end

function BjerksundStensland2002Representation(model::ConstantBlackModel, tauMax::Float64; isCall::Bool=false, method::Symbol=:twoStep, flatBoundary::Symbol=:bs2002)
    if !(method in (:twoStep, :flat, :proxy))
        throw(DomainError(method, "method must be :twoStep, :flat or :proxy"))
    end
    return BjerksundStensland2002Representation(isCall, model, tauMax, method, flatBoundary)
end

function priceAmerican(p::BjerksundStensland2002Representation, K::Float64, S::Float64)::Float64
    r = p.model.r
    b = r - p.model.q
    σ = p.model.vol
    T = p.tauMax
    if !p.isCall #put-call transformation P(S,K,T,r,b,σ) = C(K,S,T,r-b,-b,σ)
        S, K = K, S
        r, b = r - b, -b
    end
    if p.method == :twoStep
        return americanCallBS2002(S, K, T, r, b, σ)
    elseif p.method == :flat
        return americanCallBS2002Flat(S, K, T, r, b, σ, boundary=p.flatBoundary)
    else
        return 2 * americanCallBS2002(S, K, T, r, b, σ) - americanCallBS2002Flat(S, K, T, r, b, σ, boundary=p.flatBoundary)
    end
end

function europeanCallBS(S, K, T, r, b, σ)
    blackScholesFormula(true, K, S * exp(b * T), σ^2 * T, 1.0, exp(-r * T))
end

@inline function bs2002Beta(r, b, σ)
    σ2 = σ^2
    return (0.5 - b / σ2) + sqrt((b / σ2 - 0.5)^2 + 2 * r / σ2)
end

#flat boundary X_T of Eqs. (10)-(13). boundary = :bs1993 uses footnote 7 instead of Eq. (11).
#When bT + 2σ√T < 0 (very negative cost of carry, e.g. a put with large r and long maturity), h > 0 and Eq. (10) falls below B0,
#possibly below the strike, where exercise is never optimal. Unlike the paper, X_T is floored at B0.
function bs2002Boundary(K, T, r, b, σ, β; boundary::Symbol=:bs2002)
    B1 = β / (β - 1) * K
    B0 = max(K, r / (r - b) * K)
    factor = boundary == :bs1993 ? B0 / (B1 - B0) : K^2 / ((B1 - B0) * B0)
    h = -(b * T + 2σ * sqrt(T)) * factor
    return B0 + (B1 - B0) * max(0.0, -expm1(h))
end

#φ(S,T|γ,H,X) of Eq. (7), divided by scale^γ to avoid overflow of S^β.
function bs2002Phi(S, T, γ, H, X, r, b, σ, scale)
    σ2 = σ^2
    λ = -r + γ * b + γ * (γ - 1) * σ2 / 2
    κ = 2b / σ2 + 2γ - 1
    sqrtv = σ * sqrt(T)
    d = -(log(S / H) + (b + (γ - 0.5) * σ2) * T) / sqrtv
    return exp(λ * T + γ * log(S / scale)) * (normcdf(d) - (X / S)^κ * normcdf(d - 2 * log(X / S) / sqrtv))
end

#Ψ(S,T|γ,H,X,x,t) of Proposition 1, divided by scale^γ. Mp and Mm are the bivariate normal cdfs of correlation ±√(t/T).
function bs2002Psi(S, T, γ, H, X, x, t, r, b, σ, scale, Mp, Mm)
    σ2 = σ^2
    λ = -r + γ * b + γ * (γ - 1) * σ2 / 2
    κ = 2b / σ2 + 2γ - 1
    m = b + (γ - 0.5) * σ2
    vt = σ * sqrt(t)
    vT = σ * sqrt(T)
    lnSx = log(S / x)
    lnX2Sx = log(X^2 / (S * x))
    d1 = -(lnSx + m * t) / vt
    d2 = -(lnX2Sx + m * t) / vt
    d3 = -(lnSx - m * t) / vt
    d4 = -(lnX2Sx - m * t) / vt
    D1 = -(log(S / H) + m * T) / vT
    D2 = -(log(X^2 / (S * H)) + m * T) / vT
    D3 = -(log(x^2 / (S * H)) + m * T) / vT
    D4 = -(log(S * x^2 / (H * X^2)) + m * T) / vT
    value = Mp(d1, D1) - (X / S)^κ * Mp(d2, D2) - (x / S)^κ * Mm(d3, D3) + (x / X)^κ * Mm(d4, D4)
    return exp(λ * T + γ * log(S / scale)) * value
end

#ρ = √(t/T) is the same for all options as t = (√5-1)T/2.
const bs2002Mp = BivariateNormcdf(sqrt((sqrt(5.0) - 1) / 2))
const bs2002Mm = BivariateNormcdf(-sqrt((sqrt(5.0) - 1) / 2))

#American call approximation from the two-step boundary (Proposition 1), b is the cost of carry r-q.
function americanCallBS2002(S, K, T, r, b, σ)
    if b >= r
        return europeanCallBS(S, K, T, r, b, σ)
    end
    if r < 0
        throw(DomainError(r, "Bjerksund-Stensland requires r >= 0 when b < r"))
    end
    β = bs2002Beta(r, b, σ)
    t = (sqrt(5.0) - 1) / 2 * T
    X = bs2002Boundary(K, T, r, b, σ, β)
    #X_T is not monotone in T for very negative b: enforce X >= x as assumed by Proposition 1.
    x = min(X, bs2002Boundary(K, T - t, r, b, σ, β))
    if S >= X
        return S - K
    end
    aX = X - K #α(X) X^β
    ax = x - K #α(x) x^β
    return aX * exp(β * log(S / X)) - aX * bs2002Phi(S, t, β, X, X, r, b, σ, X) +
           bs2002Phi(S, t, 1, X, X, r, b, σ, 1.0) - bs2002Phi(S, t, 1, x, X, r, b, σ, 1.0) -
           K * bs2002Phi(S, t, 0, X, X, r, b, σ, 1.0) + K * bs2002Phi(S, t, 0, x, X, r, b, σ, 1.0) +
           ax * bs2002Phi(S, t, β, x, X, r, b, σ, x) - ax * bs2002Psi(S, T, β, x, X, x, t, r, b, σ, x, bs2002Mp, bs2002Mm) +
           bs2002Psi(S, T, 1, x, X, x, t, r, b, σ, 1.0, bs2002Mp, bs2002Mm) - bs2002Psi(S, T, 1, K, X, x, t, r, b, σ, 1.0, bs2002Mp, bs2002Mm) -
           K * bs2002Psi(S, T, 0, x, X, x, t, r, b, σ, 1.0, bs2002Mp, bs2002Mm) + K * bs2002Psi(S, T, 0, K, X, x, t, r, b, σ, 1.0, bs2002Mp, bs2002Mm)
end

#American call approximation from a single flat boundary (Eq. (4)).
function americanCallBS2002Flat(S, K, T, r, b, σ; boundary::Symbol=:bs2002)
    if b >= r
        return europeanCallBS(S, K, T, r, b, σ)
    end
    if r < 0
        throw(DomainError(r, "Bjerksund-Stensland requires r >= 0 when b < r"))
    end
    β = bs2002Beta(r, b, σ)
    X = bs2002Boundary(K, T, r, b, σ, β, boundary=boundary)
    if S >= X
        return S - K
    end
    aX = X - K
    return aX * exp(β * log(S / X)) - aX * bs2002Phi(S, T, β, X, X, r, b, σ, X) +
           bs2002Phi(S, T, 1, X, X, r, b, σ, 1.0) - bs2002Phi(S, T, 1, K, X, r, b, σ, 1.0) -
           K * bs2002Phi(S, T, 0, X, X, r, b, σ, 1.0) + K * bs2002Phi(S, T, 0, K, X, r, b, σ, 1.0)
end
