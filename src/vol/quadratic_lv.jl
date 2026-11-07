using AQFED.Black, GaussNewton
import AQFED.Math: normcdf, normpdf
import PolynomialRoots
export QuadraticLVParams, andersenQuadraticParams, NormalizedQuadraticParams, priceEuropean, impliedVolatilityLPP, impliedVolatilityAB,
    guessQuadraticParamsAB, calibrateQuadraticLV, averageQuadraticParams, TimeChangeAveraging, VarianceMatchAveraging, ATMOptionMatchAveraging

"""
    QuadraticLVParams(α, β, γ)

Quadratic local volatility model of the forward dF = (α + βF + γF²) dW, following Andersen (2011) "Option pricing with
quadratic volatility: a revisit". The volatility is absolute: σ(F) = α/F + β + γF in Black terms at F.
"""
struct QuadraticLVParams{T}
    α::T
    β::T
    γ::T
end

"""
    andersenQuadraticParams(f, σ, q, s)

Parameterization σ(qF + (1-q)f + s(F-f)²/(2f)) of Andersen (2011) with f the forward at time 0, σ the at-the-money volatility,
q the skew and s the convexity.
"""
andersenQuadraticParams(f, σ, q, s) = QuadraticLVParams(σ * ((1 - q) * f + s * f / 2), σ * (q - s), s * σ / (2f))

"Normalized parameterization λ(1 - b + bX + c(X-1)²/2) of the forward X = F/f of Andersen (2011), used for the time-averaging."
struct NormalizedQuadraticParams{T}
    λ::T
    b::T
    c::T
end
QuadraticLVParams(p::NormalizedQuadraticParams) = QuadraticLVParams(p.λ * (1 - p.b + p.c / 2), p.λ * (p.b - p.c), p.c * p.λ / 2)
function NormalizedQuadraticParams(p::QuadraticLVParams)
    λ = p.α + p.β + p.γ
    return NormalizedQuadraticParams(λ, (p.β + 2p.γ) / λ, 2p.γ / λ)
end
#the parameters of X = F/f
_normalizedParams(p::QuadraticLVParams, f) = NormalizedQuadraticParams(QuadraticLVParams(p.α / f, p.β, p.γ * f))

#real roots of γx²+βx+α in increasing order, or the real and imaginary parts of the complex roots.
function _quadraticRoots(p::QuadraticLVParams{T}) where {T}
    disc = p.β^2 - 4p.α * p.γ
    if disc >= 0
        sqrtDisc = sqrt(disc)
        #stable form
        q = -(p.β + copysign(sqrtDisc, p.β)) / 2
        r1, r2 = q / p.γ, p.α / q
        return disc, min(r1, r2), max(r1, r2)
    end
    return disc, -p.β / (2p.γ), sqrt(-disc) / (2abs(p.γ))
end

"""
    priceEuropean(p::QuadraticLVParams, isCall, strike, forward, τ, df=1)

Price of a European option, discounted by `df`, under the quadratic local volatility model, by the closed form formulae of Andersen
(2011) for two real roots or a double root (no absorption is needed), and by an eigenfunction expansion with absorption at 0
for complex roots.
"""
function priceEuropean(p::QuadraticLVParams{T}, isCall::Bool, strike::T, forward::T, τ::T, df::T=one(T)) where {T}
    if abs(p.γ) < eps(T) && p.β > 0
        #displaced diffusion: F + α/β is lognormal of volatility β
        shift = p.α / p.β
        return Black.blackScholesFormula(isCall, strike + shift, forward + shift, p.β^2 * τ, one(T), df)
    elseif abs(p.γ) < sqrt(eps(T))
        p = QuadraticLVParams(p.α, p.β, sqrt(eps(T)))
    end
    disc = p.β^2 - 4p.α * p.γ
    put = if disc > eps(T)
        _, l, u = _quadraticRoots(p)
        t = (p.γ * (u - l))^2 * τ #time change
        if forward < l
            #reflection around (l+u)/2
            _twoRealRootsLeftPut(l, u, l + u - strike, l + u - forward, t) + strike - forward
        elseif u < forward
            _twoRealRootsLeftPut(l, u, strike, forward, t)
        elseif strike <= l
            zero(T)
        elseif strike >= u
            strike - forward
        else
            #l < F < u, Rady (1997)
            sqrtT = sqrt(t)
            ep = (log((forward - l) / (u - forward) * (u - strike) / (strike - l)) + t / 2) / sqrtT
            em = ep - sqrtT
            call = ((u - strike) * (forward - l) * normcdf(ep) - (strike - l) * (u - forward) * normcdf(em)) / (u - l)
            call + strike - forward
        end
    elseif disc > -eps(T)
        t = p.γ^2 * τ
        u = -p.β / (2p.γ)
        if u < forward
            _singleRealRootLeftPut(u, strike, forward, t)
        else
            _singleRealRootLeftPut(u, u - strike, u - forward, t) + strike - forward
        end
    else
        _noRealRootPut(p, strike, forward, τ)
    end
    return df * (isCall ? put + forward - strike : put)
end

#put for u < F and the double root u, the process stays above u
function _singleRealRootLeftPut(u::T, strike::T, forward::T, t::T) where {T}
    strike <= u && return zero(T)
    sqrtT = sqrt(t)
    dp = (1 / (forward - u) - 1 / (strike - u)) / sqrtT
    dm = (-1 / (forward - u) - 1 / (strike - u)) / sqrtT
    return (forward - u) * (strike - u) * sqrtT * (dp * normcdf(dp) + normpdf(dp) - dm * normcdf(dm) - normpdf(dm))
end

#put for l < u < F, the process stays above u
function _twoRealRootsLeftPut(l::T, u::T, strike::T, forward::T, t::T) where {T}
    strike <= u && return zero(T)
    sqrtT = sqrt(t)
    K1 = (strike - u) * (forward - l) / (u - l)
    X1 = (forward - u) * (strike - l) / (u - l)
    K2 = (strike - l) * (forward - l) / (u - l)
    X2 = (forward - u) * (strike - u) / (u - l)
    dp1 = (log(X1 / K1) + t / 2) / sqrtT
    dp2 = (log(X2 / K2) + t / 2) / sqrtT
    return K1 * normcdf(-(dp1 - sqrtT)) - X2 * normcdf(dp2) - X1 * normcdf(-dp1) + K2 * normcdf(dp2 - sqrtT)
end

#put with complex roots a ± ib, absorption at 0: eigenfunction expansion of Andersen (2011)
function _noRealRootPut(p::QuadraticLVParams{T}, strike::T, forward::T, τ::T; nMax::Int=1000) where {T}
    _, a, b = _quadraticRoots(p)
    t = (p.γ * b)^2 * τ
    Zl = atan(-a / b)
    Zu = T(π) / 2
    Z0 = atan((forward - a) / b)
    zL = Zl - Z0
    Af = b * (1 + ((forward - a) / b)^2)
    Al = b * (1 + (a / b)^2)
    Ktilde = (strike - a) / b
    k = atan(Ktilde) - Z0
    width = Zu - Zl
    α1 = T(π)^2 / (2width^2)
    a1 = T(π) * zL / width
    expht = exp(t / 2)
    e1 = zero(T)
    e2 = zero(T)
    for n = 1:nMax
        αn = n^2 * α1
        an = n * a1
        expFactor = exp(-αn * t)
        sinan = -sin(an)
        cnp = (1 + an / zL) * k - an + Z0
        cnm = (1 - an / zL) * k + an + Z0
        denom = an^2 - zL^2
        Inc = zL / 2 * (cos(cnm) / (zL - an) - cos(cnp) / (zL + an) + 2cos(Zl) * an / denom)
        Ins = zL / 2 * (sin(cnm) / (zL - an) - sin(cnp) / (zL + an) + 2sin(Zl) * an / denom)
        e1term = expFactor * sinan * (Ktilde * Inc - Ins)
        e2term = n * T(π) * sinan * expFactor * expht / (αn - one(T) / 2)
        e1 += e1term
        e2 += e2term
        #stop on the decay of the terms, as a term may vanish when sin(an) = 0
        n > 5 && expFactor * (n * expht + abs(Ktilde) + 1) < eps(T) && break
    end
    e1 *= 2 / width
    e2 = sin(Zu - Z0) / sin(width) - e2 / width^2
    return sqrt(b * Af) * expht * e1 + sqrt(Af / Al) * strike * e2
end

"""
    impliedVolatilityLPP(p, order, strike, forward, τ)

Black implied volatility expansion of Lorig, Pagliarani and Pascucci of order 0 to 3 for the quadratic model.
"""
function impliedVolatilityLPP(p::QuadraticLVParams{T}, order::Int, strike::T, forward::T, τ::T) where {T}
    x = log(strike / forward)
    sig0 = p.α / forward + p.β + p.γ * forward
    order == 0 && return sig0
    sig0p = -p.α / forward + p.γ * forward
    sig10 = sig0p * x / 2
    order == 1 && return sig0 + sig10
    a10 = sig0p * sig0
    sig0pp = p.α / forward + p.γ * forward
    a20 = (sig0p^2 + sig0 * sig0pp) / 2
    sig20 = τ * (sig0 * a20 / 6 - a10^2 / (8sig0)) - τ^2 * sig0 * a10^2 / 96 + (4sig0^2 * a20 - 3a10^2) / (12sig0^3) * x^2
    order == 2 && return sig0 + sig10 + sig20
    a30 = (sig0 * sig0p + 3sig0p * sig0pp) / 6
    sig30 = -τ / (192sig0^3) * ((-12 + τ * sig0^2) * a10^3 + 4sig0^2 * (8 + τ * sig0^2) * a10 * a20 - 48sig0^4 * a30) * x +
            (3a10^3 - 5sig0^2 * a10 * a20 + 3sig0^4 * a30) * x^3 / (12sig0^5)
    return sig0 + sig10 + sig20 + sig30
end

"""
    impliedVolatilityAB(p, strike, forward, τ)

Black implied volatility expansion of Andersen and Brotherton-Ratcliffe (2005) at first order in τ, more accurate than LPP.
"""
function impliedVolatilityAB(p::QuadraticLVParams{T}, strike::T, forward::T, τ::T) where {T}
    D(x) = p.α + p.β * x + p.γ * x^2
    DK = D(strike)
    disc = p.β^2 - 4p.α * p.γ
    if abs(strike - forward) < sqrt(eps(T)) * forward
        ω0 = DK / strike
        ω1 = (DK / (24strike^3) - disc / (24strike * DK)) * DK^2
        return ω0 + ω1 * τ
    end
    #∫_K^F dx/D(x)
    invDKf = if disc < -eps(T)
        sqrtDisc = sqrt(-disc)
        2 / sqrtDisc * atan(2p.γ * (forward - strike) / (sqrtDisc + (p.β + 2p.γ * forward) * (p.β + 2p.γ * strike) / sqrtDisc))
    elseif disc > eps(T)
        if abs(p.γ) < eps(T)
            l = -p.α / p.β
            log((forward - l) / (strike - l)) / p.β
        else
            _, l, u = _quadraticRoots(p)
            (log((forward - l) / (forward - u)) - log((strike - l) / (strike - u))) / (p.γ * (l - u))
        end
    else
        l = -p.β / (2p.γ)
        (1 / (l - forward) - 1 / (l - strike)) / p.γ
    end
    ω0 = log(forward / strike) / invDKf
    ω1 = -ω0 / invDKf^2 * log(ω0 * sqrt(forward * strike / (D(forward) * DK)))
    return ω0 + ω1 * τ
end

"""
    guessQuadraticParamsAB(forward, τ, v0, dv0, d2v0)

Quadratic parameters matching the at-the-money level `v0`, slope `dv0` and curvature `d2v0` (towards the strike) of the Black
implied volatility, with the zeroth order of the AB expansion for β and γ, and the first order for α.
"""
function guessQuadraticParamsAB(f::T, τ::T, v0::T, dv0::T, d2v0::T) where {T}
    Dp = 2dv0 * f + v0
    Dpp = 3f * d2v0 + 2dv0^2 * f / v0 + 5dv0
    γ = Dpp / 2
    β = Dp - 2γ * f
    α = f * (v0 - β - γ * f)
    #refine α with the first order term: cubic equation
    factor = 24f^3 / τ
    bgf = β + γ * f
    a = τ / 24 * (3bgf / f^2 + 4γ / f)
    b = 1 / f + τ / 24 * (3bgf^2 / f - (β^2 - 4γ * f * bgf) / f)
    c = β + γ * f + τ / 24 * (bgf^3 - β^2 * bgf) - v0
    #monic cubic α³ + aα² + bα + c = 0 after scaling
    roots = PolynomialRoots.roots([c * factor, b * factor, a * factor, one(T)])
    realRoots = [real(r) for r in roots if abs(imag(r)) < sqrt(eps(T)) * max(1, abs(r))]
    if !isempty(realRoots)
        α = realRoots[argmin(abs.(realRoots .- α))]
    end
    return QuadraticLVParams(α, β, γ)
end

"""
    calibrateQuadraticLV(τ, forward, strikes, vols, weights)

Least-squares calibration of the quadratic local volatility model to the Black implied volatilities `vols`, starting from
`guessQuadraticParamsAB` applied to a parabola fitted in strike.

The unknowns are the values of the local volatility D at three strikes (the lowest strike, the forward and the highest strike)
rather than (α, β, γ): the terms α, βK and γK² nearly cancel each other and the least-squares problem in (α, β, γ) is badly
conditioned, so that the minimizer may stop far from the minimum.

With `useVol=false`, the measure is instead the error in out-of-the-money option prices weighted by the capped inverse Vega
min(1/ν, vegaCap/F) times the square root of the weight: it avoids the implied volatility inversion, and the floor at zero of the
implied volatility when the model price is out of the no-arbitrage bounds, which makes the residual flat in the parameters.
"""
function calibrateQuadraticLV(τ::T, forward::T, strikes::AbstractVector{T}, vols::AbstractVector{T}, weights::AbstractVector{T}=ones(T, length(strikes)); useVol::Bool=true, vegaCap::T=T(1e6)) where {T}
    #weighted parabola in strike, to estimate the at-the-money level, slope and curvature
    A = hcat(ones(T, length(strikes)), strikes .- forward, (strikes .- forward) .^ 2)
    sw = sqrt.(weights)
    coeffs = (A .* sw) \ (vols .* sw)
    guess = guessQuadraticParamsAB(forward, τ, coeffs[1], coeffs[2], 2coeffs[3])
    prices(p) = [priceEuropean(p, strike > forward, strike, forward, τ) for strike in strikes]
    return _calibrateQuadraticNodes(prices, guess, forward, strikes, vols, sw, τ, useVol=useVol, vegaCap=vegaCap)
end

#nodes where the local volatility D is the unknown: lowest strike, forward and highest strike (median if the forward is not inside)
function _quadraticNodes(forward::T, strikes::AbstractVector{T}) where {T}
    lo, hi = extrema(strikes)
    mid = lo < forward < hi ? forward : (lo + hi) / 2
    return [lo, mid, hi]
end
_quadraticNodeValues(p::QuadraticLVParams, nodes) = [p.α + p.β * k + p.γ * k^2 for k in nodes]
function _quadraticFromNodeValues(nodes::AbstractVector{T}, d::AbstractVector) where {T}
    c = hcat(ones(T, 3), nodes, nodes .^ 2) \ d
    return QuadraticLVParams(c[1], c[2], c[3])
end

#implied volatility, zero when the price is out of the no-arbitrage bounds
function _impliedVolatilityOrZero(isCall::Bool, price::T, forward::T, strike::T, τ::T) where {T}
    try
        return Black.impliedVolatility(isCall, price, forward, strike, τ, one(T))
    catch
        return zero(T)
    end
end

#weighted least squares in implied volatilities (or in capped inverse Vega weighted prices) of the out-of-the-money option
#prices returned by prices(p), in the D node values
function _calibrateQuadraticNodes(prices, guess::QuadraticLVParams{T}, forward::T, strikes::AbstractVector{T}, vols::AbstractVector{T}, sw::AbstractVector{T}, τ::T;
    useVol::Bool=true, vegaCap::T=T(1e6)) where {T}
    nodes = _quadraticNodes(forward, strikes)
    marketPrices = [Black.blackScholesFormula(strike > forward, strike, forward, vol^2 * τ, one(T), one(T)) for (strike, vol) in zip(strikes, vols)]
    priceWeights = [w * min(1 / Black.blackScholesVega(strike, forward, vol^2 * τ, one(T), one(T), τ), vegaCap / forward) for (strike, vol, w) in zip(strikes, vols, sw)]
    function residuals!(out, x)
        modelPrices = prices(_quadraticFromNodeValues(nodes, x))
        for (i, strike) in enumerate(strikes)
            out[i] = if useVol
                sw[i] * (_impliedVolatilityOrZero(strike > forward, modelPrices[i], forward, strike, τ) - vols[i])
            else
                priceWeights[i] * (modelPrices[i] - marketPrices[i])
            end
        end
        return out
    end
    x = _quadraticNodeValues(guess, nodes)
    out = zeros(T, length(strikes))
    #the minimum may be very flat (short period of a term-structure): the evaluations are limited
    GaussNewton.optimize!(residuals!, x, out, autodiff=:single, limit=200)
    return _quadraticFromNodeValues(nodes, x)
end

@enum QuadraticAveraging TimeChangeAveraging VarianceMatchAveraging ATMOptionMatchAveraging

"""
    averageQuadraticParams(τ, expiries, params::AbstractVector{<:NormalizedQuadraticParams}; method=VarianceMatchAveraging)

Constant normalized parameters equivalent to the piecewise constant term-structure `params` (valid up to `expiries`) for an
option of maturity τ, by the time-averaging of Andersen (2011). The skew b and convexity c are weighted averages, and λ is the
root mean square (`TimeChangeAveraging`), or matches the variance (`VarianceMatchAveraging`) or the at-the-money option
(`ATMOptionMatchAveraging`).
"""
function averageQuadraticParams(τ::T, expiries::AbstractVector{T}, params::AbstractVector{NormalizedQuadraticParams{T}}; method::QuadraticAveraging=VarianceMatchAveraging) where {T}
    sumWB = sumB = sumWC = sumC = sumQC = sumQB = ν = zero(T)
    lastTime = zero(T)
    for (ti, p) in zip(expiries, params)
        ti = min(ti, τ)
        dt = ti - lastTime
        λ2 = p.λ^2
        wb = λ2 * (ν * dt + λ2 * dt^2 / 2)
        sumWB += wb
        sumB += p.b * wb
        wc = ((ν + λ2 * dt)^3 - ν^3) / 3
        sumWC += wc
        sumC += p.c * wc
        sumQC += p.c * wb
        sumQB += p.b^2 * wb
        ν += λ2 * dt
        lastTime = ti
        ti == τ && break
    end
    b = sumB / sumWB
    c = sumC / sumWC
    λ = if method == VarianceMatchAveraging
        Q = ν + sumQB + sumQC
        sqrt((-1 + sqrt(1 + 2Q * (c + b^2))) / (c + b^2) / τ)
    elseif method == ATMOptionMatchAveraging
        sqrtν = sqrt(ν)
        U = sqrtν - sumC / (2sqrtν^3) + sumQC / (2sqrtν)
        δ = (-(2 + c * ν / 2) + sqrt(4 - 2c * ν - c^2 * ν^2 / 12 + 4c * sqrtν * U)) / (c * sqrtν)
        sqrt(ν / τ) + δ / sqrt(τ)
    else
        sqrt(ν / τ)
    end
    return NormalizedQuadraticParams(λ, b, c)
end

"""
    priceEuropean(expiries, params::AbstractVector{<:QuadraticLVParams}, isCall, strike, forward, τ, df=1; method=VarianceMatchAveraging)

Price for the piecewise constant quadratic model, by time-averaging the normalized parameters.
"""
function priceEuropean(expiries::AbstractVector{T}, params::AbstractVector{QuadraticLVParams{T}}, isCall::Bool, strike::T, forward::T, τ::T, df::T=one(T); method::QuadraticAveraging=VarianceMatchAveraging) where {T}
    average = averageQuadraticParams(τ, expiries, [_normalizedParams(p, forward) for p in params], method=method)
    return forward * priceEuropean(QuadraticLVParams(average), isCall, strike / forward, one(T), τ, df)
end
