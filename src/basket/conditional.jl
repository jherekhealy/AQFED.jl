using AQFED.Math
using LinearAlgebra
export ConditionalBasketPricer

#= Conditioning on one Gaussian direction, integrated numerically, and pricing of the conditional basket with another
pricer (Ju, Vorst expansions, Deelstra...). The log-returns X_i, of covariance Σ_ij = ρ_ij σ_i σ_j T, are written as
X = b Z + Y with Z standard normal and Y independent of Z, of covariance Σ - b bᵀ. Conditionally on Z = z, the assets are
still lognormal with forwards F_i exp(b_i z - b_i²/2) and covariance Σ - b bᵀ.

The direction is the one of largest variance of (w_i F_i X_i) orthogonal to the comonotone direction Λ = Σ w_i F_i X_i
(the conditioning variable of Deelstra et al.). The conditional basket then moves mostly along Λ, its assets being nearly
comonotone, which is the favorable case for lognormal approximations. With negatively correlated assets, the basket
along Z behaves like a hyperbolic cosine (bounded below, heavy right tail), which is what lognormal approximations fail
to capture: it is integrated exactly. For two assets, the conditional basket is fully comonotone. =#
struct ConditionalBasketPricer{P<:BasketPricer} <: BasketPricer
    delegate::P
    q::Quadrature #on each interval between the points where the conditional basket forward crosses the strike
    ndev::Float64
    #number of directions integrated out: with several clusters of negatively correlated assets (for example long-short
    #pairs), each cluster contributes a direction along which the basket behaves like a hyperbolic cosine. The directions
    #are integrated one after the other (the conditional basket is priced again with a ConditionalBasketPricer).
    nDirections::Int
end

ConditionalBasketPricer(delegate::BasketPricer; q::Quadrature=GaussLegendre(24), ndev::Float64=7.0, nDirections::Int=1) = ConditionalBasketPricer(delegate, q, ndev, nDirections)

function priceEuropean(
    p::ConditionalBasketPricer,
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
    sqrtVar = sqrt.(totalVariance)
    Σ = correlation .* (sqrtVar * sqrtVar')
    a = weight .* forward
    cΛ = Σ * a
    varΛ = dot(a, cΛ)
    if n < 2 || varΛ <= zero(T)
        return priceEuropean(p.delegate, isCall, strike, discountFactor, spot, forward, totalVariance, weight, correlation)
    end
    D = Diagonal(a)
    eig = eigen(Symmetric(D * (Σ - cΛ * cΛ' ./ varΛ) * D))
    ℓ = D * eig.vectors[:, end]
    varZ = dot(ℓ, Σ * ℓ)
    #nothing to integrate when the variance left besides Λ is negligible: the basket is (nearly) comonotone, and the
    #eigenvector of the (numerically zero) conditional covariance is arbitrary
    if eig.values[end] <= 1e-10 * varΛ || varZ <= sqrt(eps(T)) * varΛ
        return priceEuropean(p.delegate, isCall, strike, discountFactor, spot, forward, totalVariance, weight, correlation)
    end
    b = Σ * ℓ ./ sqrt(varZ)
    Σc = Σ - b * b'
    condVariance = max.(diag(Σc), zero(T))
    condCorrelation = Matrix{TV}(I, n, n)
    for j = 1:n, i = 1:n
        if i != j && condVariance[i] > 0 && condVariance[j] > 0
            condCorrelation[i, j] = clamp(Σc[i, j] / sqrt(condVariance[i] * condVariance[j]), -one(T), one(T))
        end
    end
    sign = isCall ? 1 : -1
    inner = p.nDirections > 1 ? ConditionalBasketPricer(p.delegate, p.q, p.ndev, p.nDirections - 1) : p.delegate
    condForward(z) = @. forward * exp(b * z - b^2 / 2)
    function condPrice(z)
        fz = condForward(z)
        return try
            priceEuropean(inner, isCall, strike, discountFactor, spot, fz, condVariance, weight, condCorrelation)
        catch #degenerate conditional basket: intrinsic value of the forward
            discountFactor * max(sign * (dot(weight, fz) - strike), zero(T))
        end
    end
    #split the integration where the conditional basket forward crosses the strike: the conditional price is then
    #close to the intrinsic value, with a kink as the conditional variance vanishes. The kink is smoothed over a width
    #proportional to the conditional volatility, which is isolated in its own subinterval.
    fwdMinusK(z) = dot(weight, condForward(z)) - strike
    knots = [-p.ndev]
    zs = range(-p.ndev, p.ndev, length=401)
    for i = 1:length(zs)-1
        lo, hi = zs[i], zs[i+1]
        flo = fwdMinusK(lo)
        if flo * fwdMinusK(hi) < 0
            for _ = 1:60
                mid = (lo + hi) / 2
                fmid = fwdMinusK(mid)
                if flo * fmid <= 0
                    hi = mid
                else
                    lo, flo = mid, fmid
                end
            end
            zc = (lo + hi) / 2
            #width of the smoothed kink: conditional log-volatility of the basket over the slope of the log forward
            fz = condForward(zc)
            wf = weight .* fz
            basket = sum(wf)
            slope = abs(dot(wf, b)) / basket
            condVol = sqrt(max(dot(wf, Σc * wf), zero(T))) / basket
            δ = slope > 0 ? 6 * condVol / slope : zero(T)
            if δ > 0 && δ < p.ndev
                append!(knots, (zc - δ, zc, zc + δ))
            else
                push!(knots, zc)
            end
        end
    end
    #when the b_i have both signs, the conditional forward is convex with an interior minimum: if the minimum is just
    #above the strike, the integrand has a smoothed kink there as well
    if any(b .> 0) && any(b .< 0)
        zmin = zero(T)
        for _ = 1:50 #Newton on the derivative of the convex conditional forward
            fz = weight .* condForward(zmin)
            d1, d2 = dot(fz, b), dot(fz, b .^ 2)
            step = d1 / d2
            zmin = clamp(zmin - step, -p.ndev, p.ndev)
            abs(step) < 1e-12 && break
        end
        wf = weight .* condForward(zmin)
        fmin = sum(wf)
        condVol = sqrt(max(dot(wf, Σc * wf), zero(T))) / fmin
        curvature = dot(wf, b .^ 2)
        if fmin > strike && fmin - strike < 6 * condVol * fmin && curvature > 0
            δ = sqrt(2 * 6 * condVol * fmin / curvature)
            append!(knots, (zmin - 2δ, zmin - δ, zmin, zmin + δ, zmin + 2δ))
        end
    end
    push!(knots, p.ndev)
    knots = sort(unique(clamp.(knots, -p.ndev, p.ndev)))
    price = zero(T)
    for i = 1:length(knots)-1
        price += Math.integrate(p.q, z -> normpdf(z) * condPrice(z), knots[i], knots[i+1])
    end
    return price
end
