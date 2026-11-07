using LinearAlgebra
export QuadraticLVForwardPDE, calibrateQuadraticLVTermStructure

"""
    QuadraticLVForwardPDE(expiries, params, forward, τ; nK=1201, stepsPerYear=400, highestStrike=0, strikeMax)

Call prices of maturity τ under the piecewise constant quadratic local volatility (`params[i]` valid up to `expiries[i]`),
from the forward PDE ∂C/∂T = ½D(K)²∂²C/∂K² on a uniform strike grid [0, strikeMax], by Crank-Nicolson with two Rannacher
steps. The forward is absorbed at zero. The default `strikeMax` is the forward times exp(6σ√τ), σ = D(F)/F of the last period,
or 1.5 times `highestStrike` if larger.
Prices are then given by `priceEuropean(pde, isCall, strike, df=1)`.
"""
struct QuadraticLVForwardPDE{T}
    strikeGrid::Vector{T}
    callPrices::Vector{T}
    forward::T
end

function QuadraticLVForwardPDE(expiries::AbstractVector{T}, params::AbstractVector{QuadraticLVParams{T}}, forward::T, τ::T;
    nK::Int=1201, stepsPerYear::Int=400, highestStrike::T=zero(T), strikeMax::T=max(_defaultStrikeMax(expiries, params, forward, τ), 3highestStrike / 2)) where {T}
    K = collect(range(zero(T), strikeMax, length=nK))
    h = K[2] - K[1]
    C = max.(forward .- K, zero(T))
    work = similar(C)
    t = zero(T)
    nRannacher = 2
    for (ti, p) in zip(expiries, params)
        ti = min(ti, τ)
        ti <= t && continue
        n = max(10, ceil(Int, (ti - t) * stepsPerYear))
        dt = (ti - t) / n
        a = [(p.α + p.β * k + p.γ * k^2)^2 / (2h^2) for k in K]
        #Dirichlet boundaries: C(0) = F and C(strikeMax) = 0 stay constant
        L = Tridiagonal([a[2:end-1]; zero(T)], [zero(T); -2a[2:end-1]; zero(T)], [zero(T); a[2:end-1]])
        implicitHalf = lu(I - (dt / 2) * L)
        explicitHalf = I + (dt / 2) * L
        for _ = 1:n
            if nRannacher > 0
                #one implicit Euler step as two half steps, to damp the kink of the initial payoff
                ldiv!(implicitHalf, C)
                ldiv!(implicitHalf, C)
                nRannacher -= 1
            else
                mul!(work, explicitHalf, C)
                ldiv!(C, implicitHalf, work)
            end
        end
        t = ti
    end
    return QuadraticLVForwardPDE(K, C, forward)
end

function _defaultStrikeMax(expiries, params, forward, τ)
    i = min(searchsortedfirst(expiries, τ), length(params))
    p = params[i]
    σ = (p.α + p.β * forward + p.γ * forward^2) / forward
    return forward * exp(6σ * sqrt(τ))
end

function priceEuropean(pde::QuadraticLVForwardPDE{T}, isCall::Bool, strike::T, df::T=one(T)) where {T}
    K, C = pde.strikeGrid, pde.callPrices
    h = K[2] - K[1]
    #cubic Lagrange interpolation on the four surrounding nodes
    i = clamp(floor(Int, strike / h) + 1, 2, length(K) - 2)
    x = (strike - K[i]) / h
    c0, c1, c2, c3 = C[i-1], C[i], C[i+1], C[i+2]
    call = c1 + x * (c2 - c0) / 2 + x^2 * (c0 - 5c1 / 2 + 2c2 - c3 / 2) + x^3 * (3(c1 - c2) / 2 + (c3 - c0) / 2)
    return df * (isCall ? call : call - pde.forward + strike)
end

"""
    calibrateQuadraticLVTermStructure(expiries, forwards, strikes, vols, weights; nK=1201, stepsPerYear=400, useVol=true, vegaCap=1e6)

Bootstrap of the piecewise constant quadratic local volatility parameters: for each expiry in increasing order, the
parameters of the last period are calibrated to the implied volatilities `vols[i]` at `strikes[i]` (weights `weights[i]`)
with the forward PDE, keeping the parameters of the previous periods. The guess is the calibration with constant parameters
for the first expiry, and the parameters of the previous period otherwise. Returns the parameters and the weighted RMSE in
implied volatilities for each expiry. The PDE strike grid extends to at least 1.5 times the highest strike.
`useVol` and `vegaCap` select the error measure as in `calibrateQuadraticLV`.
"""
function calibrateQuadraticLVTermStructure(expiries::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractVector{<:AbstractVector{T}},
    vols::AbstractVector{<:AbstractVector{T}}, weights::AbstractVector{<:AbstractVector{T}}=[ones(T, length(k)) for k in strikes];
    nK::Int=1201, stepsPerYear::Int=400, useVol::Bool=true, vegaCap::T=T(1e6)) where {T}
    params = QuadraticLVParams{T}[]
    rmses = T[]
    for (i, τ) in enumerate(expiries)
        f, ks, w = forwards[i], strikes[i], weights[i]
        guess = i == 1 ? calibrateQuadraticLV(τ, f, ks, vols[i], w, useVol=useVol, vegaCap=vegaCap) : params[end]
        function prices(p)
            pde = QuadraticLVForwardPDE(expiries[1:i], [params; p], f, τ, nK=nK, stepsPerYear=stepsPerYear, highestStrike=maximum(ks))
            return [priceEuropean(pde, k > f, k) for k in ks]
        end
        p = _calibrateQuadraticNodes(prices, guess, f, ks, vols[i], sqrt.(w), τ, useVol=useVol, vegaCap=vegaCap)
        push!(params, p)
        modelVols = [_impliedVolatilityOrZero(k > f, c, f, k, τ) for (k, c) in zip(ks, prices(p))]
        push!(rmses, sqrt(sum(w .* (modelVols .- vols[i]) .^ 2) / sum(w)))
    end
    return params, rmses
end
