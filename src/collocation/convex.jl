using COSMO
using Convex, SparseArrays, LinearAlgebra

export filterConvexPrices, isArbitrageFree

function isArbitrageFree(strikes::Vector{T}, callPrices::Vector{T}, forward::T)::Tuple{Bool,Int} where {T}
    for (xi, yi) in zip(strikes, callPrices)
        if yi < max(forward - xi, 0)
            return (false, i)
        end
    end

    for i = 2:length(callPrices)-1
        s0 = (callPrices[i] - callPrices[i-1]) / (strikes[i] - strikes[i-1])
        s1 = (callPrices[i] - callPrices[i+1]) / (strikes[i] - strikes[i+1])
        if s0 <= -1
            return (false, i)
        end
        if s0 >= s1
            return (false, i)
        end
        if s1 >= 0
            return (false, i + 1)
        end
    end

    return (true, -1)
end

#create closest set of arbitrage free prices
function filterConvexPrices(
    strikes::Vector{T},
    callPrices::Vector{T}, #undiscounted!
    weights::Vector{T},
    forward::T;
    tol = 1e-8,
)::Tuple{Vector{T},Vector{T},Vector{T}} where {T}
    if isArbitrageFree(strikes, callPrices, forward)[1]
        return strikes, callPrices, weights
    end
    n = length(callPrices)
    z = Variable(n)
    G = spzeros(T, 2 * n, n)
    h = zeros(T, 2 * n)
    for i = 2:n-1
        dym = (strikes[i] - strikes[i-1])
        dy = (strikes[i+1] - strikes[i])
        G[i, i-1] = -1 / dym
        G[i, i] = 1 / dym + 1 / dy
        G[i, i+1] = -1 / dy
    end
    G[1, 1] = 1 / (strikes[2] - strikes[1])
    G[1, 2] = -G[1, 1]
    G[n, n] = 1 / (strikes[n] - strikes[n-1])
    G[n, n-1] = -G[n, n]
    for i = 1:n
        h[i] = -tol
        G[n+i, i] = -1
        h[n+i] = -max(forward - strikes[i], 0) - tol
    end
    h[1] = 1 - tol
    W = spdiagm(weights)
    problem = minimize(square(norm(W * (z - callPrices))), G * z <= h)
    #solve!(problem, () -> SCS.Optimizer(verbose = 0))
    Convex.solve!(problem, () -> COSMO.Optimizer(verbose = false, eps_rel = 1e-8, eps_abs = 1e-8))
    #println("problem status is ", problem.status, " optimal value is ", problem.optval)
    strikesf = strikes
    pricesf = evaluate(z)
    return strikesf, pricesf, weights
end

#transform to X coordinate for collocation
function makeXFromUndiscountedPrices(strikesf::Vector{T}, pricesf::Vector{T}) where {T}
    return makeXFromUndiscountedPrices(strikesf, pricesf, s -> -norminv(-s))
end

function makeXFromUndiscountedPrices(strikesf::Vector{T}, pricesf::Vector{T}, invphi) where {T}
    n = length(strikesf)

    pif = zeros(T, n)
    xif = zeros(T, n)
    for i = 2:n-1
        dxi = strikesf[i+1] - strikesf[i]
        dxim = strikesf[i] - strikesf[i-1]
        dzi = (pricesf[i+1] - pricesf[i]) / dxi
        dzim = (pricesf[i] - pricesf[i-1]) / dxim
        s = (dxim * dzi + dxi * dzim) / (dxim + dxi)
        pif[i] = -s
        xif[i] = invphi(s)
    end
    dzi = (pricesf[n] - pricesf[n-1]) / (strikesf[n] - strikesf[n-1])
    pif[n] = -dzi
    xif[n] = invphi(dzi)
    dzim = -pif[n-2]
    slopeTolerance = 1e-8
    if dzi * dzim < 0 || dzi < dzim || abs(dzi) < slopeTolerance
        pif = pif[1:n-1]
        xif = xif[1:n-1]
        strikesf = strikesf[1:n-1]
    end
    dzim = (pricesf[2] - pricesf[1]) / (strikesf[2] - strikesf[1])
    pif[1] = -dzim
    xif[1] = invphi(dzim)
    if dzim <= -1 + slopeTolerance
        pif = pif[2:end]
        xif = xif[2:end]
        strikesf = strikesf[2:e, d]
    end
    return strikesf, pif, xif
end
