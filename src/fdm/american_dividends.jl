export AmericanFDPricer
using AQFED.TermStructure
using LinearAlgebra
using PPInterpolation
import PPInterpolation: computePP, evaluatePiece, SECOND_DERIVATIVE, C2, Bessel
import AQFED.American: priceAmerican

#Finite difference pricer of vanilla American options under Black-Scholes with discrete dividends, designed for calibration
#(many prices with the same numerical settings): it is a preallocated workspace and the time stepping does not allocate.
#TR-BDF2 on a uniform-in-log(S) grid with the strike in the middle of a cell, central discretization with linear boundary
#conditions, Brennan-Schwartz for the early exercise, dividend jump with a natural Hermite cubic of Bessel slopes,
#natural cubic spline at the spot.
#Not thread-safe: use one pricer per task (`similar`).
mutable struct AmericanFDPricer{T}
    nS::Int #number of space points, one point may be added to place the strike in the middle of a cell
    nT::Int #number of uniform time points, the ex-dividend dates are inserted
    nDev::T #number of standard deviations for the space boundaries
    dividendPolicy::DividendPolicy #for cash dividends larger than the spot
    x::Vector{T}
    t::Vector{T}
    exDividendIndex::Vector{Int}
    v::Vector{T}
    vPrev::Vector{T}
    rhs::Vector{T}
    lb::Vector{T}
    xNew::Vector{T}
    A::Tridiagonal{T,Vector{T}}
    l::Vector{T}
    ip::Vector{T} #inverse pivots
    pp::PPInterpolation.PP{3,T,T,Vector{T},Matrix{T},Vector{T}}
end

function AmericanFDPricer(::Type{T} = Float64; nS::Int = 501, nT::Int = 365, nDev = 4, dividendPolicy::DividendPolicy = Liquidator) where {T}
    nS < 4 && throw(DomainError(nS, "at least 4 space points are required"))
    nT < 2 && throw(DomainError(nT, "at least 2 time points are required"))
    m = nS + 1
    z() = zeros(T, m)
    return AmericanFDPricer{T}(nS, nT, T(nDev), dividendPolicy, z(), zeros(T, nT), zeros(Int, nT), z(), z(), z(), z(), z(),
        Tridiagonal(zeros(T, m - 1), zeros(T, m), zeros(T, m - 1)), z(), z(), PPInterpolation.PP(3, T, T, m))
end

Base.similar(p::AmericanFDPricer{T}) where {T} = AmericanFDPricer(T, nS = p.nS, nT = p.nT, nDev = p.nDev, dividendPolicy = p.dividendPolicy)

#space boundaries around the spot for a total variance and a drift discount factor.
function blackScholesBoundaries(spot::T, variance::T, driftDf::T, nDev::T) where {T}
    b = max(driftDf, one(T) / driftDf)
    l = b * exp(nDev * sqrt(variance))
    return min(T(0.95) * spot, max(spot / 1000, spot / l)), max(T(1.05) * spot, min(1000 * spot, spot * l))
end

#fills x with a geometric grid covering [xMin,xMax] of n points or n+1 points such that `special` is in the middle of a cell.
#Returns the number of points. Unlike ShiftedGrid{LogGrid}, the step is kept and the grid is extended by one point.
function makeLogGridMiddle!(x::AbstractVector{T}, xMin::T, xMax::T, n::Int, special::T) where {T}
    xMin = min(xMin, special)
    xMax = max(xMax, special)
    logMin = log(xMin)
    logMax = log(xMax)
    delta = (logMax - logMin) / (n - 1)
    logSpecial = log(special)
    j0 = trunc(Int, (logSpecial - logMin) / delta)
    lxj0 = logMin + j0 * delta
    d = logSpecial - log((exp(lxj0) + exp(lxj0 + delta)) / 2)
    if d > sqrt(eps(T))
        logMax += d
        logMin += d - delta
        n += 1
    elseif d < -sqrt(eps(T))
        logMax += d + delta
        logMin += d
        n += 1
    end
    growth = exp(delta)
    x[1] = exp(logMin)
    @inbounds for i = 2:n
        x[i] = x[i-1] * growth
    end
    x[n] = exp(logMax)
    return n
end

#fills p.t with nT uniform points on [0,τ] where the ex-dates in (0,τ) are inserted (an ex-date closer than sqrt(eps) to a
#grid point is merged with it). p.exDividendIndex[k] is the index of the dividend of ex-date p.t[k], or 0.
function makeTimeGrid!(p::AmericanFDPricer{T}, τ::T, dividends::AbstractVector{Dividend{T}}) where {T}
    nd = length(dividends)
    if length(p.t) < p.nT + nd
        resize!(p.t, p.nT + nd)
        resize!(p.exDividendIndex, p.nT + nd)
    end
    t = p.t
    exIndex = p.exDividendIndex
    dt = τ / (p.nT - 1)
    sqrtEps = sqrt(eps(T))
    k = 1 #dividends are sorted by ex-date
    while k <= nd && dividends[k].exDate <= zero(T)
        k += 1
    end
    n = 0
    @inbounds for i = 1:p.nT
        ti = dt * (i - 1)
        while k <= nd && dividends[k].exDate < τ && dividends[k].exDate < ti - sqrtEps
            n += 1
            t[n] = dividends[k].exDate
            exIndex[n] = k
            k += 1
        end
        n += 1
        t[n] = ti
        exIndex[n] = 0
        if k <= nd && dividends[k].exDate < τ && dividends[k].exDate < ti + sqrtEps
            exIndex[n] = k
            k += 1
        end
    end
    return n
end

function ensureGridSize!(p::AmericanFDPricer{T}, m::Int) where {T}
    if size(p.A, 1) != m
        p.A = Tridiagonal(zeros(T, m - 1), zeros(T, m), zeros(T, m - 1))
        p.pp = PPInterpolation.PP(3, T, T, m)
    end
end

#price of a vanilla American option of maturity τ and constant volatility σ.
function priceAmerican(p::AmericanFDPricer{T}, market::DiscreteDividendMarket{T}, isCall::Bool, strike::T, τ::T, σ::T) where {T}
    spot, r, μ, dividends = market.spot, market.r, market.μ, market.dividends
    xMin, xMax = blackScholesBoundaries(spot, σ^2 * τ, exp(-μ * τ), p.nDev)
    m = makeLogGridMiddle!(p.x, xMin, xMax, p.nS, strike)
    ensureGridSize!(p, m)
    n = makeTimeGrid!(p, τ, dividends)
    x, v, vPrev, rhs, lb, A = p.x, p.v, p.vPrev, p.rhs, p.lb, p.A
    dl, d, du = A.dl, A.d, A.du
    φ = isCall ? one(T) : -one(T)
    @inbounds for i = 1:m
        lb[i] = max(φ * (x[i] - strike), zero(T))
        v[i] = lb[i]
    end
    α = 2 - sqrt(T(2))
    w0 = 1 / (α * (2 - α))
    w1 = (1 - α)^2 / (α * (2 - α))
    halfσ2 = σ^2 / 2
    lastDt = zero(T)
    sqrtEps = sqrt(eps(T))
    @inbounds for k = n-1:-1:1
        dt = p.t[k+1] - p.t[k]
        if abs(lastDt - dt) > sqrtEps
            #A = I - dt α/2 L is the same for the trapezoidal and the BDF2 stages: one factorization per distinct step.
            dtHalf = dt * α / 2
            for i = 2:m-1
                hi = x[i+1] - x[i]
                him = x[i] - x[i-1]
                cxx = halfσ2 * x[i]^2
                cx = μ * x[i]
                dl[i-1] = (cx * hi - 2cxx) / (him * (him + hi)) * dtHalf
                du[i] = -(cx * him + 2cxx) / (hi * (him + hi)) * dtHalf
                d[i] = 1 + ((2cxx + (him - hi) * cx) / (hi * him) + r) * dtHalf
            end
            cx = μ * x[1] / (x[2] - x[1])
            d[1] = 1 - (-cx - r) * dtHalf
            du[1] = -cx * dtHalf
            cx = μ * x[m] / (x[m] - x[m-1])
            dl[m-1] = cx * dtHalf
            d[m] = 1 - (cx - r) * dtHalf
            decomposeBrennanSchwartz!(p.l, p.ip, A, m, isCall)
            lastDt = dt
        end
        #trapezoidal stage: (2I - A) v
        rhs[1] = (1 - (d[1] - 1)) * v[1] - du[1] * v[2]
        for i = 2:m-1
            rhs[i] = -dl[i-1] * v[i-1] + (1 - (d[i] - 1)) * v[i] - du[i] * v[i+1]
        end
        rhs[m] = -dl[m-1] * v[m-1] + (1 - (d[m] - 1)) * v[m]
        copyto!(vPrev, 1, v, 1, m)
        solveBrennanSchwartz!(v, rhs, p.l, p.ip, A, lb, m, isCall)
        #BDF2 stage
        for i = 1:m
            rhs[i] = w0 * v[i] - w1 * vPrev[i]
        end
        solveBrennanSchwartz!(v, rhs, p.l, p.ip, A, lb, m, isCall)
        if p.exDividendIndex[k] > 0
            jumpDividend!(p, m, dividends[p.exDividendIndex[k]])
        end
    end
    xm = view(x, 1:m)
    computePP(p.pp, xm, view(v, 1:m), SECOND_DERIVATIVE, zero(T), SECOND_DERIVATIVE, zero(T), C2())
    return PPInterpolation.evaluate(p.pp, spot)
end

#V(t⁻,S) = V(t⁺,S-D), floored by the exercise value. Linear extrapolation below the grid.
function jumpDividend!(p::AmericanFDPricer{T}, m::Int, dividend::Dividend{T}) where {T}
    x, v, lb, xNew, pp = p.x, p.v, p.lb, p.xNew, p.pp
    @inbounds for i = 1:m
        xNew[i] = if dividend.isProportional
            x[i] * (1 - dividend.amount)
        else
            xs = x[i] - dividend.amount
            if xs >= zero(T) || p.dividendPolicy == Shift
                xs
            elseif p.dividendPolicy == Survivor
                x[i]
            else
                zero(T)
            end
        end
    end
    computePP(pp, view(x, 1:m), view(v, 1:m), SECOND_DERIVATIVE, zero(T), SECOND_DERIVATIVE, zero(T), Bessel())
    slope = (v[2] - v[1]) / (x[2] - x[1])
    v1 = v[1]
    j = 1
    isSorted = dividend.isProportional || p.dividendPolicy != Survivor
    @inbounds for i = 1:m
        z = xNew[i]
        value = if z < x[1]
            slope * (z - x[1]) + v1
        else
            if isSorted
                while j < m - 1 && z >= x[j+1]
                    j += 1
                end
            else
                j = clamp(searchsortedlast(view(x, 1:m), z), 1, m - 1)
            end
            evaluatePiece(pp, j, z)
        end
        v[i] = max(value, lb[i])
    end
end
