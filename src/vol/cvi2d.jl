#Two-dimensional multiquadric RBF for the implied variance surface, in the spirit of convex volatility interpolation.
#v(x, τ) with τ = √T and x = k/(σ̄ τ): v = β₀ + β₁x + β₂τ + β₃xτ + Σ αₗ √(1 + (x-xₗ)²/cₓₗ² + (τ-τₗ)²/c_τₗ²).
#The total variance w = T v vanishes at T = 0. At fixed k, ∂w/∂T = v - x/2 ∂ₓv + τ/2 ∂_τv, linear in the parameters.
export Multiquadric2DSurface, calibrateMultiquadricSurface2D

struct Multiquadric2DSurface{T}
    xc::Vector{T} #centers in x
    τc::Vector{T} #centers in τ = √T
    cx::Vector{T}
    cτ::Vector{T}
    σbar::T
    p::Vector{T} #β₀..β₃, α₁..αₘ
    ttes::Vector{T}
    forwards::Vector{T}
end

nparams2D(m::Int) = 4 + m

"rows of v, ∂ₓv, ∂ₓₓv and ∂_τv at (x, τ)"
function basisRows2D!(f, fx, fxx, fτ, xc, τc, cx, cτ, x, τ)
    T = eltype(f)
    f[1:4] = [one(T), x, τ, x * τ]
    fx[1:4] = [zero(T), one(T), zero(T), τ]
    fxx[1:4] .= zero(T)
    fτ[1:4] = [zero(T), zero(T), one(T), x]
    for l in eachindex(xc)
        dx, dτ = x - xc[l], τ - τc[l]
        ψ = sqrt(1 + (dx / cx[l])^2 + (dτ / cτ[l])^2)
        f[4+l] = ψ
        fx[4+l] = dx / (cx[l]^2 * ψ)
        fxx[4+l] = (1 + (dτ / cτ[l])^2) / (cx[l]^2 * ψ^3)
        fτ[4+l] = dτ / (cτ[l]^2 * ψ)
    end
    return f, fx, fxx, fτ
end

"v, ∂ₓv, ∂ₓₓv and ∂_τv at (x, τ) for the parameters p, without allocation"
function values2D(xc, τc, cx, cτ, p, x, τ)
    v = p[1] + p[2] * x + p[3] * τ + p[4] * x * τ
    vx, vxx, vτ = p[2] + p[4] * τ, zero(v), p[3] + p[4] * x
    @inbounds for l in eachindex(xc)
        dx, dτ = x - xc[l], τ - τc[l]
        a = 1 + (dτ / cτ[l])^2
        ψ = sqrt(a + (dx / cx[l])^2)
        αψ = p[4+l] / ψ
        v += p[4+l] * ψ
        vx += αψ * dx / cx[l]^2
        vxx += αψ * a / (cx[l]^2 * ψ^2)
        vτ += αψ * dτ / cτ[l]^2
    end
    return v, vx, vxx, vτ
end

function rows2D(s::Multiquadric2DSurface{T}, x, τ) where {T}
    n = length(s.p)
    f, fx, fxx, fτ = zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n)
    return basisRows2D!(f, fx, fxx, fτ, s.xc, s.τc, s.cx, s.cτ, x, τ)
end

"implied variance at log-moneyness k = log(K/F(t)) and time t (beyond the last expiry, see `totalVarianceDerivatives`)"
varianceByLogmoneyness(s::Multiquadric2DSurface, k, t) = totalVarianceDerivatives(s, k, t)[1] / t

"""total variance w and its derivatives in k at (k, t), and ∂w/∂t at fixed k. Beyond the last expiry T, the total
variance grows at the at-the-money implied variance a of T: w(k, t) = w(k, T) + (t - T) a, flat in implied volatility at
the money, with ∂w/∂t = a > 0 and the same slopes in k as at T (flat volatility at every k would scale the wings
of w beyond Lee's bound)."""
function totalVarianceDerivatives(s::Multiquadric2DSurface, k, t)
    tN = s.ttes[end]
    #beyond the last expiry only (a time rebuilt as (√T)² may exceed T by a rounding error)
    if t > tN * (1 + 1e-12)
        wN, wpN, wppN, _ = totalVarianceDerivatives(s, k, tN)
        a = totalVarianceDerivatives(s, zero(k), tN)[1] / tN
        return wN + (t - tN) * a, wpN, wppN, a
    end
    τ = sqrt(t)
    x = k / (s.σbar * τ)
    v, vx, vxx, vτ = values2D(s.xc, s.τc, s.cx, s.cτ, s.p, x, τ)
    sk = s.σbar * τ
    return t * v, t * vx / sk, t * vxx / sk^2, v - x / 2 * vx + τ / 2 * vτ
end

gatheralG(s::Multiquadric2DSurface, k, t) = (w = totalVarianceDerivatives(s, k, t); gatheralG(k, w[1], w[2], w[3]))
calendarDerivative(s::Multiquadric2DSurface, k, t) = totalVarianceDerivatives(s, k, t)[4]

"forward at time t, by linear interpolation of the log-forward"
function forwardAt(ttes, forwards, t)
    t <= ttes[1] && return forwards[1]^(t / ttes[1])
    t >= ttes[end] && return forwards[end] * (forwards[end] / forwards[end-1])^((t - ttes[end]) / (ttes[end] - ttes[end-1]))
    i = searchsortedlast(ttes, t)
    θ = (t - ttes[i]) / (ttes[i+1] - ttes[i])
    return exp((1 - θ) * log(forwards[i]) + θ * log(forwards[i+1]))
end

function density(s::Multiquadric2DSurface, strike, t)
    F = forwardAt(s.ttes, s.forwards, t)
    k = log(strike / F)
    w, wp, wpp, _ = totalVarianceDerivatives(s, k, t)
    sqrtw = sqrt(w)
    return gatheralG(k, w, wp, wpp) * normpdf(-k / sqrtw - sqrtw / 2) / (strike * sqrtw)
end

"""
    calibrateMultiquadricSurface2D(ttes, forwards, logmoneyness, vols, volSpreads; m=6, kwargs...)

Fit a two-dimensional multiquadric to all the quotes. There are `m` centers per expiry, equidistant in x over the quotes of
the expiry, with the bandwidth cₓ = max(`minBandwidth`, `rangeBandwidth` × range in x) and c_τ = `rangeBandwidthτ` times the range of τ
if given, otherwise `shapeτ` times the local spacing of the expiries in τ, or `shapeτShort` times up to the expiry `shortTimes[1]`,
with a transition linear in log T up to `shortTimes[2]` (a narrower c_τ = 2h_τ fits a smooth surface more closely, but makes ∂w/∂T, hence the local volatility, oscillate in time). The butterfly condition is linearized and iterated as in `calibrateConvexVolatilitySurface`, on
the expiries and `nMid` intermediate times between consecutive expiries; the calendar condition ∂w/∂T ≥ 0 is imposed on the
same points. `exchange` steps add the negative local minima found on finer check grids, which extend to `farCheck` times the
quoted range in x (the constraint grid to `farConstraint` times). `marketPoints` adds the quotes to the constraint
points, `positivity` imposes v ≥ `gMin` σ̄² on them. With `method=:feasible`
(default), the iterates are kept free of arbitrage on the constraint points by pulling them back towards the flat surface
v = σ̄², which is arbitrage-free, and by backtracking; `method=:damped` uses a proximal and elastic SQP instead. `solver` is `:goldfarbidnani`,
`:goldfarbidnanisparse` or `:clarabel`, see `solveQuadraticProgram`; `qpStats`, a vector, collects the statistics of each QP.
"""
function calibrateMultiquadricSurface2D(ttes::AbstractVector{T}, forwards::AbstractVector{T}, logmoneyness::AbstractVector{<:AbstractVector{T}}, vols::AbstractVector{<:AbstractVector{T}}, volSpreads::AbstractVector{<:AbstractVector{T}};
    m::Int=6, rangeBandwidth=0.2, minBandwidth=2.0, shapeτ=4.0, shapeτShort=2.0, shortTimes=(0.25, 1.0), rangeBandwidthτ=nothing, nMid::Int=2, nGrid::Int=60, nCheckMid::Int=10, nCheck::Int=801, wingFactor=2.0,
    farConstraint=100.0, farCheck=3000.0, gMin=1e-4, wingGMin=1e-2, wingRamp=0.25, leeMargin=1e-2, ridge=1e-10, iterations::Int=30, xTol=1e-6, exchange::Int=10, solver::Symbol=:goldfarbidnani, method::Symbol=:feasible, calendarMargin=1e-3, marketPoints::Bool=false, positivity::Bool=false, butterfly::Bool=true, calendar::Bool=true, qpTolerance=1e-6, qpStats=nothing, verbose::Bool=false) where {T}
    nExp = length(ttes)
    σbar = sum(atmAnchor(logmoneyness[j], vols[j]) for j = 1:nExp) / nExp
    #minimum of ∂w/∂T at the calendar points: a fraction of that of the flat surface v = σ̄²
    calendarMin = calendarMargin * σbar^2
    τs = sqrt.(ttes)
    xRanges = [(minimum(logmoneyness[j]) / (σbar * τs[j]), maximum(logmoneyness[j]) / (σbar * τs[j])) for j = 1:nExp]
    #multiple of the spacing in τ for c_τ: shapeτShort up to shortTimes[1], shapeτ from shortTimes[2], linear in log T between
    function shapeAt(t)
        shapeτShort === nothing && return shapeτ
        t <= shortTimes[1] && return shapeτShort
        t >= shortTimes[2] && return shapeτ
        θ = log(t / shortTimes[1]) / log(shortTimes[2] / shortTimes[1])
        return (1 - θ) * shapeτShort + θ * shapeτ
    end
    xc, τc, cx, cτ = T[], T[], T[], T[]
    for j = 1:nExp
        xL, xR = xRanges[j]
        spacingτ = nExp == 1 ? one(T) : (j == 1 ? τs[2] - τs[1] : j == nExp ? τs[nExp] - τs[nExp-1] : (τs[j+1] - τs[j-1]) / 2)
        for xl in range(xL, xR, length=m)
            push!(xc, xl); push!(τc, τs[j])
            push!(cx, max(minBandwidth, rangeBandwidth * (xR - xL)))
            push!(cτ, rangeBandwidthτ === nothing ? shapeAt(ttes[j]) * spacingτ : rangeBandwidthτ * (τs[end] - τs[1]))
        end
    end
    np = nparams2D(length(xc))
    f, fx, fxx, fτ = zeros(T, np), zeros(T, np), zeros(T, np), zeros(T, np)
    rowsAt(x, τ) = basisRows2D!(f, fx, fxx, fτ, xc, τc, cx, cτ, x, τ)
    D = zeros(T, np, np)
    d = zeros(T, np)
    for j = 1:nExp
        N = length(logmoneyness[j])
        for i = 1:N
            rowsAt(logmoneyness[j][i] / (σbar * τs[j]), τs[j])
            wi = one(T) / (2vols[j][i] * volSpreads[j][i])^2 / N
            D .+= 2wi .* (f * f')
            d .+= 2wi * vols[j][i]^2 .* f
        end
    end
    trD = tr(D) / np
    for i = 1:np
        D[i, i] += ridge * trD
    end
    #times of the constraints: expiries and intermediate times (uniform in τ); x ranges interpolated in τ
    function timeGrid(nm)
        ts = T[]
        for j = 1:nExp
            push!(ts, ttes[j])
            j < nExp && append!(ts, [τ^2 for τ in range(τs[j], τs[j+1], length=nm + 2)[2:end-1]])
        end
        return ts
    end
    function xRangeAt(t)
        τ = sqrt(t)
        τ <= τs[1] && return xRanges[1]
        τ >= τs[end] && return xRanges[end]
        j = searchsortedlast(τs, τ)
        θ = (τ - τs[j]) / (τs[j+1] - τs[j])
        return ((1 - θ) * xRanges[j][1] + θ * xRanges[j+1][1], (1 - θ) * xRanges[j][2] + θ * xRanges[j+1][2])
    end
    #margin on g: gMin on the quoted range, rising to wingGMin at wingRamp times the quoted width beyond it (caps the local
    #volatility ∂T w / g in the extrapolated wings)
    function gTarget(x, τ)
        wingGMin === nothing && return T(gMin)
        xL, xR = xRangeAt(τ^2)
        excess = x < xL ? (xL - x) / (xR - xL) : x > xR ? (x - xR) / (xR - xL) : zero(T)
        return gMin + (wingGMin - gMin) * min(one(T), excess / wingRamp)
    end
    function xGrid(t, n, factor, nFar, farMax)
        xL, xR = xRangeAt(t)
        uL, uR = asinh(xL * factor / 2), asinh(xR * factor / 2)
        grid = [2sinh(u) for u in range(uL, uR, length=n)]
        far = [g * x for g in exp.(range(log(factor), log(farMax), length=nFar)) for x in (xL, xR)]
        return vcat(grid, far)
    end
    constraintPoints = [(x, sqrt(t)) for t in timeGrid(nMid) for x in xGrid(t, nGrid, 1 + wingFactor, 20, farConstraint)]
    if marketPoints
        for j = 1:nExp, y in logmoneyness[j]
            push!(constraintPoints, (y / (σbar * τs[j]), τs[j]))
        end
    end
    #check grid, by time (x sorted): for the exchange and for the minimum of g
    checkByTime = [(sqrt(t), sort(xGrid(t, nCheck, 3.0, nCheck ÷ 4, farCheck))) for t in timeGrid(nCheckMid)]
    calendarPoints = copy(constraintPoints)
    rows = Vector{Vector{T}}()
    rhs = T[]
    #positivity
    for (x, τ) in constraintPoints
        positivity || break
        rowsAt(x, τ)
        push!(rows, copy(f)); push!(rhs, gMin * σbar^2)
    end
    #wings: slopes in x of the asymptotes are linear in τ: monotone wings and the asymptotic calendar condition at both ends
    #of the τ range; Lee's bound on the total variance slope τ b(τ)/σ̄ on all the constraint times
    for τ in (τs[1], τs[end])
        slopeR = vcat([zero(T), one(T), zero(T), τ], one(T) ./ cx)
        slopeL = vcat([zero(T), one(T), zero(T), τ], -one(T) ./ cx)
        push!(rows, slopeR); push!(rhs, zero(T))
        push!(rows, -slopeL); push!(rhs, zero(T))
        if calendar
            #d/dτ [τ b(τ)] ≥ 0 on the right, ≤ 0 on the left
            push!(rows, vcat([zero(T), one(T), zero(T), 2τ], one(T) ./ cx)); push!(rhs, zero(T))
            push!(rows, -vcat([zero(T), one(T), zero(T), 2τ], -one(T) ./ cx)); push!(rhs, zero(T))
        end
    end
    for t in timeGrid(nMid)
        τ = sqrt(t)
        push!(rows, -τ / σbar .* vcat([zero(T), one(T), zero(T), τ], one(T) ./ cx)); push!(rhs, -(2 - leeMargin))
        push!(rows, τ / σbar .* vcat([zero(T), one(T), zero(T), τ], -one(T) ./ cx)); push!(rhs, -(2 - leeMargin))
    end
    function calendarRow(x, τ)
        rowsAt(x, τ)
        return f .- x / 2 .* fx .+ τ / 2 .* fτ
    end
    function solve(rows, rhs)
        return solveQuadraticProgram(solver, D, d, reduce(hcat, rows), rhs, tolerance=qpTolerance, stats=qpStats)
    end
    makeSurface(p) = Multiquadric2DSurface(xc, τc, cx, cτ, σbar, p, collect(ttes), collect(forwards))
    gAt(s, x, τ) = gatheralG(s, x * σbar * τ, τ^2)
    minGOf(s) = minimum(minimum(x -> gAt(s, x, τ), xs) for (τ, xs) in checkByTime)
    #min g on the constraint points: the feasible SQP history (the check grid is searched by the exchange)
    minGConstraint(s) = minimum(gAt(s, x, τ) for (x, τ) in constraintPoints)
    function runSQP()
        fixedRows, fixedRhs = copy(rows), copy(rhs)
        if calendar
            for (x, τ) in calendarPoints
                push!(fixedRows, calendarRow(x, τ)); push!(fixedRhs, calendarMin)
            end
        end
        p, crval = solve(fixedRows, fixedRhs)
        s = makeSurface(p)
        minG = [minGOf(s)]
        best = (s, crval)
        bestFeasible = !butterfly || minG[end] >= 0
        local it = 0
        if butterfly
            #damped (proximal) and elastic sequential linearization: the QP minimizes the objective plus
            #μ/2 (p-pRef)ᵀD(p-pRef) plus ρ s, where the slack s ≥ 0 relaxes all the linearized butterfly constraints
            objectiveOf(q) = -dot(d, q) + dot(q, D * q) / 2
            ρ = 1e3 * (abs(objectiveOf(p)) + one(T))
            gConstraint(q) = (sq = makeSurface(q); minimum(gAt(sq, x, τ) - gTarget(x, τ) for (x, τ) in constraintPoints))
            merit(q) = objectiveOf(q) + ρ * max(zero(T), -gConstraint(q))
            μ = one(T)
            meritRef = merit(p)
            for iteration = 1:iterations
                gRows, gRhs = Vector{Vector{T}}(), T[]
                for (x, τ) in constraintPoints
                    t = τ^2
                    k = x * σbar * τ
                    sk = σbar * τ
                    rowsAt(x, τ)
                    w, wp, wpp = t * dot(f, p), t * dot(fx, p) / sk, t * dot(fxx, p) / sk^2
                    a = 1 - k * wp / (2w)
                    gRef = gatheralG(k, w, wp, wpp)
                    gw = a * k * wp / w^2 + wp^2 / (4w^2)
                    gwp = -a * k / w - wp / 2 * (1 / w + one(T) / 4)
                    push!(gRows, vcat(t .* (gw .* f .+ gwp .* fx ./ sk .+ fxx ./ (2sk^2)), one(T)))
                    push!(gRhs, gTarget(x, τ) - gRef + gw * w + gwp * wp + wpp / 2)
                end
                allRows = vcat([vcat(r, zero(T)) for r in fixedRows], gRows, [vcat(zeros(T, np), one(T))])
                allRhs = vcat(fixedRhs, gRhs, zero(T))
                local newP
                accepted = false
                for attempt = 1:8
                    Dμ = zeros(T, np + 1, np + 1)
                    Dμ[1:np, 1:np] = (1 + μ) .* D
                    Dμ[np+1, np+1] = ridge * trD
                    dμ = vcat(d .+ μ .* (D * p), -ρ)
                    sol = try
                        solveQuadraticProgram(solver, Dμ, dμ, reduce(hcat, allRows), allRhs, tolerance=qpTolerance, stats=qpStats)[1]
                    catch e
                        verbose && println("iteration ", iteration, " failed: ", e)
                        nothing
                    end
                    sol === nothing && break
                    newP = sol[1:np]
                    if merit(newP) <= meritRef + 1e-12 * abs(meritRef)
                        accepted = true
                        μ = max(μ / 4, T(1e-4))
                        break
                    end
                    μ *= 4
                end
                accepted || break
                Δx = maximum(abs, newP - p) / maximum(abs, newP)
                p = newP
                meritRef = merit(p)
                crval = objectiveOf(p)
                it = iteration
                s = makeSurface(p)
                push!(minG, minGOf(s))
                verbose && println("iteration ", iteration, " min g ", minG[end], " objective ", crval, " μ ", μ, " Δx ", Δx)
                feasible = minG[end] >= 0
                if feasible && (!bestFeasible || crval < best[2])
                    best = (s, crval)
                    bestFeasible = true
                elseif !bestFeasible
                    best = (s, crval)
                end
                Δx < xTol && break
            end
        end
        return best[1], it, minG
    end
    #feasible SQP: the flat surface v = σ̄² is free of arbitrage; every iterate is kept feasible on the constraint points
    #(linear constraints exactly, g ≥ gMin/2) by pulling back towards it, and by backtracking along the QP steps
    pFlat = vcat(σbar^2, zeros(T, np - 1))
    objectiveOf(q) = -dot(d, q) + dot(q, D * q) / 2
    #smallest g / target over the constraint points (≥ 1 when the margins hold)
    gConstraintMin(q) = (sq = makeSurface(q); minimum(gAt(sq, x, τ) / gTarget(x, τ) for (x, τ) in constraintPoints))
    function linearRows()
        fixedRows, fixedRhs = copy(rows), copy(rhs)
        if calendar
            for (x, τ) in calendarPoints
                push!(fixedRows, calendarRow(x, τ)); push!(fixedRhs, calendarMin)
            end
        end
        return fixedRows, fixedRhs
    end
    "largest s in [0,1] such that pFlat + s (q - pFlat) satisfies the linear constraints and g ≥ gMin"
    function pullBack(q, fixedRows, fixedRhs)
        sMax = one(T)
        for (r, b) in zip(fixedRows, fixedRhs)
            a0, a1 = dot(r, pFlat), dot(r, q)
            if a1 < b && a0 > a1
                sMax = min(sMax, (a0 - b) / (a0 - a1))
            elseif a1 < b
                sMax = zero(T)
            end
        end
        s = max(sMax * (1 - sqrt(eps(T))), zero(T))
        butterfly || return pFlat .+ s .* (q .- pFlat)
        for _ = 1:60
            gConstraintMin(pFlat .+ s .* (q .- pFlat)) >= 1 && break
            s /= 1.25
        end
        return pFlat .+ s .* (q .- pFlat)
    end
    function runFeasibleSQP(pStart)
        fixedRows, fixedRhs = linearRows()
        fixedA = reduce(hcat, fixedRows)
        p = pullBack(pStart, fixedRows, fixedRhs)
        minG = [minGConstraint(makeSurface(p))]
        local it = 0
        for iteration = 1:iterations
            #QP linearized with the target gMin; if no step along its solution is acceptable, QP relaxed so that the current
            #iterate is feasible
            gRows, gRhsTarget, gRhsRelaxed = Vector{Vector{T}}(), T[], T[]
            for (x, τ) in constraintPoints
                t = τ^2
                k = x * σbar * τ
                sk = σbar * τ
                rowsAt(x, τ)
                w, wp, wpp = t * dot(f, p), t * dot(fx, p) / sk, t * dot(fxx, p) / sk^2
                a = 1 - k * wp / (2w)
                gRef = gatheralG(k, w, wp, wpp)
                gw = a * k * wp / w^2 + wp^2 / (4w^2)
                gwp = -a * k / w - wp / 2 * (1 / w + one(T) / 4)
                push!(gRows, t .* (gw .* f .+ gwp .* fx ./ sk .+ fxx ./ (2sk^2)))
                gT = gTarget(x, τ)
                push!(gRhsTarget, gT - gRef + gw * w + gwp * wp + wpp / 2)
                push!(gRhsRelaxed, min(gT, gRef) - gRef + gw * w + gwp * wp + wpp / 2)
            end
            objRef = objectiveOf(p)
            step = one(T)
            accepted = false
            pQP = p
            for gRhs in (gRhsTarget, gRhsRelaxed)
                pQP = try
                    solveQuadraticProgram(solver, D, d, reduce(hcat, vcat(fixedRows, gRows)), vcat(fixedRhs, gRhs), tolerance=qpTolerance, stats=qpStats)[1]
                catch e
                    verbose && println("iteration ", iteration, " failed: ", e)
                    continue
                end
                #backtracking: decrease of the objective, linear constraints, and g ≥ gMin/4 on the constraint points
                step = one(T)
                for _ = 1:30
                    q = p .+ step .* (pQP .- p)
                    linearOK = minimum(transpose(fixedA) * q .- fixedRhs) >= -1e-10 * (1 + maximum(abs, fixedRhs))
                    if objectiveOf(q) <= objRef + 1e-14 * abs(objRef) && linearOK && (!butterfly || gConstraintMin(q) >= 0.25)
                        accepted = true
                        break
                    end
                    step /= 2
                end
                accepted && break
            end
            accepted || break
            newP = p .+ step .* (pQP .- p)
            #stationarity: the QP solution is (nearly) the current iterate
            Δx = maximum(abs, pQP - p) / maximum(abs, pQP)
            p = newP
            it = iteration
            push!(minG, minGConstraint(makeSurface(p)))
            verbose && println("iteration ", iteration, " min g ", minG[end], " objective ", objectiveOf(p), " step ", step, " Δx ", Δx)
            #stationary, or no more progress on the objective
            (Δx < xTol || objRef - objectiveOf(p) <= 1e-10 * abs(objRef)) && break
        end
        return makeSurface(p), it, minG
    end
    if method == :feasible
        p0, _ = solve(linearRows()...)
        surface, it, minG = runFeasibleSQP(p0)
    else
        surface, it, minG = runSQP()
    end
    #exchange: negative local minima along x of g - gTarget/2 and of ∂w/∂T - calendarMin/2 on the check grid, in one pass
    function negativeMinima(s)
        newG, newC = Tuple{T,T}[], Tuple{T,T}[]
        vg, vc = zeros(T, maximum(length(last(c)) for c in checkByTime)), zeros(T, maximum(length(last(c)) for c in checkByTime))
        for (τ, xs) in checkByTime
            n = length(xs)
            for (i, x) in enumerate(xs)
                w, wp, wpp, wt = totalVarianceDerivatives(s, x * σbar * τ, τ^2)
                vg[i] = gatheralG(x * σbar * τ, w, wp, wpp) - gTarget(x, τ) / 2
                vc[i] = wt - calendarMin / 2
            end
            for (vals, added, on) in ((vg, newG, butterfly), (vc, newC, calendar))
                on || continue
                for i = 1:n
                    if vals[i] < 0 && (i == 1 || vals[i] <= vals[i-1]) && (i == n || vals[i] <= vals[i+1])
                        push!(added, (xs[i], τ))
                    end
                end
            end
        end
        return newG, newC
    end
    for exchangeIteration = 1:exchange
        newG, newC = negativeMinima(surface)
        isempty(newG) && isempty(newC) && break
        verbose && println("exchange ", exchangeIteration, ": ", length(newG), " butterfly and ", length(newC), " calendar points added")
        append!(constraintPoints, newG)
        append!(calendarPoints, newC)
        for (x, τ) in newG
            positivity || break
            rowsAt(x, τ)
            push!(rows, copy(f)); push!(rhs, gMin * σbar^2)
        end
        surface, it2, minG2 = method == :feasible ? runFeasibleSQP(surface.p) : runSQP()
        it += it2
        append!(minG, minG2)
    end
    return surface, (iterations=it, minG=minG, nParams=np, nConstraints=length(rows) + length(calendarPoints) + length(constraintPoints))
end

Base.broadcastable(s::Multiquadric2DSurface) = Ref(s)
