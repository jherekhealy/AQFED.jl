#Convex Volatility Interpolation (Deschâtres 2026, Risk) and a multiquadric RBF variant.
#The implied variance of each expiry is linear in its parameters, v(k) = φ(k)ᵀx, in the normalized log-moneyness
#z = k/(σ*√T). All expiries are fitted at once by a quadratic programme. The butterfly condition g ≥ 0 is linearized
#around the previous iterate, the calendar spread condition is imposed on a grid of log-moneyness.
using LinearAlgebra
using GoldfarbIdnaniSolver
using SparseArrays
import Clarabel
import AQFED.Math: normpdf, normcdf

"""
    solveQuadraticProgram(solver, D, d, A, b)

Minimize -dᵀx + xᵀDx/2 subject to Aᵀx ≥ b (the columns of `A` are the constraints), with `solver` `:goldfarbidnani`
(dense dual active set method, requires D positive definite) or `:clarabel` (sparse interior point method).
Returns the solution and the value of the objective. Throws an error if the problem is infeasible.
"""
function solveQuadraticProgram(solver::Symbol, D::AbstractMatrix{T}, d::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T}; tolerance=1e-9, stats=nothing) where {T}
    t0 = time()
    if solver == :clarabel
        settings = Clarabel.Settings(verbose=false, tol_gap_abs=tolerance, tol_gap_rel=tolerance, tol_feas=tolerance, max_iter=500)
        model = Clarabel.Solver()
        #Clarabel: min xᵀPx/2 + qᵀx s.t. Gx + s = h, s ≥ 0, with G = -Aᵀ, h = -b
        G = A isa SparseMatrixCSC ? -sparse(transpose(A)) : sparse(-transpose(A))
        Clarabel.setup!(model, sparse(triu(D)), -Vector(d), G, -Vector(b), [Clarabel.NonnegativeConeT(length(b))], settings)
        result = Clarabel.solve!(model)
        if !(result.status == Clarabel.SOLVED || result.status == Clarabel.ALMOST_SOLVED)
            throw(error("Clarabel: ", result.status))
        end
        x = result.x
        #statistics: constraints, variables, non-zeros of the constraint matrix, interior point iterations, time, fully solved
        isnothing(stats) || push!(stats, (size(G, 1), size(G, 2), nnz(G), Int(result.iterations), time() - t0, result.status == Clarabel.SOLVED))
        return x, -dot(d, x) + dot(x, D * x) / 2
    elseif solver == :goldfarbidnanisparse
        #Goldfarb-Idnani on the compact (sparse) storage of the constraints: the slacks cost the non-zeros only
        As = A isa SparseMatrixCSC ? A : sparse(A)
        amat, aind = convertSparse(As)
        #R⁻¹ with D = RᵀR, by LAPACK rather than by the LINPACK-style loops of the solver
        Rinv = cholesky(Symmetric(Matrix(D))).U \ Matrix{T}(I, size(D, 1), size(D, 1))
        x, _, value, _, _, iter = solveQPcompact(Matrix(Rinv), Vector(d), amat, aind, Vector(b), meq=0, factorized=true)
        isnothing(stats) || push!(stats, (size(A, 2), size(A, 1), nnz(As), iter[1], time() - t0, true))
        return x, value
    else
        x, _, value, _, _, iter = solveQP(D, d, Matrix(A), b, meq=0)
        isnothing(stats) || push!(stats, (size(A, 2), size(A, 1), count(!iszero, A), iter[1], time() - t0, true))
        return x, value
    end
end
export CVIBasis, MultiquadricBasis, VarianceSlice, CVISurface, calibrateConvexVolatilitySurface, gatheralG, density, cviKnots

abstract type VarianceBasis end

"""
    CVIBasis(knots, scale)

Cubic spline in z = k/scale with linear extrapolation beyond the edge knots. Parameters are
v(0), ∂v/∂z(0) and ∂²v/∂z² at the interior knots (zero at the edge knots). `knots` must contain 0.
"""
struct CVIBasis{T} <: VarianceBasis
    knots::Vector{T}
    scale::T
    w::Matrix{T} #truncated power weights of the hat function of each interior knot, (n-2)×3
    q0::Vector{T} #Q(0)
    q0p::Vector{T} #Q'(0)
end

function CVIBasis(knots::AbstractVector{T}, scale::T) where {T}
    n = length(knots)
    w = zeros(T, n - 2, 3)
    q0 = zeros(T, n - 2)
    q0p = zeros(T, n - 2)
    for i = 2:n-1
        a = one(T) / (knots[i] - knots[i-1])
        b = one(T) / (knots[i+1] - knots[i])
        w[i-1, :] = [a, -(a + b), b]
        for (j, t) in enumerate(knots[i-1:i+1])
            q0[i-1] += w[i-1, j] * max(-t, zero(T))^3 / 6
            q0p[i-1] += w[i-1, j] * max(-t, zero(T))^2 / 2
        end
    end
    CVIBasis(collect(knots), scale, w, q0, q0p)
end

nparams(b::CVIBasis) = length(b.knots)
edges(b::CVIBasis) = (b.knots[1] * b.scale, b.knots[end] * b.scale)

"value, first and second derivative with respect to k of the basis functions at k"
function basisRows!(f, fp, fpp, b::CVIBasis{T}, k) where {T}
    z = k / b.scale
    f[1] = one(T); fp[1] = zero(T); fpp[1] = zero(T)
    f[2] = z; fp[2] = one(T); fpp[2] = zero(T)
    n = length(b.knots)
    for i = 2:n-1
        q, qp, qpp = zero(T), zero(T), zero(T)
        for (j, t) in enumerate(@view(b.knots[i-1:i+1]))
            x = max(z - t, zero(T))
            q += b.w[i-1, j] * x^3 / 6
            qp += b.w[i-1, j] * x^2 / 2
            qpp += b.w[i-1, j] * x
        end
        f[i+1] = q - b.q0[i-1] - b.q0p[i-1] * z
        fp[i+1] = qp - b.q0p[i-1]
        fpp[i+1] = qpp
    end
    fp ./= b.scale
    fpp ./= b.scale^2
    return f, fp, fpp
end

"rows giving dv/dk in the left and right wings (exact beyond the edge knots)"
function wingSlopeRows(b::CVIBasis{T}) where {T}
    n = nparams(b)
    f, fpL, fpp = zeros(T, n), zeros(T, n), zeros(T, n)
    basisRows!(f, fpL, fpp, b, b.knots[1] * b.scale)
    fpR = zeros(T, n)
    basisRows!(f, fpR, fpp, b, b.knots[end] * b.scale)
    return fpL, fpR
end

"""
    cviKnots(zmin, zmax, n)

`n` knots on [zmin, zmax], 0 included, equidistant on each side of 0 with spacing as close as possible.
"""
function cviKnots(zmin::T, zmax::T, n::Int) where {T}
    nLeft = clamp(round(Int, (n - 1) * (-zmin) / (zmax - zmin)), 1, n - 2)
    nRight = n - 1 - nLeft
    return vcat(collect(range(zmin, zero(T), length=nLeft + 1)), collect(range(zero(T), zmax, length=nRight + 1))[2:end])
end

"""
    MultiquadricBasis(centers, c, scale; linear=true)

v(z) = a + b z + Σⱼ αⱼ √(1 + (z-zⱼ)²/cⱼ²), z = k/scale. Without the linear term, b = 0. The bandwidth `c` is either a
number, common to all centers, or a vector, one per center.
"""
struct MultiquadricBasis{T} <: VarianceBasis
    centers::Vector{T}
    c::Vector{T}
    scale::T
    linear::Bool
end

MultiquadricBasis(centers::AbstractVector{T}, c::T, scale::T; linear::Bool=true) where {T} = MultiquadricBasis(collect(centers), fill(c, length(centers)), scale, linear)
MultiquadricBasis(centers::AbstractVector{T}, c::AbstractVector{T}, scale::T; linear::Bool=true) where {T} = MultiquadricBasis(collect(centers), collect(c), scale, linear)
nparams(b::MultiquadricBasis) = length(b.centers) + (b.linear ? 2 : 1)
edges(b::MultiquadricBasis) = (b.centers[1] * b.scale, b.centers[end] * b.scale)

function basisRows!(f, fp, fpp, b::MultiquadricBasis{T}, k) where {T}
    z = k / b.scale
    f[1] = one(T); fp[1] = zero(T); fpp[1] = zero(T)
    offset = 1
    if b.linear
        f[2] = z; fp[2] = one(T); fpp[2] = zero(T)
        offset = 2
    end
    for (j, zj) in enumerate(b.centers)
        c2 = b.c[j]^2
        x = z - zj
        ψ = sqrt(1 + x^2 / c2)
        f[offset+j] = ψ
        fp[offset+j] = x / (c2 * ψ)
        fpp[offset+j] = 1 / (c2 * ψ^3)
    end
    fp ./= b.scale
    fpp ./= b.scale^2
    return f, fp, fpp
end

"rows giving the asymptotic dv/dk for k → -∞ and k → +∞"
function wingSlopeRows(b::MultiquadricBasis{T}) where {T}
    n = nparams(b)
    fpL, fpR = zeros(T, n), zeros(T, n)
    offset = 1
    if b.linear
        fpL[2] = one(T); fpR[2] = one(T)
        offset = 2
    end
    fpL[offset+1:end] .= -one(T) ./ b.c
    fpR[offset+1:end] .= one(T) ./ b.c
    return fpL ./ b.scale, fpR ./ b.scale
end

"rows giving the intercepts (value at k = 0) of the asymptotes of v for k → -∞ and k → +∞"
function wingInterceptRows(b::MultiquadricBasis{T}) where {T}
    n = nparams(b)
    iL, iR = zeros(T, n), zeros(T, n)
    iL[1] = one(T); iR[1] = one(T)
    offset = b.linear ? 2 : 1
    #ψ(z - zⱼ) ≈ |z - zⱼ|/cⱼ: intercept +zⱼ/cⱼ on the left, -zⱼ/cⱼ on the right
    iL[offset+1:end] .= b.centers ./ b.c
    iR[offset+1:end] .= -b.centers ./ b.c
    return iL, iR
end

struct VarianceSlice{B<:VarianceBasis,T}
    basis::B
    x::Vector{T}
    tte::T
    forward::T
end

function varianceDerivatives(s::VarianceSlice{B,T}, k) where {B,T}
    n = nparams(s.basis)
    f, fp, fpp = zeros(T, n), zeros(T, n), zeros(T, n)
    basisRows!(f, fp, fpp, s.basis, k)
    return dot(f, s.x), dot(fp, s.x), dot(fpp, s.x)
end

varianceByLogmoneyness(s::VarianceSlice, k) = varianceDerivatives(s, k)[1]
(s::VarianceSlice)(k) = varianceByLogmoneyness(s, k)
Base.broadcastable(s::VarianceSlice) = Ref(s)

"Gatheral's g(k) for a total variance w and its first two derivatives; the density is positive iff g ≥ 0."
gatheralG(k, w, wp, wpp) = (1 - k * wp / (2w))^2 - wp^2 / 4 * (1 / w + one(w) / 4) + wpp / 2

function gatheralG(s::VarianceSlice, k)
    v, vp, vpp = varianceDerivatives(s, k)
    return gatheralG(k, v * s.tte, vp * s.tte, vpp * s.tte)
end

"probability density of the underlying at expiry, at strike K"
function density(s::VarianceSlice, strike)
    k = log(strike / s.forward)
    v, vp, vpp = varianceDerivatives(s, k)
    w = v * s.tte
    sqrtw = sqrt(w)
    return gatheralG(k, w, vp * s.tte, vpp * s.tte) * normpdf(-k / sqrtw - sqrtw / 2) / (strike * sqrtw)
end

struct CVISurface{S}
    slices::Vector{S}
end

"""implied variance, linear interpolation of the total variance in time at fixed log-moneyness k = log(K/F(T)). Beyond the
last expiry T, the total variance grows at the at-the-money implied variance a of T: w(k, t) = w(k, T) + (t - T) a, flat in
implied volatility at the money, with ∂w/∂t = a > 0 (flat volatility at every k would scale the wings of w beyond Lee's
bound)."""
function varianceByLogmoneyness(surface::CVISurface, k, t)
    slices = surface.slices
    t <= slices[1].tte && return slices[1](k)
    if t >= slices[end].tte
        sN, tN = slices[end], slices[end].tte
        return (sN(k) * tN + (t - tN) * sN(zero(k))) / t
    end
    i = searchsortedlast([s.tte for s in slices], t)
    w0, w1 = slices[i](k) * slices[i].tte, slices[i+1](k) * slices[i+1].tte
    θ = (t - slices[i].tte) / (slices[i+1].tte - slices[i].tte)
    return ((1 - θ) * w0 + θ * w1) / t
end

"""
    makeCVIBasis(tte, logmoneyness, atmVol; n=15, zMargin=1.05, zWing=4.0)

CVI basis with `n` knots spanning the data range (in z) of the slice. When the data does not reach |z| = `zWing`, an extra
edge knot is placed there, so that d₁d₂ > 1 holds comfortably at the edges (the linear wings start far enough in the tails).
"""
function makeCVIBasis(tte::T, ys::AbstractVector{T}, atmVol::T; n::Int=15, zMargin=1.05, zWing=4.0) where {T}
    scale = atmVol * sqrt(tte)
    zs = ys ./ scale
    zmin, zmax = min(zMargin * minimum(zs), -one(T) / 2), max(zMargin * maximum(zs), one(T) / 2)
    knots = cviKnots(zmin, zmax, n)
    zmin > -zWing && pushfirst!(knots, -zWing)
    zmax < zWing && push!(knots, zWing)
    return CVIBasis(knots, scale)
end

"""
    makeMultiquadricBasis(tte, logmoneyness, atmVol; m=6, shape=2.0, bandwidth=nothing, centers=:uniform, linear=true)

`m` centers on the data range (in z), `centers=:uniform` (equidistant in z) or `:asinh` (equidistant in asinh(z/2), dense
near the money). The bandwidth is max(`minBandwidth`, `rangeBandwidth` times the data range in z) if given, or `bandwidth` (in z, the same for all
centers and expiries, in units of the at-the-money standard deviation) if given, otherwise `shape` times the local spacing
of the centers. When the quotes do not reach |z| = `zWing`, an extra center is placed there, with the bandwidth of the
outermost center.
"""
function makeMultiquadricBasis(tte::T, ys::AbstractVector{T}, atmVol::T; m::Int=6, shape=2.0, bandwidth=nothing, rangeBandwidth=nothing, minBandwidth=zero(T), centers=:uniform, linear::Bool=true, zWing=0.0) where {T}
    scale = atmVol * sqrt(tte)
    zs = ys ./ scale
    zc = if centers == :asinh
        [2sinh(u) for u in range(asinh(minimum(zs) / 2), asinh(maximum(zs) / 2), length=m)]
    else
        collect(range(minimum(zs), maximum(zs), length=m))
    end
    c = if rangeBandwidth !== nothing
        fill(max(T(minBandwidth), T(rangeBandwidth) * (zc[end] - zc[1])), m)
    elseif bandwidth === nothing
        spacing = [j == 1 ? zc[2] - zc[1] : j == m ? zc[m] - zc[m-1] : (zc[j+1] - zc[j-1]) / 2 for j = 1:m]
        shape .* spacing
    else
        fill(T(bandwidth), m)
    end
    #extra wing centers at ±zWing when the quotes do not reach it, as the extra edge knots of CVI
    if zc[1] > -zWing
        pushfirst!(zc, -T(zWing)); pushfirst!(c, c[1])
    end
    if zc[end] < zWing
        push!(zc, T(zWing)); push!(c, c[end])
    end
    return MultiquadricBasis(zc, c, scale, linear=linear)
end

"ATM volatility anchor σ*, by linear interpolation of the market vols in log-moneyness"
function atmAnchor(ys::AbstractVector{T}, vols::AbstractVector{T}) where {T}
    i = searchsortedlast(ys, zero(T))
    i <= 0 && return vols[1]
    i >= length(ys) && return vols[end]
    θ = -ys[i] / (ys[i+1] - ys[i])
    return (1 - θ) * vols[i] + θ * vols[i+1]
end

struct CVICalibrationInfo{T}
    iterations::Int
    minG::Vector{T} #minimum of g on the check grid after each solve
    objective::Vector{T} #value of the quadratic objective after each solve
    nConstraints::Int
    nVariables::Int
end

"""
    calibrateConvexVolatilitySurface(ttes, forwards, logmoneyness, vols, volSpreads; basis=:cvi, kwargs...)

Fit all the expiries at once. `logmoneyness[j]`, `vols[j]`, `volSpreads[j]` are the vectors of quotes for expiry `ttes[j]`
(log(K/F), mid vol and ask-bid vol spread). The least-squares weight is 1/(v_ask - v_bid)² = 1/(2σΔσ)².

Keyword arguments:
- `basis`: `:cvi`, `:rbf` (multiquadric), or a function `(tte, ys, atmVol) -> basis`. The bandwidth
  is `shape` times the spacing of the centers (default 1), or as set by
  `bandwidth`, `rangeBandwidth` and `minBandwidth`.
- `λ`: total variation penalty on the normalized convexities (CVI only).
- `butterfly`, `calendar`: impose the no-arbitrage constraints; `iterations`: maximum number of linearized butterfly solves,
  stopped when the relative change of the parameters is below `xTol`, or at the first feasible iterate if `stopAtFeasible`.
  The feasible iterate (g ≥ 0 on the check grid) of lowest objective is returned.
- `nGrid`: number of uniform points of the butterfly and calendar grids per expiry (the butterfly grid also contains the market
  log-moneyness if `marketPoints`, and `pointsPerKnot` points in each interval between knots or centers); `nFar`, `nFarCalendar`: number of far points per side of the RBF butterfly and calendar
  grids; `wingFactor`: extension of the grid beyond the
  edges, in multiples of the edge log-moneyness (used for RBF, where the wings are not exactly linear).
- `qpTolerance`: tolerance of the interior point solver Clarabel (`1e-6`: the multiquadric problems are too ill-conditioned
  to reach `1e-9`, and the linear constraints are checked exactly in the backtracking anyway).
- `precondition`: solve the QPs in the coordinates y = S Vᵀ x of each expiry, from the SVD B = U S Vᵀ of the basis values
  on the quoted range, so that the QP variables are well scaled (same problem, exactly).
- `qpStats`: if a vector, receives a tuple (constraints, variables, non-zeros, iterations, time, solved) per QP.
- `warmStart`: number of full-step linearized QPs from the unconstrained fit before the feasible SQP (method `:feasible`, default 1).
- `roughness`: weight μ of the penalty μ ∫ (∂³v/∂z³ / σ*²)² dz over the quoted range of each expiry, which smooths the
  convexity (and the density) independently of the bandwidth of the radial basis (default `1e-2` for the multiquadric, 0 otherwise).
- `gridType`: `:asinh` (the `nGrid` points uniform in asinh(z/2) plus `pointsPerKnot` points per knot or center interval),
  or `:uniform` (RBF only: uniform in z with `pointsPerKnot+1` points per center spacing over the whole grid range);
  `:auto` is `:asinh`.
- `edgeCondition`: impose d₁d₂ ≥ 1 at the CVI edge knots, w ≤ 2(√(1+k²)-1), and the lower branch |w'| ≤ 2w/(|k|+√(w+w²/4)),
  which makes the edge butterfly constraint sufficient
  for the whole linear wing.
- `gMin`: lower bound for the linearized g; `wingGMin` (default 0.01, `nothing` for none): the bound rises to it beyond the quoted range (over
  `wingRamp` times the quoted width), which caps the local volatility in the extrapolated wings; `leeMargin`: the total variance slopes must lie in [-2+leeMargin, 2-leeMargin].
- `method`: `:sqp` (full-step sequential linearization, the feasible iterate of lowest objective is returned) or `:feasible`
  (every iterate is kept free of arbitrage on the grids, by pulling back towards the flat surface v = σ̄² and backtracking).
- `solver`: quadratic programming solver, `:goldfarbidnani` (dense active set), `:goldfarbidnanisparse` (active set on the
  compact storage of the sparse constraints), `:clarabel` (sparse interior point) or `:auto` (Clarabel for CVI surfaces
  with the total variation penalty, whose auxiliary variables slow down the active set method, the sparse active set
  method otherwise).
- `λ2`: weight of the quadratic penalty Σ ((c_{i+1}-c_i)/v*)² on the CVI convexities, an alternative to the total
  variation penalty `λ` without auxiliary variables.
- `calendarMargin`: the calendar spreads must exceed `calendarMargin` (T_{j+1}-T_j) σ̄² at the calendar points, a fraction
  of the spread of the flat surface; the exchange adds the points where they fall below half of it (resp. where g < gMin/2).
- `exchange`: maximum number of exchange steps: the local minima of g (resp. of the calendar spreads) that are too small on the
  fine check grids are added to the butterfly grid (resp. calendar points), and the problem is solved again.
- `wingConvexity`: v'' ≥ 0 on the grid points beyond the quotes. `convexVariance` (RBF): α ≥ 0, a convex variance.
  `asymptoteOrdering` (RBF): the intercepts of the asymptotes of the total variance are ordered in time, as the slopes.
"""
function calibrateConvexVolatilitySurface(ttes::AbstractVector{T}, forwards::AbstractVector{T}, logmoneyness::AbstractVector{<:AbstractVector{T}}, vols::AbstractVector{<:AbstractVector{T}}, volSpreads::AbstractVector{<:AbstractVector{T}};
    basis=:cvi, n::Int=15, zWing=4.0, m::Int=6, shape=nothing, bandwidth=nothing, rangeBandwidth=nothing, minBandwidth=0.0, centers=:uniform, rbfWing=0.0, linear::Bool=true, λ=0.05, λ2=0.0, butterfly::Bool=true, calendar::Bool=true, iterations::Int=30, xTol=1e-6, stopAtFeasible::Bool=false,
    nGrid::Int=40, gridType::Symbol=:auto, marketPoints::Bool=false, pointsPerKnot::Int=3, nFar::Int=6, nFarCalendar::Int=8, nCalendar::Int=20, nCheck::Int=2001, nCheckPerKnot::Int=64, wingFactor=0.0, edgeCondition::Bool=true, gMin=1e-4, wingGMin=1e-2, wingRamp=0.25, leeMargin=1e-2, ridge=1e-10, exchange::Int=25, solver::Symbol=:auto, method::Symbol=:feasible, calendarMargin=1e-3, trackIterations::Bool=false, wingConvexity::Bool=false, asymptoteOrdering::Bool=false, convexVariance::Bool=false, qpTolerance=1e-6, precondition::Bool=false, qpStats=nothing, roughness=nothing, warmStart::Int=1, verbose::Bool=false) where {T}
    nExp = length(ttes)
    #the multiquadric is smoothed by the roughness penalty by default, with a bandwidth equal to the spacing of the centers
    roughness = something(roughness, basis == :rbf ? 1e-2 : 0.0)
    #the active set method on the sparse constraints, except for CVI surfaces with the total variation penalty, whose
    #auxiliary variables multiply its iterations: there, the interior point method
    solver = solver == :auto ? ((basis == :cvi && λ > 0 && nExp > 1) ? :clarabel : :goldfarbidnanisparse) : solver
    atmVols = [atmAnchor(logmoneyness[j], vols[j]) for j = 1:nExp]
    σbar2 = (sum(atmVols) / nExp)^2
    #minimum calendar spread at the constraint points: a fraction of that of the flat surface v = σ̄²
    calendarMin = [calendarMargin * (ttes[j+1] - ttes[j]) * σbar2 for j = 1:nExp-1]
    makeBasis = if basis == :cvi
        (tte, ys, atmVol) -> makeCVIBasis(tte, ys, atmVol, n=n, zWing=zWing)
    elseif basis == :rbf
        (tte, ys, atmVol) -> makeMultiquadricBasis(tte, ys, atmVol, m=m, shape=something(shape, 1.0), bandwidth=bandwidth, rangeBandwidth=rangeBandwidth, minBandwidth=minBandwidth, centers=centers, linear=linear, zWing=rbfWing)
    else
        basis
    end
    bases = [makeBasis(ttes[j], logmoneyness[j], atmVols[j]) for j = 1:nExp]
    isCVI = bases[1] isa CVIBasis
    #CVI has linear wings beyond its edge knots: exact wing conditions, no far points
    linearWings = !(bases[1] isa MultiquadricBasis)
    np = [nparams(b) for b in bases]
    offsets = cumsum(vcat(0, np))
    nx = offsets[end]
    #total variation auxiliary variables: one per difference of normalized convexities (n-1 per expiry)
    nTV = (isCVI && λ > 0) ? sum(length(b.knots) - 1 for b in bases) : 0
    nVar = nx + nTV
    D = zeros(T, nVar, nVar)
    d = zeros(T, nVar)
    for j = 1:nExp
        b = bases[j]
        ys, σs, Δσs = logmoneyness[j], vols[j], volSpreads[j]
        N = length(ys)
        f, fp, fpp = zeros(T, np[j]), zeros(T, np[j]), zeros(T, np[j])
        idx = offsets[j]+1:offsets[j+1]
        for i = 1:N
            basisRows!(f, fp, fpp, b, ys[i])
            wi = one(T) / (2σs[i] * Δσs[i])^2 / N
            D[idx, idx] .+= 2wi .* (f * f')
            d[idx] .+= 2wi * σs[i]^2 .* f
        end
    end
    trD = tr(D) / nx
    for i = 1:nx
        D[i, i] += ridge * trD
    end
    if roughness > 0
        #roughness penalty μ ∫ (∂³v/∂z³ / v*)² dz over the quoted range of each expiry (the smooth analogue of the total
        #variation of the convexity): third derivative by central differences of the second derivative, trapezoidal
        #rule in k, with ∫ (∂³v/∂z³)² dz = s⁵ ∫ (∂³v/∂k³)² dk
        for j = 1:nExp
            b = bases[j]
            s = b.scale
            vStar = atmVols[j]^2
            kL, kR = minimum(logmoneyness[j]), maximum(logmoneyness[j])
            nodes = b isa CVIBasis ? b.knots : b.centers
            nq = max(401, ceil(Int, 32 * (kR - kL) / (s * minimum(diff(nodes)))) + 1)
            kq = range(kL, kR, length=nq)
            hk = 1e-4 * s
            f, fp, fppP, fppM = zeros(T, np[j]), zeros(T, np[j]), zeros(T, np[j]), zeros(T, np[j])
            idx = offsets[j]+1:offsets[j+1]
            R = zeros(T, np[j], np[j])
            for (i, k) in enumerate(kq)
                basisRows!(f, fp, fppP, b, k + hk)
                basisRows!(f, fp, fppM, b, k - hk)
                v3 = (fppP .- fppM) ./ (2hk)
                wq = (i == 1 || i == nq) ? step(kq) / 2 : step(kq)
                R .+= wq .* (v3 * v3')
            end
            D[idx, idx] .+= (2roughness * s^5 / vStar^2) .* R
        end
    end
    if isCVI && λ2 > 0
        #quadratic penalty λ2 Σ ((c[i+1]-c[i])/v*)² on the normalized convexities: the smooth analogue of the total
        #variation penalty, without auxiliary variables
        for j = 1:nExp
            nk = length(bases[j].knots)
            vStar = atmVols[j]^2
            cIdx(i) = (i == 1 || i == nk) ? 0 : offsets[j] + i + 1
            for i = 1:nk-1
                r = zeros(T, nVar)
                cIdx(i + 1) > 0 && (r[cIdx(i + 1)] += 1 / vStar)
                cIdx(i) > 0 && (r[cIdx(i)] -= 1 / vStar)
                D .+= (2λ2) .* (r * r')
            end
        end
    end
    rows = Vector{Vector{T}}()
    rhs = T[]
    function addRow!(row, b)
        push!(rows, row)
        push!(rhs, b)
    end
    if nTV > 0
        tvOffset = nx
        for j = 1:nExp
            b = bases[j]
            nk = length(b.knots)
            vStar = atmVols[j]^2
            #normalized convexity at knot i (1-based): 0 at the edges, x[offset+i+1]/v* for interior knots i=2..n-1
            cIndex(i) = (i == 1 || i == nk) ? 0 : offsets[j] + i + 1
            for i = 1:nk-1
                t = tvOffset + i
                for sgn in (one(T), -one(T))
                    row = zeros(T, nVar)
                    row[t] = one(T)
                    ci, cn = cIndex(i), cIndex(i + 1)
                    cn > 0 && (row[cn] -= sgn / vStar)
                    ci > 0 && (row[ci] += sgn / vStar)
                    addRow!(row, zero(T)) #t ≥ ±(c[i+1]-c[i])
                end
                d[t] = -λ
                D[t, t] = ridge * trD + 1e-8 * λ
            end
            tvOffset += nk - 1
        end
    end
    grids = Vector{Vector{T}}(undef, nExp)
    checkGrids = Vector{Vector{T}}(undef, nExp)
    ranges = Vector{Tuple{T,T}}(undef, nExp)
    for j = 1:nExp
        b = bases[j]
        kL, kR = edges(b)
        kL, kR = min(kL, minimum(logmoneyness[j])), max(kR, maximum(logmoneyness[j]))
        ranges[j] = (kL * (1 + wingFactor), kR * (1 + wingFactor))
        #market log-moneyness (dense where g varies quickly, near the money for short expiries) and a uniform grid up to the wings
        #uniform in asinh(z/2): dense near the money, logarithmic in the wings
        uL, uR = asinh(kL * (1 + wingFactor) / (2b.scale)), asinh(kR * (1 + wingFactor) / (2b.scale))
        uniformGrid = [2b.scale * sinh(u) for u in range(uL, uR, length=nGrid)]
        #the RBF wings are only asymptotically linear: add a few far points
        farGrid = (linearWings || nFar == 0) ? T[] : [f * k for f in exp.(range(log(1 + wingFactor), log(100.0), length=nFar)) for k in (kL, kR)]
        #points in each knot (CVI) or center (RBF) interval, so that the grid resolves the basis
        nodes = isCVI ? b.knots : b.centers
        basisGrid = pointsPerKnot == 0 ? T[] : [b.scale * (nodes[i] + (nodes[i+1] - nodes[i]) * l / (pointsPerKnot + 1)) for i = 1:length(nodes)-1 for l = 0:pointsPerKnot]
        pointsPerKnot > 0 && push!(basisGrid, b.scale * nodes[end])
        if gridType == :uniform && !isCVI
            #RBF: the basis has the same length scale everywhere, use a uniform grid in z at the spacing of the basis over the whole constrained range
            lo, hi = ranges[j]
            step = b.scale * minimum(diff(b.centers)) / (pointsPerKnot + 1)
            uniformGrid = collect(range(lo, hi, length=max(nGrid, ceil(Int, (hi - lo) / step) + 1)))
            basisGrid = T[]
        end
        grids[j] = sort(unique(vcat(marketPoints ? logmoneyness[j] : T[], uniformGrid, farGrid, basisGrid)))
        #fine check grid: data range plus far wings
        #check grid: uniform over three times the range, geometric far out, and fine at the scale of the basis
        checkNodes = isCVI ? b.knots : b.centers
        #nCheckPerKnot points per interval between nodes, from edge to edge
        checkStep = b.scale * minimum(diff(checkNodes)) / nCheckPerKnot
        basisCheck = collect(range(kL, kR, length=max(2, ceil(Int, (kR - kL) / checkStep) + 1)))
        checkGrids[j] = sort(unique(vcat(collect(range(3kL, 3kR, length=nCheck)), [f * k for f in exp.(range(log(3.0), log(300.0), length=nCheck ÷ 4)) for k in (kL, kR)], basisCheck)))
    end
    for j = 1:nExp
        b = bases[j]
        tte = ttes[j]
        idx = offsets[j]+1:offsets[j+1]
        f, fp, fpp = zeros(T, np[j]), zeros(T, np[j]), zeros(T, np[j])
        #positivity of the variance
        for k in grids[j]
            basisRows!(f, fp, fpp, b, k)
            row = zeros(T, nVar); row[idx] = f
            addRow!(row, gMin * atmVols[j]^2)
        end
        #wings: variance non-increasing on the left, non-decreasing on the right, Lee's bound on the total variance slope
        rowL, rowR = wingSlopeRows(b)
        row = zeros(T, nVar); row[idx] = -rowL; addRow!(row, zero(T))
        row = zeros(T, nVar); row[idx] = rowR; addRow!(row, zero(T))
        row = zeros(T, nVar); row[idx] = tte .* rowL; addRow!(row, -(2 - leeMargin))
        row = zeros(T, nVar); row[idx] = -tte .* rowR; addRow!(row, -(2 - leeMargin))
        if butterfly && edgeCondition && linearWings
            for k in edges(b)
                basisRows!(f, fp, fpp, b, k)
                row = zeros(T, nVar); row[idx] = -tte .* f
                addRow!(row, -2 * (sqrt(1 + k^2) - 1) * (1 - leeMargin))
            end
        end
    end
    if !isCVI && convexVariance
        #α ≥ 0: the variance is convex everywhere
        for j = 1:nExp
            offset = offsets[j] + (bases[j].linear ? 2 : 1)
            for l = 1:length(bases[j].centers)
                row = zeros(T, nVar); row[offset+l] = one(T); addRow!(row, zero(T))
            end
        end
    end
    if wingConvexity
        #v'' ≥ 0 on the grid points beyond the quotes
        for j = 1:nExp
            idx = offsets[j]+1:offsets[j+1]
            f, fp, fpp = zeros(T, np[j]), zeros(T, np[j]), zeros(T, np[j])
            for k in grids[j]
                (logmoneyness[j][1] <= k <= logmoneyness[j][end]) && continue
                basisRows!(f, fp, fpp, bases[j], k)
                row = zeros(T, nVar); row[idx] = fpp; addRow!(row, zero(T))
            end
        end
    end
    #calendar spread points of each pair of consecutive expiries, extended by the exchange method
    calendarPoints = Vector{Vector{T}}(undef, max(nExp - 1, 0))
    calendarChecks = Vector{Vector{T}}(undef, max(nExp - 1, 0))
    function addCalendarRow!(rows, rhs, j, k)
        f0, fp0, fpp0 = zeros(T, np[j]), zeros(T, np[j]), zeros(T, np[j])
        f1, fp1, fpp1 = zeros(T, np[j+1]), zeros(T, np[j+1]), zeros(T, np[j+1])
        basisRows!(f0, fp0, fpp0, bases[j], k)
        basisRows!(f1, fp1, fpp1, bases[j+1], k)
        row = zeros(T, nVar)
        row[offsets[j+1]+1:offsets[j+2]] = ttes[j+1] .* f1
        row[offsets[j]+1:offsets[j+1]] = -ttes[j] .* f0
        push!(rows, row)
        push!(rhs, calendarMin[j])
    end
    for j = 1:nExp-1
        #CVI: both slices are linear beyond the outermost edge knots
        kL = min(ranges[j][1], ranges[j+1][1])
        kR = max(ranges[j][2], ranges[j+1][2])
        geometric = nFarCalendar == 0 ? T[] : exp.(range(log(1.02), log(100.0), length=nFarCalendar))
        #the RBF wings are only asymptotically linear: add far points, as for the butterfly grid
        farCalendar = linearWings ? T[] : [f * k for f in geometric for k in (kL, kR)]
        calendarPoints[j] = vcat(collect(range(kL, kR, length=nCalendar)), farCalendar)
        calendarChecks[j] = sort(vcat(collect(range(3kL, 3kR, length=nCheck)), [f * k for f in exp.(range(log(3.0), log(300.0), length=nCheck ÷ 4)) for k in (kL, kR)]))
    end
    if calendar
        for j = 1:nExp-1
            b0, b1 = bases[j], bases[j+1]
            idx0, idx1 = offsets[j]+1:offsets[j+1], offsets[j+1]+1:offsets[j+2]
            f0, fp0, fpp0 = zeros(T, np[j]), zeros(T, np[j]), zeros(T, np[j])
            f1, fp1, fpp1 = zeros(T, np[j+1]), zeros(T, np[j+1]), zeros(T, np[j+1])
            kL, kR = calendarPoints[j][1], calendarPoints[j][nCalendar]
            #tails: the difference of total variances must be non-increasing on the left of kL and non-decreasing on the right of kR.
            if linearWings
                for (k, sgn) in ((kL, -one(T)), (kR, one(T)))
                    basisRows!(f0, fp0, fpp0, b0, k)
                    basisRows!(f1, fp1, fpp1, b1, k)
                    row = zeros(T, nVar)
                    row[idx1] = sgn * ttes[j+1] .* fp1
                    row[idx0] = -sgn * ttes[j] .* fp0
                    addRow!(row, zero(T))
                end
            else
                rowL0, rowR0 = wingSlopeRows(b0)
                rowL1, rowR1 = wingSlopeRows(b1)
                row = zeros(T, nVar); row[idx1] = -ttes[j+1] .* rowL1; row[idx0] = ttes[j] .* rowL0; addRow!(row, zero(T))
                row = zeros(T, nVar); row[idx1] = ttes[j+1] .* rowR1; row[idx0] = -ttes[j] .* rowR0; addRow!(row, zero(T))
                if asymptoteOrdering
                    #with ordered slopes, ordered intercepts make the asymptotes of the total variances non-crossing
                    iL0, iR0 = wingInterceptRows(b0)
                    iL1, iR1 = wingInterceptRows(b1)
                    row = zeros(T, nVar); row[idx1] = ttes[j+1] .* iL1; row[idx0] = -ttes[j] .* iL0; addRow!(row, zero(T))
                    row = zeros(T, nVar); row[idx1] = ttes[j+1] .* iR1; row[idx0] = -ttes[j] .* iR0; addRow!(row, zero(T))
                end
            end
        end
    end
    #optional change of coordinates x = P y, with P = V S⁻¹ from the SVD of the basis values on the constraint grid of
    #each expiry: the columns of B P are orthonormal, so that the QP sees well-scaled variables for the same fit
    Pinvs = Matrix{T}[]
    P = if precondition
        Ib, Jb, Vb = Int[], Int[], T[]
        for j = 1:nExp
            f, fp, fpp = zeros(T, np[j]), zeros(T, np[j]), zeros(T, np[j])
            #values on a uniform grid over the range of the data (the far wing points would dominate the scaling)
            kg = range(minimum(logmoneyness[j]), maximum(logmoneyness[j]), length=max(200, 4np[j]))
            B = zeros(T, length(kg), np[j])
            for (i, k) in enumerate(kg)
                basisRows!(f, fp, fpp, bases[j], k)
                B[i, :] = f
            end
            F = svd(B)
            Sj = max.(F.S, F.S[1] * 1e-15)
            Pj = F.V * Diagonal(1 ./ Sj)
            push!(Pinvs, Diagonal(Sj) * transpose(F.V))
            for c = 1:np[j], r = 1:np[j]
                push!(Ib, offsets[j] + r); push!(Jb, offsets[j] + c); push!(Vb, Pj[r, c])
            end
        end
        for i = offsets[end]+1:nVar
            push!(Ib, i); push!(Jb, i); push!(Vb, one(T))
        end
        sparse(Ib, Jb, Vb, nVar, nVar)
    else
        nothing
    end
    #parameters in the coordinates of the QP, for the stopping criterion
    qpCoordinates(x) = isnothing(P) ? x[1:nx] : vcat([Pinvs[j] * x[offsets[j]+1:offsets[j+1]] for j = 1:nExp]...)
    Dy, dy = isnothing(P) ? (D, d) : (Matrix(Symmetric(Matrix(transpose(P) * D * P))), transpose(P) * d)
    function solve(rows, rhs)
        A = if solver == :clarabel
            #sparse constraint matrix from the non-zeros of the rows (the rows of CVI touch one or two expiries)
            I, J, V = Int[], Int[], T[]
            for (c, r) in enumerate(rows), i in eachindex(r)
                if r[i] != 0
                    push!(I, i); push!(J, c); push!(V, r[i])
                end
            end
            sparse(I, J, V, nVar, length(rows))
        else
            reduce(hcat, rows)
        end
        isnothing(P) && return solveQuadraticProgram(solver, D, d, A, rhs, tolerance=qpTolerance, stats=qpStats)
        Ay = solver == :clarabel ? sparse(transpose(P) * A) : Matrix(transpose(P) * A)
        y, value = solveQuadraticProgram(solver, Dy, dy, Ay, rhs, tolerance=qpTolerance, stats=qpStats)
        return P * y, value
    end
    makeSlices(sol) = [VarianceSlice(bases[j], sol[offsets[j]+1:offsets[j+1]], ttes[j], forwards[j]) for j = 1:nExp]
    #basis rows on the (fixed) check grids, so that the checks are matrix-vector products
    function basisMatrices(j, ks)
        F, Fp, Fpp = zeros(T, length(ks), np[j]), zeros(T, length(ks), np[j]), zeros(T, length(ks), np[j])
        f, fp, fpp = zeros(T, np[j]), zeros(T, np[j]), zeros(T, np[j])
        for (i, k) in enumerate(ks)
            basisRows!(f, fp, fpp, bases[j], k)
            F[i, :] = f; Fp[i, :] = fp; Fpp[i, :] = fpp
        end
        return F, Fp, Fpp
    end
    checkMatrices = [basisMatrices(j, checkGrids[j]) for j = 1:nExp]
    calendarMatrices = [(basisMatrices(j, calendarChecks[j])[1], basisMatrices(j + 1, calendarChecks[j])[1]) for j = 1:nExp-1]
    function gValues(j, x)
        F, Fp, Fpp = checkMatrices[j]
        w, wp, wpp = ttes[j] .* (F * x), ttes[j] .* (Fp * x), ttes[j] .* (Fpp * x)
        return [gatheralG(checkGrids[j][i], w[i], wp[i], wpp[i]) for i in eachindex(w)]
    end
    calendarValues(slices, j) = ttes[j+1] .* (calendarMatrices[j][2] * slices[j+1].x) .- ttes[j] .* (calendarMatrices[j][1] * slices[j].x)
    minGOf(slices) = minimum(minimum(gValues(j, s.x)) for (j, s) in enumerate(slices))
    "local minima of f on the sorted points ks where f < 0"
    "local minima of the values on the sorted points ks where the value is below level"
    #margin on g: gMin on the quoted range of the expiry, rising to wingGMin at wingRamp times the quoted width beyond it
    #(caps the local volatility ∂T w / g in the extrapolated wings)
    function gTarget(j, k)
        wingGMin === nothing && return T(gMin)
        kL, kR = extrema(logmoneyness[j])
        excess = k < kL ? (kL - k) / (kR - kL) : k > kR ? (k - kR) / (kR - kL) : zero(T)
        return gMin + (wingGMin - gMin) * min(one(T), excess / wingRamp)
    end
    lowMinima(values, ks, level) = [ks[i] for i in eachindex(ks) if values[i] < level && (i == 1 || values[i] <= values[i-1]) && (i == length(ks) || values[i] <= values[i+1])]
    "local minima of f on the sorted points ks where f < level"
    function negativeMinima(f, ks, level=zero(T))
        ks = sort(ks)
        values = f.(ks)
        return [ks[i] for i in eachindex(ks) if values[i] < level && (i == 1 || values[i] <= values[i-1]) && (i == length(ks) || values[i] <= values[i+1])]
    end
    nFixed = length(rows)
    #sequential linearization of the butterfly condition, for the current grids and calendar points
    "linearized butterfly constraints around sol (and the lower branch at the CVI edge knots), appended to the given rows"
    "basis rows (values, first and second derivatives in k) at the grid points of each slice"
    function gridMatrices()
        return [begin
            F, Fp, Fpp = zeros(T, length(grids[j]), np[j]), zeros(T, length(grids[j]), np[j]), zeros(T, length(grids[j]), np[j])
            f, fp, fpp = zeros(T, np[j]), zeros(T, np[j]), zeros(T, np[j])
            for (i, k) in enumerate(grids[j])
                basisRows!(f, fp, fpp, bases[j], k)
                F[i, :] = f; Fp[i, :] = fp; Fpp[i, :] = fpp
            end
            (F, Fp, Fpp)
        end for j = 1:nExp]
    end
    """
    linearized butterfly constraints around sol (and the lower branch at the CVI edge knots), appended to the given rows.
    With `relax`, the bound of each linearized constraint is lowered to its value at sol when sol does not reach it, so that
    sol is feasible for the QP.
    """
    function linearizedRows(sol, fixedRows, fixedRhs, matrices; relax::Bool=false)
        bRows = copy(fixedRows)
        bRhs = copy(fixedRhs)
        for j = 1:nExp
            b = bases[j]
            tte = ttes[j]
            idx = offsets[j]+1:offsets[j+1]
            xRef = sol[idx]
            F, Fp, Fpp = matrices[j]
            ws, wps, wpps = tte .* (F * xRef), tte .* (Fp * xRef), tte .* (Fpp * xRef)
            for (i, k) in enumerate(grids[j])
                w, wp, wpp = ws[i], wps[i], wpps[i]
                a = 1 - k * wp / (2w)
                gRef = gatheralG(k, w, wp, wpp)
                gw = a * k * wp / w^2 + wp^2 / (4w^2)
                gwp = -a * k / w - wp / 2 * (1 / w + one(T) / 4)
                #g ≈ gRef + gw (w-wRef) + gwp (w'-w'Ref) + (w''-w''Ref)/2 ≥ gMin (or the wing margin)
                gT = gTarget(j, k)
                row = zeros(T, nVar)
                row[idx] = tte .* (gw .* view(F, i, :) .+ gwp .* view(Fp, i, :) .+ view(Fpp, i, :) ./ 2)
                push!(bRows, row)
                push!(bRhs, (relax ? min(gT, gRef) : gT) - gRef + gw * w + gwp * wp + wpp / 2)
            end
            if linearWings && edgeCondition
                #lower branch at the edge knots: |w'| ≤ R(w) = 2w/(|k|+√(w+w²/4)), linearized in w
                f, fp, fpp = zeros(T, np[j]), zeros(T, np[j]), zeros(T, np[j])
                for (k, sgn) in zip(edges(b), (-one(T), one(T)))
                    basisRows!(f, fp, fpp, b, k)
                    w = tte * dot(f, xRef)
                    sw = sqrt(w + w^2 / 4)
                    R = 2w / (abs(k) + sw)
                    Rp = (2 * (abs(k) + sw) - 2w * (1 + w / 2) / (2sw)) / (abs(k) + sw)^2
                    #R'w - sgn w' ≥ R'w_ref - R (1-leeMargin)
                    row = zeros(T, nVar)
                    row[idx] = tte .* (Rp .* f .- sgn .* fp)
                    rhsValue = Rp * w - R * (1 - leeMargin)
                    push!(bRows, row)
                    push!(bRhs, relax ? min(rhsValue, dot(row, sol)) : rhsValue)
                end
            end
        end
        return bRows, bRhs
    end
    function runSQP()
        fixedRows, fixedRhs = copy(rows), copy(rhs)
        if calendar
            for j = 1:nExp-1, k in calendarPoints[j]
                addCalendarRow!(fixedRows, fixedRhs, j, k)
            end
        end
        sol, crval = solve(fixedRows, fixedRhs)
        slices = makeSlices(sol)
        minG = [minGOf(slices)]
        objective = [crval]
        verbose && println("iteration 0 min g ", minG[end], " objective ", crval)
        local it = 0
        best = (sol, slices, crval)
        bestFeasible = !butterfly || minG[end] >= 0
        if butterfly
            for iteration = 1:iterations
                bRows, bRhs = linearizedRows(sol, fixedRows, fixedRhs, gridMatrices())
                newSol, crval = try
                    solve(bRows, bRhs)
                catch e
                    verbose && println("iteration ", iteration, " failed: ", e)
                    break
                end
                Δx = maximum(abs, qpCoordinates(newSol) - qpCoordinates(sol)) / maximum(abs, qpCoordinates(newSol))
                sol = newSol
                it = iteration
                slices = makeSlices(sol)
                push!(minG, minGOf(slices))
                push!(objective, crval)
                verbose && println("iteration ", iteration, " min g ", minG[end], " objective ", crval, " Δx ", Δx)
                feasible = minG[end] >= 0
                if feasible && (!bestFeasible || crval < best[3])
                    best = (sol, slices, crval)
                    bestFeasible = true
                elseif !bestFeasible
                    best = (sol, slices, crval)
                end
                (Δx < xTol || (stopAtFeasible && feasible)) && break
            end
        end
        return best[2], it, minG, objective
    end
    #feasible SQP: the flat surface v = σ̄² is free of arbitrage and satisfies all the linear constraints; every iterate is
    #kept feasible on the grids by pulling back towards it and by backtracking along the QP steps
    pFlat = zeros(T, nVar)
    for j = 1:nExp
        pFlat[offsets[j]+1] = σbar2
    end
    objectiveOf(q) = -dot(d, q) + dot(q, D * q) / 2
    #g ≥ level times the margin (gMin, or the wing margin) on the grids
    function gridFeasible(q, level, matrices, A=nothing, b=nothing)
        #linear constraints, in case the QP solution is inexact (Clarabel's reduced accuracy)
        if A !== nothing
            minimum(transpose(A) * q .- b) >= -1e-10 * (1 + maximum(abs, b)) || return false
        end
        for j = 1:nExp
            x = view(q, offsets[j]+1:offsets[j+1])
            F, Fp, Fpp = matrices[j]
            tte = ttes[j]
            ws, wps, wpps = tte .* (F * x), tte .* (Fp * x), tte .* (Fpp * x)
            for (i, k) in enumerate(grids[j])
                ws[i] > 0 || return false
                gatheralG(k, ws[i], wps[i], wpps[i]) >= level * gTarget(j, k) || return false
            end
            if linearWings && edgeCondition
                sl = VarianceSlice(bases[j], q[offsets[j]+1:offsets[j+1]], ttes[j], forwards[j])
                for k in edges(bases[j])
                    v, vp, _ = varianceDerivatives(sl, k)
                    w, wp = ttes[j] * v, ttes[j] * vp
                    abs(wp) <= 2w / (abs(k) + sqrt(w + w^2 / 4)) || return false
                end
            end
        end
        return true
    end
    function linearFeasibleStep(q, fixedRows, fixedRhs)
        sMax = one(T)
        for (r, b) in zip(fixedRows, fixedRhs)
            a0, a1 = dot(r, pFlat), dot(r, q)
            if a1 < b - 1e-12 * (1 + abs(b))
                sMax = a0 > a1 ? min(sMax, (a0 - b) / (a0 - a1)) : zero(T)
            end
        end
        return max(sMax * (1 - sqrt(eps(T))), zero(T))
    end
    function runFeasibleSQP(pStart)
        fixedRows, fixedRhs = copy(rows), copy(rhs)
        if calendar
            for j = 1:nExp-1, k in calendarPoints[j]
                addCalendarRow!(fixedRows, fixedRhs, j, k)
            end
        end
        if !butterfly
            #only linear constraints: the QP solution is the answer
            sol = pStart
            slices = makeSlices(sol)
            return slices, 0, [minGOf(slices)], [objectiveOf(sol)], sol
        end
        matrices = gridMatrices()
        sPull = linearFeasibleStep(pStart, fixedRows, fixedRhs)
        for _ = 1:80
            gridFeasible(pFlat .+ sPull .* (pStart .- pFlat), one(T), matrices) && break
            sPull /= 1.25
        end
        sol = pFlat .+ sPull .* (pStart .- pFlat)
        fixedA = reduce(hcat, fixedRows)
        objective = [objectiveOf(sol)]
        minG = trackIterations ? [minGOf(makeSlices(sol))] : T[]
        verbose && println("pull back ", sPull)
        local it = 0
        for iteration = 1:iterations
            #QP linearized with the target gMin; if no step along its solution is acceptable (the current iterate may violate
            #the linearization), QP relaxed so that the current iterate is feasible
            objRef = objectiveOf(sol)
            accepted = false
            step = one(T)
            pQP = sol
            for relax in (false, true)
                bRows, bRhs = linearizedRows(sol, fixedRows, fixedRhs, matrices, relax=relax)
                pQP = try
                    solve(bRows, bRhs)[1]
                catch e
                    verbose && println("iteration ", iteration, " relax=", relax, " failed: ", e)
                    continue
                end
                step = one(T)
                for _ = 1:30
                    q = sol .+ step .* (pQP .- sol)
                    if objectiveOf(q) <= objRef + 1e-14 * abs(objRef) && gridFeasible(q, T(0.25), matrices, fixedA, fixedRhs)
                        accepted = true
                        break
                    end
                    step /= 2
                end
                accepted && break
            end
            accepted || break
            newSol = sol .+ step .* (pQP .- sol)
            #stationarity: the QP solution is (nearly) the current iterate
            Δx = maximum(abs, qpCoordinates(pQP) - qpCoordinates(sol)) / maximum(abs, qpCoordinates(pQP))
            sol = newSol
            it = iteration
            push!(objective, objectiveOf(sol))
            trackIterations && push!(minG, minGOf(makeSlices(sol)))
            verbose && println("iteration ", iteration, " objective ", objective[end], " step ", step, " Δx ", Δx)
            #stationary, or no more progress on the objective
            (Δx < xTol || objRef - objective[end] <= 1e-10 * abs(objRef)) && break
        end
        slices = makeSlices(sol)
        trackIterations || push!(minG, minGOf(slices))
        return slices, it, minG, objective, sol
    end
    currentSol = zeros(T, nVar)
    if method == :feasible
        startRows, startRhs = copy(rows), copy(rhs)
        if calendar
            for j = 1:nExp-1, k in calendarPoints[j]
                addCalendarRow!(startRows, startRhs, j, k)
            end
        end
        p0, _ = solve(startRows, startRhs)
        #optional full steps of the linearized QP before the feasible iteration, which bring the start closer to the
        #arbitrage-free set and reduce the pull-back towards the flat surface
        if butterfly && warmStart > 0
            warmMatrices = gridMatrices()
            for _ = 1:warmStart
                wRows, wRhs = linearizedRows(p0, startRows, startRhs, warmMatrices)
                pNext = try
                    solve(wRows, wRhs)[1]
                catch e
                    verbose && println("warm start step failed: ", e)
                    break
                end
                p0 = pNext
            end
        end
        slices, it, minG, objective, currentSol = runFeasibleSQP(p0)
    else
        slices, it, minG, objective = runSQP()
    end
    #exchange method: add the local minima of g and of the calendar spreads that are negative on the check grids, and solve again
    for exchangeIteration = 1:exchange
        added = 0
        if butterfly
            for j = 1:nExp
                newPoints = lowMinima(gValues(j, slices[j].x) .- gTarget.(j, checkGrids[j]) ./ 2, checkGrids[j], zero(T))
                grids[j] = sort(unique(vcat(grids[j], newPoints)))
                added += length(newPoints)
            end
        end
        if calendar
            for j = 1:nExp-1
                newPoints = lowMinima(calendarValues(slices, j), calendarChecks[j], calendarMin[j] / 2)
                append!(calendarPoints[j], newPoints)
                added += length(newPoints)
            end
        end
        added == 0 && break
        verbose && println("exchange ", exchangeIteration, ": ", added, " points added")
        if method == :feasible
            slices, it2, minG2, objective2, currentSol = runFeasibleSQP(currentSol)
        else
            slices, it2, minG2, objective2 = runSQP()
        end
        it += it2
        append!(minG, minG2)
        append!(objective, objective2)
    end
    nC = nFixed + (calendar ? sum(length.(calendarPoints); init=0) : 0) + (butterfly && it > 0 ? sum(length.(grids)) : 0)
    return CVISurface(slices), CVICalibrationInfo(it, minG, objective, nC, nVar)
end
