#B-spline Andreasen-Huge: the Andreasen-Huge discretization of the forward Dupire equation (one implicit step per expiry
#interval, each step starting from the prices of the previous expiry, the payoff for the first one), with a smooth
#representation of the local volatility of each step (a quadratic B-spline with positive coefficients, with the C3
#condition at the forward on the first step, as for the quadratic LVG) instead of piecewise constant values on the grid.
import BSplines: BSplineBasis, Derivative
import AQFED.Math
import AQFED.Black: impliedVolatilityThiophene
import SpecialFunctions: erfc

"""
    BSplineAHSurface

Implied volatility surface of the B-spline Andreasen-Huge model (one implicit step per expiry interval), in forward moneyness x = K/F(T) and normalized call prices
c = C/F(T) (no drift with proportional dividends). Each expiry Tᵢ solves on the grid `z` (which contains the forward and
all the quoted strikes)

    cᵢ - ½ (Tᵢ - Tᵢ₋₁) σᵢ(x)² x² cᵢ'' = cᵢ₋₁,    c₀ = (1 - x)⁺, T₀ = 0,

with σᵢ a quadratic B-spline on `knots[i]` (flat beyond them): cᵢ ≥ cᵢ₋₁ (no calendar arbitrage) and cᵢ is convex in x (no
butterfly arbitrage on the grid). Between Tᵢ₋₁ and Tᵢ, the prices are those of a step of size t - Tᵢ₋₁ with the same σᵢ,
so that σᵢ is the Dupire local volatility on (Tᵢ₋₁, Tᵢ].

The first step starts from the payoff, whose kink at the forward makes a kink (a peak) in the density unless the local
volatility has a matching kink: with a = ½σ²x²T₁ and p = c'' the density, (a p)'' = p - δ(x-1), and p is C¹ at the
forward when the derivative of σ jumps there by -σ(1)/(2c(1)) (the C3 condition of the quadratic LVG). The first spline has
a double knot at the forward and its coefficient at the forward is set by this condition.
"""
struct BSplineAHSurface{T}
    expiries::Vector{T}
    forwards::Vector{T}
    z::Vector{T}
    knots::Vector{Vector{T}}  # knots[i]: knots of the local volatility spline of the step to expiry i
    coeffs::Vector{Vector{T}} # coeffs[i]: its B-spline coefficients
    prices::Vector{Vector{T}} # normalized call prices on z at each expiry
    priceMap::Symbol          # :grid, :density, :box or :schaback (see calibrateBSplineAHSurface)
    cache::Dict{Int,Any}      # price maps of the expiries built after the calibration (:schaback)
    scheme::Symbol            # :adjoint (Fokker-Planck masses, default) or :forward (forward Dupire prices)
    masses::Vector{Vector{T}} # :adjoint, the masses of the grid points at each expiry (empty for :forward)
end
Base.broadcastable(p::BSplineAHSurface) = Ref(p)

ahBasis(knots) = BSplineBasis(3, knots)
"local volatility of a step at x (flat beyond the knots)"
function ahLocalVol(knots, coeffs, x)
    xc = min(max(x, knots[1]), knots[end])
    return BSplines.Spline(ahBasis(knots), coeffs)(xc)
end
"matrix of the B-spline basis functions at the points z (flat beyond the knots)"
function ahBasisMatrix(knots, z)
    basis = ahBasis(knots)
    B = zeros(length(z), length(basis))
    for (j, zj) in enumerate(z)
        xc = min(max(zj, knots[1]), knots[end])
        for k = 1:length(basis)
            B[j, k] = basis[k](xc)
        end
    end
    return B
end

"""
one implicit step on the grid z (log-moneyness discretization, non-uniform), Dirichlet boundaries equal to the previous
prices; the off-diagonal coefficients are non-positive (an M-matrix) when the log spacing is below 2.
"""
function ahStep(z::AbstractVector{T}, prev::AbstractVector, σ::AbstractVector{TS}, dt; exactForward=false) where {T,TS}
    n = length(z)
    TV = promote_type(TS, eltype(prev))
    dl = zeros(TV, n - 1); d = ones(TV, n); du = zeros(TV, n - 1)
    rhs = Vector{TV}(prev)
    for j = 2:n-1
        hm = log(z[j] / z[j-1]); hp = log(z[j+1] / z[j]); H = hm + hp
        κ = exactForward ? ((exp(hp) - exp(-hm)) / H) / (2 / H * (expm1(hp) / hp + expm1(-hm) / hm)) : one(T)
        a = σ[j]^2 * dt / 2
        dl[j-1] = -a * (2κ / (H * hm) + 1 / H)
        du[j] = -a * (2κ / (H * hp) - 1 / H)
        d[j] = 1 + a * 2κ / H * (1 / hm + 1 / hp)
    end
    #Thomas algorithm (diagonally dominant: no pivoting needed)
    for j = 2:n
        w = dl[j-1] / d[j-1]
        d[j] -= w * du[j-1]
        rhs[j] -= w * rhs[j-1]
    end
    c = similar(rhs)
    c[n] = rhs[n] / d[n]
    for j = n-1:-1:1
        c[j] = (rhs[j] - du[j] * c[j+1]) / d[j]
    end
    return c
end

"""
one implicit step of the discrete Fokker-Planck equation adjoint to the backward scheme in the spot (log-moneyness,
non-uniform, absorbing boundaries): Aᵀ P = Pprev, with A = I - diag(a) (κ ⊙ D_yy - D_y), a = ½σ²dt. The factor
κⱼ = D_y(eʸ)ⱼ / D_yy(eʸ)ⱼ (exact forward, Healy's book) makes A eʸ = eʸ; as A 1 = 1, the masses keep their sum (1) and
their mean (the forward 1) exactly. A is an M-matrix when the log spacing is below 2: the masses stay positive.
"""
function ahAdjointStep(z::AbstractVector{T}, Pprev::AbstractVector, σ::AbstractVector{TS}, dt) where {T,TS}
    n = length(z)
    TV = promote_type(TS, eltype(Pprev))
    dl = zeros(TV, n - 1); d = ones(TV, n); du = zeros(TV, n - 1) # rows of A
    for j = 2:n-1
        hm = log(z[j] / z[j-1]); hp = log(z[j+1] / z[j]); H = hm + hp
        κ = ((exp(hp) - exp(-hm)) / H) / (2 / H * (expm1(hp) / hp + expm1(-hm) / hm))
        a = σ[j]^2 * dt / 2
        dl[j-1] = -a * (2κ / (H * hm) + 1 / H)
        du[j] = -a * (2κ / (H * hp) - 1 / H)
        d[j] = 1 + a * 2κ / H * (1 / hm + 1 / hp)
    end
    #Thomas algorithm on the transpose: sub-diagonal du, super-diagonal dl
    rhs = Vector{TV}(Pprev)
    for j = 2:n
        w = du[j-1] / d[j-1]
        d[j] -= w * dl[j-1]
        rhs[j] -= w * rhs[j-1]
    end
    P = similar(rhs)
    P[n] = rhs[n] / d[n]
    for j = n-1:-1:1
        P[j] = (rhs[j] - dl[j] * P[j+1]) / d[j]
    end
    return P
end
"""
interpolated density of the Fokker-Planck masses P: the density fⱼ = Pⱼ / (½(zⱼ₊₁ - zⱼ₋₁)) at the inner grid points, zero
at the two ends, linear in x in between (each inner mass becomes the hat function of its grid point, of mass exactly Pⱼ),
tilted by a + b x to the mass and the mean of the inner masses; the masses absorbed at the two ends stay point masses.
Positive (for a small tilt), continuous.
"""
function ahMassDensityMap(z::AbstractVector, P::AbstractVector{W}) where {W}
    n = length(z)
    f = [j == 1 || j == n ? zero(W) : P[j] / ((z[j+1] - z[j-1]) / 2) for j = 1:n]
    inner = 2:n-1
    return ahDensityMap(z, f; mass=sum(P[inner]), mean=sum(P[j] * z[j] for j in inner), top=P[n])
end
"normalized call prices on the grid from the masses P of the grid points: cₖ = Σⱼ Pⱼ (zⱼ - zₖ)⁺, in one pass"
function ahMassPrices(z::AbstractVector, P::AbstractVector{W}) where {W}
    n = length(z)
    c = zeros(W, n)
    A = zero(W); B = zero(W) # Σ_{j>k} Pⱼ zⱼ and Σ_{j>k} Pⱼ
    for k = n-1:-1:1
        A += P[k+1] * z[k+1]; B += P[k+1]
        c[k] = A - z[k] * B
    end
    return c
end

"""
data of the C3 condition on the first spline (double knot at the forward): index `kc` of the basis function equal to one
at the forward, the jumps `D` of the first derivatives of the basis functions there; the condition reads
Σⱼ coefⱼ Dⱼ + coef_kc / (2 c(1)) = 0.
"""
function ahC3Data(knotsFirst)
    basis = ahBasis(knotsFirst)
    h = 1e-9
    D = [basis[k](1 + h, Derivative(1)) - basis[k](1 - h, Derivative(1)) for k = 1:length(basis)]
    kc = argmax([basis[k](1.0) for k = 1:length(basis)])
    return kc, D
end
function ahApplyC3!(coef, kc, D, cAtm)
    s = zero(eltype(coef))
    for k in eachindex(coef)
        k == kc || (s += coef[k] * D[k])
    end
    coef[kc] = -s / (D[kc] + 1 / (2cAtm))
    return coef
end

"knots of the first step: the double knot at the forward, and knots close enough for the C3 condition to be well posed"
function ahFirstKnots(knots::AbstractVector{T}, cAtm) where {T}
    #a knot much closer to the forward than the spacing below (a uniform point that happens to fall next to it) would make
    #a B-spline interval of a fraction of cAtm next to the double knot, where the local volatility bends abruptly, and the
    #density a spike at the forward despite the C3 condition: such knots are dropped
    k = sort(vcat(filter(x -> abs(x - 1) > cAtm / 2, knots), one(T), one(T)))
    #the coefficient at the forward is -Σ/(D_kc + 1/(2cAtm)) with D_kc ≈ -2/h₋ - 2/h₊: the knot spacing around the forward
    #must be well below 8 cAtm (as adjustKnots! for the quadratic LVG)
    h = 2cAtm
    i = searchsortedfirst(k, one(T))
    if i > 1 && 1 - k[i-1] > h
        push!(k, 1 - h)
    end
    if i + 2 <= length(k) && k[i+2] - 1 > h
        push!(k, 1 + h)
    end
    return sort(k)
end

#price map :density: the option prices as the exact integral of a positive density, the density of the step at the
#nodes, M_j = (c_j - c_prev,j) / (dt σ_j² z_j² / 2), linear between them, tilted by (a + b x) to mass 1 and mean 1
#(3-point Gauss-Legendre per cell, exact for these polynomials). Butterfly arbitrage free by construction.
const AH_GX = (-sqrt(3 / 5), 0.0, sqrt(3 / 5)); const AH_GW = (5 / 9, 8 / 9, 5 / 9)
function ahDensityMap(zk::AbstractVector{T}, M::AbstractVector{W}; mass=one(W), mean=one(W), top=zero(W)) where {T,W}
    m = length(zk)
    plin(j, x) = M[j] + (M[j+1] - M[j]) * (x - zk[j]) / (zk[j+1] - zk[j])
    gauss(f, l, r) = (r - l) / 2 * sum(AH_GW[q] * f((l + r) / 2 + (r - l) / 2 * AH_GX[q]) for q in 1:3)
    m0 = zero(W); m1 = zero(W); m2 = zero(W)
    for j = 1:m-1
        m0 += gauss(x -> plin(j, x), zk[j], zk[j+1]); m1 += gauss(x -> x * plin(j, x), zk[j], zk[j+1]); m2 += gauss(x -> x^2 * plin(j, x), zk[j], zk[j+1])
    end
    #tilt a + b x: a m0 + b m1 = mass, a m1 + b m2 = mean
    b = (mass * m1 - mean * m0) / (m1^2 - m0 * m2); a = (mass - b * m1) / m0
    A = zeros(W, m); B = zeros(W, m) # ∫ from zk[j] to the upper bound of x q and of q
    for j = m-1:-1:1
        A[j] = A[j+1] + gauss(x -> x * plin(j, x) * (a + b * x), zk[j], zk[j+1])
        B[j] = B[j+1] + gauss(x -> plin(j, x) * (a + b * x), zk[j], zk[j+1])
    end
    return (zk=zk, M=M, a=a, b=b, A=A, B=B, top=top) # top: a point mass at zk[end]
end
function ahDensityPrice(d, K)
    zk = d.zk; m = length(zk)
    K <= zk[1] && return d.A[1] - K * d.B[1] + d.top * (zk[end] - K)
    K >= zk[end] && return zero(d.a)
    j = clamp(searchsortedlast(zk, K), 1, m - 1)
    plin(x) = d.M[j] + (d.M[j+1] - d.M[j]) * (x - zk[j]) / (zk[j+1] - zk[j])
    l, r = K, zk[j+1]
    tail = (r - l) / 2 * sum(AH_GW[q] * (x -> (x - K) * plin(x) * (d.a + d.b * x))((l + r) / 2 + (r - l) / 2 * AH_GX[q]) for q in 1:3)
    return d.A[j+1] - K * d.B[j+1] + tail + d.top * (zk[end] - K)
end
function ahDensityAt(d, x)
    zk = d.zk
    (x <= zk[1] || x >= zk[end]) && return zero(d.a)
    j = clamp(searchsortedlast(zk, x), 1, length(zk) - 1)
    return (d.M[j] + (d.M[j+1] - d.M[j]) * (x - zk[j]) / (zk[j+1] - zk[j])) * (d.a + d.b * x)
end
#the density map of a step from the grid prices: c (time t), cPrev (previous expiry or payoff), σ on the grid, dt
function ahStepDensityMap(z, keep, c, cPrev, σ, dt)
    M = [max((c[j] - cPrev[j]) / (dt * σ[j]^2 * z[j]^2 / 2), zero(c[j])) for j in keep]
    return ahDensityMap(z[keep], M)
end

#price map :box (the more precise estimate of the Andreasen-Huge section of Healy's book): the masses πⱼ of the grid
#points (the jumps of the slope of the grid prices, which reproduce them exactly) are spread over boxes between the
#mid-points, each with the same mass and mean (linear, or two constant levels where the box is too asymmetric for a
#positive linear piece); prices for strikes outside a box are those of the point masses exactly, the density is positive,
#piecewise linear or constant, possibly discontinuous. Best on a log-uniform grid (nearly symmetric boxes, smooth masses).
#With the masses P of the scheme (P₁ the point mass absorbed at the lower boundary), the box masses are taken from them
#rather than from the second differences of the prices, and out-of-the-money puts are priced from the masses below the
#strike (ahBoxPut), to full relative precision: c - (1 - K) loses everything below the rounding of c, about 1e-16.
function ahBoxMap(z::AbstractVector{T}, c::AbstractVector{W}; P=nothing) where {T,W}
    n = length(z)
    m = n - 2
    π = P === nothing ? [(c[j+1] - c[j]) / (z[j+1] - z[j]) - (c[j] - c[j-1]) / (z[j] - z[j-1]) for j = 2:n-1] : W.(P[2:n-1])
    #the point mass at the lower boundary: one plus the slope of the first segment, -∫_{z₁⁺} q
    P1 = P === nothing ? 1 + (c[2] - c[1]) / (z[2] - z[1]) : W(P[1])
    zc = z[2:n-1] #the mean of each box: its grid point
    lo = [(z[j-1] + z[j]) / 2 for j = 2:n-1]; hi = [(z[j] + z[j+1]) / 2 for j = 2:n-1]
    lo[1] = z[1]; hi[end] = z[end]
    cc = (lo .+ hi) ./ 2
    #linear density α + β (x - c) with mass π and mean z: β = 12 π (z - c) / w³; positive on the box when
    #|β| w / 2 <= |α|, i.e. when the grid point is within w/6 of the centre of its box; otherwise (asymmetric box, as next to
    #a nearly coincident grid point), two constant levels a on [lo, z] and b on [z, hi], with a (z - lo) = π (hi - z) / w and
    #b (hi - z) = π (z - lo) / w: exact mass and mean, positive
    α = [π[k] / (hi[k] - lo[k]) for k = 1:m]
    β = [12π[k] * (zc[k] - cc[k]) / (hi[k] - lo[k])^3 for k = 1:m]
    lin = [abs(β[k]) * (hi[k] - lo[k]) / 2 <= abs(α[k]) for k = 1:m]
    la = [lin[k] ? zero(W) : π[k] * (hi[k] - zc[k]) / ((hi[k] - lo[k]) * (zc[k] - lo[k])) for k = 1:m]
    lb = [lin[k] ? zero(W) : π[k] * (zc[k] - lo[k]) / ((hi[k] - lo[k]) * (hi[k] - zc[k])) for k = 1:m]
    #an antiderivative in x of (x - K) q_k(x)
    G(x, K) = x^2 / 2 - K * x
    prim(k, x, K) = lin[k] ? α[k] * G(x, K) + β[k] * (x^3 / 3 - (K + cc[k]) * x^2 / 2 + K * cc[k] * x) :
                    la[k] * G(min(x, zc[k]), K) + lb[k] * (G(max(x, zc[k]), K) - G(zc[k], K))
    A = zeros(W, m + 1); B = zeros(W, m + 1) # tails from box k: ∫ x q = Σ π z (mean kept) and ∫ q = Σ π
    for k = m:-1:1
        A[k] = A[k+1] + π[k] * zc[k]
        B[k] = B[k+1] + π[k]
    end
    Al = zeros(W, m + 1); Bl = zeros(W, m + 1) # tails from below, boxes 1 to k - 1: Σ π z and Σ π
    for k = 2:m+1
        Al[k] = Al[k-1] + π[k-1] * zc[k-1]
        Bl[k] = Bl[k-1] + π[k-1]
    end
    #the grid prices are, exactly, the affine function of the last segment plus the point masses:
    #c(x) = c[n] + s (x - z[n]) + Σⱼ πⱼ (zⱼ - x)⁺, with s the slope of the last segment
    s = (c[n] - c[n-1]) / (z[n] - z[n-1])
    return (lo=lo, hi=hi, zc=zc, α=α, β=β, cc=cc, lin=lin, la=la, lb=lb, A=A, B=B, Al=Al, Bl=Bl, P1=P1, z1=z[1], cn=c[n], s=s, zn=z[n], prim=prim)
end
"normalized out-of-the-money put price of the box map, E[(K - x)⁺], from the masses below K (K below the last box)"
function ahBoxPut(d, K)
    K <= d.z1 && return zero(K) * d.cn
    K >= d.hi[end] && return ahBoxPrice(d, K) - (1 - K)
    put = d.P1 * (K - d.z1)
    K <= d.lo[1] && return put
    k = clamp(searchsortedlast(d.lo, K), 1, length(d.lo))
    #the boxes below k exactly, and the part of box k below K: ∫_{lo}^{K} (K - x) q = -(prim(K) - prim(lo))
    return put + K * d.Bl[k] - d.Al[k] - (d.prim(k, K, K) - d.prim(k, d.lo[k], K))
end
function ahBoxPrice(d, K)
    K >= d.hi[end] && return d.cn + d.s * (K - d.zn)
    aff = d.cn + d.s * (K - d.zn)
    K <= d.lo[1] && return aff + d.A[1] - K * d.B[1]
    k = clamp(searchsortedlast(d.lo, K), 1, length(d.lo))
    return aff + d.A[k+1] - K * d.B[k+1] + d.prim(k, d.hi[k], K) - d.prim(k, K, K)
end
function ahBoxDensityAt(d, x)
    (x <= d.lo[1] || x >= d.hi[end]) && return zero(d.cn)
    k = clamp(searchsortedlast(d.lo, x), 1, length(d.lo))
    return d.lin[k] ? d.α[k] + d.β[k] * (x - d.cc[k]) : (x < d.zc[k] ? d.la[k] : d.lb[k])
end

#price map :schaback (output only, the calibration uses the grid prices): the convexity preserving rational spline of
#Schaback through the grid prices, on the grid points whose out-of-the-money price exceeds 1e-13 and where the grid
#prices are convex; outside, the cubic spline of the grid prices
function ahSchabackMap(z, c)
    keep = [j for j in eachindex(z) if (z[j] >= 1 ? c[j] : c[j] - (1 - z[j])) > 1e-13]
    #the spline needs convex data: drop the middle point of every non-convex triple (deep in the money the grid prices
    #are convex only up to the discretisation of the drift term, about 1e-7), until the prices left are convex
    while true
        zk = z[keep]; δ = diff(c[keep]) ./ diff(zk)
        bad = findfirst(k -> δ[k+1] <= δ[k], 1:length(δ)-1)
        bad === nothing && break
        deleteat!(keep, bad + 1)
    end
    zk = z[keep]; ck = c[keep]
    δ = diff(ck) ./ diff(zk)
    #end slopes outside the outer chords (convex), half a chord difference away: the algorithm needs a positive curvature
    sp = Math.makeConvexSchabackRationalSpline(zk, ck, Math.FirstDerivativeBoundary(δ[1] - (δ[2] - δ[1]) / 2),
        Math.FirstDerivativeBoundary(δ[end] + (δ[end] - δ[end-1]) / 2); iterations=200)
    fb = ahNaturalSpline(z, c)
    return (sp=sp, lo=zk[1], hi=zk[end], fallback=x -> ahSplineAt(fb, x))
end
ahSchabackPrice(d, x) = d.lo < x < d.hi ? Math.evaluate(d.sp, x) : d.fallback(x)

#price map :variance (output only, the calibration uses the grid prices): the grid prices are inverted into implied
#volatilities (out-of-the-money option, ThiopheneIV), the total variance is interpolated by a natural cubic spline in
#log-moneyness, and the prices follow from the Black formula. Grid points whose out-of-the-money price is below 1e-14
#(beyond the reach of the inversion) are left out.
function ahVarianceMap(z::AbstractVector{T}, c::AbstractVector, t) where {T}
    otm(i) = z[i] >= 1 ? c[i] : c[i] - (1 - z[i])
    keep = [i for i in eachindex(z) if otm(i) > 1e-14]
    w = [impliedVolatilityThiophene(z[i] >= 1, Float64(otm(i)), 1.0, Float64(z[i]), Float64(t), 1.0)^2 * t for i in keep]
    return (w=CubicSplineNatural(log.(z[keep]), w),)
end
ahVariancePrice(d, x) = blackScholesFormula(true, x, one(x), max(d.w(log(x)), zero(x)), one(x), one(x))

#natural cubic spline of the values c on the nodes z, generic in the type of c (automatic differentiation): second
#derivatives at the nodes, then evaluation
function ahNaturalSpline(z::AbstractVector{T}, c::AbstractVector{W}) where {T,W}
    n = length(z)
    h = diff(z)
    #tridiagonal system for the second derivatives M (M₁ = Mₙ = 0)
    dl = zeros(W, n - 3); d = zeros(W, n - 2); du = zeros(W, n - 3); r = zeros(W, n - 2)
    for j = 2:n-1
        k = j - 1
        d[k] = (h[j-1] + h[j]) / 3
        k > 1 && (dl[k-1] = h[j-1] / 6)
        k < n - 2 && (du[k] = h[j] / 6)
        r[k] = (c[j+1] - c[j]) / h[j] - (c[j] - c[j-1]) / h[j-1]
    end
    for k = 2:n-2 #Thomas
        w = dl[k-1] / d[k-1]; d[k] -= w * du[k-1]; r[k] -= w * r[k-1]
    end
    M = zeros(W, n)
    M[n-1] = r[n-2] / d[n-2]
    for k = n-3:-1:1
        M[k+1] = (r[k] - du[k] * M[k+2]) / d[k]
    end
    return (z=z, c=c, M=M)
end
function ahSplineAt(s, x)
    z = s.z; j = clamp(searchsortedlast(z, x), 1, length(z) - 1); h = z[j+1] - z[j]; t = (x - z[j]) / h
    return (1 - t) * s.c[j] + t * s.c[j+1] + h^2 / 6 * (((1 - t)^3 - (1 - t)) * s.M[j] + (t^3 - t) * s.M[j+1])
end

"""
C2 quintic Hermite spline of the grid prices c on the grid z: the second derivatives at the inner grid points are the
second divided differences (the discrete density of the scheme, 2Pⱼ/(zⱼ₊₁-zⱼ₋₁)), zero at the ends; the slopes are those
of the natural cubic spline of the prices (slopes=:spline), those which minimize ∫c'''² for the given values and second
derivatives (slopes=:smooth, the smoothest density, a symmetric tridiagonal system), or the derivative of the parabola
through three points (slopes=:parabola, local, for comparison: its O(h²) error makes an O(h) density noise).
"""
function ahQuinticSpline(z::AbstractVector{T}, c::AbstractVector{W}; slopes::Symbol=:spline) where {T,W}
    n = length(z)
    s = zeros(W, n)
    for j = 2:n-1
        s[j] = 2 * ((c[j+1] - c[j]) / (z[j+1] - z[j]) - (c[j] - c[j-1]) / (z[j] - z[j-1])) / (z[j+1] - z[j-1])
    end
    d = zeros(W, n)
    if slopes == :spline
        sp = ahNaturalSpline(z, c)
        for j = 1:n-1
            h = z[j+1] - z[j]
            d[j] = (c[j+1] - c[j]) / h - h * (2sp.M[j] + sp.M[j+1]) / 6
        end
        h = z[n] - z[n-1]
        d[n] = (c[n] - c[n-1]) / h + h * (sp.M[n-1] + 2sp.M[n]) / 6
    elseif slopes == :smooth
        #∫c'''² over each interval by the 3-point Gauss-Legendre rule (exact: c''' is quadratic in t), c''' affine in (dⱼ, dⱼ₊₁)
        dl = zeros(W, n - 1); dd = zeros(W, n); du = zeros(W, n - 1); r = zeros(W, n)
        gt = (0.5 - sqrt(15) / 10, 0.5, 0.5 + sqrt(15) / 10); gw = (5 / 18, 8 / 18, 5 / 18)
        for j = 1:n-1
            h = z[j+1] - z[j]
            for (t, w) in zip(gt, gw)
                v0 = -(36 - 192t + 180t^2) / h^2
                v1 = -(24 - 168t + 180t^2) / h^2
                f = 60 * (c[j+1] - c[j]) * (6t^2 - 6t + 1) / h^3 - (s[j] * (9 - 36t + 30t^2) - s[j+1] * (3 - 24t + 30t^2)) / h
                dd[j] += w * h * v0^2; dd[j+1] += w * h * v1^2
                du[j] += w * h * v0 * v1; dl[j] += w * h * v0 * v1
                r[j] -= w * h * v0 * f; r[j+1] -= w * h * v1 * f
            end
        end
        d = Tridiagonal(dl, dd, du) \ r
    else
        d[1] = (c[2] - c[1]) / (z[2] - z[1]); d[n] = (c[n] - c[n-1]) / (z[n] - z[n-1])
        for j = 2:n-1
            hm, hp = z[j] - z[j-1], z[j+1] - z[j]
            d[j] = (hm^2 * (c[j+1] - c[j]) + hp^2 * (c[j] - c[j-1])) / (hm * hp * (hm + hp))
        end
    end
    return (z=z, c=c, d=d, s=s)
end
function ahQuinticAt(q, x)
    z = q.z; j = clamp(searchsortedlast(z, x), 1, length(z) - 1); h = z[j+1] - z[j]; t = (x - z[j]) / h; u = 1 - t
    H0(t) = 1 - t^3 * (10 - 15t + 6t^2); H1(t) = t - t^3 * (6 - 8t + 3t^2); H2(t) = t^2 * (1 - 3t + 3t^2 - t^3) / 2
    return q.c[j] * H0(t) + q.c[j+1] * H0(u) + h * (q.d[j] * H1(t) - q.d[j+1] * H1(u)) + h^2 * (q.s[j] * H2(t) + q.s[j+1] * H2(u))
end
"cardinal B-spline of degree p (1, 2 or 3), centered at 0, support [-(p+1)/2, (p+1)/2]"
function cardinalBSpline(p::Int, u)
    a = abs(u)
    if p == 1
        return a < 1 ? 1 - a : zero(u)
    elseif p == 2
        return a < 0.5 ? 0.75 - a^2 : a < 1.5 ? (1.5 - a)^2 / 2 : zero(u)
    end
    return a < 1 ? (4 - 6a^2 + 3a^3) / 6 : a < 2 ? (2 - a)^3 / 6 : zero(u)
end
const AH_GAUSS5 = ((-0.906179845938664, 0.236926885056189), (-0.538469310105683, 0.478628670499366), (0.0, 0.568888888888889),
    (0.538469310105683, 0.478628670499366), (0.906179845938664, 0.236926885056189))

"""
kernel density of the Fokker-Planck masses P on the grid z: each inner mass Pⱼ is spread with the cardinal B-spline of
degree p in y = log x, of width the local spacing hⱼ = (yⱼ₊₁ - yⱼ₋₁)/2, centered at cⱼ = yⱼ - log Mₚ(hⱼ), with
Mₚ(h) = (sinh(h/2)/(h/2))^(p+1) = E[exp(h U)], such that its mean in x is exactly xⱼ; the masses at the two ends stay point
masses. The density is positive, of class C^(p-1), and keeps the mass and the forward exactly. The prices are computed
in one pass: the kernels above the strike price Pⱼ (xⱼ - K) (tail sums), the few kernels around the strike are integrated
by a 5-point Gauss-Legendre rule on each polynomial piece (exact to rounding for the spacings of a grid).
"""
function ahKernelMap(z::AbstractVector{T}, P::AbstractVector{W}; degree::Int=2) where {T,W}
    n = length(z)
    y = log.(z)
    h = [j == 1 || j == n ? zero(T) : (y[j+1] - y[j-1]) / 2 for j = 1:n]
    w = (degree + 1) / 2
    c = [j == 1 || j == n ? y[j] : y[j] - (degree + 1) * log(sinh(h[j] / 2) / (h[j] / 2)) for j = 1:n]
    lo = [c[j] - w * h[j] for j = 1:n]; hi = [c[j] + w * h[j] for j = 1:n]
    #tail sums over the inner kernels j ≥ i: Σ Pⱼ xⱼ and Σ Pⱼ
    A = zeros(W, n + 1); B = zeros(W, n + 1)
    for j = n-1:-1:2
        A[j] = A[j+1] + P[j] * z[j]; B[j] = B[j+1] + P[j]
    end
    A[1] = A[2]; B[1] = B[2]
    #tail sums over the inner kernels 2 ≤ j ≤ i: Σ Pⱼ xⱼ and Σ Pⱼ (the puts)
    Alo = zeros(W, n); Blo = zeros(W, n)
    for j = 2:n-1
        Alo[j] = Alo[j-1] + P[j] * z[j]; Blo[j] = Blo[j-1] + P[j]
    end
    #the supports are not ordered on an irregular grid (a wide kernel next to a large spacing covers narrow ones): suffix
    #minimum of the lower ends and prefix maximum of the upper ends of the inner kernels, both non-decreasing
    loMin = fill(T(Inf), n); hiMax = fill(T(-Inf), n)
    for j = n-1:-1:2
        loMin[j] = min(lo[j], loMin[j+1])
    end
    loMin[1] = loMin[2]
    for j = 2:n-1
        hiMax[j] = max(hi[j], hiMax[j-1])
    end
    hiMax[n] = hiMax[n-1]
    return (z=z, y=y, P=P, h=h, c=c, lo=lo, hi=hi, A=A, B=B, Alo=Alo, Blo=Blo, degree=degree, w=w, loMin=loMin, hiMax=hiMax)
end
"E[(exp(Y) - k)⁺] for the kernel j, restricted to Y > log k (the kernel straddles log k)"
function ahKernelPartial(d, j, k)
    yk = log(k); p = d.degree; hj = d.h[j]; cj = d.c[j]
    u0 = (yk - cj) / hj
    acc = zero(eltype(d.P)) * zero(k)
    #polynomial pieces of the cardinal B-spline: unit intervals starting at -w
    for a in (-d.w):1:(d.w-1)
        lo = max(a, u0); hi = a + 1
        lo >= hi && continue
        mid = (lo + hi) / 2; half = (hi - lo) / 2
        for (g, wg) in AH_GAUSS5
            u = mid + half * g
            acc += wg * half * (exp(cj + hj * u) - k) * cardinalBSpline(p, u)
        end
    end
    return acc
end
function ahKernelPrice(d, k)
    z = d.z; n = length(z)
    price = d.P[n] * max(z[n] - k, zero(k)) + d.P[1] * max(z[1] - k, zero(k))
    yk = log(k)
    #the inner kernels j ≥ jAbove are entirely above log k (tail sums); below, those whose support ends above log k
    jAbove = clamp(searchsortedfirst(d.loMin, yk), 2, n)
    price += d.A[jAbove] - k * d.B[jAbove]
    j = jAbove - 1
    while j >= 2 && d.hiMax[j] > yk
        d.hi[j] > yk && (price += d.P[j] * ahKernelPartial(d, j, k))
        j -= 1
    end
    return price
end
"normalized put price E[(k - x)⁺] of the kernel map, from the kernels below k (full relative precision far out of the money)"
function ahKernelPut(d, k)
    z = d.z; n = length(z)
    put = d.P[1] * max(k - z[1], zero(k)) + d.P[n] * max(k - z[n], zero(k))
    yk = log(k)
    #the inner kernels 2 ≤ j ≤ jBelow are entirely below log k (tail sums); above, those whose support starts below log k
    jBelow = min(searchsortedlast(d.hiMax, yk), n - 1)
    put += k * d.Blo[jBelow] - d.Alo[jBelow]
    j = jBelow + 1
    while j <= n - 1 && d.loMin[j] < yk
        if d.hi[j] <= yk
            put += d.P[j] * (k - z[j])
        elseif d.lo[j] < yk #straddles log k: E[(k - X)⁺] = E[(X - k)⁺] - (zⱼ - k), the kernel having the mean zⱼ
            put += d.P[j] * (ahKernelPartial(d, j, k) - (z[j] - k))
        end
        j += 1
    end
    return put
end
"density of x at x (forward 1) of the kernel map"
function ahKernelDensityAt(d, x)
    yx = log(x); n = length(d.z)
    f = zero(eltype(d.P)) * zero(x)
    #the kernels whose support may contain log x
    for j in max(2, searchsortedfirst(d.hiMax, yx)):min(n - 1, searchsortedlast(d.loMin, yx))
        d.lo[j] < yx < d.hi[j] && (f += d.P[j] * cardinalBSpline(d.degree, (yx - d.c[j]) / d.h[j]) / d.h[j])
    end
    return f / x
end
const AH_KERNEL_DEGREE = Dict(:kernel => 2, :kernel1 => 1, :kernel2 => 2, :kernel3 => 3)
isKernelMap(m::Symbol) = haskey(AH_KERNEL_DEGREE, m)

"""
one implicit step of the linear finite elements with a consistent mass matrix for the Fokker-Planck equation
pₜ = (S p)ₓₓ, S = ½σ²x²: the density is linear in x between the grid points, p = Σⱼ qⱼ φⱼ (φⱼ the hat functions of the grid,
qⱼ the density at zⱼ, zero at the two ends), tested by the hat functions and integrated by parts twice:
Σⱼ Mₖⱼ qⱼ - Δt Σᵢ wₖᵢ Sᵢ qᵢ = Lₖ_prev for the inner grid points, Mₖⱼ = ∫ φₖ φⱼ, wₖᵢ the weights of the Dirac masses of φₖ''.
The state is L, Lₖ = ∫ φₖ p (the weak masses, the Dirac at the forward is exactly a unit L) plus the two boundary point
masses in L₁ and Lₘ; as Σₖ φₖ = 1 and Σₖ zₖ φₖ = x, Σₖ Lₖ and Σₖ zₖ Lₖ are the mass and the mean of the density, kept exactly
(the boundary entries take the flux absorbed during the step). Tridiagonal; an M-matrix, and the density positive, when
Δt Sⱼ ≥ (zⱼ₊₁ - zⱼ)² / 6 and (zⱼ - zⱼ₋₁)² / 6 at every inner grid point.
"""
function ahP1Step(z::AbstractVector{T}, Lprev::AbstractVector, σ::AbstractVector{TS}, dt) where {T,TS}
    n = length(z)
    TV = promote_type(TS, eltype(Lprev))
    m = n - 2
    dl = zeros(TV, m - 1); d = zeros(TV, m); du = zeros(TV, m - 1)
    S = [σ[i]^2 * z[i]^2 * (dt / 2) for i = 1:n]
    for k = 2:n-1
        a, b = z[k] - z[k-1], z[k+1] - z[k]
        d[k-1] = (a + b) / 3 + S[k] * (1 / a + 1 / b)
        k > 2 && (dl[k-2] = a / 6 - S[k-1] / a)
        k < n - 1 && (du[k-1] = b / 6 - S[k+1] / b)
    end
    #Thomas algorithm
    rhs = Vector{TV}(Lprev[2:n-1])
    for k = 2:m
        f = dl[k-1] / d[k-1]
        d[k] -= f * du[k-1]
        rhs[k] -= f * rhs[k-1]
    end
    q = zeros(TV, n)
    q[n-1] = rhs[m] / d[m]
    for k = m-1:-1:1
        q[k+1] = (rhs[k] - du[k] * q[k+2]) / d[k]
    end
    L = zeros(TV, n)
    for k = 1:n
        k > 1 && (L[k] += (z[k] - z[k-1]) * (q[k-1] + 2q[k]) / 6)
        k < n && (L[k] += (z[k+1] - z[k]) * (2q[k] + q[k+1]) / 6)
    end
    #the boundary entries keep the mass and the mean: the previous ones plus the flux absorbed during the step (Σₖ φₖ'' = 0
    #and Σₖ zₖ φₖ'' = 0 leave only the boundary terms when the inner rows are summed). In exact arithmetic these are the
    #differences of the sums of L and z L before and after the step; computed as such, they are rounding noise of 1e-16
    #(possibly negative) when the absorbed mass is smaller, which corrupts the far out-of-the-money prices
    L[n] = Lprev[n] + S[n-1] * q[n-1] / (z[n] - z[n-1])
    L[1] = Lprev[1] + S[2] * q[2] / (z[2] - z[1])
    return L
end
"""
local volatility of one implicit step of size dt between two flat Black smiles of total variances wPrev (zero: the payoff)
and wCur, at the forward moneyness x: σ² = 2 (c - cPrev) / (dt x² c''), from the out-of-the-money prices (parity makes the
call and put differences equal)
"""
function ahFlatStepVol(x, wPrev, wCur, dt)
    otm(w) = w <= 0 ? zero(x) : (sw = sqrt(w); d1 = -log(x) / sw + sw / 2; d2 = d1 - sw;
        x >= 1 ? Math.normcdf(d1) - x * Math.normcdf(d2) : x * Math.normcdf(-d2) - Math.normcdf(-d1))
    sw = sqrt(wCur); d2 = -log(x) / sw - sw / 2
    density = Math.normpdf(d2) / (x * sw)
    return sqrt(max(2 * (otm(wCur) - otm(wPrev)), zero(x)) / (dt * x^2 * density))
end
"the density map of the state L of ahP1Step: the density at the grid points and the boundary masses"
function ahP1Map(z::AbstractVector, L::AbstractVector{W}) where {W}
    n = length(z)
    #Mq = L on the inner rows (q₁ = qₘ = 0), tridiagonal
    m = n - 2
    d = [(z[k+1] - z[k-1]) / 3 for k = 2:n-1]; off = [(z[k+1] - z[k]) / 6 for k = 2:n-2]
    rhs = Vector{W}(L[2:n-1]); dd = Vector{W}(d)
    for k = 2:m
        f = off[k-1] / dd[k-1]
        dd[k] -= f * off[k-1]; rhs[k] -= f * rhs[k-1]
    end
    q = zeros(W, n)
    q[n-1] = rhs[m] / dd[m]
    for k = m-1:-1:1
        q[k+1] = (rhs[k] - off[k] * q[k+2]) / dd[k]
    end
    top = L[n] - (z[n] - z[n-1]) * q[n-1] / 6 #the point mass at the upper boundary
    #the density is exactly linear between the grid points, with the mass and the mean of the scheme: no tilt (a = 1,
    #b = 0, as ahDensityMap returns them), and the tail sums in closed form, ∫ q = h (qⱼ + qⱼ₊₁)/2 and
    #∫ x q = h (qⱼ (2zⱼ + zⱼ₊₁) + qⱼ₊₁ (zⱼ + 2zⱼ₊₁))/6 on [zⱼ, zⱼ₊₁] (ahDensityMap integrates them by quadrature)
    A = zeros(W, n); B = zeros(W, n)
    for j = n-1:-1:1
        h = z[j+1] - z[j]
        A[j] = A[j+1] + h * (q[j] * (2z[j] + z[j+1]) + q[j+1] * (z[j] + 2z[j+1])) / 6
        B[j] = B[j+1] + h * (q[j] + q[j+1]) / 2
    end
    return (zk=z, M=q, a=one(W), b=zero(W), A=A, B=B, top=top)
end

"""
    ahP1Slice(z, L)

an expiry of the linear finite elements (scheme=:p1): the density map of the state L (ahP1Map), the point mass at the lower
boundary, and the probability and first moment below each grid point (accumulated from the lower boundary, for the put
price and the slope of the price to full relative precision far below the forward)
"""
function ahP1Slice(z::AbstractVector, L::AbstractVector{W}) where {W}
    d = ahP1Map(z, L); q = d.M; n = length(z)
    bottom = L[1] - (z[2] - z[1]) * q[2] / 6 #the point mass at the lower boundary
    F = zeros(W, n); M1 = zeros(W, n)
    F[1] = bottom; M1[1] = bottom * z[1]
    for j = 1:n-1
        h = z[j+1] - z[j]
        F[j+1] = F[j] + h * (q[j] + q[j+1]) / 2
        M1[j+1] = M1[j] + h * (q[j] * (2z[j] + z[j+1]) + q[j+1] * (z[j] + 2z[j+1])) / 6
    end
    return (d=d, bottom=bottom, F=F, M1=M1)
end
"""
out-of-the-money price at K of an expiry of the linear finite elements (ahP1Slice), the call above the forward and the put
below, with the slope of the out-of-the-money price (the probability above K, negated, for the call; below K for the put)
and the density at K
"""
function ahP1OTM(sl, K)
    d = sl.d; z = d.zk; q = d.M; n = length(z)
    if K >= 1
        K >= z[n] && return (zero(K) * q[1], zero(K) * q[1], zero(K) * q[1])
        j = clamp(searchsortedlast(z, K), 1, n - 1); h = z[j+1] - z[j]
        qK = q[j] + (q[j+1] - q[j]) * (K - z[j]) / h
        above = d.B[j+1] + (z[j+1] - K) * (qK + q[j+1]) / 2 + d.top
        return (ahDensityPrice(d, K), -above, qK)
    end
    K <= z[1] && return (zero(K) * q[1], zero(K) * q[1], zero(K) * q[1])
    j = clamp(searchsortedlast(z, K), 1, n - 1); h = z[j+1] - z[j]; u = K - z[j]
    Δ = (q[j+1] - q[j]) / h
    FK = sl.F[j] + q[j] * u + Δ * u^2 / 2
    MK = sl.M1[j] + z[j] * q[j] * u + (z[j] * Δ + q[j]) * u^2 / 2 + Δ * u^3 / 3
    return (K * FK - MK, FK, q[j] + Δ * u)
end
"""
total implied variance w and its first two derivatives in the log-moneyness y at an expiry T of the linear finite elements:
w from the out-of-the-money price, w' from its slope (∂ₖc = ∂ₖB + ∂_wB w'/K for the Black price B(K, w)), and w'' from the
density by c'' = φ(d₂) g / (K√w), g = (1 - y w'/2w)² - w'²/4 (1/4 + 1/w) + w''/2 (Gatheral's function of the Dupire formula)
"""
function ahP1TotalVariance(sl, y, T)
    K = exp(y); isCall = K >= 1
    p, slope, dens = ahP1OTM(sl, K)
    vol = impliedVolatilityThiophene(isCall, max(p, 1e-300), one(K), K, T, one(K))
    w = vol^2 * T; sw = sqrt(w)
    d1 = -y / sw + sw / 2; d2 = d1 - sw
    #∂ₖc + N(d₂) for the call, ∂ₖp - N(-d₂) for the put (the same number), over ∂_wB = φ(d₁)/(2√w)
    wy = K * (isCall ? slope + Math.normcdf(d2) : slope - Math.normcdf(-d2)) / (Math.normpdf(d1) / (2sw))
    g = dens * K * sw / Math.normpdf(d2)
    wyy = 2 * (g - (1 - y * wy / (2w))^2 + wy^2 / 4 * (1 / 4 + 1 / w))
    return (w=w, wy=wy, wyy=wyy)
end

#finite element schemes: degree of the kernels; the suffix n puts S at the grid points (centred=false, positive only when
#σ²x² varies slowly between neighbours)
const AH_FEM_DEGREE = Dict(:bspline1 => 1, :bspline2 => 2, :bspline3 => 3, :bspline2n => 2, :bspline3n => 3)
isFEMScheme(s::Symbol) = haskey(AH_FEM_DEGREE, s)
femCentred(s::Symbol) = !endswith(string(s), "n")
"the values ψⱼ(zᵢ) of the kernels at the inner grid points, as (i, j, ψ) triplets, and the band of the step matrix"
function ahKernelStepData(z::AbstractVector{T}, degree::Int) where {T}
    n = length(z)
    y = log.(z)
    h = [j == 1 || j == n ? zero(T) : (y[j+1] - y[j-1]) / 2 for j = 1:n]
    c = [j == 1 || j == n ? y[j] : y[j] - (degree + 1) * log(sinh(h[j] / 2) / (h[j] / 2)) for j = 1:n]
    w = (degree + 1) / 2
    #triplets (k, j, i, wₖᵢ ψⱼ(zᵢ)) for the inner rows k: the weights of the Dirac masses of φₖ'' at zᵢ, i = k-1, k, k+1
    triplets = Tuple{Int,Int,Int,T}[]
    bw = 1
    for j = 2:n-1
        for i = max(2, searchsortedfirst(y, c[j] - w * h[j])):min(n - 1, searchsortedlast(y, c[j] + w * h[j]))
            ψ = cardinalBSpline(degree, (y[i] - c[j]) / h[j]) / (h[j] * z[i])
            ψ == 0 && continue
            bw = max(bw, abs(i - j) + 1) #band of W S B
            for k = max(2, i - 1):min(n - 1, i + 1)
                wki = k == i ? -(1 / (z[i] - z[i-1]) + 1 / (z[i+1] - z[i])) : 1 / abs(z[i] - z[k])
                push!(triplets, (k, j, i, wki * ψ))
            end
        end
    end
    #with S at the kernel centres, the entries (k, j) are Sⱼ Σᵢ wₖᵢ ψⱼ(zᵢ): the sums, independent of the local volatility
    sums = Dict{Tuple{Int,Int},T}()
    for (k, j, i, v) in triplets
        sums[(k, j)] = get(sums, (k, j), zero(T)) + v
    end
    pairs = sort([(k, j, v) for ((k, j), v) in sums])
    return (triplets=triplets, pairs=pairs, bw=bw, degree=degree)
end
"""
one implicit step of the Petrov-Galerkin discretization of the Fokker-Planck equation pₜ = (S p)ₓₓ, S = ½σ²x²: the density
is p = Σⱼ Pⱼ ψⱼ, ψⱼ the kernel of the grid point zⱼ of the kernel map (a cardinal B-spline of degree `degree` in log x, of
unit mass and mean zⱼ), the test functions are the hat functions φₖ of the grid in x, and the mass matrix is lumped by
columns (the identity, as Σₖ φₖ = 1); S p is approximated by Σⱼ Sⱼ Pⱼ ψⱼ: for the inner grid points,
Pₖ - Δt Σⱼ (Σᵢ wₖᵢ ψⱼ(zᵢ)) Sⱼ Pⱼ = Pₖ_prev, with wₖᵢ the weights of the Dirac masses of φₖ''. Σₖ φₖ = 1 and Σₖ zₖ φₖ = x: the two
boundary masses take what keeps the sum and the mean of the masses, which are the mass and the mean of the density, exactly;
the density is zero at the boundaries. The off-diagonal entries Sⱼ Σᵢ wₖᵢ ψⱼ(zᵢ) are non-negative whatever the local
volatility (with S at the grid points instead, `centred=false`, only when σ²x² varies slowly between neighbours), and the columns sum to
one or more: the matrix is an M-matrix, and the masses, and thus the density, stay positive.
"""
function ahKernelStep(z::AbstractVector{T}, Pprev::AbstractVector, σ::AbstractVector{TS}, dt; degree::Int=2, data=ahKernelStepData(z, degree), centred::Bool=true) where {T,TS}
    n = length(z)
    TV = promote_type(TS, eltype(Pprev))
    bw = data.bw
    #A = I - Δt W S B on the inner nodes 2..n-1, stored by diagonals: A[k, bw + 1 + (j - k)]
    m = n - 2
    A = zeros(TV, m, 2bw + 1)
    for k = 1:m
        A[k, bw+1] = one(TV)
    end
    S = [σ[i]^2 * z[i]^2 * (dt / 2) for i = 1:n]
    #S at the centre of the kernel, S p ≈ Σⱼ Sⱼ Pⱼ ψⱼ (centred), or at the grid point, S p ≈ Sᵢ p(zᵢ)
    if centred
        for (k, j, v) in data.pairs
            A[k-1, bw+1+(j-k)] -= v * S[j]
        end
    else
        for (k, j, i, v) in data.triplets
            A[k-1, bw+1+(j-k)] -= v * S[i]
        end
    end
    #banded Gaussian elimination without pivoting (column-dominant M-matrix)
    rhs = Vector{TV}(Pprev[2:n-1])
    for k = 1:m-1
        piv = A[k, bw+1]
        for r = k+1:min(m, k + bw)
            f = A[r, bw+1+(k-r)] / piv
            f == 0 && continue
            for col = k:min(m, k + bw)
                A[r, bw+1+(col-r)] -= f * A[k, bw+1+(col-k)]
            end
            rhs[r] -= f * rhs[k]
        end
    end
    Q = similar(rhs)
    for k = m:-1:1
        acc = rhs[k]
        for col = k+1:min(m, k + bw)
            acc -= A[k, bw+1+(col-k)] * Q[col]
        end
        Q[k] = acc / A[k, bw+1]
    end
    P = zeros(TV, n)
    P[2:n-1] = Q
    #the boundary masses: the sum and the mean of the masses are kept
    dMass = sum(Q) - sum(view(Pprev, 2:n-1)); dMean = sum(Q[k] * z[k+1] for k = 1:m) - sum(Pprev[k] * z[k] for k = 2:n-1)
    P[n] = Pprev[n] + (z[1] * dMass - dMean) / (z[n] - z[1])
    P[1] = Pprev[1] - dMass - (P[n] - Pprev[n])
    return P
end

"second derivative (density) of the quintic Hermite spline at x"
function ahQuinticSecondAt(q, x)
    z = q.z; j = clamp(searchsortedlast(z, x), 1, length(z) - 1); h = z[j+1] - z[j]; t = (x - z[j]) / h; u = 1 - t
    H0(t) = -t * (60 - 180t + 120t^2); H1(t) = -t * (36 - 96t + 60t^2); H2(t) = (2 - 18t + 36t^2 - 20t^3) / 2
    return (q.c[j] * H0(t) + q.c[j+1] * H0(u)) / h^2 + (q.d[j] * H1(t) - q.d[j+1] * H1(u)) / h + q.s[j] * H2(t) + q.s[j+1] * H2(u)
end
ahLinearAt(z, c, x) = (j = clamp(searchsortedlast(z, x), 1, length(z) - 1); u = (x - z[j]) / (z[j+1] - z[j]); (1 - u) * c[j] + u * c[j+1])

"""
    calibrateBSplineAHSurface(ttes, forwards, strikes::Vector{Vector}, callPrices::Vector{Vector}, weights::Vector{Vector};
        knots=nothing, size=10, location="Uniform", grid=:strikes, gridSize=401, L=nothing, U=nothing, useVol=false, optimizer="LM-ALG",
        nRefine=3, penaltyStrike=0.0, penaltyTime=0.0, minVol=0.01, maxVol=5.0, smoothWings=false, knotExtension=0.0)

Calibrate the B-spline Andreasen-Huge model expiry by expiry: each expiry fits the coefficients of the local volatility spline of
its step to its quotes, starting from the prices of the previous expiry (the payoff for the first one, with the C3
condition at the forward). The local volatility knots (in K/F) of each step are `size` points uniform in log-moneyness
over the quoted strikes of its expiry (the default, `location="Uniform"`: fewer coefficients than quotes on dense
surfaces), or the Mid-XX knots of its strikes plus the forward with the two outer coefficients on each side tied
(`location="Mid-XX"`: as many free coefficients as quotes, as for the quadratic LVG), or `knots`, shared by all steps.
The grid has `gridSize` points in [`L`, `U`] (in K/F; by default six at-the-money standard deviations of the longest
expiry on each side, and at least [1/6, 6]), plus the forward: with `grid=:strikes` (the default), uniform in
y = log(K/F) plus all the quoted strikes, which are priced exactly; the spacing is irregular (the strikes of different
expiries interleave in K/F), so that second differences of the grid prices are noisy: use `densityNormalized` for the
density. Alternatives: `grid=:uniform`, uniform in y, or `grid=:sinh`, y = β sinh(ξ) with ξ uniform and β the
at-the-money standard deviation of the first expiry, the quotes then priced by a 4-point Lagrange interpolation in y of
the logarithm of the out-of-the-money price (these need more points for the deep wings of very short expiries).
With `smoothWings`, the two outer coefficients on each side are tied, so that the local volatility has a zero slope at
the outer knots and continues flat beyond them without a kink, which would otherwise make a kink in the density at the
last quotes. `knotExtension` extends the uniform points beyond the quoted range by this fraction of its width on each
side. Recommended: `size=12, smoothWings=true, knotExtension=0.1` (as many free coefficients as `size=10` without ties).
Optional penalties on the second differences of the coefficients (`penaltyStrike`) and
on the change of the local volatility from the previous step at 2`size` points over the quoted range (`penaltyTime`), in the units of the residuals (volatility with `useVol`, else
weighted prices, with the weights normalized to a mean of one per expiry).

Scheme (`scheme`): `:adjoint` (default), the masses P of the grid points solve the discrete Fokker-Planck equation
Aᵀ Pᵢ = Pᵢ₋₁ adjoint to the backward scheme in the spot (as in Healy's book), from all the mass at the forward, with the
exact-forward adjustment (mass and mean exactly 1), and the grid call prices are Σⱼ Pⱼ (zⱼ - K)⁺; `:forward`, the
forward Dupire equation in the strike solved on the call prices (the original Andreasen-Huge scheme), here without the
exact-forward adjustment. With it (`ahStep(...; exactForward=true)`) the two are the same scheme, A C = C Aᵀ with
C[k,j] = (zⱼ - zₖ)⁺, on any grid; without it, the prices differ at the order of the discretization error, put-call
parity is not exact, and the slope jumps of the prices (masses) are noisy on an irregular grid.
`:bspline2`, `:bspline3` (`:bspline1`): finite elements (`ahKernelStep`), the density a sum of the kernels of the kernel
map of that degree, positive whatever the local volatility, priced by that kernel map (`priceMap` is ignored);
`:bspline2n`, `:bspline3n`: the same with S = ½σ²x² at the grid points, more accurate, but positive only when σ²x² varies
slowly between neighbouring grid points. `:p1`: linear finite elements with a consistent mass matrix (`ahP1Step`), the density
linear between the grid points with the exact mass and mean (no tilt), priced by it (`priceMap` is ignored); positive when
Δt σ²x² ≥ h²/3 at every grid point, h the spacing in x.

Prices between grid points (`priceMap`): `:grid`, a natural cubic spline of the grid call prices (the calibration
fits the grid prices at the quotes on the grid, the spline prices otherwise); `:density`, the interpolated density: the
density of the masses at the grid points (:adjoint; of the step for :forward), linear in between, tilted to the mass and
mean of the masses, integrated exactly (positive density; the calibration fits these prices); `:box`, the cell density: the
masses of the grid spread over the cells around the grid points, keeping mass and mean (positive, possibly discontinuous
density; prices exact outside the box of the strike; the calibration fits these prices; best with `grid=:uniform`);
`:schaback`, the calibration as `:grid`, and afterwards a convexity preserving rational spline through the grid prices;
`:variance`, the calibration as `:grid`, and afterwards a natural cubic spline in log-moneyness of the total implied
variance of the grid prices (out-of-the-money options inverted with ThiopheneIV), priced with the Black formula;
`:linear`, the calibration as `:grid`, and the prices linear between grid points (the point masses of the grid: the
one-pass Fokker-Planck formula of Andreasen and Huge); `:kernel` (`:kernel1`, `:kernel2`, `:kernel3`), the kernel
density: each mass spread by a B-spline of degree 2 (1, 2, 3) in log x of the local width, shifted to keep its mean
(positive density of class C¹ (C⁰, C¹, C²), forward exact; the calibration fits these prices; requires scheme=:adjoint).

`exactPuts` (default true): with the box and kernel maps, the quotes below the forward are fitted through the put price
computed from the masses below the strike, to full relative precision, and the box map takes the masses of the scheme;
false prices them by put-call parity, c - (1 - K), with the box masses from the second differences of the grid prices,
whose rounding (about 1e-16) swamps the far out-of-the-money puts of short expiries and can stall the calibration.

`positivityMargin` (scheme=:p1): the consistent mass step is an M-matrix when Δt ½σ(zⱼ)²zⱼ² ≥ max(hⱼ₋₁, hⱼ)²/6 at every
inner grid point, that is σ(zⱼ) ≥ max(hⱼ₋₁, hⱼ) / (zⱼ √(3Δt)); each coefficient of the spline of the step is bounded below
by (1 + positivityMargin) times the largest of these over the grid points its B-spline covers: the density of each expiry
is then positive. Not the densities between expiries, from steps shorter than Δt.
"""
function calibrateBSplineAHSurface(ttes::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractVector{<:AbstractVector{T}},
    callPrices::AbstractVector{<:AbstractVector{T}}, weights::AbstractVector{<:AbstractVector{T}}; knots=nothing, size=10,
    gridSize=401, L=nothing, U=nothing, useVol=false, optimizer="LM-ALG", nRefine=3, penaltyStrike=0.0, penaltyTime=0.0,
    minVol=0.01, maxVol=5.0, c3=true, location="Uniform", grid=:strikes, priceMap=:grid, scheme=:adjoint, smoothWings=false, knotExtension=0.0, firstSubsteps=1, initialFraction=0.0, initialKind=:lognormal, firstKnots=:default, exactPuts=true, positivityMargin=0.01, wingFloor=0.0, wingFloorKind=:atm, monotoneWings=false, capFree=nothing) where {T}
    n = length(ttes)
    #the finite element schemes come with their own price map, the kernel map of the same degree
    isFEMScheme(scheme) && (priceMap = Symbol("kernel", AH_FEM_DEGREE[scheme]))
    scheme == :p1 && (priceMap = :density) #the linear finite elements: their own density, linear between the grid points
    ks = [strikes[i] ./ forwards[i] for i = 1:n]
    cs = [callPrices[i] ./ forwards[i] for i = 1:n]
    #implied volatility from a normalized call price, through the out-of-the-money option (a put below the forward: the
    #time value of a deep in-the-money call is lost in its intrinsic value)
    vol(price, k, t) = k >= 1 ? impliedVolatilityThiophene(true, max(price, 1e-64), one(T), k, t, one(T)) :
                       impliedVolatilityThiophene(false, max(price - (1 - k), 1e-64), one(T), k, t, one(T))
    #at-the-money volatility of an expiry, interpolated linearly in the strike between the two quotes around the forward
    function atmVol(i)
        j = clamp(searchsortedfirst(ks[i], one(T)), 2, length(ks[i]))
        v1, v2 = vol(cs[i][j-1], ks[i][j-1], ttes[i]), vol(cs[i][j], ks[i][j], ttes[i])
        return v1 + (v2 - v1) * (1 - ks[i][j-1]) / (ks[i][j] - ks[i][j-1])
    end
    #grid bounds: six standard deviations of the longest expiry, at least [1/6, 6], beyond the quotes (absorbing boundaries)
    sdMax = maximum(atmVol(i) * sqrt(ttes[i]) for i = 1:n)
    L = something(L, min(T(1 / 6), exp(-6sdMax), minimum(minimum.(ks)) / 2))
    U = something(U, max(T(6), exp(6sdMax), 2maximum(maximum.(ks))))
    #grid: the forward and the quoted strikes, and uniform log-moneyness points that are not too close to them (a
    #fraction of the uniform spacing), so that the grid stays smoothly varying
    #grid=:strikes (default): uniform in log-moneyness plus the forward and all the quoted strikes, priced exactly; the
    #strikes of different expiries interleave in K/F, and the spacing is very irregular, which leaves small errors in the
    #prices that second differences (the density) amplify. grid=:uniform or :sinh: the forward only, the quotes priced by
    #4-point Lagrange interpolation in log-moneyness
    #grid=:sinh, a smoothly graded grid y = β sinh(ξ), ξ uniform, concentrated around the forward on the scale β of the
    #at-the-money standard deviation of the first expiry
    fixed = grid == :strikes ? sort(unique(vcat(one(T), filter(k -> L < k < U, vcat(ks...))))) : [one(T)]
    hz = log(U / L) / (gridSize - 1)
    base = if grid == :sinh
        β = atmVol(1) * sqrt(ttes[1])
        exp.(β .* sinh.(range(asinh(log(L) / β), stop=asinh(log(U) / β), length=gridSize)))
    else
        exp.(range(log(L), stop=log(U), length=gridSize))
    end
    hloc(i) = log(base[min(i + 1, end)] / base[max(i - 1, 1)]) / 2
    filler = [x for (i, x) in enumerate(base) if i == 1 || i == length(base) || minimum(abs.(log.(fixed ./ x))) > (grid == :strikes ? hz : hloc(i)) / 3]
    z = sort(unique(vcat(filler, fixed)))
    iAtm = findfirst(==(one(T)), z)
    keepNodes = interpolationNodes(z)
    #Mid-XX (as the quadratic LVG, without its far bounds L and U: the local volatility is flat beyond the outer knots) or
    #`size` points uniform in log-moneyness, over the quoted strikes of each expiry
    #(Mid-XX with the forward added, as the quadratic LVG)
    stepKnots(k, sz=size) = location == "Mid-XX" ? sort(unique(vcat(makeKnots(zero(T), one(T), k, 0; location="Mid-XX", L=k[1] / 2, U=2k[end])[2:end-1], one(T)))) :
                            (e = knotExtension * log(k[end] / k[1]); exp.(range(log(k[1]) - e, stop=log(k[end]) + e, length=sz)))
    knotsList = [collect(T, knots === nothing ? stepKnots(ks[i]) : knots) for i = 1:n]
    #points where the local volatilities of consecutive steps are compared (time penalty)
    xPen = exp.(collect(range(log(minimum(minimum.(ks))), stop=log(maximum(maximum.(ks))), length=2size)))
    cAtm1 = blackScholesFormula(true, one(T), one(T), atmVol(1)^2 * ttes[1], one(T), one(T))
    c3 && (knotsList[1] = ahFirstKnots(knotsList[1], cAtm1))
    #EXPERIMENTAL firstKnots without the C3 condition: :c3, the knots of the C3 condition (double knot at the forward and
    #knots at 1 ± 2c(1)), all coefficients free; :near, a single knot at the forward and knots at 1 ± 2c(1)
    if !c3 && firstKnots == :c3
        knotsList[1] = ahFirstKnots(knotsList[1], cAtm1)
    elseif !c3 && firstKnots == :near
        knotsList[1] = sort(unique(vcat(knotsList[1], one(T), 1 - 2cAtm1, 1 + 2cAtm1)))
    end
    #capFree = m (an integer, nothing to disable): at most (quotes - m) free coefficients on a step (uniform knots), fewer
    #uniform points on the steps with few quotes. A quadratic B-spline on k knots has k + 1 coefficients, less two with tied
    #wings and one for the C3 condition
    if capFree !== nothing && knots === nothing && location != "Mid-XX"
        for i = 1:n
            nFree(kn) = length(kn) + 1 - (smoothWings ? 2 : 0) - (i == 1 && c3 ? 1 : 0)
            sz = size
            while sz > 3 && nFree(knotsList[i]) > length(ks[i]) - capFree
                sz -= 1
                kn = collect(T, stepKnots(ks[i], sz))
                knotsList[i] = i == 1 && c3 ? ahFirstKnots(kn, cAtm1) : kn
            end
        end
    end
    kc, D = c3 ? ahC3Data(knotsList[1]) : (0, T[])
    payoff = [max(1 - zj, zero(T)) for zj in z]
    dirac = [j == iAtm ? one(T) : zero(T) for j in eachindex(z)] #the initial masses: all at the forward
    #EXPERIMENTAL initialFraction > 0: the scheme starts at t0 = initialFraction T1 from a smooth density instead of the
    #Dirac at the forward, not calibrated, with the at-the-money volatility of the first expiry: the lognormal density
    #(initialKind=:lognormal, cell masses tilted to mass and mean one) or one implicit step with a flat volatility from the
    #Dirac (initialKind=:flatstep). The calibrated first step then runs from t0 to T1 (intermediate times before T1
    #are not adapted yet)
    tStart = initialFraction * ttes[1]
    if tStart > 0
        σ0 = atmVol(1)
        if initialKind == :flatstep
            dirac = ahAdjointStep(z, dirac, fill(σ0, length(z)), tStart)
        else
            v0 = σ0^2 * tStart
            edges = vcat(zero(T), [(z[j] + z[j+1]) / 2 for j in 1:length(z)-1], T(Inf))
            cdf(x) = x <= 0 ? zero(T) : (isinf(x) ? one(T) : erfc(-(log(x) + v0 / 2) / sqrt(2v0)) / 2)
            mlog = [cdf(edges[j+1]) - cdf(edges[j]) for j in eachindex(z)]
            M = sum(mlog .* z); S2 = sum(mlog .* z .^ 2)
            β = (1 - M) / (S2 - M^2); α = 1 - β * M
            dirac = mlog .* (α .+ β .* z)
        end
    end
    #the state of the scheme: the masses of the grid points (:adjoint) or the call prices (:forward)
    femData = isFEMScheme(scheme) ? ahKernelStepData(z, AH_FEM_DEGREE[scheme]) : nothing
    advance(state, σ, dt) = scheme == :adjoint ? ahAdjointStep(z, state, σ, dt) : scheme == :p1 ? ahP1Step(z, state, σ, dt) :
                            isFEMScheme(scheme) ? ahKernelStep(z, state, σ, dt; degree=AH_FEM_DEGREE[scheme], data=femData, centred=femCentred(scheme)) : ahStep(z, state, σ, dt)
    pricesOf(state) = scheme == :adjoint ? ahMassPrices(z, state) : scheme == :p1 ? (d = ahP1Map(z, state); [ahDensityPrice(d, zk) for zk in z]) :
                      isFEMScheme(scheme) ? (d = ahKernelMap(z, state; degree=AH_FEM_DEGREE[scheme]); [ahKernelPrice(d, zk) for zk in z]) : state
    #the at-the-money price only (the C3 refinement)
    priceAtm(state) = scheme == :p1 ? ahDensityPrice(ahP1Map(z, state), one(T)) : isFEMScheme(scheme) ? ahKernelPrice(ahKernelMap(z, state; degree=AH_FEM_DEGREE[scheme]), one(T)) : pricesOf(state)[iAtm]
    states = Vector{Vector{T}}(undef, n)
    prices = Vector{Vector{T}}(undef, n)
    coeffs = Vector{Vector{T}}(undef, n)
    wingMinVols = Vector{Tuple{T,T}}(undef, n)
    #scheme=:p1: the consistent mass step keeps the density positive when it is an M-matrix, Δt ½σ(zⱼ)²zⱼ² ≥ max(hⱼ₋₁, hⱼ)²/6
    #at every inner grid point
    #the floor of each grid point, then of each coefficient: the largest floor of the grid points its B-spline covers
    #(σ(zⱼ) is a convex combination of the coefficients covering zⱼ); a single floor, the largest of the grid, would be set
    #by the wide spacings of the far wings of a grid with the strikes, far above the volatility of a short expiry
    p1Floor(dt) = [j == 1 || j == length(z) ? zero(T) : (1 + positivityMargin) * max(z[j] - z[j-1], z[j+1] - z[j]) / (z[j] * sqrt(3dt)) for j in eachindex(z)]
    p1CoefficientFloor(B, dt) = (f = p1Floor(dt); [maximum(f[j] for j in axes(B, 1) if B[j, k] > 0; init=zero(T)) for k in axes(B, 2)])
    for i = 1:n
        t0 = i == 1 ? tStart : ttes[i-1]
        dt = ttes[i] - t0
        prevState = i == 1 ? (scheme == :adjoint || scheme == :p1 || isFEMScheme(scheme) ? dirac : payoff) : states[i-1]
        prev = i == 1 ? payoff : prices[i-1]
        B = ahBasisMatrix(knotsList[i], z)
        nb = Base.size(B, 2)
        BPen = ahBasisMatrix(knotsList[i], xPen)
        σPrevPen = i > 1 ? abs.(ahBasisMatrix(knotsList[i-1], xPen) * coeffs[i-1]) : T[]
        #second differences of the coefficients, not across the double knot of the first step (the C3 kink)
        dIdx = [k for k = 2:nb-1 if !(i == 1 && c3 && abs(k - kc) <= 1)]
        #the two outer coefficients on each side are tied: the spline has a zero slope at its outer knots and continues flat
        #beyond them without a kink (a kink of the local volatility makes a kink in the density); Mid-XX: one free
        #coefficient per quote
        tied = smoothWings || (location == "Mid-XX" && knots === nothing)
        free = [k for k = 1:nb if !(i == 1 && c3 && k == kc) && !(tied && (k == 1 || k == nb))]
        #price at each quote: the grid value (strike on the grid) or a 4-point Lagrange interpolation in log-moneyness
        stencil = map(ks[i]) do k
            i0 = searchsortedfirst(z, k)
            if i0 <= length(z) && abs(z[i0] / k - 1) < 1e-8
                ([i0], [one(T)])
            else
                ii = clamp(i0 - 2, 1, length(z) - 3):clamp(i0 - 2, 1, length(z) - 3)+3
                yy = log.(z[ii]); y = log(k)
                (collect(ii), [prod((y - yy[m]) / (yy[l] - yy[m]) for m in 1:4 if m != l) for l in 1:4])
            end
        end
        inside = [j for j in eachindex(ks[i]) if L < ks[i][j] < U]
        w = weights[i] ./ (sum(weights[i]) / length(weights[i]))
        vols = useVol ? [vol(cs[i][j], ks[i][j], ttes[i]) for j in eachindex(ks[i])] : T[]
        #initial guess: flat forward volatility at the money
        σ0 = if i == 1
            atmVol(1)
        else
            vPrev = vol(prices[i-1][iAtm], one(T), ttes[i-1])
            sqrt(max((atmVol(i)^2 * ttes[i] - vPrev^2 * ttes[i-1]) / dt, (2minVol)^2))
        end
        minVols = scheme == :p1 ? max.(minVol, p1CoefficientFloor(B, dt / firstSubsteps^(i == 1 ? 1 : 0))) : fill(minVol, nb)
        if tied #the outer coefficient copies its neighbour, which must then meet both floors
            minVols[2] = minVols[1] = max(minVols[1], minVols[2]); minVols[nb-1] = minVols[nb] = max(minVols[nb-1], minVols[nb])
        end
        #EXPERIMENTAL wingFloor: the outer (tied) coefficients at least this fraction of the forward volatility at the money of
        #the step, against a wall of low local volatility just beyond the last quotes (a density bump before it, and a
        #slice without right tail, which the total variance linear in time between expiries turns into a butterfly)
        #wingFloorKind: :atm, wingFloor σ0 (σ0 the forward volatility at the money of the step); :flat, wingFloor σ_flat(x_w),
        #the local volatility of the step between flat smiles at the at-the-money volatilities of the two expiries, at the
        #outer knot x_w of each side, which decays as σ²√(2T)/|y| in the wings of the first step; :min, min(wingFloor σ0,
        #σ_flat(x_w)), the simple floor capped by what a flat smile needs at the outer knot (below it on short first steps far
        #out in standard deviations, where the simple floor would force a dip of the smile inside the quotes)
        if tied && wingFloor > 0
            if wingFloorKind == :flat || wingFloorKind == :min
                #the at-the-money total variances of the previous expiry of the model and of the quotes of this one, both at
                #x = 1 (forward moneyness); the variance of the step at least (2 minVol)² dt, as for σ0, so that a calendar
                #inversion of the quotes at the money does not remove the floor
                wPrev = i == 1 ? zero(T) : vol(prices[i-1][iAtm], one(T), ttes[i-1])^2 * ttes[i-1]
                wCur = max(atmVol(i)^2 * ttes[i], wPrev + (2minVol)^2 * dt)
                for (ks_, xk) in (((1, 2), knotsList[i][1]), ((nb - 1, nb), knotsList[i][end]))
                    σf = ahFlatStepVol(xk, wPrev, wCur, dt)
                    f = wingFloorKind == :min ? min(wingFloor * σ0, σf) : wingFloor * σf
                    for k in ks_
                        minVols[k] = max(minVols[k], f)
                    end
                end
            else
                for k in (1, 2, nb - 1, nb)
                    minVols[k] = max(minVols[k], wingFloor * σ0)
                end
            end
        end
        maximum(minVols) < maxVol / 2 || throw(DomainError(maximum(minVols), "the positivity floor of the step exceeds maxVol / 2: refine the grid"))
        transforms = [makeTransform(minVols[k], maxVol, optimizer) for k = 1:nb]
        guess = [min(max(σ0, 2minVols[k]), maxVol / 2) for k = 1:nb]
        p0 = [inv(transforms[k], guess[k]) for k in free]
        nPenS = penaltyStrike > 0 ? length(dIdx) : 0
        nPenT = penaltyTime > 0 && i > 1 ? length(xPen) : 0
        nres = length(inside) + nPenS + nPenT
        function coefficients(p, W)
            coef = zeros(W, nb)
            for (r, k) in enumerate(free)
                coef[k] = transforms[k](p[r])
            end
            #EXPERIMENTAL monotoneWings: the outer free coefficient between its inner neighbour and maxVol, so that the local
            #volatility does not decrease outwards beyond the last inner coefficient
            if tied && monotoneWings
                for (k, kin) in ((2, 3), (nb - 1, nb - 2))
                    lo = max(abs(coef[kin]), minVols[k])
                    u = (coef[k] - minVols[k]) / (maxVol - minVols[k])
                    coef[k] = lo + (maxVol - lo) * u
                end
            end
            if tied
                coef[1] = coef[2]; coef[nb] = coef[nb-1]
            end
            return coef
        end
        function solveStep(coef)
            if i == 1
                cA = cAtm1
                local st
                for _ = 0:(c3 ? nRefine : 0)
                    c3 && ahApplyC3!(coef, kc, D, cA)
                    #EXPERIMENTAL firstSubsteps: several implicit sub-steps from the payoff smooth the kink of the density
                    #at the forward (intermediate times before the first expiry still use a single step)
                    st = prevState
                    for _ = 1:firstSubsteps
                        st = advance(st, B * coef, dt / firstSubsteps)
                    end
                    cA = priceAtm(st)
                end
                return st
            end
            return advance(prevState, B * coef, dt)
        end
        function obj!(fvec, p)
            coef = coefficients(p, eltype(p))
            st = solveStep(coef)
            c = isKernelMap(priceMap) || scheme == :p1 ? nothing : pricesOf(st) #these maps price the quotes from the state
            dmap = priceMap == :density ? (scheme == :adjoint ? ahMassDensityMap(z, st) : scheme == :p1 ? ahP1Map(z, st) : ahStepDensityMap(z, keepNodes, c, prev, B * coef, dt)) :
                   isKernelMap(priceMap) ? ahKernelMap(z, st; degree=AH_KERNEL_DEGREE[priceMap]) :
                   priceMap == :box ? ahBoxMap(z, c; P=exactPuts && scheme == :adjoint ? st : nothing) :
                   (grid != :strikes && priceMap in (:grid, :schaback, :variance)) ? ahNaturalSpline(z, c) : nothing
            for (r, j) in enumerate(inside)
                #out-of-the-money puts from the masses below the strike: the far left quotes of a short expiry are worth
                #less than the rounding of a call price, and their implied volatility through c - (1 - K) is noise
                if exactPuts && ks[i][j] < 1 && (priceMap == :box || isKernelMap(priceMap))
                    put = priceMap == :box ? ahBoxPut(dmap, ks[i][j]) : ahKernelPut(dmap, ks[i][j])
                    fvec[r] = useVol ? w[j] * (impliedVolatilityThiophene(false, max(put, 1e-64), one(T), ks[i][j], ttes[i], one(T)) - vols[j]) :
                              w[j] * (put + (1 - ks[i][j]) - cs[i][j])
                    continue
                end
                st = stencil[j]
                mPrice = if priceMap == :density
                    ahDensityPrice(dmap, ks[i][j])
                elseif isKernelMap(priceMap)
                    ahKernelPrice(dmap, ks[i][j])
                elseif priceMap == :box
                    ahBoxPrice(dmap, ks[i][j])
                elseif length(st[1]) == 1
                    c[st[1][1]] #a grid point: the grid price
                elseif priceMap == :linear
                    ahLinearAt(z, c, ks[i][j])
                elseif priceMap in (:grid, :schaback, :variance)
                    ahSplineAt(dmap, ks[i][j])
                elseif length(st[1]) == 1
                    c[st[1][1]]
                else
                    #interpolate the logarithm of the out-of-the-money price (smooth, nearly quadratic in y; the price
                    #itself falls by orders of magnitude between grid points in the far wings)
                    otm(q) = (zq = z[st[1][q]]; zq >= 1 ? c[st[1][q]] : c[st[1][q]] - (1 - zq))
                    if all(q -> otm(q) > 1e-200, eachindex(st[1]))
                        v = exp(sum(st[2][q] * log(otm(q)) for q in eachindex(st[1])))
                        ks[i][j] >= 1 ? v : v + (1 - ks[i][j])
                    else #prices at the rounding level: interpolate the prices themselves
                        sum(st[2][q] * c[st[1][q]] for q in eachindex(st[1]))
                    end
                end
                fvec[r] = useVol ? w[j] * (vol(mPrice, ks[i][j], ttes[i]) - vols[j]) : w[j] * (mPrice - cs[i][j])
            end
            r = length(inside)
            if nPenS > 0
                for k in dIdx
                    r += 1; fvec[r] = penaltyStrike * (coef[k+1] - 2coef[k] + coef[k-1])
                end
            end
            if nPenT > 0
                σPen = BPen * coef
                for k = 1:nPenT
                    r += 1; fvec[r] = penaltyTime * (abs(σPen[k]) - σPrevPen[k])
                end
            end
            fvec
        end
        fvec = zeros(T, nres)
        p = if startswith(optimizer, "GN")
            pp = copy(p0)
            GaussNewton.optimize!(obj!, pp, fvec, autodiff=:single, iscale=1, abstol=length(fvec) * eps())
            pp
        else
            obj1 = OnceDifferentiable(obj!, p0, copy(fvec); autodiff=LsqFit.AutoForwardDiff(), inplace=true)
            LsqFit.levenberg_marquardt(obj1, p0).minimizer
        end
        coef = coefficients(p, T)
        states[i] = solveStep(coef)
        prices[i] = pricesOf(states[i])
        coeffs[i] = coef
        wingMinVols[i] = (minVols[1], minVols[nb])
    end
    #the cache keeps, under the key 0, the lower bounds of the outer coefficients of each step (left, right)
    return BSplineAHSurface(collect(ttes), collect(forwards), z, knotsList, coeffs, prices, priceMap, Dict{Int,Any}(0 => wingMinVols), scheme,
        scheme == :adjoint || scheme == :p1 || isFEMScheme(scheme) ? states : Vector{T}[])
end
"one step of the scheme of the surface from the masses Pprev"
massStep(s::BSplineAHSurface, Pprev, σ, dt) = s.scheme == :p1 ? ahP1Step(s.z, Pprev, σ, dt) : isFEMScheme(s.scheme) ? ahKernelStep(s.z, Pprev, σ, dt; degree=AH_FEM_DEGREE[s.scheme], centred=femCentred(s.scheme)) : ahAdjointStep(s.z, Pprev, σ, dt)

"""
time interpolation of the linear finite elements (scheme=:p1). One step of size τ < Tᵢ - Tᵢ₋₁ from the previous expiry is not
covered by the floor that keeps the step of the expiry an M-matrix (the floor grows as 1/√τ): its density turns negative
and its prices can fall below those of the previous expiry. Between two expiries the total implied variance is linear
in time at a fixed log-moneyness, w(t, y) = (1 - α) wᵢ₋₁(y) + α wᵢ(y), α = (t - Tᵢ₋₁)/(Tᵢ - Tᵢ₋₁), w₀ = 0, from the exact
prices of the two expiries (two implied volatility lookups per price, four with the derivatives of the local
volatility). It keeps the calendar spreads of the expiries; the butterflies in between are not guaranteed. Beyond the
last expiry, see `p1TotalVariance`.
"""
isP1Between(s::BSplineAHSurface, t) = s.scheme == :p1 && (t > s.expiries[end] || (i = searchsortedfirst(s.expiries, t); abs(t - s.expiries[i]) >= 1e-12))
"""
the slice of the expiry i (ahP1Slice) and the range of log-moneyness of the inner grid points where its out-of-the-money
price is above 1e-250, within which the implied variance and its derivatives are accurate
"""
function p1SliceAt(s::BSplineAHSurface, i::Int)
    return get!(s.cache, length(s.expiries) + i) do
        sl = ahP1Slice(s.z, s.masses[i])
        ok = [j for j = 2:length(s.z)-1 if ahP1OTM(sl, s.z[j])[1] > 1e-250]
        (slice=sl, yMin=log(s.z[ok[1]]), yMax=log(s.z[ok[end]]))
    end
end
"""
the total variance of the expiry i at log-moneyness y (zero for i = 0) and its derivatives in y; linear in y beyond the
range of `p1SliceAt`, where the prices are negligible
"""
function p1ExpiryVariance(s::BSplineAHSurface, i::Int, y)
    i == 0 && return (w=zero(y), wy=zero(y), wyy=zero(y))
    c = p1SliceAt(s, i)
    yc = clamp(y, c.yMin, c.yMax)
    v = ahP1TotalVariance(c.slice, yc, s.expiries[i])
    return yc == y ? v : (w=max(v.w + v.wy * (y - yc), v.w / 2), wy=v.wy, wyy=zero(y))
end
"""
total implied variance of the linear finite elements at log-moneyness y and time t: linear in t between the expiries (see
`isP1Between`), with its derivatives in y and in t. At an expiry, ∂ₜw is that of the interval before it (the left limit).
Beyond the last expiry Tₙ, the total variance grows by a constant forward variance, the at-the-money forward variance of
the last interval: w(t, y) = wₙ(y) + (t - Tₙ) v∞, v∞ = (wₙ(0) - wₙ₋₁(0)) / (Tₙ - Tₙ₋₁). It keeps the calendar spreads
(∂ₜw = v∞ ≥ 0) and the skew ∂ᵧw of the last expiry, within Lee's bounds at all times, and the Dupire local volatility
stays near √v∞ instead of vanishing (a frozen w), while continuing the last interval linearly makes ∂ᵧw grow with t and
breaks the butterflies.
"""
function p1TotalVariance(s::BSplineAHSurface, y, t)
    n = length(s.expiries)
    if t > s.expiries[n]
        tn = s.expiries[n]; t0 = n == 1 ? zero(t) : s.expiries[n-1]
        vInf = max((p1ExpiryVariance(s, n, zero(y)).w - p1ExpiryVariance(s, n - 1, zero(y)).w) / (tn - t0), zero(y))
        b = p1ExpiryVariance(s, n, y)
        return (w=b.w + (t - tn) * vInf, wy=b.wy, wyy=b.wyy, wt=vInf)
    end
    i = searchsortedfirst(s.expiries, t)
    t0 = i == 1 ? zero(t) : s.expiries[i-1]
    a = p1ExpiryVariance(s, i - 1, y); b = p1ExpiryVariance(s, i, y)
    α = (t - t0) / (s.expiries[i] - t0)
    return (w=(1 - α) * a.w + α * b.w, wy=(1 - α) * a.wy + α * b.wy, wyy=(1 - α) * a.wyy + α * b.wyy, wt=(b.w - a.w) / (s.expiries[i] - t0))
end
"normalized call price of the linear finite elements between expiries, from the interpolated total variance"
p1PriceBetween(s::BSplineAHSurface, x, t) = blackScholesFormula(true, x, one(x), max(p1TotalVariance(s, log(x), t).w, zero(x)), one(x), one(x))

"normalized call prices on the grid at time t"
function pricesOnGrid(s::BSplineAHSurface, t)
    if t >= s.expiries[end]
        #finite elements: the extrapolated total variance; the other schemes have no step beyond the last expiry
        t > s.expiries[end] && s.scheme == :p1 && return [j == 1 || j == length(s.z) ? max(1 - zj, zero(zj)) : p1PriceBetween(s, zj, t) for (j, zj) in enumerate(s.z)]
        return s.prices[end]
    end
    i = searchsortedfirst(s.expiries, t)
    abs(t - s.expiries[i]) < 1e-12 && return s.prices[i]
    s.scheme == :p1 && return [j == 1 || j == length(s.z) ? max(1 - zj, zero(zj)) : p1PriceBetween(s, zj, t) for (j, zj) in enumerate(s.z)]
    t0 = i == 1 ? zero(t) : s.expiries[i-1]
    σ = ahBasisMatrix(s.knots[i], s.z) * s.coeffs[i]
    if s.scheme == :adjoint || s.scheme == :p1 || isFEMScheme(s.scheme)
        prevP = i == 1 ? [zj == 1 ? one(t) : zero(t) for zj in s.z] : s.masses[i-1]
        P = massStep(s, prevP, σ, t - t0)
        s.scheme == :p1 && (d = ahP1Map(s.z, P); return [ahDensityPrice(d, zk) for zk in s.z])
        isFEMScheme(s.scheme) && (d = ahKernelMap(s.z, P; degree=AH_FEM_DEGREE[s.scheme]); return [ahKernelPrice(d, zk) for zk in s.z])
        return ahMassPrices(s.z, P)
    end
    prev = i == 1 ? [max(1 - zj, zero(zj)) for zj in s.z] : s.prices[i-1]
    return ahStep(s.z, prev, σ, t - t0)
end

"""
nodes of the interpolation between the grid points: the grid points at least a fifth of the median spacing in
log-moneyness apart (and the forward). The inserted strikes of different expiries can be 3e-6 apart, and a spline
through such nearly coincident points turns the small errors of the scheme into large oscillations of its derivatives
(the density); the prices at the dropped points are still matched to the accuracy of the scheme.
"""
function interpolationNodes(z::AbstractVector)
    y = log.(z)
    gap = sort(diff(y))[div(length(y), 2)] / 5
    keep = [1]
    for j = 2:length(z)
        (y[j] - y[keep[end]] > gap || z[j] == 1 || j == length(z)) && push!(keep, j)
    end
    return keep
end
gridSpline(s::BSplineAHSurface, c) = (sp = ahNaturalSpline(s.z, c); x -> ahSplineAt(sp, x))

"normalized call price (forward 1) at forward moneyness x and time t, by the price map of the surface"
function priceNormalized(s::BSplineAHSurface, x, t)
    (x <= s.z[1] || x >= s.z[end]) && return max(1 - x, zero(x))
    isP1Between(s, t) && return p1PriceBetween(s, x, t)
    s.priceMap == :density && return ahDensityPrice(densityMapAt(s, t), x)
    isKernelMap(s.priceMap) && return ahKernelPrice(kernelMapAt(s, t), x)
    s.priceMap == :box && return ahBoxPrice(boxMapAt(s, t), x)
    if s.priceMap == :linear #the point masses of the grid (one-pass Fokker-Planck formula): linear in the strike
        c = pricesOnGrid(s, t); j = clamp(searchsortedlast(s.z, x), 1, length(s.z) - 1); u = (x - s.z[j]) / (s.z[j+1] - s.z[j])
        return (1 - u) * c[j] + u * c[j+1]
    end
    if s.priceMap == :variance
        i = searchsortedfirst(s.expiries, t)
        d = if i <= length(s.expiries) && abs(t - s.expiries[i]) < 1e-12
            get!(() -> ahVarianceMap(s.z, s.prices[i], t), s.cache, i)
        else
            ahVarianceMap(s.z, pricesOnGrid(s, t), t)
        end
        return ahVariancePrice(d, x)
    end
    if s.priceMap == :schaback
        i = searchsortedfirst(s.expiries, t)
        if i <= length(s.expiries) && abs(t - s.expiries[i]) < 1e-12
            return ahSchabackPrice(get!(() -> ahSchabackMap(s.z, s.prices[i]), s.cache, i), x)
        end
        return ahSchabackPrice(ahSchabackMap(s.z, pricesOnGrid(s, t)), x)
    end
    return gridSpline(s, pricesOnGrid(s, t))(x)
end

"""
the density price map at time t (the step containing t, of size t - T_{i-1}; the last expiry beyond it). For scheme=:p1
between expiries, this one step is not the model, whose prices there come from the total variance (`isP1Between`).
"""
function densityMapAt(s::BSplineAHSurface, t)
    i = min(searchsortedfirst(s.expiries, t), length(s.expiries))
    tt = min(t, s.expiries[end])
    t0 = i == 1 ? zero(tt) : s.expiries[i-1]
    if s.scheme == :p1
        abs(tt - s.expiries[i]) < 1e-12 && return get!(() -> ahP1Map(s.z, s.masses[i]), s.cache, -i)
        prevL = i == 1 ? [zj == 1 ? one(tt) : zero(tt) for zj in s.z] : s.masses[i-1]
        return ahP1Map(s.z, ahP1Step(s.z, prevL, ahBasisMatrix(s.knots[i], s.z) * s.coeffs[i], tt - t0))
    end
    if s.scheme == :adjoint
        abs(tt - s.expiries[i]) < 1e-12 && return get!(() -> ahMassDensityMap(s.z, s.masses[i]), s.cache, -i)
        prevP = i == 1 ? [zj == 1 ? one(tt) : zero(tt) for zj in s.z] : s.masses[i-1]
        return ahMassDensityMap(s.z, ahAdjointStep(s.z, prevP, ahBasisMatrix(s.knots[i], s.z) * s.coeffs[i], tt - t0))
    end
    prev = i == 1 ? [max(1 - zj, zero(zj)) for zj in s.z] : s.prices[i-1]
    c = abs(tt - s.expiries[i]) < 1e-12 ? s.prices[i] : pricesOnGrid(s, tt)
    return ahStepDensityMap(s.z, interpolationNodes(s.z), c, prev, ahBasisMatrix(s.knots[i], s.z) * s.coeffs[i], tt - t0)
end

"the kernel price map at time t (the masses of the step containing t; the last expiry beyond it)"
function kernelMapAt(s::BSplineAHSurface, t)
    i = min(searchsortedfirst(s.expiries, t), length(s.expiries))
    tt = min(t, s.expiries[end])
    deg = AH_KERNEL_DEGREE[s.priceMap]
    abs(tt - s.expiries[i]) < 1e-12 && return get!(() -> ahKernelMap(s.z, s.masses[i]; degree=deg), s.cache, -i)
    t0 = i == 1 ? zero(tt) : s.expiries[i-1]
    prevP = i == 1 ? [zj == 1 ? one(tt) : zero(tt) for zj in s.z] : s.masses[i-1]
    return ahKernelMap(s.z, massStep(s, prevP, ahBasisMatrix(s.knots[i], s.z) * s.coeffs[i], tt - t0); degree=deg)
end

"""
calibrated local volatility spline σᵢ of the step containing t, at forward moneyness x: the local volatility of one
implicit step from Tᵢ₋₁ to Tᵢ, not the Dupire local volatility of the surface (see `dupireLocalVolatilityAt`)
"""
function localVolatility(s::BSplineAHSurface, x, t)
    i = min(searchsortedfirst(s.expiries, t), length(s.expiries))
    return abs(ahLocalVol(s.knots[i], s.coeffs[i], x))
end

"""
    dupireLocalVolatilityAt(s::BSplineAHSurface, t; minDensity=1e-10)

Dupire local volatility of the surface at time t, σ²(t, x) = ∂ₜc / (½x²∂ₓₓc), as a function of the forward moneyness x:
the local volatility of the diffusion with the prices of the model at all times. Between two expiries, the prices of the
model are those of one implicit step of size τ = t - Tᵢ₋₁, c(τ) = (I - τGᵢ)⁻¹cᵢ₋₁ (arbitrage-free in τ), whose time
derivative is exactly ∂τc = ((I - τGᵢ)⁻¹c(τ) - c(τ)) / τ: one more step of the same size. It is evaluated at the grid
points, ∂ₓₓc being the density Pₖ / ½(zₖ₊₁ - zₖ₋₁) (the second divided difference of the grid prices), and interpolated by a
natural cubic spline in log x through the grid points where the density exceeds `minDensity` times its peak, flat beyond.
It equals σᵢ just after Tᵢ₋₁ and departs from it during the step: σᵢ itself, used in a continuous-time diffusion, does not
reprice the vanillas of the model. Needs scheme=:adjoint or :p1.

For scheme=:p1, the prices between expiries are those of the total variance linear in time (`isP1Between`), and the local
volatility is Gatheral's form of the Dupire formula, σ² = ∂ₜw / g(w, ∂ᵧw, ∂ᵧᵧw), at any log-moneyness (the derivatives in y
from the exact prices, slopes and densities of the two expiries), constant in t between expiries for ∂ₜw, with the
constant forward variance of `p1TotalVariance` beyond the last expiry; zero where g ≤ 0 (no density) and beyond the inner
grid points the variance is flat in y.
"""
function dupireLocalVolatilityAt(s::BSplineAHSurface, t; minDensity=1e-10)
    if s.scheme == :p1
        return function (x)
            y = log(x); v = p1TotalVariance(s, y, t)
            g = (1 - y * v.wy / (2v.w))^2 - v.wy^2 / 4 * (1 / 4 + 1 / v.w) + v.wyy / 2
            return g > 0 && v.wt > 0 ? sqrt(v.wt / g) : zero(x)
        end
    end
    s.scheme == :adjoint || throw(ArgumentError("dupireLocalVolatilityAt needs scheme=:adjoint or :p1"))
    z = s.z; n = length(z)
    i = min(searchsortedfirst(s.expiries, t), length(s.expiries))
    t0 = i == 1 ? zero(t) : s.expiries[i-1]
    τ = max(t - t0, 1e-8)
    σ = ahBasisMatrix(s.knots[i], z) * s.coeffs[i]
    Pprev = i == 1 ? [zj == 1 ? one(eltype(z)) : zero(eltype(z)) for zj in z] : s.masses[i-1]
    P = ahAdjointStep(z, Pprev, σ, τ)
    c = ahMassPrices(z, P); c2 = ahMassPrices(z, ahAdjointStep(z, P, σ, τ))
    f = [k == 1 || k == n ? zero(eltype(P)) : P[k] / ((z[k+1] - z[k-1]) / 2) for k = 1:n]
    fMax = maximum(f)
    keep = Int[]; v = eltype(P)[]
    for k in interpolationNodes(z)
        (k == 1 || k == n || f[k] <= minDensity * fMax) && continue
        v2 = (c2[k] - c[k]) / (τ * z[k]^2 * f[k] / 2)
        v2 > 0 && (push!(keep, k); push!(v, sqrt(v2)))
    end
    length(keep) < 3 && return x -> abs(ahLocalVol(s.knots[i], s.coeffs[i], x)) #t → Tᵢ₋₁: the limit is σᵢ
    sp = ahNaturalSpline(log.(z[keep]), v)
    yMin, yMax = log(z[keep[1]]), log(z[keep[end]])
    return x -> ahSplineAt(sp, clamp(log(x), yMin, yMax))
end

"""
    dupireLocalVolatility(s::BSplineAHSurface; kw...)

the Dupire local volatility of the surface as a function (x, t) -> σ, `dupireLocalVolatilityAt` memoized by t (not
thread-safe: one per thread)
"""
function dupireLocalVolatility(s::BSplineAHSurface; kw...)
    memo = Dict{Float64,Any}()
    return (x, t) -> get!(() -> dupireLocalVolatilityAt(s, t; kw...), memo, Float64(t))(x)
end

"""
    densityNormalized(s::BSplineAHSurface, x, i)

Density of x = K/F at the expiry Tᵢ, from the equation of the step: cᵢ'' = (cᵢ - cᵢ₋₁) / (½ (Tᵢ - Tᵢ₋₁) σᵢ(x)² x²). It is the
density of the model, without a numerical second derivative (the second differences of the grid prices, or of an
interpolation of them, amplify the small errors of the prices on an irregular grid).
"""
function densityNormalized(s::BSplineAHSurface, x, i::Int)
    (x <= s.z[1] || x >= s.z[end]) && return zero(x)
    s.priceMap == :density && return ahDensityAt(densityMapAt(s, s.expiries[i]), x)
    isKernelMap(s.priceMap) && return ahKernelDensityAt(kernelMapAt(s, s.expiries[i]), x)
    s.priceMap == :box && return ahBoxDensityAt(boxMapAt(s, s.expiries[i]), x)
    if s.scheme == :adjoint #the Fokker-Planck masses over their widths, linear in between
        z = s.z; P = s.masses[i]; n = length(z)
        dens(j) = j == 1 || j == n ? zero(x) : P[j] / ((z[j+1] - z[j-1]) / 2)
        j = clamp(searchsortedlast(z, x), 1, n - 1); u = (x - z[j]) / (z[j+1] - z[j])
        return (1 - u) * dens(j) + u * dens(j + 1)
    end
    t0 = i == 1 ? zero(x) : s.expiries[i-1]
    c = gridSpline(s, s.prices[i])(x)
    cPrev = i == 1 ? max(1 - x, zero(x)) : gridSpline(s, s.prices[i-1])(x)
    σ = ahLocalVol(s.knots[i], s.coeffs[i], x)
    return (c - cPrev) / ((s.expiries[i] - t0) * σ^2 * x^2 / 2)
end

"the masses of the grid points at time t (schemes on masses): those of the expiry, or one step of size t - Tᵢ₋₁"
function massesAt(s::BSplineAHSurface, t)
    i = min(searchsortedfirst(s.expiries, t), length(s.expiries))
    tt = min(t, s.expiries[end])
    abs(tt - s.expiries[i]) < 1e-12 && return s.masses[i]
    t0 = i == 1 ? zero(tt) : s.expiries[i-1]
    prevP = i == 1 ? [zj == 1 ? one(tt) : zero(tt) for zj in s.z] : s.masses[i-1]
    return massStep(s, prevP, ahBasisMatrix(s.knots[i], s.z) * s.coeffs[i], tt - t0)
end
"the box map at time t, from the masses of the scheme when it has them"
boxMapAt(s::BSplineAHSurface, t) = s.scheme == :adjoint ? (P = massesAt(s, t); ahBoxMap(s.z, ahMassPrices(s.z, P); P=P)) : ahBoxMap(s.z, pricesOnGrid(s, t))
"""
normalized out-of-the-money put price E[(x - X)⁺] at forward moneyness x and time t: from the masses below x for the box
and kernel maps (full relative precision far out of the money), by put-call parity otherwise
"""
function putNormalized(s::BSplineAHSurface, x, t)
    x <= s.z[1] && return zero(x)
    x >= s.z[end] && return x - 1
    s.priceMap == :box && return ahBoxPut(boxMapAt(s, t), x)
    isKernelMap(s.priceMap) && return ahKernelPut(kernelMapAt(s, t), x)
    if s.scheme == :p1
        isP1Between(s, t) && return blackScholesFormula(false, x, one(x), max(p1TotalVariance(s, log(x), t).w, zero(x)), one(x), one(x))
        x < 1 && return ahP1OTM(p1SliceAt(s, min(searchsortedfirst(s.expiries, t), length(s.expiries))).slice, x)[1]
    end
    return priceNormalized(s, x, t) - (1 - x)
end
"Black variance at log forward moneyness y and time t"
function varianceByLogmoneyness(s::BSplineAHSurface, y, t)
    s.scheme == :p1 && return p1TotalVariance(s, y, t).w / t
    x = exp(y)
    isCall = x >= 1
    price = isCall ? max(priceNormalized(s, x, t), 1e-16) : max(putNormalized(s, x, t), 1e-64)
    vol = impliedVolatilityThiophene(isCall, price, one(x), x, t, one(x))
    return vol^2
end
