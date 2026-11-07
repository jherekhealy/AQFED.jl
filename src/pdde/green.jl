#The one-step local volatility model of Andreasen and Huge with a smooth (quadratic B-spline) local volatility per expiry
#interval, as in calibrateBSplineAHSurface, solved exactly by the Green's function of the step instead of a grid. One
#implicit step of the forward Dupire equation, c - S c'' = c_prev with S(x) = ½ Δt σ(x)² x², is, for the density p = c'' of
#x = K/F,
#    p(x) = (1/S(x)) ∫ G(x, ξ) p_prev(ξ) dξ,   G(x, ξ) = φ₋(min(x, ξ)) φ₊(max(x, ξ)) / W,
#with φ₋, φ₊ the solutions of φ'' = φ/S that decay at x → 0 and x → ∞ and W their (constant) Wronskian: G > 0. In z = ln x,
#ψ = ln φ solves the Riccati equation v' = q + v - v², v = ψ', q = 2/(Δt σ²), integrated by Chebyshev collocation on elements
#between the knots of σ (where q is analytic), and in closed form beyond the outer knots (σ flat). The density of each
#expiry is kept as ln p at Gauss-Legendre nodes of segments between all the knots, and every integral is a sum of
#exponentials with positive Gauss weights: the density is positive by construction, at every point, and the prices
#increase with the expiry. The convergence is spectral in the Chebyshev and Gauss orders. Optionally, each expiry interval
#is made of several implicit substeps with the same local volatility (and the same Green's function): the model then tends
#to the diffusion with a local volatility constant in time over each interval, whose Dupire local volatility is σᵢ.
#Reference: F.L.Y. LeFloch and C. Opus, "The one-step local volatility model solved by its Green's function" (2026).
module GreenAH
using LinearAlgebra
import ForwardDiff, QuadGK
import AQFED
import AQFED.Black: blackScholesFormula, impliedVolatilityThiophene
import ..PDDE
import ..PDDE: priceNormalized, putNormalized, varianceByLogmoneyness, densityNormalized, localVolatility
const PD = PDDE
const LsqFit = PDDE.LsqFit
export GreenAHSurface, calibrateGreenAHSurface, greenDupireAtExpiry, greenDupireLocalVolatility

#--- Chebyshev and Gauss-Legendre tools
"Chebyshev points of the second kind on [-1, 1] (descending) and the differentiation matrix"
function chebD(N)
    x = cos.(π * (0:N) / N); c = [2; ones(N - 1); 2] .* (-1) .^ (0:N)
    X = repeat(x, 1, N + 1); dX = X - X'
    D = (c * (1 ./ c)') ./ (dX + I); D = D - Diagonal(vec(sum(D, dims=2)))
    D, x
end
const CHEB = Dict{Int,Tuple{Matrix{Float64},Vector{Float64},Vector{Float64}}}()
function cheb(N)
    get!(CHEB, N) do
        D, x = chebD(N)
        D, x, [(-1.0)^k * (k == 0 || k == N ? 0.5 : 1.0) for k in 0:N]
    end
end
const GAUSS = Dict{Int,Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}}()
"Gauss-Legendre nodes, weights and barycentric weights on [-1, 1]"
function gauss(n)
    get!(GAUSS, n) do
        x, w = QuadGK.gauss(n)
        x, w, [(-1.0)^j * sqrt((1 - x[j]^2) * w[j]) for j in eachindex(x)]
    end
end
"barycentric interpolation of f at the nodes z (barycentric weights bw) at t"
function bary(z, f::AbstractVector{T}, bw, t) where {T}
    num = zero(T); den = 0.0
    @inbounds for k in eachindex(z)
        d = t - z[k]
        d == 0 && return f[k]
        a = bw[k] / d
        num += a * f[k]; den += a
    end
    num / den
end
function logsumexp(a::AbstractVector{T}) where {T}
    isempty(a) && return T(-Inf)
    m = maximum(a); isinf(m) && return m
    s = zero(T)
    @inbounds for x in a
        s += exp(x - m)
    end
    m + log(s)
end
"ln of the running sums of exp.(a) (cumulative from the left, or from the right with rev), one exp per term"
function logcumsumexp(a::AbstractVector{T}; rev=false) where {T}
    out = similar(a); m = T(-Inf); sc = zero(T)
    for r in (rev ? reverse(eachindex(a)) : eachindex(a))
        x = a[r]
        if x > m
            sc = sc * exp(m - x) + 1; m = x
        elseif x > -Inf
            sc += exp(x - m)
        end
        out[r] = m + log(sc)
    end
    out
end
logaddexp(a, b) = (m = max(a, b); isinf(m) && m < 0 ? m : m + log(exp(a - m) + exp(b - m)))

#--- one step: φ₋ and φ₊ through their logarithms
"""
the step of length dt with the local volatility spline (knots in K/F, coefficients coeffs, flat beyond the outer knots):
ψ = ln φ and v = ψ' of φ₋ (index 1) and φ₊ (index 2) at the Chebyshev nodes of the elements, ψ(0) = 0, and the Wronskian
W = φ₋'φ₊ - φ₋φ₊' (in x, at x = 1)
"""
struct GreenStep{T}
    dt::Float64
    knots::Vector{Float64}
    coeffs::Vector{T}
    edges::Vector{Float64}        # element boundaries in z
    nodes::Vector{Vector{Float64}}
    ψ::NTuple{2,Vector{Vector{T}}}
    v::NTuple{2,Vector{Vector{T}}}
    qL::T; qR::T
    W::T
end
roots(q) = ((1 + sqrt(1 + 4q)) / 2, (1 - sqrt(1 + 4q)) / 2)
"Riccati v' = q + v - v² on the Chebyshev nodes z of an element, from v0 at its left (forward) or right end"
function riccati(D, q::AbstractVector{T}, v0::T, forward) where {T}
    N = length(q) - 1; i0 = forward ? N + 1 : 1
    qv = ForwardDiff.value.(q); v0v = ForwardDiff.value(v0)
    v = fill(v0v, N + 1); J = similar(D)
    for _ in 1:60
        F = D * v .+ v .^ 2 .- v .- qv
        J .= D; for k in 1:N+1; J[k, k] += 2v[k] - 1; end
        F[i0] = v[i0] - v0v; J[i0, :] .= 0; J[i0, i0] = 1
        dv = J \ F; v .-= dv
        maximum(abs, dv) <= 1e-14 * max(1.0, maximum(abs, v)) && break
    end
    T === Float64 && return v
    #one Newton step in the dual numbers from the converged values: exact first derivatives
    vd = T.(v)
    F = D * vd .+ vd .^ 2 .- vd .- q
    F[i0] = vd[i0] - v0
    vd = vd .- (lu(J) \ F)
    vd[i0] = v0
    return vd
end
function integratePsi(D, v::AbstractVector{T}, ψ0::T, forward) where {T}
    N = length(v) - 1; i0 = forward ? N + 1 : 1
    A = copy(D); A[i0, :] .= 0; A[i0, i0] = 1
    r = copy(v); r[i0] = ψ0
    lu(A) \ r
end
"""
the step from the knots and coefficients: elements between the knots and the forward, each split into `sub` pieces, with
Chebyshev degree N; Bel (optional) the basis matrices of the element nodes, reused across calls with other coefficients
"""
function greenStep(knots, coeffs::AbstractVector{T}, dt; N=16, sub=2, Bel=nothing) where {T}
    b = sort(unique(vcat(log.(knots), 0.0)))
    edges = Float64[]
    for k in 1:length(b)-1, j in 0:sub-1
        push!(edges, b[k] + (b[k+1] - b[k]) * j / sub)
    end
    push!(edges, b[end])
    D0, t, _ = cheb(N)
    ne = length(edges) - 1
    nodes = [(edges[e] + edges[e+1]) / 2 .+ (edges[e+1] - edges[e]) / 2 .* t for e in 1:ne]
    Ds = [D0 .* (2 / (edges[e+1] - edges[e])) for e in 1:ne]
    Bel = something(Bel, [PD.ahBasisMatrix(knots, exp.(z)) for z in nodes])
    qs = [2 ./ (dt .* (B * coeffs) .^ 2) for B in Bel]
    qL = 2 / (dt * coeffs[1]^2); qR = 2 / (dt * coeffs[end]^2)
    ψm = Vector{Vector{T}}(undef, ne); vm = similar(ψm); ψp = similar(ψm); vp = similar(ψm)
    v0 = T(roots(qL)[1]); ψ0 = zero(T)
    for e in 1:ne
        vm[e] = riccati(Ds[e], qs[e], v0, true); ψm[e] = integratePsi(Ds[e], vm[e], ψ0, true)
        v0 = vm[e][1]; ψ0 = ψm[e][1]
    end
    v0 = T(roots(qR)[2]); ψ0 = zero(T)
    for e in ne:-1:1
        vp[e] = riccati(Ds[e], qs[e], v0, false); ψp[e] = integratePsi(Ds[e], vp[e], ψ0, false)
        v0 = vp[e][end]; ψ0 = ψp[e][end]
    end
    s = GreenStep{T}(dt, collect(Float64, knots), coeffs, edges, nodes, (ψm, ψp), (vm, vp), qL, qR, zero(T))
    #normalize at z = 0 and the Wronskian there
    a = psiAt(s, 0.0, 1); c = psiAt(s, 0.0, 2)
    for e in 1:ne
        ψm[e] .-= a[1]; ψp[e] .-= c[1]
    end
    GreenStep{T}(s.dt, s.knots, coeffs, edges, nodes, (ψm, ψp), (vm, vp), qL, qR, a[2] - c[2])
end
"ψ and ψ' of φ₋ (which = 1) or φ₊ (which = 2) at z, in closed form beyond the outer knots"
function psiAt(s::GreenStep{T}, z, which) where {T}
    ψs, vs = s.ψ[which], s.v[which]
    ne = length(s.nodes); zL, zR = s.edges[1], s.edges[end]
    if z < zL || z > zR
        left = z < zL
        vP, vM = roots(left ? s.qL : s.qR)
        e = left ? 1 : ne; k = left ? length(s.nodes[1]) : 1
        ψ0, v0 = ψs[e][k], vs[e][k]; d = z - (left ? zL : zR)
        #the decaying solution in its own tail is the pure exponential (its other coefficient is zero exactly; computed,
        #its derivative is rounding noise, which the tiny exponential it multiplies would blow up)
        which == 1 && left && return ψ0 + vP * d, vP
        which == 2 && !left && return ψ0 + vM * d, vM
        A = (v0 - vM) / (vP - vM); B = (vP - v0) / (vP - vM)
        t1 = vP * d; t2 = vM * d
        if t1 >= t2
            r = A + B * exp(t2 - t1)
            return ψ0 + t1 + log(r), (A * vP + B * vM * exp(t2 - t1)) / r
        else
            r = B + A * exp(t1 - t2)
            return ψ0 + t2 + log(r), (B * vM + A * vP * exp(t1 - t2)) / r
        end
    end
    e = clamp(searchsortedlast(s.edges, z), 1, ne)
    _, _, bw = cheb(length(s.nodes[e]) - 1)
    bary(s.nodes[e], ψs[e], bw, z), bary(s.nodes[e], vs[e], bw, z)
end
psi(s, z, which) = psiAt(s, z, which)[1]
"ln G(x, ξ) W in z, ζ (no Wronskian)"
logGW(s, z, ζ) = z <= ζ ? psi(s, z, 1) + psi(s, ζ, 2) : psi(s, ζ, 1) + psi(s, z, 2)
"S(x) = ½ dt σ(x)² x²"
Sof(s::GreenStep, x, σ) = s.dt / 2 * σ^2 * x^2
σAt(knots, coeffs, x) = (B = PD.ahBasisMatrix(knots, [x]); dot(view(B, 1, :), coeffs))

#--- the density of an expiry: ln p at the Gauss nodes of segments in z
struct Density
    edges::Vector{Float64}          # segment boundaries
    n::Int
    nodes::Vector{Vector{Float64}}  # Gauss nodes of each segment
    lp::Vector{Vector{Float64}}     # ln p (density of x) at the nodes
end
"""
the rate bound √(2/(Δ σ²)) of a step on a grid of z between its outer knots (flat beyond), for the widths of the segments
"""
struct StepRate
    z0::Float64; h::Float64; R::Vector{Float64}
end
function stepRate(knots, coeffs, dt; h=0.005)
    #just inside the outer knots (the basis vanishes at the last one)
    zs = collect(range(log(knots[1]) + 1e-10, log(knots[end]) - 1e-10, length=max(2, ceil(Int, log(knots[end] / knots[1]) / h) + 1)))
    σ = PD.ahBasisMatrix(knots, exp.(zs)) * ForwardDiff.value.(coeffs)
    StepRate(zs[1], zs[2] - zs[1], sqrt.(2 ./ (dt .* σ .^ 2)))
end
function rateAt(r::StepRate, z)
    t = (z - r.z0) / r.h
    t <= 0 && return r.R[1]
    k = floor(Int, t) + 1
    k >= length(r.R) ? r.R[end] : max(r.R[k], r.R[k+1])
end
rateOf(rs) = z -> maximum(rateAt(r, z) for r in rs)

"""
segments over [zmin, zmax], split at the breakpoints, of width at most τ/R(z) (R a bound on the exponential rate of the
integrands, from the steps so far and the floor of the next one) and hmax
"""
function makeSegments(breaks, zmin, zmax, rate; τ=6.0, hmax=0.05, hfar=0.5)
    b = sort(unique(vcat(filter(t -> zmin < t < zmax, breaks), zmin, zmax)))
    edges = Float64[b[1]]
    for k in 1:length(b)-1
        a, c = b[k], b[k+1]
        R = maximum(rate(t) for t in range(a, c, length=9))
        #hmax within the knots, hfar beyond them (the density is then nearly log-linear)
        inner = a >= breaks[1] && c <= breaks[end]
        #beyond the outer knots, the rate bound only within 2 of them: further, all the local volatilities are flat and ln p is
        #a smooth mixture of powers of x
        m = inner ? max(1, ceil(Int, (c - a) / hmax), ceil(Int, (c - a) * R / τ)) :
            max(1, ceil(Int, (c - a) / hfar), ceil(Int, min(c - a, 2.0) * R / τ))
        for j in 1:m
            push!(edges, a + (c - a) * j / m)
        end
    end
    edges
end
segmentNodes(edges, n) = (x = gauss(n)[1]; [(edges[s] + edges[s+1]) / 2 .+ (edges[s+1] - edges[s]) / 2 .* x for s in 1:length(edges)-1])
"ln p at z (-Inf outside the segments)"
function logDensity(d::Density, z)
    (z < d.edges[1] || z > d.edges[end]) && return -Inf
    s = clamp(searchsortedlast(d.edges, z), 1, length(d.nodes))
    bary(d.nodes[s], d.lp[s], gauss(d.n)[3], z)
end
"the first expiry from the Dirac at the forward: p = G(x, 1)/S, on the given segments"
function firstDensity(s::GreenStep, edges, n)
    nodes = segmentNodes(edges, n)
    lp = [[psi(s, z, z <= 0 ? 1 : 2) - log(s.W * Sof(s, exp(z), σAt(s.knots, s.coeffs, exp(z)))) for z in nz] for nz in nodes]
    Density(edges, n, nodes, ForwardDiff.value.(lp) .|> collect)
end

"""
integration data of a step from the density prev on the segments `edges`: the nodes, ln(weight) + ln p_prev + ζ at the
nodes (Float64, independent of the coefficients of the step)
"""
struct StepTable
    edges::Vector{Float64}
    nodes::Vector{Vector{Float64}}
    lw::Vector{Vector{Float64}}
end
"""
prev on other segments (ln p interpolated at their Gauss nodes; ln p is smooth within the segments, which break at all the
knots so far), -Inf beyond its range
"""
function projectDensity(prev::Density, edges, n)
    nodes = segmentNodes(edges, n)
    #beyond the range of prev, ln p linear from its end nodes (the tails are powers of x)
    zl, ll = prev.nodes[1], prev.lp[1]; zr, lr = prev.nodes[end], prev.lp[end]
    sl = (ll[2] - ll[1]) / (zl[2] - zl[1]); sr = (lr[end] - lr[end-1]) / (zr[end] - zr[end-1])
    f(z) = z < zl[1] ? ll[1] + sl * (z - zl[1]) : z > zr[end] ? lr[end] + sr * (z - zr[end]) : logDensity(prev, z)
    Density(edges, n, nodes, [[f(z) for z in nz] for nz in nodes])
end
"the step table of prev on its own segments (the substeps of an interval): no interpolation"
function stepTableSame(prev::Density)
    _, w, _ = gauss(prev.n); e = prev.edges
    StepTable(e, prev.nodes, [log.((e[k+1] - e[k]) / 2 .* w) .+ prev.lp[k] .+ prev.nodes[k] for k in eachindex(prev.nodes)])
end
function stepTable(prev::Density, edges, n)
    x, w, _ = gauss(n)
    nodes = segmentNodes(edges, n)
    lw = [[log((edges[s+1] - edges[s]) / 2 * w[j]) + logDensity(prev, nodes[s][j]) + nodes[s][j] for j in 1:n] for s in eachindex(nodes)]
    StepTable(edges, nodes, lw)
end
"""
ln(W S p) at the points zs after the step s from prev: two cumulative log-sums over the segments and a Gauss rule on the
two parts of the segment of each point (prev interpolated in ln p); `lpart` (optional) the ln p_prev + ζ + ln(weight) of
those parts, precomputed (they do not depend on the step)
"""
function logWSp(s::GreenStep{T}, tab::StepTable, prev::Density, zs; lpart=nothing) where {T}
    n = length(tab.nodes[1])
    ns = length(tab.nodes)
    LA = Vector{T}(undef, ns); LB = Vector{T}(undef, ns)
    buf1 = Vector{T}(undef, n); buf2 = Vector{T}(undef, n)
    for k in 1:ns
        for j in 1:n
            ζ = tab.nodes[k][j]
            buf1[j] = psi(s, ζ, 1) + tab.lw[k][j]; buf2[j] = psi(s, ζ, 2) + tab.lw[k][j]
        end
        LA[k] = logsumexp(buf1); LB[k] = logsumexp(buf2)
    end
    pre = similar(LA); suf = similar(LB)
    acc = T(-Inf)
    for k in 1:ns
        pre[k] = acc; acc = logaddexp(acc, LA[k])
    end
    totA = acc
    acc = T(-Inf)
    for k in ns:-1:1
        suf[k] = acc; acc = logaddexp(acc, LB[k])
    end
    totB = acc
    lpart = something(lpart, partTables(tab, prev, zs))
    out = Vector{T}(undef, length(zs))
    for (i, z) in enumerate(zs)
        ψmz, ψpz = psi(s, z, 1), psi(s, z, 2)
        if z <= tab.edges[1] #all of the previous mass above z
            out[i] = ψmz + totB; continue
        elseif z >= tab.edges[end] #all of it below
            out[i] = ψpz + totA; continue
        end
        k = clamp(searchsortedlast(tab.edges, z), 1, ns)
        (ζl, ll), (ζr, lr) = lpart[i]
        Im = pre[k]; Ip = suf[k]
        if !isempty(ζl)
            for j in 1:n
                buf1[j] = ll[j] + psi(s, ζl[j], 1)
            end
            Im = logaddexp(Im, logsumexp(buf1))
        end
        if !isempty(ζr)
            for j in 1:n
                buf2[j] = lr[j] + psi(s, ζr[j], 2)
            end
            Ip = logaddexp(Ip, logsumexp(buf2))
        end
        out[i] = logaddexp(ψpz + Im, ψmz + Ip)
    end
    out
end
"the Gauss points of the two parts [lo, z] and [z, hi] of the segment of each z, with ln(weight) + ln p_prev + ζ there"
function partTables(tab::StepTable, prev::Density, zs)
    n = length(tab.nodes[1]); x, w, _ = gauss(n); ns = length(tab.nodes)
    map(zs) do z
        (z <= tab.edges[1] || z >= tab.edges[end]) && return ((Float64[], Float64[]), (Float64[], Float64[]))
        k = clamp(searchsortedlast(tab.edges, z), 1, ns)
        lo, hi = tab.edges[k], tab.edges[k+1]
        part(a, b) = b > a ? (h = (b - a) / 2; ζ = a .+ h .* (1 .+ x); (ζ, [log(h * w[j]) + logDensity(prev, ζ[j]) + ζ[j] for j in 1:n])) : (Float64[], Float64[])
        (part(lo, z), part(z, hi))
    end
end
"""
the barycentric interpolation matrices from the n Gauss nodes of a segment to the Gauss points of the parts [lo, z_j] and
[z_j, hi] of the segment at its node j, in reference coordinates (the same for all segments)
"""
const PARTMATS = Dict{Int,Tuple{Vector{Matrix{Float64}},Vector{Matrix{Float64}}}}()
function partMatrices(n)
    get!(PARTMATS, n) do
        x, _, bw = gauss(n)
        interp(t) = (row = zeros(n); k = findfirst(==(t), x); k !== nothing && (row[k] = 1.0; return row);
            a = bw ./ (t .- x); a ./ sum(a))
        L = [reduce(vcat, [interp(-1 + (x[j] + 1) * (1 + x[l]) / 2)' for l in 1:n]) for j in 1:n]
        R = [reduce(vcat, [interp(x[j] + (1 - x[j]) * (1 + x[l]) / 2)' for l in 1:n]) for j in 1:n]
        L, R
    end
end
"""
the density after the step s from prev when prev is on the same segments (the substeps of an interval): the log-integrands
ψ± + ln p_prev + ζ, polynomials on each segment, are interpolated to the parts of the segment at each node by the fixed
matrices of partMatrices, instead of the evaluations of partTables
"""
function nextDensitySame(s::GreenStep, prev::Density)
    n = prev.n; x, w, _ = gauss(n); Lm, Rm = partMatrices(n)
    ns = length(prev.nodes); edges = prev.edges
    ψm = [[psi(s, z, 1) for z in nz] for nz in prev.nodes]; ψp = [[psi(s, z, 2) for z in nz] for nz in prev.nodes]
    Fm = [ψm[k] .+ prev.lp[k] .+ prev.nodes[k] for k in 1:ns]; Fp = [ψp[k] .+ prev.lp[k] .+ prev.nodes[k] for k in 1:ns]
    lh = [log((edges[k+1] - edges[k]) / 2) for k in 1:ns]; lwv = log.(w)
    LA = [logsumexp(Fm[k] .+ lwv) + lh[k] for k in 1:ns]; LB = [logsumexp(Fp[k] .+ lwv) + lh[k] for k in 1:ns]
    pre = vcat(-Inf, logcumsumexp(LA)[1:end-1]); suf = vcat(logcumsumexp(LB; rev=true)[2:end], -Inf)
    lW = log(s.W)
    σall = PD.ahBasisMatrix(s.knots, exp.(reduce(vcat, prev.nodes))) * s.coeffs
    lp = Vector{Vector{Float64}}(undef, ns)
    for k in 1:ns
        z = prev.nodes[k]; σ = view(σall, (k-1)*n+1:k*n)
        lp[k] = map(1:n) do j
            #the part [lo, z_j] of φ₋ and [z_j, hi] of φ₊, of widths (1 + x_j) h and (1 - x_j) h
            pa = logsumexp(Lm[j] * Fm[k] .+ lwv) + lh[k] + log((1 + x[j]) / 2)
            pb = logsumexp(Rm[j] * Fp[k] .+ lwv) + lh[k] + log((1 - x[j]) / 2)
            logaddexp(ψp[k][j] + logaddexp(pre[k], pa), ψm[k][j] + logaddexp(suf[k], pb)) - lW - log(Sof(s, exp(z[j]), σ[j]))
        end
    end
    Density(edges, n, prev.nodes, lp)
end
"the density after the step s from prev, on the segments of tab"
function nextDensity(s::GreenStep, tab::StepTable, prev::Density)
    n = length(tab.nodes[1])
    zs = reduce(vcat, tab.nodes)
    l = logWSp(s, tab, prev, zs)
    σ = PD.ahBasisMatrix(s.knots, exp.(zs)) * s.coeffs
    lpAll = [l[j] - log(s.W * Sof(s, exp(zs[j]), σ[j])) for j in eachindex(zs)]
    Density(tab.edges, n, tab.nodes, [lpAll[(k-1)*n+1:k*n] for k in eachindex(tab.nodes)])
end
#--- prices from a density: exact integrals of the positive density (out of the money: calls above the forward)
"out-of-the-money price at the strike K/F = exp(zk)"
function otmPrice(d::Density, zk)
    x, w, _ = gauss(d.n)
    K = exp(zk); call = zk >= 0
    total = 0.0
    for s in eachindex(d.nodes)
        a, b = d.edges[s], d.edges[s+1]
        if call
            b <= zk && continue
            a = max(a, zk)
        else
            a >= zk && continue
            b = min(b, zk)
        end
        h = (b - a) / 2
        for j in 1:d.n
            ζ = a + h * (1 + x[j]); X = exp(ζ)
            total += h * w[j] * exp(logDensity(d, ζ) + ζ) * (call ? X - K : K - X)
        end
    end
    total
end
"mass and mean of a density"
function moments(d::Density)
    x, w, _ = gauss(d.n); m0 = 0.0; m1 = 0.0
    for s in eachindex(d.nodes), j in 1:d.n
        h = (d.edges[s+1] - d.edges[s]) / 2; ζ = d.nodes[s][j]; f = h * w[j] * exp(d.lp[s][j] + ζ)
        m0 += f; m1 += f * exp(ζ)
    end
    m0, m1
end
"implied volatility from an out-of-the-money price at log-moneyness y"
otmVol(price, y, T) = impliedVolatilityThiophene(y >= 0, max(price, 1e-300), 1.0, exp(y), T, 1.0)

const GREEN_STATS = Ref{Any}(nothing) #(expiry, iterations, f calls, Jacobian calls) of the calibrations, when a vector

#--- the surface
struct GreenAHSurface
    expiries::Vector{Float64}
    steps::Vector{GreenStep{Float64}}  # one per expiry interval, of length Δᵢ/substeps
    densities::Vector{Density}         # at the expiries
    knots::Vector{Vector{Float64}}
    coeffs::Vector{Vector{Float64}}
    zrange::Tuple{Float64,Float64}
    n::Int
    rates::Vector{Any}   # the rate bounds of the segments of each step and its elements per knot interval
    substeps::Int        # implicit steps per expiry interval, all with the Green's function of the step
    inner::Vector{Vector{Density}}     # the densities after the substeps 1, ..., substeps - 1 of each interval
    cache::Dict{Float64,Density}       # the densities between the expiries, computed on demand
end
Base.broadcastable(s::GreenAHSurface) = Ref(s)
GreenAHSurface(expiries, steps, densities, knots, coeffs, zrange, n, rates, substeps, inner) =
    GreenAHSurface(expiries, steps, densities, knots, coeffs, zrange, n, rates, substeps, inner, Dict{Float64,Density}())
GreenAHSurface(expiries, steps, densities, knots, coeffs, zrange, n, rates) =
    GreenAHSurface(expiries, steps, densities, knots, coeffs, zrange, n, rates, 1, [Density[] for _ in expiries])

"""
the z range of a step: the density of y decays as x^(r₊-1) on the left and x p as x^r₋ on the right, r(r-1) = q, beyond
the outer knots; the range extends until the mass and the mean beyond are negligible (e⁻³⁰), at most 100 further
"""
function tailRange(st::GreenStep, zminI, zmaxI)
    rL = roots(ForwardDiff.value(st.qL))[1]; rR = roots(ForwardDiff.value(st.qR))[2]
    min(zminI, max(log(st.knots[1]) - 30 / (rL - 1), zminI - 100)), max(zmaxI, min(log(st.knots[end]) + 30 / -rR, zmaxI + 100))
end
"the nodes of the segments, flattened, with ln(Gauss weight) + ζ (the measure dξ = e^ζ dζ)"
function nodeWeights(edges, n)
    x, w, _ = gauss(n)
    zs = reduce(vcat, segmentNodes(edges, n))
    lw = reduce(vcat, [[log((edges[k+1] - edges[k]) / 2 * w[j]) for j in 1:n] for k in 1:length(edges)-1]) .+ zs
    zs, lw
end
flatLp(d::Density) = reduce(vcat, d.lp)

"""
sensitivity of the call prices at the log-strikes yq to the coefficients of the step: differentiating c - S c'' = c_prev,
δc - S δc'' = δS p, so that ∂c(K)/∂θₘ = ∫ G(K, ξ) 2 Bₘ(ξ)/σ(ξ) p(ξ) dξ, a positive integral. With several substeps, the sum of
these over the densities pₛ of the substeps (the propagation of the earlier ones by the later steps neglected, O(Δ)). G
separates, G(K, ξ) = φ₋(min)φ₊(max)/W: the integrals are prefix sums over the nodes zs (sorted), per B-spline over its
support only, in logarithms. Bσ the matrix 2 Bₘ(ξ)/σ(ξ) at the nodes, lp the ln Σₛ pₛ there, lw the ln(weight) + ζ.
"""
function priceJacobian(st::GreenStep, zs, lw, lp, Bσ, yq; ψm=[psi(st, z, 1) for z in zs], ψp=[psi(st, z, 2) for z in zs])
    quoteKernel(st, zs, lw, lp .+ log.(Bσ), yq; ψm=ψm, ψp=ψp)
end
"""
∫ G(K, ξ) h_m(ξ) dξ at the log-strikes yq for the columns of LH = ln h_m at the nodes zs (sorted), by prefix sums over the
nodes where h_m > 0
"""
function quoteKernel(st::GreenStep, zs, lw, LH, yq; ψm=[psi(st, z, 1) for z in zs], ψp=[psi(st, z, 2) for z in zs])
    lW = log(st.W)
    ψmy = [psi(st, y, 1) for y in yq]; ψpy = [psi(st, y, 2) for y in yq]
    pos = [searchsortedlast(zs, y) for y in yq] #nodes ≤ y
    J = zeros(length(yq), Base.size(LH, 2))
    for m in axes(LH, 2)
        idx = findall(>(-Inf), view(LH, :, m))
        isempty(idx) && continue
        a = [ψm[j] + lw[j] + LH[j, m] for j in idx]
        b = [ψp[j] + lw[j] + LH[j, m] for j in idx]
        pre = logcumsumexp(a); suf = logcumsumexp(b; rev=true)
        for q in eachindex(yq)
            r = searchsortedlast(idx, pos[q]) #support nodes ≤ y
            lA = r >= 1 ? pre[r] : -Inf; lB = r < length(idx) ? suf[r+1] : -Inf
            J[q, m] = exp(logaddexp(ψpy[q] + lA, ψmy[q] + lB) - lW)
        end
    end
    J
end
"""
the resolvent of the step at the nodes, R[h](x_j) = ∫ G(x_j, ξ) h(ξ) dξ, for the columns of LH = ln h (cumulative Gauss sums
of the nodes before each one, half its own weight), in logarithms: for the Jacobian of the substeps only
"""
function resolventNodes(st::GreenStep, lw, LH, ψm, ψp)
    M = length(lw); lW = log(st.W); out = similar(LH)
    for m in axes(LH, 2)
        a = ψm .+ lw .+ view(LH, :, m); b = ψp .+ lw .+ view(LH, :, m)
        ca = logcumsumexp(a); cb = logcumsumexp(b; rev=true)
        for j in 1:M
            A = logaddexp(j > 1 ? ca[j-1] : -Inf, a[j] - log(2)); B = logaddexp(j < M ? cb[j+1] : -Inf, b[j] - log(2))
            out[j, m] = logaddexp(ψp[j] + A, ψm[j] + B) - lW
        end
    end
    out
end
"""
ln p after the step s from prev at the nodes of tab, for the Jacobian only: the integrals to each node are the cumulative Gauss
sums of the nodes before it (half its own weight), without the partial segments of nextDensity
"""
function approxNextLp(st::GreenStep, tab::StepTable, zs, σ; ψm=[psi(st, z, 1) for z in zs], ψp=[psi(st, z, 2) for z in zs])
    lw = reduce(vcat, tab.lw)
    a = ψm .+ lw; b = ψp .+ lw
    M = length(zs)
    #the nodes before each one, and half its own weight
    ca = logcumsumexp(a); cb = logcumsumexp(b; rev=true)
    A = [logaddexp(j > 1 ? ca[j-1] : -Inf, a[j] - log(2)) for j in 1:M]
    Bv = [logaddexp(j < M ? cb[j+1] : -Inf, b[j] - log(2)) for j in 1:M]
    [logaddexp(ψp[j] + A[j], ψm[j] + Bv[j]) - log(st.W * Sof(st, exp(zs[j]), σ[j])) for j in 1:M]
end
"Black vega of the out-of-the-money option at log-moneyness y (forward one), implied volatility v"
function blackVega(y, v, T)
    sw = v * sqrt(T)
    d1 = -y / sw + sw / 2
    exp(-d1^2 / 2) / sqrt(2π) * sqrt(T)
end

"""
calibrate to implied volatilities (vols[i] at log-moneyness ys[i], expiries ttes); the defaults of the resolution (N = n = 12,
τ = 16) are within 2e-4 bp of a much finer one on SPX. Settings of SOLV (12 uniform knots over the
quotes extended by 10%, tied outer coefficients, C3 condition at the forward, minVol 0.01, wing floor 0.5 :min); N the
Chebyshev degree of the elements of φ±, sub their number per knot interval, n the Gauss points per segment, τ the
resolution of the segments. jacobian = :analytic (the sensitivity kernel of priceJacobian) or :ad (forward automatic
differentiation through the step, one substep only). substeps: implicit steps of Δᵢ/substeps per expiry interval with the
same local volatility; with two or more, the C3 condition is not used (the second substep removes the peak at the forward,
and the kink of the C3 condition would put a kink in the density of the later substeps).
"""
function calibrateGreenVols(ttes, ys, vols; size=12, knotExtension=0.1, minVol=0.01, maxVol=5.0, wingFloor=0.5, c3=true,
    nRefine=1, N=12, sub=1, n=12, τ=16.0, hmax=0.1, L=nothing, U=nothing, location="Uniform", tailLeft=10.0, tailRight=6.0,
    massTol=1e-9, maxRefine=2, jacobian=:analytic, substeps=1, weights=nothing)
    jacobian == :ad && substeps > 1 && throw(ArgumentError("jacobian=:ad needs substeps=1"))
    c3 = c3 && substeps == 1
    K = substeps
    wq = weights === nothing ? [ones(length(y)) for y in ys] : weights
    nT = length(ttes)
    ks = [exp.(y) for y in ys]
    function atmVol(i)
        j = clamp(searchsortedfirst(ys[i], 0.0), 2, length(ys[i]))
        v1, v2 = vols[i][j-1], vols[i][j]
        v1 + (v2 - v1) * (1 - ks[i][j-1]) / (ks[i][j] - ks[i][j-1])
    end
    Lb, Ub = gridBounds(ttes, ys, vols; L=L, U=U)
    #no boundary: the segments extend far beyond the range of the grid of SOLV, where the density is analytic (all the local
    #volatilities flat), so that the mass beyond them is negligible
    zmin, zmax = log(Lb) - tailLeft, log(Ub) + tailRight
    stepKnots(k) = location == "Mid-XX" ? sort(unique(vcat(PD.makeKnots(0.0, 1.0, k, 0; location="Mid-XX", L=k[1] / 2, U=2k[end])[2:end-1], 1.0))) :
                   (e = knotExtension * log(k[end] / k[1]); exp.(range(log(k[1]) - e, stop=log(k[end]) + e, length=size)))
    knotsList = [collect(Float64, stepKnots(ks[i])) for i = 1:nT]
    cAtm1 = blackScholesFormula(true, 1.0, 1.0, atmVol(1)^2 * ttes[1], 1.0, 1.0)
    c3 && (knotsList[1] = PD.ahFirstKnots(knotsList[1], cAtm1))
    kc, Dc3 = c3 ? PD.ahC3Data(knotsList[1]) : (0, Float64[])
    steps = GreenStep{Float64}[]; dens = Density[]; inner = Vector{Density}[]; coeffsL = Vector{Float64}[]; rates = Any[]
    breaks = [0.0]
    prevRates = StepRate[]
    for i = 1:nT
        dt = ttes[i] - (i == 1 ? 0.0 : ttes[i-1]); dtk = dt / K
        kn = knotsList[i]
        B = PD.ahBasisMatrix(kn, ks[i]); nb = Base.size(B, 2)
        tied = true
        free = [k for k = 1:nb if !(i == 1 && c3 && k == kc) && !(tied && (k == 1 || k == nb))]
        minVols = fill(minVol, nb)
        σ0 = i == 1 ? atmVol(1) : (vPrev = otmVol(otmPrice(dens[i-1], 0.0), 0.0, ttes[i-1]);
            sqrt(max((atmVol(i)^2 * ttes[i] - vPrev^2 * ttes[i-1]) / dt, (2minVol)^2)))
        if wingFloor > 0
            wPrev = i == 1 ? 0.0 : otmVol(otmPrice(dens[i-1], 0.0), 0.0, ttes[i-1])^2 * ttes[i-1]
            wCur = max(atmVol(i)^2 * ttes[i], wPrev + (2minVol)^2 * dt)
            for (kk, xk) in (((1, 2), kn[1]), ((nb - 1, nb), kn[end]))
                f = min(wingFloor * σ0, PD.ahFlatStepVol(xk, wPrev, wCur, dt))
                for k in kk
                    minVols[k] = max(minVols[k], f)
                end
            end
        end
        transforms = [PD.makeTransform(minVols[k], maxVol, "LM-ALG") for k = 1:nb]
        guess = [min(max(σ0, 2minVols[k]), maxVol / 2) for k = 1:nb]
        p0 = [AQFED.Math.inv(transforms[k], guess[k]) for k in free]
        #segments of the step: the knots so far, resolved for the rate of the previous density and of this step at its floors
        breaks = sort(unique(vcat(breaks, log.(kn))))
        local rate, edges, st, d, coef, subL, sd, prevOnEdges
        #the mass of the exact step is one: a mass defect above massTol refines the step (twice the elements, half the
        #widths of the segments) and calibrates it again from the coarser solution
        for level = 0:maxRefine
        subL = sub * 2^level; τL, hmaxL = τ / 2^level, hmax / 2^level
        rate = rateOf(vcat(prevRates, [stepRate(kn, minVols, dtk)])) #the steps so far and this one at its floors
        #the range of the previous density
        zminI, zmaxI = i == 1 ? (zmin, zmax) : (min(zmin, dens[i-1].edges[1]), max(zmax, dens[i-1].edges[end]))
        edges = makeSegments(breaks, zminI, zmaxI, rate; τ=τL, hmax=hmaxL)
        tab = i == 1 ? nothing : stepTable(dens[i-1], edges, n)
        prevOTM = i == 1 ? zeros(length(ys[i])) : [otmPrice(dens[i-1], y) for y in ys[i]]
        lpart = i == 1 ? nothing : partTables(tab, dens[i-1], ys[i])
        prevOnEdges = i == 1 ? nothing : projectDensity(dens[i-1], edges, n)
        #the basis matrices at the element nodes (the elements depend on the knots only)
        Bel = [PD.ahBasisMatrix(kn, exp.(z)) for z in greenStep(kn, guess, dtk; N=N, sub=subL).nodes]
        function coefficients(p)
            coef = zeros(eltype(p), nb)
            for (r, k) in enumerate(free)
                coef[k] = transforms[k](p[r])
            end
            coef[1] = coef[2]; coef[nb] = coef[nb-1]
            coef
        end
        function build(coef)
            if i == 1 && c3
                cA = cAtm1; st = nothing
                for _ = 0:nRefine
                    PD.ahApplyC3!(coef, kc, Dc3, cA)
                    st = greenStep(kn, coef, dtk; N=N, sub=subL, Bel=Bel)
                    cA = 1 / st.W #the at-the-money price after the first step: G(1, 1)
                end
                return st
            end
            greenStep(kn, coef, dtk; N=N, sub=subL, Bel=Bel)
        end
        #the substeps from the previous expiry: the out-of-the-money prices at the quotes are those of the previous expiry plus
        #the increments uₛ = Sₛpₛ of the substeps; the densities after the substeps 1, ..., K - 1 (all K with `last`)
        function substepsOf(st::GreenStep{T}, edges; last=false) where {T}
            otm = T.(prevOTM); ds = Density[]
            prevProj = i == 1 ? nothing : (tab !== nothing && edges === tab.edges ? prevOnEdges : projectDensity(dens[i-1], edges, n))
            prev = i == 1 ? nothing : dens[i-1]
            for s = 1:K
                if prev === nothing #from the Dirac at the forward: u = G(K, 1)
                    otm .+= [exp(psi(st, y, y <= 0 ? 1 : 2)) / st.W for y in ys[i]]
                    (s < K || last) && (prev = firstDensity(st, edges, n); push!(ds, prev))
                else
                    own = s == 1 && tab !== nothing && edges === tab.edges #the tables of the previous expiry, precomputed
                    same = s > 1 #a substep: prev on the segments of the step
                    tb = own ? tab : same ? stepTableSame(prev) : stepTable(prev, edges, n)
                    lpt = own ? lpart : partTables(tb, prev, ys[i])
                    otm .+= exp.(logWSp(st, tb, prev, ys[i]; lpart=lpt)) ./ st.W
                    (s < K || last) && (prev = nextDensitySame(st, same ? prev : prevProj); push!(ds, prev))
                end
            end
            otm, ds
        end
        function model(coef)
            st = build(coef)
            otm, ds = substepsOf(st, edges)
            otm, st, ds
        end
        fvec = zeros(length(ys[i]))
        if jacobian == :ad
            function objAD!(fvec, p)
                otm, _ = model(coefficients(p))
                for j in eachindex(otm)
                    fvec[j] = wq[i][j] * (otmVol(otm[j], ys[i][j], ttes[i]) - vols[i][j])
                end
                fvec
            end
            od = LsqFit.OnceDifferentiable(objAD!, p0, copy(fvec); autodiff=LsqFit.AutoForwardDiff(), inplace=true)
        else
            zsN, lwN = nodeWeights(edges, n)
            Bσbase = PD.ahBasisMatrix(kn, exp.(zsN))
            cache = Ref{Any}(nothing) #the last evaluation: p, step, implied volatilities, densities of the substeps
            function evaluate(p)
                c = cache[]
                c !== nothing && c[1] == p && return c
                coef = coefficients(p)
                otm, st, ds = model(coef)
                iv = [otmVol(otm[j], ys[i][j], ttes[i]) for j in eachindex(otm)]
                cache[] = (copy(p), st, iv, ds)
            end
            function obj!(fvec, p)
                _, _, iv, _ = evaluate(p)
                fvec .= wq[i] .* (iv .- vols[i])
            end
            function jac!(J, p)
                _, st, iv, ds = evaluate(p)
                #the density of the last substep, then ln Σₛ pₛ at the nodes
                σn = Bσbase * st.coeffs
                ψm = [psi(st, z, 1) for z in zsN]; ψp = [psi(st, z, 2) for z in zsN]
                lastLp = i == 1 && K == 1 ? flatLp(firstDensity(st, edges, n)) :
                         approxNextLp(st, stepTableSame(K == 1 ? prevOnEdges : ds[end]), zsN, σn; ψm=ψm, ψp=ψp)
                lps = vcat([flatLp(x) for x in ds], [lastLp])
                lB = log.(Bσbase .* (2 ./ σn))
                #δcₛ - S δcₛ'' = δS pₛ + δcₛ₋₁: δcₛ = R[2Bₘ/σ pₛ + δcₛ₋₁/S] at the nodes, then at the quotes (the put and the
                #call have the same)
                LD = fill(-Inf, length(zsN), nb); lS = [log(Sof(st, exp(zsN[j]), σn[j])) for j in eachindex(zsN)]
                for s = 1:K-1
                    LD = resolventNodes(st, lwN, logaddexp.(lB .+ lps[s], LD .- lS), ψm, ψp)
                end
                Jc = quoteKernel(st, zsN, lwN, logaddexp.(lB .+ lps[K], LD .- lS), ys[i]; ψm=ψm, ψp=ψp)
                Bσ = exp.(lB); lp = lps[K]
                #the coefficients from the free parameters: transforms, tied wings, the C3 coefficient
                Dp = ForwardDiff.jacobian(coefficients, p)
                if i == 1 && c3
                    cA = 1 / st.W; den = Dc3[kc] + 1 / (2cA)
                    sC = sum(st.coeffs[k] * Dc3[k] for k in eachindex(Dc3) if k != kc)
                    a = [k == kc ? 0.0 : -Dc3[k] / den for k in 1:nb] #∂θ_kc/∂θ_k at a fixed at-the-money price
                    b = -sC / (2cA^2 * den^2)                           #∂θ_kc/∂cA
                    Jatm = vec(priceJacobian(st, zsN, lwN, lp, Bσ, [0.0]; ψm=ψm, ψp=ψp)) #∂c(1)/∂θ
                    g = (a .+ b .* [k == kc ? 0.0 : Jatm[k] for k in 1:nb]) ./ (1 - b * Jatm[kc])
                    Dp[kc, :] .= vec(g' * Dp)
                end
                J .= (Jc * Dp) .* (wq[i] ./ [blackVega(ys[i][j], iv[j], ttes[i]) for j in eachindex(iv)])
            end
            od = LsqFit.OnceDifferentiable(obj!, jac!, p0, copy(fvec), zeros(length(fvec), length(p0)); inplace=true)
        end
        res = LsqFit.levenberg_marquardt(od, p0)
        p = res.minimizer
        GREEN_STATS[] !== nothing && push!(GREEN_STATS[], (i, res.iterations, od.f_calls[1], od.df_calls[1]))
        coef = coefficients(p)
        st = build(coef)
        coef = st.coeffs
        zl, zr = tailRange(st, zminI, zmaxI)
        if zl < zminI || zr > zmaxI
            edges = makeSegments(breaks, zl, zr, rate; τ=τL, hmax=hmaxL)
            tab = i == 1 ? nothing : stepTable(dens[i-1], edges, n)
            lpart = i == 1 ? nothing : partTables(tab, dens[i-1], ys[i])
            prevOnEdges = i == 1 ? nothing : projectDensity(dens[i-1], edges, n)
        end
        _, sd = substepsOf(st, edges; last=true)
        d = sd[end]
        m = moments(d)
        (level == maxRefine || max(abs(m[1] - 1), abs(m[2] - 1)) <= massTol) && break
        p0 = p
        end
        push!(steps, st); push!(dens, d); push!(inner, sd[1:end-1]); push!(coeffsL, collect(Float64, coef)); push!(rates, (rate, subL))
        push!(prevRates, stepRate(kn, coef, st.dt))
    end
    GreenAHSurface(collect(Float64, ttes), steps, dens, knotsList, coeffsL, (zmin, zmax), n, rates, K, inner)
end
"the bounds of the grid of SOLV: six at-the-money standard deviations of the longest expiry, at least [1/6, 6]"
function gridBounds(ttes, ys, vols; L=nothing, U=nothing)
    atm(i) = (j = argmin(abs.(ys[i])); vols[i][j])
    sdMax = maximum(atm(i) * sqrt(ttes[i]) for i in eachindex(ttes))
    ks = [exp.(y) for y in ys]
    something(L, min(1 / 6, exp(-6sdMax), minimum(minimum.(ks)) / 2)), something(U, max(6.0, exp(6sdMax), 2maximum(maximum.(ks))))
end
"the z range of the densities: the grid bounds extended by tailLeft and tailRight"
zRange(ttes, ys, vols; tailLeft=10.0, tailRight=6.0) = (b = gridBounds(ttes, ys, vols); (log(b[1]) - tailLeft, log(b[2]) + tailRight))
"model implied volatility of an expiry at log-moneyness y"
modelVolGreen(s::GreenAHSurface, i, y) = otmVol(otmPrice(s.densities[i], y), y, s.expiries[i])
"density of y at an expiry (p(x) x)"
densityY(s::GreenAHSurface, i, y) = exp(logDensity(s.densities[i], y) + y)

"""
the surface of the knots and coefficients of s recomputed at another resolution (no calibration): the segments resolve the
rates of the actual local volatilities, over the ranges of tailRange, with the substeps of s
"""
function rebuildGreen(s::GreenAHSurface; N=12, sub=1, n=12, τ=16.0, hmax=0.1, hfar=0.5)
    zmin, zmax = s.zrange; K = s.substeps
    steps = GreenStep{Float64}[]; dens = Density[]; inner = Vector{Density}[]
    breaks = [0.0]; prevRates = StepRate[]
    for i in eachindex(s.expiries)
        dt = s.expiries[i] - (i == 1 ? 0.0 : s.expiries[i-1])
        st = greenStep(s.knots[i], s.coeffs[i], dt / K; N=N, sub=sub)
        breaks = sort(unique(vcat(breaks, log.(s.knots[i]))))
        push!(prevRates, stepRate(st.knots, st.coeffs, st.dt)); rate = rateOf(copy(prevRates))
        zl, zr = tailRange(st, i == 1 ? zmin : min(zmin, dens[i-1].edges[1]), i == 1 ? zmax : max(zmax, dens[i-1].edges[end]))
        edges = makeSegments(breaks, zl, zr, rate; τ=τ, hmax=hmax, hfar=hfar)
        ds = Density[]; d = i == 1 ? nothing : dens[i-1]
        for k = 1:K
            d = d === nothing ? firstDensity(st, edges, n) : nextDensitySame(st, k > 1 ? d : projectDensity(d, edges, n))
            push!(ds, d)
        end
        push!(steps, st); push!(dens, d); push!(inner, ds[1:end-1])
    end
    GreenAHSurface(s.expiries, steps, dens, s.knots, s.coeffs, s.zrange, n, Any[], K, inner)
end

#--- measures (as solv_common.jl)
"""
the density at time t (0 < t ≤ last expiry): the expiry itself, or a partial step of the local volatility of the next
expiry from the last substep before t (the one-step model between its substeps), positive by construction
"""
function densityAtTime(s::GreenAHSurface, t; N=12, sub=nothing, τ=16.0, hmax=0.1)
    i = min(searchsortedfirst(s.expiries, t - 1e-12), length(s.expiries)) #beyond the last expiry: its local volatility
    abs(t - s.expiries[i]) < 1e-12 && return s.densities[i]
    sub === nothing && haskey(s.cache, t) && return s.cache[t]
    d = partialDensity(s, i, t; N=N, sub=sub)
    sub === nothing && (s.cache[t] = d)
    d
end
function partialDensity(s::GreenAHSurface, i, t; N=12, sub=nothing)
    t0 = i == 1 ? 0.0 : s.expiries[i-1]
    sub = something(sub, isempty(s.rates) ? 1 : s.rates[i][2]) #the elements of the calibrated step
    dtk = (s.expiries[i] - t0) / s.substeps
    k = min(floor(Int, (t - t0) / dtk + 1e-12), s.substeps - 1) #the substeps completed before t
    start = k == 0 ? (i == 1 ? nothing : s.densities[i-1]) : s.inner[i][k]
    r = t - t0 - k * dtk
    r < 1e-12 && return start
    st = greenStep(s.knots[i], s.coeffs[i], r; N=N, sub=sub)
    edges = s.densities[i].edges
    start === nothing ? firstDensity(st, edges, s.n) : nextDensitySame(st, k > 0 ? start : projectDensity(start, edges, s.n))
end
"""
the Dupire local volatility of the surface just before the expiry Tᵢ, exactly: within a substep of length Δ from c₋, c(t) =
(I - (t - t₀)L)⁻¹c₋, so that ∂ₜc(Tᵢ⁻) = R[L cᵢ] = u_next/Δ, with u_next the increment of one more substep from pᵢ, and
σ_D² = 2∂ₜc/(x²pᵢ) = σᵢ² u_next/uᵢ, uᵢ = Sᵢpᵢ. At Tᵢ₋₁⁺, σ_D = σᵢ.
"""
function dupireAtExpiry(s::GreenAHSurface, i, yq)
    st = s.steps[i]; d = s.densities[i]
    lu = logWSp(st, stepTable(d, d.edges, s.n), d, yq) .- log(st.W)
    σ = PD.ahBasisMatrix(st.knots, exp.(yq)) * st.coeffs
    [σ[j] * sqrt(exp(lu[j] - logDensity(d, yq[j]) - log(Sof(st, exp(yq[j]), σ[j])))) for j in eachindex(yq)]
end
"""
the Dupire local volatility of the surface at the time t (T₀ < t ≤ Tₙ), exactly: at the partial step of length r from c₋
(the last substep before t), c(t) = (I - rL)⁻¹c₋ and ∂ₜc = (I - rL)⁻¹Lc(t) = u⁺/r, with u⁺ the increment of one more step of
length r from c(t), so that σ_D² = σᵢ² u⁺/(S_r p(t)), S_r = ½rσᵢ²x². At the start of a substep, σ_D = σᵢ.
"""
function dupireAtTime(s::GreenAHSurface, t, yq)
    i = min(searchsortedfirst(s.expiries, t - 1e-12), length(s.expiries))
    abs(t - s.expiries[i]) < 1e-12 && return dupireAtExpiry(s, i, yq)
    t0 = i == 1 ? 0.0 : s.expiries[i-1]
    dtk = (s.expiries[i] - t0) / s.substeps
    r = t - t0 - min(floor(Int, (t - t0) / dtk + 1e-12), s.substeps - 1) * dtk
    σ = PD.ahBasisMatrix(s.knots[i], exp.(yq)) * s.coeffs[i]
    r < 1e-12 && return abs.(σ)
    d = densityAtTime(s, t)
    st = greenStep(s.knots[i], s.coeffs[i], r; N=12, sub=isempty(s.rates) ? 1 : s.rates[i][2])
    lu = logWSp(st, stepTable(d, d.edges, s.n), d, yq) .- log(st.W)
    [σ[j] * sqrt(exp(lu[j] - logDensity(d, yq[j]) - log(Sof(st, exp(yq[j]), σ[j])))) for j in eachindex(yq)]
end
#--- the public interface, as for BSplineAHSurface (normalized prices c = C/(P F) at the forward moneyness x = K/F)
"""
    calibrateGreenAHSurface(ttes, forwards, strikes, callPrices, weights; size=12, knotExtension=0.1, minVol=0.01,
        maxVol=5.0, wingFloor=0.5, c3=true, location="Uniform", substeps=1, N=12, n=12, kw...)

Calibrate the one-step local volatility surface solved by its Green's function to the (undiscounted) call prices
`callPrices[i]` at `strikes[i]` of the expiries `ttes`, with the forwards `forwards`, minimizing the weighted errors of the
implied volatilities by Levenberg-Marquardt, expiry by expiry. The local volatility of each expiry interval is a quadratic
B-spline with `size` knots uniform in ln(K/F) over the quotes extended by `knotExtension` of their range on each side (or the
Mid-XX knots with `location="Mid-XX"`), tied outer coefficients and flat beyond them, between `minVol` and `maxVol`, with
the floor `wingFloor` on the wings (capped at the flat-smile local volatility, as `wingFloorKind=:min` of
calibrateBSplineAHSurface) and the C3 condition at the forward on the first step (`c3`). `substeps` implicit steps per
interval (two or more: no C3 condition). N and n are the Chebyshev and Gauss orders (within 2e-4 basis point of a much
finer resolution at the defaults); the Jacobian is analytic (`jacobian=:ad` for automatic differentiation).
"""
function calibrateGreenAHSurface(ttes::AbstractVector, forwards::AbstractVector, strikes::AbstractVector{<:AbstractVector},
    callPrices::AbstractVector{<:AbstractVector}, weights::AbstractVector{<:AbstractVector}; kw...)
    ys = [log.(strikes[i] ./ forwards[i]) for i in eachindex(ttes)]
    vols = map(eachindex(ttes)) do i
        map(eachindex(ys[i])) do j
            x = exp(ys[i][j]); c = callPrices[i][j] / forwards[i]
            #the out-of-the-money option (the time value of a deep in-the-money call is lost in its intrinsic value)
            x >= 1 ? impliedVolatilityThiophene(true, max(c, 1e-300), 1.0, x, ttes[i], 1.0) :
            impliedVolatilityThiophene(false, max(c - (1 - x), 1e-300), 1.0, x, ttes[i], 1.0)
        end
    end
    calibrateGreenVols(collect(Float64, ttes), ys, vols; weights=weights, kw...)
end
"normalized call price E[(X - x)⁺] at the forward moneyness x and time t (beyond the last expiry: its local volatility)"
function priceNormalized(s::GreenAHSurface, x, t)
    y = log(x); c = otmPrice(densityAtTime(s, t), y)
    y >= 0 ? c : c + (1 - x)
end
"normalized put price E[(x - X)⁺] (out of the money below the forward: full relative precision)"
function putNormalized(s::GreenAHSurface, x, t)
    y = log(x); c = otmPrice(densityAtTime(s, t), y)
    y < 0 ? c : c - (1 - x)
end
"Black variance at the log forward moneyness y and time t"
function varianceByLogmoneyness(s::GreenAHSurface, y, t)
    otmVol(otmPrice(densityAtTime(s, t), y), y, t)^2
end
"density of x = K/F at the expiry Tᵢ"
densityNormalized(s::GreenAHSurface, x, i::Int) = exp(logDensity(s.densities[i], log(x)))
"the local volatility σᵢ(x) of the expiry interval containing t (the last one beyond)"
function localVolatility(s::GreenAHSurface, x, t)
    i = min(searchsortedfirst(s.expiries, t), length(s.expiries))
    abs(σAt(s.knots[i], s.coeffs[i], x))
end
"""
    greenDupireAtExpiry(s, i, ys)

The Dupire local volatility of the surface just before the expiry Tᵢ at the log forward moneyness ys, exactly (at Tᵢ₋₁⁺ it
is σᵢ): it tends to σᵢ with the number of substeps.
"""
greenDupireAtExpiry(s::GreenAHSurface, i, ys) = dupireAtExpiry(s, i, collect(ys))
"""
    greenDupireLocalVolatility(s, t, ys)

The Dupire local volatility 2∂ₜc/(x²∂ₓₓc) of the surface at the time t and the log forward moneyness ys, exactly, between the
expiries as at them (at Tᵢ, the limit from the left; just after Tᵢ₋₁, σᵢ).
"""
greenDupireLocalVolatility(s::GreenAHSurface, t, ys) = dupireAtTime(s, t, collect(ys))
end
