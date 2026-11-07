#Two-dimensional multiquadric RBF fitted by plain least squares, without any no-arbitrage condition: a smooth prior for the
#implied variance surface (for example before an arbitrage-free projection such as the convex spline of Fengler). Same
#representation as Multiquadric2DSurface: v(x, τ) = β₀ + β₁x + β₂τ + β₃xτ + Σ αₗ √(1 + (x-xₗ)²/cₓₗ² + (τ-τₗ)²/c_τₗ²), with
#τ = √T and x = k/(σ̄ τ). Nothing keeps the variance positive beyond the quotes: the extrapolation is left to the user.
export fitMultiquadricSurface2D

"""
    fitMultiquadricSurface2D(ttes, forwards, logmoneyness, vols; weights=nothing, m=8, rangeBandwidth=0.2, minBandwidth=2.0,
        shapeτ=4.0, shapeτShort=2.0, shortTimes=(0.25, 1.0), ridge=1e-10)

Least-squares fit of the implied variance of the quotes `vols[j]` at log-moneyness `logmoneyness[j]` for the expiries
`ttes[j]`, with `m` centers per expiry (equidistant in x over its quotes) and the bandwidths of
`calibrateMultiquadricSurface2D`. The residuals are in variance, weighted by `weights[j][i]` (default: 1/(2σ), so that they
are approximately volatility residuals), and averaged per expiry. With `bandwidth=:spacing`, the bandwidth in x is `shape` times
the spacing of the centers; `roughness` μ adds the penalty μ Σⱼ ∫ (∂³v/∂x³ / σ̄²)² dx over the quoted range of each expiry. The
normal equations, with a ridge of `ridge` times
their mean diagonal, are solved by a Cholesky factorization.
"""
function fitMultiquadricSurface2D(ttes::AbstractVector{T}, forwards::AbstractVector{T}, logmoneyness::AbstractVector{<:AbstractVector{T}},
    vols::AbstractVector{<:AbstractVector{T}}; weights=nothing, m::Int=8, rangeBandwidth=0.2, minBandwidth=2.0, shapeτ=4.0,
    shapeτShort=2.0, shortTimes=(0.25, 1.0), ridge=1e-10, bandwidth::Symbol=:range, shape=1.0, roughness=0.0, nQuadrature::Int=200) where {T}
    nExp = length(ttes)
    #reference volatility: mean over the expiries of the volatility interpolated linearly at the money
    function atm(ys, vs)
        i = searchsortedlast(ys, zero(T))
        i <= 0 && return vs[1]
        i >= length(ys) && return vs[end]
        θ = -ys[i] / (ys[i+1] - ys[i])
        return (1 - θ) * vs[i] + θ * vs[i+1]
    end
    σbar = sum(atm(logmoneyness[j], vols[j]) for j = 1:nExp) / nExp
    τs = sqrt.(ttes)
    function shapeAt(t)
        shapeτShort === nothing && return shapeτ
        t <= shortTimes[1] && return shapeτShort
        t >= shortTimes[2] && return shapeτ
        θ = log(t / shortTimes[1]) / log(shortTimes[2] / shortTimes[1])
        return (1 - θ) * shapeτShort + θ * shapeτ
    end
    xc, τc, cx, cτ = T[], T[], T[], T[]
    for j = 1:nExp
        xL, xR = minimum(logmoneyness[j]) / (σbar * τs[j]), maximum(logmoneyness[j]) / (σbar * τs[j])
        spacingτ = nExp == 1 ? one(T) : (j == 1 ? τs[2] - τs[1] : j == nExp ? τs[nExp] - τs[nExp-1] : (τs[j+1] - τs[j-1]) / 2)
        for xl in range(xL, xR, length=m)
            push!(xc, xl); push!(τc, τs[j])
            push!(cx, bandwidth == :spacing ? shape * (xR - xL) / (m - 1) : max(minBandwidth, rangeBandwidth * (xR - xL)))
            push!(cτ, shapeAt(ttes[j]) * spacingτ)
        end
    end
    np = nparams2D(length(xc))
    f, fx, fxx, fτ = zeros(T, np), zeros(T, np), zeros(T, np), zeros(T, np)
    D = zeros(T, np, np)
    d = zeros(T, np)
    for j = 1:nExp
        N = length(logmoneyness[j])
        for i = 1:N
            basisRows2D!(f, fx, fxx, fτ, xc, τc, cx, cτ, logmoneyness[j][i] / (σbar * τs[j]), τs[j])
            w = weights === nothing ? one(T) / (2vols[j][i]) : weights[j][i]
            wi = w^2 / N
            D .+= wi .* (f * f')
            d .+= wi * vols[j][i]^2 .* f
        end
    end
    #roughness penalty μ Σⱼ ∫ (∂³v/∂x³ / σ̄²)² dx over the quoted range of each expiry (trapezoidal rule): the smooth analogue
    #of the total variation penalty on the convexities of CVI; still a linear least-squares problem
    if roughness > 0
        f3 = zeros(T, np)
        for j = 1:nExp
            xL, xR = minimum(logmoneyness[j]) / (σbar * τs[j]), maximum(logmoneyness[j]) / (σbar * τs[j])
            xs = range(xL, xR, length=nQuadrature)
            hq = step(xs)
            for (q, x) in enumerate(xs)
                f3 .= zero(T)
                for l in eachindex(xc)
                    u, r = (x - xc[l]) / cx[l], (τs[j] - τc[l]) / cτ[l]
                    ψ = sqrt(1 + u^2 + r^2)
                    f3[4+l] = -3u * (1 + r^2) / (cx[l]^3 * ψ^5)
                end
                ωq = (q == 1 || q == nQuadrature ? hq / 2 : hq) / σbar^4
                D .+= roughness * ωq .* (f3 * f3')
            end
        end
    end
    trD = tr(D) / np
    for i = 1:np
        D[i, i] += ridge * trD
    end
    p = cholesky(Symmetric(D)) \ d
    return Multiquadric2DSurface(xc, τc, cx, cτ, σbar, p, collect(ttes), collect(forwards))
end

#Smoothing thin plate spline of the implied variance (the prior of Fengler (2009)), in the coordinates x = k/(σ̄√T) and
#a τ = a√T: f(p) = β₀ + β₁x + β₂τ + Σᵢ αᵢ φ(|p - pᵢ|), with φ(r) = r² log r (two dimensions), centered at every quote.
#On a single expiry, φ(r) = r³, which is the natural cubic smoothing spline. Minimizes (1/n) Σ wᵢ² (f(pᵢ) - vᵢ/σ̄²)² + λ J(f),
#where J is the thin plate energy (up to a constant): (A + nλ W⁻²) α + P β = v/σ̄², Pᵀα = 0.
export ThinPlateSpline2D, fitThinPlateSpline2D

struct ThinPlateSpline2D{T}
    xs::Vector{T}
    τs::Vector{T}
    α::Vector{T}
    β::Vector{T}
    σbar::T
    timeScale::T
    oneD::Bool
end

tpsKernel(r2, oneD) = oneD ? r2 * sqrt(r2) : (r2 <= 0 ? zero(r2) : r2 * log(r2) / 2)

"total implied variance of the thin plate spline at log-moneyness k and expiry tte"
function totalVarianceDerivatives(s::ThinPlateSpline2D{T}, k, tte) where {T}
    x, τ = k / (s.σbar * sqrt(tte)), s.timeScale * sqrt(tte)
    f = s.β[1] + s.β[2] * x + (s.oneD ? zero(T) : s.β[3] * τ)
    for i in eachindex(s.α)
        f += s.α[i] * tpsKernel((x - s.xs[i])^2 + (τ - s.τs[i])^2, s.oneD)
    end
    return (f * s.σbar^2 * tte,)
end

"""
    fitThinPlateSpline2D(ttes, logmoneyness, vols; weights=nothing, λ=1e-4, timeScale=1.0)

Smoothing thin plate spline of the implied variance of all the quotes (a center per quote), with residual weights `weights[j][i]`
(default 1/(2σ), approximately volatility residuals). With a single expiry, it is the natural cubic smoothing spline in x.
"""
function fitThinPlateSpline2D(ttes::AbstractVector{T}, logmoneyness::AbstractVector{<:AbstractVector{T}}, vols::AbstractVector{<:AbstractVector{T}};
    weights=nothing, λ=1e-4, timeScale=one(T)) where {T}
    nExp = length(ttes)
    oneD = nExp == 1
    function atm(ys, vs)
        i = searchsortedlast(ys, zero(T))
        i <= 0 && return vs[1]
        i >= length(ys) && return vs[end]
        θ = -ys[i] / (ys[i+1] - ys[i])
        return (1 - θ) * vs[i] + θ * vs[i+1]
    end
    σbar = sum(atm(logmoneyness[j], vols[j]) for j = 1:nExp) / nExp
    xs, τs, v, w = T[], T[], T[], T[]
    for j = 1:nExp, i in eachindex(logmoneyness[j])
        push!(xs, logmoneyness[j][i] / (σbar * sqrt(ttes[j])))
        push!(τs, timeScale * sqrt(ttes[j]))
        push!(v, vols[j][i]^2 / σbar^2)
        wi = weights === nothing ? one(T) / (2vols[j][i]) : weights[j][i]
        push!(w, wi * σbar^2) #residual weight on the scaled variance, as a weight in variance
    end
    n = length(xs)
    np = oneD ? 2 : 3
    M = zeros(T, n + np, n + np)
    for i = 1:n, l = 1:n
        M[i, l] = tpsKernel((xs[i] - xs[l])^2 + (τs[i] - τs[l])^2, oneD)
    end
    wbar = sum(abs2, w) / n
    for i = 1:n
        M[i, i] += n * λ * wbar / w[i]^2
        M[i, n+1] = 1; M[n+1, i] = 1
        M[i, n+2] = xs[i]; M[n+2, i] = xs[i]
        if !oneD
            M[i, n+3] = τs[i]; M[n+3, i] = τs[i]
        end
    end
    sol = M \ vcat(v, zeros(T, np))
    return ThinPlateSpline2D(xs, τs, sol[1:n], sol[n+1:end], σbar, timeScale, oneD)
end
