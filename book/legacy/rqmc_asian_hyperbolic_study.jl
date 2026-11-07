using Random
using Statistics
using LinearAlgebra
using Printf

# Reproduction of the Asian-option experiment in Hok & Kucherenko,
# "The importance of being scrambled: supercharged Quasi Monte Carlo".
#
# Model:
#   dS = r*S*dt + sigma_tilde(S)*dW
#   Y = log(S)
#   dY = (r - 0.5*sigma(Y)^2)dt + sigma(Y)dW
#
# The paper uses S0=100, r=3%, T=1, nu=30%, beta=0.5,
# d=256 monitoring steps, and strikes 80, 100, 120.
#
# This file is self-contained apart from the Sobol/RQMC backend.
# It uses QuasiMonteCarlo.jl for Sobol points and random digital shifts.
# If AQFED.Random.ScrambledSobolSeq/Burley is available, replace the
# point-generation functions below with the corresponding AQFED calls.

import Pkg

try
    @eval using QuasiMonteCarlo
catch
    error("Install QuasiMonteCarlo.jl first: import Pkg; Pkg.add(\"QuasiMonteCarlo\")")
end

struct ExperimentResult
    method::Symbol
    strike::Float64
    n::Int
    K::Int
    estimate::Float64
    rqmc_se::Float64
    replicate_sd::Float64
    elapsed::Float64
end

# Hyperbolic local volatility from equation (10) of the paper.
# The formula is evaluated in terms of S, with sigma_tilde(S) having
# units of absolute price volatility.
@inline function hyperbolic_local_vol(S::Float64, ν::Float64, β::Float64)
    s = max(S, eps(Float64))
    a = (1.0 - β + β^2) / β * s
    b = (β - 1.0) / β
    rad = a^2 + b * b * (1.0 - s)^2 - β
    return ν * sqrt(max(rad, 0.0))
end

@inline function log_local_vol(y::Float64, ν::Float64, β::Float64)
    S = exp(y)
    return hyperbolic_local_vol(S, ν, β) / S
end

# Box-Muller inverse normal. The Sobol point is clipped away from 0 and 1.
@inline function invnorm(u::Float64)
    # Acklam approximation
    p = clamp(u, 1.0e-15, 1.0 - 1.0e-15)
    a1 = -3.969683028665376e1
    a2 = 2.209460984245205e2
    a3 = -2.759285104469687e2
    a4 = 1.383577518672690e2
    a5 = -3.066479806614716e1
    a6 = 2.506628277459239e0
    b1 = -5.447609879822406e1
    b2 = 1.615858368580409e2
    b3 = -1.556989798598866e2
    b4 = 6.680131188771972e1
    b5 = -1.328068155288572e1
    c1 = -7.784894002430293e-3
    c2 = -3.223964580411365e-1
    c3 = -2.400758277161838e0
    c4 = -2.549732539343734e0
    c5 = 4.374664141464968e0
    c6 = 2.938163982698783e0
    d1 = 7.784695709041462e-3
    d2 = 3.224671290700398e-1
    d3 = 2.445134137142996e0
    d4 = 3.754408661907416e0
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow
        q = sqrt(-2.0 * log(p))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
               ((((d1*q+d2)*q+d3)*q+d4)*q+1.0)
    elseif p > phigh
        q = sqrt(-2.0 * log(1.0-p))
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                ((((d1*q+d2)*q+d3)*q+d4)*q+1.0)
    else
        q = p - 0.5
        r = q*q
        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
               (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0)
    end
end

# In-place Brownian bridge for equally spaced times t_i=i*T/d.
# z contains d independent N(0,1) coordinates. w contains d Brownian
# increments whose cumulative sums produce W(t_i).
function brownian_bridge_increments!(w::Vector{Float64}, z::Vector{Float64}, T::Float64)
    n = length(z)
    W = zeros(Float64, n + 1)
    filled = falses(n + 1)
    filled[1] = true
    filled[end] = true

    W[end] = sqrt(T) * z[1]
    left = 1
    right = n + 1
    nextz = 2

    # At each level, split all currently available intervals. This is the
    # usual midpoint bridge ordering and is sufficient for the experiment.
    while nextz <= n
        intervals = Tuple{Int,Int}[]
        for i in 1:n
            if filled[i] && filled[i+1]
                # This is not used: initially only [1,n+1] is filled.
            end
        end

        # Find unfilled points with the largest interval priority. The
        # priority ordering gives midpoint-first, then recursively finer
        # levels. For arbitrary n, use the largest temporal gap.
        candidates = Tuple{Int,Int,Int}[]
        i = 1
        while i <= n
            while i <= n && filled[i]
                i += 1
            end
            i > n && break
            l = i - 1
            while i <= n && !filled[i]
                i += 1
            end
            r = i
            push!(candidates, (r-l, l, r))
        end

        isempty(candidates) && break
        sort!(candidates, by=x -> (-x[1], x[2]))

        for (_, l, r) in candidates
            nextz > n && break
            m = (l + r) ÷ 2
            if filled[m]
                continue
            end
            tl = T * (l - 1) / n
            tm = T * (m - 1) / n
            tr = T * (r - 1) / n
            γ = (tm - tl) / (tr - tl)
            mean = (1.0 - γ) * W[l] + γ * W[r]
            var = (tm - tl) * (tr - tm) / (tr - tl)
            W[m] = mean + sqrt(max(var, 0.0)) * z[nextz]
            filled[m] = true
            nextz += 1
        end
    end

    # Fill any points missed by the midpoint priority, using conditional
    # interpolation with the next unused normal.
    while nextz <= n
        for m in 2:n
            if !filled[m]
                l = m - 1
                while l >= 1 && !filled[l]
                    l -= 1
                end
                r = m + 1
                while r <= n && !filled[r]
                    r += 1
                end
                tl = T * (l - 1) / n
                tm = T * (m - 1) / n
                tr = T * (r - 1) / n
                γ = (tm - tl) / (tr - tl)
                W[m] = (1-γ)*W[l] + γ*W[r] +
                       sqrt((tm-tl)*(tr-tm)/(tr-tl))*z[nextz]
                filled[m] = true
                nextz += 1
                break
            end
        end
    end

    @inbounds for i in 1:n
        w[i] = W[i+1] - W[i]
    end
    return w
end

function sobol_points(n::Int, d::Int)
    # QuasiMonteCarlo's API has changed across releases. This helper handles
    # the common generate(nsamples, SobolSample()) interface.
    sampler = SobolSample()
    try
        return QuasiMonteCarlo.generate_samples(sampler, d, n)'
    catch
        try
            return QuasiMonteCarlo.sample(n, d, sampler)'
        catch
            error("Could not find a compatible QuasiMonteCarlo Sobol API")
        end
    end
end

function digital_shift!(U::Matrix{Float64}, rng::AbstractRNG)
    d, n = size(U)
    shift = rand(rng, d)
    @inbounds for j in 1:d, i in 1:n
        U[j,i] = mod(U[j,i] + shift[j], 1.0)
        U[j,i] = clamp(U[j,i], 1.0e-15, 1.0 - 1.0e-15)
    end
    return U
end

# Simulate one Asian geometric-average call estimate.
function asian_estimate(
    U::Matrix{Float64};
    S0 = 100.0,
    r = 0.03,
    T = 1.0,
    ν = 0.30,
    β = 0.50,
    K = 100.0,
    bridge = false,
)
    d, n = size(U)
    h = T / d
    sqrt_h = sqrt(h)
    disc = exp(-r*T)
    y0 = log(S0)
    total = 0.0
    z = Vector{Float64}(undef, d)
    dw = Vector{Float64}(undef, d)

    @inbounds for p in 1:n
        for i in 1:d
            z[i] = invnorm(U[i,p])
        end
        if bridge
            brownian_bridge_increments!(dw, z, T)
        else
            for i in 1:d
                dw[i] = sqrt_h * z[i]
            end
        end

        y = y0
        logavg = 0.0
        for i in 1:d
            σ = log_local_vol(y, ν, β)
            y += (r - 0.5*σ*σ)*h + σ*dw[i]
            logavg += y
        end
        avg = exp(logavg / d)
        total += max(avg - K, 0.0)
    end
    return disc * total / n
end

function run_rqmc(
    method::Symbol,
    strike::Float64,
    n::Int,
    Krep::Int;
    S0=100.0,
    r=0.03,
    T=1.0,
    ν=0.30,
    β=0.50,
    bridge=(method != :mc),
    seed=1234,
)
    d = 256
    estimates = zeros(Float64, Krep)
    t0 = time()

    if method == :mc
        rng = MersenneTwister(seed)
        for k in 1:Krep
            U = rand(rng, d, n)
            estimates[k] = asian_estimate(
                U; S0=S0, r=r, T=T, ν=ν, β=β, K=strike, bridge=false
            )
        end
    elseif method == :qmc || method == :rqmc
        base = sobol_points(n, d)
        for k in 1:Krep
            U = copy(base)
            if method == :rqmc
                digital_shift!(U, MersenneTwister(seed + k - 1))
            end
            estimates[k] = asian_estimate(
                U; S0=S0, r=r, T=T, ν=ν, β=β, K=strike, bridge=bridge
            )
        end
    else
        error("method must be :mc, :qmc, or :rqmc")
    end

    μ = mean(estimates)
    sd = Krep > 1 ? std(estimates; corrected=true) : NaN
    se = sd / sqrt(Krep)
    return ExperimentResult(method, strike, n, Krep, μ, se, sd, time()-t0)
end

function print_result(x::ExperimentResult)
    @printf("%-6s K=%6.1f n=%7d R=%4d value=% .10f RQMC-SE=% .4e replicate-SD=% .4e time=% .2fs\n",
            String(x.method), x.strike, x.n, x.K, x.estimate,
            x.rqmc_se, x.replicate_sd, x.elapsed)
end

function main()
    Ns = 2 .^ (9:14)
    strikes = [80.0, 100.0, 120.0]

    # The paper uses K=10 for RQMC error tables. Increase to 32, 64, or 128
    # for a more stable modern experiment.
    Krep = 32

    for strike in strikes
        println("\nStrike = ", strike)
        for n in Ns
            # Plain MC: use one independent random estimate per replicate.
            print_result(run_rqmc(:mc, strike, n, Krep; bridge=false))
            # Digital-shift RQMC with Brownian bridge.
            print_result(run_rqmc(:rqmc, strike, n, Krep; bridge=true))
            # Unscrambled Sobol is deterministic; replicate-SD is not an error
            # estimate. We report one point-set estimate only.
            print_result(run_rqmc(:qmc, strike, n, 1; bridge=true))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
