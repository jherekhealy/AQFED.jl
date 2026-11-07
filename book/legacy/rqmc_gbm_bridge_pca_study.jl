#DOES NOT WORK WITH BLACK AS BB => 1 Z only.  HESTON FT OK.
using Statistics
using Printf
using CairoMakie
using CSV
using DataFrames
using SpecialFunctions: erfinv
using LinearAlgebra

import AQFED.Random: next!, DigitalSobolSeq, ScrambledSobolSeq, NoScrambling, AbstractRNGSeq, niederreiter_nut_21201, joe_kuo
import AQFED.Random: Burley, OwenFaureTezuka, ScramblingRngAdapter, BurleySobolSeq
import Random123

const EXACT = 1.0
const S0 = 1.0
const SIGMA = 1.0
const T_EXP = 1.0

# Quantile function for N(0,1)
@inline function quantile_normal(x::Float64)
   x_clamped = clamp(x, 2^(-33), 1.0 - 2^(-33))
     return sqrt(2.0) * erfinv(2.0 * x_clamped - 1.0)
end


function make_sequence(D::Int, N::Int, rep::Int;
                       method::Symbol=:burley,
                       seed=(3, 14), direction_source=joe_kuo)
    key = (seed[1], seed[2]+ (rep - 1) )
    rng = Random123.Philox4x(UInt32, key)
    #rng = Chacha8SIMD(UInt32,key[2])
    if method == :digital
        return ScrambledSobolSeq(D, N, Digital{D}(rng), direction_source) #DigitalSobolSeq(D, N, rng, direction_source)
    elseif method == :modulo
        return ModuloSobolSeq(D,N, rng, direction_source)
        elseif method == :burley4d
        return BurleySobolSeq(D, rng)
        elseif method == :burley
        return ScrambledSobolSeq(D, N, Burley{D}(rng), direction_source)
        elseif method == :owen_faure_tezuka
        return ScrambledSobolSeq(
            D,
            N,
            OwenFaureTezuka(31, ScramblingRngAdapter(rng)), direction_source
            )
        elseif method == :sobol
        return ScrambledSobolSeq(D, N, NoScrambling(), direction_source)
        elseif method == :philox
        return AbstractRNGSeq(rng, D)
    end
    error("Unknown method: $method")
end
# -------------------------------------------------------------------
# Precomputed Construction Cache
# -------------------------------------------------------------------

struct BrownianBridgeCache
    D::Int
    left::Vector{Int}
    right::Vector{Int}
    mid::Vector{Int}
    w_left::Vector{Float64}
    w_right::Vector{Float64}
    std_dev::Vector{Float64}
end

function BrownianBridgeCache(D::Int, T::Float64)
    left = zeros(Int, D)
    right = zeros(Int, D)
    mid = zeros(Int, D)
    w_left = zeros(Float64, D)
    w_right = zeros(Float64, D)
    std_dev = zeros(Float64, D)

    # Queue containing (l, r) boundaries for bisection
    queue = Tuple{Int, Int}[]
    push!(queue, (0, D))

    mid[1] = D
    std_dev[1] = sqrt(T)
    left[1] = 0
    right[1] = 0

    idx = 2
    while !isempty(queue) && idx <= D
        l, r = popfirst!(queue)
        if r - l <= 1
            continue
        end
        m = l + (r - l) ÷ 2

        t_l = l * (T / D)
        t_r = r * (T / D)
        t_m = m * (T / D)

        left[idx] = l
        right[idx] = r
        mid[idx] = m

        w_left[idx] = (t_r - t_m) / (t_r - t_l)
        w_right[idx] = (t_m - t_l) / (t_r - t_l)
        std_dev[idx] = sqrt((t_m - t_l) * (t_r - t_m) / (t_r - t_l))

        idx += 1
        push!(queue, (l, m))
        push!(queue, (m, r))
    end

    return BrownianBridgeCache(D, left, right, mid, w_left, w_right, std_dev)
end

struct PCACache
    D::Int
    # L_matrix is (D x D), where W = L * Z
    L_matrix::Matrix{Float64}
end

function PCACache(D::Int, T::Float64)
    dt = T / D
    # Exact analytical eigenvectors & eigenvalues for Brownian motion covariance:
    # C_ij = dt * min(i, j)
    L = Matrix{Float64}(undef, D, D)
    for j in 1:D
        lambda_j = dt / (4.0 * sin((2j - 1) * pi / (2 * (2D + 1)))^2)
        sqrt_lambda = sqrt(lambda_j)
        scaling = 2.0 / sqrt(2D + 1)
        for i in 1:D
            L[i, j] = sqrt_lambda * scaling * sin((2j - 1) * i * pi / (2D + 1))
        end
    end
    return PCACache(D, L)
end

# -------------------------------------------------------------------
# Path Construction Functions
# -------------------------------------------------------------------

@inline function construct_path_standard!(z::Vector{Float64}, w::Vector{Float64}, D::Int, dt::Float64)
    sqrt_dt = sqrt(dt)
    w[1] = sqrt_dt * z[1]
    @inbounds for i in 2:D
        w[i] = w[i-1] + sqrt_dt * z[i]
    end
end

@inline function construct_path_bridge!(z::Vector{Float64}, w::Vector{Float64}, cache::BrownianBridgeCache)
    w[cache.D] = cache.std_dev[1] * z[1]
    @inbounds for k in 2:cache.D
        l = cache.left[k]
        r = cache.right[k]
        m = cache.mid[k]

        w_l = l == 0 ? 0.0 : w[l]
        w_r = w[r]

        w[m] = cache.w_left[k] * w_l + cache.w_right[k] * w_r + cache.std_dev[k] * z[k]
    end
end

@inline function construct_path_pca!(z::Vector{Float64}, w::Vector{Float64}, cache::PCACache)
    mul!(w, cache.L_matrix, z)
end

# -------------------------------------------------------------------
# Integration Core
# -------------------------------------------------------------------

function integrate(D::Int, N::Int, rep::Int;
                   method::Symbol=:burley,
                   construction::Symbol=:standard,
                   seed=(UInt64(3), UInt64(14)))

    seq = make_sequence(D, N, rep; method=method, seed=seed)
    x = Vector{Float64}(undef, D)
    z = Vector{Float64}(undef, D)
    w = Vector{Float64}(undef, D)

    dt = T_EXP / D

    # Setup construction structures
    bb_cache = construction == :bridge ? BrownianBridgeCache(D, T_EXP) : nothing
    pca_cache = construction == :pca ? PCACache(D, T_EXP) : nothing

    total = 0.0
    @inbounds for _ in 1:N
        next!(seq, x)
        for i in 1:D
            z[i] = quantile_normal(x[i])
        end

        if construction == :standard
            construct_path_standard!(z, w, D, dt)
            elseif construction == :bridge
            construct_path_bridge!(z, w, bb_cache)
            elseif construction == :pca
            construct_path_pca!(z, w, pca_cache)
        else
            error("Unknown construction: $construction")
        end

        # S_T = S0 * exp(-0.5 * sigma^2 * T + sigma * W_T)
        W_T = w[D]
        exponent = -0.5 * (SIGMA^2) * T_EXP + SIGMA * W_T
        total += S0 * exp(exponent)
    end
    return total / N
end

# -------------------------------------------------------------------
# Parallel Runner & Results Aggregation
# -------------------------------------------------------------------

function compute_results(
    ; dimensions=collect(1:20000),
    Ns=[1024],
    R=64,
    methods=(:philox, :sobol, :digital, :burley, :burley4d, :owen_faure_tezuka),
    constructions=(:standard, :bridge, :pca),
    seed=(UInt64(3), UInt64(14)),
    csvfile="gbm_raw_results.csv",
    progress=true,
    )
    tasks = [(method=m, construction=c, dimension=D, N=N, replicate=rep)
             for m in methods
                 for c in constructions
                     for D in dimensions
                         for N in Ns
                             for rep in 1:(m == :digital ? 1 : R)]

                                 evaluate = task -> begin
                                     value = integrate(
                                         task.dimension, task.N, task.replicate;
                                         method=task.method,
                                         construction=task.construction,
                                         seed=seed,
                                         )
                                     (
                                         method=String(task.method),
                                         construction=String(task.construction),
                                         dimension=task.dimension,
                                         N=task.N,
                                         replicate=task.replicate,
                                         estimate=value,
                                         error=value - EXACT,
                                         abs_error=abs(value - EXACT),
                                         )
                                 end

                                 rows = if progress
                                     run_parallel_progress(evaluate, tasks; description="GBM evaluations")
                                 else
                                     output = Vector{Any}(undef, length(tasks))
                                     Threads.@threads for k in eachindex(tasks)
                                         output[k] = evaluate(tasks[k])
                                     end
                                     output
                                 end

                                 df = DataFrame(rows)
                                 sort!(df, [:method, :construction, :dimension, :N, :replicate])
                                 CSV.write(csvfile, df)
                                 return df
                             end

                             function aggregate_results(raw::DataFrame)
                                 grouped = groupby(raw, [:method, :construction, :dimension, :N])
                                 result = combine(grouped) do sub
                                     values = sub.estimate
                                     m = mean(values)
                                     sd = length(values) > 1 ? std(values; corrected=true) : NaN
                                     (
                                         estimate=m,
                                         error=m - EXACT,
                                         abs_error=abs(m - EXACT),
                                         replicate_sd=sd,
                                         std_err=length(values) > 1 ? sd / sqrt(length(values)) : NaN,
                                         R=length(values),
                                         )
                                 end
                                 sort!(result, [:method, :construction, :dimension, :N])
                                 result
                             end


  if abspath(PROGRAM_FILE) == @__FILE__
                             log_range = unique(exp10.(range(log10(1), log10(10000), length=200)) .|> round .|> Int)

                             raw_df = compute_results(
                                 dimensions=log_range,
                                 methods=(:philox, :digital, :burley, :burley4d, :owen_faure_tezuka),
                                 Ns=[1024],
                                 R=32,constructions = (:standard),
                                 csvfile="gemini_gbm_raw_results_1k.csv",
                                 seed=(20130129, 20109021)
                                 )
                             agg_df = aggregate_results(raw_df)
                             CSV.write("gemini_gbm_aggregated_results_1k.csv", agg_df)
                
                         end                             