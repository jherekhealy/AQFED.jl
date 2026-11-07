using Random
using Statistics
using LinearAlgebra
using Printf
using Distributed

import AQFED.Math: norminv
import AQFED.Random: next!
import AQFED.MonteCarlo: BrownianBridgeConstruction, transform!, BBCache
import AQFED.Random: ScrambledSobolSeq, Burley, DigitalSobolSeq,
                     OwenFaureTezuka, ScramblingRngAdapter,
                     AbstractRNGSeq, ModuloSobolSeq, NoScrambling
import Random123
using DataFrames
using CSV
using CairoMakie

@enum ScramblingKind begin
    BURLEY
    OWEN_FAURE_TEZUKA
    DIGITAL
    MODULO
    PHILOX
    SOBOL
end

@enum AsianKind begin
    GEOMETRIC
    ARITHMETIC
    VANILLA
end

struct SequenceSpec{F}
    name::Symbol
    dimension::Int
    npoints::Int
    replicate::Int
    scrambling::ScramblingKind
    seed::Tuple{UInt64,UInt64}
    constructor::F
end

function SequenceSpec(name::Symbol, D::Int, N::Int, rep::Int;
                      scrambling::ScramblingKind=BURLEY,
                      seed=(UInt64(3), UInt64(14)),
                      constructor=nothing)
    SequenceSpec{typeof(constructor)}(
        name, D, N, rep, scrambling, seed, constructor,
    )
end

function make_sequence(spec::SequenceSpec)
    D, N, rep = spec.dimension, spec.npoints, spec.replicate
    key = (spec.seed[1], spec.seed[2] + UInt64(rep - 1))
    spec.constructor !== nothing && return spec.constructor(D, N, rep, key)
    _make_sequence(Val(D), N, key, spec.scrambling)
end

function _make_sequence(::Val{D}, N::Int, key,
                        scrambling::ScramblingKind) where {D}
    if scrambling == BURLEY
        return ScrambledSobolSeq(D, N, Burley{D}(Random123.Philox4x(key)))
    elseif scrambling == OWEN_FAURE_TEZUKA
        return ScrambledSobolSeq(
            D, N, OwenFaureTezuka(31,
                ScramblingRngAdapter(Random123.Philox4x(key))))
    elseif scrambling == DIGITAL
        return DigitalSobolSeq(D, N, Random123.Philox4x(key))
    elseif scrambling == MODULO
        return ModuloSobolSeq(D, N, Random123.Philox4x(key))
    elseif scrambling == PHILOX
        return AbstractRNGSeq(Random123.Philox4x(key), D)
    elseif scrambling == SOBOL
        return ScrambledSobolSeq(D, N, NoScrambling())
    end
    throw(DomainError(scrambling, "unrecognized scrambling kind"))
end

function aqfed_points(seq, D::Int, N::Int)
    U = Matrix{Float64}(undef, D, N)
    @inbounds for j in 1:N
        next!(seq, @view U[:, j])
    end
    U
end

@inline function hyperbolic_local_vol(S::Float64, ν::Float64, β::Float64)
    s = max(S, eps(Float64))
    a = (1.0 - β + β^2) / β * s
    b = (β - 1.0) / β
    rad = s^2 + β^2 * (1.0 - s)^2
    ν * (a + b * (sqrt(max(rad, 0.0)) - β))
end

@inline log_local_vol(y::Float64, ν::Float64, β::Float64) =
    hyperbolic_local_vol(exp(y), ν, β) / exp(y)

@inline normal_from_uniform(u::Float64) = norminv(clamp(u, eps(), 1.0 - eps()))

function asian_estimate(U::Matrix{Float64};
                        S0=100.0, r=0.03, T=1.0, ν=0.30, β=0.50,
                        K=100.0, bridge=false, cacheSize=0,
                        asianKind::AsianKind=GEOMETRIC)
    D, N = size(U)
    h = T / D
    disc = exp(-r * T)
    bb = bridge ? BrownianBridgeConstruction(collect(range(h, T; length=D))) : nothing
    cache = nothing
    if bridge
        cacheSize = cacheSize == 0 ? 4 * ceil(Int, log2(D)) : cacheSize
        cache = BBCache{Int,Vector{Float64}}(cacheSize)
    end

    z = Vector{Float64}(undef, D)
    dw = Vector{Float64}(undef, D)
    payoffs = Vector{Float64}(undef, N)
    y0 = log(S0)

    @inbounds for p in 1:N
        if bridge
            for i in 1:D
                z[i] = normal_from_uniform(U[i, p])
            end
            transform!(bb, z, dw)
        else
            for i in 1:D
                dw[i] = sqrt(h) * normal_from_uniform(U[i, p])
            end
        end

        y = y0
        avg = 0.0
        logavg = 0.0
        for i in 1:D
            σ = log_local_vol(y, ν, β)
            y += (r - 0.5 * σ^2) * h + σ * dw[i]
            logavg += y
            asianKind == ARITHMETIC && (avg += exp(y))
        end

        if asianKind == GEOMETRIC
            avg = exp(logavg / D)
        elseif asianKind == ARITHMETIC
            avg /= D
        elseif asianKind == VANILLA
            avg = exp(y)
        else
            throw(DomainError(asianKind, "unrecognized Asian payoff"))
        end
        payoffs[p] = max(avg - K, 0.0)
    end

    μ = disc * mean(payoffs)
    se = disc * std(payoffs; corrected=true) / sqrt(N)
    return μ, se
end

struct ExperimentResult
    sequence::Symbol
    method::Symbol
    strike::Float64
    dimension::Int
    n::Int
    R::Int
    estimate::Float64
    rqmc_se::Float64
    replicate_sd::Float64
    elapsed::Float64
end

# One task computes one (sequence, dimension, N) experiment. Replicates are
# independent and can therefore be distributed without synchronization.
function run_one(
    sequence_name::Symbol,
    scrambling::ScramblingKind,
    strike::Float64,
    D::Int,
    N::Int,
    R::Int;
    S0=100.0, r=0.03, T=1.0, ν=0.30, β=0.50,
    bridge=false, seed=(UInt64(3), UInt64(14)),
    asianKind::AsianKind=VANILLA,
    useScaling=true,
)
    scale = useScaling ? S0 : 1.0
    K = useScaling ? strike / S0 : strike
    S = useScaling ? 1.0 : S0
    values = Vector{Float64}(undef, R)
    t0 = time()

    for rep in 1:R
        spec = SequenceSpec(sequence_name, D, N, rep;
                            scrambling=scrambling, seed=seed)
        U = aqfed_points(make_sequence(spec), D, N)
        value, _ = asian_estimate(U; S0=S, r=r, T=T, ν=ν, β=β,
                                  K=K, bridge=bridge, asianKind=asianKind)
        values[rep] = scale * value
    end

    μ = mean(values)
    sd = R > 1 ? std(values; corrected=true) : NaN
    ExperimentResult(sequence_name, bridge ? :rqmc_bb : :rqmc_plain,
                     strike, D, N, R, μ, sd / sqrt(R), sd, time() - t0)
end

# Distributed orchestration. Include the file on workers before calling this.
function run_all_experiments_parallel(; strikes=(80.0, 100.0, 120.0),
    Ns=2 .^ (7:16), R=32, dimensions=(10, 1000),
    S0=100.0, r=0.03, T=1.0, ν=0.30, β=0.50,
    bridge=false, seed=(UInt64(3), UInt64(14)),
    asianKind::AsianKind=VANILLA,
    scramblings=(BURLEY, OWEN_FAURE_TEZUKA, DIGITAL, PHILOX, MODULO),
    sequence_names=Symbol.(scramblings),
    add_workers=false,
    workers_to_add=Sys.CPU_THREADS,
)
    if add_workers && nworkers() < workers_to_add
        addprocs(workers_to_add - nworkers())
    end

    tasks = NamedTuple[]
    for D in dimensions, (j, scrambling) in enumerate(scramblings), N in Ns
        push!(tasks, (sequence_name=sequence_names[j], scrambling=scrambling,
                      strike=first(strikes), D=D, N=N, R=R))
    end

    results = pmap(tasks) do task
        run_one(task.sequence_name, task.scrambling, task.strike,
                task.D, task.N, task.R;
                S0=S0, r=r, T=T, ν=ν, β=β, bridge=bridge,
                seed=seed, asianKind=asianKind)
    end

    DataFrame((
        sequence=String[r.sequence for r in results],
        dimension=[r.dimension for r in results],
        N=[r.n for r in results],
        R=[r.R for r in results],
        estimate=[r.estimate for r in results],
        rqmc_se=[r.rqmc_se for r in results],
        replicate_sd=[r.replicate_sd for r in results],
        elapsed=[r.elapsed for r in results],
    ))
end

function print_result(x::ExperimentResult)
    @printf("%-20s D=%5d N=%7d R=%4d value=% .10f SE=% .5e time=% .2fs\n",
            String(x.sequence), x.dimension, x.n, x.R,
            x.estimate, x.rqmc_se, x.elapsed)
end

function plot_convergence(df::DataFrame; output="rqmc_parallel.pdf",
                          dimension=first(unique(df.dimension)))
    sub = filter(row -> row.dimension == dimension, df)
    sub.abs_error = abs.(sub.estimate .- 1.0)
    fig = Figure(size=(900, 620), fontsize=15)
    ax = Axis(fig[1, 1], xlabel="N", ylabel="absolute error",
              title="Parallel RQMC, D=$dimension", xscale=log2, yscale=log10)
    for (i, name) in enumerate(unique(sub.sequence))
        s = filter(row -> row.sequence == name, sub)
        sort!(s, :N)
        y = max.(s.abs_error, eps(Float64))
        lines!(ax, s.N, y; color=Cycled(i), linewidth=2, label=name)
        scatter!(ax, s.N, y; color=Cycled(i), markersize=7)
    end
    axislegend(ax, position=:rb)
    save(output, fig)
    fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Start Julia with, for example:
    # julia -p 8 rqmc_asian_flexible_sequences_parallel.jl
    df = run_all_experiments_parallel()
    CSV.write("rqmc_parallel.csv", df)
    plot_convergence(df)
end
