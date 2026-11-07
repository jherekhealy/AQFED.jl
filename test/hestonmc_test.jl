# Quasi Monte-Carlo with the Heston model: randomized Sobol sequences against a pseudo-random generator.
# The convergence studies and the figures are in book/ch09_heston_qmc.jl (former content: book/legacy/hestonmc_study.jl).
using AQFED: AQFED
import AQFED.Random
import AQFED.Random:
    AbstractRNGSeq,
    ZRNGSeq,
    OriginalScramblingRng,
    ScramblingRngAdapter,
    ScrambledSobolSeq,
    Owen, OwenFaureTezuka, NoScrambling, Digital, Burley

using Random123: Random123
import RandomNumbers: AbstractRNG
import Random: MersenneTwister, rand!
using Statistics
import AQFED.TermStructure:
    SVISection,
    VarianceSurfaceBySection,
    varianceByLogmoneyness,
    HestonModel,
    DoubleHestonModel,
    LocalVolatilityModel,
    ConstantBlackModel,
    TSBlackModel,
    ConstantRateCurve
import AQFED.MonteCarlo: MCConvergenceResult
using Test
using CharFuncPricing
using Statistics
using DataFrames

function make_sequence(name::Symbol, rng, nd::Int, n::Int)
    if name == :rng
        AbstractRNGSeq(rng, nd)
    elseif name == :sobol_owen
        ScrambledSobolSeq(nd, n, OwenFaureTezuka(31, ScramblingRngAdapter(rng)))
    elseif name == :sobol_digital
        ScrambledSobolSeq(nd, n, Digital{nd}(rng))
    elseif name == :sobol_burley4d
        AQFED.Random.BurleySobolSeq(nd, rng)
    elseif name == :sobol_burley
        ScrambledSobolSeq(nd, n, Burley{nd}(rng))
    elseif name == :sobol_equi
        myseq = ScrambledSobolSeq(nd, n, Digital{nd}(rng))
        myseq.shift = round.(UInt32, collect(0:(nd - 1)) ./ nd * 2^32)
        myseq.x .= myseq.shift
        myseq
    elseif name == :sobol
        ScrambledSobolSeq(nd, n, NoScrambling())
    else
        error("unknown sequence name: $name")
    end
end

@testset "Heston RQMC Daily Arithmetic Asian" begin
    # 1-year Daily Arithmetic Asian Call option setup (252 daily observations)
    τ = 1.0
    nObs = 252
    obsTimes = collect(range(1.0 / nObs, τ, length = nObs))
    spot = 1.0
    strike = 1.0
    payoff = AQFED.MonteCarlo.AsianOption(true, strike, obsTimes)

    # Heston Model Parameters (High volvol & Low volvol options)
    hParams_high = CharFuncPricing.HestonParams(0.133, 0.35, 0.321, -0.63, 1.388)
    hParams_low  = CharFuncPricing.HestonParams(0.0313, 0.111927, 0.062762, -0.755929, 0.126743)

    hParams = hParams_high
    model = AQFED.TermStructure.HestonModel(
        hParams.v0, hParams.κ, hParams.θ, hParams.ρ, hParams.σ, 0.0, 0.0
    )

    # Discretization Scheme (QE or FullTruncation or DVSS2X)
    scheme = AQFED.MonteCarlo.QE(false, true)
    timesteps = 252
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    config = AQFED.MonteCarlo.GridConfig(
        model, specTimes, 1.0 / timesteps, scheme,
        mindt = 0.5 / 252, tstop = 0.0
    )
    nd = AQFED.MonteCarlo.ndims(model, config)

    sequence_names = [:sobol_burley, :sobol_burley4d, :sobol_digital, :rng]
    n_max = 32768
    m = 32 # 32 randomizations
    start = 0

    rows = NamedTuple[]

    for name in sequence_names
        local sample_sizes = Int[]
        local all_means = Matrix{Float64}(undef, 0, 0)

        for i_randomization in 1:m
            rng = AQFED.Random.ChachaSIMD{20}(UInt32, 20100921 + i_randomization)
            seq = make_sequence(name, rng, nd, n_max)

            result = AQFED.MonteCarlo.simulateIter(
                seq, model, spot, payoff, start, n_max, config, isRestart = false
            )

            res = result isa Vector ? result[1] : result

            if i_randomization == 1
                sample_sizes = res.samples
                all_means = Matrix{Float64}(undef, m, length(sample_sizes))
            end

            all_means[i_randomization, :] .= res.means
        end

        for (j, n_paths) in enumerate(sample_sizes)
            if n_paths >= 1024
                means_at_n = all_means[:, j]
                mean_price = mean(means_at_n)
                std_err = std(means_at_n) / sqrt(m)
                push!(rows, (; n_paths = n_paths, method = String(name), mean_price = mean_price, std_err = std_err))
            end
        end
    end

    df_res = DataFrame(rows)
    @test !isempty(df_res)

    # Assert that at largest path count (n=32768), RQMC methods achieve lower standard error than PRNG
    max_n = maximum(df_res.n_paths)
    df_max = df_res[df_res.n_paths .== max_n, :]
    prng_se = df_max[df_max.method .== "rng", :std_err][1]
    burley_se = df_max[df_max.method .== "sobol_burley", :std_err][1]
    burley4d_se = df_max[df_max.method .== "sobol_burley4d", :std_err][1]

    @test burley_se < prng_se
    @test burley4d_se < prng_se
end