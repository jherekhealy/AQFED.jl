using Test, AQFED, CharFuncPricing, DataFrames, Statistics, CairoMakie

# Import helper functions from test/hestonmc_test.jl if needed
import AQFED.Random:
    AbstractRNGSeq,
    ZRNGSeq,
    OriginalScramblingRng,
    ScramblingRngAdapter,
    ScrambledSobolSeq,
    Owen, OwenFaureTezuka, NoScrambling, Digital, Burley

function make_seq(name::Symbol, rng, nd::Int, n::Int)
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
    elseif name == :sobol
        ScrambledSobolSeq(nd, n, NoScrambling())
    else
        error("unknown sequence name: $name")
    end
end

const SUPERSCRIPT_CHARS = Dict(
    '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴',
    '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹',
    '-' => '⁻',
)
superscript(n::Int) = join(SUPERSCRIPT_CHARS[c] for c in string(n))

function decade_ticks(vals)
    lo = floor(Int, log10(minimum(vals)))
    hi = ceil(Int, log10(maximum(vals)))
    exps = lo:hi
    positions = [10.0^e for e in exps]
    labels = ["10" * superscript(e) for e in exps]
    return (positions, labels)
end

function plot_asian_rqmc_convergence(
    df;
    output = nothing,
    title = "Heston Daily Arithmetic Asian RQMC Convergence (m=32)"
)
    fig = Figure(size = (800, 500), fontsize = 14)
    
    xticks_dec = decade_ticks(df.n_paths)
    yticks_dec = decade_ticks(df.std_err)

    ax = Axis(
        fig[1, 1],
        xscale = log10,
        yscale = log10,
        xlabel = "Number of paths (N)",
        ylabel = "Standard Error",
        title = title,
        xticks = xticks_dec,
        yticks = yticks_dec,
        xminorticksvisible = true,
        xminorgridvisible = true,
        yminorticksvisible = true,
        yminorgridvisible = true,
        xminorticks = IntervalsBetween(9),
        yminorticks = IntervalsBetween(9),
    )

    colors = Makie.wong_colors()
    labelmap = Dict(
        "rng"            => "ChaCha-20 PRNG",
        "sobol_burley"   => "Burley in ScrambledSobol",
        "sobol_burley4d" => "BurleySobol",
        "sobol_digital"  => "Digital ScrambledSobol"
    )
    markers = Dict(
        "rng"            => :circle,
        "sobol_burley"   => :rect,
        "sobol_burley4d" => :diamond,
        "sobol_digital"  => :utriangle
    )

    methods = ["rng", "sobol_burley", "sobol_burley4d", "sobol_digital"]

    for (i, method) in enumerate(methods)
        sub = sort(df[df.method .== method, :], :n_paths)
        isempty(sub) && continue

        scatterlines!(
            ax, sub.n_paths, sub.std_err;
            color = colors[mod1(i, length(colors))],
            linewidth = 2.2,
            marker = markers[method],
            markersize = 8,
            label = labelmap[method]
        )
    end

    axislegend(ax, position = :rt, framevisible = false)

    if output !== nothing
        save(output, fig)
    end

    return fig
end

function plot_asian_rqmc_prices(
    df;
    output = nothing,
    title = "Heston Daily Arithmetic Asian Price Convergence (m=32)"
)
    fig = Figure(size = (800, 500), fontsize = 14)
    xticks_dec = decade_ticks(df.n_paths)

    ax = Axis(
        fig[1, 1],
        xscale = log10,
        xlabel = "Number of paths (N)",
        ylabel = "Option Price",
        title = title,
        xticks = xticks_dec,
        xminorticksvisible = true,
        xminorgridvisible = true,
        yminorticksvisible = true,
        yminorgridvisible = true,
        xminorticks = IntervalsBetween(9),
    )

    colors = Makie.wong_colors()
    labelmap = Dict(
        "rng"            => "ChaCha-20 PRNG",
        "sobol_burley"   => "Burley in ScrambledSobol",
        "sobol_burley4d" => "BurleySobol",
        "sobol_digital"  => "Digital ScrambledSobol"
    )
    markers = Dict(
        "rng"            => :circle,
        "sobol_burley"   => :rect,
        "sobol_burley4d" => :diamond,
        "sobol_digital"  => :utriangle
    )

    methods = ["rng", "sobol_burley", "sobol_burley4d", "sobol_digital"]

    for (i, method) in enumerate(methods)
        sub = sort(df[df.method .== method, :], :n_paths)
        isempty(sub) && continue

        col = colors[mod1(i, length(colors))]

        band!(
            ax,
            sub.n_paths,
            sub.mean_price .- sub.std_err,
            sub.mean_price .+ sub.std_err;
            color = (col, 0.15),
        )

        scatterlines!(
            ax, sub.n_paths, sub.mean_price;
            color = col,
            linewidth = 2.2,
            marker = markers[method],
            markersize = 8,
            label = labelmap[method]
        )
    end

    axislegend(ax, position = :rt, framevisible = false)

    if output !== nothing
        save(output, fig)
    end

    return fig
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
    println("Dimensions: ", nd)

    sequence_names = [:sobol_burley, :sobol_burley4d, :sobol_digital, :rng]
    n_max = 32768
    m = 32 # 32 randomizations
    start = 0

    rows = NamedTuple[]

    for name in sequence_names
        println("Running sequence: ", name)
        local sample_sizes = Int[]
        local all_means = Matrix{Float64}(undef, 0, 0)

        for i_randomization in 1:m
            rng = AQFED.Random.ChachaSIMD{10}(UInt32, 20100921 + i_randomization)
            seq = make_seq(name, rng, nd, n_max)

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

    # Plot convergence (Standard Error vs N)
    output_plot_se = joinpath(@__DIR__, "heston_asian_rqmc_convergence.png")
    fig_se = plot_asian_rqmc_convergence(df_res; output = output_plot_se)
    @test isfile(output_plot_se)
    println("StdErr plot saved to: ", output_plot_se)

    # Alternate Plot (Price vs N with PRNG StdErr band)
    output_plot_price = joinpath(@__DIR__, "heston_asian_rqmc_prices.png")
    fig_price = plot_asian_rqmc_prices(df_res; output = output_plot_price)
    @test isfile(output_plot_price)
    println("Price plot saved to: ", output_plot_price)

    # Assert that at largest path count (n=32768), RQMC methods achieve lower standard error than PRNG
    max_n = maximum(df_res.n_paths)
    df_max = df_res[df_res.n_paths .== max_n, :]
    prng_se = df_max[df_max.method .== "rng", :std_err][1]
    burley_se = df_max[df_max.method .== "sobol_burley", :std_err][1]
    burley4d_se = df_max[df_max.method .== "sobol_burley4d", :std_err][1]

    println("Path count n=", max_n, ": PRNG SE = ", prng_se, ", Burley SE = ", burley_se, ", Burley4D SE = ", burley4d_se)

    @test burley_se < prng_se
    @test burley4d_se < prng_se
end
