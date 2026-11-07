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
using Statistics
using CairoMakie

using DataFrames
using Statistics
using CairoMakie

# ==========================================================================
# Assumes model, spot, payoff, strikes, refValues already exist, exactly as
# in your setup code (hParams, model = AQFED.TermStructure.HestonModel(...),
# payoff = AQFED.MonteCarlo.ListPayoff(...), strikes = [...]).
# ==========================================================================

# --------------------------------------------------------------------------
# Sequence constructor, factored out of your inner loop so it can be reused
# across every (bb, timesteps) combination in the sweep below.
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Log-spaced integer time-step grid, 8 -> 1000. Rounding to Int collapses
# some points at the low end, so we dedupe; use npoints to land in your
# target 100-200 range after dedup (180 -> roughly 130-160 unique points).
# --------------------------------------------------------------------------
function log_timesteps(lo::Int, hi::Int, npoints::Int)
    raw = round.(Int, exp10.(range(log10(lo), log10(hi), length = npoints)))
    return sort(unique(raw))
end

# --------------------------------------------------------------------------
# Sweep timesteps for the Full Truncation scheme, with and without the
# Brownian bridge (configs[3] / configs[4] in your list), tracking standard
# error only. Re-derives the Sobol-dimension method fallback per timestep,
# exactly like your original script (nd changes with the number of steps).
# --------------------------------------------------------------------------
function collect_fulltruncation_stderr_df(
    model, spot, payoff, strikes, timesteps_range;
    m = 64,
    n = 1024,
    start = 0,
    mindt = 0.5 / 365,
   rng_name = :philox,
   isft = true

)
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)

    bb_configs = Dict(
        false => isft ? AQFED.MonteCarlo.FullTruncation(false) : AQFED.MonteCarlo.QE(false,true),
        true  => isft ? AQFED.MonteCarlo.FullTruncation(true) : AQFED.MonteCarlo.QE(true,true)
    )

    # Each (bb, timesteps) combination is fully independent -- same model,
    # payoff, and RNG seeds either way -- so this is the natural axis to
    # parallelize over. One pre-allocated output slot per combo means no
    # thread ever writes to another thread's data, so there's nothing to
    # lock and the result is identical to the single-threaded version.
    combos = [(bb, timesteps) for bb in (false, true) for timesteps in timesteps_range]
   order = sortperm(combos; by = c -> -c[2])
  chunk_results = Vector{Vector{NamedTuple}}(undef, length(combos))

    @sync for idx in order
        Threads.@spawn begin
        bb, timesteps = combos[idx]
        fscheme = bb_configs[bb]

        config = AQFED.MonteCarlo.GridConfig(
            model, specTimes, 1.0 / timesteps, fscheme,
            mindt = min(mindt, 0.5 / timesteps), tstop = 0.0,
        )
        nd = AQFED.MonteCarlo.ndims(model, config)

        # Sobol dimension cap -> smaller method set for fine time grids
        names_here = nd <= 21201 ?
            [:sobol_burley, :rng, :sobol_owen, :sobol_digital, :sobol_burley4d] :
            [:sobol_burley4d, :rng]

        prices = Dict(
            strike => Dict(name => AQFED.MonteCarlo.MCConvergenceResult[] for name in names_here)
            for strike in strikes
        )

        for i_randomization in 1:m
            for name in names_here
                   rng = if rng_name == :philox
                        Random123.Philox4x(UInt32, (20130129,20100921 + i_randomization))
                    else
                     AQFED.Random.ChachaSIMD{10}(UInt32, 20100921 + i_randomization)
                    end
                  
                seq = make_sequence(name, rng, nd, n)

                resultv = AQFED.MonteCarlo.simulateIter(
                    seq, model, spot, payoff, start, n, config, isRestart = false,
                )

                for (i, strike) in enumerate(strikes)
                    push!(prices[strike][name], resultv[i])
                end
            end
        end

        local_rows = NamedTuple[]
        for strike in strikes, name in names_here
            vals = prices[strike][name]
            se = std([AQFED.MonteCarlo.last_mean(r) / sqrt(m) for r in vals])
            estimate = mean([AQFED.MonteCarlo.last_mean(r) for r in vals])
            push!(local_rows, (
                dimension = timesteps,
                method    = String(name),
                bb        = bb,
                strike    = strike,
                std_err   = se,
                estimate = estimate
            ))
        end

        chunk_results[idx] = local_rows

        @info "swept" bb timesteps nd nmethods = length(names_here) thread = Threads.threadid()
    end
end

    rows = reduce(vcat, chunk_results)

    return DataFrame(rows)
end

# Major ticks at integer powers of ten only (e.g. 10^1, 10^2, 10^3), spanning
# whatever decades the data actually falls in -- keeps a log axis from being
# cluttered with 2×10^2, 3×10^2, ... style minor labels.
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Plot: standard error vs. number of time steps, one panel per BB setting,
# one curve per method -- same visual language as the earlier convergence
# plots (wong_colors, distinct markers, log-log, minor grid, clean legend).
# --------------------------------------------------------------------------
function plot_fulltruncation_stderr(df; strike = 1.0, output = nothing,  scheme_title="Full Truncation", labelmap = Dict("rng" => "Philox", "sobol_burley" => "Burley(HDD)", "sobol_burley4d" => "Burley(4DP)", "sobol_digital"=>"digital shift", "sobol_owen"=>"ACM-823"))

    fig = Figure(size = (800, 400), fontsize = 14)
    colors  = Makie.wong_colors()
    markers = [:circle, :diamond, :rect, :utriangle, :dtriangle, :star5, :hexagon, :cross]

    d = df[df.strike .== strike, :]
    methods = sort(unique(d.method))

    # Shared tick set across both panels so they stay directly comparable
    xticks_dec = decade_ticks(d.dimension)
    yticks_dec = decade_ticks(d.std_err)

    function make_panel!(fig, position, bb_flag, title)

        ax = Axis(
            fig[1, position],
            xscale = log10,
             #yscale = log10,
            xlabel = "Number of time steps",
            ylabel = position == 1 ? "Standard error" : "",
            title = title,
            xticks = xticks_dec,
            #yticks = yticks_dec,
            xminorticksvisible = true,
            xminorgridvisible = true,
            yminorticksvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(9),
            yminorticks = IntervalsBetween(9),
        )

        sub_bb = d[d.bb .== bb_flag, :]

        for (i, method) in enumerate(methods)
            s = sort(sub_bb[sub_bb.method .== method, :], :dimension)
            isempty(s) && continue

            lines!(
                ax, s.dimension, s.std_err;
                color = colors[mod1(i, length(colors))],
                linewidth = 2.0,                
                label = labelmap[method],
            )
        end
        ylims!(ax, low = 0.0)
        return ax
    end

    ax1 = make_panel!(fig, 1, false, string(scheme_title," (no BB)"))
    ax2 = make_panel!(fig, 2, true,  string(scheme_title," (with BB)"))

    # Shared legend spanning both panels, placed below the plots. Pull the
    # entries from whichever axis actually has all the series (falls back
    # to ax1 if ax2 dropped some methods at high nd).
    legend_ax = length(ax2.scene.plots) >= length(ax1.scene.plots) ? ax2 : ax1

    Legend(
        fig[2, 1:2],
        legend_ax,
        orientation = :horizontal,
        framevisible = false,
        labelsize = 11,
        nbanks = 1,
        tellwidth = false,
        tellheight = true,
    )

    rowsize!(fig.layout, 2, Auto(0.12))
    rowgap!(fig.layout, 1, 0)

    if output !== nothing
        save(output, fig)
    end

    return fig
end

function plot_fulltruncation_error(df; strike = 1.0, output = nothing,  scheme_title="Full Truncation", labelmap = Dict("rng" => "Philox", "sobol_burley" => "Burley(HDD)", "sobol_burley4d" => "Burley(4DP)", "sobol_digital"=>"digital shift", "sobol_owen"=>"ACM-823"))

    fig = Figure(size = (800, 400), fontsize = 14)
    colors  = Makie.wong_colors()
    markers = [:circle, :diamond, :rect, :utriangle, :dtriangle, :star5, :hexagon, :cross]

    d = df[df.strike .== strike, :]
    methods = sort(unique(d.method))

    # Shared tick set across both panels so they stay directly comparable
    xticks_dec = decade_ticks(d.dimension)
    yticks_dec = decade_ticks(abs.(d.error))

    function make_panel!(fig, position, bb_flag, title)

        ax = Axis(
            fig[1, position],
            xscale = log10,
             #yscale = log10,
            xlabel = "Number of time steps",
            ylabel = position == 1 ? "Error" : "",
            title = title,
            xticks = xticks_dec,
            #yticks = yticks_dec,
            xminorticksvisible = true,
            xminorgridvisible = true,
            yminorticksvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(9),
            yminorticks = IntervalsBetween(9),
        )

        sub_bb = d[d.bb .== bb_flag, :]

        for (i, method) in enumerate(methods)
            s = sort(sub_bb[sub_bb.method .== method, :], :dimension)
            isempty(s) && continue

            lines!(
                ax, s.dimension, abs.(s.error);
                color = colors[mod1(i, length(colors))],
                linewidth = 2.0,                
                label = labelmap[method],
            )
        end
        ylims!(ax, low = 0.0)
        return ax
    end

    ax1 = make_panel!(fig, 1, false, string(scheme_title," (no BB)"))
    ax2 = make_panel!(fig, 2, true,  string(scheme_title," (with BB)"))

    # Shared legend spanning both panels, placed below the plots. Pull the
    # entries from whichever axis actually has all the series (falls back
    # to ax1 if ax2 dropped some methods at high nd).
    legend_ax = length(ax2.scene.plots) >= length(ax1.scene.plots) ? ax2 : ax1

    Legend(
        fig[2, 1:2],
        legend_ax,
        orientation = :horizontal,
        framevisible = false,
        labelsize = 11,
        nbanks = 1,
        tellwidth = false,
        tellheight = true,
    )

    rowsize!(fig.layout, 2, Auto(0.12))
    rowgap!(fig.layout, 1, 0)

    if output !== nothing
        save(output, fig)
    end

    return fig
end

# --------------------------------------------------------------------------
# Sweep timesteps for the Full Truncation scheme, with and without the
# Brownian bridge (configs[3] / configs[4] in your list), tracking standard
# error only. Re-derives the Sobol-dimension method fallback per timestep,
# exactly like your original script (nd changes with the number of steps).
# --------------------------------------------------------------------------
function collect_fulltruncation_stderr_df_singlethread(
    model, spot, payoff, strikes, timesteps_range;
    m = 64,
    n = 1024,
    start = 0,
    mindt = 0.5 / 365,
    rng_name = :philox
)
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)

    bb_configs = Dict(
        false => AQFED.MonteCarlo.FullTruncation(false),
        true  => AQFED.MonteCarlo.FullTruncation(true),
    )

    rows = NamedTuple[]

    for bb in (false, true)
        fscheme = bb_configs[bb]

        for timesteps in timesteps_range

            config = AQFED.MonteCarlo.GridConfig(
                model, specTimes, 1.0 / timesteps, fscheme,
                mindt = min(mindt,0.5/timesteps), tstop = 0.0,
            )
            nd = AQFED.MonteCarlo.ndims(model, config)

            # Sobol dimension cap -> smaller method set for fine time grids
            names_here = nd <= 21201 ?
                [:sobol_burley, :rng, :sobol_owen, :sobol_digital, :sobol_burley4d] :
                [:sobol_burley4d, :rng]

            prices = Dict(
                strike => Dict(name => AQFED.MonteCarlo.MCConvergenceResult[] for name in names_here)
                for strike in strikes
            )

            for i_randomization in 1:m
                for name in names_here
                    rng = if rng_name == :philox
                        Random123.Philox4x(UInt32, (20130129,20100921 + i_randomization))
                    else
                     AQFED.Random.Chacha8SIMD(UInt32, 20100921 + i_randomization)
                    end
                    seq = make_sequence(name, rng, nd, n)

                    resultv = AQFED.MonteCarlo.simulateIter(
                        seq, model, spot, payoff, start, n, config, isRestart = false,
                    )

                    for (i, strike) in enumerate(strikes)
                        push!(prices[strike][name], resultv[i])
                    end
                end
            end

            for strike in strikes, name in names_here
                vals = prices[strike][name]
                se = std([AQFED.MonteCarlo.last_mean(r) / sqrt(m) for r in vals])

                push!(rows, (
                    dimension = timesteps,
                    method    = String(name),
                    bb        = bb,
                    strike    = strike,
                    std_err   = se,
                ))
            end

            @info "swept" bb timesteps nd nmethods = length(names_here)
        end
    end

    return DataFrame(rows)
end


# --------------------------------------------------------------------------
# Plot: standard error vs. number of time steps, one panel per BB setting,
# one curve per method -- same visual language as the earlier convergence
# plots (wong_colors, distinct markers, log-log, minor grid, clean legend).
# --------------------------------------------------------------------------
function plot_fulltruncation_stderr(df; strike = 1.0, output = nothing)

    fig = Figure(size = (1000, 450), fontsize = 14)
    colors  = Makie.wong_colors()
    markers = [:circle, :diamond, :rect, :utriangle, :dtriangle, :star5, :hexagon, :cross]
    labelmap = Dict("rng" => "ChaCha-8", "sobol_burley" => "Burley-HD", "sobol_burley4d" => "Burley-4D", "sobol_digital"=>"digital shift", "sobol_owen"=>"ACM-823")
    d = df[df.strike .== strike, :]
    methods = sort(unique(d.method))

    function make_panel!(fig, position, bb_flag, title)

        ax = Axis(
            fig[1, position],
            xscale = log10,
            #yscale = log10,
            xlabel = "Number of time steps",
            ylabel = position == 1 ? "Standard error" : "",
            title = title,
            xminorticksvisible = true,
            xminorgridvisible = true,
            yminorticksvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(9),
            yminorticks = IntervalsBetween(9),
        )

        sub_bb = d[d.bb .== bb_flag, :]

        for (i, method) in enumerate(methods)
            s = sort(sub_bb[sub_bb.method .== method, :], :dimension)
            isempty(s) && continue

            lines!(
                ax, s.dimension, s.std_err;
                color = colors[mod1(i, length(colors))],
                linewidth = 2.0,
                label = labelmap[method],
            )
        end

        return ax
    end

    ax1 = make_panel!(fig, 1, false, "Full Truncation (no BB)")
    ax2 = make_panel!(fig, 2, true,  "Full Truncation (with BB)")
    legend_ax = length(ax2.scene.plots) >= length(ax1.scene.plots) ? ax2 : ax1
 
    Legend(
        fig[2, 1:2],
        legend_ax,
        orientation = :horizontal,
        framevisible = false,
        labelsize = 11,
        nbanks = 1,
        tellwidth = false,
        tellheight = true,
    )
 
    rowsize!(fig.layout, 2, Auto(0.12))
 

    if output !== nothing
        save(output, fig)
    end

    return fig
end

#=

# --------------------------------------------------------------------------
# Run the sweep.
# NOTE ON COST: length(timesteps_range) * 2 (bb) * m * |methods| simulate
# calls, each over n=1024 paths. With ~150 timesteps and m=64 that's on the
# order of 150 * 2 * 64 * 5 ≈ 96,000 simulateIter calls. Start with a
# smaller m (e.g. 8-16) and/or fewer timestep points to sanity-check timing
# before committing to the full sweep.
# --------------------------------------------------------------------------
CSV.write(
           "/home/cian/mypapers/eqd_book/heston_ft_bb_nobb_1024_lowvol.csv",
           df_fulltrunc,
       )

timesteps_range = log_timesteps(8, 1000, 50)
@info "sweeping $(length(timesteps_range)) timestep values"

df_fulltrunc = collect_fulltruncation_stderr_df(
    model, spot, payoff, strikes, timesteps_range;
    m = 32,
    n = 1024,
)

fig = plot_fulltruncation_stderr(
    df_fulltrunc;
    strike = 1.0,
    output = "heston_fulltrunc_stderr_bb.pdf",
)
display(fig)
=#

# If you'd rather have two fully separate files instead of a two-panel
# figure, just save each panel's Figure on its own, e.g.:
#
#   fig_nobb = plot_fulltruncation_stderr(df_fulltrunc[df_fulltrunc.bb .== false, :])
#   fig_bb   = plot_fulltruncation_stderr(df_fulltrunc[df_fulltrunc.bb .== true, :])

@testset "Heston RQMC StdError Vanilla" begin
       
    τ = 10.0
    spot=1.0
    hParams = CharFuncPricing.HestonParams(0.133, 0.35, 0.321, -0.63, 1.388)
     hParams = CharFuncPricing.HestonParams(0.0313, 0.111927, 0.062762, -0.755929, 0.126743) #book params: is feller violated?
   model = AQFED.TermStructure.HestonModel(hParams.v0, hParams.κ, hParams.θ, hParams.ρ, hParams.σ, 0.0, 0.0)
    
    pricer = makeCosCharFuncPricer(CharFuncPricing.DefaultCharFunc(hParams), τ, 1024*4, 32)
    strikes = [0.2,0.8, 1.0, 1.2]
   	payoff = AQFED.MonteCarlo.ListPayoff([AQFED.MonteCarlo.VanillaOption(true, spot*K, AQFED.MonteCarlo.BulletCashFlow(τ, τ, false, 0.0), 0.0) for K in strikes])
    
    refValues = [priceEuropean(pricer, true, strike, 1.0,τ,1.0) for strike in strikes]

    configs = [
        AQFED.MonteCarlo.QE(false, true),
        AQFED.MonteCarlo.QE(true, true),
        AQFED.MonteCarlo.FullTruncation(false),
        AQFED.MonteCarlo.FullTruncation(true),
    ]
timesteps = 365
specTimes = AQFED.MonteCarlo.specificTimes(payoff)
config = AQFED.MonteCarlo.GridConfig(
    model,
    specTimes,
    1.0 / timesteps,
    configs[3],
    mindt = 0.5 / 365,
    tstop = 0.0,
)

nd = AQFED.MonteCarlo.ndims(model,config) #should nd be part of config?

sequence_names = if nd <= 21201
 [
    :sobol_burley,
    :rng,
    :sobol_owen,
    :sobol_digital,
    :sobol_burley4d,
]
else 
    [:sobol_burley4d,:rng]
end


# strike -> sequence -> vector over randomizations
prices = Dict(
    strike => Dict(name => AQFED.MonteCarlo.MCConvergenceResult[] for name in sequence_names)
    for strike in strikes
)

n = 1024
start = 0
m=64
for i_randomization = 1:m
    for name in sequence_names
rng = AQFED.Random.Chacha8SIMD(UInt32, 20100921 + i_randomization)
        isRestart = false
        seq = if name == :rng
            AbstractRNGSeq(rng, nd)
        elseif name == :sobol_owen
            ScrambledSobolSeq(nd, n, OwenFaureTezuka(31, ScramblingRngAdapter(rng)))
        elseif name == :sobol_digital
            ScrambledSobolSeq(nd, n, Digital{nd}(rng))
        elseif name == :sobol_burley4d
            AQFED.Random.BurleySobolSeq(nd, 
            #AQFED.Random.MurmurRNG(20300921 + i_randomization))
            rng)
        elseif name == :sobol_burley
                 ScrambledSobolSeq(nd, n, Burley{nd}(rng))
        elseif name == :sobol_equi
            myseq = ScrambledSobolSeq(nd, n, Digital{nd}(rng))
            myseq.shift = round.(UInt32, collect(0:(nd-1))./nd * 2^32)
            myseq.x .= myseq.shift
            myseq
        elseif name == :sobol
            ScrambledSobolSeq(nd, n, NoScrambling())
        end

        resultv = AQFED.MonteCarlo.simulateIter(            seq,            model,            spot,            payoff,            start,            n, config, isRestart=isRestart)
        # values[k] = price corresponding to strikes[k]
        # stddevs[k] = MC standard error corresponding to strikes[k]
        for (i, strike) in enumerate(strikes)
            push!(prices[strike][name], resultv[i])

            #@test isapprox(values[i], refValues[i]; atol = 1e-2)
        end
    end
end


means = Dict(
    strike => Dict(
        name => mean([AQFED.MonteCarlo.last_mean(result) for result in prices[strike][name]])
        for name in sequence_names
    )
    for strike in strikes
)

stderrs = Dict(
    strike => Dict(
        name => std([AQFED.MonteCarlo.last_mean(result) / sqrt(m) for result in prices[strike][name]])
        for name in sequence_names
    )
    for strike in strikes
)


end


@testset "Heston RQMC Convergence Vanilla" begin


    τ = 10.0
    spot=1.0
    hParams = CharFuncPricing.HestonParams(0.133, 0.35, 0.321, -0.63, 1.388)
    # hParams = CharFuncPricing.HestonParams(0.0313, 0.111927, 0.062762, -0.755929, 0.126743) #book params: is feller violated?
   model = AQFED.TermStructure.HestonModel(hParams.v0, hParams.κ, hParams.θ, hParams.ρ, hParams.σ, 0.0, 0.0)
    
    pricer = makeCosCharFuncPricer(CharFuncPricing.DefaultCharFunc(hParams), τ, 1024*16, 32)
    strikes = [0.8, 1.0, 1.2]
   	payoff = AQFED.MonteCarlo.ListPayoff([AQFED.MonteCarlo.VanillaOption(true, spot*K, AQFED.MonteCarlo.BulletCashFlow(τ, τ, false, 0.0), 0.0) for K in strikes])
    
    refValues = [priceEuropean(pricer, true, strike, 1.0,τ,1.0) for strike in strikes]

    configs = [
        AQFED.MonteCarlo.QE(false, true),
        AQFED.MonteCarlo.QE(true, true),
        AQFED.MonteCarlo.FullTruncation(false),
        AQFED.MonteCarlo.FullTruncation(true),
    ]
timesteps = 365
specTimes = AQFED.MonteCarlo.specificTimes(payoff)
config = AQFED.MonteCarlo.GridConfig(
    model,
    specTimes,
    1.0 / timesteps,
    configs[2],
    mindt = 0.5 / 365,
    tstop = 0.0,
)

nd = AQFED.MonteCarlo.ndims(config)

sequence_names = [
      :sobol_burley,
       :sobol,    :rng,
           :sobol_digital,
           :sobol_acm823,
           #:sobol_equi,
       ]
results = Dict{Symbol, Vector{AQFED.MonteCarlo.MCConvergenceResult}}()
n = 1024*1024*4
for name in sequence_names

           rng = AQFED.Random.Chacha8SIMD(
               UInt32,
               20100921,
           )

           seq =
               if name == :rng
start = 0
                   AQFED.Random.AbstractRNGSeq(
                       rng,
                       nd,
                   )
               elseif name == :sobol_burley
                start = 0
                AQFED.Random.BurleySobolSeq(nd, rng)
               elseif name == :sobol_burley_inspired
start = 0
                   AQFED.Random.ScrambledSobolSeq(
                       nd,
                       n,
                       AQFED.Random.Burley{nd}(rng),
                   )
               elseif name == :sobol_acm823
                start = 0
                AQFED.Random.ScrambledSobolSeq(
                    nd, n, OwenFaureTezuka(31, ScramblingRngAdapter(rng))
                )

               elseif name == :sobol_digital
start = 0
                   AQFED.Random.ScrambledSobolSeq(
                       nd,
                       n,
                       AQFED.Random.Digital{nd}(rng),
                   )

               elseif name == :sobol_equi
start = 1
                   # Gurrieri et al. equidistant digital scrambling
                   seq = AQFED.Random.ModuloSobolSeq(
                       nd,
                       n + 1,
                       rng,
                   )

                   seq.upoint =
                           collect(0:(nd - 1)) ./ nd

                   seq

               elseif name ==:sobol
      start = 1
                   AQFED.Random.ScrambledSobolSeq(nd, n, AQFED.Random.NoScrambling())
               else
                   error("Unknown sequence: $name")
               end

           resultv = AQFED.MonteCarlo.simulateIter(
               seq,
               model,
               spot,
               payoff,
               start,
               n,
               config,
               isRestart = false,
           )

           # ATM strike = strikes[2]
           results[name] = resultv
       end
end

#=
save:
using DataFrames
using CSV

rows = NamedTuple[]

for (name, resultvec) in results
    for (i, result) in enumerate(resultvec)

        for j in eachindex(result.samples)

            push!(
                rows,
                (
                    Method   = String(name),
                    Strike   = strikes[i],
                    N        = result.samples[j],
                    Price    = result.means[j],
                    Error    = result.means[j] - refValues[i],
                    AbsError = abs(result.means[j] - refValues[i]),
                    StdErr   = result.stderrs[j],
                )
            )
        end
    end
end

df = DataFrame(rows)
CSV.write(
    "/home/cian/mypapers/eqd_book/heston_qe_qmc_convergence.csv",
    df,
)
#QE low vol-of-vol
df = CSV.read(
    "/home/cian/mypapers/eqd_book/heston_qe_qmc_convergence.csv",
    DataFrame,
)
#FT low vol-of-vol

#FT high vol-of-vol

#QE high vol-of-vol


metadata = Dict(
    "experiment" => "Heston QE RQMC convergence",
    "spot"       => spot,
    "maturity"   => τ,
    "timesteps"  => timesteps,
    "n_max"      => n,
    "dimensions" => nd,
    "v0"         => hParams.v0,
    "kappa"      => hParams.κ,
    "theta"      => hParams.θ,
    "rho"        => hParams.ρ,
    "sigma"      => hParams.σ,
    "ref_price_K08" => refValues[1],
    "ref_price_K10" => refValues[2],
    "ref_price_K12" => refValues[3],
    "seed"       => 20100921,
)
    using JSON3

JSON3.write(
    "/home/cian/mypapers/eqd_book/heston_qe_qmc_metadata.json",
    metadata,
)
using CairoMakie
using DataFrames

CairoMakie.activate!()

function plot_qe_qmc_convergence(
    df;
    strike = 1.0,
    output = nothing,
        method_order = [
        "rng",
        "sobol_burley",
        "sobol_digital",
        "sobol_equi",
        "sobol",
    ]
)

    # ------------------------------------------------------------
    # Select strike
    # ------------------------------------------------------------

    d = df[df.Strike .== strike, :]



    labels = Dict(
        "rng"           => "ChaCha8",
        "sobol_burley"  => "Sobol Burley",
        "sobol_digital" => "Sobol digital shift",
       "sobol_equi"    => "Sobol Gurrieri shift",
        "sobol" => "Sobol"
     )

    markers = Dict(
        "rng"           => :circle,
        "sobol_burley"  => :rect,
        "sobol_digital" => :diamond,
        "sobol_equi"    => :utriangle,
        "sobol" => :dtriangle,
    )

    colors = Makie.wong_colors()

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig = Figure(
        size = (760, 480),
        fontsize = 14,
    )

    ax = Axis(
        fig[1, 1],
        xscale = log10,
        yscale = log10,
        xlabel = "Number of paths",
        ylabel = "Absolute pricing error",
        xminorticksvisible = true,
        yminorticksvisible = true,
        title = "Heston QE-M, strike=$(strike*100)%",
    )

    # ------------------------------------------------------------
    # Curves
    # ------------------------------------------------------------

    for (i, method) in enumerate(method_order)
         dm = sort(
            d[d.Method .== method, :],
            :N,
        )

        scatterlines!(
            ax,
            dm.N,
            dm.AbsError;
            color = colors[i],
            linewidth = 2.4,
                        marker = markers[method],
            markersize = 8,
               label = labels[method],
        )


    end

    # ------------------------------------------------------------
    # MC standard error
    # ------------------------------------------------------------

    mc = sort(
        d[d.Method .== "rng", :],
        :N,
    )

    band!(
        ax,
        mc.N,
        mc.AbsError,
        mc.AbsError .+ mc.StdErr;
        color = (colors[1], 0.15),
    )

    lines!(
        ax,
        mc.N,
        mc.StdErr;
        color = (:black, 0.6),
        linewidth = 1.5,
        linestyle = :dash,
           label = "MC standard error",
    )

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------
axislegend(
        ax;
        position = :rt,
        framevisible = false,
    )


    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    if output !== nothing
        save(output, fig)
    end

    return fig
end

Convergence plot:
function plot_qe_qmc_convergence(
    results,
    ref_price;
    strike_index = 2,
    output = nothing,
    title = nothing,
)

    fig = Figure(
        size = (760, 480),
        fontsize = 14,
    )

    ax = Axis(
        fig[1, 1],
        xscale = log10,
        yscale = log10,
        xlabel = "Number of paths",
        ylabel = "Absolute pricing error",
        xminorticksvisible = true,
        yminorticksvisible = true,
    )

    names = [
        :rng,
        :sobol_burley,
        :sobol_digital,
        :sobol_equi,
        :sobol,
    ]

    labels = Dict(
        :rng           => "ChaCha8",
        :sobol_burley  => "Sobol + Burley",
        :sobol_digital => "Sobol + digital shift",
        :sobol_equi    => "Sobol + Gurrieri shift",
        :sobol => "Sobol"
    )

    colors = Makie.wong_colors()
    markers = [:circle, :rect, :diamond, :utriangle, :dtriangle]

    for (i, name) in enumerate(names)
        if haskey(results, name)
        result = results[name][strike_index]

        N = result.samples
        error = abs.(result.means .- ref_price)

        scatterlines!(
            ax, N, error;
            color = colors[i],
            linewidth = 2.5,
            label = labels[name],
            marker = markers[i],
            markersize = 8,
        )
        end
    end

    # MC standard error and its band
    rng_result = results[:rng][strike_index]

    N = rng_result.samples
    error = abs.(rng_result.means .- ref_price)
    stderr = rng_result.stderrs

    band!(
        ax,
        N,
        error,
        error .+ stderr,
        color = (colors[1], 0.15),
    )

    lines!(
        ax,
        N,
        stderr;
        color = (:black, 0.6),
        linewidth = 1.5,
        linestyle = :dash,
        label = "MC standard error",
    )

    axislegend(
        ax;
        position = :rt,
        framevisible = false,
    )

    xlims!(
        ax,
        first(N) / 1.1,
        last(N) * 1.1,
    )

    if title !== nothing
        ax.title = title
    end

    if output !== nothing
        save(output, fig)
    end

    return fig
end


using CairoMakie

function plot_qe_ft_qmc(
    df_qe_low,
    df_ft_low,
    df_qe_high,
    df_ft_high;
    strike = 1.0,
    output = nothing,
)

    fig = Figure(
        size = (920, 450),
        fontsize = 13,
    )

    colors = Makie.wong_colors()

    labels = [
        "ChaCha8",
        "Sobol",
        "Sobol + Burley",
        "FT + Burley",
    ]

    markers = [
        :circle,
        :diamond,
        :rect,
        :utriangle,
    ]

    function make_panel!(
        fig,
        position,
        df_qe,
        df_ft,
        title,
    )

        # Restrict to the requested strike and N
        dqe = df_qe[
            (df_qe.Strike .== strike) .& (df_qe.N .>= 512),
            :
        ]

        dft = df_ft[
            (df_ft.Strike .== strike) .& (df_ft.N .>= 512),
            :
        ]

ax = Axis(
    fig[1, position],
    xscale = log10,
    yscale = Makie.Symlog10(1e-3),
    xlabel = "Number of paths",
    ylabel = position == 1 ? "Pricing error" : "",
    title = title,
    xminorticksvisible = true,
    yminorticksvisible = true,
    yticks = (
        [-0.02, -0.01, -0.001, 0.0, 0.001, 0.01, 0.02],
        ["-0.02", "-0.01", "-0.001", "0", "0.001", "0.01", "0.02"],
    ),
)
        hlines!(
            ax,
            0.0;
            color = (:black, 0.45),
            linewidth = 1.0,
        )

        # QE: PRNG, Sobol, Sobol + Burley
        for (i, method) in enumerate([
            "rng",
            "sobol",
            "sobol_burley",
        ])

            d = sort(
                dqe[dqe.Method .== method, :],
                :N,
            )

            scatterlines!(
                ax,
                d.N,
                d.Error;
                color = colors[i],
                linewidth = 2.2,
                marker = markers[i],
                markersize = 7,
                label = labels[i],
            )
        end

        # FT + Burley
        d = sort(
            dft[dft.Method .== "sobol_burley", :],
            :N,
        )

        scatterlines!(
            ax,
            d.N,
            d.Error;
            color = colors[4],
            linewidth = 2.2,
            marker = markers[4],
            markersize = 7,
            label = labels[4],
        )

        # --------------------------------------------------------
        # MC standard-error band around the signed ChaCha8 error
        # --------------------------------------------------------

        rng = sort(
            dqe[dqe.Method .== "rng", :],
            :N,
        )

        band!(
            ax,
            rng.N,
            rng.Error .- rng.StdErr,
            rng.Error .+ rng.StdErr;
            color = (colors[1], 0.15),
        )

        if position == 2
            axislegend(
                ax;
                position = :rt,
                framevisible = false,
            )
        end

        return ax
    end

    make_panel!(
        fig,
        1,
        df_qe_low,
        df_ft_low,
        "Low vol-of-vol",
    )

    make_panel!(
        fig,
        2,
        df_qe_high,
        df_ft_high,
        "High vol-of-vol",
    )

    if output !== nothing
        save(output, fig)
    end

    return fig
end
fig = plot_qe_ft_qmc(
    df_qe_low,
    df_ft_low,
    df_qe_high,
    df_ft_high;
    strike = 1.0,
    output = "/home/cian/mypapers/eqd_book/heston_qe_ft_qmc_atm.pdf",
)
using CairoMakie
function plot_qe_ft_qmc_linear_dense(
    df_qe_low,
    df_ft_low,
    df_qe_high,
    df_ft_high;
    strike = 1.0,
    output = nothing,
)

    fig = Figure(
        size = (920, 450),
        fontsize = 13,
    )

    colors = Makie.wong_colors()
    labels = ["ChaCha8", "Sobol", "Sobol + Burley", "Sobol + ACM823", "FT + Burley"]
    markers = [:circle, :diamond, :rect, :utriangle, :dtriangle]

    function make_panel!(fig, position, df_qe, df_ft, title)

        # Restrict data to requested strike and N >= 512
        dqe = df_qe[(df_qe.Strike .== strike) .& (df_qe.N .>= 1000), :]
        dft = df_ft[(df_ft.Strike .== strike) .& (df_ft.N .>= 1000), :]

        # --------------------------------------------------------
        # Configured for a 0.01 linear view with dense ticks
        # --------------------------------------------------------
        ax = Axis(
            fig[1, position],
            xscale = log10,
            yscale = identity,  
            xlabel = "Number of paths",
            ylabel = position == 1 ? "Pricing error" : "",
            title = title,
            xminorticksvisible = true,
            yminorticksvisible = true,
            # Explicit step-by-step linear ticks down to 0.002 intervals up to 0.01
            yticks = (
                [-0.01, -0.008, -0.006, -0.004, -0.002, 0.0, 0.002, 0.004, 0.006, 0.008, 0.01],
                ["-0.01", "-0.008", "-0.006", "-0.004", "-0.002", "0", "0.002", "0.004", "0.006", "0.008", "0.01"]
            )
        )
        
        hlines!(ax, 0.0; color = (:black, 0.45), linewidth = 1.0)

        # Plot QE Methods (ChaCha8, Sobol, Sobol + Burley)
        for (i, method) in enumerate(["rng", "sobol", "sobol_burley","sobol_acm823"])
            d = sort(dqe[dqe.Method .== method, :], :N)
            scatterlines!(
                ax, d.N, d.Error;
                color = colors[i], linewidth = 2.2,
                marker = markers[i], markersize = 7, label = labels[i],
            )
        end

        # --------------------------------------------------------
        # FT + Burley (Explicitly indexing element 4)
        # --------------------------------------------------------
        d_ft = sort(dft[dft.Method .== "sobol_burley", :], :N)
        scatterlines!(
            ax, d_ft.N, d_ft.Error;
            color = colors[end],       # Fixed: Extracted individual scalar color
            linewidth = 2.2,
            marker = markers[end],     # Fixed: Extracted individual scalar marker
            markersize = 7,
            label = labels[end],       # Fixed: Extracted individual scalar label
        )

        # --------------------------------------------------------
        # 95% MC confidence-interval band around ChaCha8 error
        # --------------------------------------------------------
        rng = sort(dqe[dqe.Method .== "rng", :], :N)
        band!(
            ax, rng.N,
            rng.Error .- (1.96 .* rng.StdErr),
            rng.Error .+ (1.96 .* rng.StdErr);
            color = (colors[1], 0.15),
        )

        # Set explicit y-limits matching our wider 0.01 boundaries
        ylims!(ax, -0.011, 0.011)

        if position == 2
            axislegend(ax; position = :rt, framevisible = false)
        end

        return ax
    end

    make_panel!(fig, 1, df_qe_low, df_ft_low, "Low vol-of-vol")
    make_panel!(fig, 2, df_qe_high, df_ft_high, "High vol-of-vol")

    if output !== nothing
        save(output, fig)
    end

    return fig
end

# Generate and save the plot
fig = plot_qe_ft_qmc_linear_dense(
    df_qe_low, df_ft_low, df_qe_high, df_ft_high;
    strike = 1.0,
    output = "/home/cian/mypapers/eqd_book/heston_qe_ft_qmc_atm_linear.pdf",
)

display(fig)
plot_qe_qmc_convergence(
    results,
    refValues[1];
    strike_index = 2,
    title = "ITM, K=0.8",
    output = "/home/cian/mypapers/eqd_book/heston_qe_qmc_atm.pdf",
)
plot_itm_otm(results, refValues,output = "/home/cian/mypapers/eqd_book/heston_qe_qmc_itm_atm.pdf")
=#

@testset "Heston RQMC StdError Cliquet" begin
    τ = 10.0
    spot=1.0
    hParams = CharFuncPricing.HestonParams(0.133, 0.35, 0.321, -0.63, 1.388)
 
obsTimes = collect(range(0.0/12, 1.0, length = 13))
strikes = [0.01, 0.04]
payoff = AQFED.MonteCarlo.ListPayoff([AQFED.MonteCarlo.CliquetOption(
                               cap, -1.0, 1e100, 0.0, 0.0,
                               AQFED.MonteCarlo.PeriodicCashFlow(obsTimes, obsTimes[end], 1, 0.0),
                               0.0) for cap in strikes])

                                model = AQFED.TermStructure.HestonModel(hParams.v0, hParams.κ, hParams.θ, hParams.ρ, hParams.σ, 0.0, 0.0)
    
    pricer = makeCosCharFuncPricer(CharFuncPricing.DefaultCharFunc(hParams), τ, 1024, 32)

    refValues = [priceEuropean(pricer, true, strike, 1.0,τ,1.0) for strike in strikes]

timesteps = 365
specTimes = AQFED.MonteCarlo.specificTimes(payoff)
config = AQFED.MonteCarlo.GridConfig(
    model,
    specTimes,
    1.0 / timesteps,
    AQFED.MonteCarlo.FullTruncation(true),
    mindt = 0.5 / 365,
    tstop = 0.0,
)

nd = AQFED.MonteCarlo.ndims(config)

sequence_names = [
    :rng,
    :sobol_owen,
    :sobol_digital,
    :sobol_burley,
]

# strike -> sequence -> vector over randomizations
prices = Dict(
    strike => Dict(name => Float64[] for name in sequence_names)
    for strike in strikes
)

# Same thing for the individual-run standard errors
run_stderrs = Dict(
    strike => Dict(name => Float64[] for name in sequence_names)
    for strike in strikes
)
n = 16*1024
start = 0
m=64
for i_randomization = 1:m
    for name in sequence_names
rng = AQFED.Random.Chacha8SIMD(UInt32, 2020 + i_randomization)
        isRestart = false
        seq = if name == :rng
            AbstractRNGSeq(rng, nd)
        elseif name == :sobol_owen
            ScrambledSobolSeq(nd, n, OwenFaureTezuka(31, ScramblingRngAdapter(rng)))
        elseif name == :sobol_digital
            ScrambledSobolSeq(nd, n, Digital{nd}(rng))
        elseif name == :sobol_burley
            ScrambledSobolSeq(nd, n, Burley{nd}(rng))
        end

        values, stddevs = AQFED.MonteCarlo.simulateIter(            seq,            model,            spot,            payoff,            start,            n,
            config, isRestart=isRestart)
        # values[k] = price corresponding to strikes[k]
        # stddevs[k] = MC standard error corresponding to strikes[k]
        for (i, strike) in enumerate(strikes)
            push!(prices[strike][name], values[i])
            push!(run_stderrs[strike][name], stddevs[i])

            #@test isapprox(values[i], refValues[i]; atol = 1e-2)
        end
    end
end


means = Dict(
    strike => Dict(
        name => mean(prices[strike][name])
        for name in sequence_names
    )
    for strike in strikes
)

stderrs = Dict(
    strike => Dict(
        name => std(prices[strike][name]) / sqrt(m)
        for name in sequence_names
    )
    for strike in strikes
)

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

    # Plot convergence (Standard Error vs N)
    output_plot_se = joinpath(@__DIR__, "heston_asian_rqmc_convergence.png")
    fig_se = plot_asian_rqmc_convergence(df_res; output = output_plot_se)
    @test isfile(output_plot_se)

    # Alternate Plot (Price vs N with PRNG StdErr band)
    output_plot_price = joinpath(@__DIR__, "heston_asian_rqmc_prices.png")
    fig_price = plot_asian_rqmc_prices(df_res; output = output_plot_price)
    @test isfile(output_plot_price)

    # Assert that at largest path count (n=32768), RQMC methods achieve lower standard error than PRNG
    max_n = maximum(df_res.n_paths)
    df_max = df_res[df_res.n_paths .== max_n, :]
    prng_se = df_max[df_max.method .== "rng", :std_err][1]
    burley_se = df_max[df_max.method .== "sobol_burley", :std_err][1]
    burley4d_se = df_max[df_max.method .== "sobol_burley4d", :std_err][1]

    @test burley_se < prng_se
    @test burley4d_se < prng_se
end