# Chapter 6: randomized quasi Monte-Carlo. Scrambling of the Sobol' sequence against the dimension, on a product integral and on a
# geometric Brownian motion, and improvement brought by the Brownian bridge on an Asian option.
# The computations come from the scripts burley_owen_function_experiment_*.jl, burley_owen_gbm_experiment_gemini.jl,
# asian_rqmc_aqfed_experiment.jl and plot_geom_asian_std.py. The figures only read the CSV files of book/data.

using AQFED, Statistics, Printf
import Random123
import SpecialFunctions: erfinv

const RQMC_SEED = (20130129, 20109021)

"""
Sequence `method` of dimension D for N points, replicate `rep`. `:digital` is DigitalSobolSeq while `:digital_xor` is the digital
scrambling inside ScrambledSobolSeq: the two are not the same stream, the product integral was run on the first, the GBM on the second.
"""
function rqmcSequence(D::Int, N::Int, rep::Int; method::Symbol=:burley, seed=(3, 14), directions=AQFED.Random.joe_kuo)
    R = AQFED.Random
    rng = Random123.Philox4x(UInt32, (seed[1], seed[2] + rep - 1))
    method == :digital && return R.DigitalSobolSeq(D, N, rng, directions)
    method == :digital_xor && return R.ScrambledSobolSeq(D, N, R.Digital{D}(rng), directions)
    method == :modulo && return R.ModuloSobolSeq(D, N, rng, directions)
    method == :burley4d && return R.BurleySobolSeq(D, rng)
    method == :burley && return R.ScrambledSobolSeq(D, N, R.Burley{D}(rng), directions)
    method == :owen_faure_tezuka && return R.ScrambledSobolSeq(D, N, R.OwenFaureTezuka(31, R.ScramblingRngAdapter(rng)), directions)
    method == :sobol && return R.ScrambledSobolSeq(D, N, R.NoScrambling(), directions)
    method == :philox && return R.AbstractRNGSeq(rng, D)
    method == :philox64 && return R.AbstractRNGSeq(Random123.Philox4x(UInt64, (seed[1], seed[2] + rep - 1)), D)
    throw(DomainError(method, "unknown sequence"))
end

function rqmcIntegrate(f, D::Int, N::Int, rep::Int; kwargs...)
    seq = rqmcSequence(D, N, rep; kwargs...)
    x = Vector{Float64}(undef, D)
    total = 0.0
    @inbounds for _ in 1:N
        AQFED.Random.next!(seq, x)
        total += f(x)
    end
    return total / N
end

#the replicates of all the (method, dimension, N) combinations, on all the threads
function rqmcEstimates(f; methods, dimensions, Ns, R, seed, directions=AQFED.Random.joe_kuo, renamed=Dict{Symbol,Symbol}())
    tasks = [(method=m, dimension=D, N=N, replicate=rep) for m in methods for D in dimensions for N in Ns for rep in 1:R]
    #the costly tasks first, they are stolen by the idle threads
    tasks = tasks[sortperm(tasks, by=t -> -t.dimension * t.N)]
    estimates = Vector{Float64}(undef, length(tasks))
    Threads.@threads :dynamic for k in eachindex(tasks)
        t = tasks[k]
        estimates[k] = rqmcIntegrate(f, t.dimension, t.N, t.replicate; method=t.method, seed=seed, directions=directions)
    end
    df = DataFrame(method=[String(get(renamed, t.method, t.method)) for t in tasks], dimension=[t.dimension for t in tasks],
        N=[t.N for t in tasks], replicate=[t.replicate for t in tasks], estimate=estimates)
    return sort!(df, [:method, :dimension, :N, :replicate])
end

## product integral: I_d = ∏(1 + a(x_i - 1/2)) has an exact value of 1, the effect of the dimension without any payoff or path construction

productIntegrand(x) = prod(xi -> 1 + 0.01 * (xi - 0.5), x)

function burleyDirectionsResults(directions, file)
    df = rqmcEstimates(productIntegrand; methods=(:philox, :sobol, :digital, :burley, :owen_faure_tezuka), dimensions=1:5000, Ns=[2^15], R=1,
        seed=(3, 14), directions=directions)
    df.absolute_error = abs.(df.estimate .- 1)
    writetable(file, df)
end

#about a day of computation
@compute "burley_directions" begin
    burleyDirectionsResults(AQFED.Random.niederreiter_nut_21201, "burley_owen_raw_results_harase.csv")
    burleyDirectionsResults(AQFED.Random.joe_kuo, "burley_owen_raw_results_32k.csv")
end

@figure "burley_owen_function_error_vs_dimension_combined" begin
    ymin, ymax = 1e-8, 1e-2
    methods = [("philox", "Philox"), ("sobol", "Sobol"), ("owen_faure_tezuka", "ACM-823"), ("burley", "Burley"), ("digital", "Digital shift")]
    panels = [("burley_owen_raw_results_harase.csv", "Direction: niederreiter-nut-s21201"), ("burley_owen_raw_results_32k.csv", "Direction: new-joe-kuo-6.21201")]
    fig = bookFigure(size=(800, 400))
    axes = map(enumerate(panels)) do (position, (file, title))
        df = readcsv(file)
        ax = bookAxis(fig[1, position], xlabel="d", ylabel=position == 1 ? "Absolute error" : "", title=title, yscale=log10,
            yticks=decadeTicks(-8:2:-2), yminorticksvisible=true)
        for (i, (method, label)) in enumerate(methods)
            d = sort(df[df.method.==method, :], :dimension)
            lines!(ax, d.dimension, clamp.(d.absolute_error, ymin, ymax), color=PALETTE[i], label=label)
        end
        ylims!(ax, ymin, ymax)
        ax
    end
    linkyaxes!(axes...)
    Legend(fig[2, 1:2], axes[1], orientation=:horizontal, framevisible=false)
    fig
end

## geometric Brownian motion with an incremental construction: E[S_T] = 1 with a volatility of 100%, one dimension per time-step

quantileNormal(x::Float64) = sqrt(2.0) * erfinv(2.0 * clamp(x, 2.0^-33, 1.0 - 2.0^-33) - 1.0)
gbmTerminalValue(x) = exp(-0.5 + sqrt(1.0 / length(x)) * sum(quantileNormal, x))

function rqmcAggregate(raw)
    result = combine(groupby(raw, [:method, :dimension, :N])) do sub
        m, sd = mean(sub.estimate), std(sub.estimate)
        (estimate=m, error=m - 1, abs_error=abs(m - 1), replicate_sd=sd, std_err=sd / sqrt(nrow(sub)), R=nrow(sub))
    end
    return sort!(result, [:method, :dimension, :N])
end

@compute "gbm_convergence" begin
    dimensions = unique(round.(Int, exp10.(range(0, 4, length=200))))
    for (N, tag) in ((1024, "1k"), (16384, "16k"))
        raw = rqmcEstimates(gbmTerminalValue; methods=(:philox, :digital_xor, :burley, :burley4d, :owen_faure_tezuka), dimensions=dimensions, Ns=[N], R=32,
            seed=RQMC_SEED, renamed=Dict(:digital_xor => :digital))
        writetable("gemini_gbm_aggregated_results_$(tag).csv", rqmcAggregate(raw))
    end
end

#dense curve with sparse markers, for the black and white print of the book (style of qe_kappa_profile_compare)
function markedLines!(ax, x, y; color, marker, label, count=14)
    lines!(ax, x, y, color=color)
    idx = round.(Int, range(1, length(x), length=count))
    scatter!(ax, x[idx], y[idx], color=color, marker=marker, markersize=10, strokecolor=:white, strokewidth=0.8, label=label)
end

const RQMC_MARKERS = [:circle, :rect, :utriangle, :diamond, :rtriangle, :dtriangle]

@figure "gbm_convergence_16k1k" begin
    methods = [("burley", "Burley(HDD)"), ("burley4d", "Burley(4DP)"), ("digital", "digital shift"), ("owen_faure_tezuka", "ACM-823"), ("philox", "Philox4x32")]
    fig = bookFigure(size=(800, 400))
    for (position, (tag, N)) in enumerate((("1k", 1024), ("16k", 16384)))
        df = readcsv("gemini_gbm_aggregated_results_$(tag).csv")
        ax = bookAxis(fig[1, position], xlabel="Number of Euler time steps", ylabel=position == 1 ? "Standard error" : "", title="N = $N", xscale=log10,
            xticks=decadeTicks(0:4), xminorticksvisible=true, xminorticks=IntervalsBetween(9), yminorticksvisible=true)
        for (i, (method, label)) in enumerate(methods)
            d = sort(df[df.method.==method, :], :dimension)
            markedLines!(ax, d.dimension, d.std_err, color=PALETTE[i], marker=RQMC_MARKERS[i], label=label)
        end
        position == 1 && Legend(fig[2, 1:2], ax, orientation=:horizontal, framevisible=false)
    end
    fig
end

## Euler simulation of the hyperbolic local volatility model of Jäckel (2006). With β = 1 this is a Black model.

function hyperbolicLocalVol(S, ν, β)
    s = max(S, eps(Float64))
    a = (1 - β + β^2) / β * s
    b = (β - 1) / β
    return ν * (a + b * (sqrt(max(s^2 + β^2 * (1 - s)^2, 0.0)) - β))
end

"""
Discounted payoff on the paths U[dimension, path], with `kind` one of :geometric (average of the Asian option) or :terminal.
"""
function hyperbolicPayoff(U::Matrix{Float64}; S0, r, T, ν, β, K, bridge::Bool, kind::Symbol)
    d, n = size(U)
    h = T / d
    bb = bridge ? AQFED.MonteCarlo.BrownianBridgeConstruction(collect(range(h, T; length=d))) : nothing
    z, dw = zeros(d), zeros(d)
    sum = 0.0
    @inbounds for p in 1:n
        if bridge
            for i in 1:d
                z[i] = AQFED.Math.norminv(clamp(U[i, p], eps(Float64), 1 - eps(Float64)))
            end
            AQFED.MonteCarlo.transform!(bb, z, dw)
        else
            for i in 1:d
                dw[i] = sqrt(h) * AQFED.Math.norminv(clamp(U[i, p], eps(Float64), 1 - eps(Float64)))
            end
        end
        y, logAverage = log(S0), 0.0
        for i in 1:d
            s = exp(y)
            σ = hyperbolicLocalVol(s, ν, β) / s
            y += (r - 0.5 * σ^2) * h + σ * dw[i]
            logAverage += y
        end
        sum += max((kind == :geometric ? exp(logAverage / d) : exp(y)) - K, 0.0)
    end
    return exp(-r * T) * sum / n
end

function rqmcPaths(seq, d::Int, n::Int)
    U = zeros(d, n)
    for i in 1:n
        AQFED.Random.next!(seq, @view(U[:, i]))
    end
    return U
end

function rqmcHyperbolic(method, d, n, R; scale=1.0, K, bridge, kind, S0=1.0, r=0.0, T=1.0, ν=1.0, β=1.0, seed=(3, 14))
    estimates = Vector{Float64}(undef, R)
    Threads.@threads :dynamic for rep in 1:R
        U = rqmcPaths(rqmcSequence(d, n, rep; method=method, seed=seed), d, n)
        estimates[rep] = scale * hyperbolicPayoff(U; S0=S0, r=r, T=T, ν=ν, β=β, K=K, bridge=bridge, kind=kind)
    end
    return mean(estimates), R > 1 ? std(estimates) : NaN
end

## E[S_T] = 1 with d = 10 or 1000 time-steps, plain time-stepping. m = 1 and m = 10 randomizations

const RQMC_SCRAMBLINGS = [("BURLEY", :burley), ("OWEN_FAURE_TEZUKA", :owen_faure_tezuka), ("DIGITAL", :digital), ("MODULO", :modulo), ("PHILOX", :philox64), ("SOBOL", :sobol)]

function rqmcConvergenceResults(R)
    rows = NamedTuple[]
    for d in (10, 1000), (name, method) in RQMC_SCRAMBLINGS, N in 2 .^ (7:17)
        (name == "SOBOL" && R != 1) && continue
        estimate, sd = rqmcHyperbolic(method, d, N, R; K=0.0, bridge=false, kind=:terminal)
        #with a single replicate the plotted measure is the absolute error
        se = R == 1 ? abs(estimate - 1) : sd / sqrt(R)
        push!(rows, (dimension=d, scrambling=name, N=N, R=R, estimate=estimate, rqmc_se=se, replicate_sd=R == 1 ? se : sd))
    end
    return DataFrame(rows)
end

@compute "rqmc_gbm_convergence" begin
    for R in (1, 10)
        writetable("rqmc_convergence_results_m$(R).csv", rqmcConvergenceResults(R))
    end
end

function convergenceComparisonFigure(file)
    df = readcsv(file)
    errors = abs.(df.estimate .- 1)
    limits = (minimum(errors) * 0.5, maximum(errors) * 2)
    labels = Dict("BURLEY" => "Burley", "OWEN_FAURE_TEZUKA" => "ACM-823", "DIGITAL" => "digital shift", "MODULO" => "modulo", "PHILOX" => "Philox", "SOBOL" => "Sobol")
    fig = bookFigure(size=(900, 360))
    axes = map(enumerate((10, 1000))) do (position, d)
        ax = bookAxis(fig[1, position], xlabel="n", ylabel=position == 1 ? "Absolute error" : "", title="d = $d", xscale=log10, yscale=log10,
            xminorticksvisible=true, yminorticksvisible=true)
        for (i, name) in enumerate(unique(df.scrambling))
            s = sort(df[(df.dimension.==d).&(df.scrambling.==name), :], :N)
            scatterlines!(ax, s.N, abs.(s.estimate .- 1), color=PALETTE[i], linewidth=2.2, marker=RQMC_MARKERS[i], markersize=7, label=labels[name])
        end
        #reference slopes n^(-1/2) and 1/n from the first point of d = 10
        first = sort(df[df.dimension.==10, :], :N)[1, :]
        Ns = sort(unique(df.N))
        lines!(ax, Ns, abs(first.estimate - 1) .* (Ns[1] ./ Ns) .^ 0.5, color=(:black, 0.6), linestyle=:dash, linewidth=1.2)
        lines!(ax, Ns, abs(first.estimate - 1) .* (Ns[1] ./ Ns), color=(:black, 0.6), linestyle=:dot, linewidth=1.2)
        ylims!(ax, limits...)
        ax
    end
    Legend(fig[1, 3], axes[1], framevisible=false)
    fig
end
@figure "convergence_d10_d1000_comparison_m1" convergenceComparisonFigure("rqmc_convergence_results_m1.csv")
@figure "convergence_d10_d1000_comparison_m10" convergenceComparisonFigure("rqmc_convergence_results_m10.csv")

## geometric Asian option with m = 64 randomizations, n = 4096 paths, d = 256 time-steps

@compute "heatmap_bb_improvement" begin
    rows = NamedTuple[]
    models = [("Hyperbolic", (S0=1.0, r=0.03, ν=0.30, β=0.50, scale=100.0)), ("Black", (S0=1.0, r=0.0, ν=1.0, β=1.0, scale=1.0))]
    for (model, parameters) in models, K in (0.8, 1.0, 1.2), (name, method) in RQMC_SCRAMBLINGS[[1, 2, 3, 5]], bridge in (false, true)
        (name == "PHILOX" && bridge) && continue #the bridge of a pseudo-random sequence is not part of the comparison
        value, sd = rqmcHyperbolic(method, 256, 4096, 64; K=K, bridge=bridge, kind=:geometric, parameters...)
        push!(rows, (model=model, strike=K, scrambling=name, bridge=bridge, value=value, rqmc_se=sd / sqrt(64), replicate_sd=sd))
    end
    writetable("heatmap_bb_improvement.csv", DataFrame(rows))
end

@figure "heatmap_bb_improvement" begin
    df = readcsv("heatmap_bb_improvement.csv")
    scramblings = [("BURLEY", "Burley"), ("DIGITAL", "digital shift"), ("OWEN_FAURE_TEZUKA", "ACM-823")]
    strikes = [(0.8, "ITM"), (1.0, "ATM"), (1.2, "OTM")]
    se(model, K, name, bridge) = only(df[(df.model.==model).&(df.strike.==K).&(df.scrambling.==name).&(df.bridge.==bridge), :rqmc_se])
    ratios(model) = [se(model, K, name, false) / se(model, K, name, true) for (K, _) in strikes, (name, _) in scramblings]
    range = extrema(vcat(ratios("Hyperbolic"), ratios("Black")))
    fig = bookFigure(size=(800, 520))
    local hm
    for (position, model) in enumerate(("Hyperbolic", "Black"))
        z = ratios(model)
        ax = bookAxis(fig[1, position], xlabel="Moneyness", ylabel=position == 1 ? "Scrambling method" : "", title=model, yreversed=true,
            xgridvisible=false, ygridvisible=false, xticks=(eachindex(strikes), last.(strikes)), yticks=(eachindex(scramblings), last.(scramblings)),
            yticklabelsvisible=position == 1)
        hm = heatmap!(ax, eachindex(strikes), eachindex(scramblings), z, colormap=SURFMAP, colorrange=range)
        for i in axes(z, 1), j in axes(z, 2)
            text!(ax, i, j, text=@sprintf("%.1f", z[i, j]), align=(:center, :center), color=z[i, j] > sum(range) / 2 ? :black : :white)
        end
    end
    Colorbar(fig[1, 3], hm, label="Plain SE / BB SE")
    fig
end
