# Chapter 9: randomized quasi Monte-Carlo with the Heston model, QE against Full Truncation.
# The convergence data heston_qe_qmc_convergence_{qe,ft}_{low,high}.csv were produced by the study now kept in
# book/legacy/hestonmc_study.jl ("Heston RQMC Convergence Vanilla").

@figure "heston_qe_ft_qmc_atm_linear" begin
    strike = 1.0
    labels = ["ChaCha8", "Sobol", "Sobol + Burley", "FT + Burley"]
    markers = [:circle, :diamond, :rect, :dtriangle]
    ticks = [-0.01, -0.008, -0.006, -0.004, -0.002, 0.0, 0.002, 0.004, 0.006, 0.008, 0.01]
    fig = bookFigure(size=(920, 450))
    for (position, (volvol, title)) in enumerate((("low", "Low vol-of-vol"), ("high", "High vol-of-vol")))
        pick(df) = df[(df.Strike.==strike).&(df.N.>=1000), :]
        qe = pick(readcsv("heston_qe_qmc_convergence_qe_$(volvol).csv"))
        ft = pick(readcsv("heston_qe_qmc_convergence_ft_$(volvol).csv"))
        ax = bookAxis(fig[1, position], xlabel="Number of paths", ylabel=position == 1 ? "Pricing error" : "", title=title, xscale=log10,
            xminorticksvisible=true, yminorticksvisible=true, yticks=(ticks, string.(ticks)))
        hlines!(ax, 0.0, color=(:black, 0.45), linewidth=1.0)
        #95% confidence band of the pseudo-random estimate
        rng = sort(qe[qe.Method.=="rng", :], :N)
        band!(ax, rng.N, rng.Error .- 1.96 .* rng.StdErr, rng.Error .+ 1.96 .* rng.StdErr, color=(PALETTE[1], 0.15))
        #the caption lists ChaCha8, Sobol, Sobol + Burley (QE) and FT + Burley
        for (i, method) in enumerate(["rng", "sobol", "sobol_burley"])
            d = sort(qe[qe.Method.==method, :], :N)
            scatterlines!(ax, d.N, d.Error, color=PALETTE[i], linewidth=2.2, marker=markers[i], markersize=7, label=labels[i])
        end
        d = sort(ft[ft.Method.=="sobol_burley", :], :N)
        scatterlines!(ax, d.N, d.Error, color=PALETTE[4], linewidth=2.2, marker=markers[4], markersize=7, label=labels[4])
        ylims!(ax, -0.011, 0.011)
        position == 2 && bookLegend!(ax, position=:rt)
    end
    fig
end

## standard error against the number of time-steps, QE scheme with high vol-of-vol, with and without the Brownian bridge.
## 10-year ATM call, 32 randomizations of 1024 paths. From the script heston_fulltruncation_stderr_sweep.jl.

function hestonStdErrSequence(name::Symbol, rng, nd::Int, n::Int)
    R = AQFED.Random
    name == :rng && return R.AbstractRNGSeq(rng, nd)
    name == :sobol_owen && return R.ScrambledSobolSeq(nd, n, R.OwenFaureTezuka(31, R.ScramblingRngAdapter(rng)))
    name == :sobol_digital && return R.ScrambledSobolSeq(nd, n, R.Digital{nd}(rng))
    name == :sobol_burley4d && return R.BurleySobolSeq(nd, rng)
    name == :sobol_burley && return R.ScrambledSobolSeq(nd, n, R.Burley{nd}(rng))
    throw(DomainError(name, "unknown sequence"))
end

"""
Standard error of the price of `payoff` for each (Brownian bridge, number of time-steps per year) and sequence, one row per combination.
The cost grows with the number of time-steps: the longest jobs start first and the idle threads pick the next one.
"""
function hestonStdErrSweep(model, spot, payoff, timesteps; m=32, n=1024, mindt=0.5 / 365, schemes=Dict(false => AQFED.MonteCarlo.QE(false, true), true => AQFED.MonteCarlo.QE(true, true)))
    MC = AQFED.MonteCarlo
    specTimes = MC.specificTimes(payoff)
    combos = sort([(bb, t) for bb in (false, true) for t in timesteps], by=c -> -c[2])
    results = [NamedTuple[] for _ in combos]
    Threads.@threads :dynamic for k in eachindex(combos)
        bb, steps = combos[k]
        config = MC.GridConfig(model, specTimes, 1.0 / steps, schemes[bb], mindt=min(mindt, 0.5 / steps), tstop=0.0)
        nd = MC.ndims(model, config)
        names = nd <= 21201 ? [:sobol_burley, :rng, :sobol_owen, :sobol_digital, :sobol_burley4d] : [:sobol_burley4d, :rng]
        for name in names
            prices = map(1:m) do i
                seq = hestonStdErrSequence(name, AQFED.Random.ChachaSIMD{10}(UInt32, 20100921 + i), nd, n)
                MC.last_mean(MC.simulateIter(seq, model, spot, payoff, 0, n, config, isRestart=false)[1])
            end
            push!(results[k], (dimension=steps, method=String(name), bb=bb, estimate=mean(prices), std_err=std(prices) / sqrt(m)))
        end
        @info "swept" bb steps nd
    end
    return DataFrame(reduce(vcat, results))
end

@compute "heston_qe_stderr_bb_1k_chacha20_high" begin
    τ, spot = 10.0, 1.0
    model = AQFED.TermStructure.HestonModel(0.133, 0.35, 0.321, -0.63, 1.388, 0.0, 0.0)
    payoff = AQFED.MonteCarlo.ListPayoff([AQFED.MonteCarlo.VanillaOption(true, spot, AQFED.MonteCarlo.BulletCashFlow(τ, τ, false, 0.0), 0.0)])
    timesteps = sort(unique(round.(Int, exp10.(range(log10(8), 3, length=180)))))
    writetable("heston_qe_stderr_bb_1k_chacha20_high.csv", hestonStdErrSweep(model, spot, payoff, timesteps))
end

@figure "heston_qe_stderr_bb_1k_chacha20_high" begin
    df = readcsv("heston_qe_stderr_bb_1k_chacha20_high.csv")
    methods = [("rng", "ChaCha20"), ("sobol_burley", "Burley(HDD)"), ("sobol_burley4d", "Burley(4DP)"), ("sobol_digital", "digital shift"), ("sobol_owen", "ACM-823")]
    fig = bookFigure(size=(800, 400))
    for (position, (bb, title)) in enumerate(((false, "QE-M (no BB)"), (true, "QE-M (with BB)")))
        ax = bookAxis(fig[1, position], xlabel="Number of time steps", ylabel=position == 1 ? "Standard error" : "", title=title, xscale=log10,
            xticks=decadeTicks(1:3), xminorticksvisible=true, xminorticks=IntervalsBetween(9), yminorticksvisible=true)
        for (i, (method, label)) in enumerate(methods)
            d = sort(df[(df.bb.==bb).&(df.method.==method), :], :dimension)
            lines!(ax, d.dimension, d.std_err, color=PALETTE[i], label=label)
        end
        ylims!(ax, 0, nothing)
        position == 1 && Legend(fig[2, 1:2], ax, orientation=:horizontal, framevisible=false)
    end
    fig
end
