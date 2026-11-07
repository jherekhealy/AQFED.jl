# Chapter 6: Monte-Carlo, theta of a vanilla and of a one-touch option.

@figure "theta_vanilla_variance" begin
    vol, T, t1 = 0.2, 1.0, 1.0 / 365
    dt = range(0, stop=t1, length=101)
    incremental = @. exp(vol^2 * T) * (1 + exp(-vol^2 * dt) - 2 * exp(vol^2 * (sqrt(t1 - dt) * sqrt(t1) - t1)))
    bb = @. exp(vol^2 * T) * (1 + exp(-vol^2 * dt) - 2 * exp(vol^2 * (sqrt(T - dt) * sqrt(T) - T)))
    #ratio of the Monte-Carlo standard errors measured with 1-step paths, dt = 0.5 and 0.95 days
    mc050, mc095 = 0.000432 / 0.000020, 0.000603 / 0.000020
    fig = bookFigure(size=(800, 400))
    ax = bookAxis(fig[1, 1], xlabel="dt in days", ylabel="Ratio of standard errors Incremental/BB")
    lines!(ax, dt .* 365, sqrt.(incremental ./ bb), label="Theoretical")
    scatter!(ax, [0.5, 0.95], [mc050, mc095], label="Monte-Carlo", color=PALETTE[2], markersize=10)
    Legend(fig[1, 2], ax, framevisible=false)
    fig
end

function onetouchThetaFigure(series)
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Number of paths", ylabel="Theta", xscale=log10)
    for (i, (label, df)) in enumerate(series)
        idx = sortperm(df.N)
        lines!(ax, df.N[idx], df.Value[idx], label=label, color=PALETTE[i])
    end
    limits!(ax, 1024, 1_000_000, -1500, 500)
    bookLegend!(ax, position=:rb)
    return fig
end

@figure "theta_lr_onetouch" begin
    lr = readtable("onetouch_theta_mrg_lr.txt")
    onetouchThetaFigure(["Likelihood ratio" => lr[lr.Estimate.=="Theta", :], "Naive" => readtable("onetouch_theta_mrg_naive.txt")])
end

@figure "theta_smooth_onetouch" begin
    onetouchThetaFigure(["Brownian-bridge" => readtable("onetouch_theta_mrg_bb_smooth.txt"), "Incremental" => readtable("onetouch_theta_mrg_inc_smooth.txt")])
end
