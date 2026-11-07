# Chapter 3: American options.

@figure "qd_halley_100" begin
    df = readtable("qd_halley_100.txt")
    fig = bookFigure(size=(800, 400))
    ax = bookAxis(fig[1, 1], xlabel="Iterations", ylabel="S*")
    hlines!(ax, [48.488698], color=:black, linestyle=:dash)
    for (i, (method, marker)) in enumerate(zip(unique(df.Method), (:circle, :rect, :utriangle, :diamond, :cross)))
        d = df[df.Method.==method, :]
        scatterlines!(ax, d.Iteration, d.S, label=string(method), color=PALETTE[i], marker=marker, markersize=7)
    end
    xlims!(ax, 0, 20)
    bookLegend!(ax, position=:rb)
    fig
end

## exercise boundaries of a 5y American put under negative rates r=-0.5%, q=-1% (TR-BDF2 with policy iteration, QD+ in the *a.txt files)
function negativeBoundaryFigure(vol; tmerge=0.0)
    r, q, T = -0.005, -0.01, 5.0
    μ = r - q
    m = readtable("exercise_boundary_put_negative5y_$(round(Int, vol * 100)).txt")
    qd = readtable("exercise_boundary_put_negative5y_$(round(Int, vol * 100))a.txt", squeeze=true)
    lower = sort(m[(m.sign.==-1).&(m.t.>=tmerge), :], :t)
    upper = sort(m[(m.sign.==2).&(m.t.>=tmerge), :], :t)
    fig = bookFigure(size=(324, 360))
    #small panels (a third of the text width): smaller text than the book default
    ax = bookAxis(fig[1, 1], xlabel="Time (year fraction)", ylabel="Asset price", xticks=0:1:5, xlabelsize=12, ylabelsize=12,
        xticklabelsize=11, yticklabelsize=11)
    band!(ax, upper.t, lower.x, upper.x, color=(PALETTE[1], 0.12))
    lines!(ax, lower.t, lower.x, color=:black)
    lines!(ax, upper.t, upper.x, color=:black)
    t = sort(qd.t)
    asymLower = @. r * 100 / q * (1 + 0.638 * vol * sqrt(T - t))
    asymUpper = [(a = log(vol^2 / (8π * (T - ti) * μ^2))) > 0 ? 100 - 100 * vol * sqrt((T - ti) * a) : NaN for ti in t]
    lines!(ax, t, asymLower, color=PALETTE[3], linestyle=:dashdot, label="Asymptotic")
    lines!(ax, t, asymUpper, color=PALETTE[3], linestyle=:dashdot)
    q2 = sort(qd, :t)
    lines!(ax, q2.t, q2.QDPu, color=PALETTE[2], linestyle=:dash, label="QD+")
    lines!(ax, q2.t, q2.QDPl, color=PALETTE[2], linestyle=:dash)
    text!(ax, 3.7, 60, text="Early-exercise\noptimal", align=(:center, :center), fontsize=11)
    limits!(ax, -0.1, 5.1, 40, 100)
    Legend(fig[2, 1], ax, "Method", orientation=:horizontal, titleposition=:left, framevisible=false, labelsize=11, titlesize=11, patchsize=(16, 10))
    fig
end
@figure "exercise_boundary_put_negative5y_4b" negativeBoundaryFigure(0.04)
@figure "exercise_boundary_put_negative5y_8b" negativeBoundaryFigure(0.08)
@figure "exercise_boundary_put_negative5y_15b" negativeBoundaryFigure(0.15, tmerge=2.4)
