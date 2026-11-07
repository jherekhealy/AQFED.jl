# Figures of the book rebuilt from the data files of the original R study (figures_old/lefloch_eqd_book_colo.r),
# persisted in book/data.

## Chapter 4: effective growth rate of the dividend yield interpolations
@figure "yield_interp" begin
    df = readtable("yield_interp.txt")
    df = df[isfinite.(df.mu), :]
    fig = bookFigure(size=(576, 300))
    ax = bookAxis(fig[1, 1], xlabel="Time", ylabel="Effective growth rate")
    styles = Dict("FlatForward" => :solid, "Linear" => :dash, "Continuous" => :dot)
    for (i, m) in enumerate(["FlatForward", "Linear", "Continuous"])
        d = sort(df[df.Method.==m, :], :t)
        lines!(ax, d.t, d.mu, label=m, color=PALETTE[i], linestyle=styles[m])
    end
    xlims!(ax, 0, 0.02)
    bookLegend!(ax, position=:rb)
    fig
end

## Chapter 6: Longstaff-Schwartz regression at the last exercise date (cubic fit of the payoff on each path)
function longstaffFigure(file, p, scale; position)
    df = readtable(file)
    fig = bookFigure(size=(288, 288))
    ax = bookAxis(fig[1, 1], xlabel="S", ylabel="Continuation value")
    scatter!(ax, df.S, df.Value, color=(C_RIBBON, 0.6), markersize=5, label="Payoff on path")
    s = range(extrema(df.S)..., length=201)
    x = s ./ scale
    lines!(ax, s, scale .* (p[1] .+ p[2] .* x .+ p[3] .* x .^ 2 .+ p[4] .* x .^ 3), color=PALETTE[1], label="Regression")
    bookLegend!(ax, position=position)
    fig
end
@figure "longstaff_schwartz_1d_last" longstaffFigure("longstaff_schwartz_1d_last.txt",
    [35.184671232972406, -0.596401737136396, -0.015121703347229773, 0.00020865930974474042], 1.0, position=:rt)
@figure "longstaff_schwartz_straddle_last" longstaffFigure("longstaff_schwartz_straddle_last.txt",
    [1.2327662438347093, -2.335318304908597, 1.4463373593446762, -0.18235160971419226], 40.0, position=:rb)

## Chapter 6: local volatility Monte-Carlo with control variates, distribution of the absolute error over the options
const LV_CONTROL_NAMES = Dict("Black-ForwardCV" => "Black-F", "Black-ATMCV" => "Black-A", "LV-Euler" => "LVE", "LV-Euler-ForwardCV" => "LVE-F",
    "LV-Euler-ATMCV" => "LVE-A", "LV-Milstein" => "LVM", "LV-Milstein-ForwardCV" => "LVM-F", "LV-Milstein-ATMCV" => "LVM-A")
function lvControlFigure(file)
    df = readtable(file)
    df.Method = [get(LV_CONTROL_NAMES, m, m) for m in df.Method]
    #zero errors are dropped (log scale), as in the original figure
    df = df[(df.Method.!="LV-Euler-Mid").&(df.Error.!=0), :]
    methods = sort(unique(df.Method))
    fig = bookFigure(size=(432, 216))
    ax = bookAxis(fig[1, 1], ylabel="Absolute error in price", yscale=log10, xticks=(1:length(methods), methods))
    idx = [findfirst(==(m), methods) for m in df.Method]
    boxplot!(ax, idx, abs.(df.Error), color=(PALETTE[1], 0.35), strokecolor=PALETTE[1], strokewidth=1.2,
        whiskercolor=PALETTE[1], mediancolor=:black, outliercolor=PALETTE[1], markersize=5)
    fig
end
@figure "lv_control_zeliade_sobolbb_001" lvControlFigure("lv_control_zeliade_sobolbb_001.txt")
@figure "lv_control_zeliade_sobolbb_016" lvControlFigure("lv_control_zeliade_sobolbb_016.txt")

## Chapter 9: Heston calibration to the Kahale example, RMSE profile in (ρ, κ) or (σ, κ) after optimizing the other parameters
function rmseGrid(file, xcol, ycol)
    df = readtable(file)
    xs, ys = sort(unique(df[!, xcol])), sort(unique(df[!, ycol]))
    z = fill(NaN, length(xs), length(ys))
    for r in eachrow(df)
        z[searchsortedfirst(xs, r[xcol]), searchsortedfirst(ys, r[ycol])] = log(r.RMSE)
    end
    return xs, ys, z
end
function rmseContourFigure(file; xcol=:Rho, ycol=:Kappa, xlims=nothing, ylims=(0, 8))
    xs, ys, z = rmseGrid(file, xcol, ycol)
    fig = bookFigure(size=(288, 300))
    ax = bookAxis(fig[1, 1], xlabel=xcol == :Rho ? "ρ" : "σ", ylabel="κ")
    hm = heatmap!(ax, xs, ys, z, colormap=SURFMAP, interpolate=true, rasterize=3)
    contour!(ax, xs, ys, z, levels=20, color=(:black, 0.6), linewidth=0.8)
    xlims === nothing || xlims!(ax, xlims...)
    ylims === nothing || ylims!(ax, ylims...)
    Colorbar(fig[2, 1], hm, label="log(RMSE)", vertical=false, flipaxis=false, height=10, width=Relative(0.9))
    fig
end
function rmseSurfaceFigure(file; xcol=:Rho, ycol=:Kappa, azimuth=-0.8π)
    xs, ys, z = rmseGrid(file, xcol, ycol)
    fig = bookFigure(size=(432, 360))
    surface3D!(fig[1, 1], xs, ys, z, xlabel=xcol == :Rho ? "ρ" : "σ", ylabel="κ", zlabel="log(RMSE)", azimuth=azimuth)
    fig
end
@figure "heston_kahale_smalltime_kapparhof2" rmseSurfaceFigure("heston_kahale_smalltime_kapparhof3.txt")
@figure "heston_kahale_smalltime_kapparhof2_contour_bw" rmseContourFigure("heston_kahale_smalltime_kapparhof3.txt", xlims=(-1, -0.5))
@figure "heston_kahale_flinn_kapparhof2_bw" rmseSurfaceFigure("heston_kahale_flinn_kapparhof3.txt")
@figure "heston_kahale_flinn_kapparhof2_contour_bw" rmseContourFigure("heston_kahale_flinn_kapparhof3.txt")
@figure "heston_kahale_lpp2_kapparhof2_bw" rmseSurfaceFigure("heston_kahale_lpp2_kapparhof3.txt")
@figure "heston_kahale_lpp2_kapparhof2_contour_bw" rmseContourFigure("heston_kahale_lpp2_kapparhof3.txt", xlims=(-1, 1))
@figure "heston_kahale_kappasigma2" rmseSurfaceFigure("heston_kahale_kappasigma3.txt", xcol=:Sigma, azimuth=0.67π)
