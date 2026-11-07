# Chapter 7: implied volatility smiles. Market data, parametric and spline interpolations, extrapolation,
# collocation, Andreasen-Huge. All data were persisted by earlier studies (Go code, AQFED tests).

using AQFED

const VOL_LABEL = "Implied volatility in %"

#market quotes as scatter, model curves as lines grouped by `group`
function smileFigure(ref, models, group; x=:Strike, y=:Vol, refx=x, refy=y, reflabel="Market", order=sort(unique(models[!, group])),
    labels=Dict(), legendtitle=nothing, size=(400, 400), xlabel="Strike", ylabel=VOL_LABEL, ylims=nothing, legendposition=:rt, yscale=100)
    fig = bookFigure(size=size)
    ax = bookAxis(fig[1, 1], xlabel=xlabel, ylabel=ylabel)
    ref === nothing || referenceScatter!(ax, ref[!, refx], ref[!, refy] .* yscale, label=reflabel)
    groupLines!(ax, models[!, x], models[!, y] .* yscale, models[!, group], order=order, labels=labels)
    ylims === nothing || ylims!(ax, ylims...)
    bookLegend!(ax, title=legendtitle, position=legendposition)
    return fig
end

## AAPL option chain of February 3, 2016, expiry January 20, 2017

function callPutFigure(df, columns::Vector{Pair{Symbol,String}}; ylims=(25.5, 50))
    fig = bookFigure(size=(400, 400))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel=VOL_LABEL)
    for (col, label) in columns
        keep = .!isnan.(df[!, col])
        lines!(ax, df.Strike[keep], df[keep, col] .* 100, label=label)
    end
    ylims!(ax, ylims...)
    bookLegend!(ax)
    return fig
end

@figure "aapl170120_160203_euro_vol" callPutFigure(readtable("aapl170120_160203_euro_vol.txt"), [:Call => "Call", :Put => "Put"])
@figure "aapl170120_160203_am_yield" callPutFigure(readtable("aapl170120_160203_am_yield_vol_tr.txt"), [:CallVol => "Call", :PutVol => "Put"])
@figure "aapl170120_160203_am_cash_vol" callPutFigure(readtable("aapl170120_160203_am_cash_vol.txt"), [:Call => "Call", :Put => "Put"])
@figure "aapl170120_160203_am_prop_vol" callPutFigure(readtable("aapl170120_160203_am_prop_vol.txt"), [:Call => "Call", :Put => "Put"])

function aaplOTM()
    df = readtable("aapl170120_160203_am_prop_vol.txt")
    df.Vol = ifelse.(df.Strike .< 95.0, df.Put, df.Call)
    return df
end
@figure "aapl170120_160203_am_prop_vol_call_otm" callPutFigure(aaplOTM(), [:CallAsk => "CallAsk", :CallBid => "CallBid", :Vol => "OTM"], ylims=(16, 50))
@figure "aapl170120_160203_am_prop_vol_put_otm" callPutFigure(aaplOTM(), [:PutAsk => "PutAsk", :PutBid => "PutBid", :Vol => "OTM"], ylims=(16, 50))

## AAPL 1m (March 4, 2016) and 2y (January 19, 2018) as of February 12, 2016

aapl1m() = readtable("aapl_160304_160212_am_prop_vol.txt")

@figure "aapl_1m_parabola" smileFigure(aapl1m(), rows(readtable("aapl_160304_160212_am_prop_vol_parabola.txt"), :Method, "Strike", "Logmoneyness"), :Method, reflabel="Mid", ylims=(29, 50))
@figure "aapl_1m_spline" smileFigure(aapl1m(), rows(readtable("aapl_160304_160212_am_prop_vol_cubic.txt"), :Method, "Parabola", "Spline3"), :Method, reflabel="Mid", ylims=(29, 50))
@figure "aapl_2y_parabola" smileFigure(readtable("aapl_180119_160212_am_prop_vol.txt"), rows(readtable("aapl_180119_160212_am_prop_vol_parabola.txt"), :Method, "Strike", "Logmoneyness"), :Method, reflabel="Mid")

@figure "aapl_1m_quartic_error" begin
    #vega-like weights of the 41 quotes
    weights = [3.0, 5.0, 6.928203230275509, 5.291502622129181, 3.3166247903554, 4.898979485566356, 5.0, 3.7416573867739413, 12.68857754044952, 3.4641016151377544, 5.830951894845301, 7.483314773547883, 12.409673645990857, 5, 11.445523142259598, 6.244997998398398, 10.392304845413264, 5.291502622129181, 5.385164807134504, 2, 7.745966692414834, 6.48074069840786, 19.313207915827967, 1.4142135623730951, 11.445523142259598, 6.164414002968976, 10.04987562112089, 4.358898943540674, 5, 5.916079783099616, 10.908712114635714, 5.196152422706632, 31.796226191169293, 14.071247279470288, 6.6332495807108, 4.898979485566356, 3.1622776601683795, 5.477225575051661, 2.8284271247461903, 6.324555320336759, 1.7320508075688772]
    nweights = weights / sum(weights) * length(weights)
    df = rows(readtable("aapl_160304_160212_am_prop_vol_quartic.txt"), :Method, "Quartic", "Spline3", "RBF")
    df.WeightedError = similar(df.Error)
    for m in unique(df.Method)
        idx = findall(==(m), df.Method)
        df.WeightedError[idx] = abs.(df.Error[idx] .* nweights .* 100)
    end
    fig = bookFigure(size=(400, 400))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Weighted volatility error in %")
    groupLines!(ax, df.Strike, df.WeightedError, df.Method, order=["Quartic", "RBF", "Spline3"])
    bookLegend!(ax)
    fig
end

@figure "aapl_1m_svi" begin
    df = readtable("aapl_1m_svi_a0.txt")
    smileFigure(rows(df, :Method, "Reference"), rows(df, :Method, "SVI", "Parabolic", "SVI-a-positive"), :Method, x=:K, ylims=(29, 50))
end

@figure "aapl_1m_svi_unbounded" begin
    df = readtable("aapl_1m_svi_unbonuded.txt")
    smileFigure(rows(df, :Method, "Reference"), rows(df, :Method, "SVI", "Parabolic"), :Method, x=:K, ylims=(29, 50))
end

@figure "aapl_1m_svi_prices" begin
    forward, tte = 93.6769575815521, 0.057534246575342465
    sviPositive = AQFED.TermStructure.SVISection(0.0, 0.6637133353, -0.4073556325, 0.1493655118, 0.0084858398, tte, forward)
    sviUnbounded = AQFED.TermStructure.SVISection(-4.0799259866, 18.5213470421, 0.8767046157, 0.4680205198, 0.9311057129, tte, forward)
    price(svi, k) = AQFED.Black.blackScholesFormula(true, k, forward, AQFED.TermStructure.varianceByLogmoneyness(svi, log(k / forward)) * tte, 1.0, 1.0)
    strikes = range(forward, stop=forward * 10, length=1001)
    fig = bookFigure(size=(400, 400))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Call price", xscale=log10, yscale=log10,
        xticks=([forward, 2forward, 4forward, 8forward], ["f", "2f", "4f", "8f"]))
    for (svi, label) in ((sviUnbounded, "unbounded"), (sviPositive, "positive"))
        p = price.(svi, strikes)
        keep = p .> 0
        lines!(ax, strikes[keep], p[keep], label=label)
    end
    bookLegend!(ax, title="a", position=:lb)
    fig
end

@figure "aapl_1m_parabola_bdk" begin
    df = readtable("aapl_1m_bdk.txt")
    smileFigure(aapl1m(), df[df.Method.!="Reference", :], :Method, legendtitle="Extrapolation", ylims=(29, 50), size=(400, 300))
end

function extrapolationFigure(methods)
    df = rows(readtable("aapl_1m_extrapolation.txt"), :Method, methods...)
    df.Vol = sqrt.(df.w)
    smileFigure(nothing, df, :Method, order=collect(methods), legendtitle="Method", ylims=(20, 400), ylabel="Volatility in %", legendposition=:ct)
end
#legend on the right side, outside of the axis
@figure "aapl_1m_extrapolation_bdk" begin
    methods = ("Parabolic", "Spline", "Parabolic-BDK", "Spline-BDK")
    df = rows(readtable("aapl_1m_extrapolation.txt"), :Method, methods...)
    fig = bookFigure(size=(600, 400))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Volatility in %")
    groupLines!(ax, df.Strike, sqrt.(df.w) .* 100, df.Method, order=collect(methods))
    ylims!(ax, 20, 400)
    Legend(fig[1, 2], ax, "Method", framevisible=false)
    fig
end
@figure "aapl_1m_extrapolation_rk" extrapolationFigure(("Parabolic", "Spline", "Parabolic-RK", "Spline-RK"))

#1m (Strike, Call) and 2y quotes, against strike or against the put delta
function aapl1m2y(byDelta)
    m2y = readtable("aapl_3y.txt")
    m1 = readtable("aapl_1m.txt")
    fig = bookFigure(size=(400, 400))
    ax = bookAxis(fig[1, 1], xlabel=byDelta ? "- Put Δ" : "Strike", ylabel=VOL_LABEL)
    for (df, forward, label) in ((m1, 93.6, "1m"), (m2y, 92.6, "2y"))
        x = byDelta ? @.(AQFED.Math.normcdf(-(-log(df.Strike / forward) + 0.5 * df.Call^2 * 30.0 / 365) / (df.Call * sqrt(30.0 / 365)))) : df.Strike
        keep = .!isnan.(df.Call)
        idx = sortperm(x[keep])
        lines!(ax, x[keep][idx], df.Call[keep][idx] .* 100, label=label)
    end
    bookLegend!(ax, title="Expiry")
    return fig
end
@figure "aapl_1m_2y_strike" aapl1m2y(false)
@figure "aapl_1m_2y_delta" aapl1m2y(true)

## SVI and SABR

@figure "bancorp_10y_svi_asymptotes" begin
    df = readtable("bancorp_10y_svi_asymptotes.txt")
    smileFigure(rows(df, :method, "Reference"), rows(df, :method, "Left-Raw", "Left-Truncated", "Right-Raw"), :method, x=:strike, y=:vol,
        reflabel="Reference", legendtitle="Method", size=(480, 360), ylabel="Volatility in %")
end

@figure "airliq_10y_svi_sigma" begin
    df = readtable("airliq_10y_svi.txt")
    data = rows(df, :method, "Reference")
    S = 95.05
    sections = [
        "Parabolic-LM" => AQFED.TermStructure.SVISection(0.03282573662747978, 0.02252578300083763, -0.9999901697624964, 0.01931401931608065, 0.36908963892694535, 1.0, S),
        "Quasi-Explicit" => AQFED.TermStructure.SVISection(2.666984783436798E-10, 0.022407412867936443, -1.0, 5.000000054946551E-4, 1.104208741949215, 1.0, S),
        "Quasi-Explicit σ ≥ 0.25" => AQFED.TermStructure.SVISection(0.0, 0.022682627008952412, -1.0, 0.25000000000001094, 1.0761732271448656, 1.0, S)]
    k = range(-1.5, stop=1.5, step=0.01)
    fig = bookFigure(size=(480, 360))
    ax = bookAxis(fig[1, 1], xlabel="Log-moneyness", ylabel="Volatility in %")
    referenceScatter!(ax, log.(data.strike ./ S), data.vol .* 100)
    for (label, svi) in sections
        lines!(ax, k, sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(svi, k)) .* 100, label=label)
    end
    bookLegend!(ax)
    fig
end

@figure "explicit_fit_sabr_0479_beta1" begin
    df = readtable("explicit_fit_sabr_0479_beta1.txt")
    smileFigure(rows(df, :Method, "Reference"), df[df.Method.!="Reference", :], :Method, y=:Volatility, reflabel="Reference",
        order=["Parabola", "α_0", "α_1"], labels=Dict("α_0" => "α₀", "α_1" => "α₁"), size=(480, 360), ylabel="Volatility in %")
end

@figure "aapl_1m_sabr" begin
    df = readtable("aapl_1m_sabr.txt")
    smileFigure(rows(df, :Method, "Reference"), df[(df.Method.!="Reference").&(df.Method.!="SVI"), :], :Method, legendtitle="Interpolation",
        size=(400, 380), ylabel="Volatility in %")
end

@figure "two_sabr_1_smile" begin
    df = readtable("two_sabr_1_smile.txt")
    #labels as in the original figure (groups sorted alphabetically: High, Low)
    smileFigure(rows(df, :alpha, "Reference"), df[df.alpha.!="Reference", :], :alpha, y=:Volatility, reflabel="Market vols",
        labels=Dict("High" => "α=0.237", "Low" => "α=0.850"), legendtitle="Parameters", size=(400, 380), ylabel="Volatility in %")
end

function spxw1wFigure(file)
    df = readtable(file)
    smileFigure(rows(df, :Model, "Reference"), rows(df, :Model, "LSpline", "SABR", "SVI"), :Model, y=:Volatility,
        labels=Dict("LSpline" => "Least-squares Spline"), legendtitle="Interpolation", size=(400, 380), ylabel="Volatility in %")
end
@figure "spxw170316_1w" spxw1wFigure("spxw170316_1w.txt")
@figure "spxw170316_1w_fulls" spxw1wFigure("spxw170316_1w_fulls.txt")

#XSSVI against SVI on AAPL, quotes as logmoneyness and total variance with weights
function xssviFigure(tte, forward, y, v, sqrtw, sviParams)
    vols = sqrt.(v)
    svi = AQFED.TermStructure.SVISection(sviParams..., tte, forward)
    xssvi = AQFED.VolatilityModels.calibrateXSSVISection(tte, forward, y, vols, sqrtw .^ 2)
    strikes = exp.(y) .* forward
    fine = range(strikes[1], stop=strikes[end], length=101)
    fig = bookFigure(size=(400, 380))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Volatility in %")
    referenceScatter!(ax, strikes, vols .* 100, label="Market")
    for (label, s) in ("SVI" => svi, "XSSVI" => xssvi)
        lines!(ax, fine, sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(s, log.(fine ./ forward))) .* 100, label=label)
    end
    bookLegend!(ax)
    return fig
end

@figure "aapl_1m_xssvi" xssviFigure(0.057534246575342465, 93.6769575815521,
    [-0.291357, -0.222364, -0.157825, -0.097201, -0.091336, -0.085505, -0.073944, -0.068213, -0.062515, -0.056849, -0.051216, -0.045613, -0.040042, -0.034502, -0.028992, -0.023513, -0.018063, -0.012643, -0.007252, -0.00189, 0.003443, 0.008748, 0.014025, 0.019274, 0.024496, 0.029691, 0.034859, 0.04, 0.045116, 0.050205, 0.055268, 0.060306, 0.065318, 0.075269, 0.085121, 0.094877, 0.104539, 0.114108, 0.123587, 0.160628, 0.20508],
    [0.324092242681, 0.24657977862400002, 0.186966300816, 0.144333887569, 0.14130081, 0.138231522025, 0.132426121216, 0.129557523481, 0.127000289641, 0.12465701262399999, 0.12244960518400001, 0.11985236280899998, 0.117753981409, 0.115628561764, 0.11343895524900001, 0.11194377639999999, 0.10989755406400002, 0.108062755441, 0.10603098937600001, 0.1045099584, 0.10286837436099999, 0.101191427236, 0.099962204224, 0.09846353652099998, 0.09741702168899999, 0.09611178036099999, 0.09492622620100001, 0.093888926569, 0.09263866195600001, 0.0920212225, 0.09084738528099999, 0.09000180000900002, 0.089596054276, 0.08850030009999998, 0.08827197523599999, 0.087838140625, 0.088819496676, 0.08968946832400002, 0.09121487232400001, 0.10995856000000001, 0.14437416115600002],
    [3, 5, 6.928203230275509, 5.291502622129181, 3.3166247903554, 4.898979485566356, 5, 3.7416573867739413, 12.68857754044952, 3.4641016151377544, 5.830951894845301, 7.483314773547883, 12.409673645990857, 5, 11.445523142259598, 6.244997998398398, 10.392304845413264, 5.291502622129181, 5.385164807134504, 2, 7.745966692414834, 6.48074069840786, 19.313207915827967, 1.4142135623730951, 11.445523142259598, 6.164414002968976, 10.04987562112089, 4.358898943540674, 5, 5.916079783099616, 10.908712114635714, 5.196152422706632, 31.796226191169293, 14.071247279470288, 6.6332495807108, 4.898979485566356, 3.1622776601683795, 5.477225575051661, 2.8284271247461903, 6.324555320336759, 1.7320508075688772],
    (0.0, 0.6637133353, -0.4073556325, 0.1493655118, 0.0084858398))

@figure "aapl_2y_xssvi" xssviFigure(1.936986301369863, 95.79205142667695,
    [-0.70145, -0.650157, -0.554847, -0.467835, -0.387793, -0.313685, -0.244692, -0.180153, -0.119529, -0.090541, -0.06237, -0.034971, -0.008303, 0.017673, 0.04299, 0.091781, 0.138301, 0.182752, 0.225312, 0.266134, 0.305355, 0.343095, 0.379463, 0.414554, 0.448455, 0.573619, 0.630777],
    [0.165566796201, 0.160235286436, 0.149636675241, 0.14268146382400002, 0.132878475625, 0.12624732796899998, 0.12292176240400002, 0.11877741888099998, 0.114789505636, 0.113083565841, 0.110983592164, 0.10859595252100002, 0.1055470144, 0.10508294722499999, 0.103609309456, 0.10128433550399998, 0.10002481528900001, 0.09683423712400001, 0.09441608198399999, 0.09437736968100001, 0.09293535160899999, 0.09121849657600001, 0.09058956236099999, 0.08931311560899999, 0.08799706944899999, 0.08668372524099999, 0.086451876729],
    [12.449899597988733, 3.1622776601683795, 3.1622776601683795, 7.416198487095663, 3.7416573867739413, 11.224972160321824, 22.80350850198276, 8.602325267042627, 5.0990195135927845, 3.3166247903554, 10.099504938362077, 8, 16.792855623746664, 3.872983346207417, 19.974984355438178, 11.090536506409418, 12.409673645990857, 5.744562646538029, 7.0710678118654755, 9.486832980505138, 15.264337522473747, 7.0710678118654755, 8.54400374531753, 7.0710678118654755, 5.830951894845301, 7.681145747868608, 1.7320508075688772],
    (0.0000010186, 0.0962772082, -0.6277030428, 1.1289857530, -0.0435839844))

## Jaeckel December example: left tail of the rational spline

function jaeckelDecFigure(y, ylabel; reference=false)
    df = readtable("jackel_dec_ratspline.txt", squeeze=true)
    df0 = readtable("jackel_dec_ratspline_0.txt", squeeze=true)
    fig = bookFigure(size=(400, 300))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel=ylabel, xscale=log10)
    if reference
        ref = readtable("jaeckel_dec5_bach_opt_x.txt", squeeze=true)
        referenceScatter!(ax, ref.Strike, ref.VolRef .* 100)
    end
    scale = reference ? 100 : 1
    groupLines!(ax, vcat(df.Strike, df0.Strike), vcat(df[!, y], df0[!, y]) .* scale,
        vcat(fill("Lognormal", nrow(df)), fill("Zero strike", nrow(df0))))
    xlims!(ax, 0.0068, 1)
    bookLegend!(ax, title="Tail")
    return fig
end
@figure "jackel_dec_vol_leftd1" jaeckelDecFigure(:Vol, "Black implied volatility in %", reference=true)
@figure "jackel_dec_dens_leftd1" jaeckelDecFigure(:Density, "Implied probability density")

## Collocation

#reference points as crosses with dotted guides to the axes
function guidedCrosses!(ax, x, y; x0, y0)
    for (xi, yi) in zip(x, y)
        lines!(ax, [x0, xi], [yi, yi], color=:black, linestyle=:dot, linewidth=0.8)
        lines!(ax, [xi, xi], [y0, yi], color=:black, linestyle=:dot, linewidth=0.8)
    end
    scatter!(ax, x, y, marker=:cross, color=:black, markersize=9)
end

@figure "blackxy" begin
    ref = readtable("black_xy_ref.txt")
    bs = readtable("black_xy.txt")
    cubic = readtable("black_xy_cubic.txt")
    fig = bookFigure(size=(400, 300))
    ax = bookAxis(fig[1, 1], xlabel="x", ylabel="y", yticks=(ref.y, string.(round.(ref.y, digits=1))))
    guidedCrosses!(ax, ref.x, ref.y, x0=-5.0, y0=0.0)
    lines!(ax, bs.x, bs.y, label="Black-Scholes", color=PALETTE[1])
    lines!(ax, cubic.x, cubic.yb, label="Cubic polynomial collocation", color=PALETTE[2])
    xlims!(ax, -5, 5)
    ylims!(ax, low=0)
    bookLegend!(ax, position=:lt)
    fig
end

@figure "black_yp" begin
    ref = readtable("black_xy_ref.txt")
    cubic = readtable("black_xy_cubic.txt")
    fig = bookFigure(size=(400, 300))
    ax = bookAxis(fig[1, 1], xlabel="y", ylabel="Cumulative probability", xticks=(ref.y, string.(round.(ref.y, digits=1))))
    guidedCrosses!(ax, ref.y, ref.p, x0=0.0, y0=0.0)
    lines!(ax, cubic.y, cubic.p, label="Black-Scholes", color=PALETTE[1])
    lines!(ax, cubic.y, cubic.pb, label="Cubic polynomial collocation", color=PALETTE[2])
    xlims!(ax, 20, 230)
    ylims!(ax, low=0)
    bookLegend!(ax, position=:rb)
    fig
end

quinticCollocation(a, x) = @. a[1] + a[2] * x + a[3] * x^2 + a[4] * x^3 + a[5] * x^4 + a[6] * x^5
const OPTIMAL_QUINTIC = [2849.4246271474985, 59.39270440282361, -6.271793622794486, 3.6687255123195968, -1.3843300393219482, 0.7362716519921044]

#collocation guesses (cubic, quintic) and optimal quintic against the strike estimate of the collocation points
function collocationGuessFigure(file, estimateLabel, guesses; size=(400, 400))
    df = readtable(file)
    x = range(-5.0, stop=5, step=0.01)
    fig = bookFigure(size=size)
    ax = bookAxis(fig[1, 1], xlabel="x", ylabel="Strike")
    scatter!(ax, df.x, df.Strike, label=estimateLabel, color=:transparent, strokecolor=C_RING, strokewidth=1.2, markersize=7)
    for (i, (label, a)) in enumerate(guesses)
        lines!(ax, x, quinticCollocation(a, x), label=label, color=PALETTE[i])
    end
    ylims!(ax, -2000, 6000)
    bookLegend!(ax, position=:lt)
    return fig
end

@figure "density_spw_1m_230118_collo5_cum_bw" collocationGuessFigure("density_spw_1m_230118_collo5_cum.txt", "raw estimate", [
        "cubic guess" => [2785.3521606355916, 146.73682894029116, 0, -0.5950579887459523, 0, 0],
        "quintic guess" => [2876.848315810806, 56.80448843832256, -49.76759382860678, 8.487744676264446, 3.9730404748463086, 0.2896176339589605],
        "optimal quintic collocation" => OPTIMAL_QUINTIC], size=(800, 400))
@figure "density_spw_1m_230118_collo5_cumqp_bw" collocationGuessFigure("density_spw_1m_230118_collo5_cumqp.txt", "QP estimate", [
    "cubic guess" => [2805.1169123299496, -11.117360654194995, 0, 23.059650244031207, 0, 0],
    "quintic guess" => [2848.9760316890156, 49.63700482922661, -5.980960073681884, 7.008424288762293, -1.3317427361985148, 0.5102028868911852],
    "optimal quintic collocation" => [2849.434477060442, 59.397024909112744, -6.288430108083293, 3.6641083110376385, -1.382067848540136, 0.7368652527688424]])
@figure "density_spw_1m_230118_collo5_cums_bw" collocationGuessFigure("density_spw_1m_230118_collo5_cums.txt", "forward sweep estimate", [
    "cubic guess" => [2859.069739671715, 5.857576611515046, -16.22072552838554, 19.177240740417457, 0, 0],
    "quintic guess" => [2851.155970143886, 65.10356540971928, -9.663751977115254, -0.047207132214036975, -0.8307915866774332, 1.2708791758944042],
    "optimal quintic collocation" => OPTIMAL_QUINTIC])

#SPX 1m: market (SVI vol minus its error), SVI, and two collocations
function spxCollocationFigure(prefix, collocations, y; size, ylabel)
    svi = readtable("$(prefix)_svi.txt")
    fig = bookFigure(size=size)
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel=ylabel)
    scale = y == :Vol ? 100 : 1
    #the quotes as rings above the curves (the collocations follow them closely and would hide a ribbon)
    y == :Vol && referenceScatter!(ax, svi.Strike, (svi.Vol .- svi.VolError) .* 100, label="Market", dense=false, markersize=5)
    lines!(ax, svi.Strike, svi[!, y] .* scale, label="SVI", color=PALETTE[1])
    for (i, (file, label)) in enumerate(collocations)
        df = readtable(file)
        lines!(ax, df.Strike, df[!, y] .* scale, label=label, color=PALETTE[i+1])
    end
    bookLegend!(ax)
    return fig
end
@figure "vol_spw_1m_230118_collo_bw" spxCollocationFigure("density_spw_1m_230118",
    ["density_spw_1m_230118_collo3.txt" => "Cubic collocation", "density_spw_1m_230118_collo5.txt" => "Quintic collocation"], :Vol, size=(800, 400), ylabel="Volatility in %")
@figure "vol_spw_1m_050218_collo_bw" spxCollocationFigure("density_spw_1m_050218",
    ["density_spw_1m_050218_collo5.txt" => "Quintic collocation", "density_spw_1m_050218_collo9.txt" => "Nonic collocation"], :Vol, size=(800, 400), ylabel="Volatility in %")
@figure "density_spw_1m_050218_collo_bw" spxCollocationFigure("density_spw_1m_050218",
    ["density_spw_1m_050218_collo5.txt" => "Quintic collocation", "density_spw_1m_050218_collo9.txt" => "Nonic collocation"], :DensityNum, size=(800, 300), ylabel="Probability density")

#same color for a method in the density and the volatility figures
const BIMODAL_COLORS = Dict("Kernel" => PALETTE[1], "Quintic" => PALETTE[2], "Nonic" => PALETTE[3], "Cubic" => PALETTE[4], "Spline" => PALETTE[5])

function bimodalFigure(methods, y; size=(400, 300), ylabel, scale=1, ylims=nothing)
    df = rows(readtable("bimodal_collo2.txt"), :Method, methods...)
    fig = bookFigure(size=size)
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel=ylabel)
    groupLines!(ax, df.Strike, df[!, y] .* scale, df.Method, order=collect(methods), colors=BIMODAL_COLORS)
    ylims === nothing || ylims!(ax, ylims...)
    bookLegend!(ax)
    return fig
end
@figure "density_bimodal_collo" bimodalFigure(("Cubic", "Kernel", "Nonic", "Quintic"), :Density, ylabel="Probability density", ylims=(0, 5))
@figure "density_bimodal_collo_s" bimodalFigure(("Kernel", "Spline"), :Density, ylabel="Probability density", ylims=(0, 5))
@figure "vol_bimodal_collo" bimodalFigure(("Kernel", "Quintic", "Nonic"), :Vol, size=(800, 400), ylabel=VOL_LABEL, scale=100)

## Andreasen-Huge discrete local volatility (rows: expiries, columns: log-strikes)

const AH_EXPIRIES = [0.175, 0.425, 0.695, 0.94, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

function discreteLocalVolFigure(file; size=(400, 300))
    m, header = readdlm(datapath(file), header=true)
    strikes = exp.(parse.(Float64, vec(header)))
    fig = bookFigure(size=size)
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Discrete local volatility")
    colors = cgrad(:Spectral_10, length(AH_EXPIRIES), categorical=true)
    for (i, expiry) in enumerate(AH_EXPIRIES)
        lines!(ax, strikes, Float64.(m[i, :]) .* 100, label=string(expiry), color=colors[i])
    end
    xlims!(ax, 450, 850)
    Legend(fig[1, 2], ax, "Maturity", framevisible=false)
    return fig
end
@figure "kahale_ah_constant_200_lm" discreteLocalVolFigure("kahale_ah_constant_200_lm.txt")
@figure "kahale_ah_constant_400_lm" discreteLocalVolFigure("kahale_ah_constant_400_lm.txt")
@figure "kahale_ah_linear_200_lm" discreteLocalVolFigure("kahale_ah_linear_200_lm.txt")
@figure "kahale_ah_linear_400_lm" discreteLocalVolFigure("kahale_ah_linear_400_lm.txt")

@figure "spx500_1y_2017_ah" begin
    df = readtable("spx500_1y_2017_ah.txt", squeeze=true)
    fig = bookFigure(size=(800, 400))
    ax = bookAxis(fig[1, 1], xlabel="Logarithm of asset price", ylabel="Local variance")
    lines!(ax, df.logspot, df.Linear .^ 2, label="Linear")
    lines!(ax, df.logspot, df.Huynh .^ 2, label="Huynh")
    xlims!(ax, 7, 8)
    bookLegend!(ax, title="Representation")
    fig
end

#compact view: first expiry of the Kahale surface for the piecewise constant and linear representations, and the linear
#and Hyman spline representations on the SPX500 1y options (data from ch07_discrete_ah.jl)
@figure "ah_discrete_lv" begin
    fig = bookFigure(size=(760, 360))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Discrete local volatility in %", title="SPX500 October 1995, T = 0.175")
    lv = readcsv("kahale_discrete_ah_lv.csv")
    for (i, (kind, m, label, style)) in enumerate((("constant", 200, "Constant, 200 points", :dash), ("constant", 400, "Constant, 400 points", :dot),
        ("linear", 200, "Linear, 200 points", :dash), ("linear", 400, "Linear, 400 points", :solid)))
        d = lv[(lv.Kind .== kind) .& (lv.Points .== m), :]
        lines!(ax, d.Strike, d.LocalVol .* 100, label=label, color=PALETTE[i], linestyle=style)
    end
    xlims!(ax, 500, 750)
    ylims!(ax, 0, 60)
    Legend(fig[0, 1], ax, orientation=:horizontal, nbanks=2, framevisible=false, tellwidth=false)
    df = readcsv("spx500_1y_2017_discrete_ah_lv.csv")
    ax2 = bookAxis(fig[1, 2], xlabel="Logarithm of asset price", ylabel="Local volatility in %", title="SPX500 March 2017, T = 1.01")
    for (i, (kind, label, style)) in enumerate((("linear", "Linear", :solid), ("hyman", "Hyman spline", :dash)))
        d = df[df.Kind .== kind, :]
        lines!(ax2, d.LogSpot, d.LocalVol .* 100, label=label, color=PALETTE[i+2], linestyle=style)
    end
    xlims!(ax2, 7, 8)
    Legend(fig[0, 2], ax2, orientation=:horizontal, nbanks=2, framevisible=false, tellwidth=false, valign=:bottom)
    fig
end
