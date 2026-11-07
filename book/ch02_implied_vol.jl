# Chapter 2: ThiopheneIV implied volatility solver along slices of fixed log-moneyness.
# Port of the figures of ~/mypapers/iv3/doc/scripts/make_mdpi_figures.py (Java data dump FigureDataDump).

using AQFED, SpecialFunctions
import AQFED.Black

const THIOPHENE_X_SLICES = [-1e-8, -0.5, -2.0, -8.0]

#log of the normalised OTM call price c = N(d1) - exp(-x) N(d2), and the Jaeckel normalised price beta = c exp(x/2), in BigFloat
function accurateNormalisedCall(x::Float64, v::Float64)
    setprecision(BigFloat, 512) do
        xb, vb = BigFloat(x), BigFloat(v)
        d1 = xb / vb + vb / 2
        Φ(z) = erfc(-z / sqrt(BigFloat(2))) / 2
        c = Φ(d1) - exp(-xb) * Φ(d1 - vb)
        c > 0 ? (Float64(log(c)), Float64(c), Float64(c * exp(xb / 2))) : (-Inf, 0.0, 0.0)
    end
end

@compute "thiophene_fixed_x_slices" begin
    df = DataFrame(x=Float64[], v_ref=Float64[], c=Float64[], err_seed=Float64[], err_cheb1=Float64[], err_cheb2=Float64[], err_cheb3=Float64[],
        err_jaeckel=Float64[], err_thiophene=Float64[], err_thiophene_plus=Float64[])
    relErr(v, vRef) = abs(v - vRef) / vRef
    for x in THIOPHENE_X_SLICES, vRef in exp10.(range(-5, log10(6.0), length=420))
        logc, c, beta = accurateNormalisedCall(x, vRef)
        (isfinite(logc) && c >= 1e-300 && c < 0.999) || continue
        #convergence of the seed and of the Euler-Chebyshev steps, applied on the whole range for the illustration
        v = Black.thiopheneChoiL3Guess(x, exp(x), c)
        (v > 1e-10 && isfinite(v)) || (v = 1e-10)
        errs = [relErr(v, vRef)]
        for _ = 1:3
            v = Black.thiopheneChebyshevLogPriceStep(x, v, logc)
            push!(errs, relErr(v, vRef))
        end
        jaeckel = Black.normalised_implied_volatility_from_a_transformed_rational_guess(beta, x, 1)
        push!(df, (x, vRef, c, errs..., relErr(jaeckel, vRef),
            relErr(Black.thiopheneTotalVolatility(x, c, logc), vRef), relErr(Black.thiopheneTotalVolatility(x, c, logc, plus=true), vRef)))
    end
    writetable("thiophene_fixed_x_slices.csv", df)
end

thiopheneSliceTitle(x) = abs(x) < 1e-7 ? L"x=-10^{-8}" : L"x=%$(x)"

decadeLabel(e) = e == 0 ? "1" : rich("10", superscript(string(e)))
const THIOPHENE_TOTAL_VOL_TICKS = logTicks([exp10.(-5:-1); 1.0; 5.0], [decadeLabel.(-5:-1); "1"; "5"])
decadeTicks(es) = logTicks(exp10.(es), decadeLabel.(es))

#2x2 panels of relative total-volatility errors against the reference total volatility, one panel per log-moneyness
function thiopheneSliceFigure(df, curves; ylims, yticks)
    fig = bookFigure(size=(720, 505))
    axes = Axis[]
    for (i, x) in enumerate(THIOPHENE_X_SLICES)
        r, col = fldmod1(i, 2)
        d = df[df.x.==x, :]
        ax = bookAxis(fig[r+1, col], xlabel=r == 2 ? L"Reference total volatility $v$" : "", ylabel=col == 1 ? "Relative total-volatility error" : "",
            title=thiopheneSliceTitle(x), xscale=log10, yscale=log10, xticks=THIOPHENE_TOTAL_VOL_TICKS, yticks=yticks)
        for (j, (key, label, linestyle)) in enumerate(curves)
            lines!(ax, d.v_ref, max.(d[!, key], 1e-18), color=PALETTE[j], linestyle=linestyle, label=label)
        end
        hlines!(ax, [eps()], color=:black, linewidth=0.75, linestyle=:dot)
        xlims!(ax, extrema(d.v_ref)...)
        ylims!(ax, ylims...)
        col > 1 && hideydecorations!(ax, grid=false)
        push!(axes, ax)
    end
    linkyaxes!(axes...)
    Legend(fig[1, 1:2], axes[1], orientation=:horizontal, framevisible=false)
    fig
end

@figure "fig-thiophene-fixed-x-convergence-slices" begin
    df = readcsv("thiophene_fixed_x_slices.csv")
    df = df[df.c.<0.99, :]
    thiopheneSliceFigure(df, [(:err_seed, "Choi L3 seed", :solid), (:err_cheb1, "Euler–Chebyshev 1", :dash),
            (:err_cheb2, "Euler–Chebyshev 2", :dashdot), (:err_cheb3, "Euler–Chebyshev 3", :solid)], ylims=(1e-18, 2e2), yticks=decadeTicks(-15:5:0))
end

@figure "fig-thiophene-jaeckel-comparison-slice" begin
    df = readcsv("thiophene_fixed_x_slices.csv")
    thiopheneSliceFigure(df, [(:err_jaeckel, "Jäckel", :dash), (:err_thiophene, "ThiopheneIV", :solid), (:err_thiophene_plus, "ThiopheneIV+", :dashdot)],
        ylims=(1e-18, 1e-10), yticks=decadeTicks(-18:2:-10))
end
