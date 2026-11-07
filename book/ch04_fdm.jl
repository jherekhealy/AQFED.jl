# Chapter 4: finite difference methods.

using AQFED, AQFED.FDM, AQFED.TermStructure, PPInterpolation

## Gamma of an American put on a uniform grid with different time-stepping schemes (test/fdm_test.jl, testset "AmericanPutGamma")
@compute "american_gamma_cnrkl" begin
    strike, spot, r, vol, tte = 110.0, 100.0, 0.01, 0.1, 0.5
    xSteps, tSteps, Smin, Smax = 1000, 100, 80.890436, 123.593012
    model = AQFED.TermStructure.TSBlackModel(FlatSurface(vol), ConstantRateCurve(r), ConstantRateCurve(r))
    dividends = Vector{CapitalizedDividend{Float64}}()
    payoff = KreissSmoothDefinition(VanillaAmerican(false, strike, tte))
    args = (payoff, spot, model, dividends)
    opts = (M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=UniformGrid(false), varianceConditioner=PartialExponentialFittingConditioner())
    solutions = [
        "Crank-Nicolson" => AQFED.FDM.priceTRBDF2(args...; opts..., timeSteppingName="CN"),
        "Rannacher" => AQFED.FDM.priceTRBDF2(args...; opts..., timeSteppingName="Rannacher4"),
        "TR-BDF2" => AQFED.FDM.priceTRBDF2(args...; opts...),
        "RKL" => AQFED.FDM.priceRKL2(args...; opts...),
        "RKG" => AQFED.FDM.priceRKG2(args...; opts...),
    ]
    out = vcat([DataFrame(Scheme=name, Spot=p.x, Price=p.(p.x), Gamma=PPInterpolation.evaluateSecondDerivative.(p, p.x)) for (name, p) in solutions]...)
    writetable("american_gamma_cnrkl.csv", out)
end

@figure "american_gamma_cnrkl" begin
    df = readcsv("american_gamma_cnrkl.csv")
    Smin, Smax = extrema(df.Spot)
    fig = bookFigure(size=(1024, 200))
    for (i, name) in enumerate(["Crank-Nicolson", "Rannacher", "TR-BDF2", "RKL", "RKG"])
        d = df[df.Scheme.==name, :]
        ax = bookAxis(fig[1, i], xlabel="Spot", ylabel=i == 1 ? "Gamma" : "", title=name, yticks=[0, 0.025, 0.05, 0.075], xticks=[90, 110])
        lines!(ax, d.Spot, d.Gamma, color=PALETTE[1])
        xlims!(ax, Smin, Smax)
        ylims!(ax, -0.005, 0.075)
        i > 1 && hideydecorations!(ax, grid=false, ticks=false)
    end
    fig
end

## Kreiss smoothing and grid deformation: error of the TR-BDF2 price against the number of space-steps m
## (test/fdm_test.jl, testset "DiscretKOTavellaRandall": σ = 20%, r = 7%, q = 2%, K = 100, T = 1, Smax = 150, 1500 time-steps)
function kreissErrors(payoff, payoffK, ref, Ms, gridnames; α, solverName="TDMA")
    spot, σ, r, q = 100.0, 0.2, 0.07, 0.02
    model = TSBlackModel(FlatSurface(σ), ConstantRateCurve(r), ConstantRateCurve(r - q))
    divs = CapitalizedDividend{Float64}[]
    grids = Dict("Uniform" => UniformGrid(false), "Cubic" => CubicGrid(α), "Sinh" => SinhGrid(α))
    price(p, g, M) = priceTRBDF2(p, spot, model, divs, M=M, N=1501, Smin=0.0, Smax=150.0, grid=g, solverName=solverName)(spot)
    refPrice = priceTRBDF2(ref, spot, model, divs, M=16001, N=1501, Smin=0.0, Smax=150.0, grid=SmoothlyDeformedGrid(UniformGrid(false)), solverName=solverName)(spot)
    out = DataFrame(m=Int[], grid=String[], error=Float64[])
    for M in Ms, name in gridnames
        g = grids[name]
        push!(out, (M, name, price(payoff, g, M) - refPrice))
        push!(out, (M, name * "-Kreiss", price(payoffK, g, M) - refPrice))
        push!(out, (M, name * "-Deformed", price(payoff, SmoothlyDeformedGrid(g), M) - refPrice))
    end
    return out
end
@compute "kreiss_time_vanilla" begin
    v = VanillaEuropean(true, 100.0, 1.0)
    a2 = kreissErrors(v, KreissSmoothDefinition(v), v, 250:501, ["Uniform", "Cubic"], α=2.0 / 150)
    a20 = kreissErrors(v, KreissSmoothDefinition(v), v, 250:501, ["Cubic"], α=20.0 / 150)
    a20.grid .= replace.(a20.grid, "Cubic" => "Cubic20")
    writetable("kreiss_time_vanilla.csv", vcat(a2, a20))
end
@compute "kreiss_time_american" begin
    v = VanillaAmerican(false, 100.0, 1.0)
    a2 = kreissErrors(v, KreissSmoothDefinition(v), v, 250:501, ["Uniform", "Cubic", "Sinh"], α=2.0 / 150, solverName="LUUL")
    a20 = kreissErrors(v, KreissSmoothDefinition(v), v, 250:501, ["Cubic", "Sinh"], α=20.0 / 150, solverName="LUUL")
    a2.alpha .= 2; a20.alpha .= 20
    writetable("kreiss_time_american.csv", vcat(a2, a20))
end
#color by treatment (raw, Kreiss smoothing, grid deformation), line style by grid family
function kreissFigure(df, prefixes; familyLabels=Dict{String,String}(), gridLegend=false)
    #a grid name is a family (one of prefixes) followed by an optional treatment suffix
    familyOf(g) = first(p for p in sort(prefixes, by=length, rev=true) if g == p || startswith(g, p * "-"))
    df = df[[any(g == p || startswith(g, p * "-") for p in prefixes) for g in df.grid], :]
    fams = [f for f in prefixes if any(familyOf.(df.grid) .== f)]
    fig = bookFigure(size=(400, 330))
    ax = bookAxis(fig[1, 1], xlabel="Number of space-steps", ylabel="Price error", yscale=log10)
    familyStyle = Dict("Uniform" => :solid, "Cubic" => :dash, "Sinh" => :dot, "Cubic20" => :dot)
    treatment = [("", "Raw", PALETTE[1]), ("-Kreiss", "Kreiss", PALETTE[2]), ("-Deformed", "Deformed", PALETTE[3])]
    for (suffix, _, color) in treatment, family in fams
        d = sort(df[df.grid.==family*suffix, :], :m)
        isempty(d) && continue
        lines!(ax, d.m, abs.(d.error) .+ 1e-12, color=color, linestyle=familyStyle[family], linewidth=suffix == "" ? 1.2 : 2.2)
    end
    treatmentEntries = ([LineElement(color=c) for (_, _, c) in treatment], [n for (_, n, _) in treatment])
    legendStyle = (framevisible=false, patchsize=(18, 10), labelsize=14, titlesize=14, tellwidth=false)
    if length(fams) == 1 && !gridLegend #a single row, without title, fits in the width of a half-page panel
        Legend(fig[2, 1], treatmentEntries...; orientation=:horizontal, colgap=10, legendStyle...)
    else #treatment and grid legends side by side, each in a column
        legends = fig[2, 1] = GridLayout()
        Legend(legends[1, 1], treatmentEntries..., "Treatment"; rowgap=0, legendStyle...)
        Legend(legends[1, 2], [LineElement(color=:black, linestyle=familyStyle[f]) for f in fams], [get(familyLabels, f, f) for f in fams], "Grid"; rowgap=0, legendStyle...)
    end
    rowgap!(fig.layout, 4)
    fig
end
#same legend layout as the cubic panel next to it, so that both plots have the same height
@figure "uniform_kreiss_time_vanilla" kreissFigure(readcsv("kreiss_time_vanilla.csv"), ["Uniform"], gridLegend=true)
@figure "cubic_kreiss_time_vanilla" kreissFigure(readcsv("kreiss_time_vanilla.csv"), ["Cubic", "Cubic20"], familyLabels=Dict("Cubic" => "Cubic, α=2", "Cubic20" => "Cubic, α=20"))

## error against the position of the strike within a cell of a fixed uniform grid: the strike moves from one grid point
## (θ=0) to the next (θ=1), the grid does not move (raw, Kreiss) or is deformed to place the strike in the middle.
@compute "kreiss_strike_position" begin
    spot, σ, r, q, Smax = 100.0, 0.2, 0.07, 0.02, 150.0
    model = TSBlackModel(FlatSurface(σ), ConstantRateCurve(r), ConstantRateCurve(r - q))
    divs = CapitalizedDividend{Float64}[]
    out = DataFrame(m=Int[], theta=Float64[], grid=String[], error=Float64[])
    for M in (250, 500)
        h = Smax / (M - 1) #M grid points
        k0 = h * floor(100.0 / h)
        for θ in range(0.0, 1.0, length=41)
            v = VanillaEuropean(true, k0 + θ * h, 1.0)
            ref = priceTRBDF2(v, spot, model, divs, M=16001, N=1501, Smin=0.0, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))(spot)
            price(p, g) = priceTRBDF2(p, spot, model, divs, M=M, N=1501, Smin=0.0, Smax=Smax, grid=g)(spot)
            push!(out, (M, θ, "Raw", price(v, UniformGrid(false)) - ref))
            push!(out, (M, θ, "Kreiss", price(KreissSmoothDefinition(v), UniformGrid(false)) - ref))
            push!(out, (M, θ, "Deformed", price(v, SmoothlyDeformedGrid(UniformGrid(false))) - ref))
        end
    end
    writetable("kreiss_strike_position.csv", out)
end
@figure "kreiss_strike_position" begin
    df = readcsv("kreiss_strike_position.csv")
    fig = bookFigure(size=(800, 330))
    names = ["Raw", "Kreiss", "Deformed"]
    colors = Dict(n => PALETTE[i] for (i, n) in enumerate(names))
    for (j, M) in enumerate((250, 500))
        d = df[df.m.==M, :]
        ax = bookAxis(fig[1, j], xlabel="Strike position θ", ylabel=j == 1 ? "Price error" : "", title="m = $M")
        hlines!(ax, [0.0], color=:black, linewidth=0.8)
        groupLines!(ax, d.theta, d.error, d.grid; order=names, colors=colors)
    end
    Legend(fig[2, :], [LineElement(color=colors[n]) for n in names], names, orientation=:horizontal, framevisible=false, tellwidth=false)
    fig
end
@figure "kreiss_time_american" (d = readcsv("kreiss_time_american.csv"); kreissFigure(d[d.alpha.==2, :], ["Uniform", "Cubic", "Sinh"]))
@figure "kreiss_time_american_alpha20" (d = readcsv("kreiss_time_american.csv"); kreissFigure(vcat(d[(d.alpha.==2).&startswith.(d.grid, "Uniform"), :], d[d.alpha.==20, :]), ["Uniform", "Cubic", "Sinh"]))

## grid points of the sinh (Tavella-Randall) and cubic stretching (mypapers/stretching/stretching.jl)
function gridPointsFigure(file, cols, labels, levels; derivative=false)
    df = readtable(file)
    fig = bookFigure(size=(400, 300))
    ax = bookAxis(fig[1, 1], xlabel="u", ylabel=derivative ? "dS/du" : "S")
    markers = [:circle, :diamond, :cross]
    for (i, (c, l)) in enumerate(zip(cols, labels))
        y = df[!, c]
        x, v = derivative ? (df.u[2:end], diff(y) ./ diff(df.u)) : (df.u, y)
        scatter!(ax, x, v, label=l, color=(PALETTE[i], 0.75), marker=markers[i], markersize=8)
    end
    derivative || hlines!(ax, levels, color=(:black, 0.6), linestyle=:dash, linewidth=1)
    bookLegend!(ax, position=derivative ? :rt : :rb)
    fig
end
@figure "sinh_cubic_points" gridPointsFigure("sinh_cubic_points.txt", [:Sinh, :Cubic, :CubicMatching], ["Sinh", "Cubic", "Cubic α=0.9"], [125])
@figure "sinh_cubic_points_multi" gridPointsFigure("sinh_cubic_points_multi.txt", [:Sinh, :Cubic], ["Tavella Randall", "Piecewise Cubic"], [90, 102, 110])
@figure "sinh_cubic_points_multi_der" gridPointsFigure("sinh_cubic_points_multi.txt", [:Sinh, :Cubic], ["Tavella Randall", "Piecewise Cubic"], [], derivative=true)

## logarithmic grid on the direct PDE versus uniform grid on the log-transformed PDE (test/fdm_test.jl, testset "GridShift"):
## European and American puts, T = 0.5833, S = 40, σ = 40%, r = 4.88%, 201 space-steps, 50 time-steps, S ∈ [15, 100].
## Grids without deformation use the Kreiss smoothing. The reference is the Black-Scholes formula (European) or the
## Andersen-Lake technique with fine settings (American).
const SCHEME_STRIKES = collect(25.0:0.25:55.0)
function schemeErrors(isAmerican::Bool)
    r, tte, spot, σ = 0.0488, 0.5833, 40.0, 0.4
    M, N, Smin, Smax, α = 201, 51, 15.0, 100.0, 0.1
    model = TSBlackModel(FlatSurface(σ), ConstantRateCurve(r), ConstantRateCurve(r))
    divs = CapitalizedDividend{Float64}[]
    opts = (M=M, N=N, Smin=Smin, Smax=Smax, solverName="LUUL", calibration=NoCalibration())
    direct = ["Uniform" => UniformGrid(false), "Uniform-Deformed" => SmoothlyDeformedGrid(UniformGrid(false)),
        "LogGrid" => LogGrid(false), "LogGrid-Deformed" => SmoothlyDeformedGrid(LogGrid(false)), "Cubic-0.1" => CubicGrid(α)]
    logScheme = ["LogScheme" => UniformGrid(false), "LogScheme-Deformed" => SmoothlyDeformedGrid(UniformGrid(false)),
        "LogScheme-Cubic" => CubicGrid(α)]
    al = AQFED.American.AndersenLakeRepresentation(AQFED.TermStructure.ConstantBlackModel(σ, r, 0.0), tte, 1e-10, 32, 32, 101, 201)
    out = DataFrame(Strike=Float64[], grid=String[], error=Float64[])
    for strike in SCHEME_STRIKES
        payoff = isAmerican ? VanillaAmerican(false, strike, tte) : VanillaEuropean(false, strike, tte)
        payoffK = KreissSmoothDefinition(payoff)
        ref = isAmerican ? AQFED.American.priceAmerican(al, strike, spot) :
              AQFED.Black.blackScholesFormula(false, strike, spot, σ^2 * tte, exp(-r * tte), exp(-r * tte))
        for (name, g) in direct
            p = g isa SmoothlyDeformedGrid ? payoff : payoffK
            push!(out, (strike, name, priceTRBDF2(p, spot, model, divs; opts..., grid=g)(spot) - ref))
        end
        for (name, g) in logScheme
            p = g isa SmoothlyDeformedGrid ? payoff : payoffK
            push!(out, (strike, name, priceLogTRBDF2(p, spot, model, divs; opts..., grid=g)(spot) - ref))
        end
        #what the original study computed under the label LogScheme-Deformed: no smoothing and no deformation
        push!(out, (strike, "LogScheme-Raw", priceLogTRBDF2(payoff, spot, model, divs; opts..., grid=UniformGrid(false))(spot) - ref))
    end
    return out
end
@compute "scheme_errors" begin
    eu, am = schemeErrors(false), schemeErrors(true)
    eu.option .= "European"; am.option .= "American"
    writetable("scheme_errors.csv", vcat(eu, am))
end
function schemeFigure(option, names)
    df = readcsv("scheme_errors.csv")
    df = df[df.option.==option, :]
    fig = bookFigure(size=(800, 300))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Absolute error in price")
    #same color for a given discretization in all four figures
    colors = Dict(n => PALETTE[mod1(i, length(PALETTE))] for (i, n) in enumerate(unique(vcat(UNIFORM_SCHEME_NAMES, LOG_SCHEME_NAMES))))
    groupLines!(ax, df.Strike, abs.(df.error), df.grid; order=names, colors=colors)
    Legend(fig[1, 2], ax, framevisible=false)
    fig
end
const LOG_SCHEME_NAMES = ["LogGrid", "LogGrid-Deformed", "LogScheme", "LogScheme-Deformed", "LogScheme-Cubic"]
const UNIFORM_SCHEME_NAMES = ["Uniform", "Uniform-Deformed", "LogGrid", "LogGrid-Deformed", "Cubic-0.1"]
@figure "loggrid_scheme_european" schemeFigure("European", LOG_SCHEME_NAMES)
@figure "uniform_scheme_european" schemeFigure("European", UNIFORM_SCHEME_NAMES)
@figure "loggrid_scheme_american" schemeFigure("American", LOG_SCHEME_NAMES)
@figure "uniform_scheme_american" schemeFigure("American", UNIFORM_SCHEME_NAMES)

## error profile against the spot for the put of the previous study (K = 40): European error, early-exercise part
## (American error minus European error) and American error. 501 time-steps so that the error is mostly spatial.
@compute "american_error_spot" begin
    r, tte, σ, K = 0.0488, 0.5833, 0.4, 40.0
    M, N, Smin, Smax, α = 201, 501, 15.0, 100.0, 0.1
    model = TSBlackModel(FlatSurface(σ), ConstantRateCurve(r), ConstantRateCurve(r))
    divs = CapitalizedDividend{Float64}[]
    al = AQFED.American.AndersenLakeRepresentation(AQFED.TermStructure.ConstantBlackModel(σ, r, 0.0), tte, 1e-10, 32, 32, 101, 201)
    opts = (M=M, N=N, Smin=Smin, Smax=Smax, solverName="LUUL")
    grids = ["Uniform" => UniformGrid(false), "Uniform-Deformed" => SmoothlyDeformedGrid(UniformGrid(false)),
        "LogGrid" => LogGrid(false), "Cubic-0.1" => CubicGrid(α)]
    spots = collect(20.0:0.25:70.0) #S_max = 100 affects the error above 70
    out = DataFrame(Spot=Float64[], grid=String[], European=Float64[], American=Float64[])
    for (name, g) in grids
        pe, pa = VanillaEuropean(false, K, tte), VanillaAmerican(false, K, tte)
        if !(g isa SmoothlyDeformedGrid)
            pe, pa = KreissSmoothDefinition(pe), KreissSmoothDefinition(pa)
        end
        fe = priceTRBDF2(pe, 40.0, model, divs; opts..., grid=g)
        fa = priceTRBDF2(pa, 40.0, model, divs; opts..., grid=g)
        for S in spots
            refE = AQFED.Black.blackScholesFormula(false, K, S, σ^2 * tte, exp(-r * tte), exp(-r * tte))
            refA = AQFED.American.priceAmerican(al, K, S)
            push!(out, (S, name, fe(S) - refE, fa(S) - refA))
        end
    end
    writetable("american_error_spot.csv", out)
end
@figure "american_error_spot" begin
    df = readcsv("american_error_spot.csv")
    names = ["Uniform", "Uniform-Deformed", "LogGrid", "Cubic-0.1"]
    colors = Dict(n => PALETTE[i] for (i, n) in enumerate(names))
    fig = bookFigure(size=(800, 420))
    panels = [("European", df.European), ("Early-exercise part", df.American .- df.European), ("American", df.American)]
    axes = Axis[]
    for (j, (title, y)) in enumerate(panels)
        ax = bookAxis(fig[1, j], xlabel="Spot", ylabel=j == 1 ? "Price error" : "", title=title)
        hlines!(ax, [0.0], color=:black, linewidth=0.8)
        vlines!(ax, [40.0], color=:gray, linewidth=0.8, linestyle=:dash)
        groupLines!(ax, df.Spot, y, df.grid; order=names, colors=colors)
        push!(axes, ax)
    end
    linkyaxes!(axes...)
    Legend(fig[2, :], [LineElement(color=colors[n]) for n in names], names, orientation=:horizontal, framevisible=false, tellwidth=false)
    fig
end
