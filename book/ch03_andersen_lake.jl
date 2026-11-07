# Chapter 3: Andersen-Lake (AL) American option pricing with term-structures of rates and proportional dividends
# (setups of test/fdm_test.jl, testset "ExerciseBoundary"). EUR and TRY discount curves as of Nov. 29, 2022.

using AQFED, AQFED.FDM, AQFED.TermStructure, PPInterpolation, ForwardDiff
const AM = AQFED.American

const EUR_TIMES = [0, 1, 2, 3, 35, 64, 93, 184, 275, 367, 735, 1099] ./ 365.0
const EUR_DF = [1.0, 1.00000232, 0.99998979, 0.9999798, 0.99937204, 0.99951607, 0.99782219, 0.99406162, 0.98825549, 0.98231822, 0.96108459, 0.94354772]
const TRY_TIMES = [0, 1, 8, 15, 22, 31, 63, 92, 184, 276, 367, 552, 734, 1098] ./ 365.0
const TRY_DF = [1.0, 0.9997808, 0.99743484, 0.99403111, 0.98925055, 0.98313541, 0.96031598, 0.92064074, 0.83544403, 0.75603321, 0.68572107, 0.56887972, 0.48014397, 0.35315213]
eurLogDF() = CubicSplineNatural(EUR_TIMES, log.(EUR_DF))
tryLogDF() = CubicSplineNatural(TRY_TIMES, log.(TRY_DF))

#the three examples: EUR discount with a -1% spread on the drift, TRY domestic / EUR foreign, flat rates
function alExample(name, σ)
    eur, try_ = eurLogDF(), tryLogDF()
    discount, drift = if name == "eur"
        AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(eur), AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(CubicSplineNatural(TRY_TIMES, eur.(TRY_TIMES) .+ 0.01 .* TRY_TIMES))
    elseif name == "try"
        AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(try_), AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(CubicSplineNatural(TRY_TIMES, try_.(TRY_TIMES) .- eur.(TRY_TIMES)))
    else
        ConstantRateCurve(0.20), ConstantRateCurve(0.19) #r = 20%, q = 1%
    end
    return TSBlackModel(FlatSurface(σ), discount, drift)
end

@compute "eur_try_rates" begin
    t = collect(range(1e-4, stop=2.0, length=401))
    out = DataFrame()
    for (ccy, pp, tmax) in (("EUR", eurLogDF(), 2.0), ("TRY", tryLogDF(), 1.0))
        tt = t[t.<=tmax]
        append!(out, DataFrame(Currency=ccy, Time=tt, Rate=-pp.(tt) ./ tt .* 100, ForwardRate=-ForwardDiff.derivative.(Ref(pp), tt) .* 100))
    end
    writetable("eur_try_rates.csv", out)
end
function rateFigure(ccy, col)
    df = readcsv("eur_try_rates.csv")
    df = df[df.Currency.==ccy, :]
    fig = bookFigure(size=(400, 200))
    ax = bookAxis(fig[1, 1], xlabel="Time in ACT/365", ylabel="Rate in %")
    lines!(ax, df.Time, df[!, col], color=PALETTE[1])
    fig
end
@figure "eur_rate" rateFigure("EUR", :Rate)
@figure "eur_forward_rate" rateFigure("EUR", :ForwardRate)
@figure "try_rate" rateFigure("TRY", :Rate)
@figure "try_forward_rate" rateFigure("TRY", :ForwardRate)

#l = p = 101 integration nodes as stated in the book (the test uses fewer). The TR-BDF2 reference (16000 space-steps, 8000 time-steps as in the book) is accurate to about 5e-6.
#price error of AL with a single polynomial (nPP=1) of m nodes and with m polynomial pieces of 5 nodes, against TR-BDF2
@compute "alpp_errors" begin
    tte, strike, spot = 1.0, 100.0, 100.0
    payoff = VanillaAmerican(false, strike, tte)
    out = DataFrame()
    #σ = 30% for all examples, as in the caption of fig:alpp_eur_errors
    for (name, σ) in (("eur", 0.3), ("try", 0.3), ("flat", 0.3))
        model = alExample(name, σ)
        ref = priceTRBDF2(payoff, spot, model, CapitalizedDividend{Float64}[], M=16001, N=8001, grid=SmoothlyDeformedGrid(CubicGrid(0.2)), solverName="LUUL", ndev=4)(spot)
        for m in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 31)
            al = AM.priceAmerican(AM.AndersenLakeRepresentation(model, tte, 1e-8, m, 32, 101, 101, isCall=false), strike, spot)
            alpp = AM.priceAmerican(AM.AndersenLakePPRepresentation(model, tte, false, atol=1e-8, nC=5, nIter=32, nTS1=101, nTS2=101, nPP=m), strike, spot)
            push!(out, (Example=name, m=m, AL=abs(al - ref), ALPP=abs(alpp - ref)), promote=true)
        end
    end
    writetable("alpp_errors.csv", out)
end
function alppErrorFigure(name)
    df = readcsv("alpp_errors.csv")
    df = df[df.Example.==name, :]
    #three panels side by side: taller than wide-format so that the y label fits, legend below to keep the curves visible
    fig = bookFigure(size=(300, 330))
    ax = bookAxis(fig[1, 1], xlabel="m", ylabel="Price error", yscale=log10)
    lines!(ax, df.m, df.AL .+ 1e-16, label="nₚₚ = 1")
    lines!(ax, df.m, df.ALPP .+ 1e-16, label="nₚₚ = m")
    Legend(fig[2, 1], ax, orientation=:horizontal, framevisible=false, tellheight=true)
    fig
end
@figure "alpp_eur_error" alppErrorFigure("eur")
@figure "alpp_try_error_vol20" alppErrorFigure("try")
@figure "alpp_flat_error_vol20" alppErrorFigure("flat")

#early-exercise boundaries: TR-BDF2 (reference), AL with one polynomial of m nodes, AL with m pieces of 5 nodes
@compute "alpp_boundaries" begin
    tte, strike, spot = 1.0, 100.0, 100.0
    t = collect(range(0.0, stop=tte, length=501))
    out = DataFrame()
    #σ = 30% with the EUR curve, σ = 20% for EUR/TRY, as in the caption of fig:alpp_eur
    for (name, σ, m) in (("eur", 0.3, 9), ("try", 0.2, 7))
        model = alExample(name, σ)
        payoffB = AQFED.FDM.VanillaAmericanWithExerciseBoundary(VanillaAmerican(false, strike, tte))
        priceTRBDF2(payoffB, spot, model, CapitalizedDividend{Float64}[], M=8001, N=4001, grid=SmoothlyDeformedGrid(CubicGrid(0.3)), solverName="LUUL")
        append!(out, DataFrame(Example=name, Method="TR-BDF2", Time=payoffB.exerciseTimes, Boundary=payoffB.exerciseBoundary))
        al = AM.AndersenLakeRepresentation(model, tte, 1e-8, m, 32, 101, 101, isCall=false)
        alpp = AM.AndersenLakePPRepresentation(model, tte, false, atol=1e-8, nC=5, nIter=32, nTS1=101, nTS2=101, nPP=m)
        append!(out, DataFrame(Example=name, Method="Andersen-Lake m=$m nₚₚ=1", Time=t, Boundary=AM.exerciseBoundary(al, strike, t)))
        append!(out, DataFrame(Example=name, Method="Andersen-Lake m=5 nₚₚ=$m", Time=t, Boundary=AM.exerciseBoundary(alpp, strike, t)))
    end
    writetable("alpp_boundaries.csv", out)
end
function boundaryFigure(file, example; xlabel="Time", size=(400, 300), position=:lt, ylims=nothing)
    df = readcsv(file)
    df = df[df.Example.==example, :]
    methods = unique(df.Method)
    order = vcat(filter(!=("TR-BDF2"), methods), "TR-BDF2")
    fig = bookFigure(size=size)
    ax = bookAxis(fig[1, 1], xlabel=xlabel, ylabel="Asset price")
    for (i, m) in enumerate(order)
        d = sort(df[(df.Method.==m).&isfinite.(df.Boundary), :], :Time)
        lines!(ax, d.Time, d.Boundary, label=m, color=m == "TR-BDF2" ? :black : PALETTE[i], linewidth=m == "TR-BDF2" ? 1.4 : 2.0,
            linestyle=m == "TR-BDF2" ? :dash : :solid)
    end
    ylims === nothing || ylims!(ax, ylims...)
    bookLegend!(ax, position=position)
    fig
end
@figure "alpp_eur_spread" boundaryFigure("alpp_boundaries.csv", "eur")
@figure "alpp_try_eur_vol20" boundaryFigure("alpp_boundaries.csv", "try")

#four proportional dividends of 1% (r = 3%, q = -7%): AL with m = 15 on the price with and without log transformation
@compute "al_prop_boundaries" begin
    tte, strike, spot = 1.0, 100.0, 100.0
    dividends = [AQFED.TermStructure.Dividend(0.01, t, t, true, false) for t in (0.2, 0.45, 0.70, 0.95)]
    cutDates = vcat(0.0, [d.exDate for d in dividends], tte)
    model = TSBlackModel(FlatSurface(0.3), ConstantRateCurve(0.03), ConstantRateCurve(0.03 + 0.07))
    payoffB = AQFED.FDM.VanillaAmericanWithExerciseBoundary(VanillaAmerican(false, strike, tte))
    priceTRBDF2(payoffB, spot, model, [CapitalizedDividend(d, 1.0) for d in dividends], M=8001, N=4001, grid=SmoothlyDeformedGrid(CubicGrid(0.3)), solverName="LUUL")
    t = collect(range(0.0, stop=tte, length=1001))
    out = DataFrame(Example=String[], Method=String[], Time=Float64[], Boundary=Float64[])
    for (example, transform) in (("log", AQFED.Math.LogTransformation()), ("nolog", AQFED.Math.IdentityTransformation{Float64}()))
        alg = AM.AndersenLakeGRepresentation(model, tte, false, nIter=32, collocation=AM.ChebyshevPPCollocation(cutDates, 15),
            quadrature=AM.TanhSinhPPQuadrature(cutDates, 51), dividends=dividends, transform=transform)
        append!(out, DataFrame(Example=example, Method="Andersen-Lake m=15", Time=t, Boundary=AM.exerciseBoundary(alg, spot, t)))
        append!(out, DataFrame(Example=example, Method="TR-BDF2", Time=payoffB.exerciseTimes, Boundary=payoffB.exerciseBoundary))
    end
    #one proportional dividend of 8% at 6.9 on a 7-year put, r = 6%
    tte = 7.0
    div7 = [AQFED.TermStructure.Dividend(0.08, 6.9, 6.9, true, false)]
    cut7 = [0.0, 6.9, tte]
    model7 = TSBlackModel(FlatSurface(0.3), ConstantRateCurve(0.06), ConstantRateCurve(0.06))
    payoff7 = AQFED.FDM.VanillaAmericanWithExerciseBoundary(VanillaAmerican(false, strike, tte))
    priceTRBDF2(payoff7, spot, model7, [CapitalizedDividend(d, 1.0) for d in div7], M=8001, N=4001, grid=SmoothlyDeformedGrid(CubicGrid(0.3)), solverName="LUUL")
    t7 = collect(range(0.0, stop=tte, length=1401))
    for m in (3, 5, 7)
        alg = AM.AndersenLakeGRepresentation(model7, tte, false, nIter=32, collocation=AM.ChebyshevPPCollocation(cut7, m),
            quadrature=AM.TanhSinhPPQuadrature(cut7, 51), dividends=div7)
        append!(out, DataFrame(Example="1div_t7", Method="Andersen-Lake m=$m", Time=t7, Boundary=AM.exerciseBoundary(alg, spot, t7)))
    end
    append!(out, DataFrame(Example="1div_t7", Method="TR-BDF2", Time=payoff7.exerciseTimes, Boundary=payoff7.exerciseBoundary))
    writetable("al_prop_boundaries.csv", out)
end
@figure "al_prop_log" boundaryFigure("al_prop_boundaries.csv", "log", position=:rt, ylims=(0, 300))
@figure "al_prop_nolog" boundaryFigure("al_prop_boundaries.csv", "nolog")
@figure "al_prop_1div_t7_boundary" boundaryFigure("al_prop_boundaries.csv", "1div_t7", size=(800, 300), position=:lb)
