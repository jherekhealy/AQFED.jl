# Chapter 10: finite difference pricing of barrier options with ghost points, Kreiss smoothing or a deformed grid
# (setups of test/fdmbarrier_test.jl). Uniform grid, or cubic grid concentrated around the strike with α = 0.1 ("cubic01").

using AQFED, AQFED.FDM, AQFED.TermStructure, PPInterpolation

const BARRIER_SPOT = 6317.80
barrierGrid(kind) = kind == "uniform" ? UniformGrid(false) : CubicGrid(0.1)

#knock-out, knock-in, and knock-in then knock-out up-and-in down-and-out call options
function barrierSetup(kind)
    S, tte, σ = BARRIER_SPOT, 1.0, 0.2
    strike, up, down = 0.8S, 1.2S, 0.9S
    r, q = kind == :ko ? (0.01, -0.02) : (0.0, 0.0)
    model = TSBlackModel(FlatSurface(σ), ConstantRateCurve(r), ConstantRateCurve(r - q))
    v, vk = VanillaEuropean(true, strike, tte), KreissSmoothDefinition(VanillaEuropean(true, strike, tte))
    payoff, payoffK, analytic = if kind == :ko
        AQFED.FDM.ContinuousKO(v, up, false, 0.0, 0.0, tte), AQFED.FDM.ContinuousKO(vk, up, false, 0.0, 0.0, tte),
        s -> AQFED.Black.calculate(AQFED.Black.AnalyticBarrierEngine(s, σ^2 * tte, exp(-(r - q) * tte), exp(-r * tte), exp(-r * tte)), AQFED.Black.BarrierPayoff(true, strike, up, 0.0, AQFED.Black.UpOut))
    elseif kind == :ki
        AQFED.FDM.ContinuousKI(v, up, false, 0.0, 0.0, tte), AQFED.FDM.ContinuousKI(vk, up, false, 0.0, 0.0, tte),
        s -> AQFED.Black.calculate(AQFED.Black.AnalyticBarrierEngine(s, σ^2 * tte, 1.0, 1.0, 1.0), AQFED.Black.BarrierPayoff(true, strike, up, 0.0, AQFED.Black.UpIn))
    else
        AQFED.FDM.ContinuousKI(AQFED.FDM.ContinuousKO(v, down, true, 0.0, 0.0, tte), up, false, 0.0, 0.0, tte),
        AQFED.FDM.ContinuousKI(AQFED.FDM.ContinuousKO(vk, down, true, 0.0, 0.0, tte), up, false, 0.0, 0.0, tte),
        s -> AQFED.Black.calculateUpInDownOutEnd(AQFED.Black.AnalyticBarrierEngine(s, σ^2 * tte, 1.0, 1.0, 1.0), true, strike, down, up)
    end
    #knock-out: worthless above the barrier, the grid ends at 2L+; knock-in: a vanilla above the barrier, the grid is truncated
    #as for a vanilla, at 4 standard deviations (Smax = 0 lets the pricer choose)
    Smax = kind == :ko ? 2up : 0.0
    return (; S, strike, model, payoff, payoffK, analytic, Smax, spotMax=kind == :kiko ? 1.19 : 1.2)
end

@compute "barrier_ghost" begin
    out = DataFrame()
    for kind in (:ko, :ki, :kiko), grid in ("uniform", "cubic01")
        b = barrierSetup(kind)
        g = barrierGrid(grid)
        price(p, gr, M) = priceTRBDF2(p, b.S, b.model, CapitalizedDividend{Float64}[], M=M, N=1001, Smin=0.0, Smax=b.Smax, grid=gr)
        ghost, kreiss, deformed = price(b.payoff, g, 401), price(b.payoffK, g, 401), price(b.payoff, SmoothlyDeformedGrid(g), 401)
        spots = collect(range(0.5b.S, stop=b.spotMax * b.S, length=101))
        spots = spots[spots.<0.999 * 1.2b.S]
        an = b.analytic.(spots)
        for (name, p) in ("Ghost" => ghost, "Ghost Kreiss" => kreiss, "Deformed" => deformed)
            append!(out, DataFrame(Kind=string(kind), Grid=grid, Plot="spot", Method=name, X=spots ./ b.strike, Error=abs.(p.(spots) .- an)))
        end
        if kind == :ko
            an0 = b.analytic(b.strike)
            steps = collect(201:301)
            for (name, mk) in ("Ghost" => M -> price(b.payoff, g, M), "Ghost Kreiss" => M -> price(b.payoffK, g, M), "Deformed" => M -> price(b.payoff, SmoothlyDeformedGrid(g), M))
                append!(out, DataFrame(Kind=string(kind), Grid=grid, Plot="steps", Method=name, X=Float64.(steps), Error=[abs(mk(M)(b.strike) - an0) for M in steps]))
            end
        end
    end
    writetable("barrier_ghost.csv", out)
end

#one-touch and KIKO prices with RKG, TR-BDF2 and BDF1 Richardson extrapolation (tables tbl:onetouch_error and tbl:kiko_error):
#N time-steps and N space-steps (N+1 points), deformed grids such that the barriers are on the grid, minimum time of
#repeated runs (machine dependent)
@compute "barrier_scheme_timing" begin
    S, tte, σ = BARRIER_SPOT, 1.0, 0.2
    strike, up, down = 0.8S, 1.2S, 0.9S
    model = TSBlackModel(FlatSurface(σ), ConstantRateCurve(0.0), ConstantRateCurve(0.0))
    divs = CapitalizedDividend{Float64}[]
    engine = AQFED.Black.AnalyticBarrierEngine(S, σ^2 * tte, 1.0, 1.0, 1.0)
    call = VanillaEuropean(true, strike, tte)
    products = [
        ("One-touch", AQFED.FDM.ContinuousKO(Binary(true, up, tte), up, false, 1.0, 0.0, tte), AQFED.Black.calculateOneTouch(engine, false, up, 1.0)),
        ("KIKO", AQFED.FDM.ContinuousKI(AQFED.FDM.ContinuousKO(call, down, true, 0.0, 0.0, tte), up, false, 0.0, 0.0, tte), AQFED.Black.calculateUpInDownOutEnd(engine, true, strike, down, up))]
    bdf1(p, n, nt, g) = priceTRBDF2(p, S, model, divs, M=n + 1, N=nt + 1, Smin=0.0, Smax=0.0, grid=g, timeSteppingName="BDF1")(S)
    schemes = [
        "RKG" => (p, n, g) -> AQFED.FDM.priceRKG2(p, S, model, divs, M=n + 1, N=n + 1, Smin=0.0, Smax=0.0, grid=g)(S),
        "TR-BDF2" => (p, n, g) -> priceTRBDF2(p, S, model, divs, M=n + 1, N=n + 1, Smin=0.0, Smax=0.0, grid=g)(S),
        "BDF1 Richardson" => (p, n, g) -> 2bdf1(p, n, 2n, g) - bdf1(p, n, n, g)] #twice the number of time-steps
    function bestTime(f; minTime=2.0)
        f()
        t, total = Inf, 0.0
        while total < minTime
            e = @elapsed f()
            t, total = min(t, e), total + e
        end
        return t
    end
    out = DataFrame()
    for (product, p, reference) in products, (grid, g) in ("Uniform" => SmoothlyDeformedGrid(UniformGrid(false)), "Cubic" => SmoothlyDeformedGrid(CubicGrid(0.01))), n in (100, 400), (scheme, price) in schemes
        append!(out, DataFrame(Product=product, Grid=grid, N=n, Scheme=scheme, Reference=reference, Error=abs(price(p, n, g) - reference), TimeMs=1e3 * bestTime(() -> price(p, n, g))))
    end
    writetable("barrier_scheme_timing.csv", out)
end

#error against the spot (M = 401) and, for the up-and-out, against the number of space steps (at-the-money), uniform and
#cubic grid side by side with shared axes per row, a single legend on top
function barrierGhostGridFigure(kind, plots; size=(767, 300 + 260 * (length(plots) - 1)))
    df = readcsv("barrier_ghost.csv")
    df = df[df.Kind.==kind, :]
    methods = ["Ghost", "Ghost Kreiss", "Deformed"]
    fig = bookFigure(size=size)
    for (r, plot) in enumerate(plots), (c, grid) in enumerate(("uniform", "cubic01"))
        d = df[(df.Grid.==grid).&(df.Plot.==plot), :]
        ax = bookAxis(fig[r+1, c], xlabel=plot == "spot" ? "Spot / strike" : rich("Number of space steps ", rich("M", font=:italic)),
            ylabel=c == 1 ? "Absolute error in price" : "",
            title=r == 1 ? (grid == "uniform" ? "Uniform grid" : rich("Cubic grid, ", rich("α", font=:italic), " = 0.1")) : "", titlealign=:left)
        #line styles keep coinciding curves visible (Ghost and Ghost Kreiss for the up-and-in); the error of the ghost point
        #oscillates with the parity of M: thinner solid lines for the steps
        groupLines!(ax, d.X, d.Error, d.Method, order=methods, linewidth=plot == "steps" ? 1.2 : 2.0, linestyles=plot == "spot")
        c == 2 && linkyaxes!(contents(fig[r+1, 1])[1], ax)
        c == 2 && hideydecorations!(ax, grid=false, ticks=false)
        r == 1 && c == 1 && Legend(fig[1, 1:2], ax, orientation=:horizontal, tellwidth=false, tellheight=true)
    end
    rowgap!(fig.layout, 1, 4)
    fig
end
@figure "barrier_ko_ghost" barrierGhostGridFigure("ko", ("spot", "steps"), size=(767, 560))
@figure "barrier_ki_ghost" barrierGhostGridFigure("ki", ("spot",))
@figure "barrier_kiko_ghost" barrierGhostGridFigure("kiko", ("spot",))

## double one-touch (double no-touch paying 1) of one week: Δ with different time-stepping schemes
@compute "doubleonetouch_delta" begin
    S, up, down, tte, σ = 100.0, 101.0, 99.9, 1.0 / 52, 0.2
    model = TSBlackModel(FlatSurface(σ), ConstantRateCurve(0.0), ConstantRateCurve(0.0))
    payoff = AQFED.FDM.ContinuousDKO(AQFED.FDM.Binary(true, up, tte), up, down, 1.0, 0.0, tte)
    opts = (M=1501, N=201, Smin=0.0, Smax=0.0, grid=SmoothlyDeformedGrid(CubicGrid(0.01)))
    divs = CapitalizedDividend{Float64}[]
    schemes = ["TR-BDF2" => priceTRBDF2(payoff, S, model, divs; opts...), "RKG" => AQFED.FDM.priceRKG2(payoff, S, model, divs; opts...),
        "Crank-Nicolson" => priceTRBDF2(payoff, S, model, divs; opts..., timeSteppingName="CN"),
        "Rannacher" => priceTRBDF2(payoff, S, model, divs; opts..., timeSteppingName="Rannacher"),
        "Rannacher4" => priceTRBDF2(payoff, S, model, divs; opts..., timeSteppingName="Rannacher4")]
    out = vcat([(x = filter(x -> down <= x <= up, p.x); DataFrame(Scheme=n, Spot=x, Delta=PPInterpolation.evaluateDerivative.(p, x))) for (n, p) in schemes]...)
    writetable("doubleonetouch_delta.csv", out)
end
function doubleOneTouchFigure(schemes)
    df = readcsv("doubleonetouch_delta.csv")
    fig = bookFigure(size=(400, 200))
    ax = bookAxis(fig[1, 1], xlabel="Underlying spot price", ylabel="Option Δ")
    groupLines!(ax, df.Spot, df.Delta, df.Scheme, order=schemes)
    length(schemes) > 1 && bookLegend!(ax, position=:rb)
    fig
end
@figure "doubleonetouch_delta_trbdf2" doubleOneTouchFigure(["TR-BDF2", "RKG"])
@figure "doubleonetouch_delta_cn" doubleOneTouchFigure(["Crank-Nicolson"])
@figure "doubleonetouch_delta_ran" doubleOneTouchFigure(["Rannacher"])
@figure "doubleonetouch_delta_ran4" doubleOneTouchFigure(["Rannacher4"])
