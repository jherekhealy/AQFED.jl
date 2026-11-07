# Chapter 5: basket options.
# deelstra_vol100_err.pdf is regenerated below (deelstra_vol100_err) with a quasi Monte-Carlo reference.

using AQFED, AQFED.Basket, AQFED.Bachelier, AQFED.Collocation, PPInterpolation, LinearAlgebra

#SABR basket approximations against the 2D quadrature, on two assets with the normal SABR fit to a flat 20% Black smile
@compute "deelstra_correlation_sabr" begin
    vols = fill(0.2, 10)
    tte, r, q, spot = 1.0, 0.05, 0.0, 1.0
    f = spot * exp((r - q) * tte)
    strikes = [0.85, 0.90, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4] .* f
    discountFactor = exp(-r * tte)
    prices, _ = Collocation.weightedPrices(true, strikes, vols, ones(length(vols)), f, discountFactor, tte, vegaFloor=1e-7)
    bvols = Bachelier.bachelierImpliedVolatility.(prices, true, strikes, tte, f, discountFactor)
    bvolSpline = CubicSplineNatural(log.(strikes ./ f), bvols)
    guess = AQFED.VolatilityModels.initialGuessNormalATM(f, tte, 0.0, bvolSpline(0.0), PPInterpolation.evaluateDerivative(bvolSpline, 0.0), PPInterpolation.evaluateSecondDerivative(bvolSpline, 0.0))
    section = AQFED.VolatilityModels.calibrateNormalSABRSectionFromGuess(tte, f, strikes, bvols, ones(length(bvols)), guess)
    w = [0.5, 0.5]
    tvar = fill(0.2^2 * tte, 2)
    spots = ones(2)
    forwards = spots .* exp((r - q) * tte)
    strike = 1.1
    ρ = section.params.ρ
    out = DataFrame(rho=Float64[], reference=Float64[], sabrSingle=Float64[], sabrIndependent=Float64[], Ju=Float64[], Deelstra=Float64[], CurranLB=Float64[])
    #the 2D quadrature reference is inaccurate for a singular correlation matrix (|ρ| = 1)
    for rho in range(-0.999, stop=0.999, length=81)
        correlation = [1.0 rho; rho 1.0]
        psiIndep = diagm([ρ, ρ])
        correlationIndep = [correlation psiIndep; psiIndep' Matrix(1.0I, 2, 2)]
        refPrice = Basket.priceEuropean(Basket.QuadBasketPricer(AQFED.Math.GaussLegendre(128)), true, strike, discountFactor, spots, forwards, tvar, w, correlation, isSplit=true)
        price = Basket.priceEuropean(Basket.SABRBasketPricer{Basket.SingleStochasticVol}(), true, strike, discountFactor, spots, forwards, [section, section], w, correlation)
        priceI = Basket.priceEuropean(Basket.SABRBasketPricer{Basket.FullCorrelation}(), true, strike, discountFactor, spots, forwards, [section, section], w, correlationIndep)
        ju = Basket.priceEuropean(JuBasketPricer(), true, strike, discountFactor, spots, forwards, tvar, w, correlation)
        #64 points so that the quadrature error (around 1e-5 with the default 21 points) does not hide the approximation error
        deelstra = Basket.priceEuropean(DeelstraBasketPricer(2, 3, AQFED.Math.GaussLegendre(64)), true, strike, discountFactor, spots, forwards, tvar, w, correlation)
        lb = Basket.priceEuropean(DeelstraLBBasketPricer(2, 3), true, strike, discountFactor, spots, forwards, tvar, w, correlation)
        push!(out, (rho, refPrice, price, priceI, ju, deelstra, lb))
    end
    writetable("deelstra_correlation_sabr.csv", out)
end

@figure "deelstra_correlation" begin
    sabr = readcsv("deelstra_correlation_sabr.csv")
    fig = bookFigure(size=(400, 400))
    ax = bookAxis(fig[1, 1], xlabel="Correlation", ylabel="Relative error", yscale=log10,
        yticks=([1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], ["10⁻¹⁰", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "1"]))
    for (i, (col, label)) in enumerate((("Ju", "Ju"), ("Deelstra", "Deelstra"), ("CurranLB", "Curran LB"),
        ("sabrSingle", "SABR single"), ("sabrIndependent", "SABR indep.")))
        y = abs.(sabr[!, col] ./ sabr.reference .- 1)
        keep = isfinite.(y) .& (y .> 0)
        lines!(ax, sabr.rho[keep], y[keep], label=label, color=PALETTE[i])
    end
    ylims!(ax, 1e-10, 1)
    #compact legend in the empty lower left corner
    bookLegend!(ax, position=:lb, labelsize=12, rowgap=0, patchsize=(14, 8), padding=(2, 2, 2, 2))
    fig
end

@figure "deelstra_vol100" begin
    df = readtable("deelstra_krekel_vol100.txt")
    fig = bookFigure(size=(400, 400))
    ax = bookAxis(fig[1, 1], xlabel="Volatility in %", ylabel="Price")
    for (col, label) in (("Ju", "Ju"), ("MM3", "MM3"), ("Deelstra", "Deelstra FA2 s=3"), ("VL2", "VL2"), ("VL3", "VL3"))
        lines!(ax, df.sigma .* 100, df[!, col] .+ df.price, label=label)
    end
    lines!(ax, df.sigma .* 100, df.price, label="QMC", color=:black)
    bookLegend!(ax, position=:rb)
    fig
end

@figure "deelstra_sabr_vol100_2y" begin
    df = readtable("deelstra_krekel_sabr_vol100_2y.txt")
    fig = bookFigure()
    #the column Rho of the legacy data is the volatility of the assets 2 to 4 (the first asset has a volatility of 100%)
    ax = bookAxis(fig[1, 1], xlabel="Volatility of assets 2 to 4 in %", ylabel="Price")
    for (col, label) in (("Ju", "Ju"), ("Deelstra", "Deelstra"), ("SABRs", "SABR single"), ("SABRi", "SABR independent"))
        lines!(ax, df.Rho .* 100, df[!, col] .* 100, label=label)
    end
    bookLegend!(ax, position=:lt)
    fig
end

## normal SABR approximation of a flat Black volatility (test/sabr_test.jl, testset "flat-basket")
@compute "black_normal_sabr_collo" begin
    vols = fill(0.2, 10)
    tte, r, spot = 1.0, 0.05, 1.0
    f = spot * exp(r * tte)
    strikes = [0.85, 0.90, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4] .* f
    df = exp(-r * tte)
    prices, weights = Collocation.weightedPrices(true, strikes, vols, ones(length(vols)), f, df, tte, vegaFloor=1e-7)
    bvols = Bachelier.bachelierImpliedVolatility.(prices, true, strikes, tte, f, df)
    spline = PPInterpolation.CubicSplineNatural(log.(strikes ./ f), bvols)
    guess = AQFED.VolatilityModels.initialGuessNormalATM(f, tte, 0.0, spline(0.0), PPInterpolation.evaluateDerivative(spline, 0.0), PPInterpolation.evaluateSecondDerivative(spline, 0.0))
    section = AQFED.VolatilityModels.calibrateNormalSABRSectionFromGuess(tte, f, strikes, bvols, ones(length(bvols)), guess)
    isoc, _ = Collocation.makeIsotonicCollocation(strikes, prices, weights, tte, f, df, deg=3, degGuess=1)
    sol = Collocation.Polynomial(isoc)
    x = range(0.5, stop=2.0, length=201)
    sabrVol = @. AQFED.Black.impliedVolatility(true, Bachelier.bachelierFormula(true, x, f, sqrt(AQFED.TermStructure.normalVarianceByMoneyness(section, x - f)), tte, df), f, x, tte, df)
    colloVol = AQFED.Black.impliedVolatility.(true, Collocation.priceEuropean.(sol, true, x, f, df), f, x, tte, df)
    writetable("black_normal_sabr_collo.csv", DataFrame(Strike=collect(x), NormalSABR=(sabrVol .- 0.2) .* 100, CubicCollocation=(colloVol .- 0.2) .* 100))
end

@figure "black_normal_sabr_collo" begin
    df = readcsv("black_normal_sabr_collo.csv")
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Error in Black volatility %")
    lines!(ax, df.Strike, df.NormalSABR, label="Normal SABR")
    lines!(ax, df.Strike, df.CubicCollocation, label="Cubic collocation")
    ylims!(ax, -0.02, 0.04)
    bookLegend!(ax)
    fig
end

## normal SABR fitted to SPX500 and TSLA 1m options, and normal SABR basket against the 2D quadrature on Black-Scholes assets.
## The calibration is the body of test/sabr_test.jl testset "basket-smile" (run as is, without its plotting comments).
@compute "normal_sabr_basket_1m" begin
    src = read(joinpath(@__DIR__, "..", "test", "sabr_test.jl"), String)
    i = findfirst("@testset \"basket-smile\" begin", src)
    body = src[last(i)+1:end]
    body = body[1:findfirst("\n    #=\n    ## fit of sabr", body)[1]]
    mod = Module(:SabrBasketSmile)
    Core.eval(mod, :(using AQFED, AQFED.Collocation, AQFED.Bachelier, AQFED.Basket, PPInterpolation, LinearAlgebra, Test))
    Base.include_string(mod, body)
    g(s) = Base.invokelatest(getfield, mod, s) #globals created by include_string are only visible in the latest world
    out = DataFrame()
    for (name, f, strikes, vols, section) in (("SPX500", g(:f), g(:strikes), g(:vols), g(:sectionSPX)), ("TSLA", g(:f2), g(:strikes2), g(:vols2), g(:sectionTSLA)))
        x = collect(range(strikes[1], stop=strikes[end], length=201))
        p = [Bachelier.bachelierFormula(true, k, f, sqrt(AQFED.TermStructure.normalVarianceByMoneyness(section, k - f)), g(:tte), 1.0) for k in x]
        append!(out, DataFrame(Underlying=name, Series="Normal SABR", X=x ./ f, Y=abs.(AQFED.Black.impliedVolatility.(true, p, f, x, g(:tte), 1.0)) .* 100))
        append!(out, DataFrame(Underlying=name, Series="Market", X=strikes ./ f, Y=vols .* 100))
    end
    b = collect(g(:bstrikes))
    pS, pB, pBA, pSB = Float64.(g(:priceSABR)), Float64.(g(:priceBlack)), Float64.(g(:priceBlackATM)), Float64.(g(:priceSABRBlack))
    append!(out, DataFrame(Underlying="Basket", Series="SABR / Black", X=b, Y=abs.(pS ./ pB .- 1) .* 100))
    append!(out, DataFrame(Underlying="Basket", Series="SABR / Black ATM", X=b, Y=abs.(pS ./ pBA .- 1) .* 100))
    append!(out, DataFrame(Underlying="Basket", Series="SABR Flat ATM / Black ATM", X=b, Y=abs.(pSB ./ pBA .- 1) .* 100))
    writetable("normal_sabr_basket_1m.csv", out)
end
function normalSabrSmileFigure(name)
    df = readcsv("normal_sabr_basket_1m.csv")
    df = df[df.Underlying.==name, :]
    fig = bookFigure(size=(400, 300))
    ax = bookAxis(fig[1, 1], xlabel="Strike/Forward", ylabel="Volatility in %")
    m, s = df[df.Series.=="Market", :], df[df.Series.=="Normal SABR", :]
    lines!(ax, s.X, s.Y, label="Normal SABR", color=PALETTE[1])
    referenceScatter!(ax, m.X, m.Y, label="Market")
    bookLegend!(ax, position=:rt)
    fig
end
@figure "normal_sabr_spx500_1m" normalSabrSmileFigure("SPX500")
@figure "normal_sabr_tsla_1m" normalSabrSmileFigure("TSLA")
@figure "normal_sabr_basket_1m" begin
    df = readcsv("normal_sabr_basket_1m.csv")
    df = df[df.Underlying.=="Basket", :]
    #third panel of a row with the two smile fits: legend below the axis
    fig = bookFigure(size=(400, 330))
    ax = bookAxis(fig[1, 1], xlabel="Basket strike", ylabel="Difference", yscale=log10,
        yticks=([0.1, 10], ["0.1%", "10%"]))
    groupLines!(ax, df.X, df.Y .+ 1e-12, df.Series, order=["SABR / Black", "SABR / Black ATM", "SABR Flat ATM / Black ATM"])
    Legend(fig[2, 1], ax, framevisible=false, rowgap=0, patchsize=(16, 8), tellwidth=false)
    rowgap!(fig.layout, 2)
    fig
end

## A Greek mistake: Δ towards each asset of an equally weighted basket call (σ₁ = 10%, σ₂ = 90%, S₁ = S₂ = K = 1, T = 1,
## zero rates) against the naive scaled index Δ, q Δ_A, with forward finite differences of relative shift ε = 1e-4, and
## the hedge value Δ₁ × 2% − Δ₂ × 2% after the move (S₁ + 0.02, S₂ − 0.02). Prices by 2D Gauss-Legendre quadrature.
@compute "basket_delta_correlation" begin
    q, strike, tte, ε = [0.5, 0.5], 1.0, 1.0, 1e-4
    tvar = [0.1, 0.9] .^ 2 .* tte
    pricer = Basket.QuadBasketPricer(AQFED.Math.GaussLegendre(128))
    price(spots, corr) = Basket.priceEuropean(pricer, true, strike, 1.0, spots, spots, tvar, q, corr, isSplit=true)
    out = DataFrame(rho=Float64[], Delta1=Float64[], Delta2=Float64[], DeltaA=Float64[], HedgeValue=Float64[])
    #the 2D quadrature is inaccurate for a singular correlation matrix (|ρ| = 1): stop at |ρ| = 0.999
    for ρ in range(-0.999, 0.999, length=81)
        corr = [1.0 ρ; ρ 1.0]
        S = [1.0, 1.0]
        v = price(S, corr)
        Δ1 = (price(S .* [1 + ε, 1], corr) - v) / (S[1] * ε)
        Δ2 = (price(S .* [1, 1 + ε], corr) - v) / (S[2] * ε)
        ΔA = (price(S .* (1 + ε), corr) - v) / (sum(q .* S) * ε)
        push!(out, (ρ, Δ1, Δ2, ΔA, Δ1 * 0.02 - Δ2 * 0.02))
    end
    writetable("basket_delta_correlation.csv", out)
end
@figure "basket_delta_correlation_small" begin
    df = readcsv("basket_delta_correlation.csv")
    fig = bookFigure(size=(400, 300))
    ax = bookAxis(fig[1, 1], xlabel="Correlation", ylabel="Δ value")
    lines!(ax, df.rho, df.Delta1, color=PALETTE[1], label=rich("Δ", subscript("1")))
    lines!(ax, df.rho, 0.5 .* df.DeltaA, color=PALETTE[2], label=rich("qΔ", subscript("A")))
    lines!(ax, df.rho, df.Delta2, color=PALETTE[3], label=rich("Δ", subscript("2")))
    Legend(fig[1, 2], ax, framevisible=false)
    fig
end
@figure "basket_delta_hedge_correlation_small" begin
    df = readcsv("basket_delta_correlation.csv")
    fig = bookFigure(size=(400, 300))
    ax = bookAxis(fig[1, 1], xlabel="Correlation", ylabel="Δ hedge value")
    lines!(ax, df.rho, df.HedgeValue, color=PALETTE[1])
    fig
end

## error of the Deelstra variants (first order approximation FA1, FA2, FA3 and s = 1, 3) on the four assets example of
## Krekel et al. (2006): T = 5, K = 100, S = 100, ρ = 0.5, w = 0.25, σ₁ = 100%, σ₂ = σ₃ = σ₄ ∈ [0, 100%]. (a) zero rates,
## (b) the forward of the first asset is different from its spot (drift of 5%). Reference: quasi Monte-Carlo (Sobol,
## Brownian bridge, 2^24 paths) on the put, with the put-call parity.
@compute "deelstra_vol100_err" begin
    tte, strike, w = 5.0, 100.0, fill(0.25, 4)
    corr = fill(0.5, 4, 4); corr[diagind(corr)] .= 1.0
    spots = fill(100.0, 4)
    variants = [(a, s) for a in 1:3 for s in (1, 3)]
    out = DataFrame(case=String[], sigma=Float64[], reference=Float64[], method=String[], price=Float64[])
    for (case, fwd) in (("equal", copy(spots)), ("drift", [100.0 * exp(0.05 * tte), 100.0, 100.0, 100.0]))
        for σ in 0.0:0.05:1.0
            tvar = [1.0, σ, σ, σ] .^ 2 .* tte
            put = Basket.priceEuropean(Basket.MonteCarloEngine(true, 2^24), false, strike, 1.0, spots, fwd, tvar, w, corr)
            ref = put + sum(w .* fwd) - strike
            for (a, s) in variants
                push!(out, (case, σ, ref, "FA$a s=$s", Basket.priceEuropean(DeelstraBasketPricer(a, s), true, strike, 1.0, spots, fwd, tvar, w, corr)))
            end
        end
    end
    writetable("deelstra_vol100_err.csv", out)
end
@figure "deelstra_vol100_err" begin
    df = readcsv("deelstra_vol100_err.csv")
    names = ["FA1 s=1", "FA1 s=3", "FA2 s=1", "FA2 s=3", "FA3 s=1", "FA3 s=3"]
    colors = Dict(n => PALETTE[i] for (i, n) in enumerate(names))
    styles = Dict("FA1" => :dot, "FA2" => :solid, "FA3" => :dash)
    fig = bookFigure(size=(800, 240))
    axs = Axis[]
    for (j, (case, title)) in enumerate((("equal", "Equal forwards"), ("drift", "Forward of the first asset 28% above spot")))
        d = df[df.case.==case, :]
        ax = bookAxis(fig[1, j], xlabel="Volatility of assets 2 to 4 in %", ylabel=j == 1 ? "Price error" : "", title=title)
        hlines!(ax, [0.0], color=:black, linewidth=0.8)
        for n in names
            dn = sort(d[d.method.==n, :], :sigma)
            lines!(ax, dn.sigma .* 100, dn.price .- dn.reference, color=colors[n], linestyle=styles[n[1:3]], linewidth=n[end] == '3' ? 2.0 : 1.2)
        end
        push!(axs, ax)
    end
    linkyaxes!(axs...)
    Legend(fig[2, :], [LineElement(color=colors[n], linestyle=styles[n[1:3]]) for n in names], names, orientation=:horizontal, framevisible=false, tellwidth=false)
    fig
end

## Error against the strike at negative correlation: two assets σ = 20%, S = 1, r = 5%, T = 1, w = 0.5 (as in
## fig:deelstra_correlation), reference by 2D Gauss-Legendre quadrature.
@compute "deelstra_strike_negative" begin
    tte, r = 1.0, 0.05
    df, spots, w, tvar = exp(-r * tte), ones(2), [0.5, 0.5], fill(0.2^2 * tte, 2)
    forwards = spots .* exp(r * tte)
    out = DataFrame(rho=Float64[], strike=Float64[], reference=Float64[], Ju=Float64[], Deelstra=Float64[], CurranLB=Float64[], VL3=Float64[],
        Deelstra3=Float64[])
    for ρ in (-0.5, -0.9), strike in 0.7:0.01:1.4
        corr = [1.0 ρ; ρ 1.0]
        ref = Basket.priceEuropean(Basket.QuadBasketPricer(AQFED.Math.GaussLegendre(128)), true, strike, df, spots, forwards, tvar, w, corr, isSplit=true)
        push!(out, (ρ, strike, ref,
            Basket.priceEuropean(JuBasketPricer(), true, strike, df, spots, forwards, tvar, w, corr),
            Basket.priceEuropean(DeelstraBasketPricer(2, 3, AQFED.Math.GaussLegendre(64)), true, strike, df, spots, forwards, tvar, w, corr),
            Basket.priceEuropean(DeelstraLBBasketPricer(2, 3), true, strike, df, spots, forwards, tvar, w, corr),
            Basket.priceEuropean(AQFED.Basket.VorstLevyExpansion(3), true, strike, df, spots, forwards, tvar, w, corr),
            Basket.priceEuropean(DeelstraBasketPricer(2, 3, q=AQFED.Math.GaussLegendre(64), nMoments=3), true, strike, df, spots, forwards, tvar, w, corr)))
    end
    writetable("deelstra_strike_negative.csv", out)
end
@figure "deelstra_strike_negative" begin
    df = readcsv("deelstra_strike_negative.csv")
    names = [("Ju", "Ju"), ("Deelstra", "Deelstra FA2 s=3"), ("CurranLB", "Curran lower bound"), ("VL3", "VL3"),
        ("Deelstra3", "Deelstra 3 moments")]
    fig = bookFigure(size=(800, 380))
    axs = Axis[]
    for (j, ρ) in enumerate((-0.5, -0.9))
        d = sort(df[df.rho.==ρ, :], :strike)
        ax = bookAxis(fig[1, j], xlabel="Strike", ylabel=j == 1 ? "Relative error" : "", title="ρ = $ρ", yscale=log10,
            yticks=([1e-8, 1e-6, 1e-4, 1e-2, 1.0], ["10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²", "1"]))
        for (i, (col, _)) in enumerate(names)
            e = abs.(d[!, col] ./ d.reference .- 1)
            keep = isfinite.(e) .& (e .> 0)
            lines!(ax, d.strike[keep], e[keep], color=PALETTE[i], linestyle=i > 4 ? :dash : :solid)
        end
        vlines!(ax, [1.05], color=:gray, linewidth=0.8, linestyle=:dash) #basket forward
        ylims!(ax, 1e-9, 10)
        push!(axs, ax)
    end
    linkyaxes!(axs...)
    Legend(fig[2, :], [LineElement(color=PALETTE[i], linestyle=i > 4 ? :dash : :solid) for i in eachindex(names)], last.(names), orientation=:horizontal, nbanks=2, framevisible=false, tellwidth=false)
    fig
end

## Deelstra with two (FA2, s = 3) or three conditional moments on the four assets of Krekel et al. (Table 4: T = 5,
## S = K = 100, w = 0.25, ρ = -0.25) and with ρ = 0.5. Reference: quasi Monte-Carlo (2^22 paths) on the put, put-call parity.
@compute "deelstra_3m_krekel" begin
    out = DataFrame(rho=Float64[], sigma=Float64[], reference=Float64[], Deelstra=Float64[], Deelstra3=Float64[])
    forwards, w, tte, strike = fill(100.0, 4), fill(0.25, 4), 5.0, 100.0
    for ρ in (-0.25, 0.5), σ in (0.25, 0.5, 0.75)
        corr = fill(ρ, 4, 4); corr[diagind(corr)] .= 1.0
        tvar = fill(σ^2 * tte, 4)
        put = Basket.priceEuropean(Basket.MonteCarloEngine(true, 2^22), false, strike, 1.0, forwards, forwards, tvar, w, corr)
        ref = put + sum(w .* forwards) - strike
        push!(out, (ρ, σ, ref,
            Basket.priceEuropean(DeelstraBasketPricer(2, 3), true, strike, 1.0, forwards, forwards, tvar, w, corr),
            Basket.priceEuropean(DeelstraBasketPricer(2, 3, nMoments=3), true, strike, 1.0, forwards, forwards, tvar, w, corr)))
    end
    writetable("deelstra_3m_krekel.csv", out)
end
