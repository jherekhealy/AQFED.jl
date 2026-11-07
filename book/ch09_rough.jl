# Chapter 9: rough Heston (setups of test/rough_test.jl).

using AQFED, AQFED.Rough, AQFED.Black, CharFuncPricing, PPInterpolation, GaussNewton, Random123
import SpecialFunctions: gamma

roughVol(isCall, price, forward, strike, τ) = try
    Black.impliedVolatility(isCall, max(price, 1e-32), forward, strike, τ, 1.0)
catch e
    e isa DomainError ? NaN : rethrow()
end

## Padé approximation against Adams scheme with 1024 steps, at T = 1 for several Hurst exponents
@compute "rheston_h_pade" begin
    η, forward, τ = 0.8, 1.0, 1.0
    out = DataFrame(H=Float64[], Logmoneyness=Float64[], Method=String[], Vol=Float64[])
    #as in the test, ν is computed once for H = 0.05 and kept for all H
    nu = η * sqrt(2 * 0.05) * gamma(0.05 + 0.5)
    for (H, upper) in zip([0.01, 0.05, 0.25, 0.45], [120.0, 150.0, 360.0, 600.0])
        params = RoughHestonParams(H, -0.65, nu, s -> 0.025)
        kk = collect(-0.75:0.01:0.75)
        strikes = exp.(kk) ./ forward
        for (name, pricer) in ("Pade" => CharFuncPricing.AdaptiveFilonCharFuncPricer(PadeCharFunc(params), τ, qTol=1e-12),
            "Adams-1024" => CharFuncPricing.AdaptiveFilonCharFuncPricer(AdamsCharFunc(params, 1024), τ, qTol=1e-12, myTrans=CharFuncPricing.IdentityTransformation(0.0, upper)))
            vols = [roughVol(k >= forward, CharFuncPricing.priceEuropean(pricer, k >= forward, k, forward, τ, 1.0), forward, k, τ) for k in strikes]
            append!(out, DataFrame(H=H, Logmoneyness=kk, Method=name, Vol=vols .* 100))
        end
    end
    writetable("rheston_h_pade.csv", out)
end
@figure "rheston_h_pade" begin
    df = readcsv("rheston_h_pade.csv")
    fig = bookFigure(size=(1024, 256))
    axs = Axis[]
    for (i, H) in enumerate([0.01, 0.05, 0.25, 0.45])
        d = df[df.H.==H, :]
        ax = bookAxis(fig[1, i], xlabel="Log-moneyness", ylabel=i == 1 ? "Implied volatility in %" : "", title="H = $H")
        groupLines!(ax, d.Logmoneyness, d.Vol, d.Method, order=["Pade", "Adams-1024"], labels=Dict("Pade" => "Padé"))
        ylims!(ax, 7.5, 35)
        i == 4 && bookLegend!(ax, position=:rt)
        push!(axs, ax)
    end
    fig
end

## SPX500 as of October 2010: variance swaps, forward variance, and calibrated rough Heston / Heston smiles
const SPX2010_STRIKES = [1008, 1209.6, 1411.2, 1612.8, 1814.4, 1915.2, 2016, 2116.8, 2217.6, 2419.2, 2620.8, 3024]
const SPX2010_TTES = [0.057534247, 0.153424658, 0.230136986, 0.479452055, 0.728767123, 1.22739726, 1.726027397, 2.243835616, 2.742465753, 3.24109589, 4.238356164]
#implied volatilities (expiries in rows), from heston_cos3d / Andersen-Brotherton-Ratcliffe
spx2010Vols() = Matrix{Float64}(readtable("spx2010_vols.txt"))

@compute "rheston_spx2010" begin
    strikes, ttes, volM = Float64.(SPX2010_STRIKES), SPX2010_TTES, spx2010Vols()
    r, q, spot = 0.02, 0.01, 2016.0
    forwards = [spot * exp((r - q) * t) for t in ttes]
    varswap = [AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.FukasawaVarianceSwapReplication(true), t, log.(strikes ./ forwards[i]),
        volM[i, :] .^ 2, 1.0, u=1.0) for (i, t) in enumerate(ttes)]
    spline = CubicSplineNatural(vcat(0.0, ttes), vcat(0.0, varswap .* ttes))
    xi(u) = u <= 0 ? varswap[1] : PPInterpolation.evaluateDerivative(spline, min(u, ttes[end]))
    linear = PPInterpolation.makeLinearPP(vcat(0.0, ttes), vcat(0.0, varswap .* ttes))
    xiLinear(u) = u <= 0 ? varswap[1] : PPInterpolation.evaluateDerivative(linear, min(u, ttes[end] - 1e-8))
    #Heston fitted to the variance swaps only
    objV(fvec, c) = (@. fvec = [c[2] + (1 - exp(-c[3] * t)) / (c[3] * t) * (c[1] - c[2]) for t in ttes] - varswap)
    _, fit = GaussNewton.optimize!(objV, [varswap[1], varswap[end], 1.0], zeros(length(ttes)), autodiff=:single, iscale=1, abstol=length(ttes) * eps())
    v0, θ, κ = fit.minimizer
    refPriceM, weightM = zeros(size(volM)), zeros(size(volM))
    for (i, t) in enumerate(ttes)
        refPriceM[i, :], weightM[i, :] = AQFED.Collocation.weightedPrices(true, strikes, volM[i, :], ones(length(strikes)), forwards[i], 1.0, t, vegaFloor=1e-4)
    end
    tv = [AQFED.Math.AlgebraicTransformation(0.01, 0.5), AQFED.Math.AlgebraicTransformation(-0.99, 0.99), AQFED.Math.MQMinTransformation(0.01, 1.0)]
    n = length(strikes)
    function fill!(fvec, pricer)
        for (i, t) in enumerate(ttes), (j, x) in enumerate(strikes)
            fvec[j+(i-1)*n] = weightM[i, j] * (CharFuncPricing.priceEuropean(pricer(i), true, x, forwards[i], t, 1.0) - refPriceM[i, j])
        end
    end
    roughCF(c) = PadeCharFunc{RoughHestonParams{Float64},Complex,3}(RoughHestonParams(tv[1](c[1]), tv[2](c[2]), tv[3](c[3]), xi), AQFED.Math.Chebyshev{Float64,1}(64))
    objR(fvec, c) = (cf = roughCF(c); fill!(fvec, i -> CharFuncPricing.AdaptiveFilonCharFuncPricer(cf, ttes[i], qTol=1e-12)))
    _, fitR = GaussNewton.optimize!(objR, [AQFED.Math.inv(tv[k], x) for (k, x) in enumerate([0.1, -0.65, 0.4])], zeros(length(refPriceM)), autodiff=:single, iscale=1, abstol=length(refPriceM) * eps())
    cfR = roughCF(fitR.minimizer)
    objH(fvec, c) = (cf = DefaultCharFunc(HestonParams(v0, κ, θ, tv[2](c[1]), tv[3](c[2]))); p = CharFuncPricing.ALCharFuncPricer(cf); fill!(fvec, i -> p))
    _, fitH = GaussNewton.optimize!(objH, [AQFED.Math.inv(tv[2], -0.65), AQFED.Math.inv(tv[3], 0.4)], zeros(length(refPriceM)), autodiff=:single, iscale=1, abstol=length(refPriceM) * eps())
    pH = HestonParams(v0, κ, θ, tv[2](fitH.minimizer[1]), tv[3](fitH.minimizer[2]))
    fullP(c) = HestonParams(tv[3](c[1]), tv[3](c[2]), tv[3](c[3]), tv[2](c[4]), tv[3](c[5]))
    objF(fvec, c) = (p = CharFuncPricing.ALCharFuncPricer(DefaultCharFunc(fullP(c))); fill!(fvec, i -> p))
    x0 = [v0, κ, θ, -0.65, 0.4]
    _, fitF = GaussNewton.optimize!(objF, [AQFED.Math.inv(tv[k == 4 ? 2 : 3], x) for (k, x) in enumerate(x0)], zeros(length(refPriceM)), autodiff=:single, iscale=1, abstol=length(refPriceM) * eps())
    pF = fullP(fitF.minimizer)
    #curves
    t = collect(range(0.01, stop=ttes[end], length=301))
    quad = AQFED.Math.Simpson(length(t) * 16)
    curves = DataFrame(Time=t, XiLinear=xiLinear.(t), XiSpline=xi.(t), XiHeston=@.(θ + exp(-κ * t) * (v0 - θ)), XiHestonFull=@.(pF.θ + exp(-pF.κ * t) * (pF.v0 - pF.θ)),
        VarSpline=[AQFED.Math.integrate(quad, xi, 0.0, u) / u for u in t], VarHeston=@.(θ + (1 - exp(-κ * t)) / (κ * t) * (v0 - θ)),
        VarHestonFull=@.(pF.θ + (1 - exp(-pF.κ * t)) / (pF.κ * t) * (pF.v0 - pF.θ)))
    writetable("rheston_spx2010_curves.csv", curves)
    writetable("rheston_spx2010_varswap.csv", DataFrame(Time=ttes, Variance=varswap))
    #smiles
    kFine = collect(range(strikes[1], stop=strikes[end], length=201))
    smiles = DataFrame(Expiry=Float64[], Moneyness=Float64[], Model=String[], Vol=Float64[])
    #the calibrated rough Heston smiles are drawn with the Adams scheme and a Black control variate: the Padé approximation is
    #accurate enough for the calibration, but its error (~1e-9) exceeds the deep out-of-the-money prices of short maturities
    paramsR = CharFuncPricing.model(cfR)
    adamsCV = AQFED.Rough.RoughHestonCVCharFunc(AdamsCharFunc(paramsR, 512))
    pricers = ["Rough Heston" => i -> CharFuncPricing.AdaptiveFilonCharFuncPricer(adamsCV, ttes[i], qTol=1e-12,
            myTrans=CharFuncPricing.IdentityTransformation(0.0, 100.0 / sqrt(ttes[i]))),
        "Heston varswap" => i -> CharFuncPricing.ALCharFuncPricer(DefaultCharFunc(pH)), "Heston full" => i -> CharFuncPricing.ALCharFuncPricer(DefaultCharFunc(pF))]
    for (i, t) in enumerate(ttes)
        for (name, mk) in pricers
            p = mk(i)
            #out-of-the-money prices below 1e-10 F are below the absolute accuracy of the Fourier integration: not shown
            vols = map(kFine) do k
                price = CharFuncPricing.priceEuropean(p, k >= forwards[i], k, forwards[i], t, 1.0)
                price < 1e-10 * forwards[i] ? NaN : roughVol(k >= forwards[i], price, forwards[i], k, t)
            end
            append!(smiles, DataFrame(Expiry=t, Moneyness=kFine ./ forwards[i], Model=name, Vol=vols .* 100))
        end
        append!(smiles, DataFrame(Expiry=t, Moneyness=strikes ./ forwards[i], Model="Market", Vol=volM[i, :] .* 100))
    end
    writetable("rheston_spx2010_smiles.csv", smiles)
    writetable("rheston_spx2010_params.csv", DataFrame(Model=["Rough Heston H,rho,nu", "Heston varswap", "Heston full"],
        Params=[string(CharFuncPricing.model(cfR).H, " ", CharFuncPricing.model(cfR).ρ, " ", CharFuncPricing.model(cfR).ν), string(pH), string(pF)]))
end
@figure "rheston_forwardvar_spx2010" begin
    c = readcsv("rheston_spx2010_curves.csv")
    fig = bookFigure(size=(400, 300))
    ax = bookAxis(fig[1, 1], xlabel="Time", ylabel="Forward variance")
    lines!(ax, c.Time, c.XiLinear, label="Linear")
    lines!(ax, c.Time, c.XiSpline, label="Spline")
    lines!(ax, c.Time, c.XiHeston, label="Heston varswap")
    lines!(ax, c.Time, c.XiHestonFull, label="Heston full")
    bookLegend!(ax, position=:rb)
    fig
end
@figure "rheston_varswap_spx2010" begin
    c, v = readcsv("rheston_spx2010_curves.csv"), readcsv("rheston_spx2010_varswap.csv")
    fig = bookFigure(size=(400, 300))
    ax = bookAxis(fig[1, 1], xlabel="Time to maturity", ylabel="Variance")
    lines!(ax, c.Time, c.VarSpline, label="Spline", color=PALETTE[2])
    lines!(ax, c.Time, c.VarHeston, label="Heston varswap", color=PALETTE[3])
    lines!(ax, c.Time, c.VarHestonFull, label="Heston full", color=PALETTE[4])
    referenceScatter!(ax, v.Time, v.Variance, label="Synthetic swap")
    bookLegend!(ax, position=:rb)
    fig
end
#3 columns of 4 rows on a tall page (the 4 x 3 grid of the original made the panels too small), the legend in the last slot
@figure "rheston_vol_spx2010" begin
    df = readcsv("rheston_spx2010_smiles.csv")
    ttes = sort(unique(df.Expiry))
    ncol = 3
    nrow = cld(length(ttes) + 1, ncol)
    fig = bookFigure(size=(737, 960))
    for (i, t) in enumerate(ttes)
        r, c = fldmod1(i, ncol)
        d = df[df.Expiry.==t, :]
        ax = bookAxis(fig[r, c], title=rich(rich("T", font=:italic), " = $(round(t, digits=3))"), titlealign=:left,
            xlabel=i + ncol > length(ttes) ? "Moneyness" : "", ylabel=c == 1 ? "Implied volatility in %" : "", xticks=0.6:0.2:1.4)
        m = d[d.Model.=="Market", :]
        referenceScatter!(ax, m.Moneyness, m.Vol, label="Market")
        for (j, model) in enumerate(["Rough Heston", "Heston varswap", "Heston full"])
            dd = d[d.Model.==model, :]
            lines!(ax, dd.Moneyness, dd.Vol, color=PALETTE[j], label=model)
        end
        i == length(ttes) && Legend(fig[nrow, ncol], ax, framevisible=false, tellwidth=false, tellheight=false)
    end
    rowgap!(fig.layout, 10)
    fig
end

## single slice: SPX500 options of maturity one week as of March 16, 2017 (test/rough_test.jl, "Rough-Calibration").
## rHeston with a constant ξ0 from the variance swap replication, calibrated with Padé (3,3) on |log(K/F)| < 0.4√T; its smile
## drawn with Padé and with the Adams scheme (2000 steps, no control variate), both breaking down in the wings; reduced Heston
## (v0 = θ, κ = 1) and SABR (β = 1) calibrated on the same quotes
@compute "rheston_spx1w" begin
    q = readcsv("rheston_spx1w_quotes.csv")
    tte, forward = 8 / 365, 2385.099980
    logm, variance, w = q.Logmoneyness, q.Variance, q.Weight
    volA = sqrt.(variance)
    strikeA = exp.(logm) .* forward
    refPrices, weights = AQFED.Collocation.weightedPrices(true, strikeA, volA, w, forward, 1.0, tte, vegaFloor=1e-4)
    subset = findall(x -> abs(x) < 0.4 * sqrt(tte), logm)
    volSpline = CubicSplineNatural(logm[subset], volA[subset])
    guess = AQFED.VolatilityModels.initialGuessBlackATM(forward, tte, 1.0, volSpline(0.0), PPInterpolation.evaluateDerivative(volSpline, 0.0), PPInterpolation.evaluateSecondDerivative(volSpline, 0.0))
    sabr = AQFED.VolatilityModels.calibrateSABRSectionFromGuess(tte, forward, logm[subset], volA[subset], w[subset], guess)
    xi = AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.FukasawaVarianceSwapReplication(true), tte, logm[subset], variance[subset], 1.0, u=1.0)
    transformV = [AQFED.Math.AlgebraicTransformation(0.01, 0.5), AQFED.Math.AlgebraicTransformation(-0.99, 0.99), AQFED.Math.MQMinTransformation(0.01, 1.0)]
    fitPrices(makePricer, x0, toParams) = begin
        obj = function (fvec, coeff)
            pricer = makePricer(toParams(coeff))
            @. fvec = weights[subset] * ($(map(x -> CharFuncPricing.priceEuropean(pricer, true, x, forward, tte, 1.0), strikeA[subset])) - refPrices[subset])
        end
        coeff0 = [AQFED.Math.inv(t, x) for (t, x) in zip(x0[2], x0[1])]
        _, fit = GaussNewton.optimize!(obj, coeff0, zeros(length(subset)), autodiff=:single, iscale=1, abstol=length(subset) * eps())
        toParams(fit.minimizer)
    end
    roughParams(c) = RoughHestonParams(transformV[1](c[1]), transformV[2](c[2]), transformV[3](c[3]), s -> xi)
    pR = fitPrices(p -> CharFuncPricing.AdaptiveFilonCharFuncPricer(PadeCharFunc(p), tte, qTol=1e-8), ([0.1, -0.65, 0.4], transformV), roughParams)
    hestonParams(c) = HestonParams(transformV[3](c[1]), 1.0, transformV[3](c[1]), transformV[2](c[2]), transformV[3](c[3]))
    pH = fitPrices(p -> CharFuncPricing.ALCharFuncPricer(DefaultCharFunc(p)), ([xi, -0.65, 0.4], transformV[[3, 2, 3]]), hestonParams)
    kFine = forward .* exp.(range(-sqrt(tte), sqrt(tte), length=201))
    pricers = ["rHeston Padé" => CharFuncPricing.AdaptiveFilonCharFuncPricer(PadeCharFunc(pR), tte, qTol=1e-8),
        "rHeston Adams" => CharFuncPricing.AdaptiveFilonCharFuncPricer(AdamsCharFunc(pR, 2000), tte, qTol=1e-8, myTrans=CharFuncPricing.IdentityTransformation(0.0, 12 * 150.0)),
        "Heston" => CharFuncPricing.ALCharFuncPricer(DefaultCharFunc(pH))]
    smiles = DataFrame(Model=String[], Logmoneyness=Float64[], Vol=Float64[])
    for (name, pricer) in pricers
        vols = [roughVol(k >= forward, CharFuncPricing.priceEuropean(pricer, k >= forward, k, forward, tte, 1.0), forward, k, tte) for k in kFine]
        append!(smiles, DataFrame(Model=name, Logmoneyness=log.(kFine ./ forward), Vol=100 .* vols))
    end
    append!(smiles, DataFrame(Model="SABR", Logmoneyness=log.(kFine ./ forward), Vol=100 .* sqrt.(AQFED.TermStructure.varianceByLogmoneyness.(sabr, log.(kFine ./ forward)))))
    append!(smiles, DataFrame(Model="Market", Logmoneyness=logm, Vol=100 .* volA))
    writetable("rheston_spx1w_smiles.csv", smiles)
    writetable("rheston_spx1w_params.csv", DataFrame(Model=["rHeston H,rho,nu,xi", "Heston v0=theta,rho,sigma"],
        Params=[string(pR.H, " ", pR.ρ, " ", pR.ν, " ", xi), string(pH.v0, " ", pH.ρ, " ", pH.σ)]))
end

@figure "rheston_vol_spx1w" begin
    df = readcsv("rheston_spx1w_smiles.csv")
    tte = 8 / 365
    fig = bookFigure(size=(576, 288))
    ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Implied volatility in %")
    m = df[df.Model.=="Market", :]
    m = m[abs.(m.Logmoneyness).<=sqrt(tte), :]
    #calibration range shaded
    vspan!(ax, -0.4 * sqrt(tte), 0.4 * sqrt(tte), color=(:black, 0.05))
    referenceScatter!(ax, m.Logmoneyness, m.Vol, label="Market")
    for (j, model) in enumerate(["rHeston Padé", "rHeston Adams", "Heston", "SABR"])
        d = df[df.Model.==model, :]
        lines!(ax, d.Logmoneyness, d.Vol, color=PALETTE[j], label=model, linewidth=model == "rHeston Adams" ? 1.2 : 2.0)
    end
    xlims!(ax, -sqrt(tte), sqrt(tte))
    ylims!(ax, 5, 30)
    Legend(fig[1, 2], ax, framevisible=false)
    fig
end

## Monte-Carlo time discretization: relative error against the number of time-steps
@compute "rheston_mc_convergence" begin
    τ, strike, spot, H, forward = 1.0, 1.0, 1.0, 0.05, 1.0
    model = AQFED.Rough.FlatRoughBergomi(sqrt(0.025), H, 2.3, -0.9)
    nu = 0.8 * sqrt(2H) * gamma(H + 0.5)
    params = RoughHestonParams(H, -0.65, nu, s -> 0.025)
    payoff = AQFED.MonteCarlo.VanillaOption(true, strike, AQFED.MonteCarlo.BulletCashFlow(τ, τ, false, 0.0), 0.0)
    cfc = AQFED.Rough.RoughHestonCVCharFunc(AdamsCharFunc(params, 1000))
    refH = CharFuncPricing.priceEuropean(CharFuncPricing.AdaptiveFilonCharFuncPricer(cfc, τ, qTol=1e-12, myTrans=CharFuncPricing.IdentityTransformation(0.0, 160.0)), true, strike, forward, τ, 1.0)
    hParams = AQFED.TermStructure.HestonModel(0.025, 1.0, 0.025, -0.9, 1.0, 0.0, 0.0)
    refHeston = CharFuncPricing.priceEuropean(ALCharFuncPricer(DefaultCharFunc(HestonParams(0.025, 1.0, 0.025, -0.9, 1.0))), true, strike, forward, τ, 1.0)
    refR = 0.05331004660994802 #rBergomi hybrid scheme, 1024 steps, 16M paths
    rng(s) = AQFED.Random.ZRNGSeq(Random123.Philox4x(UInt64, (s, 20100921), 10), 1)
    n = 1024 * 64 * 4
    steps = [4, 8, 16, 32, 64, 128, 256]
    out = DataFrame(Steps=Int[], Method=String[], Error=Float64[])
    for s in steps
        push!(out, (s, "rHeston HQE", abs(AQFED.Rough.simulateHQE(rng(20230129), params, 1.0, payoff, 1, n, s)[1] / refH - 1)))
        push!(out, (s, "rBergomi hybrid", abs(AQFED.Rough.simulateHybrid(rng(20130129), model, 1.0, payoff, 1, n, s)[1] / refR - 1)))
        push!(out, (s, "rBergomi standard", abs(AQFED.Rough.simulate(rng(20130129), model, 1.0, payoff, 1, n, 1 / s)[1] / refR - 1)))
        push!(out, (s, "Heston DVSS2X", abs(AQFED.MonteCarlo.simulateDVSS2X(rng(20130129), hParams, spot, payoff, 0, n, 1 / s)[1] / refHeston - 1)))
    end
    writetable("rheston_mc_convergence.csv", out)
end
@figure "rheston_mc_convergence" begin
    df = readcsv("rheston_mc_convergence.csv")
    steps = sort(unique(df.Steps))
    fig = bookFigure(size=(800, 400))
    ax = bookAxis(fig[1, 1], xlabel="Time-steps", ylabel="Relative error in price", xscale=log10, yscale=log10, xticks=(steps, string.(steps)))
    groupLines!(ax, df.Steps, df.Error .+ 1e-12, df.Method, order=["rHeston HQE", "rBergomi hybrid", "rBergomi standard", "Heston DVSS2X"])
    Legend(fig[1, 2], ax, framevisible=false)
    fig
end
