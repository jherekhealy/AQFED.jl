# Chapter 7: implied probability densities of the quadratic LVG and Andreasen-Huge models fitted to a flat Black volatility of
# 20% (setups of test/lvg_test.jl).

using AQFED, AQFED.PDDE, AQFED.Collocation, AQFED.Black, PPInterpolation

lvgDensityAt(l, k) = (PDDE.derivativePrice(l, true, k + 1e-4) - PDDE.derivativePrice(l, true, k)) * 1e4
lognormalDensity(k, forward, vol, tte) = AQFED.Math.normpdf((log(forward / k) - 0.5vol^2 * tte) / (vol * sqrt(tte))) / (k * vol * sqrt(tte))

@compute "lvgq_lognormal_density" begin
    strikes, vol, tte, forward = [0.85, 0.90, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4], 0.2, 0.25, 1.025
    prices, weights = Collocation.weightedPrices(true, strikes, fill(vol, 10), ones(10), forward, 1.0, tte, vegaFloor=1e-7)
    c3 = PDDE.calibrateEQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, L=0.25, U=4.0, model="Quadratic", location="Mid-XX")
    c1 = PDDE.calibrateEQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, L=0.25, U=4.0, model="Quadratic", isC3=false, location="Mid-XX")
    k = collect(range(0.8, stop=strikes[end], length=401))
    out = vcat(DataFrame(Strike=k, Model="Lognormal", Density=lognormalDensity.(k, forward, vol, tte)),
        DataFrame(Strike=k, Model="C1 quadratic B-spline", Density=lvgDensityAt.(Ref(c1), k)),
        DataFrame(Strike=k, Model="Quadratic B-spline with C3 condition", Density=lvgDensityAt.(Ref(c3), k)))
    #sets of 10 strikes (forward 101): sensitivity of the density to the location of the knots
    sets = Dict("A" => [88.765135169592, 92.85494511181187, 93.38043933991867, 99.36630953103418, 107.9930908363475, 120.28927001531655, 122.03374332554367, 123.89923835687514, 134.7053265024023, 135.43198727132489],
        "B" => [85.02320730206546, 101.92119620448395, 103.54923033678908, 114.44768735553629, 121.8544889307987, 123.69290176127272, 125.07479849838776, 125.57703562269575, 131.63046140652378, 133.85728148034934],
        "C" => [98.06881859879773, 100.92597117414611, 101.05812725277457, 106.87903884044188, 109.1197408201026, 110.92917871923213, 119.7631067321654, 119.82704376084504, 132.1932799832046, 138.26522048510896])
    f2 = 101.0
    sets = Dict(n => round.(s, digits=2) for (n, s) in sets)
    sets["D"] = sort(vcat(f2, ([0.85, 0.90, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4] .* 100)[1:end-1]))
    k2 = collect(range(80.0, stop=140.0, length=401))
    for (name, s) in sets
        p, w = Collocation.weightedPrices(true, s, fill(vol, 10), ones(10), f2, 1.0, tte, vegaFloor=1e-7)
        append!(out, DataFrame(Strike=k2, Model="Set$name-Lognormal", Density=lognormalDensity.(k2, f2, vol, tte)))
        for loc in ["Strikes", "Mid-X", "Mid-XX", "Mid-Strikes", "Uniform"]
            l = PDDE.calibrateQuadraticLVG(tte, f2, s, p, w, useVol=false, L=f2 / 3, U=f2 * 3, model=PDDE.Quadratic(), location=loc)
            append!(out, DataFrame(Strike=k2, Model="Set$name-$loc", Density=lvgDensityAt.(Ref(l), k2)))
        end
    end
    writetable("lvgq_lognormal_density.csv", out)
end
@figure "lvgq_lognormal_density" begin
    df = readcsv("lvgq_lognormal_density.csv")
    fig = bookFigure(size=(480, 320))
    ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Probability density")
    models = ["Lognormal", "C1 quadratic B-spline", "Quadratic B-spline with C3 condition"]
    for (i, m) in enumerate(models)
        d = df[df.Model.==m, :]
        lines!(ax, d.Strike, d.Density, label=m, color=i == 1 ? :black : PALETTE[i-1], linestyle=i == 1 ? :dot : :solid)
    end
    bookLegend!(ax, position=:rt)
    fig
end
for name in ("A", "B", "C", "D")
    FIGURES["lvgq_lognormal_density_set_$name"] = () -> begin
        df = readcsv("lvgq_lognormal_density.csv")
        fig = bookFigure(size=(480, 320))
        ax = bookAxis(fig[1, 1], xlabel="Strike", ylabel="Probability density")
        d = df[df.Model.=="Set$name-Lognormal", :]
        lines!(ax, d.Strike, d.Density, label="Lognormal", color=:black, linestyle=:dot)
        for (i, loc) in enumerate(["Strikes", "Mid-X", "Mid-XX", "Mid-Strikes", "Uniform"])
            d = df[df.Model.=="Set$name-$loc", :]
            lines!(ax, d.Strike, d.Density, label=loc, color=PALETTE[i])
        end
        limits!(ax, 80, 140, 0, 0.06)
        bookLegend!(ax, position=:rt)
        fig
    end
end

## Andreasen-Huge (discrete LVG) on 101 and 401 points from a linear Black LVG, with or without the C3 condition at the forward
@compute "ah_lvg_linearblack" begin
    strikes, vol, tte, forward = [0.85, 0.90, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4], 0.2, 0.25, 1.025
    prices, _ = Collocation.weightedPrices(true, strikes, fill(vol, 10), ones(10), forward, 1.0, tte, vegaFloor=1e-7)
    k = exp.(range(log(strikes[1]), stop=log(strikes[end]), length=301))
    out = DataFrame()
    for (name, c3) in (("notc3", false), ("c3", true))
        lvg = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, ones(10), useVol=true, model=PDDE.LinearBlack(), location="Strikes", size=0,
            L=0.25, U=4.0, optimizer="LM-ALG", isC3=c3, nRefine=c3 ? 3 : 0)
        ah1, ah4 = PDDE.DiscreteLVG(lvg, 101), PDDE.DiscreteLVG(lvg, 401)
        append!(out, DataFrame(Case=name, Logmoneyness=log.(k ./ forward), Model="LVG LinearBlack", Density=lvgDensityAt.(Ref(lvg), k)))
        #density from a natural cubic spline through the discrete prices, as in the original figure
        d2(ah) = (sp = CubicSplineNatural(ah.z, ah.pp.(ah.z)); PPInterpolation.evaluateSecondDerivative.(Ref(sp), k))
        append!(out, DataFrame(Case=name, Logmoneyness=log.(k ./ forward), Model="AH l=101", Density=d2(ah1)))
        append!(out, DataFrame(Case=name, Logmoneyness=log.(k ./ forward), Model="AH l=401", Density=d2(ah4)))
    end
    writetable("ah_lvg_linearblack.csv", out)
end
function ahLvgFigure(case)
    df = readcsv("ah_lvg_linearblack.csv")
    df = df[df.Case.==case, :]
    fig = bookFigure(size=(480, 320))
    ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Probability density")
    for (i, m) in enumerate(["AH l=101", "AH l=401", "LVG LinearBlack"])
        d = df[df.Model.==m, :]
        lines!(ax, d.Logmoneyness, d.Density, label=m, color=i == 3 ? :black : PALETTE[i], linestyle=i == 3 ? :dot : :solid)
    end
    bookLegend!(ax, position=:rt)
    fig
end
@figure "ah_lvg_notc3_linearblack" ahLvgFigure("notc3")
@figure "ah_lvg_c3_linearblack" ahLvgFigure("c3")
