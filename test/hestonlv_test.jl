using AQFED, Test
import AQFED.MonteCarlo as MC
import AQFED.TermStructure: SVISection, VarianceSurfaceBySection, HestonModel, LocalVolatilityModel, varianceByLogmoneyness, discreteLocalVarianceByLogmoneyness
import AQFED.Random: ScrambledSobolSeq, Owen, ScramblingRngAdapter, Blabla8
import AQFED.Black: blackScholesFormula
import AQFED.TermStructure: SchobelZhuModel
using CharFuncPricing

#Van der Stoep et al. example of goquant (mc/simulation/slv_test.go TestSLVVanderStoepSingle): SVI sections calibrated by goquant.
vdsSurface() = VarianceSurfaceBySection([
        SVISection(0.02211047690170513, 0.08728514140099171, -0.7470846191571271, 0.01399832600143515, 0.05952343749999972, 0.5, 1.0),
        SVISection(0.007098341000109986, 0.08061891341040876, -0.43536114522531977, 0.14912190613260817, 0.07722949218749897, 2.0, 1.0),
        SVISection(6.729216331508232e-05, 0.05413804595661382, -0.3492177313652438, 0.3205809004034747, 0.1382236328125005, 5.0, 1.0),
        SVISection(0.009964200677573882, 0.03157899101219913, -0.5202476879856001, 0.2325780180852694, 0.16806298828124872, 8.0, 1.0),
        SVISection(0.0044141318643527916, 0.03190262256058419, -0.42036142370466967, 0.41444576466757904, 0.21499192810058354, 10.0, 1.0)],
    [0.5, 2.0, 5.0, 8.0, 10.0])

@testset "DiscreteLocalVariance" begin
    surface = vdsSurface()
    #goquant dupire.DiscreteLocalVarianceByLogmoneyness
    for (y, t0, t1, ref) in ((-0.5, 0.0, 0.01, 0.25840505452464346), (-0.5, 4.99, 5.0, 0.09366541784846788), (-0.1, 0.49, 0.5, 0.072878800936287375),
        (-0.1, 9.0, 9.01, 0.026148875006222691), (0.0, 1.0, 1.01, 0.019746052388189637), (0.2, 0.0, 0.01, 0.03022929613859195),
        (0.2, 4.99, 5.0, 0.010247041844094083), (0.6, 0.49, 0.5, 0.052594569079047804), (0.6, 9.0, 9.01, 0.016448782514932446))
        @test isapprox(discreteLocalVarianceByLogmoneyness(surface, y, t0, t1), ref, rtol=1e-12)
    end
end

@testset "LocalComponentCalculators" begin
    #deterministic particles, sorted, against goquant localcomp
    n = 10000
    x = [0.3 * ((i - 0.5) / n - 0.5) * (1 + 2 * ((i - 0.5) / n - 0.5)^2) for i = 1:n]
    v = [0.04 + 0.1 * x[i]^2 - 0.05 * x[i] + 0.01 * sin((i - 1) * 0.7) for i = 1:n]
    refs = [(MC.BinCalculator(gridSize=64), (0.046111239009775096, 0.039296028879344906, 0.035126265558684401)),
        (MC.NonConformCalculator(gridSize=128), (0.045716912167483456, 0.03987620066329866, 0.035865978450933776)),
        (MC.NonConformCalculator(gridSize=32), (0.04606074394732107, 0.039513403301103672, 0.035492684404927796)),
        (MC.CompactSupportCalculator(gridSize=1, maxGridSize=32, bandwidthFactor=1.5, threshold=1e-4), (0.045917210957239739, 0.039513174400632158, 0.035513221109574382)),
        (MC.BinSplineCalculator(gridSize=16), (0.046054389048761711, 0.039504471461601583, 0.035441598958932022))]
    for (calc, values) in refs
        f = MC.conditionalVariance(calc, x, v, 0.2, sqrt(2.0))
        for (z, ref) in zip((-0.1, 0.011, 0.12), values)
            @test isapprox(f(z), ref, rtol=1e-14)
        end
        @test f(-1.0) == f(x[1]) && f(1.0) == f(x[end]) #flat extrapolation
    end
    f = MC.conditionalVariance(MC.LocalLinearCalculator(), x, v, 0.2, sqrt(2.0))
    @test all(z -> isapprox(f(z), 0.04 + 0.1z^2 - 0.05z, atol=5e-4), (-0.1, 0.0, 0.1))
    #particle sort, including ties and a clustered cloud (sortperm fallback)
    sorter = MC.ParticleSorter(Float64, n)
    for xs in (sin.(1:n), round.(sin.(1:n), digits=2), vcat(zeros(n - 10), 1:10))
        sx, sv = MC.sortParticles!(sorter, xs, collect(1.0:n))
        @test issorted(sx) && sx == xs[Int.(sv)]
    end
end

@testset "HestonLV Van der Stoep" begin
    surface = vdsSurface()
    heston = HestonModel(0.031328999999999996, 0.11192673972631374, 0.06276225235639056, -0.7559289460184544, 0.1267434853943588, 0.0, 0.0)
    model = MC.HestonLVModel(heston, LocalVolatilityModel(surface, 0.0, 0.0))
    tte, n = 2.0, 1024 * 16
    config = MC.GridConfig(model, [tte], 1.0 / 100, MC.QE(withBB=true, withMartingale=false))
    nd = MC.ndims(model, config)
    sobol() = ScrambledSobolSeq(nd, n, Owen(30, ScramblingRngAdapter(Blabla8())))
    for calculator in (MC.NonConformCalculator(gridSize=32), MC.CompactSupportCalculator(maxGridSize=32))
        leverage = MC.calibrateLeverage(ScrambledSobolSeq(nd, n, Owen(30, ScramblingRngAdapter(Blabla8(UInt64[1, 2, 3, 4])))), model, 1.0, config, n, calculator=calculator)
        for strike in (0.75, 1.0, 1.41) #the model reproduces the vanilla options of the surface
            isCall = strike >= 1.0
            payoff = MC.VanillaOption(isCall, strike, MC.BulletCashFlow(tte, tte, false, 0.0), 0.0)
            price, stderr = MC.simulateScheme(sobol(), model, 1.0, payoff, 0, n, config, leverage=leverage)
            ref = blackScholesFormula(isCall, strike, 1.0, varianceByLogmoneyness(surface, log(strike), tte) * tte, 1.0, 1.0)
            @test abs(price - ref) < 3 * stderr
        end
    end
    #the calibration does not depend on the number of threads, pricing by blocks is the same as in one go
    calc = MC.NonConformCalculator(gridSize=32)
    l1 = MC.calibrateLeverage(sobol(), model, 1.0, config, n, calculator=calc, nThreads=1)
    l4 = MC.calibrateLeverage(sobol(), model, 1.0, config, n, calculator=calc, nThreads=4)
    @test all(l1.conditionalVariances[k](x) == l4.conditionalVariances[k](x) for k in (1, 2, 100, 200) for x in (-0.3, 0.0, 0.2))
    payoff = MC.VanillaOption(true, 1.0, MC.BulletCashFlow(tte, tte, false, 0.0), 0.0)
    price, stderr = MC.simulateScheme(sobol(), model, 1.0, payoff, 0, n, config, leverage=l1)
    priceParallel, stderrParallel = MC.simulateSchemeParallel(sobol(), model, 1.0, payoff, 0, n, config, leverage=l1)
    @test isapprox(price, priceParallel, rtol=1e-12)
    #calibration included, with the default calculator
    priceDefault, _ = MC.simulateScheme(sobol(), model, 1.0, payoff, 0, n, config, rngCalibration=sobol())
    @test abs(priceDefault - price) < 3 * stderr
    otherConfig = MC.GridConfig(model, [tte], 1.0 / 50, MC.QE(withBB=true, withMartingale=false))
    @test_throws ArgumentError MC.simulateScheme(sobol(), model, 1.0, payoff, 0, n, otherConfig, leverage=l1)
end

@testset "SchobelZhu EAE" begin
    #goquant mc/simulation/slv_test.go TestSZLVVanderStoepSingle: smart guess of the Schöbel-Zhu parameters from the surface
    params = CharFuncPricing.SchobelZhuParams(0.17700060676549295, 0.47004835147845592, 0.12560749278392161, -0.55573864712677212, 0.13330983125180917)
    model = SchobelZhuModel(params, 0.0, 0.0)
    tte, n = 2.0, 1024 * 16
    config = MC.GridConfig(model, [tte], 1.0 / 100, MC.EAE(withBB=true))
    nd = MC.ndims(model, config)
    pricer = makeCosCharFuncPricer(CharFuncPricing.DefaultCharFunc(params), tte, 1024 * 4, 16)
    for strike in (0.75, 1.0, 1.41)
        isCall = strike >= 1.0
        payoff = MC.VanillaOption(isCall, strike, MC.BulletCashFlow(tte, tte, false, 0.0), 0.0)
        price, stderr = MC.last_stat(MC.simulateScheme(ScrambledSobolSeq(nd, n, Owen(30, ScramblingRngAdapter(Blabla8()))), model, 1.0, payoff, 0, n, config))
        @test abs(price - CharFuncPricing.priceEuropean(pricer, isCall, strike, 1.0, tte, 1.0)) < 3 * stderr
    end
    #without the Brownian bridge, and in parallel
    config = MC.GridConfig(model, [tte], 1.0 / 100, MC.EAE())
    payoff = MC.VanillaOption(true, 1.0, MC.BulletCashFlow(tte, tte, false, 0.0), 0.0)
    price, stderr = MC.last_stat(MC.simulateScheme(ScrambledSobolSeq(nd, n, Owen(30, ScramblingRngAdapter(Blabla8()))), model, 1.0, payoff, 0, n, config))
    @test abs(price - CharFuncPricing.priceEuropean(pricer, true, 1.0, 1.0, tte, 1.0)) < 3 * stderr
    priceParallel, _ = MC.simulateSchemeParallel(ScrambledSobolSeq(nd, n, Owen(30, ScramblingRngAdapter(Blabla8()))), model, 1.0, payoff, 0, n, config)
    @test isapprox(price, priceParallel, rtol=1e-12)
end

@testset "SchobelZhuLV Van der Stoep" begin
    surface = vdsSurface()
    params = CharFuncPricing.SchobelZhuParams(0.17700060676549295, 0.47004835147845592, 0.12560749278392161, -0.55573864712677212, 0.13330983125180917)
    model = MC.SchobelZhuLVModel(SchobelZhuModel(params, 0.0, 0.0), LocalVolatilityModel(surface, 0.0, 0.0))
    tte, n = 2.0, 1024 * 16
    config = MC.GridConfig(model, [tte], 1.0 / 100, MC.EAE(withBB=true))
    nd = MC.ndims(model, config)
    sobol() = ScrambledSobolSeq(nd, n, Owen(30, ScramblingRngAdapter(Blabla8())))
    calc = MC.NonConformCalculator(gridSize=64)
    leverage = MC.calibrateLeverage(sobol(), model, 1.0, config, n, calculator=calc, nThreads=1)
    for strike in (0.75, 1.0, 1.41) #the model reproduces the vanilla options of the surface, unlike the Schöbel-Zhu model
        isCall = strike >= 1.0
        payoff = MC.VanillaOption(isCall, strike, MC.BulletCashFlow(tte, tte, false, 0.0), 0.0)
        price, stderr = MC.last_stat(MC.simulateScheme(sobol(), model, 1.0, payoff, 0, n, config, leverage=leverage))
        @test abs(price - blackScholesFormula(isCall, strike, 1.0, varianceByLogmoneyness(surface, log(strike), tte) * tte, 1.0, 1.0)) < 3 * stderr
    end
    l4 = MC.calibrateLeverage(sobol(), model, 1.0, config, n, calculator=calc, nThreads=4)
    @test all(leverage.conditionalVariances[k](x) == l4.conditionalVariances[k](x) for k in (1, 2, 100, 200) for x in (-0.3, 0.0, 0.2))
    #the Schöbel-Zhu leverage does not apply to the Heston-LV model (different scheme)
    @test_throws MethodError MC.simulateScheme(sobol(), model, 1.0, MC.VanillaOption(true, 1.0, MC.BulletCashFlow(tte, tte, false, 0.0), 0.0), 0, n, MC.GridConfig(model, [tte], 1.0 / 100, MC.QE()), leverage=leverage)
end
