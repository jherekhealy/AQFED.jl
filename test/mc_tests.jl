using AQFED: AQFED
using Test
import AQFED.Random: AbstractRNGSeq, ScramblingRngAdapter, ScrambledSobolSeq, Owen
using Random123: Random123
import Random: MersenneTwister
import AQFED.TermStructure:
    SVISection,
    VarianceSurfaceBySection,
    varianceByLogmoneyness,
    HestonModel,
    DoubleHestonModel,
    LocalVolatilityModel,
    ConstantBlackModel
using CharFuncPricing
#the studies formerly in this file (generator comparisons, VVIX, variance paths) are in book/legacy/mc_study.jl

@testset "BlackSim" begin
    model = ConstantBlackModel(1.0, 0.0, 0.0)
    spot = 1.0
    strike = 1.0
    payoff = AQFED.MonteCarlo.VanillaOption(true, strike, AQFED.MonteCarlo.BulletCashFlow(1.0, 1.0, false, 0.0), 0.0)
    refValue = AQFED.Black.blackScholesFormula(true, strike, spot, 1.0, 1.0, 1.0)
    gens = [
        AQFED.Random.MersenneTwister64(UInt64(20130129)),
        AQFED.Random.Mixmax17(UInt64(20130129)),
        AQFED.Random.Well1024a(UInt32(20130129)),
        AQFED.Random.Chacha8SIMD(),
        AQFED.Random.Blabla8(),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
    ]
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    nd = 1
    for rng in gens
        time = @elapsed value = AQFED.MonteCarlo.simulate(AbstractRNGSeq(rng, nd), model, spot, payoff, 64 * 1024)
        println(typeof(rng), " ", value, " ", refValue, " ", value - refValue, " ", time)
        @test isapprox(refValue, value, atol=1e-2)
    end
end

@testset "FlatLocalVolSim" begin
    spot = 100.0
    strike = 100.0
    payoff = AQFED.MonteCarlo.VanillaOption(true, strike, AQFED.MonteCarlo.BulletCashFlow(10.0, 10.0, false, 0.0), 0.0)
    section = SVISection(0.01, 0.0, 0.0, 0.01, 0.0, 1.0, spot)
    surface = VarianceSurfaceBySection([section], [1.0])
    model = LocalVolatilityModel(surface, 0.0, 0.0)

    refValue = AQFED.Black.blackScholesFormula(true, strike, spot, 0.01 * 10.0, 1.0, 1.0)
    gens = [
        AQFED.Random.MersenneTwister64(UInt64(20130129)),
        AQFED.Random.Mixmax17(UInt64(20130129)),
        AQFED.Random.Well1024a(UInt32(20130129)),
        AQFED.Random.Chacha8SIMD(),
        AQFED.Random.Blabla8(),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
    ]
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    nd = AQFED.MonteCarlo.ndims(model, specTimes, 1.0 / 16)
    for rng in gens
        time = @elapsed value, serr =
            AQFED.MonteCarlo.simulate(AbstractRNGSeq(rng, nd), model, spot, payoff, 0, 1024 * 64, 1.0 / 16)
        println(
            typeof(rng),
            " ",
            value,
            " ",
            refValue,
            " ",
            value - refValue,
            " ",
            serr,
            " ",
            time,
        )
        @test isapprox(refValue, value, atol=serr * 3)
    end
end


@testset "DAXLocalVolSim" begin
    spot = 100.0
    strike = spot
    tte = 1.0
    payoff = AQFED.MonteCarlo.VanillaOption(true, strike, AQFED.MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0)
    sections = [
        SVISection(0.030, 0.125, -1.0, 0.050, 0.074, 0.16, spot),
        SVISection(0.032, 0.094, -1.0, 0.041, 0.093, 0.26, spot),
        SVISection(0.028, 0.105, -1.0, 0.072, 0.096, 0.33, spot),
        SVISection(0.026, 0.080, -1.0, 0.098, 0.127, 0.58, spot),
        SVISection(0.026, 0.066, -1.0, 0.113, 0.153, 0.83, spot),
        SVISection(0.031, 0.047, -1.0, 0.065, 0.171, 1.33, spot),
        SVISection(0.037, 0.039, -1.0, 0.030, 0.152, 1.83, spot),
        SVISection(0.036, 0.036, -1.0, 0.083, 0.200, 2.33, spot),
        SVISection(0.038, 0.036, -1.0, 0.139, 0.170, 2.82, spot),
        SVISection(0.034, 0.032, -1.0, 0.199, 0.246, 3.32, spot),
        SVISection(0.044, 0.028, -1.0, 0.069, 0.188, 4.34, spot),
    ]

    surface = VarianceSurfaceBySection(
        sections,
        [0.16, 0.26, 0.33, 0.58, 0.83, 1.33, 1.83, 2.33, 2.82, 3.32, 4.34],
    )
    model = LocalVolatilityModel(surface, 0.0, 0.0)

    refValue = AQFED.Black.blackScholesFormula(
        true,
        100.0,
        100.0,
        tte * varianceByLogmoneyness(surface, 0.0, tte),
        1.0,
        1.0,
    )
    gens = [
        AQFED.Random.MersenneTwister64(UInt64(20130129)),
        AQFED.Random.Mixmax17(UInt64(20130129)),
        AQFED.Random.Well1024a(UInt32(20130129)),
        AQFED.Random.Chacha8SIMD(),
        AQFED.Random.Blabla8(),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
    ]
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    nd = AQFED.MonteCarlo.ndims(model, specTimes, 1.0 / 32)
    for rng in gens
        time = @elapsed value, serr =
            AQFED.MonteCarlo.simulate(AbstractRNGSeq(rng, nd), model, spot, payoff, 0, 1024 * 64, 1.0 / 32)
        println(
            typeof(rng),
            " ",
            value,
            " ",
            refValue,
            " ",
            value - refValue,
            " ",
            serr,
            " ",
            time,
        )
        @test isapprox(refValue, value, atol=serr * 5)
    end
end

@testset "HestonSim" begin
    hParams = AQFED.TermStructure.HestonModel(0.04, 0.5, 0.04, -0.9, 1.0, 0.0, 0.0)
    refValue = 13.08467
    gens = [
        MersenneTwister(20130129),
        AQFED.Random.MersenneTwister64(UInt64(20130129)),
        AQFED.Random.Well1024a(UInt32(20130129)),
        AQFED.Random.Mixmax17(UInt64(20130129)),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
        AQFED.Random.Chacha8SIMD(),
        AQFED.Random.Blabla8(),
    ]
    spot = 100.0
    payoff = AQFED.MonteCarlo.VanillaOption(true, 100.0, AQFED.MonteCarlo.BulletCashFlow(10.0, 10.0, false, 0.0), 0.0)
    timesteps = 8
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    config = AQFED.MonteCarlo.GridConfig(hParams, specTimes, 1.0 / timesteps, AQFED.MonteCarlo.DVSS2X(false), mindt=0.5/365, tstop=0.0)
    ndim = AQFED.MonteCarlo.ndims(hParams, config)
    start = 0
    n = 1024 * 64
    for rng in gens
        seq = AbstractRNGSeq(rng, ndim)
        time = @elapsed value, stderror =
            AQFED.MonteCarlo.simulateScheme(seq, hParams, spot, payoff, start, n, config)
        println(
            typeof(rng),
            " ",
            value,
            " ",
            refValue,
            " ",
            value - refValue,
            " ",
            stderror,
            " ",
            time,
        )
        @test isapprox(refValue, value, atol=3 * stderror)
    end


    seq =
        ScrambledSobolSeq(ndim, n, Owen(30, ScramblingRngAdapter(AQFED.Random.Blabla8())))
    time = @elapsed value, stderrorl = AQFED.MonteCarlo.simulateScheme(
        seq,
        hParams,
        spot,
        payoff,
        start,
        n,
       config
    )
    println(
        typeof(seq),
        " ",
        value,
        " ",
        refValue,
        " ",
        value - refValue,
        " ",
        stderrorl,
        " ",
        time,
    )
    @test isapprox(refValue, value, atol=2 * stderrorl)

     seq =
        ScrambledSobolSeq(ndim, n, AQFED.Random.Burley{ndim}())
    time = @elapsed value, stderrorl = AQFED.MonteCarlo.simulateScheme(
        seq,
        hParams,
        spot,
        payoff,
        start,
        n,
       config
    )
    println(
        typeof(seq),
        " ",
        value,
        " ",
        refValue,
        " ",
        value - refValue,
        " ",
        stderrorl,
        " ",
        time,
    )
    @test isapprox(refValue, value, atol=2 * stderrorl)

    payoff = AQFED.MonteCarlo.VanillaOption(true, 100.0, AQFED.MonteCarlo.BulletCashFlow(1.0, 1.0, false, 0.0), 0.0)
    refValue = 4.4031768153784405
    #for some reasons, FT scheme is really bad on this example => require 1000 steps for a good accuracy
    timesteps = 1000
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    ndim = AQFED.MonteCarlo.ndims(hParams, specTimes, 1.0 / timesteps)
    seq = ScrambledSobolSeq(ndim, n, Owen(30, ScramblingRngAdapter(AQFED.Random.Blabla8())))
    #FIXME NoScrambling() leads to NaN 
    time = @elapsed value, stderrorl = AQFED.MonteCarlo.simulateFullTruncation(
        seq,
        hParams,
        spot,
        payoff,
        1,
        n,
        1.0 / timesteps,
        withBB=false,
    )
    println(
        typeof(seq),
        " ",
        value,
        " ",
        refValue,
        " ",
        value - refValue,
        " ",
        stderrorl,
        " ",
        time,
    )
    @test isapprox(refValue, value, atol=stderrorl)
end

@testset "DoubleHestonDVSS2X" begin
    gens = [
        AQFED.Random.MersenneTwister64(UInt64(20130129)),
        AQFED.Random.Mixmax17(UInt64(20130129)),
        AQFED.Random.Well1024a(UInt32(20130129)),
        AQFED.Random.Chacha8SIMD(),
        AQFED.Random.Blabla8(),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
    ]
    rng = gens[3]
    τs = [1.0, 10.0]
    spot = 61.90
    strikes = [0.7, 1.0, 1.3] .* spot
    r = 0.03
    params1 = HestonParams(0.6^2, 0.9, 0.1, -0.5, 0.1)
    params2 = HestonParams(0.7^2, 1.2, 0.15, -0.5, 0.2)
    hParams = DoubleHestonModel(DoubleHestonParams(params1, params2), r, 0.0)
    refPrices = [27.6092 19.4569 13.9299;
        45.2866 41.4006 38.2780
    ]
    for (i, tte) = enumerate(τs)
        df = exp(-r * tte)
        for (j, strike) = enumerate(strikes)

            payoff = AQFED.MonteCarlo.VanillaOption(true, strike, AQFED.MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0)
            nd = AQFED.MonteCarlo.ndims(hParams, AQFED.MonteCarlo.specificTimes(payoff), 100.0)
            value, serr =
                AQFED.MonteCarlo.simulateDVSS2X(AbstractRNGSeq(rng, nd), hParams, spot, payoff, 0, 1024 * 64, 0.16)
            println(tte, " ", strike, " ", value, " ", serr)
            @test isapprox(refPrices[i, j], value, atol=serr * 2)
        end
    end
end
