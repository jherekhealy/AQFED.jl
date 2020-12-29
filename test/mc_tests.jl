import AQFED.Random
import AQFED
import Random123
import Random: MersenneTwister
using Statistics
import AQFED.TermStructure:
    SVISection,
    VarianceSurfaceBySection,
    varianceByLogmoneyness,
    HestonModel,
    LocalVolatilityModel,
    ConstantBlackModel

function simulateGBMAnti(rng, nSim, nSteps)
    tte = 1.0
    genTimes = LinRange(0.0, tte, ceil(Int, nSteps * tte) + 1)
    logpayoffValues = Vector{Float64}(undef, nSim * 2)
    logpayoffValues .= 0.0
    t0 = 0.0
    local payoffValues
    for t1 in genTimes[2:end]
        h = t1 - t0
        u = rand(rng, Float64, nSim)
        z = @. AQFED.Math.norminv(u)
        z = vcat(z, -z)
        @. logpayoffValues += -0.5 * h + z * sqrt(h)
        if (t1 == tte)
            payoffValues = @. exp(logpayoffValues)
            payoffValues = @. max(payoffValues, 1/payoffValues)
        end
        t0 = t1
    end

    payoffMean = mean(payoffValues)
    payoffValuesA = (payoffValues[1:nSim] .+ payoffValues[nSim+1:end]) ./ 2
    return payoffMean, stdm(payoffValuesA, payoffMean) / sqrt(length(payoffValuesA))
end

function simulateGBM(rng, nSim, nSteps)
    tte = 1.0
    genTimes = LinRange(0.0, tte, ceil(Int, nSteps * tte) + 1)
    logpayoffValues = Vector{Float64}(undef, nSim)
    logpayoffValues .= 0.0
    t0 = genTimes[1]
    local payoffValues
    for t1 in genTimes[2:end]
        h = t1 - t0
        u = rand(rng, Float64, nSim)
        z = @. AQFED.Math.norminv(u)
        @. logpayoffValues += - h / 2 + z * sqrt(h)
        if (t1 == tte)
            payoffValues = @. exp(logpayoffValues)
            payoffValues = @. max(payoffValues, 1/payoffValues)
        end
        t0 = t1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end

@testset "Antithetic" begin
    nSteps = 10
    gens = [
        MersenneTwister(201300129),
        AQFED.Random.MersenneTwister64(UInt64(20130129)),
        AQFED.Random.Mixmax17(UInt64(20130129)),
        AQFED.Random.Well1024a(UInt32(20130129)),
        AQFED.Random.ChachaSIMD(),
        AQFED.Random.Blabla(),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
    ]
    for rng in gens
        value, mcerr = simulateGBMAnti(rng, 32 * 1024, nSteps)
        println(typeof(rng), " ", value, " ", mcerr)
    end
    gens = [
        MersenneTwister(201300129),
        AQFED.Random.MersenneTwister64(UInt64(20130129)),
        AQFED.Random.Mixmax17(UInt64(20130129)),
        AQFED.Random.Well1024a(UInt32(20130129)),
        AQFED.Random.ChachaSIMD(),
        AQFED.Random.Blabla(),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
    ]
    for rng in gens
        value, mcerr = simulateGBM(rng, 32 * 2 * 1024, nSteps)
        println(typeof(rng), " ", value, " ", mcerr)
    end
end

@testset "BlackSim" begin
    model = ConstantBlackModel(1.0, 1.0, 0.0, 0.0)
    payoff = AQFED.MonteCarlo.VanillaOption(true, 1.0, 1.0)
    refValue = AQFED.Black.blackScholesFormula(true, 1.0, 1.0, 1.0, 1.0, 1.0)
    gens = [
        AQFED.Random.MersenneTwister64(UInt64(20130129)),
        AQFED.Random.Mixmax17(UInt64(20130129)),
        AQFED.Random.Well1024a(UInt32(20130129)),
        AQFED.Random.Chacha8SIMD(),
        AQFED.Random.Blabla8(),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
    ]
    for rng in gens
        time = @elapsed value = AQFED.MonteCarlo.simulate(
            rng,
            model,
            payoff,
            64 * 1024,
        )
        println(typeof(rng), " ", value, " ", refValue, " ", value - refValue, " ", time)
        @test isapprox(refValue, value, atol = 1e-2)
    end
end

@testset "FlatLocalVolSim" begin
    payoff = AQFED.MonteCarlo.VanillaOption(true, 100.0, 10.0)
    section = SVISection(0.01, 0.0, 0.0, 0.01, 0.0, 1.0, 100.0)
    surface = VarianceSurfaceBySection([section], [1.0])
    model = LocalVolatilityModel(100.0, surface, 0.0, 0.0)

    refValue = AQFED.Black.blackScholesFormula(true, 100.0, 100.0, 0.01 * 10.0, 1.0, 1.0)
    gens = [
        AQFED.Random.MersenneTwister64(UInt64(20130129)),
        AQFED.Random.Mixmax17(UInt64(20130129)),
        AQFED.Random.Well1024a(UInt32(20130129)),
        AQFED.Random.Chacha8SIMD(),
        AQFED.Random.Blabla8(),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
    ]
    for rng in gens
        time = @elapsed value, serr =
            AQFED.MonteCarlo.simulate(rng, model, payoff, 1024 * 64, 16)
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
        @test isapprox(refValue, value, atol = serr * 3)
    end
end


@testset "DAXLocalVolSim" begin
    spot = 100.0
    tte = 1.0
    payoff = AQFED.MonteCarlo.VanillaOption(true, spot, tte)
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
    model = LocalVolatilityModel(100.0, surface, 0.0, 0.0)

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
    for rng in gens
        time = @elapsed value, serr =
            AQFED.MonteCarlo.simulate(rng, model, payoff, 1024 * 64, 32)
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
        @test isapprox(refValue, value, atol = serr * 5)
    end
end

@testset "HestonSim" begin
    hParams = HestonModel(0.04, 0.5, 0.04, -0.9, 1.0, 100.0, 0.0, 0.0)
    refValue = 13.08467
    gens = [
        AQFED.Random.MersenneTwister64(UInt64(20130129)),
        AQFED.Random.Mixmax17(UInt64(20130129)),
        AQFED.Random.Well1024a(UInt32(20130129)),
        AQFED.Random.Chacha8SIMD(),
        AQFED.Random.Blabla8(),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
    ]
    payoff = AQFED.MonteCarlo.VanillaOption(true, 100.0, 10.0)

    for rng in gens
        time = @elapsed value, stderror =
            AQFED.MonteCarlo.simulateDVSS2X(rng, hParams, payoff, 1024 * 64, 8)
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
        @test isapprox(refValue, value, atol = 3 * stderror)
    end
end
# for gen in gens
#        global rng = gen
#        b = @benchmark  value,stderror = AQFED.MonteCarlo.simulateHestonDVSS2X(rng, hParams, payoff,1024*64,8)
#        println(typeof(rng)," ",b)
#        end
