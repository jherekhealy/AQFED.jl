import AQFED
import AQFED.Random
import AQFED.Random:
    AbstractRNGSeq,
    ZRNGSeq,
    OriginalScramblingRng,
    ScramblingRngAdapter,
    ScrambledSobolSeq,
    Owen, FaureTezuka, NoScrambling

import Random123
import RandomNumbers: AbstractRNG
import Random: MersenneTwister, rand!
using Statistics
import AQFED.TermStructure:
    SVISection,
    VarianceSurfaceBySection,
    varianceByLogmoneyness,
    HestonModel,
    LocalVolatilityModel,
    ConstantBlackModel,
    TSBlackModel


function estimateError(values::Vector{Float64}, k::Int, rng)
    n = length(values)
    mv = Vector{Float64}(undef, k)
    #stderr =  stdm(values, meanv) / sqrt(length(values))
    for i = 1:k
        indices = (rand(rng, UInt32, 100) .% n) .+ 1
        v = values[indices]
        mv[i] = mean(v)
    end
    stderr = stdm(mv, mean(values)) / sqrt(length(values) / 100)
    return stderr
end
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
            payoffValues = @. max(payoffValues, 1 / payoffValues)
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
    u = Vector{Float64}(undef, nSim)
    @inbounds for j = 2:length(genTimes)
        t1 = genTimes[j]
        h = t1 - t0
        dim = j - 1
        rand!(rng, u)
        z = @. AQFED.Math.norminv(u)
        @. logpayoffValues += -h / 2 + z * sqrt(h)
        if (t1 == tte)
            payoffValues = @. exp(logpayoffValues)
            payoffValues = @. max(payoffValues, 1 / payoffValues)
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
#        AQFED.Random.Well1024a(UInt32(20130129)),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
        AQFED.Random.Chacha8SIMD(),
        AQFED.Random.Blabla8(),
    ]
    for rng in gens
        value, mcerr = simulateGBMAnti(rng, 32 * 1024, nSteps)
        println(typeof(rng), " ", value, " ", mcerr)
    end
    gens = [
        MersenneTwister(201300129),
        AQFED.Random.MersenneTwister64(UInt64(20130129)),
        AQFED.Random.Mixmax17(UInt64(20130129)),
#        AQFED.Random.Well1024a(UInt32(20130129)),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
        AQFED.Random.Chacha8SIMD(),
        AQFED.Random.Blabla8(),
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
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    nd = 1
    for rng in gens
        time = @elapsed value = AQFED.MonteCarlo.simulate(AbstractRNGSeq(rng,nd), model, payoff, 64 * 1024)
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
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    nd = AQFED.MonteCarlo.ndims(model, specTimes, 1.0/16)
    for rng in gens
        time = @elapsed value, serr =
            AQFED.MonteCarlo.simulate(AbstractRNGSeq(rng,nd), model, payoff, 0, 1024 * 64, 1.0 / 16)
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
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    nd = AQFED.MonteCarlo.ndims(model, specTimes, 1.0/32)
    for rng in gens
        time = @elapsed value, serr =
        AQFED.MonteCarlo.simulate(AbstractRNGSeq(rng, nd), model, payoff, 0, 1024 * 64, 1.0 / 32)
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
    MersenneTwister(20130129),
        AQFED.Random.MersenneTwister64(UInt64(20130129)),
        AQFED.Random.Well1024a(UInt32(20130129)),
        AQFED.Random.Mixmax17(UInt64(20130129)),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
        AQFED.Random.Chacha8SIMD(),
        AQFED.Random.Blabla8(),
    ]
    payoff = AQFED.MonteCarlo.VanillaOption(true, 100.0, 10.0)
    timesteps = 8
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    ndims = AQFED.MonteCarlo.ndims(hParams, specTimes, 1.0 / timesteps)
    start = 0
    n = 1024 * 64
    for rng in gens
        seq = AbstractRNGSeq(rng, ndims)
        time = @elapsed value, stderror =
            AQFED.MonteCarlo.simulateDVSS2X(seq, hParams, payoff, start, n, 1.0/timesteps)
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


    seq =
        ScrambledSobolSeq(ndims, n, Owen(30, ScramblingRngAdapter(AQFED.Random.Blabla8())))
    time = @elapsed value, stderror = AQFED.MonteCarlo.simulateDVSS2X(
        seq,
        hParams,
        payoff,
        start,
        n,
        1.0/timesteps,
        withBB = false,
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
        stderror,
        " ",
        time,
    )
    @test isapprox(refValue, value, atol = 2 * stderror)

    payoff = AQFED.MonteCarlo.VanillaOption(true, 100.0, 1.0)
    refValue = 4.4031768153784405
    #for some reasons, FT scheme is really bad on this example => require 1000 steps for a good accuracy
    timesteps = 1000
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    ndims = AQFED.MonteCarlo.ndims(hParams, specTimes, 1.0 / timesteps)
    seq = ScrambledSobolSeq(ndims, n, NoScrambling())
    time = @elapsed value, stderror = AQFED.MonteCarlo.simulateFullTruncation(
        seq,
        hParams,
        payoff,
        1,
        n,
        1.0/timesteps,
        withBB = false,
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
        stderror,
        " ",
        time,
    )
    @test isapprox(refValue, value, atol = stderror)

    seq =
        ScrambledSobolSeq(ndims, n, NoScrambling())
    time = @elapsed value, stderror = AQFED.MonteCarlo.simulateFullTruncation(
        seq,
        hParams,
        payoff,
        1,
        n,
        1.0/timesteps,
        withBB = true,
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
        stderror,
        " ",
        time,
    )
    @test isapprox(refValue, value, atol = stderror/10)

    # ns = [1,2,4,8,16,32,64,128,256,512,1024]
    # vArray = Vector{Float64}(undef, length(ns))
    # for (i,n) in enumerate(ns)
    #     seq = ScrambledSobolSeq(ndims, 1<<29, FaureTezukaScrambling(OriginalScramblingRng()))
    #     value, stderror = AQFED.MonteCarlo.simulateDVSS2X(
    #             seq,
    #             hParams,
    #             payoff,
    #             1,
    #             n*1024,
    #             1.0/timesteps,
    #             withBB = false,
    #         )
    #     vArray[i] = value
    # end
end
# for gen in gens
#        global rng = gen
#        b = @benchmark  value,stderror = AQFED.MonteCarlo.simulateHestonDVSS2X(rng, hParams, payoff,1024*64,8)
#        println(typeof(rng)," ",b)
#        end

#using CharFuncPricing

@testset "ZigHeston" begin
    strike = 1.0
τ = 1.0
hParams = HestonParams(0.133, 0.35, 0.321, -0.63, 1.388)
m=1024
l=32
pricer = makeCosCharFuncPricer(Complex, Float64, Float64(MathConstants.pi), hParams, τ, m, l)
#priceEuropean(pricer, false, strike, spot, 1.0)
n = 1024*64
r = LinRange(0.95,1.05,11)
ve = Vector{Float64}(undef, length(r))
va = Vector{Float64}(undef, length(r))
for (i,spot) in enumerate(r)
#    model = HestonModel(hParams.v0, hParams.κ, hParams.θ, hParams.ρ, hParams.σ, spot, 0.0, 0.0)
    model = HestonModel(0.133, 0.35, 0.321, -0.63, 1.388, spot, 0.0, 0.0)
    payoff = AQFED.MonteCarlo.VanillaOption(true, strike,τ)
    refValue = 0.0 #priceEuropean(pricer, true, strike, spot,τ)
    timesteps = 100
    specTimes = AQFED.MonteCarlo.specificTimes(payoff)
    nd = AQFED.MonteCarlo.ndims(model, specTimes, 1.0 / timesteps)
    rng = MersenneTwister(2020) #AQFED.Random.Blabla8()
    seq = ZRNGSeq(rng, nd)
# ScrambledSobolSeq(ndims, n, NoScrambling())
    time = @elapsed value, stderror = AQFED.MonteCarlo.simulateFullTruncation(
    seq,
    model,
    payoff,
    0,
    n,
    1.0/timesteps,
    withBB = false, cacheSize=1000
)
ve[i] = value - refValue
va[i] = value
println(spot, " ",
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
end
end

@testset "DAXSims" begin
    spot = 100.0
    hParams = HestonModel(0.04, 0.5, 0.04, -0.9, 1.0, 100.0, 0.0, 0.0)

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
    gens = [
        AQFED.Random.MersenneTwister64(UInt64(20130129)),
        AQFED.Random.Mixmax17(UInt64(20130129)),
        AQFED.Random.Well1024a(UInt32(20130129)),
        AQFED.Random.Chacha8SIMD(),
        AQFED.Random.Blabla8(),
        Random123.Philox4x(UInt64, (20130129, 20100921), 10),
    ]

    ttes = [0.16, 1.0, 4.34]
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]

    for rng in gens
        time = @elapsed for tte in ttes
            for strike in strikes
                payoff = AQFED.MonteCarlo.VanillaOption(true, strike, tte)
                refValue = AQFED.Black.blackScholesFormula(
                    true,
                    strike,
                    100.0,
                    tte * varianceByLogmoneyness(surface, 0.0, tte),
                    1.0,
                    1.0,
                )

                model = TSBlackModel(100.0, surface, 0.0, 0.0)
                nd = AQFED.MonteCarlo.ndims(model, AQFED.MonteCarlo.specificTimes(payoff), 100.0)
                value, serr = AQFED.MonteCarlo.simulate(AbstractRNGSeq(rng,nd), model, payoff, 0, 1024 * 64)
                println(
                    typeof(rng),
                    " ",
                    typeof(model),
                    " ",
                    value,
                    " ",
                    refValue,
                    " ",
                    value - refValue,
                    " ",
                    serr,
                )
            end
        end
        println("elapsed ", time)

        time = @elapsed for tte in ttes
            for strike in strikes
                payoff = AQFED.MonteCarlo.VanillaOption(true, strike, tte)
                refValue = AQFED.Black.blackScholesFormula(
                    true,
                    strike,
                    100.0,
                    tte * varianceByLogmoneyness(surface, log(strike/spot), tte),
                    1.0,
                    1.0,
                )

                model = LocalVolatilityModel(100.0, surface, 0.0, 0.0)
                nd = AQFED.MonteCarlo.ndims(model, AQFED.MonteCarlo.specificTimes(payoff), 100.0)
                    value, serr =
                    AQFED.MonteCarlo.simulate(AbstractRNGSeq(rng,nd), model, payoff, 0, 1024 * 64, 0.16)
                println(
                    typeof(rng),
                    " ",
                    typeof(model),
                    " ",
                    value,
                    " ",
                    refValue,
                    " ",
                    value - refValue,
                    " ",
                    serr,
                )
            end
        end
        println(typeof(rng)," elapsed ", time)

        time = @elapsed for tte in ttes
            for strike in strikes
                payoff = AQFED.MonteCarlo.VanillaOption(true, strike, tte)
                refValue = AQFED.Black.blackScholesFormula(
                    true,
                    strike,
                    100.0,
                    tte * varianceByLogmoneyness(surface, log(strike/spot), tte),
                    1.0,
                    1.0,
                )

                nd = AQFED.MonteCarlo.ndims(hParams, AQFED.MonteCarlo.specificTimes(payoff), 100.0)
                    value, serr =
                    AQFED.MonteCarlo.simulateDVSS2X(AbstractRNGSeq(rng,nd), hParams, payoff, 0, 1024 * 64, 0.16)
                println(
                    typeof(rng),
                    " ",
                    typeof(hParams),
                    " ",
                    value,
                    " ",
                    refValue,
                    " ",
                    value - refValue,
                    " ",
                    serr,
                )
            end
        end
        println("elapsed ", time)
    end
    #        @test isapprox(refValue, value, atol = serr * 5)
end
