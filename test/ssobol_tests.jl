import AQFED
using AQFED.Random, Statistics
import AQFED.MonteCarlo: BrownianBridgeConstruction, transform!, transformByDim!



function simulateGBM(rng::AbstractSeq, nSim, nSteps; withBB = false, size=ceil(Int, log2(nSteps))*4)
    tte = 1.0
    genTimes = collect(LinRange(0.0, tte, ceil(Int, nSteps * tte) + 1))
    logpayoffValues = Vector{Float64}(undef, nSim)
    logpayoffValues .= 0.0
    t0 = genTimes[1]
    local payoffValues
    u = Vector{Float64}(undef, nSim)
    z = u

    cache =  AQFED.MonteCarlo.BBCache{Int, Vector{Float64}}(size) #Dict{Int, Vector{Float64}}()
    local bb
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
    end

    @inbounds for j = 2:length(genTimes)
        t1 = genTimes[j]
        h = t1 - t0
        dim = j - 1
        if withBB
            transformByDim!(bb, rng, 1, dim, z, cache)
        else
            skipTo(rng, dim, 1) #don't use first point for Sobol
            nextn!(rng, dim, u)
            @. z = u * sqrt(h)
        end

        @. logpayoffValues += -h / 2 + z
        if (t1 == tte)
            payoffValues = @. exp(logpayoffValues)
            #payoffValues = @. max(payoffValues, 1/payoffValues)
        end
        t0 = t1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end

function simulateGBMIter(rng::AbstractSeq, nSim, nSteps; withBB = false)
    tte = 1.0
    genTimes = collect(LinRange(0.0, tte, ceil(Int, nSteps * tte) + 1))
    #println(length(genTimes) - 1)
    payoffValues = Vector{Float64}(undef, nSim)
    u = Vector{Float64}(undef, length(genTimes) - 1)
    z = u
    w = Vector{Float64}(undef, length(u))
    sqrth = Vector{Float64}(undef, length(u))
    @. sqrth = sqrt(genTimes[2:end] - genTimes[1:end-1])
    local bb
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
    end

    #mean = 0.0
    @inbounds for i = 1:nSim
        nextn!(rng, z)
        if withBB
            transform!(bb, z, w)
        else
            @. w = z * sqrth
        end
        #println(u," ",z)
        logpayoffValues = 0.0
        t0 = genTimes[1]
        @inbounds for j = 2:length(genTimes)
            t1 = genTimes[j]
            h = t1 - t0
            logpayoffValues += -h / 2 + w[j-1]
            t0 = t1
        end
        payoffValues[i] = exp(logpayoffValues)
        #mean += payoffValue
    end
    #@. payoffValues = exp(payoffValues)
    #payoffMean = mean/nSim
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues)), 0
end

@testset "GBM" begin
    n = 32 * 2 * 1024 - 1
    nSteps = 1000
    rng = ScrambledSobolSeq(nSteps, n, NoScrambling())
    value, mcerr,mcerr2 = simulateGBMIter(rng, n, nSteps)
    println(typeof(rng.scrambling), " ", value, " ", mcerr, " ",mcerr2)
    @test isapprox(value, 0.968395, atol = 1e-6)

    rng = ScrambledSobolSeq(nSteps, n, NoScrambling())
    value, mcerr = simulateGBM(rng, n, nSteps)
    println(typeof(rng.scrambling), " ", value, " ", mcerr)
    @test isapprox(value, 0.968395, atol = 1e-6)

    rng = ScrambledSobolSeq(nSteps, n, NoScrambling())
    value, mcerr,mcerr2 = simulateGBMIter(rng, n, nSteps, withBB = true)
    println(typeof(rng.scrambling), " ", value, " ", mcerr," ",mcerr2)
    @test isapprox(0.999562, value, atol= 1e-6)

    maxd = 30 #min(32,ceil(Int,log2(n)+10))
    rng = ScrambledSobolSeq(nSteps, n, Owen(maxd, OriginalScramblingRng()))
    value, mcerr,mcerr2 = simulateGBMIter(rng, n, nSteps)
    println(typeof(rng.scrambling), " ", value, " ", mcerr," ",mcerr2)
    #    @test isapprox(value, 1.000178, atol=1e-6)

    rng = ScrambledSobolSeq(
        nSteps,
        n,
        Owen(
            maxd,
            ScramblingRngAdapter(AQFED.Random.Blabla8(
                [
                    0xB105F00DBAAAAAAD,
                    0xdeadbeefcafebabe,
                    0xFEE1DEADFEE1DEAD,
                    0xdeadbeefcafebabe,
                ],
                UInt64,
            )),
        ),
    )
    value, mcerr, mcerr2 = simulateGBMIter(rng, n, nSteps)
    println(typeof(rng.scrambling), " ", value, " ", mcerr," ", mcerr2)
    #    @test isapprox(value, 1.000178, atol=1e-6)
    rng = ScrambledSobolSeq(nSteps, 1 << (maxd - 1), FaureTezuka(OriginalScramblingRng()))
    value, mcerr, mcerr2 = simulateGBMIter(rng, n, nSteps)
    println(typeof(rng.scrambling), " ", value, " ", mcerr," ", mcerr2)
    #    @test isapprox(value, 1.000178, atol=1e-6)

    rng = ScrambledSobolSeq(
        nSteps,
        1 << (maxd - 1),
        FaureTezuka(ScramblingRngAdapter(AQFED.Random.Blabla8(
            [
                0xB105F00DBAAAAAAD,
                0xdeadbeefcafebabe,
                0xFEE1DEADFEE1DEAD,
                0xdeadbeefcafebabe,
            ],
            UInt64,
        ))),
    )
    value, mcerr, mcerr2 = simulateGBMIter(rng, n, nSteps)
    println(typeof(rng.scrambling), " ", value, " ", mcerr," ", mcerr2)

    rng = DigitalSobolSeq(
        nSteps,
        n,
        AQFED.Random.Blabla8(
            [
                0xB105F00DBAAAAAAD,
                0xdeadbeefcafebabe,
                0xFEE1DEADFEE1DEAD,
                0xdeadbeefcafebabe,
            ],
            UInt32,
        ),
    )
    #AQFED.Random.Chacha8SIMD(UInt32))
    value, mcerr, mcerr2 = simulateGBMIter(rng, n, nSteps)
    println(typeof(rng), " ", value, " ", mcerr," ", mcerr2)

    rng = ModuloSobolSeq(
        nSteps,
        n,
        AQFED.Random.Blabla8(
            [
                0xB105F00DBAAAAAAD,
                0xdeadbeefcafebabe,
                0xFEE1DEADFEE1DEAD,
                0xdeadbeefcafebabe,
            ],
            Float64,
        ),
    )
    # AQFED.Random.Chacha8SIMD(Float64))
    value, mcerr, mcerr2 = simulateGBMIter(rng, n, nSteps)
    println(typeof(rng), " ", value, " ", mcerr," ", mcerr2)

end

@testset "ffz" begin
    n = AQFED.Random.ffz(1)
    println(n)
    @test isequal(1, n)
end

@testset "joekuofirst" begin
    nDim = 3
    n = 10
    #found via external Fortran implementation
    ref = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.75, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.375, 0.375, 0.625],
        [0.875, 0.875, 0.125],
        [0.625, 0.125, 0.875],
        [0.125, 0.625, 0.375],
        [0.1875, 0.3125, 0.9375],
        [0.6875, 0.8125, 0.4375],
    ]
    sobol = ScrambledSobolSeq(nDim, n, NoScrambling())
    points = zeros(nDim)
    for i = 2:n
        next!(sobol, points)
        println(points)
        for j = 1:nDim
            @test isapprox(ref[i][j], points[j], atol = 1e-15)
        end
    end
    skipTo(sobol, 0)
    for i = 1:n
        next!(sobol, points)
        #println(points)
        for j = 1:nDim
            @test isapprox(ref[i][j], points[j], atol = 1e-15)
        end
    end
end


@testset "joekuofirstbydim" begin
    nDim = 3
    n = 10
    #found via external Fortran implementation
    ref = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.75, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.375, 0.375, 0.625],
        [0.875, 0.875, 0.125],
        [0.625, 0.125, 0.875],
        [0.125, 0.625, 0.375],
        [0.1875, 0.3125, 0.9375],
        [0.6875, 0.8125, 0.4375],
    ]
    sobol = ScrambledSobolSeq(nDim, n, NoScrambling())
    points = zeros(n)
    for j = 1:nDim
        skipTo(sobol, j, 0)
        next!(sobol, j, points)
        #println(points)
        for i = 1:n
            @test isapprox(ref[i][j], points[i], atol = 1e-15)
        end
    end
end

@testset "owenfirst" begin
    nDim = 2
    n = 1 << 29
    #found via external Fortran implementation
    ref = [
        [0.102750, 0.861468],
        [0.705979, 0.113136],
        [0.978561, 0.532342],
        [0.329897, 0.282107],
        [0.408047, 0.647400],
        [0.775438, 0.397518],
        [0.534102, 0.943155],
        [0.149600, 0.191606],
        [0.214946, 0.586519],
        [0.601325, 0.337305],
        [0.842920, 0.761363],
        [0.473622, 0.010117],
        [0.270487, 0.933886],
        [0.921027, 0.185250],
        [0.648217, 0.733849],
        [0.043080, 0.482944],
        [0.015017, 0.694562],
        [0.676498, 0.444115],
        [0.887070, 0.894538],
        [0.304228, 0.146482],
        [0.444827, 0.785095],
        [0.871443, 0.033268],
        [0.567125, 0.610189],
        [0.249417, 0.360518],
        [0.181935, 0.981963],
        [0.501550, 0.230932],
        [0.806098, 0.686269],
        [0.377605, 0.436783],
        [0.361989, 0.509107],
        [0.946739, 0.258475],
        [0.735908, 0.838295],
        [0.072550, 0.089444],
        [0.078716, 0.559680],
        [0.729743, 0.309457],
        [0.956690, 0.857568],
        [0.352040, 0.109226],
        [0.398905, 0.962679],
        [0.784797, 0.211146],
        [0.526634, 0.635692],
        [0.156850, 0.385789],
        [0.224332, 0.773072],
        [0.592210, 0.021847],
        [0.850143, 0.566996],
        [0.466128, 0.317766],
        [0.294277, 0.737751],
        [0.897020, 0.486855],
        [0.670332, 0.906549],
        [0.021182, 0.157901],
        [0.053517, 0.882825],
        [0.637781, 0.134757],
    ]
    rng = OriginalScramblingRng()
    sobol = ScrambledSobolSeq(nDim, n, Owen(30, rng))
    points = zeros(nDim)
    for i = 1:length(ref)
        next!(sobol, points)
        #println(points)
        for j = 1:nDim
            @test isapprox(ref[i][j], points[j], atol = 1e-5)
        end
    end
    skipTo(sobol, 0)
    for i = 1:length(ref)
        next!(sobol, points)
        #println(points)
        for j = 1:nDim
            @test isapprox(ref[i][j], points[j], atol = 1e-5)
        end
    end
end

@testset "owenmaxd20" begin
    nDim = 2
    rng = OriginalScramblingRng()
    points = zeros(nDim)
    #found via external Fortran implementation
    eiRefs =
        [0.99712205451487534, 0.99931521505777710, 0.99886019692965533, 0.99998703162784253]
    n = 1024 * 4 + 1
    sobol = ScrambledSobolSeq(nDim, n, Owen(20, rng))
    sum = 0.0
    twoK = 512
    indexTwoK = 1
    for i = 1:n
        f = 1.0
        next!(sobol, points)
        for j = 1:nDim
            f = f * abs(4 * points[j] - 2)
        end
        if (i % twoK == 0)
            println("i=", i)
            println("ei=", (sum / i))
            @test isapprox(eiRefs[indexTwoK], sum / i, atol = 1e-15)
            indexTwoK += 1
            twoK *= 2
        end
        sum += f
    end
end

@testset "owenblabla" begin
    nDim = 2
    rng = ScramblingRngAdapter(AQFED.Random.Blabla8())
    points = zeros(nDim)
    #non reg values
    eiRefs =
        [0.9986280015086777, 0.9989421327867944, 0.9997705799667642, 0.9995579290533669]
    n = 1024 * 4 + 1
    sobol = ScrambledSobolSeq(nDim, n, Owen(20, rng))
    sum = 0.0
    twoK = 512
    indexTwoK = 1
    for i = 1:n
        f = 1.0
        next!(sobol, points)
        for j = 1:nDim
            f = f * abs(4 * points[j] - 2)
        end
        if (i % twoK == 0)
            println("i=", i)
            println("ei=", (sum / i))
            @test isapprox(eiRefs[indexTwoK], sum / i, atol = 1e-15)
            indexTwoK += 1
            twoK *= 2
        end
        sum += f
    end
end


@testset "fauretezukamaxd20" begin
    nDim = 2
    rng = OriginalScramblingRng()
    points = zeros(nDim)
    n = 1024 * 4
    sobol = ScrambledSobolSeq(nDim, n, FaureTezuka(rng))
    #found via external Fortran implementation
    firstNumbers = [
        [0.1610107421875, 0.0618896484375],
        [0.6610107421875, 0.5618896484375],
        [0.9110107421875, 0.3118896484375],
        [0.4110107421875, 0.8118896484375],
        [0.7860107421875, 0.9368896484375],
        [0.2860107421875, 0.4368896484375],
        [0.0360107421875, 0.6868896484375],
        [0.5360107421875, 0.1868896484375],
        [0.2235107421875, 0.9993896484375],
    ]
    eiRefs =
        [0.99863761523738503, 0.99997501377947628, 0.99842982890550047, 0.99996583949541673]
    sum = 0.0
    twoK = 512
    indexTwoK = 1
    for i = 1:n
        f = 1.0
        next!(sobol, points)
        if (i < 10)
            for j = 1:nDim
                @test isapprox(firstNumbers[i][j], points[j], atol = 1e-15)
            end
        end
        for j = 1:nDim
            f = f * abs(4 * points[j] - 2)
        end
        if (i % twoK == 0)
            println("i=", i)
            println("ei=", (sum / i))
            @test isapprox(eiRefs[indexTwoK], sum / i, atol = 1e-15)
            indexTwoK += 1
            twoK *= 2
        end
        sum += f
    end
end

@testset "owenfauretezukafirst" begin
    nDim = 2
    rng = OriginalScramblingRng()
    points = zeros(nDim)
    n = 1 << 29
    sobol = ScrambledSobolSeq(nDim, n, OwenFaureTezuka(30, rng))
    #found via external Fortran implementation
    ref = [
        [0.171337, 0.890245],
        [0.520179, 0.139048],
        [0.793268, 0.715480],
        [0.398030, 0.466157],
        [0.723320, 0.805123],
        [0.093221, 0.056202],
        [0.351145, 0.604769],
        [0.965125, 0.354211],
        [0.852922, 0.062542],
        [0.455807, 0.814408],
        [0.228854, 0.268739],
        [0.579604, 0.518367],
        [0.283907, 0.242083],
        [0.899795, 0.990335],
        [0.657731, 0.411004],
        [0.025754, 0.661258],
        [0.251729, 0.967130],
        [0.931755, 0.218420],
        [0.627906, 0.638114],
        [0.055796, 0.387280],
        [0.822363, 0.853245],
        [0.486095, 0.101836],
        [0.196431, 0.557142],
        [0.612299, 0.308095],
        [0.695144, 0.017341],
        [0.121127, 0.765866],
        [0.317320, 0.315412],
        [0.999222, 0.565451],
        [0.136776, 0.162291],
        [0.554521, 0.913885],
        [0.764845, 0.489338],
        [0.426669, 0.739181],
        [0.387960, 0.933425],
        [0.803337, 0.181814],
        [0.514136, 0.727452],
        [0.177378, 0.477631],
        [0.943946, 0.761955],
        [0.372326, 0.013440],
        [0.068013, 0.592802],
        [0.748529, 0.342749],
        [0.573562, 0.113545],
        [0.234897, 0.864975],
        [0.445738, 0.288573],
        [0.862992, 0.537603],
        [0.000546, 0.191084],
        [0.682939, 0.939780],
        [0.878615, 0.391182],
        [0.305087, 0.642026],
        [0.035106, 0.978609],
        [0.648597, 0.230371],
    ]
    for i = 1:length(ref)
        next!(sobol, points)
        for j = 1:nDim
            @test isapprox(ref[i][j], points[j], atol = 1e-5)
        end
    end

end
