using AQFED, Test
using Statistics
import AQFED.Random: AbstractRNGSeq
import AQFED.TermStructure: ConstantBlackModel, TSBlackModel, ConstantRateCurve
import AQFED.MonteCarlo:
    simulateLongstaffSchwartz,
    BermudanOption,
    BermudanBasketOption,
    QRRegression,
    SVDRegression,
    CholeskyRegression

@testset "LongstaffSchwartz 1D Black" begin
    spot = 40.0
    strike = 40.0
    vol = 0.4
    r = 0.06
    tte = 1.0
    nEx = 50
    exerciseTimes = [tte / nEx * i for i in 1:nEx]
    model = ConstantBlackModel(vol, r, 0.0)
    payoff = BermudanOption(false, strike, exerciseTimes; fixing=spot, degree=3)

    priceRef = 5.308
    nTraining = 8192
    nSim = 32768
    rng = AQFED.Random.MersenneTwister64(UInt64(20130129))
    seq = AbstractRNGSeq(rng, nEx)

    regMethods = [
        QRRegression(),
        SVDRegression(thin=true),
        SVDRegression(thin=false),
        CholeskyRegression(lambda=1e-5, withConditioning=true)
    ]

    for reg in regMethods
        price, stdErr = simulateLongstaffSchwartz(
            seq,
            model,
            spot,
            payoff,
            nTraining,
            nSim,
            tte / nEx;
            regMethod = reg
        )
        println("1D Bermudan Put ($(typeof(reg))): price = $price, err = $stdErr, diff = $(price - priceRef)")
        @test isapprox(priceRef, price, atol = 0.15)
    end
end

@testset "LongstaffSchwartz 2D Basket Black" begin
    spots = [100.0, 100.0]
    strike = 1.0
    vols = [0.4, 0.2]
    weights = [0.4, 0.6]
    r = 0.0
    correlation = [1.0 0.5; 0.5 1.0]
    tte = 1.0
    exTimes = [tte / 4.0, tte / 2.0, tte * 0.75, tte]

    models = [ConstantBlackModel(vols[1], r, r), ConstantBlackModel(vols[2], r, r)]
    payoff = BermudanBasketOption(false, strike, weights, exTimes; fixings=spots)

    priceRef = 0.0966674
    nTraining = 8192
    nSim = 32768
    timestepSize = 0.25

    rng = AQFED.Random.MersenneTwister64(UInt64(20130129))
    seq = AbstractRNGSeq(rng, 4 * 2)

    regMethods = [
        QRRegression(),
        SVDRegression(thin=true),
        CholeskyRegression(lambda=0.0, withConditioning=true)
    ]

    for reg in regMethods
        price, stdErr = simulateLongstaffSchwartz(
            seq,
            models,
            spots,
            correlation,
            payoff,
            nTraining,
            nSim,
            timestepSize;
            regMethod = reg
        )
        println("2D Bermudan Basket Put ($(typeof(reg))): price = $price, err = $stdErr, diff = $(price - priceRef)")
        @test isapprox(priceRef, price, atol = 0.01)
    end
end

@testset "LongstaffSchwartz Iterative" begin
    spot = 40.0
    strike = 40.0
    vol = 0.4
    r = 0.06
    tte = 1.0
    nEx = 50
    exerciseTimes = [tte / nEx * i for i in 1:nEx]
    model = ConstantBlackModel(vol, r, 0.0)
    payoff = BermudanOption(false, strike, exerciseTimes; fixing=spot, degree=3)

    priceRef = 5.308
    nTraining = 8192
    nSim = 32768
    rng = AQFED.Random.MersenneTwister64(UInt64(20130129))
    seq = AbstractRNGSeq(rng, nEx)

    res = AQFED.MonteCarlo.simulateLongstaffSchwartzIter(
        seq,
        model,
        spot,
        payoff,
        nTraining,
        nSim,
        tte / nEx;
        regMethod = SVDRegression(thin=true)
    )

    price, stdErr = AQFED.MonteCarlo.last_stat(res)
    println("Iterative 1D Bermudan Put: price = $price, err = $stdErr, samples = $(res.samples[end])")
    @test isapprox(priceRef, price, atol = 0.15)
end

