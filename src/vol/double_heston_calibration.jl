using CharFuncPricing, Random123
using AQFED.Black, AQFED.GlobalOptimization, GaussNewton
import AQFED.Math:
    MQMinTransformation,
    LogisticTransformation,
    inv
export calibrateDoubleHestonFromPrices
using FiniteDifferences

function calibrateDoubleHestonFromVolsFixed(
    ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T},
    vols::AbstractMatrix{T};
    weights::AbstractMatrix=ones(size(vols)),
    method=:Cos,
    reduction::Reduction=NoReduction(),
    minimizer="DE", deviations=4.0,
    lower=[1e-4, 1e-4, 1e-4, -0.99, 1e-2, 1e-4, 1e-4, 1e-4, -0.9, 1e-2],
    upper=[4.0, 10.0, 4.0, 0.1, 4.0, 4.0, 10.0, 4.0, 0.99, 4.0,],
    autodiff=:single, trace=false, # :forward is ok for DNN1D
    initialGuess=DoubleHestonParams(HestonParams(0.0, 1.0, 0.0, -0.5, 0.5), HestonParams(0.0, 1.0, 0.0, -0.5, 0.5))
) where {T}
    #weights = filterWeights(uweights, ts, forwards, strikes, vols, deviations=3)
    function objectiveImmutableN(xr)
        x = expandX(reduction, DoubleHestonParams, xr)
        hParams = DoubleHestonParams(HestonParams(x[1], x[2], x[3], x[4], x[5]), HestonParams(x[6], x[7], x[8], x[9], x[10]))
        cf = DefaultCharFunc(hParams)

        mVols = impliedVolatilities(pricerMethod(method), cf, strikes, weights, forwards, ts)
        return vec(@. weights * (mVols - vols))
    end
    function objectiveN(F, xr)
        F .= objectiveImmutableN(xr)
        F
    end
    totalSize = size(vols, 1) * size(vols, 2)
    out = zeros(totalSize)
    function objective1(x)
        objectiveN(out, x)
        norm(out) / norm(weights)
    end
    strikeShortIndex = floor(Int, size(strikes, 2) / 2)
    volShort = vols[1, strikeShortIndex]
    volLong = vols[end, strikeShortIndex]
    x0 = [initialGuess.heston1.v0, initialGuess.heston1.κ, initialGuess.heston1.θ, initialGuess.heston1.ρ, initialGuess.heston1.σ,
    initialGuess.heston2.v0, initialGuess.heston2.κ, initialGuess.heston2.θ, initialGuess.heston2.ρ, initialGuess.heston2.σ]    
    if iszero(initialGuess.heston1.v0)
        x0[1] = volShort^2  / 2
        x0[6] = volShort^2  / 2
    end
    if iszero(initialGuess.heston1.θ)
        x0[3] = volLong^2 / 2
        x0[10] = volLong^2 / 2
    end

    rx0 = reduceX(reduction, DoubleHestonParams, x0)
    rlower = reduceX(reduction, DoubleHestonParams, lower)
    rupper = reduceX(reduction, DoubleHestonParams, upper)
    rng = Random123.Philox4x(UInt64, (20130129, 20100921), 10)
    problem = GlobalOptimization.Problem(length(rx0), objective1, rlower, rupper)
    rx = if minimizer == "DE"
        optim = GlobalOptimization.makeDifferentialEvolutionOptimizer(GlobalOptimization.OptimizerParams(30),
            problem,
            rng, GlobalOptimization.Best1Bin(0.9, 0.5))
        result = GlobalOptimization.optimize(optim, GlobalOptimization.TerminationCriteria(1000, 100, 1e-6, 1e-5))
        if trace
            println("Result DE ", result, " ", GlobalOptimization.minimizer(optim))
        end
        GlobalOptimization.minimizer(optim) #calibrated params.
    elseif minimizer == "SA"
        optim = GlobalOptimization.SimulatedAnnealing(problem, rng)
        result = GlobalOptimization.optimize(optim, rx0)
        if trace
            println("Result SA ", result, " ", GlobalOptimization.minimizer(optim))
        end
        rx = GlobalOptimization.minimizer(optim)
    elseif minimizer == "BFGS"
        result = Optim.optimize(objective1, rlower, rupper, rx0, Optim.Fminbox())
        rx = Optim.minimizer(result)
        if trace
            println("Result BFGS ", result, " ", rx)
        end
        rx
    else
        rx0
    end

    transforms = map((low, high) -> LogisticTransformation(low, high), rlower, rupper)

    function objectiveC(F, xinv)
        x = map((xi, transformi) -> transformi(xi), xinv, transforms)
        #objectiveN(F, x)
        F .= objectiveImmutableN(x)
    end
    rxinv = map((xi, transformi) -> inv(transformi, xi), rx, transforms)
    gnRmse = if autodiff == :zygote
        function objectiveCDer(Jac, xinv)
            Jac[:, :] .= Zygote.jacobian(xinv -> begin
                    x = map((xi, transformi) -> transformi(xi), xinv, transforms)
                    objectiveImmutableN(x)
                end, xinv)[1]

        end

        GaussNewton.optimize!(objectiveC, objectiveCDer, rxinv, out)
    else
        GaussNewton.optimize!(objectiveC, rxinv, out, autodiff=autodiff)
    end
    if trace
        println("Result GN ", gnRmse)
    end
    rx = map((xi, transformi) -> transformi(xi), rxinv, transforms)
    x = expandX(reduction, DoubleHestonParams, rx)
    rmse = gnRmse
    params = DoubleHestonParams(HestonParams(x[1], x[2], x[3], x[4], x[5]), HestonParams(x[6], x[7], x[8], x[9], x[10]))
    return params, rmse
end