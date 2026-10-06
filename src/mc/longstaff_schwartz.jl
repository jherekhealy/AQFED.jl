using Statistics
using LinearAlgebra
import AQFED.TermStructure: ConstantBlackModel, TSBlackModel, discountFactor, forward, logForward, varianceByLogmoneyness
import AQFED.Random: skipTo!, nextn!

export simulateLongstaffSchwartz, simulateLongstaffSchwartzIter, ndims

function ndims(models::AbstractVector{<:Union{ConstantBlackModel, TSBlackModel}}, specificTimes::Vector{Float64}, timestepSize::Float64)
    genTimes = pathgenTimes(models[1], specificTimes, timestepSize)
    return (length(genTimes) - 1) * length(models)
end

function ndims(models::AbstractVector{<:Union{ConstantBlackModel, TSBlackModel}}, config::GridConfig)
    return (length(config.genTimes) - 1) * length(models)
end

function ndims(model::Union{ConstantBlackModel, TSBlackModel}, config::GridConfig)
    return length(config.genTimes) - 1
end

function trainLongstaffSchwartz(
    rng,
    models::AbstractVector{<:Union{ConstantBlackModel, TSBlackModel}},
    spots::AbstractVector{T},
    correlation::AbstractMatrix{T},
    payoff::EarlyExerciseMCPayoff,
    nTraining::Int,
    timestepSize::Float64;
    regMethod::RegressionMethod = QRRegression(),
    withBB::Bool = false,
    cacheSize::Int = 0,
    useAllPaths::Bool = false,
    isTsitsiklisVanRoy::Bool = false
) where {T}
    specTimes = specificTimes(payoff)
    exTimes = exerciseTimes(payoff)
    nEx = length(exTimes)
    nAssets = length(spots)
    regDim = regressionVariablesDimension(payoff)

    genTimes = pathgenTimes(models[1], specTimes, timestepSize)
    nSteps = length(genTimes) - 1

    dfEx = [discountFactor(models[1], t) for t in exTimes]
    L = cholesky(Symmetric(correlation)).L

    local bb, cache
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
        if cacheSize == 0
            cacheSize = ceil(Int, log2(nSteps)) * 4
        end
        cache = BBCache{Int, Array{Float64, 2}}(cacheSize)
    end

    # Phase 1: Training paths
    exerciseFlowsTr = zeros(T, nTraining, nEx)
    regVarsTr = zeros(T, nTraining, nEx, regDim)
    logPathTr = zeros(T, nTraining, nAssets)
    for a in 1:nAssets
        logPathTr[:, a] .= log(spots[a])
    end

    exIndexTr = 1
    z_stepTr = zeros(T, nTraining, nAssets)
    W_stepTr = zeros(T, nTraining, nAssets)

    t0 = genTimes[1]
    for (dim, t1) in enumerate(genTimes[2:end])
        dt = t1 - t0
        sqrtdt = sqrt(dt)

        if withBB
            transformByDim!(bb, rng, 0, dim, z_stepTr, cache)
        else
            for a in 1:nAssets
                dimRng = (dim - 1) * nAssets + a
                skipTo!(rng, dimRng, 0, nTraining)
                nextn!(rng, dimRng, @view(z_stepTr[:, a]))
            end
        end

        mul!(W_stepTr, z_stepTr, L')

        for a in 1:nAssets
            m = models[a]
            lnspot = log(spots[a])
            lnf0 = logForward(m, lnspot, t0)
            lnf1 = logForward(m, lnspot, t1)

            if m isa ConstantBlackModel
                volSq = m.vol^2
            else
                v1 = varianceByLogmoneyness(m.surface, 0.0, t1) * t1
                v0 = varianceByLogmoneyness(m.surface, 0.0, t0) * t0
                volSq = max(v1 - v0, 0.0) / dt
            end
            vol = sqrt(volSq)
            drift = (lnf1 - lnf0) - 0.5 * volSq * dt

            @. logPathTr[:, a] += drift + vol * sqrtdt * W_stepTr[:, a]
        end

        if exIndexTr <= nEx && abs(t1 - exTimes[exIndexTr]) < 1e-8
            for i in 1:nTraining
                spotVec = exp.(@view(logPathTr[i, :]))
                flow = evaluateEarlyExercise!(payoff, spotVec, @view(regVarsTr[i, exIndexTr, :]))
                exerciseFlowsTr[i, exIndexTr] = flow * dfEx[exIndexTr]
            end
            exIndexTr += 1
        end
        t0 = t1
    end

    # Backward induction to find decision rules
    fits = Vector{LinearRegressionFit{T}}(undef, nEx - 1)
    for k in (nEx - 1):-1:1
        itmIndices = Int[]
        for i in 1:nTraining
            if exerciseFlowsTr[i, k] > 0 || useAllPaths
                push!(itmIndices, i)
            end
        end

        nITM = length(itmIndices)
        if nITM > 0
            X = zeros(T, nITM, regDim)
            y = zeros(T, nITM)
            for (idx, i) in enumerate(itmIndices)
                X[idx, :] .= regVarsTr[i, k, :]
                y[idx] = exerciseFlowsTr[i, k + 1]
            end

            debugTime = @elapsed fit = regress(regMethod, X, y)
            fits[k] = fit
            #println(debugTime, " seconds to fit regression at exercise time ", exTimes[k]," ", typeof(regMethod))
            for i in 1:nTraining
                immVal = exerciseFlowsTr[i, k]
                if immVal > 0 || useAllPaths
                    contVal = predict(fit, @view(regVarsTr[i, k, :]))
                    if contVal > immVal
                        exerciseFlowsTr[i, k] = isTsitsiklisVanRoy ? contVal : exerciseFlowsTr[i, k + 1]
                    end
                else
                    exerciseFlowsTr[i, k] = exerciseFlowsTr[i, k + 1]
                end
            end
        else
            fits[k] = LinearRegressionFit(zeros(T, regDim))
            for i in 1:nTraining
                exerciseFlowsTr[i, k] = exerciseFlowsTr[i, k + 1]
            end
        end
    end

    return fits, genTimes, dfEx, L, nSteps, nEx, nAssets, regDim
end

function simulateLongstaffSchwartz(
    rng,
    model::Union{ConstantBlackModel, TSBlackModel},
    spot::Float64,
    payoff::EarlyExerciseMCPayoff,
    nTraining::Int,
    nSim::Int,
    timestepSize::Float64;
    regMethod::RegressionMethod = QRRegression(),
    withBB::Bool = false,
    cacheSize::Int = 0,
    isAntithetic::Bool = false
)
    return simulateLongstaffSchwartz(
        rng,
        [model],
        [spot],
        ones(1, 1),
        payoff,
        nTraining,
        nSim,
        timestepSize;
        regMethod = regMethod,
        withBB = withBB,
        cacheSize = cacheSize,
        isAntithetic = isAntithetic
    )
end

function simulateLongstaffSchwartz(
    rng,
    models::AbstractVector{<:Union{ConstantBlackModel, TSBlackModel}},
    spots::AbstractVector{T},
    correlation::AbstractMatrix{T},
    payoff::EarlyExerciseMCPayoff,
    config::GridConfig,
    nTraining::Int,
    nSim::Int;
    regMethod::RegressionMethod = QRRegression(),
    withBB::Bool = false,
    cacheSize::Int = 0,
    isAntithetic::Bool = false
) where {T}
    return simulateLongstaffSchwartz(
        rng,
        models,
        spots,
        correlation,
        payoff,
        nTraining,
        nSim,
        config.mindt;
        regMethod = regMethod,
        withBB = withBB,
        cacheSize = cacheSize,
        isAntithetic = isAntithetic
    )
end

function simulateLongstaffSchwartz(
    rng,
    models::AbstractVector{<:Union{ConstantBlackModel, TSBlackModel}},
    spots::AbstractVector{T},
    correlation::AbstractMatrix{T},
    payoff::EarlyExerciseMCPayoff,
    nTraining::Int,
    nSim::Int,
    timestepSize::Float64;
    regMethod::RegressionMethod = QRRegression(),
    withBB::Bool = false,
    cacheSize::Int = 0,
    isAntithetic::Bool = false,
    useAllPaths::Bool = false,
    isTsitsiklisVanRoy::Bool = false
) where {T}
    fits, genTimes, dfEx, L, nSteps, nEx, nAssets, regDim = trainLongstaffSchwartz(
        rng, models, spots, correlation, payoff, nTraining, timestepSize;
        regMethod = regMethod, withBB = withBB, cacheSize = cacheSize,
        useAllPaths = useAllPaths, isTsitsiklisVanRoy = isTsitsiklisVanRoy
    )

    specTimes = specificTimes(payoff)
    exTimes = exerciseTimes(payoff)

    local bb, cache
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
        if cacheSize == 0
            cacheSize = ceil(Int, log2(nSteps)) * 4
        end
        cache = BBCache{Int, Array{Float64, 2}}(cacheSize)
    end

    # Phase 2: Simulation / Pricing pass
    exerciseFlowsSim = zeros(T, nSim, nEx)
    regVarsSim = zeros(T, nSim, nEx, regDim)
    logPathSim = zeros(T, nSim, nAssets)
    for a in 1:nAssets
        logPathSim[:, a] .= log(spots[a])
    end

    exIndexSim = 1
    z_stepSim = zeros(T, nSim, nAssets)
    W_stepSim = zeros(T, nSim, nAssets)

    startSimIndex = 1
    t0 = genTimes[1]
    for (dim, t1) in enumerate(genTimes[2:end])
        dt = t1 - t0
        sqrtdt = sqrt(dt)

        if withBB
            transformByDim!(bb, rng, startSimIndex, dim, z_stepSim, cache)
        else
            for a in 1:nAssets
                dimRng = (dim - 1) * nAssets + a
                skipTo!(rng, dimRng, startSimIndex, nSim)
                nextn!(rng, dimRng, @view(z_stepSim[:, a]))
            end
        end

        mul!(W_stepSim, z_stepSim, L')

        for a in 1:nAssets
            m = models[a]
            lnspot = log(spots[a])
            lnf0 = logForward(m, lnspot, t0)
            lnf1 = logForward(m, lnspot, t1)

            if m isa ConstantBlackModel
                volSq = m.vol^2
            else
                v1 = varianceByLogmoneyness(m.surface, 0.0, t1) * t1
                v0 = varianceByLogmoneyness(m.surface, 0.0, t0) * t0
                volSq = max(v1 - v0, 0.0) / dt
            end
            vol = sqrt(volSq)
            drift = (lnf1 - lnf0) - 0.5 * volSq * dt

            @. logPathSim[:, a] += drift + vol * sqrtdt * W_stepSim[:, a]
        end

        if exIndexSim <= nEx && abs(t1 - exTimes[exIndexSim]) < 1e-8
            for i in 1:nSim
                spotVec = exp.(@view(logPathSim[i, :]))
                flow = evaluateEarlyExercise!(payoff, spotVec, @view(regVarsSim[i, exIndexSim, :]))
                exerciseFlowsSim[i, exIndexSim] = flow * dfEx[exIndexSim]
            end
            exIndexSim += 1
        end
        t0 = t1
    end

    payoffValues = zeros(T, nSim)
    for i in 1:nSim
        exVal = exerciseFlowsSim[i, nEx]
        for k in (nEx - 1):-1:1
            immVal = exerciseFlowsSim[i, k]
            if immVal > 0 || useAllPaths
                contVal = predict(fits[k], @view(regVarsSim[i, k, :]))
                if contVal > immVal
                    exerciseFlowsSim[i, k] = isTsitsiklisVanRoy ? contVal : exerciseFlowsSim[i, k + 1]
                end
            else
                exerciseFlowsSim[i, k] = exerciseFlowsSim[i, k + 1]
            end
            exVal = exerciseFlowsSim[i, k]
        end
        payoffValues[i] = exVal
    end

    payoffMean = mean(payoffValues)
    payoffStdErr = stdm(payoffValues, payoffMean) / sqrt(nSim)
    return payoffMean, payoffStdErr
end

function simulateLongstaffSchwartzIter(
    rng,
    model::Union{ConstantBlackModel, TSBlackModel},
    spot::Float64,
    payoff::EarlyExerciseMCPayoff,
    nTraining::Int,
    nSim::Int,
    timestepSize::Float64;
    regMethod::RegressionMethod = QRRegression(),
    withBB::Bool = false,
    cacheSize::Int = 0,
    isAntithetic::Bool = false
)
    return simulateLongstaffSchwartzIter(
        rng,
        [model],
        [spot],
        ones(1, 1),
        payoff,
        nTraining,
        nSim,
        timestepSize;
        regMethod = regMethod,
        withBB = withBB,
        cacheSize = cacheSize,
        isAntithetic = isAntithetic
    )
end

function simulateLongstaffSchwartzIter(
    rng,
    models::AbstractVector{<:Union{ConstantBlackModel, TSBlackModel}},
    spots::AbstractVector{T},
    correlation::AbstractMatrix{T},
    payoff::EarlyExerciseMCPayoff,
    config::GridConfig,
    nTraining::Int,
    nSim::Int;
    regMethod::RegressionMethod = QRRegression(),
    withBB::Bool = false,
    cacheSize::Int = 0,
    isAntithetic::Bool = false
) where {T}
    return simulateLongstaffSchwartzIter(
        rng,
        models,
        spots,
        correlation,
        payoff,
        nTraining,
        nSim,
        config.mindt;
        regMethod = regMethod,
        withBB = withBB,
        cacheSize = cacheSize,
        isAntithetic = isAntithetic
    )
end

function simulateLongstaffSchwartzIter(
    rng,
    models::AbstractVector{<:Union{ConstantBlackModel, TSBlackModel}},
    spots::AbstractVector{T},
    correlation::AbstractMatrix{T},
    payoff::EarlyExerciseMCPayoff,
    nTraining::Int,
    nSim::Int,
    timestepSize::Float64;
    regMethod::RegressionMethod = QRRegression(),
    withBB::Bool = false,
    cacheSize::Int = 0,
    isAntithetic::Bool = false,
    useAllPaths::Bool = false,
    isTsitsiklisVanRoy::Bool = false,
    start::Int = 1,
    isRestart::Bool = true
) where {T}
    fits, genTimes, dfEx, L, nSteps, nEx, nAssets, regDim = trainLongstaffSchwartz(
        rng, models, spots, correlation, payoff, nTraining, timestepSize;
        regMethod = regMethod, withBB = withBB, cacheSize = cacheSize,
        useAllPaths = useAllPaths, isTsitsiklisVanRoy = isTsitsiklisVanRoy
    )

    specTimes = specificTimes(payoff)
    exTimes = exerciseTimes(payoff)

    payoffValues = zeros(T, nSim)

    # Pre-allocate small path buffers to avoid per-path memory allocations
    logPath = zeros(T, nAssets)
    z_step = zeros(T, nAssets)
    W_step = zeros(T, nAssets)
    pathFlows = zeros(T, nEx)
    pathRegVars = zeros(T, nEx, regDim)
    spotVec = zeros(T, nAssets)

    startSimIndex = isRestart ? (nTraining + 1) : start

    for sim in 1:nSim
        logPath .= log.(spots)
        exIdx = 1
        t0 = genTimes[1]

        for (dim, t1) in enumerate(genTimes[2:end])
            dt = t1 - t0
            sqrtdt = sqrt(dt)

            for a in 1:nAssets
                dimRng = (dim - 1) * nAssets + a
                skipTo!(rng, dimRng, startSimIndex + sim - 1, 1)
                nextn!(rng, dimRng, @view(z_step[a:a]))
            end

            mul!(W_step, L, z_step)

            for a in 1:nAssets
                m = models[a]
                lnspot = log(spots[a])
                lnf0 = logForward(m, lnspot, t0)
                lnf1 = logForward(m, lnspot, t1)

                if m isa ConstantBlackModel
                    volSq = m.vol^2
                else
                    v1 = varianceByLogmoneyness(m.surface, 0.0, t1) * t1
                    v0 = varianceByLogmoneyness(m.surface, 0.0, t0) * t0
                    volSq = max(v1 - v0, 0.0) / dt
                end
                vol = sqrt(volSq)
                drift = (lnf1 - lnf0) - 0.5 * volSq * dt

                logPath[a] += drift + vol * sqrtdt * W_step[a]
            end

            if exIdx <= nEx && abs(t1 - exTimes[exIdx]) < 1e-8
                spotVec .= exp.(logPath)
                flow = evaluateEarlyExercise!(payoff, spotVec, @view(pathRegVars[exIdx, :]))
                pathFlows[exIdx] = flow * dfEx[exIdx]
                exIdx += 1
            end
            t0 = t1
        end

        exVal = pathFlows[nEx]
        for k in (nEx - 1):-1:1
            immVal = pathFlows[k]
            if immVal > 0 || useAllPaths
                contVal = predict(fits[k], @view(pathRegVars[k, :]))
                if contVal > immVal
                    pathFlows[k] = isTsitsiklisVanRoy ? contVal : pathFlows[k + 1]
                end
            else
                pathFlows[k] = pathFlows[k + 1]
            end
            exVal = pathFlows[k]
        end
        payoffValues[sim] = exVal
    end

    return mc_convergence_detailed(payoffValues)
end
