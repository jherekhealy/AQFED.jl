import AQFED.Math: norminv
using Statistics
import AQFED.TermStructure: ConstantBlackModel, TSBlackModel, varianceByLogmoneyness


function simulate(rng, model::ConstantBlackModel, payoff::VanillaOption, nSim::Int)
    tte = payoff.maturity
    sqrtte = sqrt(tte)
    df = exp(-model.r * tte)
    z = Vector{Float64}(undef, nSim)
    nextn!(rng, z)
    pathValues =
        @. exp(model.vol * z * sqrtte + (model.r - model.q - 0.5 * model.vol^2) * tte)
    mean(x -> evaluatePayoff(payoff, x, df), pathValues)
end

function ndims(model::TSBlackModel, specificTimes::Vector{Float64}, timestepSize::Float64)
    genTimes = pathgenTimes(model, specificTimes, timestepSize)
    return (length(genTimes) - 1) #0.0 does not count
end

function pathgenTimes(model, specificTimes::Vector{Float64}, timestepSize::Float64)
    t0 = 0.0
    pathTimes = Vector{Float64}(undef, 1)
    pathTimes[1] = t0
    for t1 in specificTimes
        currentSize = t1 - t0
        if timestepSize >= currentSize
            push!(pathTimes, t1)
        else
            nSubsteps = trunc(Int, currentSize / timestepSize + 0.5)
            dt = currentSize / nSubsteps
            for j = 1:nSubsteps-1
                push!(pathTimes, t0 + dt * j)
            end
            push!(pathTimes, t1)
        end
        t0 = t1
    end
    return pathTimes
end


function simulate(
    rng,
    model::TSBlackModel{S},
    payoff::VanillaOption,
    start::Int,
    nSim::Int;
    withBB = false,
) where {S}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    genTimes = pathgenTimes(model, specTimes, tte)

    local bb, cache
    if withBB
        cacheSize = ceil(Int, log2(length(genTimes))) * 4
        bb = BrownianBridgeConstruction(genTimes[2:end])
        cache = BBCache{Int,Vector{Float64}}(cacheSize)
    end
    logpathValues = Vector{Float64}(undef, nSim)
    z  = Vector{Float64}(undef, nSim)
    pathValues = Vector{Float64}(undef, nSim)
    local payoffValues

    t0 = genTimes[1]
    lnspot = log(model.spot)
    logpathValues .= lnspot

    for (dim, t1) in enumerate(genTimes[2:end])
        h = t1 - t0
        sqrth = sqrt(h)
        if withBB
            transformByDim!(bb, rng, start, dim, z, cache)
        else
            skipTo(rng, dim, start)
            nextn!(rng, dim, z)
            @. z *= sqrth
        end
        volSq = (varianceByLogmoneyness(model.surface, 0.0, t1)*t1 - varianceByLogmoneyness(model.surface, 0.0, t0)*t0)/h
        vol = sqrt(volSq/h)
        @. logpathValues += z * vol + (model.r - model.q - 0.5 * volSq) * h
        if t1 == tte
            df = exp(-model.r * t1)
            @. pathValues = exp(logpathValues)
            payoffValues = map(x -> evaluatePayoff(payoff, x, df), pathValues)
        end
        t0 = t1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end
