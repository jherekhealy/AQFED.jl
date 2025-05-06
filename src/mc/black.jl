import AQFED.Math: norminv
using Statistics
import AQFED.TermStructure: ConstantBlackModel, TSBlackModel, varianceByLogmoneyness, discountFactor, forward, logForward
using LinearAlgebra

function simulate(rng, model::ConstantBlackModel, spot::Float64, payoff::VanillaOption, nSim::Int)
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    sqrtte = sqrt(tte)
    df = discountFactor(model, tte)
    z = Vector{Float64}(undef, nSim)
    nextn!(rng, z)
    pathValues = map(z -> forward(model, spot, tte)*exp(model.vol * z * sqrtte - 0.5 * model.vol^2 * tte),z)        
    mean(x -> evaluatePayoffOnPath(payoff, x, df), pathValues)
end


function simulate(rng, model::AbstractArray{ConstantBlackModel}, spot::AbstractArray{T}, correlation::AbstractMatrix{T}, payoff::Union{VanillaBasketOption}, nSim::Int) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = discountFactor(model[1], tte)
    nd = length(spot)
    varianceSqrtV = [sqrt(m.vol^2 * tte) for m = model]
    varianceSqrtM = diagm(varianceSqrtV)    
    covar = Symmetric(varianceSqrtM * correlation * varianceSqrtM)
    covarSqrt = sqrt(covar)
    f = [forward(m, s, tte) for (m,s) = zip(model,spot)]
    z = zeros(nSim,nd)
    varianceMM =  ones(nSim,nd) * diagm(varianceSqrtV.^2)
    logfMM =  ones(nSim,nd) * diagm(log.(f))
    sqrtdt = sqrt(tte)
    for i = 1:size(z,1)
        nextn!(rng, @view(z[i,:]))
    end
    pathValues = exp.(logfMM + z*covarSqrt  - 0.5 * varianceMM)
    payoffValue = mapslices(x -> evaluatePayoffOnPath(payoff, x, df), pathValues,dims=2)
    mean(payoffValue)
end


function simulate(rng, model::AbstractArray{ConstantBlackModel}, spot::AbstractArray{T}, correlation::AbstractMatrix{T}, payoff::ListPayoff, nSim::Int) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = discountFactor(model[1], tte)
    nd = length(spot)
    varianceSqrtV = [sqrt(m.vol^2 * tte) for m = model]
    varianceSqrtM = diagm(varianceSqrtV)    
    covar = Symmetric(varianceSqrtM * correlation * varianceSqrtM)
    covarSqrt = sqrt(covar)
    f = [forward(m, s, tte) for (m,s) = zip(model,spot)]
    z = zeros(nSim,nd)
    varianceMM =  ones(nSim,nd) * diagm(varianceSqrtV.^2)
    logfMM =  ones(nSim,nd) * diagm(log.(f))
    sqrtdt = sqrt(tte)
    for i = 1:size(z,1)
        nextn!(rng, @view(z[i,:]))
    end
    pathValues = exp.(logfMM + z*covarSqrt  - 0.5 * varianceMM)
    [mean( mapslices(x -> evaluatePayoffOnPath(p, x, df), pathValues,dims=2)) for p = payoff.list]
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
    spot::Float64,
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
    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, t0)
    logpathValues .= lnf0
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
        vol = sqrt(volSq)
        lnf1 = logForward(model, lnspot, t1)
        @. logpathValues += z * vol - 0.5 * volSq * h + lnf1 - lnf0        
        if t1 == tte
            df = discountFactor(model, t1)
            @. pathValues = exp(logpathValues)
            payoffValues = map(x -> evaluatePayoff(payoff, x, df), pathValues)
        end
        t0 = t1
        lnf0 = lnf1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end
