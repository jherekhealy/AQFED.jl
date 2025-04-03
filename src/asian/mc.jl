using Statistics,LinearAlgebra
using AQFED.Random
using AQFED.MonteCarlo
using AQFED.Basket
export MonteCarloByPathEngine

function priceAsianFixedStrike(
    p::MonteCarloEngine,
    isCall::Bool,
    strike::AbstractFloat,
    discountFactor::AbstractFloat,
    spot::AbstractFloat,
    forward::AbstractArray{TV}, #forward to each Asian observation t_i
    totalVariance::AbstractArray{TV}, #vol^2 * t_i
    weight::AbstractArray{TV}; start=1,
) where {TV<:Number}
return priceAsianFixedStrike(p,isCall,[strike],discountFactor,spot,forward,totalVariance,weight)
end


function priceAsianBasketFixedStrike(
    p::Basket.MonteCarloEngine,
    isCall::Bool,
    strikes::AbstractVector{TV},
    discountFactor::AbstractFloat,
    spot::AbstractVector{TV},
    forward::AbstractMatrix{TV}, #forward for each asset F_i to each Asian observation t_j
    totalVariance::AbstractMatrix{TV}, #vol^2 * t_i
    weight::AbstractMatrix{TV},
    correlation::AbstractMatrix{TV}, #S_k S_j
    obsTimes::AbstractVector{TV}
) where {TV<:Number}
nSim = p.nSim
nAsset = length(spot)
nDim = nAsset*length(obsTimes)
rng = DigitalSobolSeq(nDim, nSim, Chacha8SIMD(UInt32))
local bb, cache
genTimes = vcat(0.0, obsTimes)
sign = if isCall
    1
else
    -1
end
logpathValues = Array{Float64}(undef, (nSim,nAsset))
u = Array{Float64}(undef, (nSim,nAsset))
if p.withBB
    cacheSize = ceil(Int, log2(length(genTimes))) * 4
    bb = BrownianBridgeConstruction(genTimes[2:end])
    cache = BBCache{Int,typeof(u)}(cacheSize)
end
currentAverage = zeros(Float64, nSim)
local payoffValues

t0 = genTimes[1]
lnf0 = @. log(spot)
lnf1 = copy(lnf0)
for iAsset = 1:nAsset
    logpathValues[:,iAsset] .= lnf0[iAsset]
end

forwardVariance = zeros(nAsset,length(genTimes)-1)
   for (dim, t1) in enumerate(genTimes[2:end])
    for i=1:nAsset
        forwardVariance[i,dim] = totalVariance[i,dim]
        if dim > 1 
            forwardVariance[i,dim] -= totalVariance[i,dim-1]
        end
        forwardVariance[i,dim]/=(genTimes[dim+1]-genTimes[dim])
    end
end

for (dim, t1) in enumerate(genTimes[2:end])
    h = t1 - t0
    C = diagm(sqrt.(forwardVariance[:,dim])) * correlation * diagm(sqrt.(forwardVariance[:,dim]))
    A = sqrt(C)
    @. lnf1 = log(forward[:,dim])
    sqrth = sqrt(h)
    if p.withBB
        transformByDim!(bb, rng, 1, dim, u, cache)
    else
        for iAsset = 1:nAsset
            d = nAsset*(dim-1) + iAsset
        # skipTo(rng, d+i-1, start)
        nextn!(rng, d, @view(u[:,iAsset]))
        @view(u[:,i]) .*= sqrth
        end
    end
    for iAsset = 1:nAsset
        @inbounds for p=1:nSim
         logpathValues[p,iAsset] += dot(@view(u[p,:]), @view(A[:,iAsset])) - 0.5 * forwardVariance[iAsset,dim]* h + lnf1[iAsset] - lnf0[iAsset]
        end
    #@. pathValues = exp(logpathValues)
    @. currentAverage += exp(@view(logpathValues[:,iAsset])) * weight[iAsset,dim]
    end
    t0 = t1
    lnf0 .= lnf1
end
payoffMeans = map(strike ->  mean(@.(max(sign * (currentAverage - strike), 0) * discountFactor)),strikes)
return payoffMeans #, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end

function priceAsianFixedStrike(
    p::MonteCarloEngine,
    isCall::Bool,
    strikes::AbstractArray{TV},
    discountFactor::AbstractFloat,
    spot::AbstractFloat,
    forward::AbstractArray{TV}, #forward to each Asian observation t_i
    totalVariance::AbstractArray{TV}, #vol^2 * t_i
    weight::AbstractArray{TV}; start=1,
) where {TV<:Number}
    nSim = p.nSim
    #rng = ScrambledSobolSeq(length(totalVariance), 1024 * 1024 * 64, Owen(30,ScramblingRngAdapter( Chacha8SIMD(UInt32))))
    if totalVariance[end] < totalVariance[1]
        totalVariance = reverse(totalVariance)
        weight = reverse(weight)
        forward = reverse(forward)
    end
    rng = DigitalSobolSeq(length(totalVariance), nSim, Chacha8SIMD(UInt32))
    local bb, cache
    genTimes = vcat(0.0, totalVariance)
    sign = if isCall
        1
    else
        -1
    end
    if p.withBB
        cacheSize = ceil(Int, log2(length(genTimes))) * 4
        bb = BrownianBridgeConstruction(genTimes[2:end])
        cache = BBCache{Int,Vector{Float64}}(cacheSize)
    end
    logpathValues = Vector{Float64}(undef, nSim)
    z = zeros(Float64, nSim)
    #pathValues = zeros(Float64, nSim)
    currentAverage = zeros(Float64, nSim)
    local payoffValues

    t0 = genTimes[1]
    lnf0 = log(spot)
    logpathValues .= lnf0

    for (dim, t1) in enumerate(genTimes[2:end])
        h = t1 - t0
        lnf1 = log(forward[dim])
        sqrth = sqrt(h)
        if p.withBB
            transformByDim!(bb, rng, start, dim, z, cache)
        else
            skipTo(rng, dim, start)
            nextn!(rng, dim, z)
            @. z *= sqrth
        end
        @. logpathValues += z - 0.5 * h + lnf1 - lnf0
        #@. pathValues = exp(logpathValues)
        @. currentAverage += exp(logpathValues) * weight[dim]
        t0 = t1
        lnf0 = lnf1
    end
    payoffMeans = map(strike ->  mean(@.(max(sign * (currentAverage - strike), 0) * discountFactor)),strikes)
    return payoffMeans #, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))

end


struct MonteCarloByPathEngine
    withBB::Bool
    nSim::Int
end
MonteCarloByPathEngine() = MonteCarloByPathEngine(true, 1024 * 64)

function priceAsianFixedStrike(
    p::MonteCarloByPathEngine,
    isCall::Bool,
    strike::AbstractFloat,
    discountFactor::AbstractFloat,
    spot::AbstractFloat,
    forward::AbstractArray{TV}, #forward to each Asian observation t_i
    totalVariance::AbstractArray{TV}, #vol^2 * t_i
    weight::AbstractArray{TV}; start=1,
) where {TV<:Number}
    nSim = p.nSim
    #rng = ScrambledSobolSeq(length(totalVariance), 1024 * 1024 * 64, Owen(30,ScramblingRngAdapter( Chacha8SIMD(UInt32))))
    if totalVariance[end] < totalVariance[1]
        totalVariance = reverse(totalVariance)
        weight = reverse(weight)
        forward = reverse(forward)
    end
    genTimes = vcat(0.0, totalVariance)
    sign = if isCall
        1
    else
        -1
    end

    bb = BrownianBridgeConstruction(genTimes[2:end])
    lnf0 = log(spot)
    lnf = log.(forward)
     chunk = 1:nSim
    # chunks = Iterators.partition(1:nSim, nSim ÷ Threads.nthreads() ) #spawn is slow
    # tasks = map(chunks) do chunk
    #     Threads.@spawn begin
             local rng = ScrambledSobolSeq(length(totalVariance), nSim*2, Owen(30,ScramblingRngAdapter( Chacha8SIMD(UInt64))))#rng = DigitalSobolSeq(length(totalVariance), 1024 * 1024 * 64, Chacha8SIMD(UInt32))
            #local rng = ZRNGSeq(Blabla(),length(totalVariance))
            z = zeros(Float64, length(genTimes) - 1)
            u = zeros(Float64, length(genTimes) - 1)
            skipTo(rng, chunk[1])
            payoffValues = zeros(Float64, length(chunk))
            for sim = 1:length(chunk)
                if p.withBB
                    nextn!(rng, z)
                    transform!(bb, z, u)
                else
                    nextn!(rng, z)
                    t0 = genTimes[1]
                    @inbounds for i = 1:length(genTimes)-1
                        dim = i
                        t1 = genTimes[i+1]
                        u[i] = z[dim] * sqrt(t1 - t0)
                        t0 = t1
                    end
                end
                logpathValue = lnf0
                currentAverage = 0.0
                t0 = genTimes[1]
                @inbounds for i = 1:length(genTimes)-1
                    t1 = genTimes[i+1]
                    lnf1 = lnf[i]
                    u1 = u[i]
                    logpathValue += u1 - 0.5 * (t1 - t0) + lnf1 - lnf0
                    pathValue = exp(logpathValue)
                    currentAverage += pathValue * weight[i]
                    t0 = t1
                    lnf0 = lnf1
                end
                # println(chunk, " ",sim+chunk[1]-1, " ",z[1])
                payoffValues[sim] = max(sign * (currentAverage - strike), 0) * discountFactor
            end
            return sum(payoffValues)/nSim
    #     end
    # end
    # chunk_means = fetch.(tasks)
    # return sum(chunk_means)/nSim
end


function priceAsianSpread(
    p::MonteCarloEngine,
    isCall::Bool,
    strikeShifts::AbstractArray{TV},
    discountFactor::AbstractFloat,
    spot::AbstractFloat,
    forward::AbstractArray{TV}, #forward to each Asian observation t_i
    totalVariance::AbstractArray{TV}, #vol^2 * t_i
    weight::AbstractArray{TV},
    indexEndAverage::Int; start=1,
) where {TV<:Number} 
    nSim = p.nSim
    #rng = ScrambledSobolSeq(length(totalVariance), 1024 * 1024 * 64, Owen(30,ScramblingRngAdapter( Chacha8SIMD(UInt32))))
    if totalVariance[end] < totalVariance[1]
        totalVariance = reverse(totalVariance)
        weight = reverse(weight)
        forward = reverse(forward)
    end
    rng = DigitalSobolSeq(length(totalVariance), nSim, Chacha8SIMD(UInt32))
    local bb, cache
    genTimes = vcat(0.0, totalVariance)
    sign = if isCall
        1
    else
        -1
    end
    if p.withBB
        cacheSize = ceil(Int, log2(length(genTimes))) * 4
        bb = BrownianBridgeConstruction(genTimes[2:end])
        cache = BBCache{Int,Vector{Float64}}(cacheSize)
    end
    logpathValues = Vector{Float64}(undef, nSim)
    z = zeros(Float64, nSim)
    #pathValues = zeros(Float64, nSim)
    spotAverage = zeros(Float64, nSim)
    strikeAverage = zeros(Float64, nSim)
    local payoffValues

    t0 = genTimes[1]
    lnf0 = log(spot)
    logpathValues .= lnf0

    for (dim, t1) in enumerate(genTimes[2:end])
        h = t1 - t0
        lnf1 = log(forward[dim])
        sqrth = sqrt(h)
        if p.withBB
            transformByDim!(bb, rng, start, dim, z, cache)
        else
            skipTo(rng, dim, start)
            nextn!(rng, dim, z)
            @. z *= sqrth
        end
        @. logpathValues += z - 0.5 * h + lnf1 - lnf0
        #@. pathValues = exp(logpathValues)
        if dim <= indexEndAverage
            @. strikeAverage += exp(logpathValues) * weight[dim]
        else
            @. spotAverage += exp(logpathValues) * weight[dim]
        end
        t0 = t1
        lnf0 = lnf1
    end
    payoffMeans = map(strikeShift ->  mean(@.(max(sign * (spotAverage -strikeShift - strikeAverage), 0) * discountFactor)),strikeShifts)
    return payoffMeans #, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))

end
