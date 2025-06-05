import AQFED.Math: norminv

import AQFED.TermStructure: HestonModel, DoubleHestonModel, discountFactor, forward, PiecewiseConstantFunction
import AQFED.Random: next!, nextn!, skipTo
import CharFuncPricing: HestonTSParams
struct HestonPathGeneratorParameters{T}
    model::HestonModel{T}
    spot::T
    timestepSize::Float64
    withBB::Bool
    cacheSize::Int
end

struct DoubleHestonPathGeneratorParameters{T}
    model::DoubleHestonModel{T}
    spot::T
    timestepSize::Float64
    withBB::Bool
    cacheSize::Int
end
#mutable struct HestonPathGenerator{T}
#    u1 = Vector{Float64}(undef, nSim)
#    u2 = Vector{Float64}(undef, nSim)
#    xsi = u1
#    v = Vector{T}(undef, nSim)
#    t0 = genTimes[1]
#    lnspot = log(spot)
#    lnf0 = logForward(model, lnspot, t0)
#    logpathValues .= lnf0
#    v .= model.v0
#    ρBar = sqrt(1 - model.ρ^2)
#end
function ndims(model::HestonModel, specificTimes::Vector{Float64}, timestepSize::Float64)
    genTimes = pathgenTimes(model, specificTimes, timestepSize)
    return (length(genTimes) - 1) * 2 #0.0 does not count
end

function ndims(model::DoubleHestonModel, specificTimes::Vector{Float64}, timestepSize::Float64)
    genTimes = pathgenTimes(model, specificTimes, timestepSize)
    return (length(genTimes) - 1) * 4 #0.0 does not count
end

function ndims(model::HestonTSParams, specificTimes::Vector{Float64}, timestepSize::Float64)
    iLast = searchsortedfirst(model.startTime,specificTimes[end])
    specTimesAll = sort(unique(vcat(specificTimes, model.startTime[2:iLast-1])))
    genTimes = pathgenTimes(model, specTimesAll, timestepSize)
    return (length(genTimes) - 1) * 2 #0.0 does not count
end

#DVSS2X discretization scheme for Heston
function simulateDVSS2X(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = discountFactor(model, tte)
    genTimes = pathgenTimes(model, specTimes, timestepSize)
    logpathValues = Vector{T}(undef, nSim)
    local bb, cache
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
        if cacheSize == 0
            nSteps = length(genTimes) - 1
            cacheSize = ceil(Int, log2(nSteps)) * 4
        end
        # cache = BBCache{Int,Array{Float64,2}}(cacheSize)
        cache = BBCache{Int,Vector{Float64}}(cacheSize)
    end

    u1 = Vector{Float64}(undef, nSim)
    u2 = Vector{Float64}(undef, nSim)
    xsi = u1
    v = Vector{T}(undef, nSim)
    t0 = genTimes[1]
    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, t0)
    logpathValues .= lnf0
    v .= model.v0
    ρBar = sqrt(1 - model.ρ^2)
    ndimsh = length(genTimes) - 1
    local payoffValues
    for (dim, t1) in enumerate(genTimes[2:end])
        h = t1 - t0
        ekdth = exp(-model.κ * h / 2)
        c0 = model.θ * h / 4
        c1 = (ekdth - 1) / (model.κ * 2)
        c2 = model.θ * (1 - ekdth)
        if withBB
            d = dim
            transformByDim!(bb, rng, start, d, xsi, cache)
            skipTo(rng, d + ndimsh, start)
            next!(rng, d + ndimsh, u2)
        else
            d = 2 * dim - 1
            skipTo(rng, d, start)
            nextn!(rng, d, xsi)
            xsi .*= sqrt(h)
            skipTo(rng, d + 1, start)
            next!(rng, d + 1, u2)
        end

        vTmp = @. (v * ekdth + c2) / model.σ^2
        yr = map(
            (vTmp, u2) -> ifelse(
                vTmp > 1.896 * h,  #limit for y1 > 0 => 1.89564392373896
                dvss2_case1(vTmp, u2, h),
                dvss2_case2(vTmp, u2, h),
            ),
            vTmp,
            u2,
        )
        xb = vTmp # reuse var
        @. xb =
            logpathValues - c0 +
            c1 * (v - model.θ) +
            model.σ * (ρBar * xsi * sqrt((vTmp + yr) / 2) + model.ρ * (yr - vTmp))
        yr .= model.σ^2 * yr
        lnf1 = logForward(model, lnspot, t1)
        @. logpathValues = lnf1 - lnf0 + xb - c0 + c1 * (yr - model.θ)
        @. v = yr * ekdth + c2
        # advancePayoff(payoff, t1, )
        if t1 == tte
            pathValues = xb #reuse Var
            @. pathValues = exp(logpathValues)
            payoffValues = map(x -> evaluatePayoffOnPath(payoff, x, df), pathValues)
        end
        t0 = t1
        lnf0 = lnf1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end

function simulateDVSS2X(
    rng,
    model::DoubleHestonModel{T},
    spot::Float64,
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = discountFactor(model, tte)
	genTimes = pathgenTimes(model, specTimes, timestepSize)
    logpathValues = Vector{T}(undef, nSim)
    local bb, cache
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
        if cacheSize == 0
            nSteps = length(genTimes) - 1
            cacheSize = ceil(Int, log2(nSteps)) * 4
        end
        # cache = BBCache{Int,Array{Float64,2}}(cacheSize)
        cache = BBCache{Int,Vector{Float64}}(cacheSize)
    end

    u1 = Vector{Float64}(undef, nSim)
    u2 = Vector{Float64}(undef, nSim)
    u3 = Vector{Float64}(undef, nSim)
    u4 = Vector{Float64}(undef, nSim)
    xsi1 = u1
    xsi2 = u2
    v1 = Vector{T}(undef, nSim)
    v2 = Vector{T}(undef, nSim)
    t0 = genTimes[1]
    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, t0)
	logpathValues .= lnf0
    v1 .= model.params.heston1.v0
    v2 .= model.params.heston2.v0
    ρ1Bar = sqrt(1 - model.params.heston1.ρ^2)
    ρ2Bar = sqrt(1 - model.params.heston2.ρ^2)
    ndimsh = length(genTimes) - 1
    local payoffValues
    for (dim, t1) in enumerate(genTimes[2:end])
        h = t1 - t0
        ek1dth = exp(-model.params.heston1.κ * h / 2)
        ek2dth = exp(-model.params.heston2.κ * h / 2)
        c10 = model.params.heston1.θ * h / 4
        c11 = (ek1dth - 1) / (model.params.heston1.κ * 2)
        c12 = model.params.heston1.θ * (1 - ek1dth)
        c20 = model.params.heston2.θ * h / 4
        c21 = (ek2dth - 1) / (model.params.heston2.κ * 2)
        c22 = model.params.heston2.θ * (1 - ek2dth)

        if withBB
            d = dim
            transformByDim!(bb, rng, start, d, xsi1, cache)
            skipTo(rng, d + ndimsh, start)
            transformByDim!(bb, rng, start, d + ndimsh, xsi2, cache)
            skipTo(rng, d + 2ndimsh, start)
            next!(rng, d + 2ndimsh, u3)
            skipTo(rng, d + 3ndimsh, start)
            next!(rng, d + 3ndimsh, u4)
        else
            d = 4 * dim - 1
            skipTo(rng, d, start)
            nextn!(rng, d, xsi1)
            skipTo(rng, d + 1, start)
            nextn!(rng, d + 1, xsi2)
            xsi1 .*= sqrt(h)
            xsi2 .*= sqrt(h)
            skipTo(rng, d + 2, start)
            next!(rng, d + 2, u3)
            skipTo(rng, d + 3, start)
            next!(rng, d + 3, u4)
        end

        v1Tmp = @. (v1 * ek1dth + c12) / model.params.heston1.σ^2
        yr1 = map(
            (vTmp, u2) -> ifelse(
                vTmp > 1.896 * h,  #limit for y1 > 0 => 1.89564392373896
                dvss2_case1(vTmp, u2, h),
                dvss2_case2(vTmp, u2, h),
            ),
            v1Tmp,
            u3,
        )
        v2Tmp = @. (v2 * ek2dth + c22) / model.params.heston2.σ^2
        yr2 = map(
            (vTmp, u2) -> ifelse(
                vTmp > 1.896 * h,  #limit for y1 > 0 => 1.89564392373896
                dvss2_case1(vTmp, u2, h),
                dvss2_case2(vTmp, u2, h),
            ),
            v2Tmp,
            u4,
        )

        xb = v1Tmp # reuse var
        @. xb =
            logpathValues - c10 - c20 +
            c11 * (v1 - model.params.heston1.θ) + c21 * (v2 - model.params.heston2.θ) +
            model.params.heston1.σ * (ρ1Bar * xsi1 * sqrt((v1Tmp + yr1) / 2) + model.params.heston1.ρ * (yr1 - v1Tmp)) +
            model.params.heston2.σ * (ρ2Bar * xsi2 * sqrt((v2Tmp + yr2) / 2) + model.params.heston2.ρ * (yr2 - v2Tmp))

        yr1 .= model.params.heston1.σ^2 * yr1
        yr2 .= model.params.heston2.σ^2 * yr2
        lnf1 = logForward(model, lnspot, t1)
        @. (logpathValues = lnf1 - lnf0 + xb - c10 - c20 + c11 * (yr1 - model.params.heston1.θ) + c21 * (yr2 - model.params.heston2.θ))
        @. v1 = yr1 * ek1dth + c12
        @. v2 = yr2 * ek2dth + c22
        # advancePayoff(payoff, t1, )
        if t1 == tte
            pathValues = xb #reuse Var
            @. pathValues = exp(logpathValues)
            payoffValues = map(x -> evaluatePayoffOnPath(payoff, x, df), pathValues)
        end
        t0 = t1
        lnf0 = lnf1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end

@inline function dvss2_case1(vTmp, u2, h)
    delta = (21.0 / 4 * h + 12 * vTmp) * h
    sqrtDelta = sqrt(delta)
    s = 1.5 * h
    twovh = 2 * vTmp * h / sqrtDelta
    p1 = twovh / (sqrtDelta - s)
    p2 = twovh / (sqrtDelta + s)
    y0 = vTmp
    y1 = vTmp + (s - sqrtDelta) / 2
    y2 = y1 + sqrtDelta
    if (u2 < p1)
        return y1
    elseif u2 > 1 - p2
        return y2
    else
        return y0
    end
end

@inline function dvss2_case2(vTmp, u2, h)
    denom = (2 * vTmp + h)
    s = denom + (5 * vTmp + 2 * h) * h / denom
    delta = h * (((16 * vTmp + 33 * h) * vTmp + 18 * h^2) * vTmp + 3 * h^3)
    sqrtDelta = sqrt(delta) / denom
    p1 = vTmp * (2 * (vTmp + h) - s - sqrtDelta) / (sqrtDelta * (sqrtDelta - s))
    p2 = vTmp * (2 * (vTmp + h) - s + sqrtDelta) / (sqrtDelta * (sqrtDelta + s))
    y0 = 0.0
    y1 = (s - sqrtDelta) * 0.5
    y2 = y1 + sqrtDelta
    if (u2 < p1)
        return y1
    elseif u2 > 1 - p2
        return y2
    else
        return y0
    end
end



#Euler Full Truncation discretization scheme for Heston
function simulateFullTruncation(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = discountFactor(model, tte)
    genTimes = pathgenTimes(model, specTimes, timestepSize)
    logpathValues = Vector{T}(undef, nSim)
    local bb, cache
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
        if cacheSize == 0
            nSteps = length(genTimes) - 1
            cacheSize = ceil(Int, log2(nSteps)) * 4
        end
        cache = BBCache{Int,Array{Float64,2}}(cacheSize)
        # cache = BBCache{Int,Vector{Float64}}(cacheSize)
    end

    u = Array{Float64}(undef, (nSim, 2))
    u1 = @view u[:, 1]
    u2 = @view u[:, 2]
    v = Vector{T}(undef, nSim)
    sqrtmv = Vector{T}(undef, nSim)
    t0 = genTimes[1]
    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, t0)
    logpathValues .= lnf0
    v .= model.v0
    ρBar = sqrt(1 - model.ρ^2)
    ndimsh = length(genTimes) - 1
    local payoffValues
    for (dim, t1) in enumerate(genTimes[2:end])
        h = t1 - t0
        sqrth = sqrt(h)
        if withBB
            transformByDim!(bb, rng, start, dim, u, cache)
        else
            d = 2 * dim - 1
            skipTo(rng, d, start)
            nextn!(rng, d, u1)
            skipTo(rng, d + 1, start)
            nextn!(rng, d + 1, u2)
            #@inbounds @. u *= sqrth
            u1 .*= sqrth
            u2 .*= sqrth
        end
        lnf1 = logForward(model, lnspot, t1)
        @. sqrtmv = sqrt(max(v, 0))
        @. logpathValues +=
            lnf1 - lnf0 - (0.5 * sqrtmv^2) * h + sqrtmv * (u1 * ρBar + u2 * model.ρ)
        # for (i,p) in enumerate(logpathValues)
        #     if isnan(p)
        #         println(i," nan ",sqrtmv[i], " ",u1[i]," ", u2[i])
        #     end
        # end
        @. v += model.κ * (model.θ - sqrtmv^2) * h + model.σ * sqrtmv * u2

        if t1 == tte
            pathValues = sqrtmv #reuse Var
            @. pathValues = exp(logpathValues)
            payoffValues = map(x -> evaluatePayoffOnPath(payoff, x, df), pathValues)
        end
        t0 = t1
        lnf0 = lnf1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end


function simulateFullTruncation(
    rng,
    model::HestonTSParams,
    spot::T,
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
    leverageFunction= t -> 1
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    iLast = searchsortedfirst(model.startTime,tte)
    specTimesAll = sort(unique(vcat(specTimes, model.startTime[2:iLast-1])))
    genTimes = pathgenTimes(model, specTimesAll, timestepSize)
    logpathValues = Vector{T}(undef, nSim)
    local bb, cache
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
        if cacheSize == 0
            nSteps = length(genTimes) - 1
            cacheSize = ceil(Int, log2(nSteps)) * 4
        end
        cache = BBCache{Int,Array{Float64,2}}(cacheSize)
        # cache = BBCache{Int,Vector{Float64}}(cacheSize)
    end

    u = Array{Float64}(undef, (nSim, 2))
    u1 = @view u[:, 1]
    u2 = @view u[:, 2]
    v = Vector{T}(undef, nSim)
    sqrtmv = Vector{T}(undef, nSim)
    t0 = genTimes[1]
    lnspot = log(spot)
    lnf0 = lnspot
    logpathValues .= lnf0
    v .= model.v0
    kappaFunc = PiecewiseConstantFunction(model.startTime, model.κ)
    thetaFunc = PiecewiseConstantFunction(model.startTime, model.θ)
    rhoFunc = PiecewiseConstantFunction(model.startTime, model.ρ)
    sigmaFunc = PiecewiseConstantFunction(model.startTime, model.σ)
    ndimsh = length(genTimes) - 1
  #  println("genTimes ",genTimes)
    local payoffValues
    for (dim, t1) in enumerate(genTimes[2:end])
        h = t1 - t0
        κ = kappaFunc(t0+h/2)
        θ = thetaFunc(t0+h/2)
        ρ = rhoFunc(t0+h/2)
        σ = sigmaFunc(t0+h/2)
        lev = leverageFunction(t0+h/2)
        ρBar = sqrt(1 - ρ^2)
        sqrth = sqrt(h)
        if withBB
            transformByDim!(bb, rng, start, dim, u, cache)
        else
            d = 2 * dim - 1
            skipTo(rng, d, start)
            nextn!(rng, d, u1)
            skipTo(rng, d + 1, start)
            nextn!(rng, d + 1, u2)
            #@inbounds @. u *= sqrth
            u1 .*= sqrth
            u2 .*= sqrth
        end
        lnf1 = lnspot #logForward(model, lnspot, t1)
        @. sqrtmv = sqrt(max(v, 0))
        @. logpathValues +=
            lnf1 - lnf0 - (0.5 * (lev*sqrtmv)^2) * h + lev*sqrtmv * (u1 * ρBar + u2 * ρ)
        # for (i,p) in enumerate(logpathValues)
        #     if isnan(p)
        #         println(i," nan ",sqrtmv[i], " ",u1[i]," ", u2[i])
        #     end
        # end
        @. v += κ * (θ - sqrtmv^2) * h + σ * sqrtmv * u2

        if t1 == tte
            pathValues = sqrtmv #reuse Var
            @. pathValues = exp(logpathValues)
            payoffValues = map(x -> evaluatePayoffOnPath(payoff, x, 1.0), pathValues)
        end
        t0 = t1
        lnf0 = lnf1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end




function simulateFullTruncationHagan(
    rng,
    model::HestonTSParams,
    spot::T,
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    specTimesAll = sort(vcat(specTimes, model.startTime[2:end]))
    genTimes = pathgenTimes(model, specTimesAll, timestepSize)
    logpathValues = Vector{T}(undef, nSim)
    local bb, cache
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
        if cacheSize == 0
            nSteps = length(genTimes) - 1
            cacheSize = ceil(Int, log2(nSteps)) * 4
        end
        cache = BBCache{Int,Array{Float64,2}}(cacheSize)
        # cache = BBCache{Int,Vector{Float64}}(cacheSize)
    end

    u = Array{Float64}(undef, (nSim, 2))
    u1 = @view u[:, 1]
    u2 = @view u[:, 2]
    v = Vector{T}(undef, nSim)
    sqrtmv = Vector{T}(undef, nSim)
    t0 = genTimes[1]
    lnspot = log(spot)
    lnf0 = lnspot
    logpathValues .= lnf0
    v .= 1.0
    kappaFunc = PiecewiseConstantFunction(model.startTime, model.κ)
    thetaFunc = PiecewiseConstantFunction(model.startTime, model.θ)
    rhoFunc = PiecewiseConstantFunction(model.startTime, model.ρ)
    sigmaFunc = PiecewiseConstantFunction(model.startTime, model.σ)
    ndimsh = length(genTimes) - 1
  #  println("genTimes ",genTimes)
    local payoffValues
    θ0 = model.θ[1]
    jump = 0.0
    for (dim, t1) in enumerate(genTimes[2:end])
        h = t1 - t0
        κ = kappaFunc(t0+h/2)
        θ = thetaFunc(t0+h/2)
        ρ = rhoFunc(t0+h/2)
        σ = sigmaFunc(t0+h/2)
        ρBar = sqrt(1 - ρ^2)
        sqrth = sqrt(h)
        if withBB
            transformByDim!(bb, rng, start, dim, u, cache)
        else
            d = 2 * dim - 1
            skipTo(rng, d, start)
            nextn!(rng, d, u1)
            skipTo(rng, d + 1, start)
            nextn!(rng, d + 1, u2)
            #@inbounds @. u *= sqrth
            u1 .*= sqrth
            u2 .*= sqrth
        end
        lnf1 = lnspot #logForward(model, lnspot, t1)
        @. sqrtmv = sqrt(max(v, 0))        
        @. logpathValues +=
        lnf1 - lnf0 - (0.5 * max(0,θ*sqrtmv^2)) * h + sqrt(max(0,sqrtmv^2*θ)) * (u1 * ρBar + u2 * ρ)
        # for (i,p) in enumerate(logpathValues)
        #     if isnan(p)
        #         println(i," nan ",sqrtmv[i], " ",u1[i]," ", u2[i])
        #     end
        # end
        @. v += κ * (1 - sqrtmv^2) * h + σ/sqrt(θ) * sqrtmv * u2 
        #@. v /= θ/θ0 

        if t1 == tte
            pathValues = sqrtmv #reuse Var
            @. pathValues = exp(logpathValues)
            payoffValues = map(x -> evaluatePayoffOnPath(payoff, x, 1.0), pathValues)
        end
        t0 = t1
        θ0 = θ
        lnf0 = lnf1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end


function simulateFullTruncation(
    rng,
    model::DoubleHestonModel{T},
    spot::Float64,
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = discountFactor(model, tte)
    genTimes = pathgenTimes(model, specTimes, timestepSize)
    logpathValues = Vector{T}(undef, nSim)
    local bb, cache
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
        if cacheSize == 0
            nSteps = length(genTimes) - 1
            cacheSize = ceil(Int, log2(nSteps)) * 4
        end
        cache = BBCache{Int,Array{Float64,2}}(cacheSize)
        # cache = BBCache{Int,Vector{Float64}}(cacheSize)
    end

    u = Array{Float64}(undef, (nSim, 4))
    u1 = @view u[:, 1]
    u2 = @view u[:, 2]
	u3 = @view u[:, 3]
	u4 = @view u[:, 4]
    v1 = Vector{T}(undef, nSim)
	v2 = Vector{T}(undef, nSim)
    sqrtmv1 = Vector{T}(undef, nSim)
	sqrtmv2 = Vector{T}(undef, nSim)
    t0 = genTimes[1]
    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, t0)
    logpathValues .= lnf0
    v1 .= model.params.heston1.v0
	v2 .= model.params.heston2.v0
    ρ1Bar = sqrt(1 - model.params.heston1.ρ^2)
	ρ2Bar = sqrt(1 - model.params.heston2.ρ^2)
    ndimsh = length(genTimes) - 1
    local payoffValues
    for (dim, t1) in enumerate(genTimes[2:end])
        h = t1 - t0
        sqrth = sqrt(h)
        if withBB
            transformByDim!(bb, rng, start, dim, u, cache)
        else
            d = 2 * dim - 1
            skipTo(rng, d, start)
            nextn!(rng, d, u1)
            skipTo(rng, d+1, start)
            nextn!(rng, d+1, u2)
            skipTo(rng, d + 2, start)
            nextn!(rng, d + 2, u3)
            skipTo(rng, d + 3, start)
            nextn!(rng, d + 3, u4)
            #@inbounds @. u *= sqrth
            u1 .*= sqrth
            u2 .*= sqrth
			u3 .*= sqrth
			u4 .*= sqrth
        end
        lnf1 = logForward(model, lnspot, t1)
        @. sqrtmv1 = sqrt(max(v1, 0))
		@. sqrtmv2 = sqrt(max(v2, 0))
        @. logpathValues +=
            lnf1 - lnf0 - (0.5 * (sqrtmv1^2+sqrtmv2^2)) * h + sqrtmv1 * (u1 * ρ1Bar + u3 * model.params.heston1.ρ) +  sqrtmv2 * (u2 * ρ2Bar + u4 * model.params.heston2.ρ)
        # for (i,p) in enumerate(logpathValues)
        #     if isnan(p)
        #         println(i," nan ",sqrtmv[i], " ",u1[i]," ", u2[i])
        #     end
        # end
        @. v1 += model.params.heston1.κ * (model.params.heston1.θ - sqrtmv1^2) * h + model.params.heston1.σ * sqrtmv1 * u3
		@. v2 += model.params.heston2.κ * (model.params.heston2.θ - sqrtmv2^2) * h + model.params.heston2.σ * sqrtmv2 * u4

        if t1 == tte
            pathValues = sqrtmv1 #reuse Var
            @. pathValues = exp(logpathValues)
            payoffValues = map(x -> evaluatePayoffOnPath(payoff, x, df), pathValues)
        end
        t0 = t1
        lnf0 = lnf1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end

function simulateFullTruncationIter(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = exp(-model.r * tte)
    genTimes = pathgenTimes(model, specTimes, timestepSize)
    local bb
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
    end
    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, tte)
    ρBar = sqrt(1 - model.ρ^2)
    z = Vector{Float64}(undef, (length(genTimes) - 1) * 2)
    u = Array{Float64}(undef, (2, length(genTimes) - 1))
    skipTo(rng, start)
    payoffMean = 0.0
    for sim ∈ 1:nSim
        if withBB
            nextn!(rng, z)
            transform!(bb, z, u)
        else
            nextn!(rng, z)
            t0 = genTimes[1]
            @inbounds for i ∈ 1:length(genTimes)-1
                dim = 2 * i - 1
                t1 = genTimes[i+1]
                u[1, i] = z[dim] * sqrt(t1 - t0)
                u[2, i] = z[dim+1] * sqrt(t1 - t0)
                t0 = t1
            end
        end
        t0 = genTimes[1]
        logpathValue = lnf0
        v = model.v0
        local payoffValue
        @inbounds for i ∈ 1:length(genTimes)-1
            t1 = genTimes[i+1]
            # dim = 2 * i - 1
            u1 = u[1, i]
            u2 = u[2, i]
            h = t1 - t0
            sqrtmv = sqrt(max(v, 0))
            lnf1 = logForward(model, lnspot, t1)
            logpathValue += lnf1 - lnf0 - 0.5 * sqrtmv^2 * h + sqrtmv * (u1 * ρBar + u2 * model.ρ)
            v += model.κ * (model.θ - sqrtmv^2) * h + model.σ * sqrtmv * u2

            if t1 == tte
                pathValue = exp(logpathValue)
                payoffValue = evaluatePayoff(payoff, pathValue, df)
            end
            t0 = t1
            lnf0 = lnf1
        end

        payoffMean += payoffValue
    end
    payoffMean /= nSim
    return payoffMean, NaN
end

struct VVIX{T}
    model::HestonModel{T}
end

function VIXSquare(model::HestonModel{T}, vt; dt=30.0 / 365) where {T}
    ektm1 = (exp(-model.κ * dt) - 1) / (model.κ * dt)
    return @. -ektm1 * vt + model.θ * (ektm1 + 1)
end

function logsqrtVix(model::HestonModel{T}, vt) where {T}
    return log.(sqrt.(VIXSquare(model, max.(vt, 0))))
end

function evaluatePayoff(payoff::VVIX{T}, t, x, v, df) where {T}
    return sqrt.(VIXSquare(payoff.model, max.(v, 0), dt=30 / 365))
    # return (VIXSquare(payoff, max.(v, 0),dt=22/252)
    # return -2/t * (logsqrtVix(payoff.model, v) .- logsqrtVix(payoff.model, payoff.model.v0))
end

function specificTimes(payoff::VVIX)
    return [30 / 365] #[5 / 252]
end

function simulateDVSS2X(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::VVIX,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = discountFactor(model, tte)
    genTimes = pathgenTimes(model, specTimes, timestepSize)
    logpathValues = Vector{T}(undef, nSim)
    local bb, cache
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
        if cacheSize == 0
            nSteps = length(genTimes) - 1
            cacheSize = ceil(Int, log2(nSteps)) * 4
        end
        # cache = BBCache{Int,Array{Float64,2}}(cacheSize)
        cache = BBCache{Int,Vector{Float64}}(cacheSize)
    end

    u1 = Vector{Float64}(undef, nSim)
    u2 = Vector{Float64}(undef, nSim)
    xsi = u1
    v = Vector{T}(undef, nSim)
    t0 = genTimes[1]
    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, t0)
    logpathValues .= lnf0
    v .= model.v0
    ρBar = sqrt(1 - model.ρ^2)
    ndimsh = length(genTimes) - 1
    local payoffValues
    for (dim, t1) in enumerate(genTimes[2:end])
        h = t1 - t0
        ekdth = exp(-model.κ * h / 2)
        c0 = model.θ * h / 4
        c1 = (ekdth - 1) / (model.κ * 2)
        c2 = model.θ * (1 - ekdth)
        if withBB
            d = dim
            transformByDim!(bb, rng, start, d, xsi, cache)
            skipTo(rng, d + ndimsh, start)
            next!(rng, d + ndimsh, u2)
        else
            d = 2 * dim - 1
            skipTo(rng, d, start)
            nextn!(rng, d, xsi)
            xsi .*= sqrt(h)
            skipTo(rng, d + 1, start)
            next!(rng, d + 1, u2)
        end

        vTmp = @. (v * ekdth + c2) / model.σ^2
        yr = map(
            (vTmp, u2) -> ifelse(
                vTmp > 1.896 * h,  #limit for y1 > 0 => 1.89564392373896
                dvss2_case1(vTmp, u2, h),
                dvss2_case2(vTmp, u2, h),
            ),
            vTmp,
            u2,
        )
        xb = vTmp # reuse var
        @. xb =
            logpathValues - c0 +
            c1 * (v - model.θ) +
            model.σ * (ρBar * xsi * sqrt((vTmp + yr) / 2) + model.ρ * (yr - vTmp))
        yr .= model.σ^2 * yr
        lnf1 = logForward(model, lnspot, t1)
        @. logpathValues = lnf1 - lnf0 + xb - c0 + c1 * (yr - model.θ)
        @. v = yr * ekdth + c2
        # advancePayoff(payoff, t1, )
        if t1 == tte
            pathValues = xb #reuse Var
            @. pathValues = exp(logpathValues)
            payoffValues = evaluatePayoff(payoff, t1, pathValues, v, df)
        end
        t0 = t1
        lnf0 = lnf1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end

function simulateFullTruncation(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::VVIX,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = discountFactor(model, tte)
    genTimes = pathgenTimes(model, specTimes, timestepSize)
    logpathValues = Vector{T}(undef, nSim)
    local bb, cache
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
        if cacheSize == 0
            nSteps = length(genTimes) - 1
            cacheSize = ceil(Int, log2(nSteps)) * 4
        end
        cache = BBCache{Int,Array{Float64,2}}(cacheSize)
        # cache = BBCache{Int,Vector{Float64}}(cacheSize)
    end

    u = Array{Float64}(undef, (nSim, 2))
    u1 = @view u[:, 1]
    u2 = @view u[:, 2]
    v = Vector{T}(undef, nSim)
    sqrtmv = Vector{T}(undef, nSim)
    t0 = genTimes[1]
    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, t0)
    logpathValues .= lnf0
    v .= model.v0
    ρBar = sqrt(1 - model.ρ^2)
    ndimsh = length(genTimes) - 1
    local payoffValues
    for (dim, t1) in enumerate(genTimes[2:end])
        h = t1 - t0
        sqrth = sqrt(h)
        if withBB
            transformByDim!(bb, rng, start, dim, u, cache)
        else
            d = 2 * dim - 1
            skipTo(rng, d, start)
            nextn!(rng, d, u1)
            skipTo(rng, d + 1, start)
            nextn!(rng, d + 1, u2)
            #@inbounds @. u *= sqrth
            u1 .*= sqrth
            u2 .*= sqrth
        end
        lnf1 = logForward(model, lnspot, t1)
        @. sqrtmv = sqrt(max(v, 0))
        @. logpathValues +=
            lnf1 - lnf0 - (0.5 * sqrtmv^2) * h + sqrtmv * (u1 * ρBar + u2 * model.ρ)
        # for (i,p) in enumerate(logpathValues)
        #     if isnan(p)
        #         println(i," nan ",sqrtmv[i], " ",u1[i]," ", u2[i])
        #     end
        # end
        @. v += model.κ * (model.θ - sqrtmv^2) * h + model.σ * sqrtmv * u2

        if t1 == tte
            pathValues = sqrtmv #reuse Var
            @. pathValues = exp(logpathValues)
            payoffValues = evaluatePayoff(payoff, t1, pathValues, v, df)
        end
        t0 = t1
        lnf0 = lnf1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end
