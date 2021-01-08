import AQFED.Math: norminv
using Statistics
import AQFED.TermStructure: HestonModel
import AQFED.Random: next!, nextn!, skipTo

function ndims(model::HestonModel, specificTimes::Vector{Float64}, timestepSize::Float64)
    genTimes = pathgenTimes(model, specificTimes, timestepSize)
    return (length(genTimes) - 1) * 2 #0.0 does not count
end

#DVSS2X discretization scheme for Heston
function simulateDVSS2X(
    rng,
    model::HestonModel{T},
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB = false,
    cacheSize = 0,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = exp(-model.r * tte)
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
    lnspot = log(model.spot)
    logpathValues .= lnspot
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
        @. logpathValues = (model.r - model.q) * h + xb - c0 + c1 * (yr - model.θ)
        @. v = yr * ekdth + c2
        if t1 == tte
            pathValues = xb #reuse Var
            @. pathValues = exp(logpathValues)
            payoffValues = map(x -> evaluatePayoff(payoff, x, df), pathValues)
        end
        t0 = t1
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
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB = false,
    cacheSize = 0,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = exp(-model.r * tte)
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
    lnspot = log(model.spot)
    logpathValues .= lnspot
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

        @. sqrtmv = sqrt(max(v, 0))
        @. logpathValues +=
            (model.r - model.q - 0.5 * sqrtmv^2) * h + sqrtmv * (u1 * ρBar + u2 * model.ρ)
        # for (i,p) in enumerate(logpathValues)
        #     if isnan(p)
        #         println(i," nan ",sqrtmv[i], " ",u1[i]," ", u2[i])
        #     end
        # end
        @. v += model.κ * (model.θ - sqrtmv^2) * h + model.σ * sqrtmv * u2

        if t1 == tte
            pathValues = sqrtmv #reuse Var
            @. pathValues = exp(logpathValues)
            payoffValues = map(x -> evaluatePayoff(payoff, x, df), pathValues)
        end
        t0 = t1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end


function simulateFullTruncationIter(
    rng,
    model::HestonModel{T},
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB = false,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = exp(-model.r * tte)
    genTimes = pathgenTimes(model, specTimes, timestepSize)
    local bb
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
    end
    lnspot = log(model.spot)
    ρBar = sqrt(1 - model.ρ^2)
    z = Vector{Float64}(undef, (length(genTimes) - 1) * 2)
    u = Array{Float64}(undef, (2, length(genTimes)-1))
    skipTo(rng, start)
    payoffMean = 0.0
    for sim = 1:nSim
        if withBB
            nextn!(rng, z)
            transform!(bb, z, u)
        else
            nextn!(rng, z)
            t0 = genTimes[1]
            @inbounds for i=1:length(genTimes)-1
                dim = 2*i-1
                t1 = genTimes[i+1]
                u[1,i] = z[dim]*sqrt(t1 - t0)
                u[2,i] = z[dim+1]*sqrt(t1 - t0)
                t0 = t1
            end
        end
        t0 = genTimes[1]
        logpathValue = lnspot
        v = model.v0
        local payoffValue
        @inbounds for i=1:length(genTimes)-1
            t1 = genTimes[i+1]
            dim = 2*i -1
            u1 = u[1,i]
            u2 = u[2,i]
            h = t1 - t0
            sqrth = sqrt(h)
            sqrtmv = sqrt(max(v, 0))
            logpathValue +=
                (model.r - model.q - 0.5 * sqrtmv^2) * h +
                sqrtmv * (u1 * ρBar + u2 * model.ρ)
            v += model.κ * (model.θ - sqrtmv^2) * h + model.σ * sqrtmv * u2

            if t1 == tte
                pathValue = exp(logpathValue)
                payoffValue = evaluatePayoff(payoff, pathValue, df)
            end
            t0 = t1
        end

        payoffMean += payoffValue
    end
    payoffMean /= nSim
    return payoffMean, NaN
end
