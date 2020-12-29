import AQFED.Math: norminv
using Statistics
import  AQFED.TermStructure: HestonModel

#DVSS2X discretization scheme for Heston
function simulateDVSS2X(
    rng,
    model::HestonModel{T},
    payoff::VanillaOption,
    nSim::Int,
    nSteps::Int,
) where {T}
    tte = payoff.maturity
    df = exp(-model.r * tte)
    genTimes = LinRange(0.0, tte, ceil(Int, nSteps * tte) + 1)
    logpathValues = Vector{T}(undef, nSim)
    u1 = Vector{Float64}(undef, nSim)
    u2 = Vector{Float64}(undef, nSim)
    v = Vector{T}(undef, nSim)
    t0 = genTimes[1]
    lnspot = log(model.spot)
    logpathValues .= lnspot
    v .= model.v0
    ρBar = sqrt(1 - model.ρ^2)
    local payoffValues
    for t1 in genTimes[2:end]
        h = t1 - t0
        ekdth = exp(-model.κ * h / 2)
        c0 = model.θ * h / 4
        c1 = (ekdth - 1) / (model.κ * 2)
        c2 = model.θ * (1 - ekdth)
        rand!(rng, u1)
        xsi = u1
        @. xsi = norminv(u1)
        rand!(rng, u2)
        #xsi, u2 = norminv.( rand(rng, Float64, nSim) ), rand(rng, Float64, nSim)
        #xsi, u2 = randn(rng, Float64,nSim), rand(rng, Float64, nSim)
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
            model.σ * (ρBar * xsi * sqrt((vTmp + yr) / 2 * h) + model.ρ * (yr - vTmp))
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
