import AQFED.Math: norminv

import AQFED.TermStructure: HestonModel, DoubleHestonModel, SchobelZhuModel, discountFactor, forward, PiecewiseConstantFunction, HaganHestonPiecewiseParams, Curve
import AQFED.Random: next!, nextn!, skipTo!
import CharFuncPricing: HestonTSParams
export last_stat, QE, FullTruncation, DVSS2X, GridConfig, simulateIter, simulateScheme, MCConvergenceResult
import DataStructures: SortedSet
#one-factor stochastic volatility models simulated by the vectorized engine (simulateScheme)
const StochasticVolatilityModel{T} = Union{HestonModel{T},SchobelZhuModel{T}}

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

abstract type PathGenerator end
struct LogEuler <: PathGenerator
    withBB::Bool
end
struct DVSS2X <: PathGenerator
    withBB::Bool
end
function DVSS2X(; withBB::Bool=false)
    return DVSS2X(withBB)
end
struct FullTruncation <: PathGenerator
    withBB::Bool
end
FullTruncation(; withBB::Bool=false) = FullTruncation(withBB)

struct QE <: PathGenerator
    withBB::Bool
    withMartingale::Bool
end
function QE(; withBB::Bool=false, withMartingale::Bool=true)
    return QE(withBB, withMartingale)
end

struct IVI <: PathGenerator
    withBB::Bool
end
IVI(; withBB::Bool=false) = IVI(withBB)

variatesPerStep(::HestonModel, ::DVSS2X) = 2
variatesPerStep(::HestonModel,::FullTruncation) = 2
variatesPerStep(::HestonModel,::QE) = 2
variatesPerStep(::HestonModel,::IVI) = 3

struct GridConfig{M<:PathGenerator}
    specTimes::Vector{Float64}
    mindt::Float64
    tstop::Float64
    maxdt::Float64
    genTimes::Vector{Float64}
    method::M
end

function GridConfig(
    model::HestonModel,
    specTimes::Vector{Float64},
    timestepSize::Float64,
    method::PathGenerator;
    mindt::Float64=timestepSize,
    tstop::Float64=0.0,
    maxdt::Float64=timestepSize,
)
    genTimes = pathgenTimesRamp(specTimes, mindt, tstop, maxdt)
    return GridConfig(specTimes, mindt, tstop, maxdt, genTimes, method)
end

function pathgenTimesUniform(model::HestonModel, obs::Vector{Float64}, dt_target::Float64)
    if length(obs) == 0 || (length(obs) > 1 && obs[1] != 0.0)
        obs = vcat(0.0, obs)
    end
    t = sort(unique(obs))
    isempty(t) && return Float64[]
    out = Float64[t[1]]

    for k in 1:(length(t)-1)
        a, b = t[k], t[k+1]
        h = b - a
        n = max(1, ceil(Int, h / dt_target))
        append!(out, collect(range(a, b; length=n+1))[2:end])
    end
    #println("pathGenTimes=",out)
    out
end


"""
Build a non-uniform time grid for Monte Carlo simulation.

The grid has two phases:
  - Ramp [0, tstop]: local step size grows linearly from mindt to maxdt,
    implemented via a time-change φ(t) = ∫ ds/h(s) where h(s) = mindt + (maxdt-mindt)*s/tstop.
    Points are placed at uniform φ-spacing within each observation sub-interval.
  - Post-ramp [tstop, tte]: constant step maxdt, split uniformly between observations.

Observation times (specTimes) and tstop are used as anchors: within each
anchor sub-interval, n = ceil(Δφ) points are placed at equal φ-spacing,
then inverted back to physical time. A final pass splits any remaining
step > maxdt.
"""
function pathgenTimes(
    model::HestonModel{T},
    specTimes::AbstractVector{<:Real},
    timestepSize::Float64;
    mindt::Float64=timestepSize,
    tstop::Float64=0.0,
    maxdt::Float64=timestepSize,
) where {T}
    return pathgenTimesRamp(specTimes, mindt, tstop, maxdt)
end

function pathgenTimesRamp(
    specTimes::AbstractVector{<:Real},
    mindt::Float64,
    tstop::Float64,
    maxdt::Float64;
    atol::Float64=1e-12,
)
    @assert !isempty(specTimes) "specTimes must not be empty"
    @assert all(s >= 0 for s in specTimes) "specTimes must be nonnegative"
    @assert issorted(specTimes) "specTimes must be sorted in increasing order"
    @assert mindt > 0.0 "mindt must be positive"
    @assert maxdt >= mindt "maxdt must be >= mindt"
    @assert tstop >= 0.0 "tstop must be >= 0"

    tte = Float64(specTimes[end])
    T = min(tstop, tte)
    a, b = mindt, maxdt
    ramp = (b > a) && (T > atol)

    phi_T = ramp ? (T / (b - a)) * log(b / a) : (T > atol ? T / a : 0.0)

    phi(t::Float64) =
        if t <= T + atol
            if ramp
                (T / (b - a)) * log((a + (b - a) * t / T) / a)
            else
                t / a
            end
        else
            phi_T + (t - T) / b
        end

    phi_inv(p::Float64) =
        if T > atol && p <= phi_T + atol
            if ramp
                (a * T / (b - a)) * (exp((b - a) * p / T) - 1.0)
            else
                a * p
            end
        else
            T + b * (p - phi_T)
        end

    anchor_set = SortedSet{Float64}()
    push!(anchor_set, 0.0)
    for s in specTimes
        push!(anchor_set, Float64(s))
    end
    if atol < T < tte - atol
        push!(anchor_set, T)
    end
    anchors = collect(anchor_set)

    times = Float64[0.0]
    for k in 2:length(anchors)
        ta, tb = anchors[k-1], anchors[k]
        if tb - ta < atol
            continue
        end
        pa, pb = phi(ta), phi(tb)
        dphi = pb - pa
        n = max(1, ceil(Int, dphi - 1e-10))
        n = max(n, ceil(Int, (tb - ta) / maxdt - 1e-10))
        for j in 1:n
            pj = pa + j * dphi / n
            tj = phi_inv(pj)
            tj = clamp(tj, ta, tb)
            if tj > times[end] + atol
                push!(times, tj)
            end
        end
    end

    for s in specTimes
        sf = Float64(s)
        if all(abs(sf - x) > 1e-8 for x in times)
            push!(times, sf)
        end
    end
    sort!(times)

    i = 1
    while i < length(times)
        dt = times[i+1] - times[i]
        if dt > maxdt + 1e-10
            n_split = ceil(Int, dt / maxdt)
            dt_new = dt / n_split
            for j in 1:(n_split-1)
                insert!(times, i + j, times[i] + j * dt_new)
            end
            i += n_split
        else
            i += 1
        end
    end

    sort!(times)

    out = Float64[times[1]]
    for t in times[2:end]
        if t > out[end] + atol
            push!(out, t)
        end
    end

    return out
end
function ndims(model, config::GridConfig)
    return (length(config.genTimes) - 1) * variatesPerStep(model, config.method)
end
#dimension for the simulate* functions that take a timestepSize and build the default GridConfig.
ndims(model::HestonModel, specificTimes::Vector{Float64}, timestepSize::Float64; method::PathGenerator=FullTruncation()) =
    ndims(model, GridConfig(model, specificTimes, timestepSize, method))


function ndims(model::DoubleHestonModel, specificTimes::Vector{Float64}, timestepSize::Float64)
    genTimes = pathgenTimes(model, specificTimes, timestepSize)
    return (length(genTimes) - 1) * 4 #0.0 does not count
end

function ndims(model::Union{HestonTSParams,HaganHestonPiecewiseParams}, specificTimes::Vector{Float64}, timestepSize::Float64)
    iLast = searchsortedfirst(model.startTime, specificTimes[end])
    specTimesAll = sort(unique(vcat(specificTimes, model.startTime[2:(iLast-1)])))
    genTimes = pathgenTimes(model, specTimesAll, timestepSize)
    return (length(genTimes) - 1) * 2 #0.0 does not count
end


function inverseGaussian(μ, λ, normalZ, uniformU)
    y = normalZ^2
    x = y
    x = μ + (μ^2 * y) / (2λ) - μ / (2λ) * sqrt(4 * μ * λ * y + μ^2 * y^2)

    ifelse(uniformU <= μ / (μ + x), x, μ^2 / x)
end


function simulateDVSS2X(
    rng,
    model::DoubleHestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
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
            skipTo!(rng, d + ndimsh, start, length(xsi1))
            transformByDim!(bb, rng, start, d + ndimsh, xsi2, cache)
            skipTo!(rng, d + 2ndimsh, start, length(u3))
            next!(rng, d + 2ndimsh, u3)
            skipTo!(rng, d + 3ndimsh, start, length(u4))
            next!(rng, d + 3ndimsh, u4)
        else
            d = 4 * dim - 1
            skipTo!(rng, d, start, length(xsi1))
            nextn!(rng, d, xsi1)
            skipTo!(rng, d + 1, start, length(xsi2))
            nextn!(rng, d + 1, xsi2)
            xsi1 .*= sqrt(h)
            xsi2 .*= sqrt(h)
            skipTo!(rng, d + 2, start, length(u3))
            next!(rng, d + 2, u3)
            skipTo!(rng, d + 3, start, length(u4))
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
    return if (u2 < p1)
        y1
    elseif u2 > 1 - p2
        y2
    else
        y0
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
    return if (u2 < p1)
        y1
    elseif u2 > 1 - p2
        y2
    else
        y0
    end
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
    leverageFunction=t -> 1
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    iLast = searchsortedfirst(model.startTime, tte)
    specTimesAll = sort(unique(vcat(specTimes, model.startTime[2:(iLast-1)])))
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
        κ = kappaFunc(t0 + h / 2)
        θ = thetaFunc(t0 + h / 2)
        ρ = rhoFunc(t0 + h / 2)
        σ = sigmaFunc(t0 + h / 2)
        lev = leverageFunction(t0 + h / 2)
        ρBar = sqrt(1 - ρ^2)
        sqrth = sqrt(h)
        if withBB
            transformByDim!(bb, rng, start, dim, u, cache)
        else
            d = 2 * dim - 1
            skipTo!(rng, d, start, length(u1))
            nextn!(rng, d, u1)
            skipTo!(rng, d + 1, start, length(u2))
            nextn!(rng, d + 1, u2)
            #@inbounds @. u *= sqrth
            u1 .*= sqrth
            u2 .*= sqrth
        end
        lnf1 = lnspot #logForward(model, lnspot, t1)
        @. sqrtmv = sqrt(max(v, 0))
        @. logpathValues +=
            lnf1 - lnf0 - (0.5 * (lev * sqrtmv)^2) * h + lev * sqrtmv * (u1 * ρBar + u2 * ρ)
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



function simulateDVSS2X(
    rng,
    model::HaganHestonPiecewiseParams{T},
    spot::Float64,
    forecastCurve::Curve,
    discountCurve::Curve,
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = discountFactor(discountCurve, tte)
    iLast = searchsortedfirst(model.startTime, tte)
    specTimesAll = sort(unique(vcat(specTimes, model.startTime[2:(iLast-1)])))
    genTimes = pathgenTimes(model, specTimesAll, timestepSize)
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
    lnf0 = lnspot
    v .= 1.0
    levFunc = PiecewiseConstantFunction(model.startTime, model.leverage)
    rhoFunc = PiecewiseConstantFunction(model.startTime, model.ρ)
    sigmaFunc = PiecewiseConstantFunction(model.startTime, model.σ)

    logpathValues .= lnf0
    ndimsh = length(genTimes) - 1
    local payoffValues
    for (dim, t1) in enumerate(genTimes[2:end])
        h0 = (t1 - t0)
        θ = levFunc(t0 + h0 / 2)
        ρ = rhoFunc(t0 + h0 / 2)
        σ = sigmaFunc(t0 + h0 / 2)
        ρBar = sqrt(1 - ρ^2)
        h = (t1 - t0) * θ^2
        ekdth = exp(-model.κ * h0 / 2)
        c0 = h / 4
        c1 = (ekdth - 1) / (model.κ * 2) * θ^2
        c2 = (1 - ekdth)
        if withBB
            d = dim
            transformByDim!(bb, rng, start, d, xsi, cache)
            skipTo!(rng, d + ndimsh, start, length(u2))
            next!(rng, d + ndimsh, u2)
        else
            d = 2 * dim - 1
            skipTo!(rng, d, start, length(xsi))
            nextn!(rng, d, xsi)
            xsi .*= sqrt(h0)
            skipTo!(rng, d + 1, start, length(u2))
            next!(rng, d + 1, u2)
        end

        vTmp = @. (v * ekdth + c2) / σ^2 * θ^2
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
            c1 * (v - 1) +
            σ / θ * (θ * ρBar * xsi * sqrt((vTmp + yr) / 2) + ρ * (yr - vTmp))
        yr .= (σ / θ)^2 * yr
        lnf1 = logForward(forecastCurve, lnspot, t1)
        @. logpathValues = lnf1 - lnf0 + xb - c0 + c1 * (yr - 1)
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

function simulateFullTruncationHagan(
    rng,
    model::HaganHestonPiecewiseParams{T},
    spot::T,
    forecastCurve::Curve,
    discountCurve::Curve,
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = discountFactor(discountCurve, tte)
    iLast = searchsortedfirst(model.startTime, tte)
    specTimesAll = sort(unique(vcat(specTimes, model.startTime[2:(iLast-1)])))
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
    levFunc = PiecewiseConstantFunction(model.startTime, model.leverage)
    rhoFunc = PiecewiseConstantFunction(model.startTime, model.ρ)
    sigmaFunc = PiecewiseConstantFunction(model.startTime, model.σ)
    ndimsh = length(genTimes) - 1
    #  println("genTimes ",genTimes)
    local payoffValues
    jump = 0.0
    for (dim, t1) in enumerate(genTimes[2:end])
        h = t1 - t0
        κ = model.κ
        θ = levFunc(t0 + h / 2)
        ρ = rhoFunc(t0 + h / 2)
        σ = sigmaFunc(t0 + h / 2)
        ρBar = sqrt(1 - ρ^2)
        sqrth = sqrt(h)
        if withBB
            transformByDim!(bb, rng, start, dim, u, cache)
        else
            d = 2 * dim - 1
            skipTo!(rng, d, start, length(u1))
            nextn!(rng, d, u1)
            skipTo!(rng, d + 1, start, length(u2))
            nextn!(rng, d + 1, u2)
            #@inbounds @. u *= sqrth
            u1 .*= sqrth
            u2 .*= sqrth
        end
        lnf1 = logForward(forecastCurve, lnspot, t1)
        @. sqrtmv = sqrt(max(v, 0))
        @. logpathValues +=
            lnf1 - lnf0 - (0.5 * max(0, (θ * sqrtmv)^2)) * h + sqrt(max(0, (sqrtmv * θ)^2)) * (u1 * ρBar + u2 * ρ)
        # for (i,p) in enumerate(logpathValues)
        #     if isnan(p)
        #         println(i," nan ",sqrtmv[i], " ",u1[i]," ", u2[i])
        #     end
        # end
        @. v += κ * (1 - sqrtmv^2) * h + σ * sqrtmv * u2
        #@. v /= θ/θ0 

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
        κ = kappaFunc(t0 + h / 2)
        θ = thetaFunc(t0 + h / 2)
        ρ = rhoFunc(t0 + h / 2)
        σ = sigmaFunc(t0 + h / 2)
        ρBar = sqrt(1 - ρ^2)
        sqrth = sqrt(h)
        if withBB
            transformByDim!(bb, rng, start, dim, u, cache)
        else
            d = 2 * dim - 1
            skipTo!(rng, d, start, length(u1))
            nextn!(rng, d, u1)
            skipTo!(rng, d + 1, start, length(u2))
            nextn!(rng, d + 1, u2)
            #@inbounds @. u *= sqrth
            u1 .*= sqrth
            u2 .*= sqrth
        end
        lnf1 = lnspot #logForward(model, lnspot, t1)
        @. sqrtmv = sqrt(max(v, 0))
        @. logpathValues +=
            lnf1 - lnf0 - (0.5 * max(0, θ * sqrtmv^2)) * h + sqrt(max(0, sqrtmv^2 * θ)) * (u1 * ρBar + u2 * ρ)
        # for (i,p) in enumerate(logpathValues)
        #     if isnan(p)
        #         println(i," nan ",sqrtmv[i], " ",u1[i]," ", u2[i])
        #     end
        # end
        @. v += κ * (1 - sqrtmv^2) * h + σ / sqrt(θ) * sqrtmv * u2
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
            skipTo!(rng, d, start, length(u1))
            nextn!(rng, d, u1)
            skipTo!(rng, d + 1, start, length(u2))
            nextn!(rng, d + 1, u2)
            skipTo!(rng, d + 2, start, length(u3))
            nextn!(rng, d + 2, u3)
            skipTo!(rng, d + 3, start, length(u4))
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
            lnf1 - lnf0 - (0.5 * (sqrtmv1^2 + sqrtmv2^2)) * h + sqrtmv1 * (u1 * ρ1Bar + u3 * model.params.heston1.ρ) + sqrtmv2 * (u2 * ρ2Bar + u4 * model.params.heston2.ρ)
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
function simulateIter(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{FullTruncation};
    withBB=config.method.withBB, isRestart::Bool=true, isInterleaved=true
) where {T}
    specTimes = config.specTimes
    tte = specTimes[end]
    df = discountFactor(model, tte)
    genTimes = config.genTimes
    obsIdx = _obs_indices(genTimes, specTimes)

    local bb
    if withBB
        bb = MultiAssetBridge(genTimes[2:end], 2, isInterleaved ? StepInterleaved() : AssetBlocked())
    end

    lnspot = log(spot)
    ρBar = sqrt(1 - model.ρ^2)

    lnf = map(t -> logForward(model, lnspot, t), genTimes)

    z = Vector{Float64}(undef, (length(genTimes) - 1) * 2)
    u = Array{Float64}(undef, 2, length(genTimes) - 1)
    obsPath = zeros(length(specTimes))

    if isRestart
        skipTo!(rng, start)
    end
    dimOut = dimensionOut(payoff)[1]
    payoffValues = zeros(dimOut, nSim)

    for sim in 1:nSim
        if withBB
            nextn!(rng, z)
            transform!(bb, z, u)
        else
            nextn!(rng, z)
            t0 = genTimes[1]
            @inbounds for i in 1:(length(genTimes)-1)
                dim = 2 * i - 1
                t1 = genTimes[i+1]
                sdt = sqrt(t1 - t0)
                u[1, i] = z[dim] * sdt
                u[2, i] = z[dim+1] * sdt
                t0 = t1
            end
        end

        # Start from lnf[1], but if t=0, this is just log(spot)
        logpathValue = lnf[1]
        v = model.v0
        nextObs = 1
        nextObs = _record_observation!(obsPath, nextObs, obsIdx, 0, logpathValue)


        @inbounds for i in 1:(length(genTimes)-1)
            u1i = u[1, i]
            u2i = u[2, i]
            h = genTimes[i+1] - genTimes[i]

            # Full truncation: use max(v, 0) everywhere
            v_pos = max(v, 0.0)  # ← Key fix
            sqrtmv = sqrt(v_pos)

            logpathValue += lnf[i+1] - lnf[i] - 0.5 * v_pos * h + sqrtmv * (u1i * ρBar + u2i * model.ρ)
            v += model.κ * (model.θ - v_pos) * h + model.σ * sqrtmv * u2i

            nextObs = _record_observation!(obsPath, nextObs, obsIdx, i, logpathValue)
        end
        #println("path ",obsPath)
        _store_payoff!(payoffValues, sim, payoff, obsPath, df)
        _check_nan(payoffValues, sim, obsPath, u)
    end

    return mc_convergence_detailed(payoffValues)
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
    config = GridConfig(model, specificTimes(payoff), timestepSize, FullTruncation(withBB=withBB))
    return simulateIter(rng, model, spot, payoff, start, nSim, config; withBB=withBB)
end
function simulateIter(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{QE};
    withBB=config.method.withBB,
    withMartingale=config.method.withMartingale,
    isRestart::Bool=true
) where {T}
    specTimes = config.specTimes
    tte = specTimes[end]
    df = discountFactor(model, tte)
    genTimes = config.genTimes
    obsIdx = _obs_indices(genTimes, specTimes)

    local bb
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
    end
    if isRestart
        skipTo!(rng, start)
    end

    lnspot = log(spot)
    ρBar = sqrt(1 - model.ρ^2)
    dimOut = dimensionOut(payoff)[1]
    payoffValues = zeros(T, (dimOut, nSim))

    h = genTimes[2:end] - genTimes[1:(end-1)]
    ekdt = @. exp(-model.κ * h)

    u = zeros(2, length(h))
    u1 = @view u[1, :]
    u2 = @view u[2, :]
    z1 = zeros(length(h))
    psiC = 1.5

    obsPath = zeros(T, length(specTimes))
    lnf = map(t -> logForward(model, lnspot, t), genTimes)

    for sim in 1:nSim
        next!(rng, u)
        @. u1 = norminv(u1)

        if withBB
            transform!(bb, u1, z1)
        else
            @. z1 = u1 * sqrt(h)
        end

        logpathValue = lnf[1]
        v = model.v0
        nextObs = 1
        nextObs = _record_observation!(obsPath, nextObs, obsIdx, 0, logpathValue)

        @inbounds for i in 1:length(h)
            m = model.θ + (v - model.θ) * ekdt[i]
            s2 = v * model.σ^2 * ekdt[i] / model.κ * (1 - ekdt[i]) + (model.θ * model.σ^2) / (2 * model.κ) * (1 - ekdt[i])^2
            psi = s2 / (m^2)
            vNew = 0.0
            martingaleCorrection = 0.0

            if psi <= psiC
                psiInv = 1 / psi
                b2 = 2 * psiInv - 1 + sqrt(2 * psiInv * (2 * psiInv - 1))
                a = m / (1 + b2)
                b = sqrt(b2)
                zv = norminv(u2[i])
                vNew = a * (b + zv)^2
                if withMartingale
                    A = h[i] / 2 * (model.κ * model.ρ / model.σ - 1 / 2) + model.ρ / model.σ + h[i] / 4 * (1 - model.ρ^2)
                    B = h[i] / 2 * (model.κ * model.ρ / model.σ - 1 / 2) - model.ρ / model.σ + h[i] / 4 * (1 - model.ρ^2)
                    C = model.ρ / model.σ * model.κ * model.θ * h[i]
                    martingaleCorrection = C - A * b2 * a / (1 - 2 * A * a) + 0.5 * log(1 - 2 * A * a) - B * v
                end
            else
                p = (psi - 1) / (psi + 1)
                invBeta = m / (1 - p)
                if u2[i] <= p
                    vNew = 0.0
                else
                    vNew = log((1 - p) / (1 - u2[i])) * invBeta
                end
                if withMartingale
                    A = h[i] / 2 * (model.κ * model.ρ / model.σ - 1 / 2) + model.ρ / model.σ + h[i] / 4 * (1 - model.ρ^2)
                    B = h[i] / 2 * (model.κ * model.ρ / model.σ - 1 / 2) - model.ρ / model.σ + h[i] / 4 * (1 - model.ρ^2)
                    C = model.ρ / model.σ * model.κ * model.θ * h[i]
                    martingaleCorrection = C - log(p + (1 - p) / (1 - invBeta * A)) - B * v
                end
            end

            vIntegral = (vNew + v) / 2
            logpathValue += lnf[i+1] - lnf[i] + model.ρ / model.σ * (vNew - v - model.κ * model.θ * h[i]) +
                            (model.κ * model.ρ / model.σ - 1 / 2) * vIntegral * h[i] +
                            ρBar * sqrt(vIntegral) * z1[i] + martingaleCorrection
            v = vNew

            nextObs = _record_observation!(obsPath, nextObs, obsIdx, i, logpathValue)
        end

        _store_payoff!(payoffValues, sim, payoff, obsPath, df)
        _check_nan(payoffValues, sim, obsPath, u)
    end

    return mc_convergence_detailed(payoffValues)
end
function simulateQEIter(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{QE};
    withBB=config.method.withBB,
    withMartingale=config.method.withMartingale,
) where {T}
    return simulateIter(rng, model, spot, payoff, start, nSim, config; withBB=withBB, withMartingale=withMartingale)
end

function simulateQEIter(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false, withMartingale=true
) where {T}
    config = GridConfig(model, specificTimes(payoff), timestepSize, QE(withBB=withBB, withMartingale=withMartingale))
    return simulateIter(rng, model, spot, payoff, start, nSim, config; withBB=withBB, withMartingale=withMartingale)
end

function simulateIter(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{IVI};
    withBB=config.method.withBB,
    isRestart::Bool=true
) where {T}
    specTimes = config.specTimes
    tte = specTimes[end]
    df = discountFactor(model, tte)
    genTimes = config.genTimes
    obsIdx = _obs_indices(genTimes, specTimes)

    local bb
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
    end

    if isRestart
        skipTo!(rng, start)
    end
    lnspot = log(spot)
    ρBar = sqrt(1 - model.ρ^2)
    h = genTimes[2:end] - genTimes[1:(end-1)]

    u = zeros(3, length(h))
    u1 = @view u[1, :]
    u2 = @view u[2, :]
    u3 = @view u[3, :]
    z1 = zeros(length(h))
    z3 = zeros(length(h))
    dimOut = dimensionOut(payoff)[1]
    payoffValues = zeros(dimOut, nSim)

    a = model.κ * model.θ
    b = -model.κ
    c = model.σ
    obsPath = zeros(length(specTimes))

    lnf = map(t -> logForward(model, lnspot, t), genTimes)
    ebm1 = @. (exp(b * h) - 1) / b

    for sim in 1:nSim
        next!(rng, u)
        @. u1 = norminv(u1)
        @. u3 = norminv(u3)

        if withBB
            transform!(bb, u1, z1)
            transform!(bb, u3, z3)
            @. z1 /= sqrt(h)
            @. z3 /= sqrt(h)
        else
            @. z1 = u1
            @. z3 = u3
        end

        logpathValue = lnf[1]
        v = model.v0
        nextObs = 1
        nextObs = _record_observation!(obsPath, nextObs, obsIdx, i, logpathValue)

        @inbounds for i in 1:length(h)
            aFactor = a / b * (ebm1[i] - h[i])
            σFactor = c * ebm1[i]
            α = v * ebm1[i] + aFactor
            uig = inverseGaussian(α, (α / σFactor)^2, z1[i], u2[i])
            zig = (uig - α) / σFactor
            v += a * h[i] + b * uig + c * zig
            logpathValue += lnf[i+1] - lnf[i] - uig / 2 + model.ρ * zig + ρBar * sqrt(uig) * z3[i]
            nextObs = _record_observation!(obsPath, nextObs, obsIdx, i, logpathValue)
        end

        _store_payoff!(payoffValues, sim, payoff, obsPath, df)
        _check_nan(payoffValues, sim, obsPath, u)
    end

    return mc_convergence_detailed(payoffValues)
end
function simulateIVIIter(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
) where {T}
    config = GridConfig(model, specificTimes(payoff), timestepSize, IVI(withBB=withBB))
    return simulateIter(rng, model, spot, payoff, start, nSim, config; withBB=withBB)
end
function simulateIter(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{DVSS2X};
    withBB=config.method.withBB,
    isRestart::Bool=true
) where {T}
    specTimes = config.specTimes
    tte = specTimes[end]
    df = discountFactor(model, tte)
    genTimes = config.genTimes
    obsIdx = _obs_indices(genTimes, specTimes)

    local bb
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
    end
    if isRestart
        skipTo!(rng, start)
    end

    lnspot = log(spot)
    ρBar = sqrt(1 - model.ρ^2)
    dimOut = dimensionOut(payoff)[1]
    payoffValues = zeros(dimOut, nSim)

    h = genTimes[2:end] - genTimes[1:(end-1)]
    ekdth = @. exp(-model.κ * h / 2)

    u = zeros(2, length(h))
    u1 = @view u[1, :]
    u2 = @view u[2, :]
    z1 = zeros(length(h))
    obsPath = zeros(length(specTimes))
    lnf = map(t -> logForward(model, lnspot, t), genTimes)

    for sim in 1:nSim
        next!(rng, u)
        @. u1 = norminv(u1)

        if withBB
            transform!(bb, u1, z1)
        else
            @. z1 = u1 * sqrt(h)
        end

        logpathValue = lnf[1]
        v = model.v0
        nextObs = 1
        nextObs = _record_observation!(obsPath, nextObs, obsIdx, 0, logpathValue)

        @inbounds for i in 1:length(h)
            c0 = model.θ * h[i] / 4
            c1 = (ekdth[i] - 1) / (model.κ * 2)
            c2 = model.θ * (1 - ekdth[i])
            vTmp = (v * ekdth[i] + c2) / model.σ^2

            yr = ifelse(
                vTmp > 1.896 * h[i],
                dvss2_case1(vTmp, u2[i], h[i]),
                dvss2_case2(vTmp, u2[i], h[i]),
            )

            xb = logpathValue - c0 + c1 * (v - model.θ) +
                 model.σ * (ρBar * z1[i] * sqrt((vTmp + yr) / 2) + model.ρ * (yr - vTmp))

            yr = model.σ^2 * yr
            logpathValue = lnf[i+1] - lnf[i] + xb - c0 + c1 * (yr - model.θ)
            v = yr * ekdth[i] + c2



            nextObs = _record_observation!(obsPath, nextObs, obsIdx, i, logpathValue)
        end

        _store_payoff!(payoffValues, sim, payoff, obsPath, df)
        _check_nan(payoffValues, sim, obsPath, u)
    end

    return mc_convergence_detailed(payoffValues)
end
function simulateDVSS2XIter(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{DVSS2X};
    withBB=config.method.withBB,
) where {T}
    return simulateIter(rng, model, spot, payoff, start, nSim, config; withBB=withBB)
end

function simulateDVSS2XIter(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
) where {T}
    config = GridConfig(model, specificTimes(payoff), timestepSize, DVSS2X(withBB=withBB))
    return simulateDVSS2XIter(rng, model, spot, payoff, start, nSim, config; withBB=withBB)
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
            skipTo!(rng, d + ndimsh, start, length(u2))
            next!(rng, d + ndimsh, u2)
        else
            d = 2 * dim - 1
            skipTo!(rng, d, start, length(xsi))
            nextn!(rng, d, xsi)
            xsi .*= sqrt(h)
            skipTo!(rng, d + 1, start, length(u2))
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
            skipTo!(rng, d, start, length(u1))
            nextn!(rng, d, u1)
            skipTo!(rng, d + 1, start, length(u2))
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


struct Alfonsi2 <: PathGenerator
    withBB::Bool
end

Alfonsi2(; withBB::Bool=false) = Alfonsi2(withBB)

variatesPerStep(::HestonModel, ::Alfonsi2) = 2
const variatesPerStepAlfonsi2 = 2

function simulate(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{Alfonsi2};
    withBB=config.method.withBB,
    cacheSize=0,
) where {T}
    specTimes = config.specTimes
    genTimes = config.genTimes
    tte = specTimes[end]
    df = discountFactor(model, tte)

    logpathValues = Vector{T}(undef, nSim)
    obsPath = Matrix{T}(undef, nSim, length(specTimes))

    local bb, cache
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
        if cacheSize == 0
            nSteps = length(genTimes) - 1
            cacheSize = ceil(Int, log2(nSteps)) * 4
        end
        cache = BBCache{Int,Vector{Float64}}(cacheSize)
    end

    u1 = Vector{Float64}(undef, nSim)   # Gaussian for asset leg
    u2 = Vector{Float64}(undef, nSim)   # Uniform for variance branch
    w = u1
    v = Vector{T}(undef, nSim)

    t0 = genTimes[1]
    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, t0)
    logpathValues .= lnf0
    v .= model.v0

    κ = model.κ
    θ = model.θ
    σ = model.σ
    ρ = model.ρ
    ρBar = sqrt(1 - ρ^2)

    ndimsh = length(genTimes) - 1

    ekappadt = Vector{Float64}(undef, ndimsh)
    K2 = Vector{Float64}(undef, ndimsh)

    lastTime = t0
    for (i, time) in enumerate(genTimes[2:end])
        dt = time - lastTime
        ekdt = exp(-κ * dt)
        ekappadt[i] = ekdt
        K2[i] = 0.0
        if σ^2 > 4 * κ * θ
            ψ = (1 - sqrt(ekdt)) / κ
            ekhalf = sqrt(1.0 / ekdt)
            term = sqrt(ekhalf * (σ^2 / 4 - θ * κ) * ψ) + σ * 0.5 * sqrt(3 * dt)
            K2[i] = ekhalf * ((σ^2 / 4 - θ * κ) * ψ + term * term)
        end
        lastTime = time
    end

    payoffValues = zeros(nSim)
    iObs = 1

    for (dim, t1) in enumerate(genTimes[2:end])
        dt = t1 - t0
        ekdt = ekappadt[dim]

        if withBB
            transformByDim!(bb, rng, start, dim, w, cache)
            skipTo!(rng, dim + ndimsh, start, length(u2))
            next!(rng, dim + ndimsh, u2)
        else
            d = 2 * dim - 1
            skipTo!(rng, d, start, length(u1))
            nextn!(rng, d, w)
            w .*= sqrt(dt)
            skipTo!(rng, d + 1, start, length(u2))
            next!(rng, d + 1, u2)
        end

        lnf1 = logForward(model, lnspot, t1)

        @inbounds for j in eachindex(v)
            vOld = v[j]
            uv = u2[j]
            vNew = vOld

            if vOld >= K2[dim]
                y = 0.0
                sixu = 6 * uv
                if sixu < 1.0
                    y = sqrt(3.0)
                elseif sixu < 2.0
                    y = -sqrt(3.0)
                end

                ψ = (1 - sqrt(ekdt)) / κ
                term = sqrt((κ * θ - σ^2 * 0.25) * ψ + sqrt(ekdt) * vOld) + σ * 0.5 * sqrt(dt) * y
                vNew = sqrt(ekdt) * term * term + (κ * θ - σ^2 * 0.25) * ψ
            else
                m = θ + (vOld - θ) * ekdt
                s2 = vOld * σ^2 * ekdt / κ * (1 - ekdt) + (θ * σ^2) / (2 * κ) * (1 - ekdt) * (1 - ekdt)
                u2mom = s2 + m * m
                π = 0.5 - 0.5 * sqrt(1 - m * m / u2mom)
                if uv <= π
                    vNew = m / (2 * π)
                else
                    vNew = m / (2 * (1 - π))
                end
            end

            vIntegral = 0.5 * (vNew + vOld)

            logpathValues[j] +=
                (lnf1 - lnf0) +
                ρ / σ * (vNew - vOld - κ * θ * dt) +
                (κ * ρ / σ - 0.5) * vIntegral * dt +
                ρBar * sqrt(vIntegral) * w[j]

            v[j] = vNew
        end

        if abs(t1 - specTimes[iObs]) < 1e-8
            obsPath[:, iObs] = exp.(logpathValues)
            iObs += 1
        end
        if abs(t1 - tte) < 1e-8
            payoffValues .= [evaluatePayoffOnPath(payoff, path, df) for path in eachrow(obsPath)]
        end

        t0 = t1
        lnf0 = lnf1
    end

    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end

function simulate(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
) where {T}
    config = GridConfig(model, specificTimes(payoff), timestepSize, Alfonsi2(withBB=withBB))
    return simulate(rng, model, spot, payoff, start, nSim, config; withBB=withBB, cacheSize=cacheSize)
end

function simulateIter(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{Alfonsi2};
    withBB=config.method.withBB,
    isRestart::Bool=true
) where {T}
    specTimes = config.specTimes
    genTimes = config.genTimes
    tte = specTimes[end]
    df = discountFactor(model, tte)

    local bb
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
    end

    if isRestart
        skipTo!(rng, start)
    end

    lnspot = log(spot)
    lnf = map(t -> logForward(model, lnspot, t), genTimes)

    κ = model.κ
    θ = model.θ
    σ = model.σ
    ρ = model.ρ
    ρbar = sqrt(1 - ρ^2)
    a = κ * θ

    h = genTimes[2:end] .- genTimes[1:(end-1)]
    nSteps = length(h)
    sqrth = sqrt.(h)
    e = exp.(-κ .* h)
    ehalf = exp.(-κ .* h ./ 2)

    ψh = similar(h)
    ψh2 = similar(h)
    K2 = zeros(Float64, nSteps)

    unrestricted = σ^2 > 4a

    for i in 1:nSteps
        ψh[i] = (1 - e[i]) / κ
        ψh2[i] = (1 - ehalf[i]) / κ
        if unrestricted
            tmp = (σ^2 / 4 - a) * ψh2[i]
            etmp = exp(κ * h[i] / 2)
            inside = max(etmp * tmp, 0.0)
            K2[i] = etmp * (tmp + (sqrt(inside) + 0.5 * σ * sqrt(3h[i]))^2)
        else
            K2[i] = 0.0
        end
    end

    obsIdx = _obs_indices(genTimes, specTimes)

    u = Matrix{Float64}(undef, 2, nSteps)
    u1 = @view u[1, :]
    u2 = @view u[2, :]
    w = Vector{Float64}(undef, nSteps)

    obsPath = Vector{Float64}(undef, length(specTimes))
    dimOut = dimensionOut(payoff)[1]
    payoffValues = zeros(dimOut, nSim)

    sqrt3 = sqrt(3.0)

    for sim in 1:nSim

        next!(rng, u)
        @. u1 = norminv(u1)

        if withBB
            transform!(bb, u1, w)
        else
            @. w = u1 * sqrth
        end

        logpathValue = lnf[1]
        v = model.v0
        nextObs = 1
        #observation at the start of the path
        if nextObs <= length(obsIdx) && obsIdx[nextObs] == 0
            obsPath[nextObs] = exp(logpathValue)
            nextObs += 1
        end

        @inbounds for i in 1:nSteps
            hi = h[i]
            uv = u2[i]
            vNew = 0.0

            if unrestricted && v <= K2[i]
                ν1 = v * e[i] + a * ψh[i]
                ν2 = ν1^2 + σ^2 * ψh[i] * (0.5 * a * ψh[i] + v * e[i])
                rad = max(0.0, 1.0 - ν1^2 / max(ν2, eps(Float64)))
                π = 0.5 * (1.0 - sqrt(rad))

                if uv <= π
                    vNew = ν1 / max(2π, eps(Float64))
                else
                    vNew = ν1 / max(2 * (1 - π), eps(Float64))
                end
            else
                y = if uv <= 1 / 6
                    sqrt3
                elseif uv <= 5 / 6
                    0.0
                else
                    -sqrt3
                end

                inner = (a - σ^2 / 4) * ψh2[i] + ehalf[i] * v
                inner = max(inner, 0.0)

                tmp = sqrt(inner) + 0.5 * σ * sqrth[i] * y
                vNew = ehalf[i] * tmp^2 + (a - σ^2 / 4) * ψh2[i]
            end

            vNew = max(vNew, 0.0)
            I = 0.5 * hi * (v + vNew)

            if I < 0.0
                I = 0.0
            end

            logpathValue +=
                lnf[i+1] - lnf[i] +
                (ρ / σ) * (vNew - v - a * hi + κ * I) -
                0.5 * I +
                ρbar * sqrt(I / hi) * w[i]

            v = vNew

            if nextObs <= length(obsIdx) && i == obsIdx[nextObs]
                obsPath[nextObs] = exp(logpathValue)
                nextObs += 1
            end
        end

        payoffValues[:, sim] .= evaluatePayoffOnPath(payoff, obsPath, df)
        if isnan(payoffValues[1, sim])
            throw(ErrorException("NaN value, obsPath=$obsPath, u=$u"))
        end
    end
    return mc_convergence_detailed(payoffValues)
end



struct MCConvergenceResult
    means::Vector{Float64}
    stderrs::Vector{Float64}
    samples::Vector{Int}
end
@inline last_mean(r::MCConvergenceResult) = r.means[end]
@inline last_stderr(r::MCConvergenceResult) = r.stderrs[end]
@inline last_sample(r::MCConvergenceResult) = r.samples[end]
last_stat(r::MCConvergenceResult) = (r.means[end], r.stderrs[end])
#allows value, stderror = simulate...(...) as with the former tuple results
Base.iterate(r::MCConvergenceResult, state::Int=1) = state == 1 ? (last_mean(r), 2) : state == 2 ? (last_stderr(r), 3) : nothing
Base.length(::MCConvergenceResult) = 2
Base.getindex(r::MCConvergenceResult, i::Int) = i == 1 ? last_mean(r) : i == 2 ? last_stderr(r) : throw(BoundsError(r, i))
function last_stat(v::Vector{MCConvergenceResult})
    return ([r.means[end] for r in v], [r.stderrs[end] for r in v])
end

@inline function mc_convergence_detailed(payoffValues::AbstractVector;
    min_pow=2^7)
    n = length(payoffValues)
    max_pow = floor(Int, log2(n))

    means = Float64[]
    stderrs = Float64[]
    samples = Int[]
    n_samples = 0
    for pow in min_pow:max_pow
        n_samples = 2^pow
        if n_samples <= n
            subset = @view payoffValues[strike_idx, 1:n_samples]
            m = mean(subset)
            s = stdm(subset, m) / sqrt(n_samples)
            push!(means, m)
            push!(stderrs, s)
            push!(samples, n_samples)
        end
    end

    if n_samples < n
        n_samples = n
        subset = @view payoffValues[1:n_samples]
        m = mean(subset)
        s = stdm(subset, m) / sqrt(n_samples)
        push!(means, m)
        push!(stderrs, s)
        push!(samples, n_samples)
    end
    return MCConvergenceResult(means, stderrs, samples)
end

@inline function mc_convergence_detailed(payoffValues::AbstractMatrix;
    min_pow=2^7)
    n_strikes, n_samples = size(payoffValues)
    max_pow = floor(Int, log2(n_samples))

    results = Vector{MCConvergenceResult}()

    for strike_idx in 1:n_strikes
        means = Float64[]
        stderrs = Float64[]
        samples = Int[]
        n_samp = 0
        for pow in 7:max_pow
            n_samp = 2^pow
            if n_samp <= n_samples
                subset = @view payoffValues[strike_idx, 1:n_samp]
                m = mean(subset)
                s = stdm(subset, m) / sqrt(n_samp)
                push!(means, m)
                push!(stderrs, s)
                push!(samples, n_samp)
            end
        end
        if n_samp < n_samples
            n_samp = n_samples
            subset = @view payoffValues[strike_idx, 1:n_samples]
            m = mean(subset)
            s = stdm(subset, m) / sqrt(n_samples)
            push!(means, m)
            push!(stderrs, s)
            push!(samples, n_samples)
        end
        push!(results, MCConvergenceResult(means, stderrs, samples))
    end

    return results
end

@inline function _obs_indices(genTimes, specTimes; atol=1e-8)
    obsIdx = Vector{Int}(undef, length(specTimes))
    j = 1
    # handle observation at t = genTimes[1] (typically 0)
    if abs(specTimes[1] - genTimes[1]) < atol
        obsIdx[1] = 0
        j = 2
    end
    @inbounds for i in 1:(length(genTimes)-1)
        if j <= length(specTimes) && abs(genTimes[i+1] - specTimes[j]) < atol
            obsIdx[j] = i
            j += 1
        end
    end
    return obsIdx
end

@inline function _record_observation!(obsPath, nextObs::Int, obsIdx, i::Int, logpathValue)
    if nextObs <= length(obsIdx) && i == obsIdx[nextObs]
        obsPath[nextObs] = exp(logpathValue)
        return nextObs + 1
    end
    return nextObs
end

@inline function _store_payoff!(payoffValues::AbstractMatrix, sim::Int, payoff, obsPath, df)
    payoffValues[:, sim] .= evaluatePayoffOnPath(payoff, obsPath, df)
    return nothing
end

@inline function _store_payoff!(payoffValues::AbstractVector, sim::Int, payoff, obsPath, df)
    payoffValues[sim] = evaluatePayoffOnPath(payoff, obsPath, df)
    return nothing
end

@inline function _check_nan(payoffValues::AbstractMatrix, sim::Int, obsPath, u)
    if isnan(payoffValues[1, sim])
        throw(ErrorException("NaN value, obsPath=$obsPath, u=$u"))
    end
    return nothing
end

@inline function _check_nan(payoffValues::AbstractVector, sim::Int, obsPath, u)
    if isnan(payoffValues[sim])
        throw(ErrorException("NaN value, obsPath=$obsPath, u=$u"))
    end
    return nothing
end

function simulateAlfonsi2Iter(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
) where {T}
    config = GridConfig(model, specificTimes(payoff), timestepSize, Alfonsi2(withBB=withBB))
    return simulateIter(rng, model, spot, payoff, start, nSim, config; withBB=withBB)
end



# ============================================================
# Common vectorized engine for HestonModel + MCPayoff schemes
# ============================================================

struct CommonSimData{T}
    specTimes::Vector{Float64}
    genTimes::Vector{Float64}
    h::Vector{Float64}
    sqrth::Vector{Float64}
    lnf::Vector{T}
    obsIdx::Vector{Int}
    df::Float64
    ρBar::T
    nSteps::Int
    nObs::Int
end

function _common_sim_data(
    model::StochasticVolatilityModel{T},
    spot::Float64,
    payoff, #unused, the observation times are those of the config
    config::GridConfig,
) where {T}
    specTimes = config.specTimes
    genTimes = config.genTimes
    h = genTimes[2:end] .- genTimes[1:(end-1)]
    sqrth = sqrt.(h)
    lnspot = log(spot)
    lnf = map(t -> logForward(model, lnspot, t), genTimes)
    obsIdx = _obs_indices(genTimes, specTimes)
    df = discountFactor(model, specTimes[end])
    ρBar = sqrt(1 - model.ρ^2)
    nSteps = length(h)
    nObs = length(specTimes)
    return CommonSimData(specTimes, genTimes, h, sqrth, lnf, obsIdx, df, ρBar, nSteps, nObs)
end

@inline function _record_observation_vectorized!(
    obsPath::AbstractMatrix,
    nextObs::Int,
    obsIdx::AbstractVector{Int},
    i::Int,
    logpathValues,
)
    if nextObs <= length(obsIdx) && i == obsIdx[nextObs]
        @views @. obsPath[:, nextObs] = exp(logpathValues)
        return nextObs + 1
    end
    return nextObs
end

@inline function _allocate_payoff_values(payoff, nSim::Int)
    dimOut = dimensionOut(payoff)[1]
    return dimOut == 1 ? zeros(nSim) : zeros(dimOut, nSim)
end

@inline function _finalize_payoffs!(payoffValues, payoff, obsPath, df, nSim::Int)
    @inbounds for sim in 1:nSim
        _store_payoff!(payoffValues, sim, payoff, @view(obsPath[sim, :]), df)
    end
    return nothing
end

bbcachetype(::FullTruncation) = Array{Float64,2}
bbcachetype(::IVI) = Array{Float64,2}
bbcachetype(::QE) = Vector{Float64}
bbcachetype(::DVSS2X) = Vector{Float64}

function _make_bbcache(method::PathGenerator, cacheSize::Int)
    TT = bbcachetype(method)
    return BBCache{Int,TT}(cacheSize)
end

# ============================================================
# Scheme-specific state / precalc
# ============================================================

struct FTState{T,PV}
    logpathValues::Vector{T}
    v::Vector{T}
    obsPath::Matrix{T}
    payoffValues::PV
    u::Matrix{Float64}
    sqrtmv::Vector{T}
end

struct QEState{T,PV}
    logpathValues::Vector{T}
    v::Vector{T}
    obsPath::Matrix{T}
    payoffValues::PV
    z1::Vector{Float64}
    u2::Vector{Float64}
end

struct IVIState{T,PV}
    logpathValues::Vector{T}
    v::Vector{T}
    obsPath::Matrix{T}
    payoffValues::PV
    u::Matrix{Float64}
    uuni::Vector{Float64}
    α::Vector{Float64}
    uig::Vector{Float64}
    zig::Vector{Float64}
end

struct DVSS2XState{T,PV}
    logpathValues::Vector{T}
    v::Vector{T}
    obsPath::Matrix{T}
    payoffValues::PV
    xsi::Vector{Float64}
    u2::Vector{Float64}
    vTmp::Vector{Float64}
    yr::Vector{Float64}
end
function init_scheme_state(::FullTruncation, ::Type{T}, payoff, nSim::Int, nObs::Int) where {T}
    return FTState(
        Vector{T}(undef, nSim),
        Vector{T}(undef, nSim),
        Matrix{T}(undef, nSim, nObs),
        _allocate_payoff_values(payoff, nSim),
        Matrix{Float64}(undef, nSim, 2),
        Vector{T}(undef, nSim),
    )
end

function init_scheme_state(::QE, ::Type{T}, payoff, nSim::Int, nObs::Int) where {T}
    return QEState(
        Vector{T}(undef, nSim),
        Vector{T}(undef, nSim),
        Matrix{T}(undef, nSim, nObs),
        _allocate_payoff_values(payoff, nSim),
        Vector{Float64}(undef, nSim),
        Vector{Float64}(undef, nSim),
    )
end

function init_scheme_state(::IVI, ::Type{T}, payoff, nSim::Int, nObs::Int) where {T}
    return IVIState(
        Vector{T}(undef, nSim),
        Vector{T}(undef, nSim),
        Matrix{T}(undef, nSim, nObs),
        _allocate_payoff_values(payoff, nSim),
        Matrix{Float64}(undef, nSim, 2),
        Vector{Float64}(undef, nSim),
        Vector{Float64}(undef, nSim),
        Vector{Float64}(undef, nSim),
        Vector{Float64}(undef, nSim),
    )
end

function init_scheme_state(::DVSS2X, ::Type{T}, payoff, nSim::Int, nObs::Int) where {T}
    return DVSS2XState(
        Vector{T}(undef, nSim),
        Vector{T}(undef, nSim),
        Matrix{T}(undef, nSim, nObs),
        _allocate_payoff_values(payoff, nSim),
        Vector{Float64}(undef, nSim),
        Vector{Float64}(undef, nSim),
        Vector{Float64}(undef, nSim),
        Vector{Float64}(undef, nSim),
    )
end

@inline function reset_state!(st::FTState, model::HestonModel, lnf0)
    fill!(st.logpathValues, lnf0)
    fill!(st.v, model.v0)
    return nothing
end

@inline function reset_state!(st::QEState, model::HestonModel, lnf0)
    fill!(st.logpathValues, lnf0)
    fill!(st.v, model.v0)
    return nothing
end

@inline function reset_state!(st::IVIState, model::HestonModel, lnf0)
    fill!(st.logpathValues, lnf0)
    fill!(st.v, model.v0)
    return nothing
end

@inline function reset_state!(st::DVSS2XState, model::HestonModel, lnf0)
    fill!(st.logpathValues, lnf0)
    fill!(st.v, model.v0)
    return nothing
end

struct FTPrecalc end
precompute_scheme(::FullTruncation, model::HestonModel, common::CommonSimData) = FTPrecalc()



struct QEPrecalc{T}
    ekdt::Vector{T}
    A::Vector{T}
    B::Vector{T}
    C::Vector{T}
    psiC::T
end

function precompute_scheme(method::QE, model::HestonModel{T}, common::CommonSimData{T}) where {T}
    h = common.h
    ekdt = @. exp(-model.κ * h)
    A = @. h / 2 * (model.κ * model.ρ / model.σ - 1 / 2) + model.ρ / model.σ + h / 4 * (1 - model.ρ^2)
    B = @. h / 2 * (model.κ * model.ρ / model.σ - 1 / 2) - model.ρ / model.σ + h / 4 * (1 - model.ρ^2)
    C = @. model.ρ / model.σ * model.κ * model.θ * h
    return QEPrecalc(ekdt, A, B, C, T(1.5))
end



struct IVIPrecalc{T}
    ebm1::Vector{T}
    a::T
    b::T
    c::T
end

function precompute_scheme(::IVI, model::HestonModel{T}, common::CommonSimData{T}) where {T}
    a = model.κ * model.θ
    b = -model.κ
    c = model.σ
    ebm1 = @. (exp(b * common.h) - 1) / b
    return IVIPrecalc(ebm1, a, b, c)
end



struct DVSS2XPrecalc{T}
    ekdth::Vector{T}
end

function precompute_scheme(::DVSS2X, model::HestonModel{T}, common::CommonSimData{T}) where {T}
    return DVSS2XPrecalc(@. exp(-model.κ * common.h / 2))
end

# ============================================================
# Variate loading
# ============================================================

function load_step_variates!(
    st::FTState{T,PV},
    ::FullTruncation,
    rng,
    start::Int,
    dim::Int,
    common::CommonSimData,
    withBB::Bool,
    bb,
    cache,
) where {T,PV}
    u = st.u
    if withBB
        transformByDim!(bb, rng, start, dim, u, cache)
    else
        @views begin
            u1 = u[:, 1]
            u2 = u[:, 2]
            d = 2 * dim - 1
            skipTo!(rng, d, start, length(u1))
            nextn!(rng, d, u1)
            skipTo!(rng, d + 1, start, length(u2))
            nextn!(rng, d + 1, u2)
            @. u1 *= common.sqrth[dim]
            @. u2 *= common.sqrth[dim]
        end
    end
    return nothing
end

function load_step_variates!(
    st::QEState{T,PV},
    ::QE,
    rng,
    start::Int,
    dim::Int,
    common::CommonSimData,
    withBB::Bool,
    bb,
    cache,
) where {T,PV}
    if withBB
        transformByDim!(bb, rng, start, dim, st.z1, cache)
        skipTo!(rng, dim + common.nSteps, start, length(st.u2))
        next!(rng, dim + common.nSteps, st.u2)
    else
        d = 2 * dim - 1
        skipTo!(rng, d, start, length(st.z1))
        nextn!(rng, d, st.z1)
        @. st.z1 *= common.sqrth[dim]
        skipTo!(rng, d + 1, start, length(st.u2))
        next!(rng, d + 1, st.u2)
    end
    return nothing
end

function load_step_variates!(
    st::IVIState{T,PV},
    ::IVI,
    rng,
    start::Int,
    dim::Int,
    common::CommonSimData,
    withBB::Bool,
    bb,
    cache,
) where {T,PV}
    u = st.u
    if withBB
        transformByDim!(bb, rng, start, dim, u, cache)
        skipTo!(rng, 2 * common.nSteps + dim, start, length(st.uuni))
        next!(rng, 2 * common.nSteps + dim, st.uuni)
        invsqrth = inv(common.sqrth[dim])
        @views begin
            xsi = u[:, 1]
            xsi2 = u[:, 2]
            @. xsi *= invsqrth
            @. xsi2 *= invsqrth
        end
    else
        @views begin
            xsi = u[:, 1]
            xsi2 = u[:, 2]
            d = 3 * dim - 2
            skipTo!(rng, d, start, length(xsi))
            nextn!(rng, d, xsi)
            skipTo!(rng, d + 1, start, length(st.uuni))
            next!(rng, d + 1, st.uuni)
            skipTo!(rng, d + 2, start, length(xsi2))
            nextn!(rng, d + 2, xsi2)
        end
    end
    return nothing
end

function load_step_variates!(
    st::DVSS2XState{T,PV},
    ::DVSS2X,
    rng,
    start::Int,
    dim::Int,
    common::CommonSimData,
    withBB::Bool,
    bb,
    cache,
) where {T,PV}
    if withBB
        transformByDim!(bb, rng, start, dim, st.xsi, cache)
        skipTo!(rng, dim + common.nSteps, start, length(st.u2))
        next!(rng, dim + common.nSteps, st.u2)
    else
        d = 2 * dim - 1
        skipTo!(rng, d, start, length(st.xsi))
        nextn!(rng, d, st.xsi)
        @. st.xsi *= common.sqrth[dim]
        skipTo!(rng, d + 1, start, length(st.u2))
        next!(rng, d + 1, st.u2)
    end
    return nothing
end

# ============================================================
# Step kernels
# ============================================================

function step!(
    st::FTState{T,PV},
    ::FullTruncation,
    model::HestonModel{T},
    common::CommonSimData{T},
    ::FTPrecalc,
    dim::Int,
) where {T,PV}
    h = common.h[dim]
    lndrift = common.lnf[dim+1] - common.lnf[dim]
    @views begin
        u1 = st.u[:, 1]
        u2 = st.u[:, 2]
        @. st.sqrtmv = sqrt(max(st.v, 0))
        @. st.logpathValues += lndrift - 0.5 * st.sqrtmv^2 * h + st.sqrtmv * (u1 * common.ρBar + u2 * model.ρ)
        @. st.v += model.κ * (model.θ - st.sqrtmv^2) * h + model.σ * st.sqrtmv * u2
    end
    return nothing
end

function step!(
    st::QEState{T,PV},
    method::QE,
    model::HestonModel{T},
    common::CommonSimData{T},
    precalc::QEPrecalc{T},
    dim::Int,
) where {T,PV}
    h = common.h[dim]
    lndrift = common.lnf[dim+1] - common.lnf[dim]
    ekdt = precalc.ekdt[dim]
    A = precalc.A[dim]
    B = precalc.B[dim]
    C = precalc.C[dim]
    psiC = precalc.psiC

    κ = model.κ
    θ = model.θ
    σ = model.σ
    ρ = model.ρ
    ρBar = common.ρBar

    @inbounds for j in eachindex(st.v)
        vold = st.v[j]
        m = θ + (vold - θ) * ekdt
        s2 = vold * σ^2 * ekdt / κ * (1 - ekdt) + (θ * σ^2) / (2 * κ) * (1 - ekdt)^2
        psi = s2 / (m * m)

        vnew = zero(T)
        mcorr = zero(T)

        if psi <= psiC
            psiInv = inv(psi)
            b2 = 2 * psiInv - 1 + sqrt(2 * psiInv * (2 * psiInv - 1))
            a = m / (1 + b2)
            b = sqrt(b2)
            zv = norminv(st.u2[j])
            vnew = a * (b + zv)^2
            if method.withMartingale
                mcorr = C - A * b2 * a / (1 - 2 * A * a) + 0.5 * log(1 - 2 * A * a) - B * vold
            end
        else
            p = (psi - 1) / (psi + 1)
            invBeta = m / (1 - p)
            vnew = st.u2[j] <= p ? zero(T) : log((1 - p) / (1 - st.u2[j])) * invBeta
            if method.withMartingale
                mcorr = C - log(p + (1 - p) / (1 - invBeta * A)) - B * vold
            end
        end

        vIntegral = (vnew + vold) / 2
        st.logpathValues[j] += lndrift +
                               ρ / σ * (vnew - vold - κ * θ * h) +
                               (κ * ρ / σ - 1 / 2) * vIntegral * h +
                               ρBar * sqrt(vIntegral) * st.z1[j] + mcorr
        st.v[j] = vnew
    end
    return nothing
end

function step!(
    st::IVIState{T,PV},
    ::IVI,
    model::HestonModel{T},
    common::CommonSimData{T},
    precalc::IVIPrecalc{T},
    dim::Int,
) where {T,PV}
    h = common.h[dim]
    lndrift = common.lnf[dim+1] - common.lnf[dim]
    ebm1 = precalc.ebm1[dim]
    a = precalc.a
    b = precalc.b
    c = precalc.c
    aFactor = a / b * (ebm1 - h)
    σFactor = c * ebm1
    @views begin
        xsi = st.u[:, 1]
        xsi2 = st.u[:, 2]
        @. st.α = st.v * ebm1 + aFactor
        @. st.uig = inverseGaussian(st.α, (st.α / σFactor)^2, xsi, st.uuni)
        @. st.zig = (st.uig - st.α) / σFactor
        @. st.v += a * h + b * st.uig + c * st.zig
        @. st.logpathValues += lndrift - st.uig / 2 + model.ρ * st.zig + common.ρBar * sqrt(st.uig) * xsi2
    end
    return nothing
end

function step!(
    st::DVSS2XState{T,PV},
    ::DVSS2X,
    model::HestonModel{T},
    common::CommonSimData{T},
    precalc::DVSS2XPrecalc{T},
    dim::Int,
) where {T,PV}
    h = common.h[dim]
    lndrift = common.lnf[dim+1] - common.lnf[dim]
    ekdth = precalc.ekdth[dim]
    c0 = model.θ * h / 4
    c1 = (ekdth - 1) / (model.κ * 2)
    c2 = model.θ * (1 - ekdth)
    σ = model.σ
    σ2 = σ^2

    @. st.vTmp = (st.v * ekdth + c2) / σ2
    @inbounds for j in eachindex(st.vTmp)
        st.yr[j] = st.vTmp[j] > 1.896 * h ? dvss2_case1(st.vTmp[j], st.u2[j], h) : dvss2_case2(st.vTmp[j], st.u2[j], h)
    end
    @. st.logpathValues =
        lndrift +
        (st.logpathValues - c0 + c1 * (st.v - model.θ) +
         σ * (common.ρBar * st.xsi * sqrt((st.vTmp + st.yr) / 2) + model.ρ * (st.yr - st.vTmp))) -
        c0 + c1 * (σ2 * st.yr - model.θ)
    @. st.v = (σ2 * st.yr) * ekdth + c2
    return nothing
end

# ============================================================
# Generic driver
# ============================================================

function simulateScheme(
    rng,
    model::StochasticVolatilityModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{S};
    withBB=config.method.withBB,
    cacheSize=0,
) where {T,S<:PathGenerator}
    common = _common_sim_data(model, spot, payoff, config)
    method = config.method
    precalc = precompute_scheme(method, model, common)
    st = init_scheme_state(method, T, payoff, nSim, common.nObs)
    reset_state!(st, model, common.lnf[1])
    _evolve_scheme!(st, rng, start, method, model, common, precalc, withBB, cacheSize)
    _finalize_payoffs!(st.payoffValues, payoff, st.obsPath, common.df, nSim)
    return mc_convergence_detailed(st.payoffValues)
end

#Brownian bridge on the simulation grid and its cache, or nothing.
function _make_bb(method::PathGenerator, common::CommonSimData, withBB::Bool, cacheSize::Int)
    withBB || return nothing, nothing
    if cacheSize == 0
        cacheSize = ceil(Int, log2(common.nSteps)) * 4
    end
    return BrownianBridgeConstruction(common.genTimes[2:end]), _make_bbcache(method, cacheSize)
end

#evolves all the paths of the state over the whole grid, recording the observations.
function _evolve_scheme!(st, rng, start::Int, method::PathGenerator, model, common::CommonSimData, precalc, withBB::Bool, cacheSize::Int)
    bb, cache = _make_bb(method, common, withBB, cacheSize)
    nextObs = 1
    for dim in 1:common.nSteps
        load_step_variates!(st, method, rng, start, dim, common, withBB, bb, cache)
        step!(st, method, model, common, precalc, dim)
        nextObs = _record_observation_vectorized!(st.obsPath, nextObs, common.obsIdx, dim, st.logpathValues)
    end
    return nothing
end

# ============================================================
# Public vectorized wrappers
# ============================================================

function simulateFullTruncation(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{FullTruncation};
    withBB=config.method.withBB,
    cacheSize=0,
) where {T}
    return simulateScheme(rng, model, spot, payoff, start, nSim, config; withBB=withBB, cacheSize=cacheSize)
end

function simulateQE(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{QE};
    withBB=config.method.withBB,
    cacheSize=0,
) where {T}
    return simulateScheme(rng, model, spot, payoff, start, nSim, config; withBB=withBB, cacheSize=cacheSize)
end

function simulateIVI(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{IVI};
    withBB=config.method.withBB,
    cacheSize=0,
) where {T}
    return simulateScheme(rng, model, spot, payoff, start, nSim, config; withBB=withBB, cacheSize=cacheSize)
end

function simulateDVSS2X(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{DVSS2X};
    withBB=config.method.withBB,
    cacheSize=0,
) where {T}
    return simulateScheme(rng, model, spot, payoff, start, nSim, config; withBB=withBB, cacheSize=cacheSize)
end

# ============================================================
# Convenience overloads
# ============================================================

function simulateIVI(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
) where {T}
    config = GridConfig(model, specificTimes(payoff), timestepSize, IVI(withBB=withBB))
    return simulateIVI(rng, model, spot, payoff, start, nSim, config; withBB=withBB, cacheSize=cacheSize)
end

function simulateDVSS2X(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
) where {T}
    config = GridConfig(model, specificTimes(payoff), timestepSize, DVSS2X(withBB=withBB); mindt=0.5 / 365, tstop=0.75, maxdt=timestepSize)
    return simulateDVSS2X(rng, model, spot, payoff, start, nSim, config; withBB=withBB, cacheSize=cacheSize)
end

function simulateFullTruncation(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    cacheSize=0,
) where {T}
    config = GridConfig(model, specificTimes(payoff), timestepSize, FullTruncation(withBB=withBB))
    return simulateFullTruncation(rng, model, spot, payoff, start, nSim, config; withBB=withBB, cacheSize=cacheSize)
end

function simulateQE(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
    withMartingale=true,
    cacheSize=0,
) where {T}
    config = GridConfig(model, specificTimes(payoff), timestepSize, QE(withBB=withBB, withMartingale=withMartingale))
    return simulateQE(rng, model, spot, payoff, start, nSim, config; withBB=withBB, cacheSize=cacheSize)
end



using Base.Threads

@inline function _block_bounds(tid::Int, n::Int, nt::Int)
    q, r = divrem(n, nt)
    lo = (tid - 1) * q + min(tid - 1, r) + 1
    hi = lo + q - 1 + (tid <= r ? 1 : 0)
    return lo, hi
end

@inline function _as_vectors(vals, errs)
    if vals isa Number
        return [Float64(vals)], [Float64(errs)]
    else
        return Float64.(vals), Float64.(errs)
    end
end

@inline function _clone_rng(rng)
    return deepcopy(rng) #copy may share the scrambling generator between threads
end

@inline function _reduce_block_results(sum1::AbstractVector, sum2::AbstractVector, nTotal::Int)
    n = Float64(nTotal)
    meanv = sum1 ./ n
    if length(meanv) == 1
        varv = max(sum2[1] / n - meanv[1]^2, 0.0)
        return meanv[1], sqrt(varv / n)
    else
        stderr = similar(meanv)
        @inbounds for k in eachindex(meanv)
            varv = max(sum2[k] / n - meanv[k]^2, 0.0)
            stderr[k] = sqrt(varv / n)
        end
        return collect(meanv), collect(stderr)
    end
end

function simulateParallelReuse(
    simfun,
    rng,
    model,
    spot,
    payoff,
    start::Int,
    nSim::Int,
    config;
    kwargs...
)
    nt = Threads.nthreads()
    part_sum = Vector{Vector{Float64}}(undef, nt)
    part_sum2 = Vector{Vector{Float64}}(undef, nt)

    Threads.@threads :static for tid in 1:nt
        lo, hi = _block_bounds(tid, nSim, nt)
        if hi < lo
            part_sum[tid] = Float64[]
            part_sum2[tid] = Float64[]
            continue
        end

        nLocal = hi - lo + 1
        startLocal = start + lo - 1
        rngLocal = _clone_rng(rng)

        result = simfun(
            rngLocal,
            model,
            spot,
            deepcopy(payoff), #payoffs keep observation buffers: one copy per thread
            startLocal,
            nLocal,
            config;
            kwargs...
        )
        #simulate*Iter return MCConvergenceResult (one per payoff): keep the estimate on all the paths
        vals, errs = result isa Tuple ? result : last_stat(result)

        v, e = _as_vectors(vals, errs)

        ex2 = similar(v)
        @inbounds for k in eachindex(v)
            ex2[k] = v[k]^2 + e[k]^2 * nLocal
        end

        part_sum[tid] = nLocal .* v
        part_sum2[tid] = nLocal .* ex2
    end

    first_nonempty = findfirst(!isempty, part_sum)
    total1 = zeros(Float64, length(part_sum[first_nonempty]))
    total2 = zeros(Float64, length(part_sum2[first_nonempty]))

    for tid in 1:nt
        isempty(part_sum[tid]) && continue
        total1 .+= part_sum[tid]
        total2 .+= part_sum2[tid]
    end

    return _reduce_block_results(total1, total2, nSim)
end

function simulateIterParallel(
    rng,
    model::HestonModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{M};
    withBB=config.method.withBB,
) where {T,M<:PathGenerator}
    return simulateParallelReuse(
        simulateIter,
        rng, model, spot, payoff, start, nSim, config;
        withBB=withBB,
    )
end

function simulateSchemeParallel(
    rng,
    model::StochasticVolatilityModel{T},
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{M};
    withBB=config.method.withBB,
    cacheSize::Int=0,
) where {T,M<:PathGenerator}
    return simulateParallelReuse(
        simulateScheme,
        rng, model, spot, payoff, start, nSim, config;
        withBB=withBB,
        cacheSize=cacheSize,
    )
end#=== LV ==#




import AQFED.TermStructure:
    LocalVolatilityModel, VarianceSurfaceBySection, localVarianceByLogmoneyness, discountFactor, logForward

function GridConfig(
    model::LocalVolatilityModel,
    specTimes::Vector{Float64},
    timestepSize::Float64,
    method::PathGenerator;
    mindt::Float64=timestepSize,
    tstop::Float64=0.0,
    maxdt::Float64=timestepSize,
)
    genTimes = pathgenTimesRamp(specTimes, mindt, tstop, maxdt)
    return GridConfig(specTimes, mindt, tstop, maxdt, genTimes, method)
end

import AQFED.TermStructure:
    LocalVolatilityModel,
    localVarianceByLogmoneyness,
    discountFactor,
    logForward

import AQFED.Random: next!, nextn!, skipTo!
using Statistics

function simulateIter(
    rng,
    model::LocalVolatilityModel,
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{FullTruncation};
    withBB::Bool=config.method.withBB,
    isRestart::Bool=true
)
    specTimes = config.specTimes
    tte = specTimes[end]
    df = discountFactor(model, tte)
    genTimes = config.genTimes
    h = genTimes[2:end] .- genTimes[1:(end-1)]
    sqrth = sqrt.(h)

    obsIdx = _obs_indices(genTimes, specTimes)
    df = discountFactor(model, specTimes[end])
    nSteps = length(h)

    local bb
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
    end

    lnspot = log(spot)
    lnf = map(t -> logForward(model, lnspot, t), genTimes)

    if isRestart
        skipTo!(rng, start)
    end
    dimOut = dimensionOut(payoff)[1]
    payoffValues = dimOut == 1 ? zeros(nSim) : zeros(dimOut, nSim)
    obsPath = zeros(length(specTimes))
    z = Vector{Float64}(undef, length(genTimes) - 1)


    for sim in 1:nSim
        logpath = lnf[1]
        # Observation at t=0 if present

        # Generate one Gaussian path: one variate per time step

        if withBB
            nextn!(rng, z)
            transform!(bb, z, z)  # in-place
        else
            nextn!(rng, z)
            z .*= sqrth
        end
        nextObs = 1
        nextObs = _record_observation!(obsPath, nextObs, obsIdx, 0, logpath)


        # Time stepping
        @inbounds for i in 1:(length(genTimes)-1)
            t0 = genTimes[i]
            t1 = genTimes[i+1]
            hi = h[i]

            # Local variance at current log-moneyness
            logmoneyness = logpath - lnf[i+1]
            lv = localVarianceByLogmoneyness(model.surface, logmoneyness, t0, t1)

            # Euler–Maruyama:
            # dX = (forward drift) dt - ½ σ² dt + σ dW
            logpath +=
                lnf[i+1] - lnf[i] -
                0.5 * lv * hi +
                sqrt(lv) * z[i]

            nextObs = _record_observation!(
                obsPath, nextObs, obsIdx, i, logpath
            )
        end

        # Payoff
        _store_payoff!(payoffValues, sim, payoff, obsPath, df)
        _check_nan(payoffValues, sim, obsPath, z)
    end

    return mc_convergence_detailed(payoffValues)
end

function simulateScheme(
    rng,
    model::LocalVolatilityModel,
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{T};
    withBB::Bool=config.method.withBB,
) where {T <: Union{LogEuler, FullTruncation}}
    specTimes = config.specTimes
    genTimes = config.genTimes
    h = genTimes[2:end] .- genTimes[1:(end-1)]
    sqrth = sqrt.(h)

    obsIdx = _obs_indices(genTimes, specTimes)
    df = discountFactor(model, specTimes[end])
    nSteps = length(h)
    nObs = length(specTimes)

    local bb, cache

    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
        cacheSize = max(1, ceil(Int, log2(nSteps)) * 4)
        cache = BBCache{Int,Array{Float64,2}}(cacheSize)
    else
        bb = nothing
        cache = nothing
    end

    lnspot = log(spot)
    lnf = map(t -> logForward(model, lnspot, t), genTimes)

    # State
    logpath = fill(lnf[1], nSim)
    obsPath = Matrix{Float64}(undef, nSim, nObs)
    z = Vector{Float64}(undef, nSim)

    dimOut = dimensionOut(payoff)[1]
    payoffValues = dimOut == 1 ? zeros(nSim) : zeros(dimOut, nSim)

    nextObs = 1

    # Observation at t=0 if present
    if nextObs <= nObs && obsIdx[nextObs] == 0
        @. obsPath[:, nextObs] = exp(logpath)
        nextObs += 1
    end

    for i in 1:nSteps
        t0 = genTimes[i]
        t1 = genTimes[i+1]
        hi = h[i]

        # Load Gaussian variates for this step
        if withBB
            # transformByDim! expects an nSim × 1 matrix or similar
            u = reshape(z, nSim, 1)
            transformByDim!(bb, rng, start, i, u, cache)
            z .= u[:, 1]
        else
            d = i
            skipTo!(rng, d, start, length(z))
            nextn!(rng, d, z)
            z .*= sqrth[i]
        end

        # Local variance: vectorized over paths
        logmoneyness = @. logpath - lnf[i+1]
        lv = localVarianceByLogmoneyness(model.surface, logmoneyness, t0, t1)

        @. logpath +=
            lnf[i+1] - lnf[i] -
            0.5 * lv * hi +
            sqrt(lv) * z

        # Record observations
        nextObs = _record_observation_vectorized!(
            obsPath, nextObs, obsIdx, i, logpath
        )
    end

    # Evaluate payoffs
    @inbounds for sim in 1:nSim
        _store_payoff!(
            payoffValues,
            sim,
            payoff,
            @view(obsPath[sim, :]),
            df,
        )
    end

    return mc_convergence_detailed(payoffValues)
end

function simulateIter(
    rng,
    model::LocalVolatilityModel,
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB::Bool=false,
)
    specTimes = specificTimes(payoff)
    config = GridConfig(
        model,
        specTimes,
        timestepSize,
        FullTruncation(withBB=withBB),
    )
    return simulateIter(rng, model, spot, payoff, start, nSim, config; withBB=withBB)
end

function simulateScheme(
    rng,
    model::LocalVolatilityModel,
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB::Bool=false,
)
    specTimes = specificTimes(payoff)
    config = GridConfig(
        model,
        specTimes,
        timestepSize,
        FullTruncation(withBB=withBB),
    )
    return simulateScheme(rng, model, spot, payoff, start, nSim, config; withBB=withBB)
end#==== Black TS =====#



function GridConfig(
    model::TSBlackModel,
    specTimes::Vector{Float64},
    timestepSize::Float64,
    method::PathGenerator;
    mindt::Float64=timestepSize,
    tstop::Float64=0.0,
    maxdt::Float64=timestepSize,
)
    genTimes = pathgenTimesRamp(specTimes, mindt, tstop, maxdt)
    return GridConfig(
        specTimes,
        mindt,
        tstop,
        maxdt,
        genTimes,
        method,
    )
end



LogEuler(; withBB::Bool=false) = LogEuler(withBB)

variatesPerStep(::LocalVolatilityModel, ::LogEuler) = 1
variatesPerStep(::LocalVolatilityModel, ::FullTruncation) = 1


struct BlackSimData{T}
    specTimes::Vector{Float64}
    genTimes::Vector{Float64}
    h::Vector{Float64}
    sqrth::Vector{Float64}
    logForward::Vector{T}
    variance::Vector{T}
    volatility::Vector{T}
    obsIdx::Vector{Int}
    df::Float64
    nSteps::Int
    nObs::Int
end

function _black_sim_data(
    model::TSBlackModel{S},
    spot::Float64,
    payoff::VanillaOption,
    config::GridConfig,
) where {S}
    specTimes = config.specTimes
    genTimes = config.genTimes

    h = genTimes[2:end] .- genTimes[1:(end-1)]
    sqrth = sqrt.(h)

    lnspot = log(spot)
    lnf = S.(map(t -> logForward(model, lnspot, t), genTimes))

    # Total variance at the grid points.
    totalVariance = S.([
        varianceByLogmoneyness(model.surface, 0.0, t) * t
        for t in genTimes
    ])

    variance = similar(totalVariance)
    volatility = similar(totalVariance)

    @inbounds for i in eachindex(h)
        vi = (totalVariance[i+1] - totalVariance[i]) / h[i]

        if !isfinite(vi) || vi < zero(S)
            throw(DomainError(
                vi,
                "Invalid Black forward variance at step $i",
            ))
        end

        variance[i] = vi
        volatility[i] = sqrt(vi)
    end

    # The last element is unused but makes indexing by grid point convenient.
    variance[end] = zero(S)
    volatility[end] = zero(S)

    obsIdx = _obs_indices(genTimes, specTimes)
    df = discountFactor(model, specTimes[end])

    return BlackSimData(
        specTimes,
        genTimes,
        h,
        sqrth,
        lnf,
        variance,
        volatility,
        obsIdx,
        df,
        length(h),
        length(specTimes),
    )
end

function simulateScheme(
    rng,
    model::TSBlackModel{S},
    spot::Float64,
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    config::GridConfig{LogEuler};
    withBB::Bool=config.method.withBB,
    cacheSize::Int=0,
) where {S}
    data = _black_sim_data(model, spot, payoff, config)

    logpath = fill(data.logForward[1], nSim)
    z = Vector{Float64}(undef, nSim)
    obsPath = Matrix{Float64}(undef, nSim, data.nObs)

    dimOut = dimensionOut(payoff)[1]

    payoffValues =
        dimOut == 1 ?
        zeros(Float64, nSim) :
        zeros(Float64, dimOut, nSim)

    local bb, cache

    if withBB
        bb = BrownianBridgeConstruction(data.genTimes[2:end])

        if cacheSize == 0
            cacheSize = max(
                1,
                ceil(Int, log2(data.nSteps)) * 4,
            )
        end

        cache = BBCache{Int,Vector{Float64}}(cacheSize)
    else
        bb = nothing
        cache = nothing
    end

    nextObs = 1

    if nextObs <= data.nObs &&
       data.obsIdx[nextObs] == 0
        @views obsPath[:, nextObs] .= exp.(logpath)
        nextObs += 1
    end

    for i in 1:data.nSteps
        if withBB
            transformByDim!(
                bb,
                rng,
                start,
                i,
                z,
                cache,
            )
        else
            skipTo!(rng, i, start, length(z))
            nextn!(rng, i, z)
            @. z *= data.sqrth[i]
        end

        @. logpath +=
            data.logForward[i+1] -
            data.logForward[i] -
            0.5 * data.variance[i] * data.h[i] +
            data.volatility[i] * z

        if nextObs <= data.nObs &&
           i == data.obsIdx[nextObs]
            @views obsPath[:, nextObs] .= exp.(logpath)
            nextObs += 1
        end
    end

    @inbounds for sim in 1:nSim
        path = @view obsPath[sim, :]

        if dimOut == 1
            payoffValues[sim] =
                evaluatePayoffOnPath(payoff, path, data.df)
        else
            @views payoffValues[:, sim] .=
                evaluatePayoffOnPath(payoff, path, data.df)
        end
    end

    return mc_convergence_detailed(payoffValues)
end

function simulateIter(
    rng,
    model::TSBlackModel{S},
    spot::Float64,
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    config::GridConfig{LogEuler};
    withBB::Bool=config.method.withBB,
    isRestart::Bool=true
) where {S}
    data = _black_sim_data(model, spot, payoff, config)

    local bb, cache

    if withBB
        bb = BrownianBridgeConstruction(data.genTimes[2:end])

        cacheSize = max(
            1,
            ceil(Int, log2(data.nSteps)) * 4,
        )

        cache = BBCache{Int,Vector{Float64}}(cacheSize)
    else
        bb = nothing
        cache = nothing
    end

    dimOut = dimensionOut(payoff)[1]

    payoffValues =
        dimOut == 1 ?
        zeros(Float64, nSim) :
        zeros(Float64, dimOut, nSim)

    z = Vector{Float64}(undef, data.nSteps)
    obsPath = Vector{Float64}(undef, data.nObs)

    # Each path consumes nSteps Sobol dimensions.
    if isRestart
        skipTo!(rng, start)
    end

    @inbounds for sim in 1:nSim
        logpath = data.logForward[1]
        nextObs = 1

        fill!(obsPath, 0.0)

        if nextObs <= data.nObs &&
           data.obsIdx[nextObs] == 0
            obsPath[nextObs] = exp(logpath)
            nextObs += 1
        end

        # One complete Gaussian vector for this path.
        nextn!(rng, z)

        if withBB
            transform!(bb, z, z)
        else
            @simd for i in 1:data.nSteps
                z[i] *= data.sqrth[i]
            end
        end

        for i in 1:data.nSteps
            logpath +=
                data.logForward[i+1] -
                data.logForward[i] -
                0.5 * data.variance[i] * data.h[i] +
                data.volatility[i] * z[i]

            if nextObs <= data.nObs &&
               i == data.obsIdx[nextObs]
                obsPath[nextObs] = exp(logpath)
                nextObs += 1
            end
        end

        if dimOut == 1
            payoffValues[sim] =
                evaluatePayoffOnPath(payoff, obsPath, data.df)
        else
            @views payoffValues[:, sim] .=
                evaluatePayoffOnPath(payoff, obsPath, data.df)
        end
    end

    return mc_convergence_detailed(payoffValues)
end