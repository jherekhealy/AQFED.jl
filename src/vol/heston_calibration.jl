using CharFuncPricing, Random123, LinearAlgebra
import CharFuncPricing: makeCharFuncPricer, pricerMethod
using AQFED.Black, AQFED.GlobalOptimization, GaussNewton
import AQFED.Math:
    MQMinTransformation,
    LogisticTransformation,
    inv
export convertVolsToPricesOTMWeights, calibrateHestonFromPrices, makeCharFuncPricer
using QuadGK, Roots, SpecialFunctions
using DoubleExponentialFormulas
using FiniteDifferences
abstract type Reduction end
struct NoReduction <: Reduction
end

function expandX(r::NoReduction, ::Type, x)
    return x
end
function reduceX(r::NoReduction, ::Type, x)
    return x
end
struct SigmaKappaReduction{T} <: Reduction
    σ::T
    κ::T
end
function expandX(r::SigmaKappaReduction{T}, ::Type{HestonParams}, x::AbstractArray{TX}) where {T,TX}
    if !iszero(r.σ) && !iszero(r.κ)
        return [x[1], r.κ, x[2], x[3], r.σ]#v0::T, κ::T, θ::T, ρ::T, σ::T)
    elseif !iszero(r.σ)
        return [x[1], x[2], x[3], x[4], r.σ]
    elseif !iszero(r.κ)
        return [x[1], r.κ, x[2], x[3], x[4]]
    else
        return x
    end
end

function reduceX(r::SigmaKappaReduction{T}, ::Type{HestonParams}, x::AbstractArray{TX}) where {T,TX}
    if !iszero(r.σ) && !iszero(r.κ)
        return [x[1], x[3], x[4]]
    elseif !iszero(r.σ)
        return [x[1], x[2], x[3], x[4]]
    elseif !iszero(r.κ)
        return [x[1], x[3], x[4], x[5]]
    else
        return x
    end
end
struct V0KappaThetaReduction{T} <: Reduction
    v0::T
    κ::T
    θ::T
end
function expandX(r::V0KappaThetaReduction{T}, ::Type{HestonParams}, x::AbstractArray{TX}) where {T,TX}
    return [r.v0, r.κ, r.θ, x[1], x[2]] #v0::T, κ::T, θ::T, ρ::T, σ::T)
end
function reduceX(r::V0KappaThetaReduction{T}, ::Type{HestonParams}, x::AbstractArray{TX}) where {T,TX}
    return [x[4], x[5]]
end

struct V0EqualDoubleHestonReduction{T} <: Reduction
end
function expandX(r::V0EqualDoubleHestonReduction{T}, ::Type{DoubleHestonParams}, x::AbstractArray{TX}) where {T,TX}
    return vcat(x[1:5], x[1], x[6:9])
end
function reduceX(r::V0EqualDoubleHestonReduction{T}, ::Type{DoubleHestonParams}, x::AbstractArray{TX}) where {T,TX}
    return vcat(x[1:5], x[7:10])
end

struct V0KappaThetaDoubleHestonReduction{T} <: Reduction
    v01::T
    κ1::T
    θ1::T
    v02::T
    κ2::T
    θ2::T
end
function expandX(r::V0KappaThetaDoubleHestonReduction{T}, ::Type{DoubleHestonParams}, x::AbstractArray{TX}) where {T,TX}
    return [r.v01, r.κ1, r.θ1, x[1], x[2], r.v02, r.κ2, r.θ2, x[3], x[4]]
end
function reduceX(r::V0KappaThetaDoubleHestonReduction{T}, ::Type{DoubleHestonParams}, x::AbstractArray{TX}) where {T,TX}
    return [x[4], x[5], x[9], x[10]]
end

struct VVIXReduction{T} <: Reduction
    VVIX::T
end
using ForwardDiff
function expandX(r::VVIXReduction{T}, ::Type{HestonParams}, x::AbstractArray{TX}) where {T,TX}
    minBound = 0.15
    upperBound = 1.5
    # f(z) = VVIXLogContract(x[1],x[2],x[3],z)-r.VVIX
    # local sigma
    # try
    # sigma = Roots.find_zero(f, (minBound, upperBound), A42())	
    # catch e
    # 	sigma = 1e-6
    # end
    f(z) = VVIXApproxSimple(x[1], x[2], x[3], z) - r.VVIX
    D(f) = x -> ForwardDiff.derivative(f, float(x))
    sigma = Roots.find_zero((f, D(f)), r.VVIX / 200, Roots.Newton())
    return [x[1], x[2], x[3], x[4], sigma] #v0::T, κ::T, θ::T, ρ::T, σ::T)
end
function reduceX(r::VVIXReduction{T}, ::Type{HestonParams}, x::AbstractArray{TX}) where {T,TX}
    return [x[1], x[2], x[3], x[4]]
end


function VVIXApproxSimple(hParams::HestonParams{T}; dt=30 / 365) where {T}
    return VVIXApproxSimple(hParams.v0, hParams.κ, hParams.θ, hParams.σ, dt=dt)
end

function VVIXApproxSimple(v0, κ, θ, σ; dt=30 / 365)
    ekt = exp(-κ * dt)
    evT = (v0 - θ) * ekt + θ
    evsq = integratedVarianceExpectation(evT, κ, θ, dt=dt)
    evT2 = (σ^2 / 2κ + θ - (θ - v0 + σ^2 / (2κ)) * ekt)^2
    ektm1 = (ekt - 1) / (κ * dt)
    ev2 = -2 * ektm1 * evT * (θ * (ektm1 + 1)) + (θ * (ektm1 + 1))^2 + ektm1^2 * evT2
    varvix = (ev2 - evsq^2) / 4evsq
    vixapprox = sqrt(max(zero(typeof(σ)), log(1 + varvix / evsq) / dt)) * 100
    return vixapprox
end


function hestonDensityBK(hParams::HestonParams{T}, s; dt=30.0 / 365) where {T}
    d = 4 * hParams.κ * hParams.θ / hParams.σ^2
    lambda = 4 * hParams.κ * exp(-hParams.κ * dt) / (hParams.σ^2 * (1 - exp(-hParams.κ * dt))) * hParams.v0
    C1 = hParams.σ^2 * (1 - exp(-hParams.κ * dt)) / (4 * hParams.κ)
    C2 = @. exp(-(s + lambda) / 2) / 2 * (s / lambda)^(d / 4 - 1 / 2)
    return C1, C2 .* besseli.(d / 2 - 1, sqrt.(lambda * s))
end

function hestonDensity(hParams::HestonParams{T}, s; dt=30.0 / 365) where {T}
    return hestonDensity(hParams.v0, hParams.κ, hParams.θ, hParams.σ, s, dt=dt)
end

function hestonDensity(v0, κ, θ, σ, s; dt=30.0 / 365)
    q = 2 * κ * θ / σ^2 - 1
    c = 2 * κ / (σ^2 * (1 - exp(-κ * dt)))
    u = c * v0 * exp(-κ * dt)
    v = c .* s
    factor = c * exp.(-u - v) .* (v ./ u) .^ (q / 2)
    ifelse.(factor == 0, factor, factor * besseli(q, 2 * sqrt.(u * v)))
end

function hestonLogDensity(hParams::HestonParams{T}, logs; dt=30.0 / 365) where {T}
    return hestonLogDensity(hParams.v0, hParams.κ, hParams.θ, hParams.σ, logs, dt=dt)
end


function hestonLogDensity(v0, κ, θ, σ, logs; dt=30.0 / 365) #u=log(s), ds/s =  du
    q = 2 * κ * θ / σ^2 - 1
    c = 2 * κ / (σ^2 * (1 - exp(-κ * dt)))
    u = c * v0 * exp(-κ * dt)
    s = exp.(logs)
    v = c .* s
    factor = c * exp.(-u - v) .* (v ./ u) .^ (q / 2) .* s
    ifelse.(factor == 0, factor, factor * besseli(q, 2 * sqrt.(u * v)))
end

function vixIntrinsic(hParams::HestonParams, strike, signValue, x; dt=30 / 365)
    signValue * (sqrt(integratedVarianceExpectation(hParams, x, dt=dt)) - strike)
end
function vixIntrinsic(x, κ, θ, strike, signValue; dt=30 / 365)
    signValue * (sqrt(integratedVarianceExpectation(x, κ, θ, dt=dt)) - strike)
end
function vixCallIntegrand(hParams::HestonParams, strike, signValue, x; t=30 / 365, dt=30 / 365)
    d = hestonDensity(hParams, x, dt=t)
    max(vixIntrinsic(hParams, strike, signValue, x, dt=dt), 0) * d
end

function vixCallIntegrand(v0, κ, θ, σ, strike, signValue, x; t=30 / 365, dt=30 / 365)
    d = hestonDensity(v0, κ, θ, σ, x, dt=t)
    max(vixIntrinsic(x, κ, θ, strike, signValue, dt=dt), 0) * d
end
function VIXFuture(hParams::HestonParams; t=30 / 365, dt=30 / 365)
    upperBound = 10.0 * hParams.σ^2
    minBound = 0.0
    return quadgk(x -> vixCallIntegrand(hParams, 0.0, 1.0, x, t=t, dt=dt), minBound, upperBound)[1]
end

function logContractIntegrand(hParams, x; dt=30 / 365)
    d = hestonDensity(hParams, x)
    -(log(sqrt(integratedVarianceExpectation(hParams, x)))) * 2d / dt
end
function logContractIntegrand(v0, κ, θ, σ, x; dt=30 / 365)
    d = hestonDensity(v0, κ, θ, σ, x)
    -(log(sqrt(integratedVarianceExpectation(x, κ, θ)))) * 2d / dt
end

function VVIXLogContract(hParams::HestonParams; dt=30 / 365)
    upperBound = 10.0 * hParams.σ^2
    minBound = 1e-64
    try
        vixfut = quadgk(x -> vixCallIntegrand(hParams, 0.0, 1.0, x, dt=dt), minBound, upperBound)[1]
        valueRef = quadgk(x -> logContractIntegrand(hParams, x, dt=dt), minBound, upperBound)[1] + 2log(vixfut) / dt
        return sqrt(valueRef) * 100
    catch e
        return 1000.0
    end
end
function VVIXLogContract(v0, κ, θ, σ; dt=30 / 365)
    upperBound = 10.0 * σ^2
    minBound = 1e-64
    try
        vixfut = quadgk(x -> vixCallIntegrand(v0, κ, θ, σ, 0.0, 1.0, x, dt=dt), minBound, upperBound)[1]
        valueRef = quadgk(x -> logContractIntegrand(v0, κ, θ, σ, x, dt=dt), minBound, upperBound)[1] + 2log(vixfut) / dt
        return sqrt(valueRef) * 100
    catch e
        return 1000.0
    end
end

function VVIXReplicationEstimate(hParams::HestonParams; strikesVIX=range(0.1, 0.5, length=50))
    upperBound = 10.0 * hParams.σ^2
    minBound = 1e-64
    vixfut = quadgk(x -> vixCallIntegrand(hParams, 0.0, 1.0, x), minBound, upperBound)[1]
    refPricesCall = zeros(length(strikesVIX))
    refPricesPut = zeros(length(strikesVIX))
    for (i, strike) ∈ enumerate(strikesVIX)
        lowerBound = if vixIntrinsic(hParams, strike, 1.0, eps()) > 0
            minBound
        else
            Roots.find_zero(x -> vixIntrinsic(hParams, strike, 1.0, x), (minBound, upperBound), A42())
        end
        refPricesCall[i] = quadgk(x -> vixCallIntegrand(hParams, strike, 1.0, x), lowerBound, upperBound)[1]
        refPricesPut[i] = quadgk(x -> vixCallIntegrand(hParams, strike, -1.0, x), minBound, lowerBound)[1]
    end
    vvix = 0.0
    for (k, strike) ∈ enumerate(strikesVIX)
        dstrike = if k == 1
            strikesVIX[k+1] - strikesVIX[k]
        elseif k == length(strikesVIX)
            strikesVIX[k] - strikesVIX[k-1]
        else
            (strikesVIX[k+1] - strikesVIX[k-1]) / 2
        end
        vvix += ifelse(strikesVIX[k] >= vixfut, refPricesCall[k], refPricesPut[k]) * dstrike / (strike^2)
    end
    vvix = sqrt(vvix * 2 * 365 / 30) * 100

end

function integratedVarianceExpectation(model::HestonParams{T}, vt; dt=30.0 / 365) where {T}
    return integratedVarianceExpectation(vt, model.κ, model.θ, dt=dt)
end

function integratedVarianceExpectation(vt, κ, θ; dt=30.0 / 365)
    ektm1 = (exp(-κ * dt) - 1) / (κ * dt)
    return @. -ektm1 * vt + θ * (ektm1 + 1)
end
function filter_monotone(vs, ts; tol=0.0)
    @assert length(vs) == length(ts)

    if length(vs) <= 1
        return vs, ts, collect(eachindex(vs))
    end

    increasing = vs[end] >= vs[1]

    keep = trues(length(vs))
    last_kept = vs[1]

    for i in 2:length(vs)
        ok = increasing ? (vs[i] >= last_kept - tol) : (vs[i] <= last_kept + tol)
        keep[i] = ok
        if ok
            last_kept = vs[i]
        end
    end

    idx = findall(keep)
    return vs[idx], ts[idx], idx
end
function makeVarianceSwapReduction(ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractVector{T},
    vols::AbstractMatrix{T}; isLinearExtrapolation=false, thetaUpperBoundFactor=0.0, v0UpperBoundFactor=0.0, maturities=ts) where {T}
    return makeVarianceSwapReduction(ts, forwards, reshape(strikes, 1, :), vols, isLinearExtrapolation=isLinearExtrapolation, thetaUpperBoundFactor=thetaUpperBoundFactor, v0UpperBoundFactor=v0UpperBoundFactor, maturities=maturities)
end

function computeVSTheo(ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T},
    vols::AbstractMatrix{T}; isLinearExtrapolation=false, maturities=ts) where {T}
    vsTheo = if size(strikes, 1) == 1
        [priceVarianceSwap(FukasawaVarianceSwapReplication(isLinearExtrapolation), t, log.(strikes[1, :] / forwards[i]), vols[i, :] .^ 2, 1.0) for (i, t) ∈ enumerate(ts)]
    else
        [priceVarianceSwap(FukasawaVarianceSwapReplication(isLinearExtrapolation), t, log.(strikes[i, :] / forwards[i]), vols[i, :] .^ 2, 1.0) for (i, t) ∈ enumerate(ts)]
    end
     if maturities != ts
        interp = PPInterpolation.CubicSplineNatural(ts, vsTheo)
        vsTheo = map(t->PPInterpolation.evaluate(interp, t, extrapolation=LinearAutoExtrapolation()), maturities)
        ts = maturities
    end
    return ts, vsTheo
end
function makeVarianceSwapReduction(ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T},
    vols::AbstractMatrix{T}; isLinearExtrapolation=false, thetaUpperBoundFactor=0.0, v0UpperBoundFactor=0.0, maturities=ts) where {T}
    ts, vsTheo = computeVSTheo(ts, forwards, strikes, vols ,isLinearExtrapolation = isLinearExtrapolation, maturities=maturities)
   
    #println("filtered= ", vsTheo, ts)
    vsTrans = [if v0UpperBoundFactor == 0.0
            MQMinTransformation(1e-4, 1.0)
        else
            LogisticTransformation(1e-4, maximum(vsTheo) / 10000 * v0UpperBoundFactor)
        end,
        MQMinTransformation(1e-4, 1.0),
        if thetaUpperBoundFactor == 0.0
            MQMinTransformation(1e-4, 1.0)
        else
            LogisticTransformation(1e-4, maximum(vsTheo) / 10000 * thetaUpperBoundFactor)
        end]
    objectiveVS = function (x)
        vsModel = map(t -> integratedVarianceExpectation.(vsTrans[1](x[1]), vsTrans[2](x[2]), vsTrans[3](x[3]), dt=t) * 10000, ts)
        vsModel - vsTheo
    end

    lnx0 = [inv(vsTrans[1], vsTheo[1] / 10000), inv(vsTrans[2], 1.0), inv(vsTrans[3], vsTheo[end] / 10000)]
    vsResult = GaussNewton.optimize(objectiveVS, lnx0)
    #println("Variance Swap Heston calibration ", vsResult)
    x0 = [vsTrans[1](lnx0[1]), vsTrans[2](lnx0[2]), vsTrans[3](lnx0[3])]
    return V0KappaThetaReduction(x0[1], x0[2], x0[3])
end


function makeVarianceSwapReduction1D(ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractVector{T},
    vols::AbstractMatrix{T}; isLinearExtrapolation=false) where {T}
    return makeVarianceSwapReduction1D(ts, forwards, reshape(strikes, 1, :), vols, isLinearExtrapolation=isLinearExtrapolation)
end
function makeVarianceSwapReduction1D(ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T},
    vols::AbstractMatrix{T}; isLinearExtrapolation=false) where {T}
    ts, vsTheo = computeVSTheo(ts, forwards, strikes, vols ,isLinearExtrapolation = isLinearExtrapolation, maturities=ts)

    vsTrans = [
        MQMinTransformation(1e-4, 1.0)]
    objectiveVS = function (x)
        vsModel = map(t -> integratedVarianceExpectation.(vsTheo[1]/10000, vsTrans[1](x[1]), vsTheo[end]/10000, dt=t) * 10000, ts)
        vsModel - vsTheo
    end
    lnx0 = [inv(vsTrans[1], 1.0)]
    vsResult = GaussNewton.optimize(objectiveVS, lnx0)
    println("Variance Swap Heston calibration ", vsResult)
    x0 = [vsTrans[1](lnx0[1])]
    return V0KappaThetaReduction(vsTheo[1]/10000, x0[1], vsTheo[end]/10000)
end

function makeVarianceSwapReductionExplicit(ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T},
    vols::AbstractMatrix{T}; isLinearExtrapolation=false, tte=1.0) where {T}

    ts, vsTheo = computeVSTheo(ts, forwards, strikes, vols ,isLinearExtrapolation = isLinearExtrapolation, maturities=ts)

    v0 = vsTheo[1]/10000
    θ = vsTheo[end]/10000
    if abs(θ - v0) < 1e-4
        return V0KappaThetaReduction(v0, one(T), θ)
    else
        interp = PPInterpolation.CubicSplineNatural(ts, vsTheo)
        vartte = PPInterpolation.evaluate(interp, tte, extrapolation=PPInterpolation.LinearAutoExtrapolation())/10000
        a = (vartte - θ)/(v0 - θ)
        println("a = ", a)
        if (a <= zero(T) || a >= one(T))
            return V0KappaThetaReduction(v0, one(T), θ)
        end
        κ = (1/a + lambertW(-exp(-1/a)/a))/tte
        return V0KappaThetaReduction(v0, κ, θ)
    end
end
function makeDoubleHestonVarianceSwapReduction(ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractVector{T},
    vols::AbstractMatrix{T}; isLinearExtrapolation=false, thetaUpperBoundFactor=0.0, v0UpperBoundFactor=0.0) where {T}

    vsTheo = [priceVarianceSwap(FukasawaVarianceSwapReplication(isLinearExtrapolation), t, log.(strikes / forwards[i]), vols[i, :] .^ 2, 1.0) for (i, t) ∈ enumerate(ts)]
    vsTrans = [if v0UpperBoundFactor == 0.0
            MQMinTransformation(1e-4, 1.0)
        else
            LogisticTransformation(1e-4, maximum(vsTheo) / 10000 * v0UpperBoundFactor)
        end,
        MQMinTransformation(1e-4, 1.0),
        if thetaUpperBoundFactor == 0.0
            MQMinTransformation(1e-4, 1.0)
        else
            LogisticTransformation(1e-4, maximum(vsTheo) / 10000 * thetaUpperBoundFactor)
        end]
    objectiveVS2 = function (x)
        vsModel = map(t -> integratedVarianceExpectation(vsTrans[1](x[1]), vsTrans[2](x[2]), vsTrans[3](x[3]), dt=t) * 10000 + integratedVarianceExpectation.(vsTrans[1](x[1]), vsTrans[2](x[4]), vsTrans[3](x[5]), dt=t) * 10000, ts)
        vsModel - vsTheo
    end
    lnx02 = [inv(vsTrans[1], vsTheo[1] / 10000 / 2), inv(vsTrans[2], 1.0), inv(vsTrans[3], vsTheo[end] / 10000 / 2), inv(vsTrans[2], 2.0), inv(vsTrans[3], vsTheo[end] / 10000 / 2)]
    vsResult = GaussNewton.optimize(objectiveVS2, lnx02)
    x02 = [vsTrans[1](lnx02[1]), vsTrans[2](lnx02[2]), vsTrans[3](lnx02[3]), vsTrans[2](lnx02[4]), vsTrans[3](lnx02[5])]
    println("Variance Swap Double Heston calibration ", vsResult)
    return V0KappaThetaDoubleHestonReduction(x02[1], x02[2], x02[3], x02[1], x02[4], x02[5])
end

function calibrateHestonFromPrices(
    ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T},
    prices::AbstractMatrix{T},
    isCall::AbstractMatrix{Bool},
    weights::AbstractMatrix{T};
    reduction::Reduction=NoReduction(),
    isRelative=false,
    method=:JoshiYang,
    minimizer="DE",
    VVIX=0.0,
    weightVVIX=1.0,
    strikesVIX=range(0.1, 0.5, length=50),
    lower=[1e-4, 1e-4, 1e-4, -0.99, 1e-2],
    upper=[4.0, 15.0, 4.0, 0.9, 4.0]) where {T}
    # uPrices = zeros(T, length(ts),length(strikes))))
    # uWeights = zeros(T, length(ts),length(strikes))
    # for i  = eachindex(forwards)
    # 	@. uPrices[i,:] = prices[i,:] ./ forwards[i]
    # 	@. uWeights[i,:] = weights[i,:] .* forwards[i]
    # end
    uPrices = prices
    uWeights = weights

    function objectiveN(F, xr)
        x = expandX(reduction, HestonParams, xr)
        hParams = HestonParams(x[1], x[2], x[3], x[4], x[5])
        # TX = typeof(x[1])
        # TH = typeof(hParams)
        # cf = DefaultCharFunc{TH,Complex{TX}}(hParams)
        cf = DefaultCharFunc(hParams)

        for (i, t) ∈ enumerate(ts)
            # pricer =JoshiYangCharFuncPricer(cf, t, n=64)
            pricer = makeCharFuncPricer(pricerMethod(method), cf, t)
            for (j, strike) ∈ enumerate(strikes[i, :])
                mPrice = CharFuncPricing.priceEuropean(pricer, isCall[i, j], strike / forwards[i], 1.0, t, 1.0) * forwards[i]
                F[i+(j-1)*length(ts)] = if isRelative
                    uWeights[i, j] * (mPrice / uPrices[i, j] - 1)
                else
                    uWeights[i, j] * (mPrice - uPrices[i, j])
                end
            end
        end
        if !iszero(VVIX) && !iszero(weightVVIX)
            F[length(F)] = weightVVIX * (VVIXApproxSimple(hParams) - VVIX)
            #F[length(F)] = weightVVIX*(VVIXReplicationEstimate(hParams) - VVIX)
            # F[length(F)] = weightVVIX*(VVIXLogContract(hParams) - VVIX)
        end
        F
    end
    totalSize = size(prices, 1) * size(prices, 2)
    if !iszero(VVIX) && !iszero(weightVVIX)
        totalSize += 1
    end
    out = zeros(totalSize)
    function objective1(x)
        objectiveN(out, x)
        norm(out) / norm(uWeights)
    end
    strikeShortIndex = floor(Int, size(strikes, 2) / 2)
    priceShort = uPrices[1, strikeShortIndex]
    volShort = Black.impliedVolatility(isCall[1, strikeShortIndex], priceShort, forwards[1], strikes[1, strikeShortIndex], ts[1], 1.0)
    priceLong = uPrices[end, strikeShortIndex]
    volLong = Black.impliedVolatility(isCall[end, strikeShortIndex], priceLong, forwards[end], strikes[end, strikeShortIndex], ts[end], 1.0)
    x0 = [volShort^2, 1.0, volLong^2, -0.5, 0.5] #v0::T, κ::T, θ::T, ρ::T, σ::T)

    rx0 = reduceX(reduction, HestonParams, x0)
    rlower = reduceX(reduction, HestonParams, lower)
    rupper = reduceX(reduction, HestonParams, upper)
    rng = Random123.Philox4x(UInt64, (20130129, 20100921), 10)
    problem = GlobalOptimization.Problem(length(rx0), objective1, rlower, rupper)
    result = if minimizer == "DE"
        optim = GlobalOptimization.makeDifferentialEvolutionOptimizer(GlobalOptimization.OptimizerParams(15),
            problem,
            rng, GlobalOptimization.Best1Bin(0.9, 0.5))
        GlobalOptimization.optimize(optim, GlobalOptimization.TerminationCriteria(1000, 100, 1e-6, 1e-5))
    else
        optim = GlobalOptimization.SimulatedAnnealing(problem, rng)
        GlobalOptimization.optimize(optim, rx0)
    end
    println("Result DE ", result, " ", GlobalOptimization.minimizer(optim))
    rx = GlobalOptimization.minimizer(optim) #calibrated params.

    transforms = map((low, high) -> LogisticTransformation(low, high), rlower, rupper)

    function objectiveC(F, xinv)
        x = map((xi, transformi) -> transformi(xi), xinv, transforms)
        objectiveN(F, x)
    end
    rxinv = map((xi, transformi) -> inv(transformi, xi), rx, transforms)
    gnRmse = GaussNewton.optimize!(objectiveC, rxinv, out, autodiff=:single)
    println("Result GN ", gnRmse)
    rx = map((xi, transformi) -> transformi(xi), rxinv, transforms)
    x = expandX(reduction, HestonParams, rx)
    rmse = norm(uWeights) * GlobalOptimization.minimum(optim) / sqrt(size(prices, 1) * size(prices, 2)) #very close to truncated vol rmse.
    params = HestonParams(x[1], x[2], x[3], x[4], x[5])
    return params, rmse
end

function calibrateDoubleHestonFromPrices(
    ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractVector{T},
    prices::AbstractMatrix{T},
    isCall::AbstractMatrix{Bool},
    weights::AbstractMatrix{T};
    reduction::Reduction=NoReduction(),
    method=:JoshiYang,
    minimizer="DE",
    VVIX=0.0,
    weightVVIX=1.0,
    strikesVIX=range(0.1, 0.5, length=50)) where {T}
    # uPrices = zeros(T, length(ts),length(strikes))))
    # uWeights = zeros(T, length(ts),length(strikes))
    # for i  = eachindex(forwards)
    # 	@. uPrices[i,:] = prices[i,:] ./ forwards[i]
    # 	@. uWeights[i,:] = weights[i,:] .* forwards[i]
    # end
    uPrices = prices
    uWeights = weights
    lower1 = [1e-4, 1e-4, 1e-4, -0.99, 1e-2]
    upper1 = [4.0, 10.0, 4.0, 0.9, 4.0]
    lower = vcat(lower1, lower1)
    upper = vcat(upper1, upper1)

    function objectiveN(F, xr)
        x = expandX(reduction, DoubleHestonParams, xr)
        @. x = max(lower, min(upper, x))
        hParams = DoubleHestonParams(
            HestonParams(x[1], x[2], x[3], x[4], x[5]),
            HestonParams(x[6], x[7], x[8], x[9], x[10]))
        # TX = typeof(x[1])
        # TH = typeof(hParams)
        # cf = DefaultCharFunc{TH,Complex{TX}}(hParams)
        cf = DefaultCharFunc(hParams)

        for (i, t) ∈ enumerate(ts)
            pricer = makeCharFuncPricer(pricerMethod(method), cf, t)

            for (j, strike) ∈ enumerate(strikes)
                mPrice = CharFuncPricing.priceEuropean(pricer, isCall[i, j], strike / forwards[i], 1.0, t, 1.0) * forwards[i]
                F[i+(j-1)*length(ts)] = uWeights[i, j] * (mPrice - uPrices[i, j])
            end
        end
        if !iszero(VVIX) && !iszero(weightVVIX)
            F[length(F)] = weightVVIX * (VVIXApproxSimple(hParams) - VVIX)
            #F[length(F)] = weightVVIX*(VVIXReplicationEstimate(hParams) - VVIX)
            # F[length(F)] = weightVVIX*(VVIXLogContract(hParams) - VVIX)
        end
        F
    end
    totalSize = length(ts) * length(strikes)
    if !iszero(VVIX) && !iszero(weightVVIX)
        totalSize += 1
    end
    out = zeros(totalSize)
    function objective1(x)
        objectiveN(out, x)
        norm(out) / norm(uWeights)
    end
    strikeShortIndex = floor(Int, length(strikes) / 2)
    priceShort = uPrices[1, strikeShortIndex]
    volShort = Black.impliedVolatility(isCall[1, strikeShortIndex], priceShort, forwards[1], strikes[strikeShortIndex], ts[1], 1.0)
    priceLong = uPrices[end, strikeShortIndex]
    volLong = Black.impliedVolatility(isCall[end, strikeShortIndex], priceLong, forwards[end], strikes[strikeShortIndex], ts[end], 1.0)
    x01 = [volShort^2, 1.0, volLong^2, -0.5, 0.5] #v0::T, κ::T, θ::T, ρ::T, σ::T)
    x0 = [x01[1] / 2, x01[2] / 2, x01[3] / 2, x01[4], x01[5], x01[1] / 2, x01[2] * 2, x01[3] / 2, x01[4], x01[5]]
    rx0 = reduceX(reduction, DoubleHestonParams, x0)
    lower1g = [0.1, 1e-2, 1e-2, -0.9, 1e-2]
    upper1g = [4.0, 10.0, 4.0, 0.5, 2.0]
    lowerg = vcat(lower1g, lower1g)
    upperg = vcat(upper1g, upper1g)

    rlower = reduceX(reduction, DoubleHestonParams, lowerg)
    rupper = reduceX(reduction, DoubleHestonParams, upperg)
    rng = Random123.Philox4x(UInt64, (20130129, 20100921), 10)
    problem = GlobalOptimization.Problem(length(rx0), objective1, rlower, rupper)
    result = if minimizer == "DE"
        optim = GlobalOptimization.makeDifferentialEvolutionOptimizer(GlobalOptimization.OptimizerParams(15),
            problem,
            rng, GlobalOptimization.Best1Bin(0.9, 0.5))
        GlobalOptimization.optimize(optim, GlobalOptimization.TerminationCriteria(1000, 100, 1e-6, 1e-5))
    else
        optim = GlobalOptimization.SimulatedAnnealing(problem, rng)
        GlobalOptimization.optimize(optim, rx0)
    end
    println("Result DE ", result, " ", GlobalOptimization.minimizer(optim))
    rx = GlobalOptimization.minimizer(optim) #calibrated params.
    gnRmse = GaussNewton.optimize!(objectiveN, rx, out, autodiff=:single)
    println("Result GN ", gnRmse)
    x = expandX(reduction, DoubleHestonParams, rx)
    @. x = max(lower, min(upper, x))
    rmse = norm(uWeights) * GlobalOptimization.minimum(optim) / sqrt(length(strikes) * length(ts)) #very close to truncated vol rmse.
    params = DoubleHestonParams(
        HestonParams(x[1], x[2], x[3], x[4], x[5]),
        HestonParams(x[6], x[7], x[8], x[9], x[10]))
    return params, rmse
end



function convertVolsToPricesOTMWeights(
    t::T,
    forward::T,
    strikes::AbstractVector{T},
    vols::AbstractVector{T};
    weights::AbstractVector{T}=ones(T, size(vols)),
    vegaFloor=1e-2,
    isInverse=true,
    truncationDev=3.0,
) where {T}
    uPrices = zeros(size(vols))
    uVegas = zeros(size(vols))
    uWeights = zeros(size(vols))
    isCall = zeros(Bool, size(vols))
   

    for i ∈ eachindex(strikes)
            strike = strikes[i]
            isCall[i] = strike >= forward
            uPrices[i] = Black.blackScholesFormula(isCall[i], strike, forward, vols[i]^2 * t, 1.0, 1.0)
            uVegas[i] = Black.blackScholesVega(strike, forward, vols[i]^2 * t, 1.0, 1.0, t)
        end

        if isInverse
            @. uWeights = weights / max(uVegas, vegaFloor * forward)
        else
            sumVega = sum(uVegas)
            @. uWeights = weights * uVegas / sumVega
        end
        indicator = @. truncationDev * vols * sqrt(t) > abs(log(strikes/forward))
        uWeights .*= indicator
    
    return uPrices, isCall, uWeights
end

function convertVolsToPricesOTMWeights(
    ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractVecOrMat{T}, #a vector of strikes is shared by all maturities
    vols::AbstractMatrix{T};
    weights::AbstractMatrix{T}=ones(T, size(vols)),
    vegaFloor=1e-2,
    isInverse=true,
    truncationDev=3.0,
) where {T}
    uPrices = zeros(size(vols))
    uVegas = zeros(size(vols))
    uWeights = zeros(size(vols))
    isCall = zeros(Bool, size(vols))
    mStrikes = if Base.ndims(strikes) == 1
        ones(length(ts)) * strikes'
    else
        strikes
    end


    for i ∈ axes(vols, 1)
        for j ∈ axes(vols, 2)
            strike = mStrikes[i, j]
            isCall[i, j] = strike >= forwards[i]
            uPrices[i, j] = Black.blackScholesFormula(isCall[i, j], strike, forwards[i], vols[i, j]^2 * ts[i], 1.0, 1.0)
            uVegas[i, j] = Black.blackScholesVega(strike, forwards[i], vols[i, j]^2 * ts[i], 1.0, 1.0, ts[i])
        end
        if isInverse
            @. uWeights[i, :] = weights[i, :] / max(uVegas[i, :], vegaFloor * forwards[i])
        else
            sumVega = sum(uVegas[i, :])
            @. uWeights[i, :] = weights[i, :] * uVegas[i, :] / sumVega
        end
        indicator = @. truncationDev * vols[i, :] * sqrt(ts[i]) > abs(log(mStrikes[i, :] / forwards[i]))
        uWeights[i, :] .*= indicator
    end
    return uPrices, isCall, uWeights
end

function estimateVolError(params, ts::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractMatrix{T}, vols; weights=ones(T, size(vols)), pricerName="Joshi-Yang") where {T}
    cf = DefaultCharFunc(params)

    volError = zeros(size(vols))
    count = 0
    for (i, t) ∈ enumerate(ts)
        #pricer = CharFuncPricing.ALCharFuncPricer(cf, n=128)
        pricer = if pricerName == "Joshi-Yang"
            CharFuncPricing.JoshiYangCharFuncPricer(cf, t, n=128)
        elseif pricerName == "Cos"
            CharFuncPricing.makeCosCharFuncPricer(cf, t, 256, 10)
        else
            CharFuncPricing.AdaptiveFilonCharFuncPricer(cf, t)
        end
        for (j, strike) ∈ enumerate(strikes[i, :])
            isCall = strike > forwards[i]
            mPrice = CharFuncPricing.priceEuropean(pricer, isCall, strike / forwards[i], 1.0, t, 1.0)
            if !iszero(weights[i, j])
                try
                    volError[i, j] = weights[i, j] * (Black.impliedVolatility(isCall, mPrice, 1.0, strike / forwards[i], t, 1.0) - vols[i, j])
                    count += 1

                catch e
                    println(e)
                    volError[i, j] = NaN
                end
            end
        end
    end
    rmseVol = norm(volError) / sqrt(count) #around 0.015
    return volError, rmseVol
end

function calibrateHestonFromVols(ts::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractMatrix{T}, vols::AbstractMatrix{T}; weights::AbstractMatrix{T}=ones(size(vols)), vegaFloor=1e-2) where {T}
    uPrices, isCall, uWeights = convertVolsToPricesOTMWeights(ts, forwards, strikes, vols, weights=weights, vegaFloor=vegaFloor, truncationDev=3.0)
    println(uWeights)
    params, rmse = calibrateHestonFromPrices(ts, forwards, strikes, uPrices, isCall, uWeights, method=:Cos)
    volError, rmseVol = estimateVolError(params, ts, forwards, strikes, vols, weights=weights)
    return params, rmseVol
end

function filterVolsByMoneyness(ts::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractMatrix{T}, vols::AbstractMatrix{T}; moneyness=[-0.5, 0.0, 0.5]) where {T}
    volsF = zeros((length(ts), length(moneyness)))
    strikesF = zeros((length(ts), length(moneyness)))
    for (i, t) ∈ enumerate(ts)
        spl = PPInterpolation.CubicSplineNatural(log.(strikes[i, :] ./ forwards[i]), vols[i, :])
        atmVol = spl(0.0)
        y = moneyness .* (atmVol * sqrt(t))
        volsF[i, :] = spl.(y)
        strikesF[i, :] = exp.(y) .* forwards[i]
    end
    return volsF, strikesF
end

function filterVolsBySimpleDelta(ts::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractMatrix{T}, vols::AbstractMatrix{T}; deltas=[0.25, 0.5, 0.75]) where {T}
    volsF = zeros((length(ts), length(deltas)))
    strikesF = zeros((length(ts), length(deltas)))
    for (i, ti) ∈ enumerate(ts)
        logm = log.(strikes[i, :] ./ forwards[i])
        spl = PPInterpolation.CubicSplineNatural(logm, vols[i, :])
        atmVol = spl(0.0)
        y = @. norminv(deltas) .* (atmVol * sqrt(ti))
        volsF[i, :] = spl.(y)
        strikesF[i, :] = exp.(y) .* forwards[i]
    end
    return volsF, strikesF
end

function filterWeights(uweights::AbstractMatrix,
    ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T},
    vols::AbstractMatrix{T}; deviations=3.0,
) where {T}
    weights = copy(uweights)
    for (i, t) ∈ enumerate(ts)
        for (j, strike) ∈ enumerate(strikes[i, :])
            x = log(strike / forwards[i]) / (vols[i, j] * sqrt(t))
            if abs(x) > deviations
                weights[i, j] = 0
            end
        end
    end
    return weights
end
function updateWeights(weights, ts, strikes, spot; moneynessThreshold=0.2, weightFactor=2.0)
    for i ∈ eachindex(ts)
        for (j, strike) ∈ enumerate(strikes[i, :])
            x = log(strike / spot)
            if abs(x) < moneynessThreshold
                weights[i, j] *= weightFactor
            end
        end
    end
    weights
end


function impliedVolatility(pricer, isCall::Bool, strike, forward, t; minPrice=1e-16)
    mPrice = CharFuncPricing.priceEuropean(pricer, isCall, strike / forward, 1.0, t, 1.0)
    mPrice = max(minPrice, mPrice)
    return Black.impliedVolatility(isCall, mPrice, 1.0, strike / forward, t, 1.0)
end

function impliedVolatilityWithJacobian(pricer, isCall::Bool, strike, forward, tte)
    throw(Exception("unsupported"))
end
using Zygote
function calibrateHestonFromVolsDirect(
    ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T},
    vols::AbstractMatrix{T};
    uweights::AbstractMatrix=ones(size(vols)),
    method=:Cos,
    reduction::Reduction=NoReduction(),
    minimizer="DE", deviations=4.0,
    lower=[1e-4, 1e-4, 1e-4, -0.99, 1e-2],
    upper=[4.0, 10.0, 4.0, 0.9, 4.0],
    autodiff=:single, # :forward is ok for DNN1D
) where {T}
    weights = filterWeights(uweights, ts, forwards, strikes, vols, deviations=3)
    function objectiveImmutableN(xr)
        x = expandX(reduction, HestonParams, xr)
        hParams = HestonParams(x[1], x[2], x[3], x[4], x[5])
        # TX = typeof(x[1])
        # TH = typeof(hParams)
        # cf = DefaultCharFunc{TH,Complex{TX}}(hParams)
        cf = DefaultCharFunc(hParams)
        sizeT = length(ts)
        sizeS = size(strikes, 2)  # max number of strikes

        return [
            begin
                t = ts[i]
                strike = strikes[i, j]
                w = weights[i, j]
                fwd = forwards[i]
                isCall = strike > fwd

                if !iszero(w)
                    pricer = makeCharFuncPricer(pricerMethod(method), cf, t)
                    try
                        mVol = impliedVolatility(pricer, isCall, strike / fwd, 1.0, t)
                        w * (mVol - vols[i, j])
                    catch
                        w * vols[i, j]
                    end
                else
                    0.0
                end
            end
            for j in 1:sizeS, i in 1:sizeT
        ] |> vec

    end
    function objectiveN(F, xr)
        x = expandX(reduction, HestonParams, xr)
        hParams = HestonParams(x[1], x[2], x[3], x[4], x[5])
        # TX = typeof(x[1])
        # TH = typeof(hParams)
        # cf = DefaultCharFunc{TH,Complex{TX}}(hParams)
        cf = DefaultCharFunc(hParams)

        for (i, t) ∈ enumerate(ts)
            # pricer =JoshiYangCharFuncPricer(cf, t, n=64)
            pricer = makeCharFuncPricer(pricerMethod(method), cf, t)

            for (j, strike) ∈ enumerate(strikes[i, :])
                isCall = strike > forwards[i]
                if !iszero(weights[i, j])
                    try
                        mVol = impliedVolatility(pricer, isCall, strike / forwards[i], 1.0, t)
                        F[i+(j-1)*length(ts)] = weights[i, j] * (mVol - vols[i, j])

                    catch e
                        #println(e)
                        F[i+(j-1)*length(ts)] = weights[i, j] * vols[i, j]
                    end
                else
                    F[i+(j-1)*length(ts)] = 0.0
                end


            end
        end
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
    x0 = [volShort^2, 1.0, volLong^2, -0.5, 0.5] #v0::T, κ::T, θ::T, ρ::T, σ::T)

    rx0 = reduceX(reduction, HestonParams, x0)
    rlower = reduceX(reduction, HestonParams, lower)
    rupper = reduceX(reduction, HestonParams, upper)
    rng = Random123.Philox4x(UInt64, (20130129, 20100921), 10)
    problem = GlobalOptimization.Problem(length(rx0), objective1, rlower, rupper)
    result = if minimizer == "DE"
        optim = GlobalOptimization.makeDifferentialEvolutionOptimizer(GlobalOptimization.OptimizerParams(15),
            problem,
            rng, GlobalOptimization.Best1Bin(0.9, 0.5))
        GlobalOptimization.optimize(optim, GlobalOptimization.TerminationCriteria(100, 50, 1e-6, 1e-5))
    else
        optim = GlobalOptimization.SimulatedAnnealing(problem, rng)
        GlobalOptimization.optimize(optim, rx0)
    end
    println("Result DE ", result, " ", GlobalOptimization.minimizer(optim))
    rx = GlobalOptimization.minimizer(optim) #calibrated params.
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
    elseif autodiff == :finitediff
        function objectiveDer(Jac, xinv)
            x = map((xi, transformi) -> transformi(xi), xinv, transforms)
            x_jac = FiniteDifferences.jacobian(central_fdm(3, 1), xinv -> map((xi, transformi) -> transformi(xi), xinv, transforms), xinv)[1]

            f_jac = zeros(size(Jac))
            xr = expandX(reduction, HestonParams, x)
            hParams = HestonParams(xr[1], xr[2], xr[3], xr[4], xr[5])
            # TX = typeof(x[1])
            # TH = typeof(hParams)
            # cf = DefaultCharFunc{TH,Complex{TX}}(hParams)
            cf = DefaultCharFunc(hParams)

            for (i, t) ∈ enumerate(ts)
                # pricer =JoshiYangCharFuncPricer(cf, t, n=64)
                pricer = makeCharFuncPricer(pricerMethod(method), cf, t)

                for (j, strike) ∈ enumerate(strikes[i, :])
                    isCall = strike > forwards[i]
                    if !iszero(weights[i, j])
                        try
                            mVolWithJ = impliedVolatilityWithJacobian(pricer, isCall, strike / forwards[i], 1.0, t)
                            mVolJac = mVolWithJ[2:end]
                            f_jac[i+(j-1)*length(ts), :] = weights[i, j] .* mVolJac

                        catch e
                            println(e)
                            f_jac[i+(j-1)*length(ts), :] .= 0.0
                        end
                    else
                        f_jac[i+(j-1)*length(ts), :] .= 0.0
                    end


                end
            end

            Jac[:, :] = f_jac * x_jac

        end

        GaussNewton.optimize!(objectiveC, objectiveDer, rxinv, out)
    else
        GaussNewton.optimize!(objectiveC, rxinv, out, autodiff=autodiff)
    end
    println("Result GN ", gnRmse)
    rx = map((xi, transformi) -> transformi(xi), rxinv, transforms)
    x = expandX(reduction, HestonParams, rx)
    rmse = norm(weights) * GlobalOptimization.minimum(optim) / sqrt(size(vols, 1) * size(vols, 2)) #very close to truncated vol rmse.
    params = HestonParams(x[1], x[2], x[3], x[4], x[5])
    return params, rmse
end


function calibrateHestonFromVolsFixed(
    ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T},
    vols::AbstractMatrix{T};
    weights::AbstractMatrix=ones(size(vols)),
    method=:Cos,
    reduction::Reduction=NoReduction(),
    minimizer="DE", deviations=4.0,
    lower=[1e-4, 1e-4, 1e-4, -0.99, 1e-2],
    upper=[4.0, 10.0, 4.0, 0.9, 4.0],
    autodiff=:single, trace=false, # :forward is ok for DNN1D
    initialGuess=HestonParams(0.0, 1.0, 0.0, -0.5, 0.5)
) where {T}
    #weights = filterWeights(uweights, ts, forwards, strikes, vols, deviations=3)
    function objectiveImmutableN(xr)
        x = expandX(reduction, HestonParams, xr)
        hParams = HestonParams(x[1], x[2], x[3], x[4], x[5])
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
    x0 = [initialGuess.v0, initialGuess.κ, initialGuess.θ, initialGuess.ρ, initialGuess.σ]
    if iszero(initialGuess.v0)
        x0[1] = volShort^2
    end
    if iszero(initialGuess.θ)
        x0[3] = volLong^2
    end

    rx0 = reduceX(reduction, HestonParams, x0)
    rlower = reduceX(reduction, HestonParams, lower)
    rupper = reduceX(reduction, HestonParams, upper)
    rng = Random123.Philox4x(UInt64, (20130129, 20100921), 10)
    problem = GlobalOptimization.Problem(length(rx0), objective1, rlower, rupper)
    rx = if minimizer == "DE"
        optim = GlobalOptimization.makeDifferentialEvolutionOptimizer(GlobalOptimization.OptimizerParams(15),
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
    x = expandX(reduction, HestonParams, rx)
    rmse = gnRmse
    params = HestonParams(x[1], x[2], x[3], x[4], x[5])
    return params, rmse
end



function impliedVolatilities(method::Val{P}, cf, strikes, weights, forwards, ts) where {P}
    mvols = [
        begin
            t = ts[i]
            strike = strikes[i, j]
            w = weights[i, j]
            fwd = forwards[i]
            isCall = strike > fwd

            if !iszero(w)
                pricer = makeCharFuncPricer(method, cf, t)
                try
                    impliedVolatility(pricer, isCall, strike / fwd, 1.0, t)
                catch
                    0.0
                end
            else
                0.0
            end
        end
        for i in 1:size(strikes, 1), j in 1:size(strikes, 2)
    ]
    return mvols
end





struct FordeGuessInput5D{TV}
    X0::Float64 #a log-moneyness
    T1::Float64 #first maturity
    T2::Float64 #second maturity
    V00::TV
    Vx0t1::TV
    Vmx0t1::TV
    Vx0t2::TV
    Vmx0t2::TV
end

function computeHestonFordeGuess(i::FordeGuessInput5D{TV}; kappa_min=0.1, theta_min=0.01, kappa=0.0) where {TV}
    x0 = i.X0
    t1 = i.T1
    t2 = i.T2
    v00 = i.V00
    vx0t1 = i.Vx0t1
    vmx0t1 = i.Vmx0t1
    vx0t2 = i.Vx0t2
    vmx0t2 = i.Vmx0t2

    v0 = v00
    vp = (t2 * vx0t1 - t1 * vx0t2) / (t2 - t1)
    vm = (t2 * vmx0t1 - t1 * vmx0t2) / (t2 - t1)

    C = (vp - 2 * v00 + vm) / (2 * x0^2)
    C = max(0.0, C)

    S = (vp - vm) / (2 * x0)
    delta = 7 * S^2 + 12 * v00 * C
    sigma = sqrt(delta)
    rho = 2 * S / sigma

    q = [
        vx0t1 - vmx0t1 - vp + vm,
        vx0t1 + vmx0t1 - vp - vm,
    ] ./ (2t1)

    r = [
        (rho * sigma^3 * (1 - rho^2) + v0 * rho^2 * sigma^2) / (24 * v0) * x0,
        (12 * v0 * rho * sigma + (rho^2 - 4) * sigma^2) / 48 + ((176 - 712 * rho^2 + 521 * rho^4) * sigma^4 + 40 * v0 * rho^3 * sigma^3) / (7680 * v0^2) * x0^2,
    ]

    qmr = q .- r

    # md matrix in row-major order
    md = [
        -rho*sigma/(12*v0)*x0 -rho*sigma/12*x0;
        0.5+(13*rho^2-6)*sigma^2/(96*v0^2)*x0^2 -v0/2-rho^2*sigma^2/(96*v0)*x0^2
    ]

    if iszero(kappa)
        # Solve linear system md * solution = qmr
        solution = md \ qmr

        alpha = solution[1]
        kappa = solution[2]

        # Explicitly enforce constraints
        if kappa < kappa_min && alpha / kappa < theta_min
            kappa = kappa_min
            alpha = theta_min * kappa
        elseif kappa < kappa_min
            kappa = kappa_min
        elseif alpha / kappa < theta_min
            alpha = theta_min * kappa
        end

        theta = alpha / kappa
    else
        #Constraint C.x = d
        C = [0.0 1.0]
        d = [kappa]
        KKT_mat = [md'*md C';
            C zeros(1, 1)]
        KKT_rhs = [md' * qmr;
            d]
        solution = KKT_mat \ KKT_rhs
        alpha = solution[1]
        println("constrained check ", solution[2], " ==? ", kappa)
        theta = max(theta_min, alpha / kappa)
    end
    return HestonParams(v0, kappa, theta, rho, sigma)
end

"""
    computeHestonGeminiGuess(v0, x0, t1, t2, vols_t1, vols_t2)

Performs an asymptotic inversion of the Heston model to retrieve parameters 
(kappa, theta, sigma, rho) using small-time expansions.

"""
function computeHestonGeminiGuess(input::FordeGuessInput5D; sigma_min=0.05)

    # Unpack vols (Order: Put-side (negative x), Call-side (positive x))
    iv_m_t1, iv_p_t1 = sqrt(input.Vmx0t1), sqrt(input.Vx0t1)
    iv_m_t2, iv_p_t2 = sqrt(input.Vmx0t2), sqrt(input.Vx0t2)
    x0 = input.X0
    t1 = input.T1
    t2 = input.T2
    v0 = input.V00

    # --- Step 1: Compute Observables (Skew and Level) ---
    # Skew S(t) ~ Slope
    S_t1 = (iv_p_t1 - iv_m_t1) / (2 * x0)
    S_t2 = (iv_p_t2 - iv_m_t2) / (2 * x0)

    # Level M(t) ~ ATM Average
    M_t1 = (iv_p_t1 + iv_m_t1) / 2
    M_t2 = (iv_p_t2 + iv_m_t2) / 2

    # --- Step 2: Retrieve Mean Reversion (kappa) ---
    # Formula: kappa = 2 * (S1 - S2) / (S1*t2 - S2*t1)    #Gatheral (2006) (The "Skew Term Structure" section)
    # Using the linear decay of skew term structure
    numerator_k = 2 * (S_t1 - S_t2)
    denominator_k = (S_t1 * t2) - (S_t2 * t1)

    if abs(denominator_k) < 1e-8
        error("Skew term structure is flat or invalid; cannot retrieve kappa.")
    end
    kappa = numerator_k / denominator_k

    # --- Step 3: Retrieve Rho-Sigma Product (A) ---
    # Extrapolate Skew to t=0
    S_0 = S_t1 / (1 - (kappa * t1) / 2)

    # Theoretical S(0) = (rho * sigma) / (4 * sqrt(v0))  Forde and Jacquier The small-time smile and term structure of implied volatility under the Heston model
    # Let A = rho * sigma
    A = 4 * sqrt(v0) * S_0

    # --- Step 4: Retrieve Long-Run Variance (theta) ---
    # Calculate slope of the ATM level term structure
    slope_M = (M_t2 - M_t1) / (t2 - t1)

    # Formula derived from M(t) expansion slope
    # slope_M = kappa * (theta - v0) / (4 * sqrt(v0))
    theta = v0 + (slope_M * 4 * sqrt(v0) / kappa)

    # --- Step 5: Retrieve Vol-of-Vol (sigma) and Correlation (rho) ---
    # Extrapolate Level to t=0 to find convexity bias
    M_0 = M_t1 - slope_M * t1

    # Theoretical M(0) = sqrt(v0) + 0.5 * Curvature * x0^2
    curvature_0 = 2 * (M_0 - sqrt(v0)) / (x0^2)

    # Define constant B based on curvature expansion  Forde, Jacquier & Lee (2012)
    # B = 24 * v0^(1.5) * C(0) = sigma^2 * (2 - 5*rho^2)
    B = 24 * (v0^1.5) * curvature_0

    # Solve system:
    # 1) rho * sigma = A
    # 2) 2*sigma^2 - 5*A^2 = B  (Substitute rho = A/sigma into curvature eq)

    sigma_sq = (B + 5 * A^2) / 2

    if sigma_sq < sigma_min^2
        @warn "Negative sigma squared detected ($sigma_sq). Data may violate Heston constraints."
        sigma_sq = sigma_min^2
    end
    sigma = sqrt(sigma_sq)
    rho = A / sigma

    # Clamp rho if it slightly exceeds boundaries due to numerical noise
    if !isnan(rho)
        rho = clamp(rho, -0.95, 0.95)
    end

    return HestonParams(v0, kappa, theta, rho, sigma)
end



struct LPPHestonExpansion{TV}
    coeffs::Vector{TV}
end
function LPPHestonExpansion(p::HestonParams{TV}, T; order=3) where {TV}
    ekt = exp(p.κ * T)
    coeffs = if order == 2
        [
            lpp2z0(p, T, ekt),
            lpp2z1(p, T, ekt),
            lpp2z2(p, T, ekt),
            zero(T)
        ]
    else
        [
            lpp3z0(p, T, ekt),
            lpp3z1(p, T, ekt),
            lpp3z2(p, T, ekt),
            lpp3z3(p, T, ekt)
        ]
    end
    return LPPHestonExpansion(coeffs)
end


function makeCharFuncPricer(::Val{:LPP}, cf, t)
    lpp = LPPHestonExpansion(CharFuncPricing.model(cf), t)
end
function makeCharFuncPricer(::Val{:LPP2}, cf, t)
    lpp = LPPHestonExpansion(CharFuncPricing.model(cf), t, order=2)
end
function impliedVolatility(pricer::LPPHestonExpansion, isCall::Bool, strike, forward, t; minPrice=1e-16)
    return impliedVolatility(pricer, strike, forward)
end

using Optim
function computeHestonLPPGuess(input::FordeGuessInput5D; order=3, trace=false, lower=[1.0e-1, 1.0e-4, -0.999, 1.0e-2], upper=[10.0, 1.0, 0.1, 5.0], guess=[1.0, 0.05, -0.5, 0.5], kappa=0.0)
    strike = exp(input.X0)
    if kappa == 0
        function obj_fun(x)
            expansion1 = LPPHestonExpansion(HestonParams(input.V00, x[1], x[2], x[3], x[4]), input.T1, order=order)
            expansion2 = LPPHestonExpansion(HestonParams(input.V00, x[1], x[2], x[3], x[4]), input.T2, order=order)
            res = [impliedVolatility(expansion1, strike, 1.0)^2 - input.Vx0t1,
                impliedVolatility(expansion1, 1 / strike, 1.0)^2 - input.Vmx0t1,
                impliedVolatility(expansion2, strike, 1.0)^2 - input.Vx0t2,
                impliedVolatility(expansion2, 1 / strike, 1.0)^2 - input.Vmx0t2]
            return sum(res .^ 2)
        end

        res = Optim.optimize(obj_fun, lower, upper, guess, Optim.Fminbox())
        if trace
            println(res)
        end
        return HestonParams(input.V00, res.minimizer[1], res.minimizer[2], res.minimizer[3], res.minimizer[4])
    else
        function obj_fun_fixed_kappa(x)
            expansion1 = LPPHestonExpansion(HestonParams(input.V00, kappa, x[1], x[2], x[3]), input.T1, order=order)
            expansion2 = LPPHestonExpansion(HestonParams(input.V00, kappa, x[1], x[2], x[3]), input.T2, order=order)
            res = [impliedVolatility(expansion1, strike, 1.0)^2 - input.Vx0t1,
                impliedVolatility(expansion1, 1 / strike, 1.0)^2 - input.Vmx0t1,
                impliedVolatility(expansion2, strike, 1.0)^2 - input.Vx0t2,
                impliedVolatility(expansion2, 1 / strike, 1.0)^2 - input.Vmx0t2]
            return sum(res .^ 2)
        end
        res = Optim.optimize(obj_fun_fixed_kappa, lower[2:4], upper[2:4], guess[2:4], Optim.Fminbox())
        if trace
            println(res)
        end
        return HestonParams(input.V00, kappa, res.minimizer[1], res.minimizer[2], res.minimizer[3])
    end
end

function impliedVolatility(e::LPPHestonExpansion,
    strike,
    forward)
    x = log(strike / forward)
    vol = e.coeffs[1] + x * (e.coeffs[2] + x * (e.coeffs[3] + x * e.coeffs[4]))
    return max(vol, 1e-8)
end
@inline fastpow(x, n) = x^n
function lpp2z0(p::HestonParams, t, ekt)
    e2kt = ekt^2
    e3kt = ekt^3
    kappa = p.κ
    theta = p.θ
    delta = p.σ
    rho = p.ρ
    y = p.v0

    return (4 * fastpow(delta, 2) * kappa * (-theta - 4 * ekt * (theta + kappa * t * (theta - y)) + e2kt * ((5 - 2 * kappa * t) * theta - 2 * y) + 2 * y) * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) + 128 * ekt * fastpow(kappa, 3) * fastpow((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y, 2) + 32 * delta * ekt * fastpow(kappa, 2) * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) + fastpow(delta, 2) * ekt * fastpow(rho, 2) * (-theta + kappa * t * theta + (theta - y) / ekt + y) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 2) + (48 * fastpow(delta, 2) * e2kt * fastpow(kappa, 2) * fastpow(rho, 2) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 2)) / ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) - fastpow(delta, 2) * fastpow(rho, 2) * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 2) + 2 * fastpow(delta, 2) * kappa * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * (theta - 2 * y + e2kt * (-5 * theta + 2 * kappa * t * theta + 2 * y + 8 * fastpow(rho, 2) * ((-3 + kappa * t) * theta + y)) + 4 * ekt * (theta + kappa * t * theta - kappa * t * y + fastpow(rho, 2) * ((6 + kappa * t * (4 + kappa * t)) * theta - (2 + kappa * t * (2 + kappa * t)) * y))) - (8 * fastpow(delta, 2) * fastpow(kappa, 2) * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * (theta - 2 * y + e2kt * (-5 * theta + 2 * kappa * t * theta + 2 * y + 8 * fastpow(rho, 2) * ((-3 + kappa * t) * theta + y)) + 4 * ekt * (theta + kappa * t * theta - kappa * t * y + fastpow(rho, 2) * ((6 + kappa * t * (4 + kappa * t)) * theta - (2 + kappa * t * (2 + kappa * t)) * y)))) / (-theta + kappa * t * theta + (theta - y) / ekt + y)) / (128 * e3kt * fastpow(kappa, 5) * fastpow(t, 2) * fastpow((-theta + kappa * t * theta + (theta - y) / ekt + y) / (kappa * t), 1.5))

end
function lpp2z1(p::HestonParams, t, ekt)
    e2kt = ekt^2
    kappa = p.κ
    theta = p.θ
    delta = p.σ
    rho = p.ρ
    y = p.v0

    return (delta * rho * (-(delta * fastpow(-1 + ekt, 2) * rho * (4 * theta - y) * y) + 2 * ekt * fastpow(kappa, 3) * fastpow(t, 2) * theta * ((2 + 2 * ekt + delta * rho * t) * theta - (2 + delta * rho * t) * y) - 2 * (-1 + ekt) * kappa * (2 * theta - y) * ((-1 + ekt) * (-2 + delta * rho * t) * theta + (-2 + 2 * ekt + delta * rho * t) * y) + fastpow(kappa, 2) * t * ((-1 + ekt) * (-4 + delta * rho * t + ekt * (-12 + delta * rho * t)) * fastpow(theta, 2) + 2 * (-4 + 4 * e2kt + delta * rho * t + 3 * delta * ekt * rho * t) * theta * y - (-4 + delta * rho * t + 2 * ekt * (2 + delta * rho * t)) * fastpow(y, 2)))) / (8 * fastpow(kappa, 2) * t * sqrt((-theta + kappa * t * theta + (theta - y) / ekt + y) / (kappa * t)) * fastpow((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y, 2))

end
function lpp2z2(p::HestonParams, t, ekt)
    e2kt = ekt^2
    kappa = p.κ
    theta = p.θ
    delta = p.σ
    rho = p.ρ
    y = p.v0

    return (fastpow(delta, 2) * sqrt((-theta + kappa * t * theta + (theta - y) / ekt + y) / (kappa * t)) * (-12 * fastpow(rho, 2) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 2) +
                                                                                                            (-theta + kappa * t * theta + (theta - y) / ekt + y) * (theta - 2 * y + e2kt * (-5 * theta + 2 * kappa * t * theta + 2 * y + 8 * fastpow(rho, 2) * ((-3 + kappa * t) * theta + y)) + 4 * ekt * (theta + kappa * t * theta - kappa * t * y + fastpow(rho, 2) * ((6 + kappa * t * (4 + kappa * t)) * theta - (2 + kappa * t * (2 + kappa * t)) * y))))) / (16 * e2kt * fastpow(-theta + kappa * t * theta + (theta - y) / ekt + y, 4))
end



function lpp3z0(p::HestonParams, t, ekt)
    e2kt = ekt^2
    e3kt = ekt^3
    e4kt = ekt^4
    kappa = p.κ
    theta = p.θ
    delta = p.σ
    rho = p.ρ
    y = p.v0

    return (
        96 * fastpow(delta, 2) * ekt * fastpow(kappa, 3) * (-theta - 4 * ekt * (theta + kappa * t * (theta - y)) + e2kt * ((5 - 2 * kappa * t) * theta - 2 * y) + 2 * y) * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) + 3072 * e2kt * fastpow(kappa, 5) * fastpow((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y, 2) + 96 * fastpow(delta, 3) * ekt * fastpow(kappa, 2) * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * (-2 * theta - kappa * t * theta - 2 * ekt * (2 + kappa * t) * (2 * theta + kappa * t * (theta - y)) + e2kt * ((10 - 3 * kappa * t) * theta - 3 * y) + 3 * y + 2 * kappa * t * y) + 768 * delta * e2kt * fastpow(kappa, 4) * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) + 6 * fastpow(delta, 3) * kappa * rho * (-theta - 4 * ekt * (theta + kappa * t * (theta - y)) + e2kt * ((5 - 2 * kappa * t) * theta - 2 * y) + 2 * y) * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) + 24 * fastpow(delta, 2) * e2kt * fastpow(kappa, 2) * fastpow(rho, 2) * (-theta + kappa * t * theta + (theta - y) / ekt + y) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 2) + (1152 * fastpow(delta, 2) * e3kt * fastpow(kappa, 4) * fastpow(rho, 2) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 2)) / ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) - 24 * fastpow(delta, 2) * ekt * fastpow(kappa, 2) * fastpow(rho, 2) * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 2) + 80 * fastpow(delta, 3) * ekt * kappa * fastpow(rho, 3) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 3) + fastpow(delta, 3) * ekt * fastpow(rho, 3) * (-theta + kappa * t * theta + (theta - y) / ekt + y) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 3) - (1440 * fastpow(delta, 3) * e3kt * fastpow(kappa, 3) * fastpow(rho, 3) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 3)) / fastpow((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y, 2) - (528 * fastpow(delta, 3) * e2kt * fastpow(kappa, 2) * fastpow(rho, 3) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 3)) / ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) - 3 * fastpow(delta, 3) * fastpow(rho, 3) * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 3) + 384 * fastpow(delta, 3) * e2kt * fastpow(kappa, 3) * rho * ((2 + kappa * t + 2 * ekt * fastpow(2 + kappa * t, 2) + e2kt * (-10 + 3 * kappa * t)) * theta + (-3 + 3 * e2kt - 2 * kappa * t - 2 * ekt * kappa * t * (2 + kappa * t)) * y) - (576 * fastpow(delta, 3) * e2kt * fastpow(kappa, 3) * rho * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) * ((1 + e2kt * (-5 + 2 * kappa * t + 4 * fastpow(rho, 2) * (-3 + kappa * t)) + 2 * ekt * (2 + 2 * kappa * t + fastpow(rho, 2) * (6 + 4 * kappa * t + fastpow(kappa, 2) * fastpow(t, 2)))) * theta + 2 * (-1 + e2kt * (1 + 2 * fastpow(rho, 2)) - ekt * (2 * kappa * t + fastpow(rho, 2) * (2 + 2 * kappa * t + fastpow(kappa, 2) * fastpow(t, 2)))) * y)) / ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) + fastpow(delta, 3) * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) * (theta * (12 * ekt * fastpow(kappa, 3) * fastpow(rho, 2) * fastpow(t, 2) + 8 * fastpow(-1 + ekt, 2) * fastpow(rho, 2) * theta - (-1 + ekt) * kappa * (3 + 8 * fastpow(rho, 2) * t * theta + ekt * (15 + 8 * fastpow(rho, 2) * (9 + t * theta))) + 2 * fastpow(kappa, 2) * t * (fastpow(rho, 2) * t * theta + 2 * ekt * (3 + fastpow(rho, 2) * (12 + t * theta)) + e2kt * (3 + fastpow(rho, 2) * (12 + t * theta)))) - 2 * (6 * ekt * fastpow(kappa, 3) * fastpow(rho, 2) * fastpow(t, 2) + 4 * fastpow(-1 + ekt, 2) * fastpow(rho, 2) * theta + 2 * fastpow(kappa, 2) * t * (fastpow(rho, 2) * t * theta + ekt * (3 + fastpow(rho, 2) * (6 + t * theta))) - (-1 + ekt) * kappa * (3 + 6 * fastpow(rho, 2) * t * theta + ekt * (3 + 2 * fastpow(rho, 2) * (6 + t * theta)))) * y + 2 * fastpow(rho, 2) * fastpow(1 - ekt + kappa * t, 2) * fastpow(y, 2)) - (40 * fastpow(delta, 3) * kappa * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) * (theta * (12 * ekt * fastpow(kappa, 3) * fastpow(rho, 2) * fastpow(t, 2) + 8 * fastpow(-1 + ekt, 2) * fastpow(rho, 2) * theta - (-1 + ekt) * kappa * (3 + 8 * fastpow(rho, 2) * t * theta + ekt * (15 + 8 * fastpow(rho, 2) * (9 + t * theta))) + 2 * fastpow(kappa, 2) * t * (fastpow(rho, 2) * t * theta + 2 * ekt * (3 + fastpow(rho, 2) * (12 + t * theta)) + e2kt * (3 + fastpow(rho, 2) * (12 + t * theta)))) - 2 * (6 * ekt * fastpow(kappa, 3) * fastpow(rho, 2) * fastpow(t, 2) + 4 * fastpow(-1 + ekt, 2) * fastpow(rho, 2) * theta + 2 * fastpow(kappa, 2) * t * (fastpow(rho, 2) * t * theta + ekt * (3 + fastpow(rho, 2) * (6 + t * theta))) - (-1 + ekt) * kappa * (3 + 6 * fastpow(rho, 2) * t * theta + ekt * (3 + 2 * fastpow(rho, 2) * (6 + t * theta)))) * y + 2 * fastpow(rho, 2) * fastpow(1 - ekt + kappa * t, 2) * fastpow(y, 2))) / (-theta + kappa * t * theta + (theta - y) / ekt + y) - 12 * fastpow(delta, 3) * kappa * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * (2 * theta + kappa * t * theta - y - kappa * t * y + ekt * ((-2 + kappa * t) * theta + y)) * (theta - 2 * y + e2kt * (-5 * theta + 2 * kappa * t * theta + 2 * y + 4 * fastpow(rho, 2) * ((-3 + kappa * t) * theta + y)) + 2 * ekt * (2 * (theta + kappa * t * (theta - y)) + fastpow(rho, 2) * ((6 + kappa * t * (4 + kappa * t)) * theta - (2 + kappa * t * (2 + kappa * t)) * y))) + (288 * fastpow(delta, 3) * fastpow(kappa, 2) * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * (2 * theta + kappa * t * theta - y - kappa * t * y + ekt * ((-2 + kappa * t) * theta + y)) * (theta - 2 * y + e2kt * (-5 * theta + 2 * kappa * t * theta + 2 * y + 4 * fastpow(rho, 2) * ((-3 + kappa * t) * theta + y)) + 2 * ekt * (2 * (theta + kappa * t * (theta - y)) + fastpow(rho, 2) * ((6 + kappa * t * (4 + kappa * t)) * theta - (2 + kappa * t * (2 + kappa * t)) * y)))) / (-theta + kappa * t * theta + (theta - y) / ekt + y) + 48 * fastpow(delta, 2) * ekt * fastpow(kappa, 3) * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * (theta - 2 * y + e2kt * (-5 * theta + 2 * kappa * t * theta + 2 * y + 8 * fastpow(rho, 2) * ((-3 + kappa * t) * theta + y)) + 4 * ekt * (theta + kappa * t * theta - kappa * t * y + fastpow(rho, 2) * ((6 + kappa * t * (4 + kappa * t)) * theta - (2 + kappa * t * (2 + kappa * t)) * y))) - (192 * fastpow(delta, 2) * ekt * fastpow(kappa, 4) * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * (theta - 2 * y + e2kt * (-5 * theta + 2 * kappa * t * theta + 2 * y + 8 * fastpow(rho, 2) * ((-3 + kappa * t) * theta + y)) + 4 * ekt * (theta + kappa * t * theta - kappa * t * y + fastpow(rho, 2) * ((6 + kappa * t * (4 + kappa * t)) * theta - (2 + kappa * t * (2 + kappa * t)) * y)))) / (-theta + kappa * t * theta + (theta - y) / ekt + y) + 3 * fastpow(delta, 3) * kappa * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) * (theta - 2 * y + e2kt * (-5 * theta + 2 * kappa * t * theta + 2 * y + 8 * fastpow(rho, 2) * ((-3 + kappa * t) * theta + y)) + 4 * ekt * (theta + kappa * t * theta - kappa * t * y + fastpow(rho, 2) * ((6 + kappa * t * (4 + kappa * t)) * theta - (2 + kappa * t * (2 + kappa * t)) * y))) - (12 * fastpow(delta, 3) * fastpow(kappa, 2) * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) * (theta - 2 * y + e2kt * (-5 * theta + 2 * kappa * t * theta + 2 * y + 8 * fastpow(rho, 2) * ((-3 + kappa * t) * theta + y)) + 4 * ekt * (theta + kappa * t * theta - kappa * t * y + fastpow(rho, 2) * ((6 + kappa * t * (4 + kappa * t)) * theta - (2 + kappa * t * (2 + kappa * t)) * y)))) / (-theta + kappa * t * theta + (theta - y) / ekt + y) + 4 * fastpow(delta, 3) * kappa * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * (3 * (theta - 2 * y) * ((2 + kappa * t) * theta - (1 + kappa * t) * y) + 3 * ekt * (6 * fastpow(theta, 2) + theta * y - 2 * fastpow(y, 2) + kappa * (13 * t * fastpow(theta, 2) + theta * (8 - 18 * t * y) + 4 * y * (-3 + t * y)) + 4 * fastpow(kappa, 2) * t * (theta + t * fastpow(theta, 2) - 2 * t * theta * y + y * (-2 + t * y))) + 3 * e3kt * (10 * fastpow(theta, 2) + 2 * fastpow(kappa, 2) * t * theta * (6 + 8 * fastpow(rho, 2) + t * theta) - 9 * theta * y + 2 * fastpow(y, 2) + kappa * (-9 * t * fastpow(theta, 2) + 4 * (3 + 4 * fastpow(rho, 2)) * y + theta * (-40 - 64 * fastpow(rho, 2) + 4 * t * y))) + e2kt * (-54 * fastpow(theta, 2) + 8 * fastpow(kappa, 4) * fastpow(rho, 2) * fastpow(t, 3) * (theta - y) + 39 * theta * y - 6 * fastpow(y, 2) + 24 * fastpow(kappa, 3) * fastpow(t, 2) * (theta + 2 * fastpow(rho, 2) * theta - (1 + fastpow(rho, 2)) * y) + 6 * fastpow(kappa, 2) * t * (3 * t * fastpow(theta, 2) - 8 * (1 + fastpow(rho, 2)) * y + theta * (16 + 24 * fastpow(rho, 2) - 3 * t * y)) - 3 * kappa * (5 * t * fastpow(theta, 2) + 2 * y * (8 * fastpow(rho, 2) + 3 * t * y) - theta * (32 + 64 * fastpow(rho, 2) + 17 * t * y)))) -
        (48 * fastpow(delta, 3) * fastpow(kappa, 2) * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * (3 * (theta - 2 * y) * ((2 + kappa * t) * theta - (1 + kappa * t) * y) + 3 * ekt * (6 * fastpow(theta, 2) + theta * y - 2 * fastpow(y, 2) + kappa * (13 * t * fastpow(theta, 2) + theta * (8 - 18 * t * y) + 4 * y * (-3 + t * y)) + 4 * fastpow(kappa, 2) * t * (theta + t * fastpow(theta, 2) - 2 * t * theta * y + y * (-2 + t * y))) + 3 * e3kt * (10 * fastpow(theta, 2) + 2 * fastpow(kappa, 2) * t * theta * (6 + 8 * fastpow(rho, 2) + t * theta) - 9 * theta * y + 2 * fastpow(y, 2) + kappa * (-9 * t * fastpow(theta, 2) + 4 * (3 + 4 * fastpow(rho, 2)) * y + theta * (-40 - 64 * fastpow(rho, 2) + 4 * t * y))) + e2kt * (-54 * fastpow(theta, 2) + 8 * fastpow(kappa, 4) * fastpow(rho, 2) * fastpow(t, 3) * (theta - y) + 39 * theta * y - 6 * fastpow(y, 2) + 24 * fastpow(kappa, 3) * fastpow(t, 2) * (theta + 2 * fastpow(rho, 2) * theta - (1 + fastpow(rho, 2)) * y) + 6 * fastpow(kappa, 2) * t * (3 * t * fastpow(theta, 2) - 8 * (1 + fastpow(rho, 2)) * y + theta * (16 + 24 * fastpow(rho, 2) - 3 * t * y)) - 3 * kappa * (5 * t * fastpow(theta, 2) + 2 * y * (8 * fastpow(rho, 2) + 3 * t * y) - theta * (32 + 64 * fastpow(rho, 2) + 17 * t * y))))) / (-theta + kappa * t * theta + (theta - y) / ekt + y) + (240 * fastpow(delta, 3) * e2kt * fastpow(kappa, 2) * rho * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) * (12 * ekt * fastpow(kappa, 3) * fastpow(rho, 2) * fastpow(t, 2) * (theta - y) + 2 * fastpow(-1 + ekt, 2) * fastpow(rho, 2) * fastpow(-2 * theta + y, 2) - (-1 + ekt) * kappa * (8 * (1 + ekt) * fastpow(rho, 2) * t * fastpow(theta, 2) + 2 * y * (-3 - 3 * ekt * (1 + 4 * fastpow(rho, 2)) + 2 * fastpow(rho, 2) * t * y) + theta * (3 - 12 * fastpow(rho, 2) * t * y + ekt * (15 + fastpow(rho, 2) * (72 - 4 * t * y)))) + 2 * fastpow(kappa, 2) * t * (e2kt * theta * (3 + fastpow(rho, 2) * (12 + t * theta)) + fastpow(rho, 2) * t * fastpow(theta - y, 2) + 2 * ekt * (fastpow(rho, 2) * t * fastpow(theta, 2) - 3 * (y + 2 * fastpow(rho, 2) * y) + theta * (3 + fastpow(rho, 2) * (12 - t * y)))))) / ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y)
    ) / (3072. * e4kt * fastpow(kappa, 7) * fastpow(t, 2) * fastpow((-theta + kappa * t * theta + (theta - y) / ekt + y) / (kappa * t), 1.5))
end

function lpp3z1(p::HestonParams, t, ekt)
    e2kt = ekt^2
    e3kt = ekt^3

    kappa = p.κ
    theta = p.θ
    delta = p.σ
    rho = p.ρ
    y = p.v0
    return (delta * (768 * e2kt * fastpow(kappa, 4) * rho * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) - (576 * delta * e2kt * fastpow(kappa, 3) * fastpow(rho, 2) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 2)) / ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) - 10 * fastpow(delta, 2) * fastpow(rho, 3) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 3) + (6 * fastpow(delta, 2) * kappa * fastpow(rho, 3) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 3)) / (-theta + kappa * t * theta + (theta - y) / ekt + y) - (3360 * fastpow(delta, 2) * e3kt * fastpow(kappa, 3) * fastpow(rho, 3) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 3)) / fastpow((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y, 3) - (288 * fastpow(delta, 2) * e2kt * fastpow(kappa, 2) * fastpow(rho, 3) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 3)) / fastpow((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y, 2) + (234 * fastpow(delta, 2) * ekt * kappa * fastpow(rho, 3) * fastpow((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y, 3)) / ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) - 96 * delta * ekt * fastpow(kappa, 3) * ((1 + 4 * ekt * (1 + kappa * t) + e2kt * (-5 + 2 * kappa * t)) * theta + 2 * (-1 + e2kt - 2 * ekt * kappa * t) * y) - 12 * fastpow(delta, 2) * kappa * rho * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) * ((1 + 4 * ekt * (1 + kappa * t) + e2kt * (-5 + 2 * kappa * t)) * theta + 2 * (-1 + e2kt - 2 * ekt * kappa * t) * y) - 192 * fastpow(delta, 2) * ekt * fastpow(kappa, 2) * rho * ((2 + kappa * t + 2 * ekt * fastpow(2 + kappa * t, 2) + e2kt * (-10 + 3 * kappa * t)) * theta + (-3 + 3 * e2kt - 2 * kappa * t - 2 * ekt * kappa * t * (2 + kappa * t)) * y) - (12 * fastpow(delta, 2) * ekt * fastpow(kappa, 2) * rho * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) * ((1 + e2kt * (-5 + 2 * kappa * t + 8 * fastpow(rho, 2) * (-3 + kappa * t)) + 4 * ekt * (1 + kappa * t + fastpow(rho, 2) * (6 + 4 * kappa * t + fastpow(kappa, 2) * fastpow(t, 2)))) * theta + 2 * (-1 + e2kt * (1 + 4 * fastpow(rho, 2)) - 2 * ekt * (kappa * t + fastpow(rho, 2) * (2 + 2 * kappa * t + fastpow(kappa, 2) * fastpow(t, 2)))) * y)) / ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) + (576 * fastpow(delta, 2) * ekt * fastpow(kappa, 2) * rho * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) * ((1 + e2kt * (-5 + 2 * kappa * t + 4 * fastpow(rho, 2) * (-3 + kappa * t)) + 2 * ekt * (2 + 2 * kappa * t + fastpow(rho, 2) * (6 + 4 * kappa * t + fastpow(kappa, 2) * fastpow(t, 2)))) * theta + 2 * (-1 + e2kt * (1 + 2 * fastpow(rho, 2)) - ekt * (2 * kappa * t + fastpow(rho, 2) * (2 + 2 * kappa * t + fastpow(kappa, 2) * fastpow(t, 2)))) * y)) / ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) + (5 * fastpow(delta, 2) * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) * (theta * (12 * ekt * fastpow(kappa, 3) * fastpow(rho, 2) * fastpow(t, 2) + 8 * fastpow(-1 + ekt, 2) * fastpow(rho, 2) * theta - (-1 + ekt) * kappa * (3 + 8 * fastpow(rho, 2) * t * theta + ekt * (15 + 8 * fastpow(rho, 2) * (9 + t * theta))) + 2 * fastpow(kappa, 2) * t * (fastpow(rho, 2) * t * theta + 2 * ekt * (3 + fastpow(rho, 2) * (12 + t * theta)) + e2kt * (3 + fastpow(rho, 2) * (12 + t * theta)))) - 2 * (6 * ekt * fastpow(kappa, 3) * fastpow(rho, 2) * fastpow(t, 2) + 4 * fastpow(-1 + ekt, 2) * fastpow(rho, 2) * theta + 2 * fastpow(kappa, 2) * t * (fastpow(rho, 2) * t * theta + ekt * (3 + fastpow(rho, 2) * (6 + t * theta))) - (-1 + ekt) * kappa * (3 + 6 * fastpow(rho, 2) * t * theta + ekt * (3 + 2 * fastpow(rho, 2) * (6 + t * theta)))) * y + 2 * fastpow(rho, 2) * fastpow(1 - ekt + kappa * t, 2) * fastpow(y, 2))) / (ekt * (-theta + kappa * t * theta + (theta - y) / ekt + y)) - (48 * fastpow(delta, 2) * kappa * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * (2 * theta + kappa * t * theta - y - kappa * t * y + ekt * ((-2 + kappa * t) * theta + y)) * (theta - 2 * y + e2kt * (-5 * theta + 2 * kappa * t * theta + 2 * y + 4 * fastpow(rho, 2) * ((-3 + kappa * t) * theta + y)) + 2 * ekt * (2 * (theta + kappa * t * (theta - y)) + fastpow(rho, 2) * ((6 + kappa * t * (4 + kappa * t)) * theta - (2 + kappa * t * (2 + kappa * t)) * y)))) / (ekt * (-theta + kappa * t * theta + (theta - y) / ekt + y)) + (96 * delta * fastpow(kappa, 3) * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * (theta - 2 * y + e2kt * (-5 * theta + 2 * kappa * t * theta + 2 * y + 8 * fastpow(rho, 2) * ((-3 + kappa * t) * theta + y)) + 4 * ekt * (theta + kappa * t * theta - kappa * t * y + fastpow(rho, 2) * ((6 + kappa * t * (4 + kappa * t)) * theta - (2 + kappa * t * (2 + kappa * t)) * y)))) / (-theta + kappa * t * theta + (theta - y) / ekt + y) + (9 * fastpow(delta, 2) * kappa * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) * (theta - 2 * y + e2kt * (-5 * theta + 2 * kappa * t * theta + 2 * y + 8 * fastpow(rho, 2) * ((-3 + kappa * t) * theta + y)) + 4 * ekt * (theta + kappa * t * theta - kappa * t * y + fastpow(rho, 2) * ((6 + kappa * t * (4 + kappa * t)) * theta - (2 + kappa * t * (2 + kappa * t)) * y)))) / (ekt * (-theta + kappa * t * theta + (theta - y) / ekt + y)) - (48 * fastpow(delta, 2) * ekt * fastpow(kappa, 2) * rho * (3 * (theta - 2 * y) * ((2 + kappa * t) * theta - (1 + kappa * t) * y) + 3 * ekt * (6 * fastpow(theta, 2) + theta * y - 2 * fastpow(y, 2) + kappa * (13 * t * fastpow(theta, 2) + theta * (8 - 18 * t * y) + 4 * y * (-3 + t * y)) + 4 * fastpow(kappa, 2) * t * (theta + t * fastpow(theta, 2) - 2 * t * theta * y + y * (-2 + t * y))) + 3 * e3kt * (10 * fastpow(theta, 2) + 2 * fastpow(kappa, 2) * t * theta * (6 + 8 * fastpow(rho, 2) + t * theta) - 9 * theta * y + 2 * fastpow(y, 2) + kappa * (-9 * t * fastpow(theta, 2) + 4 * (3 + 4 * fastpow(rho, 2)) * y + theta * (-40 - 64 * fastpow(rho, 2) + 4 * t * y))) + e2kt * (-54 * fastpow(theta, 2) + 8 * fastpow(kappa, 4) * fastpow(rho, 2) * fastpow(t, 3) * (theta - y) + 39 * theta * y - 6 * fastpow(y, 2) + 24 * fastpow(kappa, 3) * fastpow(t, 2) * (theta + 2 * fastpow(rho, 2) * theta - (1 + fastpow(rho, 2)) * y) + 6 * fastpow(kappa, 2) * t * (3 * t * fastpow(theta, 2) - 8 * (1 + fastpow(rho, 2)) * y + theta * (16 + 24 * fastpow(rho, 2) - 3 * t * y)) - 3 * kappa * (5 * t * fastpow(theta, 2) + 2 * y * (8 * fastpow(rho, 2) + 3 * t * y) - theta * (32 + 64 * fastpow(rho, 2) + 17 * t * y))))) / ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) + (12 * fastpow(delta, 2) * kappa * rho * ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y) * (3 * (theta - 2 * y) * ((2 + kappa * t) * theta - (1 + kappa * t) * y) + 3 * ekt * (6 * fastpow(theta, 2) + theta * y - 2 * fastpow(y, 2) + kappa * (13 * t * fastpow(theta, 2) + theta * (8 - 18 * t * y) + 4 * y * (-3 + t * y)) + 4 * fastpow(kappa, 2) * t * (theta + t * fastpow(theta, 2) - 2 * t * theta * y + y * (-2 + t * y))) + 3 * e3kt * (10 * fastpow(theta, 2) + 2 * fastpow(kappa, 2) * t * theta * (6 + 8 * fastpow(rho, 2) + t * theta) - 9 * theta * y + 2 * fastpow(y, 2) + kappa * (-9 * t * fastpow(theta, 2) + 4 * (3 + 4 * fastpow(rho, 2)) * y + theta * (-40 - 64 * fastpow(rho, 2) + 4 * t * y))) + e2kt * (-54 * fastpow(theta, 2) + 8 * fastpow(kappa, 4) * fastpow(rho, 2) * fastpow(t, 3) * (theta - y) + 39 * theta * y - 6 * fastpow(y, 2) + 24 * fastpow(kappa, 3) * fastpow(t, 2) * (theta + 2 * fastpow(rho, 2) * theta - (1 + fastpow(rho, 2)) * y) + 6 * fastpow(kappa, 2) * t * (3 * t * fastpow(theta, 2) - 8 * (1 + fastpow(rho, 2)) * y + theta * (16 + 24 * fastpow(rho, 2) - 3 * t * y)) - 3 * kappa * (5 * t * fastpow(theta, 2) + 2 * y * (8 * fastpow(rho, 2) + 3 * t * y) - theta * (32 + 64 * fastpow(rho, 2) + 17 * t * y))))) / (ekt * (-theta + kappa * t * theta + (theta - y) / ekt + y)) + (240 * fastpow(delta, 2) * e2kt * fastpow(kappa, 2) * rho * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) * (12 * ekt * fastpow(kappa, 3) * fastpow(rho, 2) * fastpow(t, 2) * (theta - y) + 2 * fastpow(-1 + ekt, 2) * fastpow(rho, 2) * fastpow(-2 * theta + y, 2) - (-1 + ekt) * kappa * (8 * (1 + ekt) * fastpow(rho, 2) * t * fastpow(theta, 2) + 2 * y * (-3 - 3 * ekt * (1 + 4 * fastpow(rho, 2)) + 2 * fastpow(rho, 2) * t * y) + theta * (3 - 12 * fastpow(rho, 2) * t * y + ekt * (15 + fastpow(rho, 2) * (72 - 4 * t * y)))) + 2 * fastpow(kappa, 2) * t * (e2kt * theta * (3 + fastpow(rho, 2) * (12 + t * theta)) + fastpow(rho, 2) * t * fastpow(theta - y, 2) + 2 * ekt * (fastpow(rho, 2) * t * fastpow(theta, 2) - 3 * (y + 2 * fastpow(rho, 2) * y) + theta * (3 + fastpow(rho, 2) * (12 - t * y)))))) / fastpow((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y, 2) - (120 * fastpow(delta, 2) * ekt * kappa * rho * ((2 + kappa * t + ekt * (-2 + kappa * t)) * theta + (-1 + ekt - kappa * t) * y) * (12 * ekt * fastpow(kappa, 3) * fastpow(rho, 2) * fastpow(t, 2) * (theta - y) + 2 * fastpow(-1 + ekt, 2) * fastpow(rho, 2) * fastpow(-2 * theta + y, 2) - (-1 + ekt) * kappa * (8 * (1 + ekt) * fastpow(rho, 2) * t * fastpow(theta, 2) + 2 * y * (-3 - 3 * ekt * (1 + 4 * fastpow(rho, 2)) + 2 * fastpow(rho, 2) * t * y) + theta * (3 - 12 * fastpow(rho, 2) * t * y + ekt * (15 + fastpow(rho, 2) * (72 - 4 * t * y)))) + 2 * fastpow(kappa, 2) * t * (e2kt * theta * (3 + fastpow(rho, 2) * (12 + t * theta)) + fastpow(rho, 2) * t * fastpow(theta - y, 2) + 2 * ekt * (fastpow(rho, 2) * t * fastpow(theta, 2) - 3 * (y + 2 * fastpow(rho, 2) * y) + theta * (3 + fastpow(rho, 2) * (12 - t * y)))))) / ((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y))) /
           (1536. * e3kt * fastpow(kappa, 6) * fastpow(t, 2) * fastpow((-theta + kappa * t * theta + (theta - y) / ekt + y) / (kappa * t), 1.5))
end

function lpp3z2(p::HestonParams, t, ekt)
    e2kt = ekt^2
    e3kt = ekt^3
    kappa = p.κ
    theta = p.θ
    delta = p.σ
    rho = p.ρ
    y = p.v0
    return (fastpow(delta, 2) * (8 * e3kt * fastpow(kappa, 5) * fastpow(rho, 2) * fastpow(t, 4) * (2 + delta * rho * t) * fastpow(theta, 2) * (theta - y) - delta * fastpow(-1 + ekt, 3) * rho * (2 * (-1 + ekt * (-5 + 24 * fastpow(rho, 2))) * fastpow(theta, 3) + (7 + ekt * (3 + 56 * fastpow(rho, 2))) * fastpow(theta, 2) * y - 3 * (1 + ekt * (-3 + 8 * fastpow(rho, 2))) * theta * fastpow(y, 2) + 2 * (-1 + ekt * (-1 + 2 * fastpow(rho, 2))) * fastpow(y, 3)) - fastpow(-1 + ekt, 2) * kappa * ((-4 + delta * rho * t - 8 * ekt * (2 - 12 * fastpow(rho, 2) - 4 * delta * rho * t + 25 * delta * fastpow(rho, 3) * t) + e2kt * (20 - 96 * fastpow(rho, 2) + 3 * delta * rho * t + 56 * delta * fastpow(rho, 3) * t)) * fastpow(theta, 3) - 2 * (-8 + 2 * delta * rho * t + e2kt * (24 - 80 * fastpow(rho, 2) - 9 * delta * rho * t + 24 * delta * fastpow(rho, 3) * t) - 4 * ekt * (4 - 20 * fastpow(rho, 2) - 10 * delta * rho * t + 39 * delta * fastpow(rho, 3) * t)) * fastpow(theta, 2) * y + (5 * (-4 + delta * rho * t) + ekt * (-16 + 80 * fastpow(rho, 2) + 57 * delta * rho * t - 140 * delta * fastpow(rho, 3) * t) + 2 * e2kt * (18 - 40 * fastpow(rho, 2) - 3 * delta * rho * t + 6 * delta * fastpow(rho, 3) * t)) * theta * fastpow(y, 2) + 2 * (4 + e2kt * (-4 + 8 * fastpow(rho, 2)) - delta * rho * t + ekt * rho * (-8 * rho - 7 * delta * t + 14 * delta * fastpow(rho, 2) * t)) * fastpow(y, 3)) + ekt * (-1 + ekt) * fastpow(kappa, 2) * t * ((-24 + 128 * fastpow(rho, 2) + 9 * delta * rho * t - 144 * delta * fastpow(rho, 3) * t - 4 * ekt * (6 - 8 * fastpow(rho, 2) - 9 * delta * rho * t + 6 * delta * fastpow(rho, 3) * t) + e2kt * (48 - 160 * fastpow(rho, 2) - 9 * delta * rho * t + 24 * delta * fastpow(rho, 3) * t)) * fastpow(theta, 3) - (-72 + 320 * fastpow(rho, 2) + 27 * delta * rho * t - 360 * delta * fastpow(rho, 3) * t - ekt * rho * (160 * rho - 81 * delta * t + 348 * delta * fastpow(rho, 2) * t) + 2 * e2kt * (36 - 80 * fastpow(rho, 2) - 3 * delta * rho * t + 6 * delta * fastpow(rho, 3) * t)) * fastpow(theta, 2) * y - 2 * (32 - 128 * fastpow(rho, 2) + 12 * e2kt * (-1 + 2 * fastpow(rho, 2)) - 15 * delta * rho * t + 144 * delta * fastpow(rho, 3) * t + 2 * ekt * (-10 + 52 * fastpow(rho, 2) - 13 * delta * rho * t + 58 * delta * fastpow(rho, 3) * t)) * theta * fastpow(y, 2) + 4 * (4 - 16 * fastpow(rho, 2) - 3 * delta * rho * t + 18 * delta * fastpow(rho, 3) * t + ekt * (-4 + 16 * fastpow(rho, 2) - 2 * delta * rho * t + 11 * delta * fastpow(rho, 3) * t)) * fastpow(y, 3)) - 4 * e2kt * fastpow(kappa, 4) * fastpow(t, 3) * theta * (2 * e2kt * (-1 + 2 * fastpow(rho, 2)) * fastpow(theta, 2) + fastpow(rho, 2) * (4 + 13 * delta * rho * t) * fastpow(theta - y, 2) + ekt * ((-4 + 16 * fastpow(rho, 2) - 2 * delta * rho * t + 9 * delta * fastpow(rho, 3) * t) * fastpow(theta, 2) + (4 - 32 * fastpow(rho, 2) + 2 * delta * rho * t - 19 * delta * fastpow(rho, 3) * t) * theta * y + 4 * fastpow(rho, 2) * (2 + delta * rho * t) * fastpow(y, 2))) - 2 * ekt * fastpow(kappa, 3) * fastpow(t, 2) * (-4 * fastpow(rho, 2) * (-4 + 3 * delta * rho * t) * fastpow(theta - y, 3) + e3kt * fastpow(theta, 2) * ((18 - 40 * fastpow(rho, 2) - delta * rho * t + 2 * delta * fastpow(rho, 3) * t) * theta + 12 * (-1 + 2 * fastpow(rho, 2)) * y) + 2 * ekt * ((-9 + 36 * fastpow(rho, 2) + 19 * delta * fastpow(rho, 3) * t) * fastpow(theta, 3) + 2 * (9 - 30 * fastpow(rho, 2) + 7 * delta * fastpow(rho, 3) * t) * fastpow(theta, 2) * y + (-8 + 20 * fastpow(rho, 2) + delta * rho * t - 46 * delta * fastpow(rho, 3) * t) * theta * fastpow(y, 2) + fastpow(rho, 2) * (4 + 13 * delta * rho * t) * fastpow(y, 3)) + e2kt * (8 * theta * y * (-3 * theta + 2 * y) + delta * rho * t * theta * (7 * fastpow(theta, 2) - 23 * theta * y + 8 * fastpow(y, 2)) - 8 * fastpow(rho, 2) * (6 * fastpow(theta, 3) - 18 * fastpow(theta, 2) * y + 11 * theta * fastpow(y, 2) - fastpow(y, 3)) + 4 * delta * fastpow(rho, 3) * t * (-13 * fastpow(theta, 3) + 31 * fastpow(theta, 2) * y - 14 * theta * fastpow(y, 2) + fastpow(y, 3)))))) / (64. * fastpow(kappa, 2) * t * sqrt((-theta + kappa * t * theta + (theta - y) / ekt + y) / (kappa * t)) * fastpow((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y, 4))
end

function lpp3z3(p::HestonParams, t, ekt)
    e2kt = ekt^2
    e3kt = ekt^3
    e4kt = ekt^4
    kappa = p.κ
    theta = p.θ
    delta = p.σ
    rho = p.ρ
    y = p.v0
    return (fastpow(delta, 3) * ekt * rho * ((-15 * (2 + kappa * t) + 3 * e4kt * (50 - 79 * kappa * t + 35 * fastpow(kappa, 2) * fastpow(t, 2) - 6 * fastpow(kappa, 3) * fastpow(t, 3) + 8 * fastpow(rho, 2) * (-18 + 15 * kappa * t - 6 * fastpow(kappa, 2) * fastpow(t, 2) + fastpow(kappa, 3) * fastpow(t, 3))) + ekt * (-3 * (20 + 86 * kappa * t + 29 * fastpow(kappa, 2) * fastpow(t, 2)) + fastpow(rho, 2) * (432 + 936 * kappa * t + 552 * fastpow(kappa, 2) * fastpow(t, 2) + 92 * fastpow(kappa, 3) * fastpow(t, 3))) + e2kt * (360 + 324 * kappa * t - 261 * fastpow(kappa, 2) * fastpow(t, 2) - 48 * fastpow(kappa, 3) * fastpow(t, 3) - 4 * fastpow(rho, 2) * (324 + 378 * kappa * t - 12 * fastpow(kappa, 2) * fastpow(t, 2) - 2 * fastpow(kappa, 3) * fastpow(t, 3) + 23 * fastpow(kappa, 4) * fastpow(t, 4))) + e3kt * (3 * (-140 + 62 * kappa * t + 81 * fastpow(kappa, 2) * fastpow(t, 2) - 38 * fastpow(kappa, 3) * fastpow(t, 3) + 8 * fastpow(kappa, 4) * fastpow(t, 4)) + 4 * fastpow(rho, 2) * (324 + 54 * kappa * t - 114 * fastpow(kappa, 2) * fastpow(t, 2) + 77 * fastpow(kappa, 3) * fastpow(t, 3) - 19 * fastpow(kappa, 4) * fastpow(t, 4) + 2 * fastpow(kappa, 5) * fastpow(t, 5)))) * fastpow(theta, 3) + (15 * (7 + 4 * kappa * t) + 3 * e4kt * (-79 + 70 * kappa * t - 18 * fastpow(kappa, 2) * fastpow(t, 2) + 24 * fastpow(rho, 2) * (5 - 4 * kappa * t + fastpow(kappa, 2) * fastpow(t, 2))) - 3 * ekt * (26 - 200 * kappa * t - 87 * fastpow(kappa, 2) * fastpow(t, 2) + 4 * fastpow(rho, 2) * (30 + 142 * kappa * t + 115 * fastpow(kappa, 2) * fastpow(t, 2) + 23 * fastpow(kappa, 3) * fastpow(t, 3))) + 2 * e2kt * (3 * (-66 - 195 * kappa * t + 63 * fastpow(kappa, 2) * fastpow(t, 2) + 16 * fastpow(kappa, 3) * fastpow(t, 3)) + 4 * fastpow(rho, 2) * (135 + 390 * kappa * t - 9 * fastpow(kappa, 2) * fastpow(t, 2) - 48 * fastpow(kappa, 3) * fastpow(t, 3) + 23 * fastpow(kappa, 4) * fastpow(t, 4))) + e3kt * (606 + 300 * kappa * t - 585 * fastpow(kappa, 2) * fastpow(t, 2) + 210 * fastpow(kappa, 3) * fastpow(t, 3) - 24 * fastpow(kappa, 4) * fastpow(t, 4) - 4 * fastpow(rho, 2) * (270 + 282 * kappa * t - 345 * fastpow(kappa, 2) * fastpow(t, 2) + 153 * fastpow(kappa, 3) * fastpow(t, 3) - 29 * fastpow(kappa, 4) * fastpow(t, 4) + 2 * fastpow(kappa, 5) * fastpow(t, 5)))) * fastpow(theta, 2) * y + (-93 - 75 * kappa * t + 3 * e4kt * (35 - 18 * kappa * t + 24 * fastpow(rho, 2) * (-2 + kappa * t)) + 3 * ekt * (58 - 123 * kappa * t - 86 * fastpow(kappa, 2) * fastpow(t, 2) + 4 * fastpow(rho, 2) * (12 + 80 * kappa * t + 92 * fastpow(kappa, 2) * fastpow(t, 2) + 23 * fastpow(kappa, 3) * fastpow(t, 3))) + e3kt * (-3 * (74 + 137 * kappa * t - 100 * fastpow(kappa, 2) * fastpow(t, 2) + 16 * fastpow(kappa, 3) * fastpow(t, 3)) - 16 * fastpow(rho, 2) * (-27 - 51 * kappa * t + 45 * fastpow(kappa, 2) * fastpow(t, 2) - 12 * fastpow(kappa, 3) * fastpow(t, 3) + fastpow(kappa, 4) * fastpow(t, 4))) + e2kt * (36 + 909 * kappa * t - 42 * fastpow(kappa, 2) * fastpow(t, 2) - 60 * fastpow(kappa, 3) * fastpow(t, 3) - 4 * fastpow(rho, 2) * (108 + 462 * kappa * t + 96 * fastpow(kappa, 2) * fastpow(t, 2) - 117 * fastpow(kappa, 3) * fastpow(t, 3) + 23 * fastpow(kappa, 4) * fastpow(t, 4)))) * theta * fastpow(y, 2) + 2 * (9 + 3 * e4kt * (-3 + 4 * fastpow(rho, 2)) + 15 * kappa * t + e2kt * (-3 * kappa * t * (33 + 10 * kappa * t) + fastpow(rho, 2) * (36 + 192 * kappa * t + 96 * fastpow(kappa, 2) * fastpow(t, 2) - 46 * fastpow(kappa, 3) * fastpow(t, 3))) + e3kt * (18 + 57 * kappa * t - 12 * fastpow(kappa, 2) * fastpow(t, 2) - 2 * fastpow(rho, 2) * (18 + 48 * kappa * t - 21 * fastpow(kappa, 2) * fastpow(t, 2) + 2 * fastpow(kappa, 3) * fastpow(t, 3))) + ekt * (3 * (-6 + 9 * kappa * t + 14 * fastpow(kappa, 2) * fastpow(t, 2)) - 2 * fastpow(rho, 2) * (6 + 48 * kappa * t + 69 * fastpow(kappa, 2) * fastpow(t, 2) + 23 * fastpow(kappa, 3) * fastpow(t, 3)))) * fastpow(y, 3))) / (96. * kappa * t * sqrt((-theta + kappa * t * theta + (theta - y) / ekt + y) / (kappa * t)) * fastpow((1 + ekt * (-1 + kappa * t)) * theta + (-1 + ekt) * y, 5))
end
