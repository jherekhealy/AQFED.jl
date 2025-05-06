using CharFuncPricing, Random123, LinearAlgebra
using AQFED.Black, AQFED.GlobalOptimization, GaussNewton
import AQFED.Math:
    MQMinTransformation,
    LogisticTransformation,
	inv
export convertVolsToPricesOTMWeights, calibrateHestonFromPrices
using QuadGK, Roots, SpecialFunctions
using DoubleExponentialFormulas
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

function makeVarianceSwapReduction(ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractVector{T},
    vols::AbstractMatrix{T}; isLinearExtrapolation=false, thetaUpperBoundFactor=0.0, v0UpperBoundFactor=0.0) where {T}

    vsTheo = [priceVarianceSwap(FukasawaVarianceSwapReplication(isLinearExtrapolation), t, log.(strikes / forwards[i]), vols[i, :] .^ 2, 1.0) for (i, t) = enumerate(ts)]
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
    println("Variance Swap Heston calibration ", vsResult)
    x0 = [vsTrans[1](lnx0[1]), vsTrans[2](lnx0[2]), vsTrans[3](lnx0[3])]
    return V0KappaThetaReduction(x0[1], x0[2], x0[3])
end

function makeDoubleHestonVarianceSwapReduction(ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractVector{T},
    vols::AbstractMatrix{T}; isLinearExtrapolation=false, thetaUpperBoundFactor=0.0, v0UpperBoundFactor=0.0) where {T}

    vsTheo = [priceVarianceSwap(FukasawaVarianceSwapReplication(isLinearExtrapolation), t, log.(strikes / forwards[i]), vols[i, :] .^ 2, 1.0) for (i, t) = enumerate(ts)]
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
    strikes::AbstractVector{T},
    prices::AbstractMatrix{T},
    isCall::AbstractMatrix{Bool},
    weights::AbstractMatrix{T};
    reduction::Reduction=NoReduction(),
	isRelative=false,
    method="Joshi-Yang",
    minimizer="DE",
    VVIX=0.0,
    weightVVIX=1.0,
    strikesVIX=range(0.1, 0.5, length=50),) where {T}
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
            pricer = if method == "Andersen-Lake"
                CharFuncPricing.ALCharFuncPricer(cf, n=64)
            elseif method == "Joshi-Yang"
                JoshiYangCharFuncPricer(cf, t, n=64)
            elseif method == "Cos-128"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, 128, 12)
            elseif method == "Cos"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, 256, 16)
            elseif method == "Cos-Junike"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, tol=1e-4, maxM=4096)
            elseif method == "Flinn"
                CharFuncPricing.FlinnCharFuncPricer(cf, t, tTol=1e-4, qTol=1e-8)
            elseif method == "Flinn-Transformed"
                #CharFuncPricing.makeCVCharFunc(cf,t,CharFuncPricing.InitialControlVariance())
                CharFuncPricing.AdaptiveFlinnCharFuncPricer(cf, t)
            elseif method == "Swift"
                m, _ = CharFuncPricing.findSwiftScaling(cf, t)
                CharFuncPricing.makeSwiftCharFuncPricer(cf, t, m, 3)
            end
            for (j, strike) ∈ enumerate(strikes)
                mPrice = CharFuncPricing.priceEuropean(pricer, isCall[i, j], strike / forwards[i], 1.0, t, 1.0) * forwards[i]
				F[i+(j-1)*length(ts)] =if isRelative
					uWeights[i, j] * (mPrice / uPrices[i, j]-1)
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
    x0 = [volShort^2, 1.0, volLong^2, -0.5, 0.5] #v0::T, κ::T, θ::T, ρ::T, σ::T)
    lower = [1e-4, 1e-4, 1e-4, -0.99, 1e-2]
    upper = [4.0, 10.0, 4.0, 0.9, 4.0]
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
    gnRmse = GaussNewton.optimize!(objectiveN, rx, out, autodiff=:single)
    println("Result GN ", gnRmse)
    x = expandX(reduction, HestonParams, rx)
    rmse = norm(uWeights) * GlobalOptimization.minimum(optim) / sqrt(length(strikes) * length(ts)) #very close to truncated vol rmse.
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
    method="Joshi-Yang",
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
		@. x = max(lower, min(upper,x))
        hParams = DoubleHestonParams(
            HestonParams(x[1], x[2], x[3], x[4], x[5]),
            HestonParams(x[6], x[7], x[8], x[9], x[10]))
        # TX = typeof(x[1])
        # TH = typeof(hParams)
        # cf = DefaultCharFunc{TH,Complex{TX}}(hParams)
        cf = DefaultCharFunc(hParams)

        for (i, t) ∈ enumerate(ts)
            # pricer =JoshiYangCharFuncPricer(cf, t, n=64)
            pricer = if method == "Andersen-Lake"
                CharFuncPricing.ALCharFuncPricer(cf, n=64)
            elseif method == "Joshi-Yang"
                JoshiYangCharFuncPricer(cf, t, n=64)
            elseif method == "Cos-128"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, 128, 12)
            elseif method == "Cos"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, 256, 16)
            elseif method == "Cos-Junike"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, tol=1e-4, maxM=4096)
            elseif method == "Flinn"
                CharFuncPricing.FlinnCharFuncPricer(cf, t, tTol=1e-4, qTol=1e-8)
            elseif method == "Flinn-Transformed"
                #CharFuncPricing.makeCVCharFunc(cf,t,CharFuncPricing.InitialControlVariance())
                CharFuncPricing.AdaptiveFlinnCharFuncPricer(cf, t)
            elseif method == "Swift"
                m, _ = CharFuncPricing.findSwiftScaling(cf, t)
                CharFuncPricing.makeSwiftCharFuncPricer(cf, t, m, 3)
            end
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
	x0 = [x01[1]/2, x01[2]/2, x01[3]/2, x01[4],x01[5], x01[1]/2, x01[2]*2, x01[3]/2, x01[4],x01[5]]
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
	@. x = max(lower, min(upper,x))
	rmse = norm(uWeights) * GlobalOptimization.minimum(optim) / sqrt(length(strikes) * length(ts)) #very close to truncated vol rmse.
    params = DoubleHestonParams(
        HestonParams(x[1], x[2], x[3], x[4], x[5]),
        HestonParams(x[6], x[7], x[8], x[9], x[10]))
    return params, rmse
end


function convertVolsToPricesOTMWeights(ts::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractVector{T}, vols::AbstractMatrix{T}; weights::AbstractMatrix{T}=ones(T, length(ts), length(strikes)), vegaFloor=1e-2, isInverse=true, truncationDev=3.0) where {T}
    uPrices = zeros(length(ts), length(strikes))
    uVegas = zeros(length(ts), length(strikes))
    uWeights = zeros(length(ts), length(strikes))
    isCall = zeros(Bool, length(ts), length(strikes))

    for (i, t) ∈ enumerate(ts)
        for (j, strike) ∈ enumerate(strikes)
            isCall[i, j] = strike >= forwards[i]
            uPrices[i, j] = Black.blackScholesFormula(isCall[i, j], strike, forwards[i], vols[i, j]^2 * t, 1.0, 1.0)
            uVegas[i, j] = Black.blackScholesVega(strike, forwards[i], vols[i, j]^2 * t, 1.0, 1.0, t)
        end
		if isInverse
            @. uWeights[i, :] = weights[i, :] / max(uVegas[i, :], vegaFloor * forwards[i])
		else 
			sumVega = sum(uVegas[i,:])
			@. uWeights[i, :] = weights[i, :] * uVegas[i, :]/sumVega 
		end
        indicator = @. truncationDev*vols[i,:]*sqrt(t) > abs(log(strikes[:]/forwards[i]))
        uWeights[i,:] .*= indicator
    end
    return uPrices, isCall, uWeights
end

function estimateVolError(params, ts::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractVector{T}, vols; weights=ones(T, length(ts), length(strikes))) where {T}
    cf = DefaultCharFunc(params)

    volError = zeros(length(ts), length(strikes))
    for (i, t) ∈ enumerate(ts)
        #pricer = CharFuncPricing.ALCharFuncPricer(cf, n=128)
        pricer = CharFuncPricing.JoshiYangCharFuncPricer(cf, t)
        for (j, strike) ∈ enumerate(strikes)
            isCall = strike > forwards[i]
            mPrice = CharFuncPricing.priceEuropean(pricer, isCall, strike / forwards[i], 1.0, t, 1.0)
            try
            volError[i, j] = weights[i, j] * (Black.impliedVolatility(isCall, mPrice, 1.0, strike / forwards[i], t, 1.0) - vols[i, j])
            catch e
                println(e)
                volError[i , j] = NaN
            end
        end
    end
    rmseVol = norm(volError) / sqrt(length(strikes) * length(ts)) #around 0.015
    return volError, rmseVol
end

function calibrateHestonFromVols(ts::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractMatrix{T}, vols::AbstractMatrix{T}; weights::AbstractMatrix{T}=ones(T, length(ttes), length(strikes)), vegaFloor=1e-2) where {T}
    uPrices, isCall, uWeights = convertVolsToPricesOTMWeights(ts, forwards, strikes, vols, weights=weights, vegaFloor=vegaFloor)
    params, rmse = calibrateHestonFromPrices(ts, forwards, strikes, uPrices, isCall, uWeights)
    volError, rmseVol = estimateVolError(params, ts, forwards, strikes, vols, weights=weights)
    return params, rmseVol
end

