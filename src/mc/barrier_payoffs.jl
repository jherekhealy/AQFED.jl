import AQFED.Black: BarrierKind, DownIn, UpIn, DownOut, UpOut
export DiscreteBarrierOption, DiscreteOneTouch, AutocallNote, UniformKernel, EpanechnikovKernel

#Payoffs monitored at discrete times. evaluatePayoffOnPath receives the asset values x at specificTimes(payoff).

"""
    DiscreteBarrierOption(isCall, strike, level, kind::BarrierKind, barrierTimes, maturity)

Vanilla option knocked in or out (`kind` among `DownIn`, `UpIn`, `DownOut`, `UpOut`) when the asset crosses `level` at one
of the `barrierTimes`, paid at `maturity`.
"""
struct DiscreteBarrierOption <: MCPayoff
    isCall::Bool
    strike::Float64
    level::Float64
    kind::BarrierKind
    observationTimes::Vector{Float64} #the barrier times, followed by the maturity if it is not a barrier time
    nBarrierTimes::Int
end

function DiscreteBarrierOption(isCall::Bool, strike::Float64, level::Float64, kind::BarrierKind, barrierTimes::AbstractVector{Float64}, maturity::Float64)
    times = barrierTimes[end] == maturity ? collect(barrierTimes) : vcat(barrierTimes, maturity)
    return DiscreteBarrierOption(isCall, strike, level, kind, times, length(barrierTimes))
end

specificTimes(payoff::Union{DiscreteBarrierOption}) = payoff.observationTimes

@inline _isCrossed(isUp::Bool, level, x) = isUp ? x >= level : x <= level

function evaluatePayoffOnPath(payoff::DiscreteBarrierOption, x, df)
    isUp = payoff.kind == UpIn || payoff.kind == UpOut
    isIn = payoff.kind == UpIn || payoff.kind == DownIn
    isCrossed = false
    @inbounds for i = 1:payoff.nBarrierTimes
        isCrossed |= _isCrossed(isUp, payoff.level, x[i])
    end
    isCrossed == isIn || return zero(eltype(x))
    intrinsic = x[end] - payoff.strike
    return df * max(payoff.isCall ? intrinsic : -intrinsic, 0)
end

abstract type TouchKernel end
#the payout is linear in the asset over the smoothing width
struct UniformKernel <: TouchKernel end
#the payout follows the integral of the Epanechnikov kernel over the smoothing width
struct EpanechnikovKernel <: TouchKernel end
@inline touchKernelValue(::UniformKernel, u) = (u + 1) / 2
@inline touchKernelValue(::EpanechnikovKernel, u) = 1 / 2 - u * (u * u - 3) / 4

"""
    DiscreteOneTouch(isDown, level, barrierTimes, payout; width=0, kernel=UniformKernel())

Pays `payout` at the last barrier time if the asset is beyond `level` at one of the `barrierTimes`. With a positive `width`,
the digital is smoothed over [level-width/2, level+width/2] by the cumulative `kernel`, which reduces the variance of the
Greeks by finite differences.
"""
struct DiscreteOneTouch{K<:TouchKernel} <: MCPayoff
    isDown::Bool
    level::Float64
    observationTimes::Vector{Float64}
    payout::Float64
    width::Float64
    kernel::K
end
DiscreteOneTouch(isDown::Bool, level::Float64, barrierTimes::AbstractVector{Float64}, payout::Float64; width::Float64=0.0, kernel::TouchKernel=UniformKernel()) =
    DiscreteOneTouch(isDown, level, collect(barrierTimes), payout, width, kernel)

specificTimes(payoff::DiscreteOneTouch) = payoff.observationTimes

function evaluatePayoffOnPath(payoff::DiscreteOneTouch, x, df)
    w = payoff.width / 2
    touch = zero(eltype(x))
    @inbounds for xi in x
        s = payoff.isDown ? payoff.level - xi : xi - payoff.level #positive when touched
        if s >= w
            return df * payoff.payout
        elseif s >= -w
            touch = max(touch, touchKernelValue(payoff.kernel, s / w))
        end
    end
    return df * payoff.payout * touch
end

"""
    AutocallNote(observationTimes, autocallLevels, coupons, discountFactors; reference=1.0, protectionLevel=0.0)

Note on a notional of 1 that is redeemed early at the first observation time `observationTimes[i]` where the asset is above
`autocallLevels[i]*reference`, for the amount `coupons[i]` (principal included). If it is never called, it pays at the
last observation time 1, or the performance x/reference when the asset is below `protectionLevel*reference` (a down-and-in
put observed at maturity). `discountFactors[i]` is the discount factor to `observationTimes[i]`.
"""
struct AutocallNote <: MCPayoff
    observationTimes::Vector{Float64}
    autocallLevels::Vector{Float64}
    coupons::Vector{Float64}
    discountFactors::Vector{Float64}
    reference::Float64
    protectionLevel::Float64
end

function AutocallNote(observationTimes::AbstractVector{Float64}, autocallLevels::AbstractVector{Float64}, coupons::AbstractVector{Float64}, discountFactors::AbstractVector{Float64}; reference::Float64=1.0, protectionLevel::Float64=0.0)
    n = length(observationTimes)
    length(autocallLevels) == n && length(coupons) == n && length(discountFactors) == n ||
        throw(DimensionMismatch("the autocall levels, coupons and discount factors must be given for each observation time"))
    return AutocallNote(collect(observationTimes), collect(autocallLevels), collect(coupons), collect(discountFactors), reference, protectionLevel)
end

specificTimes(payoff::AutocallNote) = payoff.observationTimes

#the discount factors of the payoff are used, df (to the last observation) is ignored
function evaluatePayoffOnPath(payoff::AutocallNote, x, df)
    @inbounds for i in eachindex(payoff.observationTimes)
        if x[i] >= payoff.autocallLevels[i] * payoff.reference
            return payoff.coupons[i] * payoff.discountFactors[i]
        end
    end
    performance = x[end] / payoff.reference
    return payoff.discountFactors[end] * (performance < payoff.protectionLevel ? performance : one(performance))
end
