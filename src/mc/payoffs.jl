abstract type MCPayoff end
abstract type MCPayoffSchedule end

export specificTimes, evaluatePayoffOnPath, evaluatePayoff
export EarlyExerciseMCPayoff, BermudanOption, BermudanBasketOption, AsianOption
export exerciseTimes, regressionVariablesDimension, evaluateEarlyExercise!

#payment schedule ? Bullet(maturity, payment, isOnGrid)  Periodic(Vector{Bullet})
mutable struct BulletCashFlow <: MCPayoffSchedule
    observationTime::Float64
    paymentTime::Float64
    isOn::Bool
    value #for a barrier, single KO flow but we don't know when => array of payment times?
end

mutable struct PeriodicCashFlow <: MCPayoffSchedule
    observationTimes::Vector{Float64}
    paymentTime::Float64
    observationIndex::Int
    value
end

# we need to evaluate the payoff or not
function isObservationTime(schedule::BulletCashFlow, time)
    return schedule.observationTime == time
end

function isObservationTime(schedule::PeriodicCashFlow, time)
    schedule.observationTimes[schedule.observationIndex] == time
end

# allow to precompute discount factors.
function paymentTimes(schedule)
    return [schedule.paymentTime]
end

function advance(schedule::BulletCashFlow, time)
    if time == schedule.observationTime
        schedule.isOn = true
    else
        schedule.isOn = false
    end
end

struct ListPayoff <: MCPayoff
    list::Vector{MCPayoff}
end
function dimensionOut(::MCPayoff)
    return (1,)
end

function dimensionOut(p::ListPayoff)
    return size(p.list)
end

function evaluatePayoffOnPath(payoff::ListPayoff, x, df)
    [evaluatePayoffOnPath(p, x, df) for p = payoff.list]
end
function specificTimes(payoff::ListPayoff)
    return specificTimes(payoff.list[1])
end

struct CliquetOption <: MCPayoff
    localCap::Float64
    localFloor::Float64
    cap::Float64
    floor::Float64
    strike::Float64
    schedule::PeriodicCashFlow
    currentValue
end

function evaluatePayoffOnPath(payoff::CliquetOption, x, df)
    s = zero(eltype(x))
    @inbounds for i in 1:(length(x)-1)
        s += max(min(x[i+1] / x[i] - 1, payoff.localCap), payoff.localFloor)
    end
    #println("s=",s,x)
    return max(min(s, payoff.cap)-payoff.strike, payoff.floor) * df
end
function specificTimes(payoff::CliquetOption)
    return payoff.schedule.observationTimes
end


struct MultiplicativeCliquetStream <: MCPayoff
    localCap::Float64
    premium::Float64
    gearing::Float64
    schedule::PeriodicCashFlow
    currentValue
end

function evaluatePayoffOnPath(payoff::MultiplicativeCliquetStream, x, df)
    s = zero(eltype(x))
    @inbounds for i in 1:(length(x)-1)
        s += max(min(x[i+1] / x[i] - 1, payoff.localCap), payoff.localFloor)
    end
    #println("s=",s,x)
    return max(min(s, payoff.cap)-payoff.strike, payoff.floor) * df
end
function specificTimes(payoff::MultiplicativeCliquetStream)
    return payoff.schedule.observationTimes
end


struct VanillaBasketOption <: MCPayoff
    isCall::Bool
    strike::Float64
    weights::Vector{Float64}
    schedule::BulletCashFlow
    currentValue
end

function evaluatePayoffOnPath(payoff::VanillaBasketOption, x, df)
    if payoff.isCall
        return df * max(sum(payoff.weights .* x) - payoff.strike, 0)
    else
        return df * max(payoff.strike - sum(payoff.weights .* x), 0)
    end
end
function specificTimes(payoff::VanillaBasketOption)
    return [payoff.schedule.observationTime]
end


struct ForwardStartOption <: MCPayoff
    isCall::Bool
    strike::Float64
    startTime::Float64
    expiryTime::Float64
end



function evaluatePayoffOnPath(payoff::ForwardStartOption, x, df)
    if payoff.isCall
        return df * max(x[2] - payoff.strike*x[1], 0)
    else
        return df * max(payoff.strike*x[1] - x[2], 0)
    end
end

function specificTimes(payoff::ForwardStartOption)
    return [payoff.startTime, payoff.expiryTime]
end
#payoff has list of schedules, may be pure obs or payment.
struct VanillaOption <: MCPayoff
    isCall::Bool
    strike::Float64
    schedule::BulletCashFlow
    currentValue
end

#advancePath(gen, pathValues, t0, t1)
function advancePayoff(payoff::VanillaOption, time, x)
    advance(payoff.schedule, time)
    if payoff.schedule.isOn
        if payoff.isCall
            payoff.schedule.value = max(x - payoff.strike, 0)
        else
            payoff.schedule.value = max(payoff.strike - x, 0)
        end
    end
end
#should we pass in currentpv usch that we do pv += df*... or have is isPayment up 1 level?
function evaluatePayoff(payoff::VanillaOption, x, df)
    if payoff.schedule.isOn
        return df * payoff.schedule.value
    else
        return zero(x)
    end
end

function evaluatePayoffOnPath(payoff::VanillaOption, x, df)
    if payoff.isCall
        return df * max(x[end] - payoff.strike, 0)
    else
        return df * max(payoff.strike - x[end], 0)
    end
end

function specificTimes(payoff::VanillaOption)
    return [payoff.schedule.observationTime]
end

struct AsianOption <: MCPayoff
    isCall::Bool
    strike::Float64
    observationTimes::Vector{Float64}
    weights::Vector{Float64}
end

function AsianOption(isCall::Bool, strike::Float64, observationTimes::Vector{Float64})
    n = length(observationTimes)
    weights = fill(1.0 / n, n)
    return AsianOption(isCall, strike, observationTimes, weights)
end

function evaluatePayoffOnPath(payoff::AsianOption, x, df)
    avg = zero(eltype(x))
    @inbounds for i in eachindex(x)
        avg += payoff.weights[i] * x[i]
    end
    if payoff.isCall
        return df * max(avg - payoff.strike, 0.0)
    else
        return df * max(payoff.strike - avg, 0.0)
    end
end

function specificTimes(payoff::AsianOption)
    return payoff.observationTimes
end

# mutable struct RealizedVariancePayoff <: MCPayoff
#     stopTimes::Vector{Float64}
# end

# function evaluatePayoffOnPath(payoff::RealizedVariancePayoff, x, df)

# end

# function specificTimes(payoff::RealizedVariancePayoff)
#     return payoff.stopTimes
# end



# change of frequency measure
struct COF
    m::Int
end

function specificTimes(payoff::COF)
    return LinRange(0.0, 1.0, m+1)
end

#function evaluatePayoff(payoff::VanillaOption, x, df)
#    if payoff.isCall
#        return df * max(x - payoff.strike, 0)
#    else
#        return df * max(payoff.strike - x, 0)
#    end
#end

abstract type EarlyExerciseMCPayoff <: MCPayoff end

struct BermudanOption <: EarlyExerciseMCPayoff
    isCall::Bool
    strike::Float64
    exerciseTimes::Vector{Float64}
    fixing::Float64
    degree::Int
end

function BermudanOption(isCall::Bool, strike::Float64, exerciseTimes::Vector{Float64}; fixing::Float64=strike, degree::Int=3)
    return BermudanOption(isCall, strike, exerciseTimes, fixing, degree)
end

exerciseTimes(payoff::BermudanOption) = payoff.exerciseTimes
specificTimes(payoff::BermudanOption) = payoff.exerciseTimes
regressionVariablesDimension(payoff::BermudanOption) = payoff.degree + 1

function evaluateEarlyExercise!(payoff::BermudanOption, spotValues, regressionVars)
    s = spotValues isa AbstractArray ? spotValues[end] : spotValues
    x = s / payoff.fixing
    pow = 1.0
    for d in 1:(payoff.degree + 1)
        regressionVars[d] = pow
        pow *= x
    end
    intrinsic = payoff.isCall ? s - payoff.strike : payoff.strike - s
    return max(intrinsic, 0.0)
end

function evaluatePayoffOnPath(payoff::BermudanOption, x, df)
    s = x isa AbstractArray ? x[end] : x
    intrinsic = payoff.isCall ? s - payoff.strike : payoff.strike - s
    return max(intrinsic, 0.0) * df
end

struct BermudanBasketOption <: EarlyExerciseMCPayoff
    isCall::Bool
    strike::Float64
    weights::Vector{Float64}
    exerciseTimes::Vector{Float64}
    fixings::Vector{Float64}
    isParametric::Bool
end

function BermudanBasketOption(
    isCall::Bool,
    strike::Float64,
    weights::Vector{Float64},
    exerciseTimes::Vector{Float64};
    fixings::Vector{Float64}=ones(length(weights)),
    isParametric::Bool=true
)
    return BermudanBasketOption(isCall, strike, weights, exerciseTimes, fixings, isParametric)
end

exerciseTimes(payoff::BermudanBasketOption) = payoff.exerciseTimes
specificTimes(payoff::BermudanBasketOption) = payoff.exerciseTimes

function regressionVariablesDimension(payoff::BermudanBasketOption)
    n = length(payoff.weights)
    if n == 1
        return 3
    elseif n == 2
        return 6
    else
        return 1 + 2*n + div(n*(n-1), 2)
    end
end

function evaluateEarlyExercise!(payoff::BermudanBasketOption, spotValues::AbstractVector, regressionVars::AbstractVector)
    n = length(spotValues)
    basketLevel = 0.0
    for j in 1:n
        basketLevel += payoff.weights[j] * (spotValues[j] / payoff.fixings[j])
    end
    
    regressionVars[1] = 1.0
    idx = 2
    for j in 1:n
        regressionVars[idx] = spotValues[j] / payoff.fixings[j]
        idx += 1
    end
    for j in 1:n
        x_j = spotValues[j] / payoff.fixings[j]
        regressionVars[idx] = x_j * x_j
        idx += 1
    end
    for i in 1:n, j in (i+1):n
        regressionVars[idx] = (spotValues[i] / payoff.fixings[i]) * (spotValues[j] / payoff.fixings[j])
        idx += 1
    end

    intrinsic = payoff.isCall ? basketLevel - payoff.strike : payoff.strike - basketLevel
    return max(intrinsic, 0.0)
end

function evaluatePayoffOnPath(payoff::BermudanBasketOption, x::AbstractMatrix, df)
    lastRow = x[end, :]
    basketLevel = 0.0
    for j in 1:length(lastRow)
        basketLevel += payoff.weights[j] * (lastRow[j] / payoff.fixings[j])
    end
    intrinsic = payoff.isCall ? basketLevel - payoff.strike : payoff.strike - basketLevel
    return max(intrinsic, 0.0) * df
end
