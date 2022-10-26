abstract type MCPayoff end
abstract type MCPayoffSchedule end


#payment schedule ? Bullet(maturity, payment, isOnGrid)  Periodic(Vector{Bullet})
mutable struct BulletCashFlow <: MCPayoffSchedule
    observationTime::Float64
    paymentTime::Float64
    isOn::Bool
    value #for a barrier, single KO flow but we don't know when => array of payment times?
end

mutable struct PeriodicCashFlow  <: MCPayoffSchedule
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

#payoff has list of schedules, may be pure obs or payment.
struct VanillaOption <: MCPayoff
    isCall::Bool
    strike::Float64
    schedule::BulletCashFlow
    currentValue
end

#advancePath(gen, pathValues, t0, t1)
function advancePayoff(payoff::VanillaOption, time, x)
    advance(payoff.schedule,time)
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
    if payoff.schedule.isCashflowReady
        return df * payoff.schedule.value
    else
        return zero(x)
    end
end

function evaluatePayoffOnPath(payoff::VanillaOption, x, df)
    if payoff.isCall
        return df * max(x[1] - payoff.strike, 0)
    else
        return df * max(payoff.strike - x[1], 0)
    end
end

function specificTimes(payoff::VanillaOption)
    return [payoff.maturity]
end


# change of frequency measure
struct COF
    m::Int
end

function specificTimes(payoff::COF) 
    return LinRange(0.0,1.0, m+1)    
end

#function evaluatePayoff(payoff::VanillaOption, x, df)
#    if payoff.isCall
#        return df * max(x - payoff.strike, 0)
#    else
#        return df * max(payoff.strike - x, 0)
#    end
#end
