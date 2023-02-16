export VanillaEuropean, VanillaAmerican, StructureDefinition, DiscreteKO, ButterflyAmerican, KreissSmoothDefinition

abstract type StructureDefinition end

abstract type FDMStructure end

mutable struct DefaultFDMStructure{T}
    currentTime::T
    currentValue::Matrix{T}
    lowerBound::Vector{T}
end

function makeFDMStructure(p::TD, underlying::AbstractArray{T}) where {T,TD<:StructureDefinition}
    DefaultFDMStructure(zero(T), zeros(T, length(underlying), 1), zeros(T, length(underlying)))
end


"advance payoff to given time. This is where one could update relevant indices/set current observation"
function advance(p::TD, s::DefaultFDMStructure{T}, time::T) where {TD<:StructureDefinition,T}
    s.currentTime = time
end

function currentValue(s::DefaultFDMStructure{T})::Matrix{T} where {T}
    return s.currentValue
end

function isLowerBoundActive(p::TD, s::DefaultFDMStructure{T})::Bool where {TD<:StructureDefinition,T}
    return false
end

function lowerBound!(s::DefaultFDMStructure{T}, v::AbstractArray{T}) where {T}
    if !isempty(s.lowerBound)
        v[1:end] = s.lowerBound
    end
end

function evaluate(p::TD, s::DefaultFDMStructure{T}, S::AbstractArray{T}) where {T,TD<:StructureDefinition}
    for i = 1:size(currentValue(s), 2)
        evaluate(p, s, S, i)
    end
end


struct DiscountBond{T} <: StructureDefinition
    timeToExpiry::T
end


"observation times sorted in ascending order"
function observationTimes(p::DiscountBond{T})::AbstractArray{T} where {T}
    return [p.timeToExpiry]
end
function isLowerBoundActive(p::DiscountBond{T}, s::DefaultFDMStructure{T})::Bool where {T}
    return false
end

function evaluate(p::DiscountBond{TS}, s::DefaultFDMStructure{T}, S::AbstractArray{T}, columnIndex::Int) where {T,TS}
    if s.currentTime == p.timeToExpiry
        @. s.currentValue[:, 1] = 1.0
    end
end

function nonSmoothPoints(p::DiscountBond{T})::AbstractArray{T} where {T}
    return []
end

mutable struct VanillaEuropean{T} <: StructureDefinition
    isCall::Bool
    strike::T
    timeToExpiry::T
    VanillaEuropean(isCall::Bool, strike::T, timeToExpiry::T) where {T} = new{T}(isCall, strike, timeToExpiry)
end
function nonSmoothPoints(p::VanillaEuropean{T})::AbstractArray{T} where {T}
    return [p.strike]
end

"observation times sorted in ascending order"
function observationTimes(p::VanillaEuropean{T})::AbstractArray{T} where {T}
    return [p.timeToExpiry]
end
function isLowerBoundActive(p::VanillaEuropean{T}, s::DefaultFDMStructure{T})::Bool where {T}
    return true
end

function evaluate(p::VanillaEuropean{T}, S::T)::T where {T}
    if isCall
        return max(S - p.strike, zero(T))
    else
        return max(p.strike - S, zero(T))
    end
end
function evaluate(p::VanillaEuropean{TS}, s::DefaultFDMStructure{T}, S::AbstractArray{T}, columnIndex::Int) where {T,TS}
    if s.currentTime == p.timeToExpiry
        if p.isCall
            @. s.currentValue[:, 1] = max(S - p.strike, zero(T))
        else
            @. s.currentValue[:, 1] = max(p.strike - S, zero(T))
        end
    end
end



struct VanillaAmerican{T} <: StructureDefinition
    isCall::Bool
    strike::T
    timeToExpiry::T
    exerciseStartTime::T
    VanillaAmerican(isCall::Bool, strike::T, timeToExpiry::T; exerciseStartTime=zero(T)) where {T} = new{T}(isCall, strike, timeToExpiry, exerciseStartTime)
end

function observationTimes(p::VanillaAmerican{T})::AbstractArray{T} where {T}
    return [p.timeToExpiry]
end

function nonSmoothPoints(p::VanillaAmerican{T})::AbstractArray{T} where {T}
    return [p.strike]
end

function makeFDMStructure(p::VanillaAmerican{TS}, underlying::AbstractArray{T}) where {T,TS}
    lowerBound = if p.isCall
        @. max(underlying - p.strike, zero(T))
    else
        @. max(p.strike - underlying, zero(T))
    end
    return DefaultFDMStructure(p.timeToExpiry, zeros(T, length(underlying), 1), lowerBound)
end


function evaluate(p::VanillaAmerican{TS}, s::DefaultFDMStructure{T}, S::AbstractArray{T}, columnIndex::Int) where {T,TS}
    if (s.currentTime >= p.exerciseStartTime)
        @. s.currentValue[:, 1] = max(s.currentValue[:, 1], s.lowerBound)
    end
end

function isLowerBoundActive(p::VanillaAmerican{T}, s::DefaultFDMStructure{T})::Bool where {T}
    return s.currentTime >= p.exerciseStartTime
end

mutable struct VanillaAmericanWithExerciseBoundary{T} <: StructureDefinition
    delegate::VanillaAmerican{T}
    exerciseBoundary::Array{T}
    exerciseTimes::Array{T}
    VanillaAmericanWithExerciseBoundary(delegate::VanillaAmerican{T}) where {T} = new{T}(delegate, Vector{T}(), Vector{T}())
end


function observationTimes(p::VanillaAmericanWithExerciseBoundary{T})::AbstractArray{T} where {T}
    return observationTimes(p.delegate)
end

function nonSmoothPoints(p::VanillaAmericanWithExerciseBoundary{T})::AbstractArray{T} where {T}
    return nonSmoothPoints(p.delegate)
end

function makeFDMStructure(p::VanillaAmericanWithExerciseBoundary{TS}, underlying::AbstractArray{T}) where {T,TS}
    empty!(p.exerciseBoundary)
    empty!(p.exerciseTimes)
    return makeFDMStructure(p.delegate, underlying)
end


function evaluate(p::VanillaAmericanWithExerciseBoundary{TS}, s::DefaultFDMStructure{T}, x::AbstractArray{T}, columnIndex::Int) where {T,TS}
    evaluate(p.delegate, s, x, columnIndex)
    y = copy(s.currentValue[:, 1])
    payoff = p.delegate
    for i = 3:length(x)-1
        if (s.currentValue[i-2, 1] <= s.lowerBound[i-2] + sqrt(eps(T)) && s.currentValue[i-1, 1] > s.lowerBound[i-1] + sqrt(eps(T))) || (s.currentValue[i-1, 1] <= s.lowerBound[i-1] + sqrt(eps(T)) && s.currentValue[i-2, 1] > s.lowerBound[i-2] + sqrt(eps(T)))
            sign = one(T)
            if !payoff.isCall
                sign = -one(T)
            end

            p2 = y[i] / ((x[i] - x[i-1]) * (x[i] - x[i+1])) + y[i-1] / ((x[i-1] - x[i]) * (x[i-1] - x[i+1])) + y[i+1] / ((x[i+1] - x[i]) * (x[i+1] - x[i-1]))
            p1 = -sign - y[i] * (x[i+1] + x[i-1]) / ((x[i] - x[i-1]) * (x[i] - x[i+1])) - y[i-1] * (x[i] + x[i+1]) / ((x[i-1] - x[i]) * (x[i-1] - x[i+1])) - y[i+1] * (x[i-1] + x[i]) / ((x[i+1] - x[i]) * (x[i+1] - x[i-1]))
            p0 = sign * payoff.strike + y[i] * x[i+1] * x[i-1] / ((x[i] - x[i-1]) * (x[i] - x[i+1])) + y[i-1] * x[i] * x[i+1] / ((x[i-1] - x[i]) * (x[i-1] - x[i+1])) + y[i+1] * x[i-1] * x[i] / ((x[i+1] - x[i]) * (x[i+1] - x[i-1]))
            delta = p1^2 - 4p2 * p0 + eps(T)
            level = x[i-2]
            if delta >= 0.0
                level1 = (-p1 - sqrt(delta)) / (2p2)
                level2 = (-p1 + sqrt(delta)) / (2p2)
                level = if (level1 < x[i+1] && x[i-2] <= level1)
                    level1
                else
                    level2
                end
            end
            if (level < x[i+1] && x[i-2] <= level)
                # println("level ",level," xim2 ",x[i-2]," xim ",x[i-1]," yim2 ",y[i-2]-s.lowerBound[i-2]," yim ",y[i-1]-s.lowerBound[i-1])
                append!(p.exerciseBoundary, level)
                append!(p.exerciseTimes, s.currentTime)
                break
            end
        end
    end

end

function isLowerBoundActive(p::VanillaAmericanWithExerciseBoundary{T}, s::DefaultFDMStructure{T})::Bool where {T}
    return isLowerBoundActive(p.delegate, s)
end



struct ButterflyAmerican{T} <: StructureDefinition
    isCall::Bool
    strike1::T
    strike2::T
    timeToExpiry::T
    exerciseStartTime::T
    ButterflyAmerican(isCall::Bool, strike1::T, strike2::T, timeToExpiry::T; exerciseStartTime=zero(T)) where {T} = new{T}(isCall, strike1, strike2, timeToExpiry, exerciseStartTime)
end

function observationTimes(p::ButterflyAmerican{T})::AbstractArray{T} where {T}
    return [p.timeToExpiry]
end

function nonSmoothPoints(p::ButterflyAmerican{T})::AbstractArray{T} where {T}
    return [p.strike1, p.strike2]
end

function makeFDMStructure(p::ButterflyAmerican{TS}, underlying::AbstractArray{T}) where {T,TS}
    lowerBound = if p.isCall
        @. (max(underlying - p.strike1, zero(T)) + max(underlying - p.strike2, zero(T)) - 2 * max(underlying - (p.strike1 + p.strike2) / 2, zero(T)))
    else
        @. (max(p.strike1 - underlying, zero(T)) + max(p.strike2 - underlying, zero(T)) - 2 * max((p.strike1 + p.strike2) / 2 - underlying, zero(T)))
    end
    return FDMStructure(p.timeToExpiry, zeros(T, length(underlying), 1), lowerBound)
end


function evaluate(p::ButterflyAmerican{TS}, s::DefaultFDMStructure{T}, S::AbstractArray{T}, columnIndex::Int) where {T,TS}
    if (s.currentTime >= p.exerciseStartTime)
        @. s.currentValue[:, 1] = max(s.currentValue[:, 1], s.lowerBound)
    end
end

function isLowerBoundActive(p::ButterflyAmerican{T}, s::DefaultFDMStructure{T})::Bool where {T}
    return s.currentTime >= p.exerciseStartTime
end


struct DiscreteKO{T} <: StructureDefinition
    vanilla::VanillaEuropean{T}
    level::T
    isDown::Bool
    rebate::T
    observationTimes::Vector{T} #TODO create mutable Schedule{T}, with currentIndex, isActive(time), advance(time),reset()
end

function observationTimes(p::DiscreteKO{T})::AbstractArray{T} where {T}
    if (p.observationTimes[end] != p.vanilla.timeToExpiry)
        return append(p.observationTimes, p.vanilla.timeToExpiry)
    else
        return p.observationTimes
    end
end

function nonSmoothPoints(p::DiscreteKO{T})::AbstractArray{T} where {T}
    return [p.vanilla.strike, p.level]
end

# function makeFDMStructure(p::DiscreteKO{TS}, underlying::AbstractArray{T})::FDMStructure{T} where {T,TS}
#     lowerBound = zeros(T, length(underlying))
#     return FDMStructure(p.vanilla.timeToExpiry, zeros(T, length(underlying), 1), lowerBound, p)
# end


function evaluate(p::DiscreteKO{TS}, s::DefaultFDMStructure{T}, S::AbstractArray{T}, columnIndex::Int) where {T,TS}
    evaluate(p.vanilla, s, S, columnIndex)
    isBarrierActive = false
    for t in p.observationTimes
        if abs(s.currentTime - t) < eps(T)
            isBarrierActive = true
            break
        end
    end
    if isBarrierActive
        for i = eachindex(S)
            if (p.isDown && S[i] <= p.level) || (!p.isDown && S[i] >= p.level)
                s.currentValue[i, 1] = p.rebate
            end
        end
    end
end

function isLowerBoundActive(p::DiscreteKO{T}, s::DefaultFDMStructure{T})::Bool where {T}
    return true
end

struct KreissSmoothDefinition{TS} <: StructureDefinition
    delegate::TS
end


observationTimes(p::KreissSmoothDefinition{TS}) where {TS} = observationTimes(p.delegate)

nonSmoothPoints(p::KreissSmoothDefinition{TS}) where {TS} = nonSmoothPoints(p.delegate)

function makeFDMStructure(p::KreissSmoothDefinition{TS}, underlying::AbstractArray{T}) where {T,TS}
    return makeFDMStructure(p.delegate, underlying)
end

function isLowerBoundActive(p::KreissSmoothDefinition{TS}, s::DefaultFDMStructure{T}) where {T,TS}
    return isLowerBoundActive(p.delegate, s)
end

function evaluate(p::KreissSmoothDefinition{VanillaEuropean{TS}}, s::DefaultFDMStructure{T}, x::AbstractArray{T}, columnIndex::Int) where {T,TS}
    payoff = p.delegate
    if abs(s.currentTime - payoff.timeToExpiry) < eps(T)
        strike = payoff.strike
        sign = one(T)
        if !payoff.isCall
            sign = -one(T)
        end
        for i = eachindex(x)
            if !isIndexBetween(i, x, strike)
                s.currentValue[i, 1] = max(sign * (x[i] - strike), zero(T))
            else
                h = (x[i+1] - x[i-1]) / 2
                obj = function (u::T)
                    max(sign * (u - strike), 0)
                end
                s.currentValue[i, 1] = applyKreissSmoothing(obj, strike, x[i], h)
            end
        end
    end
end


function evaluate(p::KreissSmoothDefinition{VanillaAmerican{TS}}, s::DefaultFDMStructure{T}, x::AbstractArray{T}, columnIndex::Int) where {T,TS}
    payoff = p.delegate
    strike = payoff.strike
    if abs(s.currentTime - payoff.timeToExpiry) < eps(T)
        sign = one(T)
        if !payoff.isCall
            sign = -one(T)
        end
        for i = eachindex(x)
            if !isIndexBetween(i, x, strike)
                s.currentValue[i, 1] = max(sign * (x[i] - strike), zero(T))
            else
                h = (x[i+1] - x[i-1]) / 2
                obj = function (u::T)
                    max(sign * (u - strike), 0)
                end
                s.currentValue[i, 1] = applyKreissSmoothing(obj, strike, x[i], h)
            end
        end
    else
        evaluate(payoff, s, x, columnIndex)
        # y = copy(s.currentValue[:,1])
        # #ideally we would search for location such that value = LB. and apply smoothing there.
        # for i = 2:length(x)-1
        #     #one direction (put?)
        #     if (s.currentValue[i-1, 1] <= s.lowerBound[i-1] + sqrt(eps(T)) && s.currentValue[i+1, 1] > s.lowerBound[i+1]) || (s.currentValue[i+1, 1] <= s.lowerBound[i+1] + sqrt(eps(T)) && s.currentValue[i-1, 1] > s.lowerBound[i-1])
        #         sign = one(T)
        #         if !payoff.isCall
        #             sign = -one(T)
        #         end
        #         #apply smoothing

        #         p2 = y[i] / ((x[i] - x[i-1]) * (x[i] - x[i+1])) + y[i-1] / ((x[i-1] - x[i]) * (x[i-1] - x[i+1])) + y[i+1] / ((x[i+1] - x[i]) * (x[i+1] - x[i-1]))
        #         p1 = -sign - y[i] * (x[i+1] + x[i-1]) / ((x[i] - x[i-1]) * (x[i] - x[i+1])) - y[i-1] * (x[i] + x[i+1]) / ((x[i-1] - x[i]) * (x[i-1] - x[i+1])) - y[i+1] * (x[i-1] + x[i]) / ((x[i+1] - x[i]) * (x[i+1] - x[i-1]))
        #         p0 = sign * strike + y[i] * x[i+1] * x[i-1] / ((x[i] - x[i-1]) * (x[i] - x[i+1])) + y[i-1] * x[i] * x[i+1] / ((x[i-1] - x[i]) * (x[i-1] - x[i+1])) + y[i+1] * x[i-1] * x[i] / ((x[i+1] - x[i]) * (x[i+1] - x[i-1]))
        #         delta = p1^2 - 4p2 * p0 + eps(T)
        #         level1 = (-p1 - sqrt(delta)) / (2p2)
        #         level2 = (-p1 + sqrt(delta)) / (2p2)
        #         level = if (level1 < x[i+1] && x[i-1] <= level1)
        #             level1
        #         else
        #             level2
        #         end
        #         if (level < x[i+1] && x[i-1] <= level)
        #             #println("applying smoothing at t ", s.currentTime)
        #             obj = function (u::T)
        #                 value = y[i] * (u - x[i-1]) * (u - x[i+1]) / ((x[i] - x[i-1]) * (x[i] - x[i+1])) + y[i-1] * (u - x[i]) * (u - x[i+1]) / ((x[i-1] - x[i]) * (x[i-1] - x[i+1])) + y[i+1] * (u - x[i-1]) * (u - x[i]) / ((x[i+1] - x[i]) * (x[i+1] - x[i-1])) #lagrange on x[i-1],x[i],x[i+1]
        #                 intrinsic = max(sign * (u - strike), 0)
        #                 return max(value, intrinsic)
        #             end
        #             s.currentValue[i, 1] = applyKreissSmoothing(obj, level, x[i], (x[i+1] - x[i-1]) / 2)
        #         end
        #     end
        # end
    end
end


function applyKreissSmoothing(obj, strike::T, xi::T, h::T) where {T}
    objK = function (u::T)
        obj(u) * (1 - abs(u - xi) / h)
    end
    # println("modified index ",i, " ",x[i])
    if xi < strike
        return (simpson(objK, xi, strike - sqrt(eps(T))) + simpson(objK, strike + sqrt(eps(T)), xi + h) + simpson(objK, xi - h, xi)) / h
    else
        return (simpson(objK, xi - h, strike - sqrt(eps(T))) + simpson(objK, strike + sqrt(eps(T)), xi) + simpson(objK, xi, xi + h)) / h
    end
end

isIndexBetween(i, x, strike) = (i < length(x) && x[i+1] >= strike) && (i > 1 && x[i-1] <= strike)

function evaluate(p::KreissSmoothDefinition{DiscreteKO{TS}}, s::DefaultFDMStructure{T}, x::AbstractArray{T}, columnIndex::Int) where {T,TS}
    payoff = p.delegate
    evaluate(KreissSmoothDefinition(payoff.vanilla), s, x, columnIndex)
    isBarrierActive = false
    for t in payoff.observationTimes
        if abs(s.currentTime - t) < eps(T)
            isBarrierActive = true
            break
        end
    end
    if isBarrierActive
        y = copy(s.currentValue[:, 1])
        for i = eachindex(x)
            if !isIndexBetween(i, x, payoff.level)
                if (payoff.isDown && x[i] <= payoff.level) || (!payoff.isDown && x[i] >= payoff.level)
                    s.currentValue[i, 1] = payoff.rebate
                end
            else
                obj = function (u::T)
                    intrinsic = y[i] * (u - x[i-1]) * (u - x[i+1]) / ((x[i] - x[i-1]) * (x[i] - x[i+1])) + y[i-1] * (u - x[i]) * (u - x[i+1]) / ((x[i-1] - x[i]) * (x[i-1] - x[i+1])) + y[i+1] * (u - x[i-1]) * (u - x[i]) / ((x[i+1] - x[i]) * (x[i+1] - x[i-1])) #lagrange on x[i-1],x[i],x[i+1]
                    if (payoff.isDown && u <= payoff.level) || (!payoff.isDown && u >= payoff.level)
                        return payoff.rebate
                    else
                        return intrinsic
                    end
                end
                s.currentValue[i, 1] = applyKreissSmoothing(obj, payoff.level, x[i], (x[i+1] - x[i-1]) / 2)
            end
        end
    end
end


function simpson(f, a::T, b::T)::T where {T}
    (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))
end