export VanillaEuropean, VanillaAmerican
abstract type FDPayoff end
    
    function evaluate(p::FDPayoff, S::AbstractArray{T}) where {T}
        for i=1:size(currentValue(p),2) 
            evaluate(p,S,i)
        end
    end

mutable struct VanillaEuropean{T} <: FDPayoff
    isCall::Bool
    strike::T
    timeToExpiry::T
    #below is internal
    currentTime::T
    currentValue::Matrix{T} #last column is full structure value, other columns are fore intermediate (dependent) values. first column is solved first
    VanillaEuropean(isCall::Bool, strike::T,timeToExpiry::T) where{T} = new{T}(isCall,strike,timeToExpiry,zero(T),zeros(T,0,0))
end

"observation times sorted in ascending order"
function observationTimes(p::VanillaEuropean{T})::AbstractArray{T} where {T}
    return [p.timeToExpiry]
end

"advance payoff to given time. This is where one could update relevant indices/set current observation"
function advance(p::VanillaEuropean{T}, time::T) where {T}
    p.currentTime = time
end
function initialize(p::VanillaEuropean{T}, S::AbstractArray{T}) where {T}
    p.currentValue = zeros(T, length(S), 1)
end

function evaluate(p::VanillaEuropean{T}, S::AbstractArray{T}, columnIndex::Int) where {T}
    if p.currentTime == p.timeToExpiry
        if p.isCall
            @. p.currentValue[:,1] = max(S - p.strike, zero(T))
        else
            @. p.currentValue[:,1] = max(p.strike - S, zero(T))
        end
    end
end

function currentValue(p::VanillaEuropean{T})::Matrix{T} where {T}
    return p.currentValue
end

mutable struct VanillaAmerican{T} <: FDPayoff
    isCall::Bool
    strike::T
    timeToExpiry::T
    exerciseStartTime::T
    #below is internal
    currentTime::T
    currentValue::Matrix{T}
    lowerBound::Vector{T}
    VanillaAmerican(isCall::Bool, strike::T,timeToExpiry::T;exerciseStartTime=zero(T)) where {T} = new{T}(isCall,strike,timeToExpiry,exerciseStartTime, zero(T),zeros(T,0,0),zeros(T,0))
end

function observationTimes(p::VanillaAmerican{T})::AbstractArray{T} where {T}
    return [p.timeToExpiry]
end
function advance(p::VanillaAmerican{T}, time::T) where {T}
    p.currentTime = time
end
function initialize(p::VanillaAmerican{T}, S::AbstractArray{T}) where {T}
    p.currentValue[:,1] = zeros(T, length(S), 1)
    if p.isCall
        @. p.lowerBound = max(S - p.strike, zero(T))
    else
        @. p.lowerBound = max(p.strike - S, zero(T))
    end
end

function evaluate(p::VanillaAmerican{T}, S::AbstractArray{T}, columnIndex::Int) where {T}
    if (p.currentTime >= p.exerciseStartTime) 
        @. p.currentValue[:,1] = max(p.currentValue[:,1], lowerBound)  
    end
end

function currentValue(p::VanillaAmerican{T})::Matrix{T} where {T}
    return p.currentValue
end