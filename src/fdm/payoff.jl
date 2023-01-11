export VanillaEuropean, VanillaAmerican, StructureDefinition

abstract type StructureDefinition end

mutable struct FDMStructure{T,TD <: StructureDefinition}
    currentTime::T
    currentValue::Matrix{T}
    lowerBound::Vector{T}
    definition::TD    
end

function makeFDMStructure(p::TD, underlying::AbstractArray{T})::FDMStructure{T} where {T,TD <: StructureDefinition}
    return FDMStructure(zero(T),zeros(T,length(underlying),1),zeros(T,length(underlying)),p)
end


"advance payoff to given time. This is where one could update relevant indices/set current observation"
function advance(p::FDMStructure, time)
    p.currentTime = time
end

function currentValue(p::FDMStructure{T,TS})::Matrix{T} where {T,TS}
    return p.currentValue
end

function isLowerBoundActive(p::FDMStructure)::Bool
    return false
end

function lowerBound!(p::FDMStructure{T,TD}, v::AbstractArray{T}) where {T,TD}
    if !isempty(p.lowerBound)
        v[1:end] = p.lowerBound
    end
end

    function evaluate(p::FDMStructure{T,TD}, S::AbstractArray{T}) where {T,TD}
        for i=1:size(currentValue(p),2) 
            evaluate(p,S,i)
        end
    end


struct DiscountBond{T}<: StructureDefinition
    timeToExpiry::T
end

function makeFDMStructure(p::DiscountBond{TS}, underlying::AbstractArray{T})::FDMStructure{T} where {T,TS}
    return FDMStructure(p.timeToExpiry,zeros(T,length(underlying),1),zeros(T,length(underlying)),p)
end

"observation times sorted in ascending order"
function observationTimes(p::DiscountBond{T})::AbstractArray{T} where {T}
    return [p.timeToExpiry]
end
function isLowerBoundActive(p::FDMStructure{T,DiscountBond{T}})::Bool where {T}
    return false
end

function evaluate(p::FDMStructure{T,DiscountBond{TS}}, S::AbstractArray{T}, columnIndex::Int) where {T,TS}
    if p.currentTime == p.definition.timeToExpiry
        @. p.currentValue[:,1] = 1.0
    end
end

function nonSmoothPoints(p::DiscountBond{T})::AbstractArray{T} where {T}
    return []
end

mutable struct VanillaEuropean{T} <: StructureDefinition
    isCall::Bool
    strike::T
    timeToExpiry::T
    VanillaEuropean(isCall::Bool, strike::T,timeToExpiry::T) where{T} = new{T}(isCall,strike,timeToExpiry)
end

function makeFDMStructure(p::VanillaEuropean{TS}, underlying::AbstractArray{T})::FDMStructure{T} where {T,TS}
    return FDMStructure(p.timeToExpiry,zeros(T,length(underlying),1),zeros(T,length(underlying)),p)
end
function nonSmoothPoints(p::VanillaEuropean{T})::AbstractArray{T} where {T}
    return [p.strike]
end

"observation times sorted in ascending order"
function observationTimes(p::VanillaEuropean{T})::AbstractArray{T} where {T}
    return [p.timeToExpiry]
end
function isLowerBoundActive(p::FDMStructure{T,VanillaEuropean{T}})::Bool where {T}
    return true
end

function evaluate(p::FDMStructure{T,VanillaEuropean{TS}}, S::AbstractArray{T}, columnIndex::Int) where {T,TS}
    if p.currentTime == p.definition.timeToExpiry
        if p.definition.isCall
            @. p.currentValue[:,1] = max(S - p.definition.strike, zero(T))
        else
            @. p.currentValue[:,1] = max(p.definition.strike - S, zero(T))
        end
    end
end



struct VanillaAmerican{T} <: StructureDefinition
    isCall::Bool
    strike::T
    timeToExpiry::T
    exerciseStartTime::T
    VanillaAmerican(isCall::Bool, strike::T,timeToExpiry::T;exerciseStartTime=zero(T)) where {T} = new{T}(isCall,strike,timeToExpiry,exerciseStartTime)
end

function observationTimes(p::VanillaAmerican{T})::AbstractArray{T} where {T}
    return [p.timeToExpiry]
end

function nonSmoothPoints(p::VanillaAmerican{T})::AbstractArray{T} where {T}
    return [p.strike]
end

function makeFDMStructure(p::VanillaAmerican{TS}, underlying::AbstractArray{T})::FDMStructure{T} where {T,TS}
    lowerBound = if p.isCall
         @. max(underlying - p.strike, zero(T))
    else
         @. max(p.strike - underlying, zero(T))
    end
    return FDMStructure(p.timeToExpiry,zeros(T,length(underlying),1),lowerBound,p)
end


function evaluate(p::FDMStructure{T,VanillaAmerican{TS}}, S::AbstractArray{T}, columnIndex::Int) where {T,TS}
    if (p.currentTime >= p.definition.exerciseStartTime) 
        @. p.currentValue[:,1] = max(p.currentValue[:,1], p.lowerBound)  
    end
end

function isLowerBoundActive(p::FDMStructure{T,VanillaAmerican{T}})::Bool where {T}
    return p.currentTime >= p.definition.exerciseStartTime
end



struct ButterflyAmerican{T} <: StructureDefinition
    isCall::Bool
    strike1::T
    strike2::T
    timeToExpiry::T
    exerciseStartTime::T
    ButterflyAmerican(isCall::Bool, strike1::T, strike2::T,timeToExpiry::T;exerciseStartTime=zero(T)) where {T} = new{T}(isCall,strike1,strike2,timeToExpiry,exerciseStartTime)
end

function observationTimes(p::ButterflyAmerican{T})::AbstractArray{T} where {T}
    return [p.timeToExpiry]
end

function nonSmoothPoints(p::ButterflyAmerican{T})::AbstractArray{T} where {T}
    return [p.strike1, p.strike2]
end

function makeFDMStructure(p::ButterflyAmerican{TS}, underlying::AbstractArray{T})::FDMStructure{T} where {T,TS}
    lowerBound = if p.isCall
        @. (max(underlying - p.strike1, zero(T)) + max(underlying-p.strike2,zero(T)) - 2*max(underlying-(p.strike1+p.strike2)/2,zero(T)) )   
     else
        @. (max(p.strike1 - underlying, zero(T)) + max(p.strike2-underlying,zero(T)) - 2*max((p.strike1+p.strike2)/2-underlying,zero(T)) )
    end
    return FDMStructure(p.timeToExpiry,zeros(T,length(underlying),1),lowerBound,p)
end


function evaluate(p::FDMStructure{T,ButterflyAmerican{TS}}, S::AbstractArray{T}, columnIndex::Int) where {T,TS}
    if (p.currentTime >= p.definition.exerciseStartTime) 
        @. p.currentValue[:,1] = max(p.currentValue[:,1], p.lowerBound)  
    end
end

function isLowerBoundActive(p::FDMStructure{T,ButterflyAmerican{T}})::Bool where {T}
    return p.currentTime >= p.definition.exerciseStartTime
end
