import CharFuncPricing: DoubleHestonParams, HestonParams, HestonTSParams
export HaganHestonPiecewiseParams, makeHestonTSParams,makeHaganHestonPiecewiseParams, HestonModel


struct HaganHestonPiecewiseParams{T}
    leverage::Vector{T}
    κ::T
    ρ::Vector{T}
    σ::Vector{T}
    startTime::Vector{T}
end

function makeHaganHestonPiecewiseParams(p::HestonParams)  
    if p.θ != p.v0 
        throw(DomainError(p.θ, "θ != v0"))
    end
    return HaganHestonPiecewiseParams([sqrt(p.θ)], p.κ, [p.ρ], [p.σ / sqrt(p.θ)], [0.0])
end


function convertTime(p::HaganHestonPiecewiseParams, t)
    lastIndex = searchsortedfirst(p.startTime,t)
    ts = vcat(p.startTime[1:lastIndex-1],t)
    τsum = zero(t)
    n = length(ts) 
    for i = 1:n-1
        dt = ts[i+1]-ts[i]
        τsum += p.leverage[i]^2 * dt              
    end
    return τsum
end
    
function makeHestonTSParams(p::HaganHestonPiecewiseParams; tte=10.0)
    #time change t'=int sigma^2 du
    lastIndex = searchsortedfirst(p.startTime,tte)
    ts = vcat(p.startTime[1:lastIndex-1],tte)
    τsum = zero(p.σ[1])
    n = length(ts) 
    κ = zeros(n-1)
    σ = zeros(n-1)
    θ = zeros(n-1)
    startTime = zeros(n-1)    
    for i = 1:n-1
        startTime[i] = τsum
        dt = ts[i+1]-ts[i]
        τsum += p.leverage[i]^2 * dt        
        κ[i] = p.κ / p.leverage[i]^2
        σ[i] = p.σ[i] / p.leverage[i]
        θ[i] = 1.0
    end
    v0 = 1.0
    return HestonTSParams(v0, κ, θ, p.ρ, σ, startTime), τsum
    #NOTE: in between times, models are not equivalent.
end

struct HestonModel{T,C1,C2}
    v0::T
    κ::T
    θ::T
    ρ::T
    σ::T
    driftCurve::C1
    discountCurve::C2
end

function HestonModel(v0::T, κ::T, θ::T, ρ::T, σ::T, r::T, q::T ) where {T}
    forwardCurve = ConstantRateCurve(r-q)
    discountCurve = ConstantRateCurve(r)
    return HestonModel(v0, κ, θ, ρ, σ, forwardCurve, discountCurve)
end

function HestonModel(p::HestonParams{T}, r::T, q::T ) where {T}
    forwardCurve = ConstantRateCurve(r-q)
    discountCurve = ConstantRateCurve(r)
    return HestonModel(p.v0, p.κ, p.θ, p.ρ, p.σ, forwardCurve, discountCurve)
end


function HestonModel(p::HestonParams{T}, driftCurve, discountCurve ) where {T}
    return HestonModel(p.v0, p.κ, p.θ, p.ρ, p.σ, driftCurve, discountCurve)
end

function discountFactor(model::HestonModel{S, C1, C2}, t) where {S, C1 <: Curve, C2 <: Curve}
	discountFactor(model.discountCurve, t)
end

function logDiscountFactor(model::HestonModel{S, C1, C2}, t) where {S, C1 <: Curve, C2 <: Curve}
	log(discountFactor(model.discountCurve, t))
end

function logForward(model::HestonModel{S, C1, C2}, lnspot, t) where {S, C1 <: Curve, C2 <: Curve}
	return lnspot - log(discountFactor(model.driftCurve, t))
end

function forward(model::HestonModel{S, C1, C2}, spot, t) where {S, C1 <: Curve, C2 <: Curve}
	return spot / discountFactor(model.driftCurve, t)
end
feller_margin(p::HestonParams) = 2 * p.κ * p.θ - p.σ^2
feller_ratio(p::HestonParams) = 2 * p.κ * p.θ / p.σ^2
feller_satisfied(p::HestonParams) = feller_margin(p) >= 0

struct DoubleHestonModel{T}
    params::DoubleHestonParams{T}
    r::T
    q::T
end


