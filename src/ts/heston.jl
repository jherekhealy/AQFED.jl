import CharFuncPricing: DoubleHestonParams, HestonParams, HestonTSParams
export HaganHestonPiecewiseParams, makeHestonTSParams,makeHaganHestonPiecewiseParams


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
    return HaganHestonPiecewiseParams([sqrt(p.params.θ)], p.κ, [p.ρ], [p.σ / sqrt(p.params.θ)], [0.0])
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

struct HestonModel{T}
    v0::T
    κ::T
    θ::T
    ρ::T
    σ::T
    r::T
    q::T
end


struct DoubleHestonModel{T}
    params::DoubleHestonParams{T}
    r::T
    q::T
end


