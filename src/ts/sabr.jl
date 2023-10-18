export SABRSection, SABRParams, Hagan2020, normalVarianceByMoneyness

struct SABRParams{T}
    α::T
    β::T
    ρ::T
    ν::T
end

abstract type SABRApprox end

struct Obloj <: SABRApprox
end

struct Hagan2020 <: SABRApprox
end


struct SABRSection{T,M<:SABRApprox} <: VarianceSection
    approx::M
    params::SABRParams{T}
    tte::Float64
    f::Float64
    shift::Float64
end

Base.broadcastable(p::SABRSection) = Ref(p)
# logmoneyness= log(strike+shift / spot +shift) to allow negative strikes/spots.
function varianceByLogmoneyness(s::SABRSection{T,Hagan2020}, y) where {T}
    fpo = s.f + s.shift
    Kpo = fpo * exp(y)
    β = s.params.β
    ν = s.params.ν
    α = s.params.α
    ρ = s.params.ρ
    Kpowonebeta = Kpo^(1 - β)
    fpowonebeta = fpo^(1 - β)

    χ = if β == 1
       ν * y / α
    else
        ν / α * (Kpowonebeta - fpowonebeta) / (1 - β)
    end
    delta0 = β * (2 - β) / (8 * fpowonebeta^2)
    e = sqrt(1 + 2 * ρ * χ + χ^2)
    y0 = log((-ρ - χ + e) / (1 - ρ))
    factor = 1 + (-delta0 / 4 * α^2 + ρ * α * ν * β / (4 * fpowonebeta) + (2 - 3 * ρ^2) * (ν^2) / 24) * s.tte    
    blackVol = if y == 0 α*factor else y/y0*ν*factor end
    return blackVol^2
end

#moneyness = strike - spot 
function normalVarianceByMoneyness(s::SABRSection{T,Hagan2020}, y) where {T}
    fpo = s.f + s.shift
    Kpo = y + fpo
    β = s.params.β
    ν = s.params.ν
    α = s.params.α
    ρ = s.params.ρ
    Kpowonebeta = Kpo^(1 - β)
    fpowonebeta = fpo^(1 - β)

    χ = if β == 1
        ν * log(Kpo / fpo) / α
    else
        ν / α * (Kpowonebeta - fpowonebeta) / (1 - β)
    end
    delta0 = β * (2 - β) / (8 * fpowonebeta^2)
    e = sqrt(1 + 2 * ρ * χ + χ^2)
    y0 = log((ρ + χ + e) / (1 + ρ))
    factor = 1 + (-delta0 / 3 * α^2 + ρ * α * ν * β / (4 * fpowonebeta) + (2 - 3 * ρ^2) * (ν^2) / 24) * s.tte
    bpvol = α / fpowonebeta * fpo * factor
    if abs(Kpo - fpo) > eps(y)
        bpvol *= ν / α * (Kpo - fpo) * fpowonebeta / (fpo * y0)
    end
    return bpvol^2
end
