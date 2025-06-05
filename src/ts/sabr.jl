export SABRSection, SABRParams, Hagan2020, HaganEffective, normalVarianceByMoneyness

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

struct LeFloch <: SABRApprox
end

#From "Implied volatility formulas for Heston models"
struct HaganEffective <: SABRApprox
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
    blackVol = if y == 0
        α * factor
    else
        y / y0 * ν * factor
    end
    return blackVol^2
end

function varianceByLogmoneyness(s::SABRSection{T,LeFloch}, y) where {T}
    fpo = s.f + s.shift
    Kpo = fpo * exp(y)
    β = s.params.β
    ν = s.params.ν
    α = s.params.α
    ρ = s.params.ρ
    Kpowonebeta = Kpo^(1 - β)
    fpowonebeta = fpo^(1 - β)

    yK = if β == 1
        y
    else 
        (Kpowonebeta - fpowonebeta) / (1 - β)
    end
    χ = ν * yK / α
  
    delta0 = β * (2 - β) / (8 * fpowonebeta^2)
    e = sqrt(1 + 2 * ρ *    χ + χ^2)
    y0 = log((-ρ - χ + e) / (1 - ρ)) #x*nu
    gamma = (Kpowonebeta-fpowonebeta)/(Kpo - fpo)
    logoverx = if y==0
        α 
    else 
        -y / y0 * ν
    end
    DK = sqrt(α^2 + 2α*ρ*ν*yK+ν^2*yK^2)*Kpo^β
    Df = α*fpo^β
    g = -nu^2/y0^2*log(logoverx* sqrt(fpo*Kpo/(Df*DK)))
    factor = 1 + (g+ρ*ν*α*gamma/4) * s.tte
    blackVol = logoverx * factor
    return blackVol^2
end

function varianceByLogmoneyness(s::SABRSection{T,Obloj}, y) where {T}
    fpo = s.f + s.shift
    Kpo = fpo * exp(y)
    β = s.params.β
    ν = s.params.ν
    α = s.params.α
    ρ = s.params.ρ
    Kpowonebeta = Kpo^(1 - β)
    fpowonebeta = fpo^(1 - β)
    i0B = if (y == 0)
        α / Kpowonebeta
    elseif (ν == 0)
        -y * α * (1 - β) / (fpowonebeta - Kpowonebeta)
    elseif (β == 1)
        z = -ν * y / α
        if abs(z)<1e-4
    #        println("small z ",z)
            #log((sqrt(1 - 2 * ρ * z + z^2) + z - ρ) / (1 - ρ)) = z + ρ * z^2/2
            α/(1 + (ρ *z)/2 + (3*ρ^2 - 1)/6* z^2  + ρ*(5*ρ^2 - 3)/8*z^3)
            # + 1/40 (35 ρ^4 - 30 ρ^2 + 3) y^5 + 1/48 ρ (63 ρ^4 - 70 ρ^2 + 15) y^6 + O(y^7)
        else
        -y * ν / log((sqrt(1 - 2 * ρ * z + z^2) + z - ρ) / (1 - ρ))
        end
    else
        z = -ν / α * (fpowonebeta - Kpowonebeta) / (1 - β)
        -y * ν / log((sqrt(1 - 2 * ρ * z + z^2) + z - ρ) / (1 - ρ))
    end
    i1H = (β - 1)^2 * α^2 / (24 * fpowonebeta * Kpowonebeta) + ρ * ν * α * β / (4 * sqrt(fpowonebeta * Kpowonebeta)) + (2 - 3*ρ^2) * ν^2 / 24
   # println("i0B ", i0B, "i1H ",i1H, " ",β, " ",ν, " ",-ν * y / α)
    return (i0B * (1 + i1H * s.tte))^2
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


function varianceByLogmoneyness(s::SABRSection{T,HaganEffective}, y) where {T}
    fpo = s.f + s.shift
    Kpo = fpo * exp(y)
    β = s.params.β
    ν = s.params.ν
    α = s.params.α
    ρ = s.params.ρ
    Kpowonebeta = Kpo^(1 - β)
    fpowonebeta = fpo^(1 - β)
    αtilde = if β==1
        α*exp(ρ*ν*α/4*s.tte) 
    elseif β==0
        α
    else
        α*exp(ρ*ν*α/4*s.tte*β/fpowonebeta) 
    end
    χ = if β == 1
        ν * y / αtilde
    else
        ν / αtilde * (Kpowonebeta - fpowonebeta) / (1 - β)
    end
    delta0 = β * (2 - β) / (8 * fpowonebeta^2)
    e = sqrt(1 + 2 * ρ * χ + χ^2)
    y0 = log((ρ + χ + e) / (1 + ρ))
    factor = 1 + (-delta0 / 4 * αtilde^2 + (2 - 3 * ρ^2) * (ν^2) / 24) * s.tte
    blackVol = if y == 0
        αtilde * factor
    else
        y / y0 * ν * factor
    end
    return blackVol^2
end

#moneyness = strike - spot 
function normalVarianceByMoneyness(s::SABRSection{T,HaganEffective}, y) where {T}
    fpo = s.f + s.shift
    Kpo = y + fpo
    β = s.params.β
    ν = s.params.ν
    α = s.params.α
    ρ = s.params.ρ
    #println("SABRSection ",s)
    Kpowonebeta = Kpo^(1 - β)
    fpowonebeta = fpo^(1 - β)
    αtilde = if β==1
        α*exp(ρ*ν*α/4*s.tte) 
    elseif β==0
        α
    else
        α*exp(ρ*ν*α/4*s.tte*β/fpowonebeta) 
    end
    χ = if β == 1
        ν * log(Kpo / fpo) / αtilde
    else
        ν / αtilde * (Kpowonebeta - fpowonebeta) / (1 - β)
    end
    delta0 = β * (2 - β) / (8 * fpowonebeta^2)
    e = sqrt(1 + 2 * ρ * χ + χ^2)
    y0 = log((ρ + χ + e) / (1 + ρ))
  #  y0 = log((-ρ - χ + e) / (1 - ρ)) #x*nu
  
    factor = 1 + (-delta0 / 3 * αtilde^2 + (2 - 3 * ρ^2) * (ν^2) / 24) * s.tte
    bpvol = αtilde / fpowonebeta * fpo * factor
    if abs(Kpo - fpo) > eps(y)
        bpvol *= ν / αtilde * (Kpo - fpo) * fpowonebeta / (fpo * y0)
    end
    return bpvol^2
end

