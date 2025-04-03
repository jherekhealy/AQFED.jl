import CharFuncPricing:
DoubleHestonParams

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
