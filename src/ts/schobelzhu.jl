import CharFuncPricing: SchobelZhuParams
export SchobelZhuModel

#Schöbel-Zhu model: d ln S = (r-q-V²/2)dt + V dW, dV = κ(θ-V)dt + σ dZ, d<W,Z> = ρ dt, V is the volatility (Ornstein-Uhlenbeck).
struct SchobelZhuModel{T,C1,C2}
    v0::T
    κ::T
    θ::T
    ρ::T
    σ::T
    driftCurve::C1
    discountCurve::C2
end

SchobelZhuModel(v0::T, κ::T, θ::T, ρ::T, σ::T, r::T, q::T) where {T} =
    SchobelZhuModel(v0, κ, θ, ρ, σ, ConstantRateCurve(r - q), ConstantRateCurve(r))
SchobelZhuModel(p::SchobelZhuParams{T}, r::T, q::T) where {T} = SchobelZhuModel(p.v0, p.κ, p.θ, p.ρ, p.σ, r, q)
SchobelZhuModel(p::SchobelZhuParams, driftCurve, discountCurve) = SchobelZhuModel(p.v0, p.κ, p.θ, p.ρ, p.σ, driftCurve, discountCurve)

discountFactor(model::SchobelZhuModel, t) = discountFactor(model.discountCurve, t)
logDiscountFactor(model::SchobelZhuModel, t) = log(discountFactor(model.discountCurve, t))
logForward(model::SchobelZhuModel, lnspot, t) = lnspot - log(discountFactor(model.driftCurve, t))
forward(model::SchobelZhuModel, spot, t) = spot / discountFactor(model.driftCurve, t)
