import AQFED.TermStructure: SchobelZhuModel
export EAE, SchobelZhuLVModel

#Schöbel-Zhu scheme: exact Ornstein-Uhlenbeck volatility over the step, the log-asset uses the exact Itô integral
#∫V dZ = (V₁²-V₀²-σ²h)/(2σ) - κ/σ ∫V(θ-V)dt with the time integrals by the trapezoidal rule.
struct EAE <: PathGenerator
    withBB::Bool
end
EAE(; withBB::Bool=false) = EAE(withBB)

variatesPerStep(::SchobelZhuModel, ::EAE) = 2
bbcachetype(::EAE) = Array{Float64,2}

function GridConfig(
    model::SchobelZhuModel,
    specTimes::Vector{Float64},
    timestepSize::Float64,
    method::PathGenerator;
    mindt::Float64=timestepSize,
    tstop::Float64=0.0,
    maxdt::Float64=timestepSize,
)
    genTimes = pathgenTimesRamp(specTimes, mindt, tstop, maxdt)
    return GridConfig(specTimes, mindt, tstop, maxdt, genTimes, method)
end
ndims(model::SchobelZhuModel, specificTimes::Vector{Float64}, timestepSize::Float64) =
    ndims(model, GridConfig(model, specificTimes, timestepSize, EAE()))

#v is the volatility, u the standard normal variates of the asset (column 1) and of the volatility (column 2)
struct SZState{T,PV}
    logpathValues::Vector{T}
    v::Vector{T}
    obsPath::Matrix{T}
    payoffValues::PV
    u::Matrix{Float64}
end

init_scheme_state(::EAE, ::Type{T}, payoff, nSim::Int, nObs::Int) where {T} =
    SZState(Vector{T}(undef, nSim), Vector{T}(undef, nSim), Matrix{T}(undef, nSim, nObs), _allocate_payoff_values(payoff, nSim), Matrix{Float64}(undef, nSim, 2))

@inline function reset_state!(st::SZState, model::SchobelZhuModel, lnf0)
    fill!(st.logpathValues, lnf0)
    fill!(st.v, model.v0)
    return nothing
end

struct EAEPrecalc{T}
    ekdt::Vector{T}
    stdv::Vector{T} #standard deviation of the volatility over the step
end

function precompute_scheme(::EAE, model::SchobelZhuModel{T}, common::CommonSimData{T}) where {T}
    ekdt = @. exp(-model.κ * common.h)
    stdv = @. model.σ * sqrt((1 - ekdt^2) / (2model.κ))
    return EAEPrecalc(ekdt, stdv)
end

function load_step_variates!(st::SZState, ::EAE, rng, start::Int, dim::Int, common::CommonSimData, withBB::Bool, bb, cache)
    u = st.u
    if withBB
        transformByDim!(bb, rng, start, dim, u, cache)
        @. u *= inv(common.sqrth[dim])
    else
        @views for k = 1:2
            d = 2 * (dim - 1) + k
            skipTo!(rng, d, start, size(u, 1))
            nextn!(rng, d, u[:, k])
        end
    end
    return nothing
end

#one step of the log-asset x and the volatility v, with the leverage L of the local volatility (L=1 for Schöbel-Zhu).
@inline function _stepSZ(x, v, z0, z1, L, lndrift, h, ekdt, stdv, κ, θ, ρσ, σ, ρBar)
    vnew = v * ekdt + θ * (1 - ekdt) + stdv * z1
    v2Integral = (vnew^2 + v^2) * h / 2
    vIntegral = (vnew + v) * h / 2
    xnew = x + lndrift + ρσ / 2 * L * (vnew^2 - v^2 - σ^2 * h) + (κ * ρσ - L / 2) * L * v2Integral - θ * κ * ρσ * L * vIntegral +
           ρBar * L * sqrt(v2Integral) * z0
    return xnew, vnew
end

function step!(st::SZState{T}, ::EAE, model::SchobelZhuModel{T}, common::CommonSimData{T}, precalc::EAEPrecalc{T}, dim::Int) where {T}
    h = common.h[dim]
    lndrift = common.lnf[dim+1] - common.lnf[dim]
    ekdt, stdv = precalc.ekdt[dim], precalc.stdv[dim]
    κ, θ, σ = model.κ, model.θ, model.σ
    ρσ = model.ρ / σ
    ρBar = common.ρBar
    @inbounds @simd for j in eachindex(st.v)
        st.logpathValues[j], st.v[j] = _stepSZ(st.logpathValues[j], st.v[j], st.u[j, 1], st.u[j, 2], one(T), lndrift, h, ekdt, stdv, κ, θ, ρσ, σ, ρBar)
    end
    return nothing
end

# ============================================================
# Schöbel-Zhu stochastic local volatility
# ============================================================

"""
    SchobelZhuLVModel(sz::SchobelZhuModel, lv::LocalVolatilityModel)

Schöbel-Zhu stochastic local volatility model: d ln S = (r-q-L²V²/2)dt + L(t,S)V dW with V the Schöbel-Zhu volatility and
the leverage L² = σ²_LV(t,S)/E[V²|S_t=S], calibrated by `calibrateLeverage` and simulated by the `EAE` scheme.
"""
struct SchobelZhuLVModel{S<:SchobelZhuModel,L<:LocalVolatilityModel} <: StochasticLocalVolatilityModel
    sz::S
    lv::L
end
stochasticVolatilityModel(model::SchobelZhuLVModel) = model.sz
variatesPerStep(::SchobelZhuLVModel, ::EAE) = 2
ndims(model::SchobelZhuLVModel, specificTimes::Vector{Float64}, timestepSize::Float64) =
    ndims(model, GridConfig(model, specificTimes, timestepSize, EAE()))

_particle_state(::EAE, ::Type{T}, m::Int) where {T} =
    SZState(Vector{T}(undef, m), Vector{T}(undef, m), Matrix{T}(undef, m, 0), T[], Matrix{Float64}(undef, m, 2))
#the particle variance is the square of the Schöbel-Zhu volatility
@views _copy_particle_variance!(v::AbstractVector, lo::Int, st::SZState) = (v[lo:lo+length(st.v)-1] .= st.v .^ 2)

function step!(st::SZState{T}, ::EAE, model::SchobelZhuLVModel, common::CommonSimData{T}, precalc::SLVPrecalc, dim::Int) where {T}
    localVariance = DiscreteLocalVarianceSlice(precalc.surface, common.genTimes[dim], common.genTimes[dim+1])
    #function barrier: the conditional variance type may change with the step
    _stepSZLV!(st, model.sz, localVariance, common, precalc.scheme, dim, precalc.conditionalVariances[dim])
end

function _stepSZLV!(st::SZState{T}, model::SchobelZhuModel, localVariance::LV, common::CommonSimData{T}, precalc::EAEPrecalc{T}, dim::Int, expectedVariance::F) where {T,LV,F}
    h = common.h[dim]
    lnf0 = common.lnf[dim]
    lndrift = common.lnf[dim+1] - lnf0
    ekdt, stdv = precalc.ekdt[dim], precalc.stdv[dim]
    κ, θ, σ = model.κ, model.θ, model.σ
    ρσ = model.ρ / σ
    ρBar = common.ρBar
    @inbounds for j in eachindex(st.v)
        x = st.logpathValues[j]
        L = sqrt(localVariance(x - lnf0) / expectedVariance(x))
        st.logpathValues[j], st.v[j] = _stepSZ(x, st.v[j], st.u[j, 1], st.u[j, 2], L, lndrift, h, ekdt, stdv, κ, θ, ρσ, σ, ρBar)
    end
    return nothing
end
