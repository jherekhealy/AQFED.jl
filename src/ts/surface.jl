#TermStructure volatility surface
export VarianceSurface, FlatSurface, FlatSection, VarianceSurfaceBySection
export varianceByLogmoneyness, localVarianceByLogmoneyness
abstract type VarianceSection end

struct FlatSection <: VarianceSection 
    σ::Float64
    tte::Float64
end

function varianceByLogmoneyness(s::FlatSection, y::Float64)
    return s.σ^2
end

abstract type VarianceSurface end
struct VarianceSurfaceBySection{T} <: VarianceSurface
    sections::Vector{T}
    expiries::Vector{Float64}
    #    indexMap::Any
end
struct FlatSurface <: VarianceSurface
    σ::Float64
end
function varianceByLogmoneyness(
    s::FlatSurface,
    y::Float64,
    t::Float64,
    indexTime::Int = 0)::Float64
    return s.σ^2
end
function localVarianceByLogmoneyness(
    surface::FlatSurface, 
    y::Array{Float64,1},
    t0::Float64,
    t1::Float64) 
    return s.σ^2
end

#abstract type LocalVarianceSurface end

function varianceByLogmoneyness(
    s::VarianceSurfaceBySection{T},
    y::Real,
    t::Real,
    indexTime::Int = 0,
) where {T}
    if t <= s.expiries[1]
        return varianceByLogmoneyness(s.sections[1], y)
    elseif t >= s.expiries[end]
        return varianceByLogmoneyness(s.sections[end], y)
    else
        if indexTime == 0
            indexTime = searchsortedlast(s.expiries, t)
        end
        t0 = s.expiries[indexTime]
        var0 = varianceByLogmoneyness(s.sections[indexTime], y)
        t1 = s.expiries[indexTime+1]
        var1 = varianceByLogmoneyness(s.sections[indexTime+1], y)
        #linear interpolation in total variance along same logmoneyness
        v = (var1 * t1 * (t - t0) + var0 * t0 * (t1 - t)) / (t1 - t0)
        return v / t
    end
end


function varianceSlopeCurvature(
    s::VarianceSurfaceBySection{T},
    y::Real, #could be a vector?
    t::Float64,
    indexTime::Int = 0, #for perf, have a version with t, tIndex
) where {T}
    if t <= s.expiries[1]
        return varianceSlopeCurvature(s.sections[1], y)
    elseif t >= s.expiries[end]
        return varianceSlopeCurvature(s.sections[end], y)
    else
        if indexTime == 0
            indexTime = searchsortedlast(s.expiries, t)
        end
        t0 = s.expiries[indexTime]
        var0, slope0, curvature0 = varianceSlopeCurvature(s.sections[indexTime], y)
        t1 = s.expiries[indexTime+1]
        var1, slope1, curvature1 = varianceSlopeCurvature(s.sections[indexTime+1], y)
        #linear interpolation in total variance along same logmoneyness
        v = (var1 * t1 * (t - t0) + var0 * t0 * (t1 - t)) / (t1 - t0)
        slope = (slope1 * t1 * (t - t0) + slope0 * t0 * (t1 - t)) / (t1 - t0)
        c = (curvature1 * t1 * (t - t0) + curvature0 * t0 * (t1 - t)) / (t1 - t0)
        return v / t, slope / t, c / t
    end
end


#length(s::VarianceSurfaceBySection{T}) where {T} = 1

function varianceByLogmoneyness(
    s::VarianceSurfaceBySection{T},
    y::AbstractVector{<:Real},
    t::Float64,
    indexTime::Int = 0,
) where {T}
    if t <= s.expiries[1]
        return varianceByLogmoneyness(s.sections[1], y)
    elseif t >= s.expiries[end]
        return varianceByLogmoneyness(s.sections[end], y)
    else
        if indexTime == 0
            indexTime = searchsortedlast(s.expiries, t)
        end
        t0 = s.expiries[indexTime]
        var0 = varianceByLogmoneyness(s.sections[indexTime], y)
        t1 = s.expiries[indexTime+1]
        var1 = varianceByLogmoneyness(s.sections[indexTime+1], y)
        #linear interpolation in total variance along same logmoneyness
        v = @. (var1 * t1 * (t - t0) + var0 * t0 * (t1 - t)) / (t1 - t0)
        @. v /= t
        return v
    end
end


function varianceSlopeCurvature(
    s::VarianceSurfaceBySection{T},
    y::AbstractVector{<:Real}, #could be a vector?
    t::Float64,
    indexTime::Int = 0, #for perf, have a version with t, tIndex
) where {T}
    if t <= s.expiries[1]
        return varianceSlopeCurvature(s.sections[1], y)
    elseif t >= s.expiries[end]
        return varianceSlopeCurvature(s.sections[end], y)
    else
        if indexTime == 0
            indexTime = searchsortedlast(s.expiries, t)
        end
        t0 = s.expiries[indexTime]
        var0, slope0, curvature0 = varianceSlopeCurvature(s.sections[indexTime], y)
        t1 = s.expiries[indexTime+1]
        var1, slope1, curvature1 = varianceSlopeCurvature(s.sections[indexTime+1], y)
        #linear interpolation in total variance along same logmoneyness
        v = @. (var1 * t1 * (t - t0) + var0 * t0 * (t1 - t)) / (t1 - t0)
        slope = @. (slope1 * t1 * (t - t0) + slope0 * t0 * (t1 - t)) / (t1 - t0)
        c = @. (curvature1 * t1 * (t - t0) + curvature0 * t0 * (t1 - t)) / (t1 - t0)
        @. v /= t
        @. slope /= t
        @. c /= t
        return v, slope, c
    end
end

function localVarianceByLogmoneyness(
    surface::VarianceSurfaceBySection{T}, #another implementation could be based on call surface
    y::V,
    t0::Float64,
    t1::Float64,
) where {T, V <: Union{AbstractVector{<:Real}, Real}}
    indexT0 = searchsortedlast(surface.expiries, t0)
    indexT1 = searchsortedlast(surface.expiries, t1)

    t = t0
    #out = map(y -> varianceSlopeCurvature(surface, y, t, indexT0), y)
    #w = getindex.(out, 1)
    #dw = getindex.(out, 2)
    #d2w = getindex.(out, 3)
    w, dw, d2w = varianceSlopeCurvature(surface, y, t, indexT0)
    denom = @. ifelse(
        w == 0,
        1,
        1 - y / w * dw + 0.25 * (-0.25 * t^2 - t / w + y^2 / w^2) * dw^2 + 0.5 * d2w * t,
    )
    #w1 = map(y -> varianceByLogmoneyness(surface, y, t1, indexT1), y)
    w1 = varianceByLogmoneyness(surface, y, t1, indexT1)
    #may be worth using mid point t= (t0+t1)/2 and w(t0) below.
    num = @. (w1 * t1 - w * t0) / (t1 - t0)
    lv = map(
        (num, denom) -> ifelse(num < 0, 0, ifelse(denom < 0, 10.0, num / denom)),
        num,
        denom,
    )

end


export discreteLocalVarianceByLogmoneyness, DiscreteLocalVarianceSlice
#sections i, j and weights a, b of the total variance a*w_i(y) + b*w_j(y) at t (linear in total variance, flat variance outside).
@inline function totalVarianceWeights(surface::VarianceSurfaceBySection, t::Real)
    expiries = surface.expiries
    n = length(expiries)
    if t <= expiries[1]
        return 1, 1, t, zero(t)
    elseif t >= expiries[n]
        return n, n, t, zero(t)
    end
    i = searchsortedlast(expiries, t)
    ti, tj = expiries[i], expiries[i+1]
    return i, i + 1, ti * (tj - t) / (tj - ti), tj * (t - ti) / (tj - ti)
end

#y ↦ local variance over [t0,t1] at the log-moneyness y: the forward difference of the total implied variance divided by
#the Dupire denominator in log-moneyness at t1. Capped to [0,10] where the surface allows arbitrages.
#The time interpolation is precomputed, for the many evaluations of a Monte-Carlo time step.
struct DiscreteLocalVarianceSlice{S,T}
    sections::Vector{S}
    i1::Int
    j1::Int
    a1::T
    b1::T
    i0::Int
    j0::Int
    a0::T
    b0::T
    t1::T
    invT1::T
    invDt::T
end

function DiscreteLocalVarianceSlice(surface::VarianceSurfaceBySection, t0::Real, t1::Real)
    i1, j1, a1, b1 = totalVarianceWeights(surface, t1)
    i0, j0, a0, b0 = totalVarianceWeights(surface, t0)
    T = typeof(a1 / 1)
    return DiscreteLocalVarianceSlice(surface.sections, i1, j1, T(a1), T(b1), i0, j0, T(a0), T(b0), T(t1), T(1 / t1), T(1 / (t1 - t0)))
end

@inline function (s::DiscreteLocalVarianceSlice)(y)
    @inbounds wi, dwi, d2wi = varianceSlopeCurvature(s.sections[s.i1], y)
    @inbounds wj, dwj, d2wj = varianceSlopeCurvature(s.sections[s.j1], y)
    totalVariance1 = s.a1 * wi + s.b1 * wj
    totalVariance0 = if s.i0 == s.i1 && s.j0 == s.j1 #the sections are evaluated once in general
        s.a0 * wi + s.b0 * wj
    else
        @inbounds s.a0 * varianceByLogmoneyness(s.sections[s.i0], y) + s.b0 * varianceByLogmoneyness(s.sections[s.j0], y)
    end
    t1 = s.t1
    w = totalVariance1 * s.invT1
    dw = (s.a1 * dwi + s.b1 * dwj) * s.invT1
    d2w = (s.a1 * d2wi + s.b1 * d2wj) * s.invT1
    invW = inv(w)
    yw = y * invW
    denom = iszero(w) ? one(w) : 1 - yw * dw + (-t1^2 / 4 - t1 * invW + yw^2) * dw^2 / 4 + d2w * t1 / 2
    num = (totalVariance1 - totalVariance0) * s.invDt
    lv = if num < 0
        zero(num)
    elseif denom <= 0
        10 * one(num)
    else
        num / denom
    end
    return min(lv, 10 * one(lv))
end

discreteLocalVarianceByLogmoneyness(surface::VarianceSurfaceBySection, y::Real, t0::Real, t1::Real) =
    DiscreteLocalVarianceSlice(surface, t0, t1)(y)
