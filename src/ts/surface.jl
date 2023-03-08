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
    y::Float64,
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
        v = (var1 * t1 * (t - t0) + var0 * t0 * (t1 - t)) / (t1 - t0)
        return v / t
    end
end


function varianceSlopeCurvature(
    s::VarianceSurfaceBySection{T},
    y::Float64, #could be a vector?
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
        s = (slope1 * t1 * (t - t0) + slope0 * t0 * (t1 - t)) / (t1 - t0)
        c = (curvature1 * t1 * (t - t0) + curvature0 * t0 * (t1 - t)) / (t1 - t0)
        return v / t, s / t, c / t
    end
end


#length(s::VarianceSurfaceBySection{T}) where {T} = 1

function varianceByLogmoneyness(
    s::VarianceSurfaceBySection{T},
    y::Array{Float64,1},
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
    y::Array{Float64,1}, #could be a vector?
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
        s = @. (slope1 * t1 * (t - t0) + slope0 * t0 * (t1 - t)) / (t1 - t0)
        c = @. (curvature1 * t1 * (t - t0) + curvature0 * t0 * (t1 - t)) / (t1 - t0)
        @. v /= t
        @. s /= t
        @. c /= t
        return v, s, c
    end
end

function localVarianceByLogmoneyness(
    surface::VarianceSurfaceBySection{T}, #another implementation could be based on call surface
    y::Array{Float64,1},
    t0::Float64,
    t1::Float64,
) where {T}
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
