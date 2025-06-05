export Curve, ConstantRateCurve, InterpolatedLogDiscountFactorCurve, SpreadCurve, OppositeCurve
export PiecewiseConstantFunction
export df
using PPInterpolation
abstract type Curve end

struct ConstantRateCurve{T} <: Curve
    r::T
end

function discountFactor(c::ConstantRateCurve{T}, time::TX)::T where {T, TX}
    return exp(-c.r*time)
end

struct InterpolatedLogDiscountFactorCurve{TX,T} <: Curve
    pp::PPInterpolation.PP{3,TX,T} #make sure to add time 0 and TODO extrapolation
end
function discountFactor(c::InterpolatedLogDiscountFactorCurve{TX,T}, time::TZ)::T where {T,TX,TZ}
    return exp(logDiscountFactor(c,time))
end

function logDiscountFactor(c::InterpolatedLogDiscountFactorCurve{TX,T}, time::TZ)::T where {T,TX,TZ}
    return c.pp(time)
end


struct OppositeCurve{T<: Curve} <: Curve
    delegate::T
end

function discountFactor(c::OppositeCurve{TX}, time::TZ) where {TX,TZ}
    return exp(logDiscountFactor(c,time))
end

function logDiscountFactor(c::OppositeCurve{TX}, time::TZ) where {TX,TZ}
    return -logDiscountFactor(c.delegate,time)
end


struct SpreadCurve{T<: Curve, TB <: Curve} <: Curve
    curve::T
    baseCurve::TB
end

function discountFactor(c::SpreadCurve{T,TB}, time::TZ) where {T,TB,TZ}
    return exp(logDiscountFactor(c,time))
end

function logDiscountFactor(c::SpreadCurve{T,TB}, time::TZ) where {T,TB,TZ}
    return logDiscountFactor(c.curve,time)-logDiscountFactor(c.baseCurve,time)
end

struct PiecewiseConstantFunction{VX,VT}
    x::VX
    a::VT
end

function makePiecewiseConstantFunction(x::AbstractArray{TX}, y::AbstractArray{T}) where {TX, T}
    return PiecewiseConstantFunction(x,y)
end
Base.length(p::PiecewiseConstantFunction) = Base.length(p.x)
Base.size(p::PiecewiseConstantFunction) = Base.size(p.x)
Base.broadcastable(p::PiecewiseConstantFunction) = Ref(p)

(spl::PiecewiseConstantFunction{TX,T})(x::TZ) where {TX,T,TZ} = evaluate(spl, x)
function (spl::PiecewiseConstantFunction{TX,T})(x::AbstractArray) where {TX,T}
    evaluate.(spl, x)
end

function evaluate(p::PiecewiseConstantFunction{TX,T}, x::TZ) where {TX,T,TZ}
    ppIndex = searchsortedlast(p.x,x)#   x[i]<=z<x[i+1]
    #ppIndex = min(max(ppIndex, 2), length(p.x))
    return p.a[ppIndex]
end