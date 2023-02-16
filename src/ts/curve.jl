export Curve, ConstantRateCurve, InterpolatedLogDiscountFactorCurve, SpreadCurve, OppositeCurve
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
