export Curve, ConstantRateCurve, InterpolatedDiscountFactorCurve
export df
using PPInterpolation
abstract type Curve end

struct ConstantRateCurve{T} <: Curve
    r::T
end

function df(c::ConstantRateCurve{T}, time::TX)::T where {T, TX}
    return exp(-c.r*time)
end

struct InterpolatedDiscountFactorCurve{TX,T} <: Curve
    pp::PPInterpolation.PP{3,TX,T} #make sure to add time 0 and TODO extrapolation
end
function df(c::InterpolatedDiscountFactorCurve{TX,T}, time::TX)::T where {T,TX}
    return c.pp(time)
end

