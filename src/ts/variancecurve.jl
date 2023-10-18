abstract type VarianceCurve end

struct ConstantVarianceCurve{T} <: VarianceCurve
    v0::T
end

variance(c::ConstantVarianceCurve{T}, t) where {T} = c.v0
    

struct InterpolatedVarianceCurve{TX,T} <: VarianceCurve
    pp::PPInterpolation.PP{3,TX,T} #make sure to add time 0 and TODO extrapolation
end

variance(c::InterpolatedVarianceCurve, t) = pp(t)


