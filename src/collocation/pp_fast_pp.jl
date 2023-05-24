using BSplines
import PPInterpolation:bspp,PP

const FastPP{N,T,TX} = PP{N,T,TX,Vector{T},Matrix{T},Vector{TX}}  
const CubicPP{T,TX} = PP{3,T,TX,Vector{T},Matrix{T},Vector{TX}} 
const QuadraticPP{T,TX} = PP{2,T,TX,Vector{T},Matrix{T},Vector{TX}} 
function Base.convert(::Type{QuadraticPP{T,TX}}, spl::BSplines.Spline) where {T,TX}
    return Base.convert(PP{2,T,TX},spl)
end

# function Base.convert(::Type{CubicPP{T,TX}}, spl::BSplines.Spline) where {T,TX}
#     return Base.convert(PP{3,T,TX},spl)
# end
function Base.convert(::Type{FastPP{N,T,TX}}, spl::BSplines.Spline) where {N,T,TX}
        t = BSplines.knots(spl.basis)
        α = spl.coeffs
        n = length(α)
        breakA, coef, l = bspp(t, α, n, 4)
        x = Vector(breakA)
        a = Vector(coef[1, :])
        b = Vector(coef[2, :])
        c = Matrix(coef[3:4, 1:l]')
        dx = (x[l+1] - x[l])
        a[l+1] = a[l] + dx * (b[l] + dx * (c[l, 1] + dx * (c[l, 2])))
        b[l+1] = b[l] + dx * (2 * c[l, 1] + dx * 3 * c[l, 2])
        return PPInterpolation.PP(3, a, b, c, x)
end
