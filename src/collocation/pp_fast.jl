using BSplines
import PPInterpolation:bspp,PP,FIRST_DERIVATIVE,evaluateDerivative,C2,makeCubicPP

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
        breakA, coef, l = bspp(t, α, n, N+1)
        x = Vector(breakA)
        a = Vector(coef[1, :])
        b = Vector(coef[2, :])
        c = Matrix(coef[3:4, 1:l]')
        dx = (x[l+1] - x[l])
        a[l+1] = a[l] + dx * (b[l] + dx * (c[l, 1] + dx * (c[l, 2])))
        b[l+1] = b[l] + dx * (2 * c[l, 1] + dx * 3 * c[l, 2])
        return PP(3, a, b, c, x)
end

function changeSupport(pp::FastPP{3,T,TX}, xa::AbstractArray{TX})::FastPP{3,T,TX} where {N,T,TX}
#xa has same endpoints as x.
    ya = @. pp(xa)
return makeCubicPP(xa,ya,FIRST_DERIVATIVE, evaluateDerivative(pp,xa[1]),FIRST_DERIVATIVE,evaluateDerivative(pp,xa[end]),C2())
end

function insertKnots(pp::FastPP{3,T,TX}, za::AbstractArray{TX})::FastPP{3,T,TX} where {N,T,TX}
newPP = PP(N,T,TX,length(pp)+length(za))
newPP.a[ppIndex]=pp.a[ppIndex]
newPP.b[ppIndex]=pp.b[ppIndex]
newPP.c[ppIndex,:] = pp.c[ppIndex,:]

for (i,z) in enumerate(za)
    while (ppIndex < length(self.x) && (self.x[ppIndex] < z)) #Si[ppIndex]<=z<Si[ppIndex+1]  
        ppIndex += 1
        newIndex == ppIndex+i-1
        newPP.a[newIndex]=pp.a[ppIndex]
        newPP.b[newIndex]=pp.b[ppIndex]
        newPP.c[newIndex,:] = pp.c[ppIndex,:]
    end
    ppIndex -= 1
    ppIndex = min(max(ppIndex, 1), length(self.x) - 1)
    newIndex == ppIndex+i-1
    newPP.a[newIndex] = evaluatePiece(pp, ppIndex, z)
    newPP.b[newIndex] = evaluateDerivativePiece(pp, ppIndex, z)
    newPP.c[newIndex,1] = evaluateSecondDerivativePiece(pp, ppIndex, z)/2
    newPP.c[newIndex,2] = evaluateThirdDerivativePiece(pp, ppIndex, z)/6
end
return newPP
end