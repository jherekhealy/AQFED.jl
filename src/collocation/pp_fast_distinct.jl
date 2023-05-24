using BSplines
import PPInterpolation:bspp


struct FastPP{N,T,TX}
    a::Vector{T}
    b::Vector{T}
    c::Matrix{T}
    x::Vector{TX}
    FastPP(N::Int,a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractMatrix{T}, x::AbstractArray{TX}) where {T<:Real,TX} =
        new{N,T,TX}(a, b, c, x)
end

const CubicPP{T,TX} = FastPP{3,T,TX} 
const QuadraticPP{T,TX} = FastPP{2,T,TX} 


Base.length(p::FastPP) = Base.length(p.x)
Base.size(p::FastPP) = Base.size(p.x)
Base.broadcastable(p::FastPP) = Ref(p)


function evaluate(self::FastPP{N,T,TX}, z::TZ) where {N,T,TX,TZ}
    if z <= self.x[1]
        return self.b[1] * (z - self.x[1]) + self.a[1]
    elseif z >= self.x[end]
        return self.b[end] * (z - self.x[end]) + self.a[end]
    end
    i = searchsortedfirst(self.x, z)  # x[i-1]<z<=x[i]
    if z != self.x[i] && i > 1
        i -= 1
    end
    return evaluatePiece(self, i, z)
end


@inline function evaluatePiece(self::FastPP{3,T,TX}, i::Int, z::TZ) where {T,TX,TZ}
    h = z - self.x[i]
    return self.a[i] + h * (self.b[i] + h * (self.c[i,1] + h * (self.c[i,2])))
end

@inline function evaluatePiece(self::FastPP{2,T,TX}, i::Int, z::TZ) where {T,TX,TZ}
    h = z - self.x[i]
    return self.a[i] + h * (self.b[i] + h * (self.c[i,1] ))
end


function evaluateDerivative(self::FastPP{N,T,TX}, z::TZ) where {N,T,TX,TZ}
    if z <= self.x[1]
        return self.b[1]
    elseif z >= self.x[end]
        rightSlope = self.b[end]
        return rightSlope
    end
    i = searchsortedfirst(self.x, z)  # x[i-1]<z<=x[i]
    if z != self.x[i] && i > 1
        i -= 1
    end
    return evaluateDerivativePiece(self,i,z)
end

@inline function evaluateDerivativePiece(self::FastPP{3,T,TX}, i::Int, z::TZ) where {T,TX,TZ}
    h = z - self.x[i]
    return  self.b[i] + h * (2*self.c[i, 1] + h * (3*self.c[i, 2]))
end

@inline function evaluateDerivativePiece(self::FastPP{2,T,TX}, i::Int, z::TZ) where {T,TX,TZ}
    h = z - self.x[i]
    return self.b[i] + 2h * self.c[i, 1]
end



function Base.convert(::Type{FastPP{2,T,TX}}, spl::BSplines.Spline) where {T,TX}
    t = BSplines.knots(spl.basis)
    n = length(spl.basis) - 1
    a = zeros(T, n)
    b = zeros(T, n)
    c = zeros(T,( n - 1,1))
    x = spl.basis.breakpoints
    α = spl.coeffs
    for i = 1:n
        a[i] = (t[i+2] - t[i+1]) / (t[i+3] - t[i+1]) * (α[i+1] - α[i]) + α[i]
        b[i] = 2 * (α[i+1] - α[i]) / (t[i+3] - t[i+1])
    end
    for i = 1:n-1
        c[i,1] = ((α[i+2] - α[i+1]) / (t[i+4] - t[i+2]) + (α[i] - α[i+1]) / (t[i+3] - t[i+1])) / (t[i+3] - t[i+2])
    end
    return FastPP(2,a, b, c, x)
end

function Base.convert(::Type{BSplines.Spline}, pp::FastPP{2,T,TX}) where {T,TX}
    basis = BSplines.BSplineBasis(3, pp.x)
    t = BSplines.knots(basis)
    n = length(pp.x)
    α = zeros(T, n + 1)
    for i = 1:n
        α[i] = pp.a[i] - pp.b[i] / 2 * (t[i+2] - t[i+1])
    end
    α[n+1] = pp.a[n]

    return BSplines.Spline(basis, α)
end

function Base.convert(::Type{BSplines.Spline}, pp::FastPP{3,T,TX}) where {T,TX}
    basis = BSplines.BSplineBasis(4, pp.x)
    t = BSplines.knots(basis)
    n = length(pp.x)
    α = zeros(T, n + 2)
    α[1] = pp.a[1]
    #spl.(t[3:end-2]) + spl.(t[3:end-2],Derivative(1))/3 .* (t[4:end-1]-t[3:end-2]-(t[3:end-2]-t[2:end-3])) - spl.(t[3:end-2],Derivative(2))/6 .* (t[4:end-1]-t[3:end-2]).*(t[3:end-2]-t[2:end-3])
    for i = 1:n-1
        dt1 = t[i+3] - t[i+2]
        dt2 = t[i+4] - t[i+3]
        α[i+1] = pp.a[i] + pp.b[i] / 3 * (dt2 - dt1) - pp.c[i,1] / 3 * dt2 * dt1
    end
    α[n+1] = pp.a[n] + pp.b[n] / 3 * (t[n+4] - 2 * t[n+3] + t[n+2])
    α[n+2] = pp.a[n]
    return BSplines.Spline(basis, α)
end


function Base.convert(::Type{FastPP{3,T,TX}}, spl::BSplines.Spline) where {T,TX}
    t = BSplines.knots(spl.basis)
    α = spl.coeffs
    n = length(α)
    breakA, coef, l = bspp(t, α, n, 4)
    x = breakA
    a = coef[1, :]
    b = coef[2, :]
    c = coef[3:4, 1:l]'
    dx = (x[l+1] - x[l])
    a[l+1] = a[l] + dx * (b[l] + dx * (c[l,1] + dx * (c[l,2])))
    b[l+1] = b[l] + dx * (2 * c[l,1] + dx * 3 * c[l,2])
    return FastPP(3,a, b, c,x)
end
