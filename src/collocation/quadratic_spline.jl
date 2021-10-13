using BSplines
struct QuadraticPP{T<:Number,U<:Real}
    a::Vector{T}
    b::Vector{T}
    c::Vector{T}
    x::Vector{U}
    #logx::Vector{T}
    QuadraticPP(T, U, n::Int) = new{T,U}(zeros(T, n), zeros(T, n), zeros(T, n - 1), zeros(U, n))
    QuadraticPP(a::Vector{T}, b::Vector{T}, c::Vector{T}, x::Vector{U}) where {T<:Real,U<:Real} = new{T,U}(a, b, c, x)
end

(p::QuadraticPP{T})(x::T) where {T} = evaluate(p, x)
Base.length(p::QuadraticPP) = length(pp.a)

function evaluate(self::QuadraticPP, z::T) where {T}
    if z <= self.x[1]
        return self.b[1] * (z - self.x[1]) + self.a[1]
    elseif z >= self.x[end]
        rightSlope = self.b[end]
        return rightSlope * (z - self.x[end]) + self.a[end]
    end
    i = searchsortedfirst(self.x, z)  # x[i-1]<z<=x[i]
    if z == self.x[i]
        return self.a[i]
    end
    if i > 1
        i -= 1
    end
    h = z - self.x[i]
    return self.a[i] + h * (self.b[i] + h * (self.c[i]))
end

function evaluateDerivative(self::QuadraticPP, z::T) where {T}
    if z <= self.x[1]
        return self.b[1]
    elseif z >= self.x[end]
        rightSlope = self.b[end]
        return rightSlope
    end
    i = searchsortedfirst(self.x, z)  # x[i-1]<z<=x[i]
    if z == self.x[i]
        return self.a[i]
    end
    if i > 1
        i -= 1
    end
    h = z - self.x[i]
    return self.b[i] + 2h * self.c[i]
end
function convert(::Type{QuadraticPP}, spl::BSplines.Spline)
    t = BSplines.knots(spl.basis)
    n = length(spl.basis) - 1
    T = typeof(spl.coeffs[1])
    a = zeros(T, n)
    b = zeros(T, n)
    c = zeros(T, n - 1)
    x = spl.basis.breakpoints
    α = spl.coeffs
    for i = 1:n
        a[i] = (t[i+2] - t[i+1]) / (t[i+3] - t[i+1]) * (α[i+1] - α[i]) + α[i]
        b[i] = 2 * (α[i+1] - α[i]) / (t[i+3] - t[i+1])
    end
    for i = 1:n-1
        c[i] = ((α[i+2] - α[i+1]) / (t[i+4] - t[i+2]) + (α[i] - α[i+1]) / (t[i+3] - t[i+1])) / (t[i+3] - t[i+2])
    end
    return QuadraticPP(a, b, c, x)
end

function convert(::Type{BSplines.Spline}, pp::QuadraticPP)
    basis = BSplines.BSplineBasis(3, pp.x)
    t = BSplines.knots(basis)
    n = length(pp.x)
    T = typeof(pp.x[1])
    α = zeros(T, n + 1)
    for i = 1:n
        α[i] = pp.a[i] - pp.b[i] / 2 * (t[i+2] - t[i+1])
    end
    α[n+1] = pp.b[n] / 2 * (t[n+3] - t[n+1]) + α[n]
    return BSplines.Spline(basis, α)
end
