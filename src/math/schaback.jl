#Schaback convexity preserving rational spline interpolation
using LinearAlgebra
using LeastSquaresOptim
import AQFED.Math: normpdf, inv, ClosedTransformation

export SchabackRationalSpline, makeConvexSchabackRationalSpline, FirstDerivativeBoundary, SecondDerivativeBoundary, FitResult,  fitConvexSchabackRationalSpline

struct SchabackRationalSpline{U<:Number,T<:Real}
    x::Vector{U}
    y::Vector{T}
    h::Vector{T}
    m::Vector{T}
    SchabackRationalSpline(U, T, n::Int) = new{U,T}(zeros(U, n), zeros(T, n), zeros(T, n), zeros(T, n))
    SchabackRationalSpline(x::Vector{U}, y::Vector{T}, h::Vector{T}, m::Vector{T}) where {U<:Real,T<:Real} =
        new{U,T}(x, y, h, m)
end

abstract type PPBoundary end
struct FirstDerivativeBoundary{T} <: PPBoundary
    value::T
end
struct SecondDerivativeBoundary{T} <: PPBoundary
    value::T
end

function initLeftBoundary(b::FirstDerivativeBoundary, delta, d, h, m)
    d[1] = delta[1] - b.value
    m[1] = sqrt(2 * d[1] / (h[1] * m[2])) * sign(d[1])
end

function initLeftBoundary(b::SecondDerivativeBoundary, delta, d, h, m)
    d[1] = b.value
    m[1] = d[1]
end

function initRightBoundary(b::FirstDerivativeBoundary, delta, d, h, m)
    d[end] = b.value - delta[end]
    m[end] = sqrt(2 * d[end] / (h[end] * m[end-1])) * sign(d[end])
end

function initRightBoundary(b::SecondDerivativeBoundary, delta, d, h, m)
    d[end] = b.value
    m[end] = d[end]
end


abstract type SecondDerivativeGuess end
struct Schaback{T} <: SecondDerivativeGuess end
struct Normal{T} <: SecondDerivativeGuess
    μ::T
    σ::T
end

function initSecondDerivative(guessType::Schaback, x, d, h, m)
    for j = 2:length(m)-1
        m[j] = (2 * d[j]) / (h[j] + h[j-1])
    end
end

function initSecondDerivative(guessType::Normal, x, d, h, m)
    for j = 2:length(m)-1
        m[j] = normpdf((x[j] - guessType.μ) / guessType.σ) / guessType.σ
    end
end

struct FitResult
    measure
    iterations
    x
    fx
    details
end
#Newton optimization
function fitConvexSchabackRationalSpline(
    x0::AbstractArray{U},
    y0::AbstractArray{T},
    weights::AbstractArray{T},
    leftBoundary::PPBoundary,
    rightBoundary::PPBoundary;
    penalty = 0.0,
    guessType::SecondDerivativeGuess = Schaback{Float64}(),
)::Tuple{SchabackRationalSpline{U,T}, FitResult} where {T,U}
    x = copy(x0)
    y = copy(y0)
    h = x[2:end] - x[1:end-1]
    delta = (y[2:end] - y[1:end-1]) ./ h
    d = zeros(T, length(x))
    m = zeros(T, length(x))

    n = length(delta)
    leftBoundaryOverride = typeof(leftBoundary) == SecondDerivativeBoundary{T} ? leftBoundary : SecondDerivativeBoundary((eps(T)))
    rightBoundaryOverride = typeof(rightBoundary) == SecondDerivativeBoundary{T} ? rightBoundary : SecondDerivativeBoundary((eps(T)))
    initLeftBoundary(leftBoundaryOverride, delta, d, h, m)
    initRightBoundary(rightBoundaryOverride, delta, d, h, m)

    sign1 = sign(d[1])
    for i = 2:n
        d[i] = delta[i] - delta[i-1] #corresponds to D[j]
        if sign(d[i]) != sign1
            d[i] = sign1 * eps(one(d[1]))
        end
    end
    #starting value
    initSecondDerivative(guessType, x, d, h, m)
    @. m = max(m, sqrt(eps(T))) #a too small estimate may create instabilities
    @. m = sign(m) * (sign(m) * m^(1 / 3))
    #tridiagonal system
    lower = zeros(T, n)
    middle = zeros(T, n + 1)
    upper = zeros(T, n)
    b = zeros(T, n + 1)
    for j = 2:n
        lower[j-1] = h[j] / h[j-1]
        upper[j] = one(T)
        middle[j] = -(upper[j] + lower[j-1])
    end
    #boundary for linear system can only be second derivative
    middle[1] = one(T)
    upper[1] = zero(T)
    middle[end] = one(T)
    lower[end] = zero(T)
    tri = Tridiagonal(lower, middle, upper)
    transform = ExpMinTransformation(eps(T)^(1/3))
    iter = 0
    #m longer if first derivative boundary ?!?
    function obj!(fvec, x::AbstractArray{W}) where {W}
        #c0 and cn are exact, for first der boundary, use eqn 5.144 5.145
        #to determine line i=1 and i=n-1 => u appears on line 2
        ml = zeros(W,n+1)
        b = zeros(W, n + 1)
        @. ml[2:end-1] = transform(x)
        ml[1] = m[1]
        ml[end] = m[end]
        b[1] = y[1]
        b[n+1] = y[n+1]
        for j = 2:n
            b[j] = h[j]*(h[j-1] * ml[j-1] + h[j] * ml[j+1] ) * ml[j]^2  / 2
        end
        yHat = tri \ b
        @. fvec[1:n-1] = @view(weights[2:end-1]) * (@view(yHat[2:end-1]) - @view(y[2:end-1])) #end points are exact
        if penalty > 0
            #Strain on M
            for j = 1:n-1
                  s2,s1,s0 = ml[j+2]^3 , ml[j+1]^3 ,  ml[j]^3
                #  g2 = 2 * ((s2 - s1) / h[j+1] - (s1 - s0) / h[j]) / (h[j+1] + h[j])
                g2 = ((log(s2) - log(s1)) / h[j+1] - (log(s1) - log(s0)) / h[j]) / (h[j+1] + h[j])
                fvec[j+n-1] = g2 * sqrt(h[j])  * penalty #strain is sometimes unstable, second derivative is better
            end
        end
        iter += 1
        fvec
    end
    c0 = @. inv(transform, @view(m[2:end-1]))
    outlen = n - 1
    if penalty > 0
        outlen += n - 1
    end
    #optimization not great for very small option prices due to machine epsilon issues in Schaback eval combined with large inverse vega weights
    fit = optimize!(
        LeastSquaresProblem(
            x = c0,
            f! = obj!,
            autodiff = :forward,
            # g! = jac!, #useful to debug issue with ForwardDiff NaNs
            output_length = outlen,
        ),
        LevenbergMarquardt();
        iterations = 1024*4,
        x_tol = T(1e-3),
        f_tol = eps(T),
        g_tol = eps(T)
    )
    c0 = fit.minimizer
    fvec = zeros(T, outlen)
    obj!(fvec, c0)
    @. y[2:end-1] += @view(fvec[1:n-1]) / @view(weights[2:end-1])
    @. m[2:end-1] = transform(c0)
    #println(iter, " Schaback fit ", fit, fvec) #obj(fit.x))  #fit.f
    return SchabackRationalSpline(x, y, h, m), FitResult(fit.ssr, iter, c0, fvec, fit)
end

#Fixed point iteration
function makeConvexSchabackRationalSpline(
    x0::AbstractArray{U},
    y0::AbstractArray{T},
    leftBoundary::PPBoundary,
    rightBoundary::PPBoundary;
    iterations::Int = 8,
) where {T,U}
    x = copy(x0)
    y = copy(y0)
    h = x[2:end] - x[1:end-1]
    delta = (y[2:end] - y[1:end-1]) ./ h
    d = zeros(T, length(x))
    m = zeros(T, length(x))

    n = length(delta)
    initLeftBoundary(leftBoundary, delta, d, h, m)
    initRightBoundary(rightBoundary, delta, d, h, m)

    sign1 = sign(d[1])
    for i = 2:n
        d[i] = delta[i] - delta[i-1] #corresponds to D[j]
        if sign(d[i]) != sign1
            d[i] = sign1 * eps(one(d[1]))
        end
    end
    #starting value
    for j = 2:n
        m[j] = (2 * d[j]) / (h[j] + h[j-1])
    end
    @. m = sign(m) * (sign(m) * m^(1 / 3))
    #Gauss Seidel iterations
    for loop = 1:iterations
        initLeftBoundary(leftBoundary, delta, d, h, m)
        initRightBoundary(rightBoundary, delta, d, h, m)
        for j = 2:n
            m[j] = sqrt(2 * d[j] / (h[j-1] * m[j-1] + h[j] * m[j+1])) * sign(d[j])
        end
    end
    return SchabackRationalSpline(x, y, h, m)
end

(p::SchabackRationalSpline{U,T})(x::U) where {U,T} = evaluate(p, x)
Base.length(p::SchabackRationalSpline) = length(p.x)
Base.broadcastable(p::SchabackRationalSpline) = Ref(p)

function evaluateRational(s::SchabackRationalSpline{U,T}, index::Int, z::U) where {U,T}
    hi = (s.x[index+1] - s.x[index])
    h = (s.x[index+1] - z)
    m11 = s.m[index+1]
    if abs(m11) <= zero(U)
        return s.y[index+1] - h * ((s.y[index+1] - s.y[index]) / hi)
    end
    m12 = m11 * m11
    m13 = m12 * m11
    mr = (s.m[index] - m11) / (hi * s.m[index])
    return s.y[index+1] - h * ((s.y[index+1] - s.y[index]) / hi + hi / 2 * s.m[index] * m12) +
           (m13 * h^2) / (2 * (1 - h * mr))
end

function evaluateRationalDerivative(s::SchabackRationalSpline{U,T}, index::Int, z::U) where {U,T}
    hi = (s.x[index+1] - s.x[index])
    h = (s.x[index+1] - z)
    m11 = s.m[index+1]
    if abs(m11) <= zero(U)
        return (s.y[index+1] - s.y[index]) / hi
    end
    m12 = m11 * m11
    m13 = m12 * m11
    mr = (s.m[index] - m11)
    mhi = (hi * s.m[index])
    return (s.y[index+1] - s.y[index]) / hi +
           hi / 2 * s.m[index] * m12 +
           m13 / 2 * h * mhi * (-2 - mr * h / (mhi - mr * h)) / (mhi - mr * h)
end


function evaluateRationalSecondDerivative(s::SchabackRationalSpline{U,T}, index::Int, z::U) where {U,T}
    hi = (s.x[index+1] - s.x[index])
    h = (s.x[index+1] - z)
    m11 = s.m[index+1]
    if abs(m11) <= zero(U)
        return zero(T)
    end
    mr = (s.m[index] - m11)
    mhi = (hi * s.m[index])
    return m11^3 * ( mhi + mr * h * mhi * (2 +  mr * h / (mhi - mr * h)) / (mhi - mr * h)) / (mhi - mr * h)
end

function locateLowerIndex(self::SchabackRationalSpline{T,U}, z::T) where {T,U}
    i = searchsortedfirst(self.x, z)  # x[i-1]<z<=x[i]
    if i >= length(self.x)
        return length(self.x) - 1
    elseif i > 1
        return i - 1
    end
end

function evaluate(self::SchabackRationalSpline{T,U}, z::T) where {T,U}
    i = locateLowerIndex(self, z)
    return evaluateRational(self, i, z)
end

function evaluateDerivative(self::SchabackRationalSpline{T,U}, z::T) where {T,U}
    i = locateLowerIndex(self, z)
    return evaluateRationalDerivative(self, i, z)
end


function evaluateSecondDerivative(self::SchabackRationalSpline{T,U}, z::T) where {T,U}
    i = locateLowerIndex(self, z)
    return evaluateRationalSecondDerivative(self, i, z)
end
