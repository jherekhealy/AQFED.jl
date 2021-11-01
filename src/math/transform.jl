export Bijection, IdentityTransformation, ClosedTransformation, SquareMinTransformation, ExpMinTransformation, inv

abstract type Bijection end

Base.broadcastable(p::Bijection) = Ref(p)

struct IdentityTransformation{T} <: Bijection end
(f::IdentityTransformation{T})(x::T) where {T} = x
inv(f::IdentityTransformation{T}, x::T) where {T} = x

struct ClosedTransformation{T} <: Bijection
    minValue::T
    maxValue::T
end

function (f::ClosedTransformation{T})(x) where {T}
    return (f.minValue + f.maxValue) / 2 + (f.minValue - f.maxValue) / 2 * cos(x)
end

function inv(f::ClosedTransformation{T}, y) where {T}
    x = (y - f.minValue + (f.minValue - f.maxValue) / 2) / (f.minValue - f.maxValue) * 2
    return acos(x)
end


struct SquareMinTransformation{T} <: Bijection
    minValue::T
end

function (f::SquareMinTransformation{T})(x) where {T}
    return x^2 + f.minValue
end

function inv(f::SquareMinTransformation{T}, y) where {T}
    return sqrt(y - f.minValue)
end

struct ExpMinTransformation{T} <: Bijection
    minValue::T
end

function (f::ExpMinTransformation{T})(x) where {T}
    return exp(x) + f.minValue
end

function inv(f::ExpMinTransformation{T}, y) where {T}
    return log(y - f.minValue)
end



struct MQMinTransformation{T} <: Bijection
    minValue::T
    r::T
end

function (f::MQMinTransformation{T})(x) where {T}
    return sqrt(1+ (f.r*x)^2) + f.minValue - 1
end

function inv(f::MQMinTransformation{T}, y) where {T}
    return sqrt((y + 1 - f.minValue)^2 - 1)/f.r
end
