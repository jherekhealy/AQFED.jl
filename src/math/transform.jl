export Bijection, IdentityTransformation, ClosedTransformation, SquareMinTransformation, ExpMinTransformation, inv
export LogTransformation
abstract type Bijection end

Base.broadcastable(p::Bijection) = Ref(p)

struct IdentityTransformation{T} <: Bijection
end

(f::IdentityTransformation{T})(x) where {T} = x
function inv(f::IdentityTransformation{T}, y) where {T} 
    return y
end

struct LogTransformation <: Bijection
end

(f::LogTransformation)(x) = log(x)
function inv(f::LogTransformation, y) 
    return exp(y)
end

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

struct LogisticTransformation{T} <: Bijection
    minValue::T
    maxValue::T
end

function (f::LogisticTransformation{T})(x) where {T}
    return f.minValue + (f.maxValue-f.minValue)/(one(T)+exp(-x))
end

function inv(f::LogisticTransformation{T}, y) where {T}
    # y = 1/1+emx == 1/y = 1 + emx == -x = log(1/y-y/y)  x = log(y/(1-y))
    y0 = (y - f.minValue)/(f.maxValue-f.minValue)
    return log(y0/(one(T)-y0))
end


struct TanhTransformation{T} <: Bijection
    minValue::T
    maxValue::T
end

function (f::TanhTransformation{T})(x) where {T}
    return (f.minValue + f.maxValue) / 2 + (f.maxValue-f.minValue)/2*tanh(x)
end

function inv(f::TanhTransformation{T}, y) where {T}
    y0 = 2(y - (f.minValue + f.maxValue) / 2)/(f.maxValue-f.minValue)
    return atanh(y0)
end

struct AtanTransformation{T} <: Bijection
    minValue::T
    maxValue::T
end

function (f::AtanTransformation{T})(x) where {T}
    return (f.minValue + f.maxValue) / 2 + (f.maxValue-f.minValue)/pi*atan(x*pi/2)
end

function inv(f::AtanTransformation{T}, y) where {T}
    y0 = (y - (f.minValue + f.maxValue) / 2)/(f.maxValue-f.minValue)*pi
    return tan(y0)*2/pi
end


struct AlgebraicTransformation{T} <: Bijection
    minValue::T
    maxValue::T
end

function (f::AlgebraicTransformation{T})(x) where {T}
    return  (f.minValue + f.maxValue) / 2 + (f.maxValue-f.minValue)/2*x/sqrt(one(T)+x^2)
end

function inv(f::AlgebraicTransformation{T}, y) where {T}
    y0 = 2(y - (f.minValue + f.maxValue) / 2)/(f.maxValue-f.minValue)
    # y = x / sqrt(1+X^2) === y^2 (1+x^2)= x^2 === x^2 (1-y^2)= y^2
    return y0/sqrt(one(T)-y0^2)
end