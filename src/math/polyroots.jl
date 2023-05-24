#QuadRootsReal returns the real roots of a*x^2+ b*x +c =0
export quadRootsReal,cubicRootsReal

function quadRootsReal(a::T, b::T, c::T)::AbstractArray{T} where {T}
    d = b^2 - 4a * c
    if iszero(d)# single root
        return [-b / (2a)]
    elseif d > zero(d)# two real roots
        d = b < zero(d) ? sqrt(d) - b : -sqrt(d) - b
        return [d / (2a), (2c) / d]
    else # two complex roots
        return []
    end
end

#finds the real roots of  x^3+ax^2+bx+c=0
function cubicRootsReal(a::T, b::T, c::T)::AbstractArray{T} where {T}
    q = a^2 - 3b
    q /= 9

    R = 2a^3 - 9a * b + 27c
    R /= 54
    R2 = R^2
    q3 = q^3
    if R2 < q3
        theta = acos(R / sqrt(q3))
        return [-2 * sqrt(q) * cos(theta / 3) - a / 3, -2 * sqrt(q) * cos((theta + 2 * π) / 3) - a / 3,
            -2 * sqrt(q) * cos((theta - 2 * π) / 3) - a / 3]
    else
        signumR = if R < zero(T)
            -one(T)
        else
            one(T)
        end
        A = -signumR * (abs(R) + sqrt(R2 - q3))^(1 / 3)
        B = if abs(A) > sqrt(eps(A))
            q / A
        else
            zero(T)
        end
        return [(A + B) - a / 3]
    end
end