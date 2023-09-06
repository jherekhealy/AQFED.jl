export ChebyshevPoly, makeInterpolation,evaluate
export chebnodevalues!, chebcoeff!, chebinterp,cheb2nodevalues!,cheb2coeff!,cheb2interp,chebnodes,cheb2nodes

struct ChebyshevPoly{T,Kind}
    nodes::Vector{T}
    coeffs::Vector{T}
    values::Vector{T}
end

ChebyshevPoly{T,1}(n::Int) where {T} = ChebyshevPoly{T,1}(chebnodes(T,n),zeros(T,n),zeros(T,n))
ChebyshevPoly{T,2}(n::Int) where {T} = ChebyshevPoly{T,2}(cheb2nodes(T,n),zeros(T,n),zeros(T,n))
ChebyshevPoly{T}(n::Int,kind::Int) where {T} = ChebyshevPoly{T,kind}(n)
ChebyshevPoly(n::Int) = ChebyshevPoly{Float64}(n,1)
ChebyshevPoly(n::Int,kind::Int) = ChebyshevPoly{Float64}(n,kind)

(p::ChebyshevPoly{T,K})(x) where {T,K} = evaluate(p, x)
Base.length(p::ChebyshevPoly) = length(p.x)
Base.broadcastable(p::ChebyshevPoly) = Ref(p)

function makeInterpolation(self::ChebyshevPoly{T,1}, f) where {T}
    fValues = f.(self.nodes)
    chebcoeff!(self.coeffs, fValues)
end

function evaluate(self::ChebyshevPoly{T,1},x::TZ) where {T,TZ}
    return chebinterp(self.coeffs, x)
end

function makeInterpolation(self::ChebyshevPoly{T,2}, f) where {T}
    fValues = f.(self.nodes)
    cheb2coeff!(self.coeffs, fValues)
end

function evaluate(self::ChebyshevPoly{T,2},x::TZ) where {T,TZ}
    return cheb2interp(self.coeffs, x)
end

@inline function chebinterp(c::AbstractArray{T}, x::TZ) where {T,TZ}
    nC = Base.length(c)
    bn=zero(T)
    bn_1 = zero(T)
    bn_2 = zero(T)
    for i = nC:-1:2
        bn = c[i] + 2x*bn_1 - bn_2
        bn_2 = bn_1    
        bn_1 = bn
    end    
    bn = 2c[1] + 2x*bn_1 - bn_2
    return (bn - bn_2)/2
end

chebnodes(T, n::Int) = @. (cos(((2:2:2n) - 1) * T(pi) / 2n))
function chebnodevalues!(fValues::AbstractArray{T},f) where {T}
    nC = Base.length(fValues)
    @. fValues = f(chebnodes(T,nC))
end
   

function chebcoeff!(coeff::AbstractArray{T}, fValues::AbstractArray{TV}) where {T,TV}
    nC = Base.length(fValues)
    coeff[1] = sum(fValues)/nC
    for sk = 1:nC-1
        sumi = zero(T)
        @inbounds @simd for i = 1:nC
            sumi += fValues[i] * cos((2i - 1) * sk * pi / 2nC)
        end
        coeff[sk+1] = 2sumi / nC
    end
end


cheb2nodes(T, n::Int) = @. (cos(((1:n) - 1) * T(pi) / (n-1)))
function cheb2nodevalues!(fValues::AbstractArray{T},f) where {T}
    nC = Base.length(fValues)-1
    @. fValues = f(cheb2nodes(T,nC+1))
end
   

function cheb2coeff!(coeff::AbstractArray{T}, fValues::AbstractArray{TV}) where {T,TV}
    nC = Base.length(fValues)-1
    for sk = 0:nC
        @inbounds sumi = fValues[1] / 2
        @inbounds @simd for i = 2:nC
            sumi += fValues[i] * cos((i - 1) * sk * pi / nC)
        end
        @inbounds sumi += fValues[nC+1] / 2 * cos(sk * pi)
        coeff[sk+1] = 2 * sumi / nC
    end
end


@inline function cheb2interp(coeff::AbstractArray{T}, zck::TZ) where {T,TZ}
    b2 = 0.0
    nC = Base.length(coeff) - 1
    b1 = coeff[nC+1] / 2
    @inbounds @fastmath for sk22 = nC:-1:2
        bd = coeff[sk22] - b2
        b2 = b1
        b1 = 2 * zck * b1 + bd
    end
    b0 = coeff[1] + 2 * zck * b1 - b2
    qck = (b0 - b2) / 2
    qck
end