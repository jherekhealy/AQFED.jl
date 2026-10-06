#load after ssobol.jl
import RandomNumbers: AbstractRNG
import Random: rand!
import AQFED.Math: norminv
export DigitalSobolSeq, ModuloSobolSeq

mutable struct DigitalSobolSeq{N} <: AbstractSeq{N}
    delegate::ScrambledSobolSeq{N, NoScrambling}
    rng::AbstractRNG{UInt32}
    upoint::Array{UInt32, 1}
end


function DigitalSobolSeq(dimension::Int, n::Int, rng::AbstractRNG{UInt32}, direction_source::DirectionSource=joe_kuo)
    delegate = ScrambledSobolSeq(dimension, n, NoScrambling(), direction_source)
    upoint = Vector{UInt32}(undef, dimension)
    rand!(rng, upoint)
    #delegate.counter = 0 #reset counter to start
    return DigitalSobolSeq{dimension}(delegate, rng, upoint)
end

@inline normalize(s::DigitalSobolSeq, x::UInt32) = ldexp(Float64(x)+0.5, Int32(-32))

@inline function next(s::DigitalSobolSeq{N}, ::Type{UInt32}) where {N}
    points = copy(next(s.delegate, UInt32))
    sx = s.delegate.x
    @inbounds for j = 1:ndims(s)
        points[j] = (sx[j]<< (32-s.delegate.l)) ⊻ s.upoint[j]
    end
    return points
end

#next vector at counter containing all dimensions
@inline function next!(s::DigitalSobolSeq, points::AbstractVector{<:AbstractFloat})
    next(s.delegate, UInt32)
    sx = s.delegate.x
    @inbounds for j = 1:ndims(s)
        points[j] = normalize(s, (sx[j]<< (32-s.delegate.l)) ⊻ s.upoint[j])
    end
    return points
end

#next vector for a given dimension (vertical) from counter to counter + length(points)
@inline function next!(s::DigitalSobolSeq, dim::Int, points::AbstractVector{<:AbstractFloat})
    j = dim
    sx = s.delegate.x
    sv = s.delegate.v
    @inbounds for i = 1:length(points)
        if s.delegate.counter != 0
            c = ffz(s.delegate.counter)
            sx[j] ⊻= sv[j, c]
        end
        points[i] = normalize(s, (sx[j]<< (32-s.delegate.l)) ⊻ s.upoint[j])
        s.delegate.counter += one(s.delegate.counter)
    end
    return points
end

@inline skipTo!(s::DigitalSobolSeq, n::Int) = skipTo!(s.delegate, n)
@inline skipTo!(s::DigitalSobolSeq, dim::Int, n::Int, size::Int=0) = skipTo!(s.delegate, dim, n, size)




mutable struct ModuloSobolSeq{N} <: AbstractSeq{N}
    delegate::ScrambledSobolSeq{N, NoScrambling}
    rng::AbstractRNG
    upoint::Array{Float64, 1}
end


function ModuloSobolSeq(dimension::Int, n::Int, rng::AbstractRNG, direction_source::DirectionSource=joe_kuo)
    delegate = ScrambledSobolSeq(dimension, n, NoScrambling(), direction_source)
    upoint = Vector{Float64}(undef, dimension)
    rand!(rng, upoint)
    return ModuloSobolSeq(delegate, rng, upoint)
end

#next vector for a given dimension (vertical) from counter to counter + length(points)
@inline function next!(s::ModuloSobolSeq, dim::Int, points::AbstractVector{<:AbstractFloat})
    next!(s.delegate, dim, points)
    uj = s.upoint[j]
    @. points = ((points + uj) % 1) + 0.5/2^32
    return points
end


@inline function next!(s::ModuloSobolSeq, points::P) where { P<:Union{AbstractVector{<:AbstractFloat}, AbstractMatrix{<:AbstractFloat}}}
    next!(s.delegate, points)
     @inbounds for j = eachindex(points)
         points[j] = ((points[j] + s.upoint[j]) % 1) + 0.5/2^32
     end
    return points
end


@inline skipTo!(s::ModuloSobolSeq, n::Int) = skipTo!(s.delegate, n)
@inline skipTo!(s::ModuloSobolSeq, dim::Int, n::Int, size::Int=0) = skipTo!(s.delegate, dim, n, size)

@inline function nextn!(s::Union{DigitalSobolSeq,ModuloSobolSeq}, points::AbstractVector{<:AbstractFloat})
    next!(s, points)
    @. points = norminv(points)
    points
end

@inline function nextn!(s::Union{DigitalSobolSeq,ModuloSobolSeq}, dim::Int, points::AbstractVector{<:AbstractFloat})
    next!(s, dim, points)
    @. points = norminv(points)
    points
end
