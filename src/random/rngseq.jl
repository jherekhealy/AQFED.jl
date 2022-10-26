import Random: rand, rand!, randn!, MersenneTwister
import RandomNumbers: AbstractRNG
import AQFED.Math: norminvAS241, norminvBSMS
#import AQFED.Random: randz

export AbstractRNGSeq, ZRNGSeq, next!, nextn!, skipTo


mutable struct AbstractRNGSeq{T,N} <: AbstractSeq{N} #where {T <: AbstractRNG}
    rng::T
    startIndex::Int
    currentIndex::Int
end

AbstractRNGSeq(rng::T, ndims::Int) where {T} = AbstractRNGSeq{T,ndims}(rng, 0, 0)

@inline function next!(s::AbstractRNGSeq, points::AbstractVector{<:AbstractFloat})
    rand!(s.rng, points)
    s.currentIndex += length(points)
end

@inline function nextn!(s::AbstractRNGSeq, points::AbstractVector{<:AbstractFloat})
    rand!(s.rng, points)
    @. points = norminv(points) #deterministic number of random numbers
    s.currentIndex += length(points)
end

@inline function nextn!(s::AbstractRNGSeq{T, N},  dim::Int, points::AbstractVector{<:AbstractFloat}) where {T,N}
    rand!(s.rng, points)
    @. points = norminv(points) #deterministic number of random numbers
     # @. points = norminvBSMS(points)
    s.currentIndex += length(points)

end
@inline function next!(s::AbstractRNGSeq{T, N}, dim::Int, points::AbstractVector{<:AbstractFloat}) where {T,N}
    rand!(s.rng, points)
    s.currentIndex += length(points)
end

function skipTo(s::AbstractRNGSeq, dim::Int, startIndex::Int)
    #the following allows to not skip at every dim. Breaks BB use in some ways,
    #but is supported by most RNGs. Counter-based RNGs may have a specific impl.
    #presupposes that RNG is not used
    n = ndims(s)
    if s.startIndex != startIndex
        skip(s.rng, startIndex*n - s.currentIndex)
        s.currentIndex = startIndex*n
        s.startIndex = startIndex
    end
end





mutable struct ZRNGSeq{T,N} <: AbstractSeq{N} #where {T <: AbstractRNG}
    rng::T
    startIndex::Int
end

ZRNGSeq(rng::T, ndims::Int) where {T} = ZRNGSeq{T,ndims}(rng, 0)

@inline function next!(s::ZRNGSeq, points::AbstractVector{<:AbstractFloat})
    rand!(s.rng, points)
end

@inline function nextn!(s::ZRNGSeq, points::AbstractVector{<:AbstractFloat})
    randn!(s.rng, points)
end

@inline function nextn!(s::ZRNGSeq,  dim::Int, points::AbstractVector{<:AbstractFloat})
    randn!(s.rng, points)
end

@inline function next!(s::ZRNGSeq, dim::Int, points::AbstractVector{<:AbstractFloat})
    rand!(s.rng, points)
end

function skipTo(s::ZRNGSeq, dim::Int, startIndex::Int)
    #the following allows to not skip at every dim. Breaks BB use in some ways,
    #but is supported by most RNGs. Counter-based RNGs may have a specific impl.
    #presupposes that RNG is not used
    n = ndims(s)
    if s.startIndex != startIndex
        skipTo(s.rng, startIndex*n)
        s.startIndex = startIndex
    end
end
