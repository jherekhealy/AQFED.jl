import Random: rand, rand!, randn!, MersenneTwister
import RandomNumbers: AbstractRNG
import AQFED.Math: norminvAS241, norminvBSMS
#import AQFED.Random: randz

export AbstractRNGSeq, ZRNGSeq, next!, nextn!, skipTo!


mutable struct AbstractRNGSeq{T,N} <: AbstractSeq{N} #where {T <: AbstractRNG}
    rng::T
    startIndex::Int
    currentIndex::Int
end

AbstractRNGSeq(rng::T, ndims::Int) where {T} = AbstractRNGSeq{T,ndims}(rng, 0, 0)

@inline function next!(s::AbstractRNGSeq, points::AbstractMatrix{<:Number})
    rand!(s.rng, points)
    s.currentIndex += length(points)
end

@inline function next!(s::AbstractRNGSeq, points::AbstractVector{<:Number})
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
@inline function next!(s::AbstractRNGSeq{T, N}, dim::Int, points::AbstractVector{<:Number}) where {T,N}
    rand!(s.rng, points)
    s.currentIndex += length(points)
end

function skipTo!(
    s::AbstractRNGSeq,
    dim::Int,
    startIndex::Int,
    npoints::Int,
)
    target =
        (startIndex) * ndims(s) * npoints + (dim - 1) * npoints

    try
        skipTo!(s.rng, target)
    catch
    end

    return s
end




function skipTo!(
    s::AbstractRNGSeq,
    startIndex::Int,
)
    n = ndims(s)
    target = startIndex * n

    if s.currentIndex != target
        try
            skipTo!(s.rng, target)
        catch
        end
        s.currentIndex = target
    end

    s.startIndex = startIndex

    return s
end


mutable struct ZRNGSeq{T,N} <: AbstractSeq{N} #where {T <: AbstractRNG}
    rng::T
    startIndex::Int
end

ZRNGSeq(rng::U, ndims::Int) where {U} = ZRNGSeq{U,ndims}(rng, 0)

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

function skipTo!(s::ZRNGSeq, dim::Int, startIndex::Int, npoints::Int=0)
    #the following allows to not skip at every dim. Breaks BB use in some ways,
    #but is supported by most RNGs. Counter-based RNGs may have a specific impl.
    #presupposes that RNG is not used
    n = ndims(s)
    if s.startIndex != startIndex
        skipTo!(s.rng, startIndex*n)
        s.startIndex = startIndex
    end
end

function skipTo!(s::ZRNGSeq, startIndex::Int)
    n = ndims(s)
    if s.startIndex != startIndex
        skipTo!(s.rng, startIndex*n)
        s.startIndex = startIndex
    end
end
