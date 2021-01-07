export BBCache, transformByDim!
import AQFED.Random: AbstractSeq, skipTo, nextn! #for vectorized way


mutable struct Entry{K,V}
    key::K
    value::V
    previous::Union{Nothing,Entry{K,V}}
    next::Union{Nothing,Entry{K,V}}
end

#LRU cache implementation (not threadsafe)
mutable struct BBCache{K,V} <: AbstractDict{K,V}
    map::Dict{K,Entry{K,V}}
    head::Union{Nothing,Entry{K,V}}
    tail::Union{Nothing,Entry{K,V}}
    maxsize::Int
    function BBCache{K,V}(size::Int) where {K,V}
        new{K,V}(Dict{K,Entry{K,V}}(), nothing, nothing, size)
    end
end

Base.haskey(cache::BBCache{K,V}, key::K) where {K,V} = haskey(cache.map, key)

function Base.getindex(cache::BBCache{K,V}, key::K) where {K,V}
    entry = cache.map[key]
    removeNode(cache, entry)
    addFirst(cache, entry)
    return entry.value
end

function Base.get(cache::BBCache{K,V}, key::K, default::Union{Nothing,V}) where {K,V}
    if haskey(cache.map, key)
        return Base.getindex(cache, key)
    else
        return default
    end
end

function Base.setindex!(cache::BBCache{K,V}, value::V, key::K) where {K,V}
    if haskey(cache.map, key)
        entry = cache.map[key]
        entry.value = value
        removeNode(cache, entry)
        addFirst(cache, entry)
    else
        newEntry = Entry(key, value, nothing, nothing)
        if length(cache.map) > cache.maxsize
            delete!(cache, cache.tail)
            addFirst(cache, newEntry)
        else
            addFirst(cache, newEntry)
        end
        cache.map[key] = newEntry
    end
end

Base.length(cache::BBCache) = Base.length(cache.map)

function Base.delete!(cache::BBCache, entry::Entry)
    delete!(cache.map, entry.key)
    removeNode(cache, entry)
end

Base.isempty(cache::BBCache) = isempty(cache.map)
function Base.sizehint!(cache::BBCache, n::Integer)
    sizehint!(cache.map, n)
    return cache
end

function addFirst(cache::BBCache, entry::Entry)
    entry.next = cache.head
    entry.previous = nothing
    if (cache.head != nothing)
        cache.head.previous = entry
    end
    cache.head = entry
    if (cache.tail == nothing)
        cache.tail = cache.head
    end
end

function removeNode(cache::BBCache, entry::Entry)
    if entry.previous != nothing
        entry.previous.next = entry.next
    else
        cache.head = entry.next
    end
    if entry.next != nothing
        entry.next.previous = entry.previous
    else
        cache.tail = entry.previous
    end
end

function Base.empty!(cache::BBCache)
    empty!(cache.map)
    return cache
end

Base.show(io::IO, cache::BBCache{K,V}) where {K,V} =
    print(io, "BBCache{$K, $V}(; maxsize = $(cache.maxsize))")

#Vectorized BB algorithm as described in Jherek Healy "Applied Quantitative Finance for Equity Derivatives"
function transformRecursive(
    bb::BrownianBridgeConstruction,
    qrng::AbstractSeq,
    start::Int,
    length::Int,
    dimIndex::Int,
    cache::AbstractDict,
)
    ent = get(cache, dimIndex, nothing)
    if ent != nothing
        return ent
    end #else...
    outputl = Vector{Float64}(undef, length)
    #println(bb.stdDev, Base.length(bb.stdDev))
    size = Base.length(bb.stdDev)
    if dimIndex == size
        skipTo(qrng, 1, start)
        nextn!(qrng, 1, outputl)
        @. outputl *= bb.stdDev[1]
        cache[dimIndex] = outputl
    else
        i = bb.pmap[dimIndex] #bijective
        j = bb.leftIndex[i]
        k = bb.rightIndex[i]
        if j != 1
            outputj = transformRecursive(bb, qrng, start, length, j - 1, cache)
            outputk = transformRecursive(bb, qrng, start, length, k, cache)
            skipTo(qrng, i, start)
            nextn!(qrng, i, outputl)
            wl = bb.leftWeight[i]
            wr = bb.rightWeight[i]
            sd = bb.stdDev[i]
            @. outputl = wl * outputj + wr * outputk + sd * outputl
            cache[dimIndex] = outputl
        else
            outputk = transformRecursive(bb, qrng, start, length, k, cache)
            skipTo(qrng, i, start)
            nextn!(qrng, i, outputl)
            wr = bb.rightWeight[i]
            sd = bb.stdDev[i]
            @. outputl = wr * outputk + sd * outputl
            cache[dimIndex] = outputl
        end
    end
    return outputl
end

#Vectorized BB algorithm as described in Jherek Healy "Applied Quantitative Finance for Equity Derivatives"
function transformByDim!(
    bb::BrownianBridgeConstruction,
    qrng::AbstractSeq,
    start::Int,
    dimIndex::Int,
    output::AbstractVector{<:AbstractFloat},
    cache::AbstractDict,
)
    outputDim = transformRecursive(bb, qrng, start, length(output), dimIndex, cache)
    if dimIndex == 1
        output .= outputDim
        return
    end
    outputm = transformRecursive(bb, qrng, start, length(output), dimIndex - 1, cache)
    @. output = outputDim - outputm
end



#Vectorized BB algorithm for a d-dimensional Brownian path.
function transformByDim!(
    bb::BrownianBridgeConstruction,
    qrng::AbstractSeq,
    start::Int,
    dimIndex::Int,
    output::Array{Float64,2}, #size (n,d) retrieve column(di) = out[:,di] (column major order)
    cache::AbstractDict,
)
    outputDim = transformRecursive(bb, qrng, start, size(output), dimIndex, cache)
    if dimIndex == 1
        output .= outputDim
        return
    end
    outputm = transformRecursive(bb, qrng, start, size(output), dimIndex - 1, cache)
    @. output = outputDim - outputm
end


function transformRecursive(
    bb::BrownianBridgeConstruction,
    qrng::AbstractSeq,
    start::Int,
    osize::Tuple{Int,Int},
    dimIndex::Int,
    cache::AbstractDict,
)
    ent = get(cache, dimIndex, nothing)
    if ent != nothing
        return ent
    end #else...
    outputl = Array{Float64}(undef, osize)  #size (n,d) retrieve column(di) = out[:,di] (column major order)
    d = osize[2]
    #println(bb.stdDev, Base.length(bb.stdDev))
    size = Base.length(bb.stdDev)
    if dimIndex == size
        @inbounds for di = 1:d
            skipTo(qrng, di, start)
            nextn!(qrng, di, @view outputl[:, di])
        end
        @. outputl *= bb.stdDev[1]
        cache[dimIndex] = outputl
    else
        i = bb.pmap[dimIndex] #bijective
        j = bb.leftIndex[i]
        k = bb.rightIndex[i]
        if j != 1
            outputj = transformRecursive(bb, qrng, start, osize, j - 1, cache)
            outputk = transformRecursive(bb, qrng, start, osize, k, cache)
            @inbounds for di = 1:d
                skipTo(qrng, d * (i - 1) + di, start) #i
                nextn!(qrng, d * (i - 1) + di, @view outputl[:, di])
            end
            wl = bb.leftWeight[i]
            wr = bb.rightWeight[i]
            sd = bb.stdDev[i]
            @. outputl = wl * outputj + wr * outputk + sd * outputl
            cache[dimIndex] = outputl
        else
            outputk = transformRecursive(bb, qrng, start, osize, k, cache)
            @inbounds for di = 1:d
                skipTo(qrng, d * (i - 1) + di, start) #i
                nextn!(qrng, d * (i - 1) + di, @view outputl[:, di])
            end
            wr = bb.rightWeight[i]
            sd = bb.stdDev[i]
            #println("sizes ",Base.size(outputl)," ",Base.size(outputk))
            @. outputl = wr * outputk + sd * outputl
            cache[dimIndex] = outputl
        end
    end
    return outputl
end
