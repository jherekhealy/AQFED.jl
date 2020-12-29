#import Random: rand!
export ScrambledSobolSeq, next!

include("ssoboldata.jl") #loads `sobol_a` and `sobol_minit`

abstract type AbstractSobolSeq{N} end


abstract type ScramblingRng end


abstract type Scrambling end

struct NoScrambling <: Scrambling end

# Owen's recommendation is maxd >= L+10 where L is the log2 of maxNumSamples. maxd=31 scrambles all bits of an int.
struct Owen <: Scrambling
    maxd::Int
    rng::ScramblingRng
end
struct FaureTezuka <: Scrambling
    rng::ScramblingRng
end
struct OwenFaureTezuka <: Scrambling
    maxd::Int
    rng::ScramblingRng
end

unirand(s::Union{Owen,FaureTezuka,OwenFaureTezuka}) = unirand(s.rng)

#An adapter to use a standard 64-bit or 32-bit integer RNG for scrambling
mutable struct ScramblingRngAdapter{T} <: ScramblingRng
    rng::AbstractRNG{T} #may be Chacha, Philox or Blabla. Not suited for MT/Well as individual bits distribution not as good
    counter::UInt8
    x::T
    ScramblingRngAdapter(rng::AbstractRNG{T}) where {T} = new{T}(rng, UInt8(64), zero(T))
end

@inline function unirand(s::ScramblingRngAdapter{UInt64})
    if s.counter >= 64
        s.x = rand(s.rng, UInt64)
        s.counter = 0
    end
    b = (s.x >> s.counter) & one(UInt64)
    s.counter += 1
    return b
end

@inline function unirand(s::ScramblingRngAdapter{UInt32})
    if s.counter >= 32
        s.x = rand(s.rng, UInt32)
        s.counter = 0
    end
    b = (s.x >> s.counter) & one(UInt32)
    s.counter += 1
    return b
end

mutable struct OriginalScramblingRng <: ScramblingRng
    seedi::Int32
    seedj::Int32
    seedcarry::Float64
    seedseeds::Vector{Float64}
end

function OriginalScramblingRng()
    seedi = 24
    seedj = 10
    seedcarry = 0
    seedseeds = [
        0.8804418,
        0.2694365,
        0.0367681,
        0.4068699,
        0.4554052,
        0.2880635,
        0.1463408,
        0.2390333,
        0.6407298,
        0.1755283,
        0.713294,
        0.4913043,
        0.2979918,
        0.1396858,
        0.3589528,
        0.5254809,
        0.9857749,
        0.4612127,
        0.2196441,
        0.7848351,
        0.40961,
        0.9807353,
        0.2689915,
        0.5140357,
    ]
    rng = OriginalScramblingRng(seedi, seedj, seedcarry, seedseeds)
    resetSeed(rng)
end

function seed(s::OriginalScramblingRng, seeds::Vector{Float64})
    s.seedi = 24
    s.seedj = 10
    s.seedcarry = 0
    s.seedseeds .= seeds
    s
end

function resetSeed(rng::OriginalScramblingRng)
    seeds = [
        0.8804418,
        0.2694365,
        0.0367681,
        0.4068699,
        0.4554052,
        0.2880635,
        0.1463408,
        0.2390333,
        0.6407298,
        0.1755283,
        0.713294,
        0.4913043,
        0.2979918,
        0.1396858,
        0.3589528,
        0.5254809,
        0.9857749,
        0.4612127,
        0.2196441,
        0.7848351,
        0.40961,
        0.9807353,
        0.2689915,
        0.5140357,
    ]
    seed(rng, seeds)
end

@inline unirand(s::OriginalScramblingRng) = trunc(UInt32, rand(s) * 1000.0) % 2

@inline function rand(s::OriginalScramblingRng)
    retVal = s.seedseeds[s.seedi] - s.seedseeds[s.seedj] - s.seedcarry
    if retVal < 0
        retVal += 1
        s.seedcarry = 5.9604644775390625e-8 #1/2^24
    else
        s.seedcarry = 0
    end
    s.seedseeds[s.seedi] = retVal
    s.seedi = 24 - (25 - s.seedi) % 24
    s.seedj = 24 - (25 - s.seedj) % 24
    return retVal
end

# N iis the dimension of sequence being generated, S is scrambling type
mutable struct ScrambledSobolSeq{N,S} <: AbstractSobolSeq{N}
    v::Array{UInt32,2} #array of size (sdim, 32)
    #points::Array{UInt32,1}
    x::Array{UInt32,1}
    shift::Array{UInt32,1}
    l::UInt32
    counter::UInt32
    scrambling::S
end

ndims(s::AbstractSobolSeq{N}) where {N} = N::Int

unirand(s::ScrambledSobolSeq) = unirand(s.scrambling)

#@inline norm(s::ScrambledSobolSeq{N}) where {N} = 1 / 2^s.l #non scrambled norm
@inline normalize(s::ScrambledSobolSeq{N,FaureTezuka}, x::UInt32) where {N} =
    ldexp(Float64(x), -s.l % Int32)
@inline normalize(s::ScrambledSobolSeq{N,Owen}, x::UInt32) where {N} =
    ldexp(Float64(x), -s.scrambling.maxd % Int32)
@inline normalize(s::ScrambledSobolSeq{N,NoScrambling}, x::UInt32) where {N} =
    ldexp(Float64(x), -s.l % Int32)

function ScrambledSobolSeq(dimension::Int, n::Int, scrambling::Scrambling)
    d = dimension
    (d < 0 || d > (length(sobol_a) + 1)) && error("invalid Sobol dimension")
    #special cases
    d == 0 && return (ScrambledSobolSeq{0})(v, UInt32[], UInt32[], zero(UInt32))
    #l=30
    l = 31
    if (n > 0)
        l = 0
        i = n
        while (i != 0)
            i >>= 1
            l += 1
        end
        l = min(l, 31)
    end
    #println("l=",l)
    #points = Vector{Float64}(undef, d)
    x = Vector{UInt32}(undef, d)
    fill!(x, 0)
    counter = 0
    v = ones(UInt32, (d, l))
    for i = 1:l
        v[1, i] = one(UInt32) << (l - i)
    end
    for j = 2:d
        a = sobol_a[j-1]
        s = floor(Int, log2(a)) #degree of poly

        #set initial values of m from table
        if l <= s
            for i = 1:l
                v[j, i] = sobol_minit[i, j-1] << (l - i)
            end
        else
            #fill in remaining values using recurrence
            for i = 1:s
                v[j, i] = sobol_minit[i, j-1] << (l - i)
            end
            for i = s+1:l
                v[j, i] = v[j, i-s] ⊻ (v[j, i-s] >> s)
                for k = 1:s-1 #or from 0?
                    @inbounds v[j, i] ⊻= (((a >> (s - 1 - k)) & one(UInt32)) * v[j, i-k])
                end
            end
        end
    end
    vt = ones(UInt32, (l, d))
    shift = Vector{UInt32}(undef, d)
    ss = ScrambledSobolSeq{d,typeof(scrambling)}(v, x, shift, l, counter, scrambling)
    scramble(ss)
    ss
end

ScrambledSobolSeq(N::Integer) = ScrambledSobolSeq(Int(N), 0, NoScrambling())

function scramble(s::ScrambledSobolSeq{D,NoScrambling}) where {D}
    s.counter = 1 #skip first item for non-scrambled version
    fill!(s.shift, 0)
end

function scramble(s::ScrambledSobolSeq{D,Owen}) where {D}
    maxd = s.scrambling.maxd
    if maxd < s.l
        throw(DomainError(string("maxd must be >= l where l = log2(n) = ", s.l)))
    end
    s.counter = 0
    fill!(s.shift, 0)
    scrambledV = zeros(UInt32, (D, s.l))
    #norm = 1/2^MAXD
    lsm = genscrml(s)
    for i = 1:D
        for j = 1:s.l
            l = 1
            temp2 = 0
            for p = maxd:-1:1
                temp1 = 0
                for k = 1:s.l
                    temp1 += lbitbits(lsm[i, p], k - 1, 1) * lbitbits(s.v[i, j], k - 1, 1)
                end
                temp1 %= 2
                temp2 += temp1 << (l - 1)
                l += 1
            end
            scrambledV[i, j] = temp2
        end
    end
    s.v .= scrambledV
    s.x .= s.shift
end

function genscrml(s::ScrambledSobolSeq{D,Owen}) where {D}
    maxd = s.scrambling.maxd
    lsm = zeros(UInt32, (D, maxd))
    for p = 1:D
        s.shift[p] = 0
        l = 1
        for i = maxd:-1:1
            lsm[p, i] = 0
            stemp = unirand(s)
            s.shift[p] += stemp << (l - 1)
            l += 1
            ll = 1
            for j = s.l:-1:1
                local temp
                if (j == i)
                    temp = 1
                elseif (j < i)
                    temp = unirand(s)
                else
                    temp = 0
                end
                lsm[p, i] += temp << (ll - 1)
                ll += 1
            end
        end
    end
    return lsm
end

@inline function lbitbits(a::UInt32, b::Int, len::Int)
    x = UInt64(a)
    y = UInt64(0xffffffffffffffff) #-1
    x >>= b
    y <<= len
    return (x & ~y) % UInt32
end

function next(s::ScrambledSobolSeq, ::Type{UInt32})
    d = ndims(s)
    c = ffz(s.counter)
    c > s.l && throw(DomainError(string("counter larger than sequence length: ",s.counter)))
    s.counter += one(s.counter)
    sx = s.x
    sv = s.v
    for j = 1:d
        sx[j] ⊻= sv[j, c]
    end

    return s.x
end

#next vector at counter containing all dimensions
@inline function next!(s::ScrambledSobolSeq, points::AbstractVector{<:AbstractFloat})
    d = ndims(s)
    length(points) != d && throw(BoundsError())
    c = ffz(s.counter)
    c > s.l && throw(DomainError(string("counter larger than sequence length: ",s.counter)))

    s.counter += one(s.counter)
    sx = s.x
    sv = s.v
    @inbounds for j = 1:d
        sx[j] ⊻= sv[j, c]
        if sx[j] == 0
            points[j] = normalize(s, one(UInt32)) / 2
        else
            points[j] = normalize(s, sx[j])
        end
    end
    return points
end

@inline function ffz(x::Integer)
    if x == 0
        return 1
    end
    return trailing_zeros(x)+1
end

#next vector for a given dimension (vertical) from counter to counter + length(points)
function next!(s::ScrambledSobolSeq, dim::Int, points::AbstractVector{<:AbstractFloat})
    j = dim
    sx = s.x
    sv = s.v
    for i = 1:length(points)
        c = ffz(s.counter)
        s.counter += one(s.counter)
        sx[j] ⊻= sv[j, c]
        points[i] =
            (sx[j] == 0) ? normalize(s, s.scrambling, one(UInt32)) / 2 :
            normalize(s, s.scrambling, sx[j])
    end
    return points
end

@inline function skip(s::ScrambledSobolSeq, n::Int)
    m = n + s.counter
    g = m ⊻ (m >> 1)
    d = ndims(s)
    sx = s.x
    sv = s.v
    @inbounds for j = 1:d
        @inbounds for index = 1:s.l
            if ((g >> index) & one(UInt32)) != 0
                sx[j] ⊻= sv[j, index]
            end
        end
    end
    s.counter += n
    s
end

@inline function skipTo(s::ScrambledSobolSeq, n::Int)
    # Convert to Gray code
    g = n ⊻ (n >> 1)
    d = ndims(s)
    sx = s.x
    sv = s.v
    @inbounds for j = 1:d
        s.x[j] = s.shift[j]
        @inbounds for index = 1:s.l
            if ((g >> index) & one(UInt32)) != 0
                sx[j] ⊻= sv[j, index]
            end
        end
    end
    s.counter = n
    s
end

#skip to position n for a given dimension dim
@inline function skipTo(s::ScrambledSobolSeq, dim::Int, n::Int)
    # Convert to Gray code
    g = n ⊻ (n >> 1)
    j = dim
    sx = s.x
    sv = s.v
    s.x[j] = s.shift[j]

    @inbounds for index = 1:s.l
        if ((g >> index) & one(UInt32)) != 0
            sx[j] ⊻= sv[j, index]
        end
    end
    s.counter = n
    s
end

#TODO various scrambles
