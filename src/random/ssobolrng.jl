import RandomNumbers: AbstractRNG

abstract type ScramblingRng end

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
