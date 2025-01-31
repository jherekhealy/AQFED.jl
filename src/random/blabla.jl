import Random: rand, seed!
import RandomNumbers: AbstractRNG
export Blabla8, Blabla, Blabla10
# Blabla Random Number Generator, based on BLAKE2b hash function.
# Adapted from https://github.com/veorq/blabla/blob/master/BlaBla.swift by J.P Aumasson (2017)
#
# Original Blabla is 10 rounds, akin to Chacha 20 rounds. BLAKE2b uses 12 rounds and is still secure with 6 rounds.

mutable struct Blabla{ROUND,T} <: AbstractRNG{T}
    v::Vector{UInt64}
    x::Vector{UInt64}
    subCounter::Int8
end

function Blabla8(
    key::Vector{UInt64} = [0xB105F00DBAAAAAAD, 0xdeadbeefcafebabe, 0xFEE1DEADFEE1DEAD, 0xdeadbeefcafebabe],
    T = UInt64
    )
    Blabla(key, 1, 8, T)
end

function Blabla10(
    key::Vector{UInt64} = [0xB105F00DBAAAAAAD, 0xdeadbeefcafebabe, 0xFEE1DEADFEE1DEAD, 0xdeadbeefcafebabe],
    T = UInt64)
    Blabla(key, 1, 10, T)
end

function Blabla(
    key::Vector{UInt64} = [0xB105F00DBAAAAAAD, 0xdeadbeefcafebabe, 0xFEE1DEADFEE1DEAD, 0xdeadbeefcafebabe],
    skipLen = 0,
    ROUND::Integer = 10,
    T = UInt64
)
    r = Blabla{ROUND, T}(Vector{UInt64}(undef, 16), Vector{UInt64}(undef, 16),Int8(1))
    v = r.v
    v[1] = 0x6170786593810fab
    v[2] = 0x3320646ec7398aee
    v[3] = 0x79622d3217318274
    v[4] = 0x6b206574babadada
    v[5:8] .= key[1:4]
    v[9] = 0x2ae36e593e46ad5f
    v[10] = 0xb68f143029225fc9
    v[11] = 0x8da1e08468303aa6
    v[12] = 0xa48a209acd50a4a7
    v[13] = 0x7fdc12f23f90778c
    v[14] = skipLen
    v[15] = 0
    v[16] = 0
    raw(r)
    r
end

@inline function raw(r::Blabla{ROUND}) where {ROUND}
    x = r.x
    x .= r.v
    # @inbounds for i = 1:16
    #     x[i] = r.v[i]
    # end
    @inbounds @simd for i = 1:ROUND
        G(x, 1, 5, 9, 13)
        G(x, 2, 6, 10, 14)
        G(x, 3, 7, 11, 15)
        G(x, 4, 8, 12, 16)
        G(x, 1, 6, 11, 16)
        G(x, 2, 7, 12, 13)
        G(x, 3, 8, 9, 14)
        G(x, 4, 5, 10, 15)
    end
    @inbounds @simd for i = 1:16
        x[i] += r.v[i]
    end
    r.v[14] += 1
end

#@inline ROTR64(word::UInt64, count::Int) = bitrotate(word, -count)
# ((word >> count) ⊻ (word << (64 - count)))

@inline function G(v::Vector{UInt64}, a::Int, b::Int, c::Int, d::Int)
    @inbounds begin
    v[a] += v[b]
    v[d] = bitrotate(v[d] ⊻ v[a], -32)
    v[c] += v[d]
    v[b] = bitrotate(v[b] ⊻ v[c], -24)
    v[a] += v[b]
    v[d] = bitrotate(v[d] ⊻ v[a], -16)
    v[c] += v[d]
    v[b] = bitrotate(v[b] ⊻ v[c], -63)
end
end

@inline function skip(r::Blabla, skipLength::UInt64)
    actualSkip = (skipLength + r.subCounter-1) ÷ 16
    r.v[14] += -1 + actualSkip
    raw(r)
    r.subCounter = (((r.subCounter + (skipLength)) % 16) +1) % Int8
end

@inline function skipTo(r::Blabla, skip::Int)
    actualSkip = (UInt64(skip) + r.subCounter-1) ÷ 16
    r.v[14] = -1 + actualSkip
    raw(r)
    r.subCounter = (((r.subCounter + UInt64(skip)) % 16) +1) % Int8
end

@inline function rand(r::Blabla, ::Type{UInt64})
    if (r.subCounter == 17)
        r.subCounter = 1
        raw(r)
    end
    value = r.x[r.subCounter]
    r.subCounter += 1
    value
end

@inline rand(r::Blabla, ::Type{UInt32}) = (rand(r, UInt64) >> 32) % UInt32

#@inline rand(r::Blabla{R}, ::Type{Float64}) = (Float64(Random.rand(r, UInt64) >> 32 )+0.5) /4294967296.0
