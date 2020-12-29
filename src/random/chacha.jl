import Random: rand, seed!
import RandomNumbers: AbstractRNG

# Chacha Random Number Generator
# DOUBLEROUNDS=10 means Chacha20
# Generates 32-bit random numbers

mutable struct Chacha{DOUBLEROUNDS} <: AbstractRNG{UInt32}
    v::Vector{UInt32}
    x::Vector{UInt32}
    subCounter::Int
end

function Chacha8()
    Chacha(
        [
            0x243F6A88,
            0x85A308D3,
            0x13198A2E,
            0x03707344,
            0xA4093822,
            0x299F31D0,
            0x082EFA98,
            0xEC4E6C89,
        ],
        UInt64(0),
        4)
    end
function Chacha(
    key::Vector{UInt32} = [
        0x243F6A88,
        0x85A308D3,
        0x13198A2E,
        0x03707344,
        0xA4093822,
        0x299F31D0,
        0x082EFA98,
        0xEC4E6C89,
    ],
    skipLen::UInt64 = UInt64(0),
    DOUBLEROUNDS::Integer = 10, #Chacha20
)
    r = Chacha{DOUBLEROUNDS}(Vector{UInt32}(undef, 16), Vector{UInt32}(undef, 16), Int8(1))
    v = r.v
    v[1] = 0x61707865
    v[2] = 0x3320646e
    v[3] = 0x79622d32
    v[4] = 0x6b206574
    v[5:12] .= key[1:8]
    v[13] = skipLen
    v[14] = skipLen >> 32
    v[15] = 0
    v[16] = 0
    raw(r)
    r
end

@inline function raw(r::Chacha{DOUBLEROUNDS}) where {DOUBLEROUNDS}
    x = r.x
    @inbounds x .= r.v
    # @inbounds for i = 1:16
    #     x[i] = r.v[i]
    # end
    @inbounds for i = 1:DOUBLEROUNDS
        quarterRound(x, 1, 5, 9, 13)
        quarterRound(x, 2, 6, 10, 14)
        quarterRound(x, 3, 7, 11, 15)
        quarterRound(x, 4, 8, 12, 16)
        quarterRound(x, 1, 6, 11, 16)
        quarterRound(x, 2, 7, 12, 13)
        quarterRound(x, 3, 8, 9, 14)
        quarterRound(x, 4, 5, 10, 15)
    end
    @inbounds @simd for i = 1:16
        x[i] += r.v[i]
    end
    r.v[13] += 1
    if (r.v[13] == 0)
        r.v[14] += 1
    end
end

#@inline ROT32(word::UInt32, count::Int) = bitrotate(word,count) #(((word) << count) | ((word) >> (32 - count)))

@inline function quarterRound(v::Vector{UInt32}, a::Int, b::Int, c::Int, d::Int)
    @inbounds begin
        v[a] += v[b]
        v[d] = bitrotate(v[d] ⊻ v[a], 16)
        v[c] += v[d]
        v[b] = bitrotate(v[b] ⊻ v[c], 12)
        v[a] += v[b]
        v[d] = bitrotate(v[d] ⊻ v[a], 8)
        v[c] += v[d]
        v[b] = bitrotate(v[b] ⊻ v[c], 7)
    end
end

@inline function skip(r::Chacha{DOUBLEROUNDS}, skipLength::UInt64) where {DOUBLEROUNDS}
    actualSkip = (skipLength + r.subCounter - 1) / 16
    this.v[13] += -1 + (actualSkip & 0xffffffff) % UInt32
    this.v[14] += (actualSkip >> 32) % UInt32
    raw(r)
    r.subCounter = (((r.subCounter + (skipLength)) % 16) + 1) % Int8
end

@inline function rand(r::Chacha{R}, ::Type{UInt32}) where {R}
    if (r.subCounter >= 17)
        r.subCounter = 1
        raw(r)
    end
    value = r.x[r.subCounter]
    r.subCounter += 1
    value
end

@inline function rand(r::Chacha{R}, ::Type{UInt64}) where {R}
    if (r.subCounter >= 17)
        r.subCounter = 1
        raw(r)
    end
    value = (UInt64(r.x[r.subCounter]) << 32) | r.x[r.subCounter+1]
    r.subCounter += 2
    value
end

#Right now, use UInt32 version
#@inline rand(r::Chacha, ::Type{Float64}) =
#(Float64(rand(r, UInt64) >> 11) + 0.5) / 9007199254740992.0
