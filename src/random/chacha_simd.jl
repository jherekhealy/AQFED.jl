import Random: rand, seed!
import RandomNumbers: AbstractRNG

# Chacha Random Number Generator with SIMD-128 in mind
# DOUBLEROUNDS=10 means Chacha20
# Generates 64-bit random numbers

mutable struct ChachaSIMD{DOUBLEROUNDS,T} <: AbstractRNG{T}
    v::Vector{NTuple{4,UInt32}}
    x::Vector{NTuple{4,UInt32}}
    subCounter::Int8
end

function Chacha8SIMD(T = UInt64) #::Union{Type{UInt32},Type{UInt64}
    ChachaSIMD(
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
        4,
        T,
    )
end

function ChachaSIMD(
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
    DOUBLEROUNDS::Integer = 10,
    T = UInt64,
)
    r = ChachaSIMD{DOUBLEROUNDS,T}(
        Vector{NTuple{4,UInt32}}(undef, 4),
        Vector{NTuple{4,UInt32}}(undef, 4),
        Int8(1),
    )
    v = r.v
    v[1] = (0x61707865, 0x3320646e, 0x79622d32, 0x6b206574)
    v[2] = (key[1], key[2], key[3], key[4])
    v[3] = (key[5], key[6], key[7], key[8])
    v[4] = (skipLen & 0xffffffff, skipLen >> 32, 0, 0)
    raw(r)
    r
end

@inline @inbounds function round(x::Vector{NTuple{4,UInt32}})
    x[1] = x[1] .+ x[2]
    a = (x[4] .⊻ x[1])
    x[4] = (a .<< 16) .| (a .>> (32 - 16))
    x[3] = x[3] .+ x[4]
    a = x[2] .⊻ x[3]
    x[2] = (a .<< 12) .| (a .>> (32 - 12))
    x[1] = x[1] .+ x[2]
    a = x[4] .⊻ x[1]
    x[4] = (a .<< 8) .| (a .>> (32 - 8))
    x[3] = x[3] .+ x[4]
    a = (x[2] .⊻ x[3])
    x[2] = (a .<< 7) .| (a .>> (32 - 7))

end

@inline @inbounds function shuffle(x::Vector{NTuple{4,UInt32}})
    x[2] = (x[2][2], x[2][3], x[2][4], x[2][1])
    x[3] = (x[3][3], x[3][4], x[3][1], x[3][2])
    x[4] = (x[4][4], x[4][1], x[4][2], x[4][3])
end

@inline @inbounds function unshuffle(x::Vector{NTuple{4,UInt32}})
    x[2] = (x[2][4], x[2][1], x[2][2], x[2][3])
    x[3] = (x[3][3], x[3][4], x[3][1], x[3][2])
    x[4] = (x[4][2], x[4][3], x[4][4], x[4][1])
end

@inline function roundPair(x::Vector{NTuple{4,UInt32}})
    round(x)
    shuffle(x)
    round(x)
    unshuffle(x)
end

@inline function raw(r::ChachaSIMD{DOUBLEROUNDS}) where {DOUBLEROUNDS}
    x = r.x
    x .= r.v
    @inbounds @simd for i = 1:DOUBLEROUNDS
        roundPair(x)
    end
    @inbounds @simd for i = 1:4
        x[i] = x[i] .+ r.v[i]
    end
    @inbounds r.v[4] = (r.v[4][1] + 1, r.v[4][2], r.v[4][3], r.v[4][4])
    @inbounds if (r.v[4][1] == 0)
        r.v[4] = (r.v[4][1], r.v[4][2] + 1, r.v[4][3], r.v[4][4])
    end
end

@inline function skip(r::ChachaSIMD{DOUBLEROUNDS}, skipLength::UInt64) where {DOUBLEROUNDS}
    actualSkip = (skipLength + r.subCounter - 1) / 8
    r.v[4] = (
        r.v[4][1] - 1 + (actualSkip & 0xffffffff) % UInt32,
        (r.v[4][2] >> 32) % UInt32,
        r.v[4][3],
        r.v[4][4],
    )
    raw(r)
    r.subCounter = (((r.subCounter + (skipLength)) % 8) + 1) % Int8
end

@inline @inbounds function rand(r::ChachaSIMD{R,UInt32}, ::Type{UInt32}) where {R}
    if (r.subCounter == 17)
        r.subCounter = 1
        raw(r)
    end
    xIndex = ((r.subCounter - 1) >> 2) + 1
    tIndex = (r.subCounter - 1 - ((xIndex - 1) << 2)) + 1
    xvalue = r.x[xIndex]
    value = xvalue[tIndex]
    r.subCounter += 1
    value
end

@inline @inbounds function rand(r::ChachaSIMD{R,UInt64}, ::Type{UInt32}) where {R}
    if (r.subCounter == 17)
        r.subCounter = 1
        raw(r)
    end
    xIndex = ((r.subCounter - 1) >> 2) + 1
    tIndex = (r.subCounter - 1 - ((xIndex - 1) << 2)) + 1
    xvalue = r.x[xIndex]
    value = xvalue[tIndex]
    r.subCounter += 2 #skip one int32 for compatibility with 64-bit
    value
end

@inline @inbounds function rand(r::ChachaSIMD{R}, ::Type{UInt64}) where {R}
    if (r.subCounter == 17)
        r.subCounter = 1
        raw(r)
    end
    xIndex = ((r.subCounter - 1) >> 2) + 1
    tIndex = (r.subCounter - 1 - ((xIndex - 1) << 2)) + 1
    @inbounds xvalue = r.x[xIndex]
    @inbounds value = (UInt64(xvalue[tIndex]) << 32) | xvalue[tIndex+1]
    r.subCounter += 2
    value
end
@inline rand(r::ChachaSIMD, ::Type{Float64}) =
    (Float64(rand(r, UInt64) >> 11) + 0.5) / 9007199254740992.0
