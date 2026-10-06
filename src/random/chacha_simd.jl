import Random: rand, seed!
import RandomNumbers: AbstractRNG

export Chacha8SIMD, ChachaSIMD


# ============================================================================
# ChaCha SIMD RNG
#
# T = UInt64
#
# One ChaCha block:
#
#   16 × UInt32
#   = 8 × UInt64
#
# subCounter:
#
#   0 ... 7
#
# and denotes the next UInt64 to return from x.
#
# skipTo!(rng, n):
#
#   positions the RNG such that the next rand(rng, UInt64)
#   returns stream element n.
#
# Thus:
#
#   skipTo!(rng, 0)
#
# restores the initial stream position.
# ============================================================================


mutable struct ChachaSIMD{DOUBLEROUNDS,T} <: AbstractRNG{T}
    v::Vector{NTuple{4,UInt32}}
    x::Vector{NTuple{4,UInt32}}
    subCounter::UInt8
end


# ============================================================================
# Constructor
# ============================================================================

function ChachaSIMD(
    key::Vector{UInt32} = UInt32[
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
    rng = ChachaSIMD{DOUBLEROUNDS,T}(
        Vector{NTuple{4,UInt32}}(undef, 4),
        Vector{NTuple{4,UInt32}}(undef, 4),
        UInt8(0),
    )

    return initstate!(rng, key, skipLen)
end

function Chacha8SIMD(T = UInt64)
    return ChachaSIMD(
        UInt32[
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

Chacha8SIMD(T, seed::Integer) = ChachaSIMD{4}(T, seed)
    
function ChachaSIMD{DOUBLEROUNDS}(T, seed::Integer) where {DOUBLEROUNDS}
    rng = ChachaSIMD(
        UInt32[
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
        DOUBLEROUNDS,
        T,
    )

    return initstate!(
        rng,
        seedkey_splitmix(seed),
        UInt64(0),
    )
end
function Base.copy(
    rng::ChachaSIMD{DOUBLEROUNDS,T},
) where {DOUBLEROUNDS,T}

    return ChachaSIMD{DOUBLEROUNDS,T}(
        Base.copy(rng.v),
        Base.copy(rng.x),
        rng.subCounter,
    )
end


# ============================================================================
# Initialization
#
# skipLen is interpreted as an absolute UInt64 stream position.
#
# Therefore:
#
#   ChachaSIMD(..., 0, ...) 
#
# starts at stream position zero.
# ============================================================================

function initstate!(
    rng::ChachaSIMD{DOUBLEROUNDS,T},
    key::AbstractVector{UInt32},
    skipLen::UInt64 = UInt64(0),
) where {DOUBLEROUNDS,T}

    length(key) == 8 ||
        throw(ArgumentError("ChaCha key must have length 8"))

    v = rng.v

    # ChaCha constants
    v[1] = (
        0x61707865,
        0x3320646e,
        0x79622d32,
        0x6b206574,
    )

    # Key
    v[2] = (
        key[1],
        key[2],
        key[3],
        key[4],
    )

    v[3] = (
        key[5],
        key[6],
        key[7],
        key[8],
    )

    # 64-bit block counter
    v[4] = (
        UInt32(skipLen & UInt64(0xffffffff)),
        UInt32(skipLen >> 32),
        UInt32(0),
        UInt32(0),
    )

    # We generate the requested block below.
    rng.subCounter = UInt8(0)

    raw(rng)

    # NOTE:
    # initstate! historically interpreted skipLen as a block counter.
    #
    # If skipLen is intended to be a stream-output position instead,
    # use skipTo! after initialization instead.
    #
    # For the normal constructor with skipLen == 0 this is exactly
    # the initial stream state.

    return rng
end


# ============================================================================
# Seeding
# ============================================================================

function seed!(
    rng::ChachaSIMD{DOUBLEROUNDS,T},
    seed::Integer,
) where {DOUBLEROUNDS,T}

    return initstate!(
        rng,
        seedkey_splitmix(seed),
        UInt64(0),
    )
end


function seed!(
    rng::ChachaSIMD{DOUBLEROUNDS,T},
    key::AbstractVector{UInt32},
) where {DOUBLEROUNDS,T}

    return initstate!(
        rng,
        key,
        UInt64(0),
    )
end


function seedkey_splitmix(seed::Integer)

    x = UInt64(seed)
    key = Vector{UInt32}(undef, 8)

    @inbounds for i in 1:4
        x, z = splitmix64_next(x)

        key[2i - 1] = UInt32(z & UInt64(0xffffffff))
        key[2i]     = UInt32(z >> 32)
    end

    return key
end


# ============================================================================
# Counter increment
#
# v[4][1:2] are the low/high UInt32 words of the 64-bit ChaCha counter.
# ============================================================================

@inline function increment_counter!(
    r::ChachaSIMD,
)

    lo = r.v[4][1]
    hi = r.v[4][2]

    if lo == typemax(UInt32)

        r.v[4] = (
            UInt32(0),
            hi + UInt32(1),
            r.v[4][3],
            r.v[4][4],
        )

    else

        r.v[4] = (
            lo + UInt32(1),
            hi,
            r.v[4][3],
            r.v[4][4],
        )

    end

    return r
end


# ============================================================================
# ChaCha quarter-round
# ============================================================================

@inline @inbounds function round(
    x::Vector{NTuple{4,UInt32}},
)

    x[1] = x[1] .+ x[2]

    a = x[4] .⊻ x[1]
    x[4] = (a .<< 16) .| (a .>> (32 - 16))

    x[3] = x[3] .+ x[4]

    a = x[2] .⊻ x[3]
    x[2] = (a .<< 12) .| (a .>> (32 - 12))

    x[1] = x[1] .+ x[2]

    a = x[4] .⊻ x[1]
    x[4] = (a .<< 8) .| (a .>> (32 - 8))

    x[3] = x[3] .+ x[4]

    a = x[2] .⊻ x[3]
    x[2] = (a .<< 7) .| (a .>> (32 - 7))

    return x
end


# ============================================================================
# SIMD permutation
# ============================================================================

@inline @inbounds function shuffle(
    x::Vector{NTuple{4,UInt32}},
)

    x[2] = (
        x[2][2],
        x[2][3],
        x[2][4],
        x[2][1],
    )

    x[3] = (
        x[3][3],
        x[3][4],
        x[3][1],
        x[3][2],
    )

    x[4] = (
        x[4][4],
        x[4][1],
        x[4][2],
        x[4][3],
    )

    return x
end


@inline @inbounds function unshuffle(
    x::Vector{NTuple{4,UInt32}},
)

    x[2] = (
        x[2][4],
        x[2][1],
        x[2][2],
        x[2][3],
    )

    x[3] = (
        x[3][3],
        x[3][4],
        x[3][1],
        x[3][2],
    )

    x[4] = (
        x[4][2],
        x[4][3],
        x[4][4],
        x[4][1],
    )

    return x
end


@inline function roundPair(
    x::Vector{NTuple{4,UInt32}},
)

    round(x)
    shuffle(x)
    round(x)
    unshuffle(x)

    return x
end


# ============================================================================
# Generate one ChaCha block
#
# x = generated block
# v = state for next block after return
# ============================================================================

@inline function raw(
    r::ChachaSIMD{DOUBLEROUNDS},
) where {DOUBLEROUNDS}

    x = r.x

    x .= r.v

    @inbounds @simd for i = 1:DOUBLEROUNDS
        roundPair(x)
    end

    @inbounds @simd for i = 1:4
        x[i] = x[i] .+ r.v[i]
    end

    increment_counter!(r)

    return r
end


# ============================================================================
# RAND UInt64
#
# One ChaCha block contains 8 UInt64 values.
#
# subCounter = 0,...,7
# ============================================================================

@inline @inbounds function rand(
    r::ChachaSIMD{R,UInt64},
    ::Type{UInt64},
) where {R}

    s = r.subCounter

    # Current block exhausted.
    if s >= UInt8(8)

        raw(r)
        s = UInt8(0)

    end

    # Each UInt64 consumes two UInt32 words.
    word = Int(s) << 1

    xIndex = (word >> 2) + 1
    tIndex = (word & 0x03) + 1

    xvalue = r.x[xIndex]

    value =
        (UInt64(xvalue[tIndex]) << 32) |
        UInt64(xvalue[tIndex + 1])

    r.subCounter = s + UInt8(1)

    return value
end


# ============================================================================
# RAND Float64
# ============================================================================

@inline function rand(
    r::ChachaSIMD{R,UInt64},
    ::Type{Float64},
) where {R}

    return (
        Float64(rand(r, UInt64) >> 12) + 0.5
    ) / 0x1p52
end


# ============================================================================
# SKIP TO
#
# skipTo!(rng, n)
#
# means:
#
#   the next rand(rng, UInt64) returns stream[n]
#
# where n is zero-based.
#
# Therefore:
#
#   skipTo!(rng, 0)
#
# restores the initial state.
#
# There are 8 UInt64 outputs per ChaCha block.
# ============================================================================

@inline function skipTo!(
    r::ChachaSIMD{R,UInt64},
    n::UInt64,
) where {R}

    # Which ChaCha block?
    block = n >> 3

    # Which UInt64 inside that block?
    offset = n & UInt64(0x07)

    # Set counter to the requested block.
    r.v[4] = (
        UInt32(block),
        UInt32(block >> 32),
        r.v[4][3],
        r.v[4][4],
    )

    # Generate requested block.
    raw(r)

    # Position inside generated block.
    r.subCounter = UInt8(offset)

    return r
end


# Convenience overload for Integer.
@inline function skipTo!(
    r::ChachaSIMD{R,UInt64},
    n::Integer,
) where {R}

    n < 0 && throw(ArgumentError("skipTo! position must be non-negative"))

    return skipTo!(r, UInt64(n))
end


# ============================================================================
# SKIP
#
# skip!(rng, n)
#
# advances n calls to rand(UInt64).
# ============================================================================

@inline function skip!(
    r::ChachaSIMD{R,UInt64},
    n::UInt64,
) where {R}

    # Current absolute position within the stream cannot be recovered
    # from v alone, because v contains the NEXT block counter.
    #
    # Therefore skip is most robustly implemented by calculating the
    # current position from the counter and subCounter.
    #
    # v[4] currently points to the block AFTER x.
    nextBlock =
        UInt64(r.v[4][1]) |
        (UInt64(r.v[4][2]) << 32)

    currentBlock = nextBlock - UInt64(1)

    currentPosition =
        (currentBlock << 3) +
        UInt64(r.subCounter)

    return skipTo!(r, currentPosition + n)
end


@inline function skip!(
    r::ChachaSIMD{R,UInt64},
    n::Integer,
) where {R}

    n < 0 && throw(ArgumentError("skip! length must be non-negative"))

    return skip!(r, UInt64(n))
end


@inline @inbounds function rand(
    r::ChachaSIMD{R,UInt32},
    ::Type{UInt32},
) where {R}

    s = r.subCounter

    # 16 UInt32 values per ChaCha block
    if s >= UInt8(16)
        raw(r)
        s = UInt8(0)
    end

    xIndex = (s >> 2) + 1
    tIndex = (s & 0x03) + 1

    value = r.x[xIndex][tIndex]

    r.subCounter = s + UInt8(1)

    return value
end
@inline function skipTo!(
    r::ChachaSIMD{R,UInt32},
    n::UInt64,
) where {R}

    # 16 UInt32 outputs per ChaCha block
    block = n >> 4
    offset = n & UInt64(0x0f)

    r.v[4] = (
        UInt32(block),
        UInt32(block >> 32),
        r.v[4][3],
        r.v[4][4],
    )

    raw(r)

    r.subCounter = UInt8(offset)

    return r
end

@inline function skipTo!(
    r::ChachaSIMD{R,UInt32},
    n::Integer,
) where {R}

    n < 0 && throw(ArgumentError("skipTo! position must be non-negative"))

    return skipTo!(r, UInt64(n))
end
@inline function skip!(
    r::ChachaSIMD{R,UInt32},
    n::UInt64,
) where {R}

    nextBlock =
        UInt64(r.v[4][1]) |
        (UInt64(r.v[4][2]) << 32)

    currentBlock = nextBlock - UInt64(1)

    currentPosition =
        (currentBlock << 4) +
        UInt64(r.subCounter)

    return skipTo!(r, currentPosition + n)
end

@inline function skip!(
    r::ChachaSIMD{R,UInt32},
    n::Integer,
) where {R}

    n < 0 && throw(ArgumentError("skip! length must be non-negative"))

    return skip!(r, UInt64(n))
end
