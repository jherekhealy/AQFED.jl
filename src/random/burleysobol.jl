# burleysobol.jl
#
# Burley-style scrambled Sobol sequence for arbitrary dimension N.
#
# The dimensions are generated in blocks of four:
#
#   block 0 -> Sobol dimensions 1:4
#   block 1 -> the same 4D Sobol construction, with a new group seed
#   ...
#   final block -> only the required coordinates are retained
#
# This follows Burley's "padding to higher dimensions" construction:
# a different seed is supplied to each group of four dimensions.
#
# This file is intended to be used alongside the existing ssobol.jl /
# ssoboldata.jl infrastructure:
#   - load_direction_numbers(...)
#   - DirectionSource / joe_kuo
#
# The implementation deliberately keeps Burley's 32-bit index scrambling.
# skipTo! therefore gives O(1) repositioning and the randomized sequence
# does not depend on a requested power-of-two sample count.

import Random: rand!, AbstractRNG
import AQFED.Math: norminv

export BurleySobolSeq, next!, skip!, skipTo!, nextn!

# ---------------------------------------------------------------------------
# Burley hash / nested uniform scramble
# ---------------------------------------------------------------------------

@inline function hash_combine(
    seed::UInt32, v::UInt32
)::UInt32
    return xor(seed, v + (seed << 6) + (seed >> 2))
end
@inline function murmur3_finalize(x::UInt32)::UInt32
    x ⊻= x >> 16
    x *= 0x85ebca6b
    x ⊻= x >> 13
    x *= 0xc2b2ae35
    x ⊻= x >> 16
    return x
end
mutable struct MurmurRNG{T} <: AbstractRNG{T}
    counter::T
    seed::T
end
function MurmurRNG(seed::Int=UInt32(0))
    MurmurRNG{UInt32}(UInt32(1), UInt32(seed) * UInt32(0x9e3779b9)) #0 is preserved.
end


function Base.copy(rng::MurmurRNG{T}) where {T}
    return MurmurRNG{T}(rng.counter, rng.seed)
end

function Random.rand(rng::MurmurRNG{T}, ::Type{T})::T where {T<:Union{UInt32,UInt64}}
    h = rng.counter ⊻ rng.seed #seed = 0 => counter is the key
    rng.counter += T(1)
    return murmur3_finalize(h)
end

function skipTo!(rng::MurmurRNG{T}, n::Integer) where {T}
    rng.counter = T(n+1) #skipTo(0) == init.
end


const _directions = UInt32[
    0x80000000 0x40000000 0x20000000 0x10000000 0x08000000 0x04000000 0x02000000 0x01000000 0x00800000 0x00400000 0x00200000 0x00100000 0x00080000 0x00040000 0x00020000 0x00010000 0x00008000 0x00004000 0x00002000 0x00001000 0x00000800 0x00000400 0x00000200 0x00000100 0x00000080 0x00000040 0x00000020 0x00000010 0x00000008 0x00000004 0x00000002 0x00000001;
    0x80000000 0xc0000000 0xa0000000 0xf0000000 0x88000000 0xcc000000 0xaa000000 0xff000000 0x80800000 0xc0c00000 0xa0a0a000 0xf0f00000 0x88880000 0xcccc0000 0xaaaa0000 0xffff0000 0x80008000 0xc000c000 0xa000a000 0xf000f000 0x88008800 0xcc00cc00 0xaa00aa00 0xff00ff00 0x80808080 0xc0c0c0c0 0xa0a0a0a0 0xf0f0f0f0 0x88888888 0xcccccccc 0xaaaaaaaa 0xffffffff;
    0x80000000 0xc0000000 0x60000000 0x90000000 0xe8000000 0x5c000000 0x8e000000 0xc5000000 0x68800000 0x9cc00000 0xee600000 0x55900000 0x80680000 0xc09c0000 0x60ee0000 0x90550000 0xe8808000 0x5cc0c000 0x8e606000 0xc5909000 0x6868e800 0x9c9c5c00 0xeeee8e00 0x5555c500 0x8000e880 0xc0005cc0 0x60008e60 0x9000c590 0xe8006868 0x5c009c9c 0x8e00eeee 0xc5005555;
    0x80000000 0xc0000000 0x20000000 0x50000000 0xf8000000 0x74000000 0xa2000000 0x93000000 0xd8800000 0x25400000 0x59e00000 0xe6d00000 0x78080000 0xb40c0000 0x82020000 0xc3050000 0x208f8000 0x51474000 0xfbea2000 0x75d93000 0xa0858800 0x914e5400 0xdbe79e00 0x25db6d00 0x58800080 0xe54000c0 0x79e00020 0xb6d00050 0x800800f8 0xc00c0074 0x200200a2 0x50050093;
    0x80000000 0x40000000 0x20000000 0xb0000000 0xf8000000 0xdc000000 0x7a000000 0x9d000000 0x5a800000 0x2fc00000 0xa1600000 0xf0b00000 0xda880000 0x6fc40000 0x81620000 0x40bb0000 0x22878000 0xb3c9c000 0xfb65a000 0xddb2d000 0x78022800 0x9c0b3c00 0x5a0fb600 0x2d0ddb00 0xa2878080 0xf3c9c040 0xdb65a020 0x6db2d0b0 0x800228f8 0x400b3cdc 0x200fb67a 0xb00ddb9d;
]

mutable struct BurleySobolSeq{N} <: AbstractSeq{N}
    # 4 x 32 Sobol direction numbers.  We intentionally reuse these four
    # dimensions for every Burley group.

    # Current scrambled coordinate values.
    x::Vector{UInt32}

    # One seed for index scrambling, then one seed per 4D group.
    #
    # length(seed_array) == 1 + cld(N, 4)
    seed_array::Vector{UInt32}

    # Current sequence position.  Position 0 is the first sample.
    counter::UInt32
end

@inline ndims(::BurleySobolSeq{N}) where {N} = N

@inline nblocks(::BurleySobolSeq{N}) where {N} = cld(N, 4)

# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

"""
    BurleySobolSeq(dimension, rng=Chacha8SIMD(UInt32);
                   direction_source=joe_kuo)

Create a Burley-style scrambled Sobol sequence with arbitrary dimension.

The first four Sobol dimensions form the base 4D construction.  For every
additional group of four dimensions, the same 4D Sobol construction is
reused with a different group seed, as described by Burley.

`seed_array[1]` is the sample-index shuffle seed.
`seed_array[g+1]` is the seed for 4D group `g`.

This constructor assumes the existing `load_direction_numbers` and
`DirectionSource` definitions from ssobol.jl are available.
"""
function BurleySobolSeq(
    dimension::Integer,
    rng::AbstractRNG=Chacha8SIMD(UInt32))
    N = Int(dimension)
    N <= 0 && throw(ArgumentError("dimension must be > 0"))

    seeds = zeros(UInt32, cld(N, 4))
    rand!(rng, seeds)

    return BurleySobolSeq{N}(
        zeros(UInt32, N),
        seeds,
        UInt32(0),
    )
end

@inline function BurleySobolSeq(
    dimension::Integer,
    seed::UInt64)

    # Deterministic convenience constructor.
    rng = Chacha8SIMD(UInt32)
    Random.seed!(rng, seed)
    return BurleySobolSeq(
        dimension,
        rng
    )
end

# ---------------------------------------------------------------------------
# Build the first four Sobol direction numbers.
# This follows the same recurrence used by ssobol.jl, but keeps 32 bits.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sobol evaluation
# ---------------------------------------------------------------------------
function sobol4d!(X::Vector{UInt32}, index::UInt32)
    @assert length(X) == 4
    for dim in 1:4
        x = UInt32(0)
        for bit in 0:31
            mask = (index >> bit) & UInt32(1)
            x ⊻= mask * _directions[dim, bit+1]  # Julia is 1-indexed
        end
        X[dim] = x
    end
end


@inline function sobol4d(index::UInt32)
    x0 = zero(UInt32); x1 = zero(UInt32); x2 = zero(UInt32); x3 = zero(UInt32)
    @inbounds @simd for bit in 0:31
        mask = (index >> bit) & UInt32(1)
        x0 ⊻= mask * _directions[1, bit+1]
        x1 ⊻= mask * _directions[2, bit+1]
        x2 ⊻= mask * _directions[3, bit+1]
        x3 ⊻= mask * _directions[4, bit+1]
    end
    return (x0, x1, x2, x3)
end
@inline function _sobolupdate(index::UInt32, dim::UInt32)
    x = UInt32(0)
    @inbounds @simd for bit in 1:32
        mask = (index >> (bit-1)) & UInt32(1)
        x ⊻= mask * _directions[dim, bit]  # Julia is 1-indexed
    end
    return x
end
@inline function _scrambled_group!(s::BurleySobolSeq{D}, group::Int, i::UInt32) where {D}
    gs = @inbounds s.seed_array[group+1]
    index = nested_uniform_scramble_base2(i, gs)
    x0, x1, x2, x3 = sobol4d(index)

    first = 4 * group + 1
    last  = min(first + 3, D)
    n     = last - first + 1
    @inbounds begin
        n >= 1 && (s.x[first]   = nested_uniform_scramble_base2(x0, hash_combine(gs, UInt32(0))))
        n >= 2 && (s.x[first+1] = nested_uniform_scramble_base2(x1, hash_combine(gs, UInt32(1))))
        n >= 3 && (s.x[first+2] = nested_uniform_scramble_base2(x2, hash_combine(gs, UInt32(2))))
        n >= 4 && (s.x[first+3] = nested_uniform_scramble_base2(x3, hash_combine(gs, UInt32(3))))
    end
    return s.x
end
"""
    next!(s, points)

Generate the current sample and advance the sequence by one.

The generated coordinates are in the open interval (0,1), using the same
32-bit midpoint normalization as the existing Burley implementation.
"""
@inline function next!(s::BurleySobolSeq{N}, ::Type{UInt32}=UInt32) where {N}
    i = s.counter
    @inbounds for g in 0:(cld(N, 4)-1)
        _scrambled_group!(s, g, i)
    end
    s.counter = i + UInt32(1)
    return s.x
end
# ---------------------------------------------------------------------------
# Positioning / iteration
# ---------------------------------------------------------------------------

"""
    skipTo!(s, n)

Position the Burley Sobol sequence so that the next call to `next!` returns
sample `n`.

As in the original ssobol.jl Burley implementation, `n` is zero-based:

    skipTo!(s, 0); next!(s, p)  # returns sample 0
    skipTo!(s, 17); next!(s, p) # returns sample 17

The operation is O(1): the Sobol point is evaluated directly at `n` after
Burley's nested uniform index permutation.
"""
@inline function skipTo!(
    s::BurleySobolSeq{N},
    n::Integer,
) where {N}
    n < 0 && throw(ArgumentError("sample index must be non-negative"))
    s.counter = UInt32(n)
    return s
end

"""
    skip!(s, n)

Advance the sequence by `n` samples.
"""
@inline function skip!(
    s::BurleySobolSeq,
    n::Integer,
)
    n < 0 && throw(ArgumentError("skip count must be non-negative"))

    new_counter = UInt64(s.counter) + UInt64(n)
    new_counter > typemax(UInt32) &&
        throw(OverflowError("Burley Sobol index exceeds UInt32 range"))

    s.counter = UInt32(new_counter)
    return s
end

# Match the dimension-specific skipTo! API used by ssobol.jl.
@inline function skipTo!(
    s::BurleySobolSeq{N},
    dim::Int,
    n::Integer,
    npoints::Integer=0,
) where {N}
    1 <= dim <= N ||
        throw(BoundsError("dimension $dim is outside 1:$N"))
    npoints < 0 &&
        throw(ArgumentError("npoints must be non-negative"))
    return skipTo!(s, n)
end


@inline function next!(
    s::BurleySobolSeq{N},
    points::P
) where {N, P<:Union{AbstractVector{<:AbstractFloat}, AbstractMatrix}}
    # length(points) == N ||
    #     throw(DimensionMismatch("points has length $(length(points)), expected $N"))

    next!(s, UInt32)

    @inbounds @simd for j in 1:N
        points[j] = to_unit(s.x[j])
    end

    return points
end

@inline function nextn!(
    s::BurleySobolSeq{N},
    points::P,
) where {N, P<:Union{AbstractVector{<:AbstractFloat}, AbstractMatrix}}
    next!(s, points)

    @inbounds for j in eachindex(points)
        points[j] = norminv(points[j])
    end

    return points
end

# ---------------------------------------------------------------------------
# Single-dimension vector generation
# ---------------------------------------------------------------------------

"""
    next!(s, dim, points)

Generate consecutive samples for one dimension.

This is evaluated directly at the Burley-shuffled sample index rather than
advancing the full N-dimensional state, which keeps this analogous to the
dimension-specific method in ssobol.jl.

`dim` is one-based.
"""
@inline function next!(
    s::BurleySobolSeq{N},
    dim::Int,
    points::AbstractVector{<:AbstractFloat},
) where {N}
    1 <= dim <= N ||
        throw(BoundsError("dimension $dim is outside 1:$N"))

    group = (dim - 1) ÷ 4
    local_dim = (dim - 1) % 4
    gs = @inbounds s.seed_array[group+1]
    coord_seed = hash_combine(gs, UInt32(local_dim))
    drow = UInt32(local_dim + 1)

    i = s.counter
    @inbounds for k in eachindex(points)
        index = nested_uniform_scramble_base2(i, gs)
        v = _sobolupdate(index, drow)
        x = nested_uniform_scramble_base2(v, coord_seed)
        points[k] = to_unit(x)
        i += UInt32(1)
    end
    s.counter = i
    return points
end

@inline function nextn!(s::BurleySobolSeq, points::AbstractVector{<:AbstractFloat})
	next!(s, points)
	@. points = norminv(points)
end

@inline function nextn!(s::BurleySobolSeq,  dim::Int, points::AbstractVector{<:AbstractFloat})
	next!(s, dim, points)
	@. points = norminv(points)
end
# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

@inline Base.copy(s::BurleySobolSeq{N}) where {N} =
    BurleySobolSeq{N}(
        Base.copy(s.x),
        Base.copy(s.seed_array),
        s.counter,
    )

