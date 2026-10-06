#import Random: rand!
import AQFED.Math: norminv
import Random: rand!, AbstractRNG
export AbstractSeq, ScrambledSobolSeq, next!, nextn!, scramble, skip!, skipTo!
export Owen, FaureTezuka, OwenFaureTezuka, NoScrambling, Burley, Digital

include("ssoboldata.jl") #loads `sobol_a` and `sobol_minit` from Joe & Kuo.
include("ssobolrng.jl") #original RNG and RNG adapter for scrambling
export DirectionSource, joe_kuo, niederreiter_nut, niederreiter_nut_21201, niederreiter_nut_s21201

abstract type AbstractSeq{N} end

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

#This is not Burley but Burley Inspired. Burley is 4D by 4D.
struct Burley{N} <: Scrambling
    rng::AbstractRNG
    seed_array::Vector{UInt32}
end

Burley{N}(rng::AbstractRNG) where {N} = Burley{N}(rng, zeros(UInt32, N+1))
Burley{N}() where {N} = Burley{N}(Chacha8SIMD(UInt32))

struct Digital{N} <: Scrambling
    rng::AbstractRNG
end
Digital{N}() where {N} = Digital{N}(Chacha8SIMD(UInt32))


unirand(s::S) where {S <: Union{Owen,FaureTezuka,OwenFaureTezuka}} = unirand(s.rng)


# N iis the dimension of sequence being generated, S is scrambling type
mutable struct ScrambledSobolSeq{N,S} <: AbstractSeq{N}
    v::Matrix{UInt32}
    x::Vector{UInt32}
    shift::Vector{UInt32}
    l::UInt32
    counter::UInt32
    scrambling::S
end

const INV32 = exp2(-32)
@inline to_unit(x::UInt32) = (Float64(x) + 0.5) * INV32

ndims(s::AbstractSeq{N}) where {N} = N::Int

unirand(s::ScrambledSobolSeq) = unirand(s.scrambling)

Base.copy(s::ScrambledSobolSeq{N,S}) where {N, S} = ScrambledSobolSeq{N,S}(Base.copy(s.v), Base.copy(s.x), Base.copy(s.shift), s.l, s.counter, Base.copy(s.scrambling))

copy_shallow_v(s::ScrambledSobolSeq{N,S}) where {N,S} = ScrambledSobolSeq{N,S}(s.v, Base.copy(s.x), Base.copy(s.shift), s.l, s.counter, Base.copy(s.scrambling))

Base.copy(s::Owen) = Owen(s.maxd, Base.copy(s.rng))
Base.copy(s::FaureTezuka) = FaureTezuka(Base.copy(s.rng))
Base.copy(s::OwenFaureTezuka) = OwenFaureTezuka(s.maxd, Base.copy(s.rng))
Base.copy(s::Burley) = Burley(Base.copy(s.rng), Base.copy(s.seed_array))
Base.copy(s::Digital{N}) where {N} = Digital{N}(Base.copy(s.rng))
@inline normalize(
    s::ScrambledSobolSeq{N,S},
    x::UInt32,
) where {N,S<:Union{NoScrambling,FaureTezuka}} = ldexp(Float64(x), -Int(s.l))

@inline normalize(
    s::ScrambledSobolSeq{N, S},
    x::UInt32,
) where {N, S <: Union{Burley{N}, Digital{N}}} = to_unit(x) #s.l must be 32!


@inline normalize(
    s::ScrambledSobolSeq{N,S},
    x::UInt32,
) where {N,S<:Union{Owen,OwenFaureTezuka}} = ldexp(Float64(x)+0.5, -Int(s.scrambling.maxd)) #generate in the open interval since 0.0 may appear.

#fl1 find last 1 index in value
@inline function fl1(value::Int)
	return 32 - leading_zeros(value % UInt32)
end

@inline function fl1Slow(n::Int)
	l = 0
	i = n
	while (i != 0)
		i >>= 1
		l += 1
	end
	l = min(l, 31)
end

function number_of_bits(scrambling::S, n::Int) where {S <: Scrambling}
    return n > 0 ? fl1(n + 1) : 31
end


function number_of_bits(scrambling::S, n::Int) where {S <:Union{Burley,Digital}}
    return 32
end

function ScrambledSobolSeq(dimension::Int, n::Int, scrambling::S, direction_source::DirectionSource=joe_kuo) where {S <: Scrambling}
    d = dimension
    a, minit, s_loaded = load_direction_numbers(direction_source)
    (d < 0 || d > (length(a) + 1)) && error("invalid Sobol dimension for source $direction_source (max $(length(a)+1))")
    #special cases
    if d == 0
    return ScrambledSobolSeq{0, typeof(scrambling)}(zeros(UInt32, 0, 0), UInt32[], UInt32[], zero(UInt32), zero(UInt32), scrambling)
    end

    l = number_of_bits(scrambling, n)
    x = Vector{UInt32}(undef, d)
    fill!(x, 0)
    counter = 0
    v = ones(UInt32, (d, l))
    for i = 1:l
        @inbounds v[1, i] = one(UInt32) << (l - i)
    end
    for j = 2:d
        a_coeff = a[j-1]
        # Get polynomial degree: use loaded s for Niederreiter, compute from a for Joe & Kuo
        if s_loaded !== nothing
            s = s_loaded[j-1]            
        else
            # For Joe & Kuo, a_coeff should never be 0, so this is safe
            s = a_coeff == 0 ? 0 : floor(Int, log2(a_coeff))
            a_coeff >>= 1
        end
        
        # If s is 0 or invalid, use default binary representation (like dimension 1)
        if s <= 0
            for i = 1:l
                v[j, i] = one(UInt32) << (l - i)
            end
            continue
        end
        
		#set initial values of m from table
        if l <= s
            for i = 1:l
                v[j, i] = minit[i, j-1] << (l - i)
            end
        else
            #fill in remaining values using recurrence
            for i = 1:s
                v[j, i] = minit[i, j-1] << (l - i)
            end
            for i = s+1:l
                @inbounds v[j, i] = v[j, i-s] ⊻ (v[j, i-s] >> s)
                for k = 1:s-1 
                    @inbounds v[j, i] ⊻= (((a_coeff >> (s - k - 1)) & one(UInt32)) * v[j, i-k])
                end
            end
        end
    end    
    shift = Vector{UInt32}(undef, d)
    ss = ScrambledSobolSeq{d,typeof(scrambling)}(v, x, shift, l, counter, scrambling)
    scramble(ss)
    ss
end

ScrambledSobolSeq(N::Integer) = ScrambledSobolSeq(Int(N), 0, NoScrambling())

function scramble(s::ScrambledSobolSeq{D,NoScrambling}) where {D}
    s.counter = 1 #skip! first item for non-scrambled version
    fill!(s.shift, 0)
end

function scramble(s::ScrambledSobolSeq{D,Digital{D}}) where {D}
    s.counter = 0
    fill!(s.shift, 0)
    rand!(s.scrambling.rng, s.shift)
    s.x .= s.shift
end
function scramble(s::ScrambledSobolSeq{D,Burley{D}}) where {D}
    s.counter = 0
    fill!(s.shift, 0)
    rand!(s.scrambling.rng, s.scrambling.seed_array)
end

function scramble(s::ScrambledSobolSeq{D,Owen}) where {D}
    s.counter = 0
    fill!(s.shift, 0)
    scrambleOwen(s, s.scrambling.maxd)
end

function scramble(s::ScrambledSobolSeq{D,FaureTezuka}) where {D}
    s.counter = 0
    fill!(s.shift, 0)
    scrambleTezuka(s, s.l)
end

function scramble(s::ScrambledSobolSeq{D,OwenFaureTezuka}) where {D}
    s.counter = 0
    fill!(s.shift, 0)
    scrambleOwen(s, s.scrambling.maxd)
    scrambleTezuka(s, s.scrambling.maxd)
end

function scrambleOwen(s::ScrambledSobolSeq{D}, maxd::Integer) where {D}
    if maxd < s.l
        throw(DomainError(string("maxd must be >= l where l = log2(n) = ", s.l)))
    end
    scrambledV = zeros(UInt32, (D, s.l))
    #norm = 1/2^MAXD
    lsm = genscrml(s)
    @inbounds for i = 1:D
        @inbounds for j = 1:s.l
            l = 1
            temp2 = 0
            @inbounds for p = maxd:-1:1
                temp1 = 0
                @inbounds for k = 1:s.l
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

function scrambleTezuka(s::ScrambledSobolSeq{D}, maxx::Integer) where {D}
    ushift = Vector{UInt32}(undef, s.l)
    usm = genscrmu(s, ushift)
    tv = zeros(UInt32, (s.l, maxx, D))

    @inbounds for i = 1:D
        @inbounds for j = 1:s.l
            p = maxx
            @inbounds for k = 1:maxx
                tv[j, p, i] = lbitbits(s.v[i, j], k - 1, 1)
                p -= 1
            end
        end
        @inbounds for pp = 1:s.l
            temp2 = 0
            temp4 = 0
            l = 1
            @inbounds for j = maxx:-1:1
                temp1 = 0
                temp3 = 0
                @inbounds for p = 1:s.l
                    temp1 += tv[p, j, i] * usm[pp, p]
                    if (pp == 1)
                        temp3 += tv[p, j, i] * ushift[p]
                    end
                end
                temp1 %= 2
                temp2 += temp1 * l
                if (pp == 1)
                    temp3 %= 2
                    temp4 += temp3 * l
                end
                l <<= 1
            end
            s.v[i, pp] = temp2
            if (pp == 1)
                s.shift[i] ⊻= temp4
            end
        end
    end
    s.x .= s.shift
end

function genscrml(s::ScrambledSobolSeq{D,S}) where {D,S<:Union{Owen,OwenFaureTezuka}}
    maxd = s.scrambling.maxd
    lsm = zeros(UInt32, (D, maxd))
    @inbounds for p = 1:D
        s.shift[p] = 0
        l = 1
        @inbounds for i = maxd:-1:1
            lsm[p, i] = 0
            stemp = unirand(s)
            s.shift[p] += stemp << (l - 1)
            l += 1
            ll = 1
            @inbounds for j = s.l:-1:1
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

function genscrmu(
    s::ScrambledSobolSeq{D,S},
    ushift::Vector{UInt32},
) where {D,S<:Union{FaureTezuka,OwenFaureTezuka}}
    usm = zeros(UInt32, (s.l, s.l))
    @inbounds for i = 1:s.l
        stemp = unirand(s)
        ushift[i] = stemp
        @inbounds for j = 1:s.l
            local temp
            if (j == i)
                temp = 1
            elseif (j > i)
                temp = unirand(s)
            else
                temp = 0
            end
            usm[j, i] = temp
        end
    end
    return usm
end

@inline function lbitbits(a::UInt32, b::Int, len::Int)
    x = UInt64(a)
    y = UInt64(0xffffffffffffffff) #-1
    x >>= b
    y <<= len
    return (x & ~y) % UInt32
end

@inline function reverse_bits(x::UInt32)::UInt32
    x = ((x & UInt32(0xaaaaaaaa)) >> 1) | ((x & UInt32(0x55555555)) << 1)
    x = ((x & UInt32(0xcccccccc)) >> 2) | ((x & UInt32(0x33333333)) << 2)
    x = ((x & UInt32(0xf0f0f0f0)) >> 4) | ((x & UInt32(0x0f0f0f0f)) << 4)
    x = ((x & UInt32(0xff00ff00)) >> 8) | ((x & UInt32(0x00ff00ff)) << 8)
    return (x >> 16) | (x << 16)
end

@inline function laine_karras_permutation(x::UInt32, seed::UInt32)::UInt32
    x += seed
    x = xor(x, x * UInt32(0x6c50b47c))
    x = xor(x, x * UInt32(0xb82f1e52))
    x = xor(x, x * UInt32(0xc7afe638))
    x = xor(x, x * UInt32(0x8d22f6e6))
    return x
end

@inline function nested_uniform_scramble_base2(x::UInt32, seed::UInt32)
    x = bitreverse(x)
    x = laine_karras_permutation(x, seed)
    x = bitreverse(x)
    return x
end
@inline function hash_combine(seed::T, v::T)::T where {T<:Unsigned}
    return xor(seed, v + (seed << 6) + (seed >> 2))
end

@inline function next(s::ScrambledSobolSeq{U,V}, ::Type{UInt32}) where {U,V}
    d = ndims(s)
    if s.counter == 0
        s.counter += one(s.counter)
        return s.x
    end
    c = ffz(s.counter)
    c > s.l &&
        throw(DomainError(string("counter larger than sequence length: ", s.counter)))
    s.counter += one(s.counter)
	#alternatively: @inbounds s.x .⊻= @view(s.v[:,c])
	sx = s.x
    sv = s.v
    @inbounds for j = 1:d
        sx[j] ⊻= sv[j, c]
    end
    return s.x
end

@inline function next(s::ScrambledSobolSeq{N,Burley{N}}, ::Type{UInt32}) where {N}
    d = ndims(s)
    index = nested_uniform_scramble_base2(s.counter, s.scrambling.seed_array[1])
    _skipToBinaryInternal(s, index)
    sx = s.x
    #_skipTo!BinaryInternal!(sx, s.v, s.shift, index, N)
    @inbounds for j = 1:d
        sx[j] = nested_uniform_scramble_base2(sx[j], s.scrambling.seed_array[j+1])
    end
    s.counter += one(s.counter)
    return s.x
end

@inline function next!(s::ScrambledSobolSeq{U,V}, points::P) where {U,V, P<:Union{AbstractVector{<:AbstractFloat}, AbstractMatrix{<:AbstractFloat}}}
    d = ndims(s)
    if s.counter == 0
        s.counter += one(s.counter)
        sx = s.x
        @inbounds @simd for j = 1:d
            points[j] = normalize(s, sx[j])
        end
        return points
    end
    c = ffz(s.counter)
    c > s.l && throw(DomainError(string("counter larger than sequence length: ", s.counter)))
    s.counter += one(s.counter)
    
    sx = s.x
    sv = s.v
    @inbounds @simd for j = 1:d
        val = sx[j] ⊻ sv[j, c]
        sx[j] = val
        points[j] = normalize(s, val)
    end
    return points
end

@inline function next!(s::ScrambledSobolSeq{N,Burley{N}}, points::P) where {N, P<:Union{AbstractVector{<:AbstractFloat}, AbstractMatrix{<:AbstractFloat}}}
    next(s, UInt32)
    sx = s.x
    @inbounds for j = eachindex(points)
        points[j] = normalize(s, sx[j])
    end
    return points
end

#next set of normal numbers
@inline function nextn!(s::ScrambledSobolSeq, points::AbstractVector{<:AbstractFloat})
	next!(s, points)
	@. points = norminv(points)
end

#next set of normal numbers
@inline function nextn!(s::ScrambledSobolSeq,  dim::Int, points::AbstractVector{<:AbstractFloat})
	next!(s, dim, points)
	@. points = norminv(points)
end

@inline function sobol_coordinate_at(
    s::ScrambledSobolSeq,
    j::Int,
    n::UInt32,
)
    x = s.shift[j]

    @inbounds while n != 0
        bit = trailing_zeros(n) + 1
        x ⊻= s.v[j, bit]
        n &= n - UInt32(1)
    end

    return x
end
    
@inline function next!(
    s::ScrambledSobolSeq{N,Burley{N}},
    dim::Int,
    points::AbstractVector{<:AbstractFloat},
) where {N}
    j = dim
    seeds = s.scrambling.seed_array

    @inbounds for i = 1:length(points)
        n = nested_uniform_scramble_base2(s.counter, seeds[1])
        x = s.shift[j]

        @inbounds while n != 0
            bit = trailing_zeros(n) + 1
            x ⊻= s.v[j, bit]
            n &= n - UInt32(1)
        end

        x = nested_uniform_scramble_base2(x, seeds[j + 1])
        points[i] = ldexp(Float64(x), -32)
        s.counter += one(s.counter)
    end

    return points
end

#next vector for a given dimension (vertical) from counter to counter + length(points)
@inline function next!(s::ScrambledSobolSeq, dim::Int, points::AbstractVector{<:AbstractFloat})
    j = dim
    sx = s.x
    sv = s.v
    len = length(points)
    len == 0 && return points
    
    start_idx = 1
    if s.counter == 0
        points[1] = normalize(s, sx[j])
        s.counter += one(s.counter)
        start_idx = 2
    end

    @inbounds @simd for i = start_idx:len
        c = ffz(s.counter)
        val = sx[j] ⊻ sv[j, c]
        sx[j] = val
        points[i] = normalize(s, val)
        s.counter += one(s.counter)
    end
    return points
end

@inline function skip!(s::ScrambledSobolSeq{D, Burley{D}}, n::Int) where {D}
    s.counter += n
    s
end


@inline function skip!(s::ScrambledSobolSeq{S,U}, n::Int) where {S,U}
   n == 0 && return s

    old = s.counter
    new = old + UInt32(n)

    # Gray-code difference
    g = (old ⊻ (old >> 1)) ⊻
        (new ⊻ (new >> 1))

    v = s.v
    x = s.x
    d = ndims(s)

    @inbounds while g != 0
        bit = trailing_zeros(g) + 1

        @simd for j = 1:d
            x[j] ⊻= v[j, bit]
        end

        g &= g - UInt32(1)
    end

    s.counter = new
    return s
end


@inline function skipTo!(s::ScrambledSobolSeq{D, Burley{D}}, n1::Int) where {D}
    s.counter = n1
    s
end
@inline function skipTo!(s::ScrambledSobolSeq{S,U}, n1::Int) where {S,U}
    # Convert to Gray code
    n = (n1-1) << 1 #first point with index n1=1
    g = n ⊻ (n >> 1)
    d = ndims(s)
    @inbounds for j = 1:d
        s.x[j] = s.shift[j]
        @inbounds for index = 1:s.l
            if ((g >> index) & one(UInt32)) != 0
                s.x[j] ⊻= s.v[j, index]
            end
        end
    end
    s.counter = n1
    s
end

@inline function _skipToBinaryInternalT(s::ScrambledSobolSeq{S,U}, g::UInt32) where {S,U}
    d = ndims(s)
    s.x .= s.shift
    @inbounds for j = 1:d
        @inbounds for index = 0:s.l-1
            mask = (g >> index) & one(UInt32)
                s.x[j] ⊻= mask *s.v[j, index+1]         
        end
    end
end
@inline function _skipToBinaryInternal!(
    x::Vector{UInt32},
    v::Matrix{UInt32},
    shift::Vector{UInt32},
    n::UInt32,
    d::Int)
    copyto!(x, shift)

    @inbounds while n != 0
        bit = trailing_zeros(n) + 1

        @simd for j = 1:d
            x[j] ⊻= v[j, bit]
        end

        n &= n - UInt32(1)
    end
end
@inline function _skipToBinaryInternal(s::ScrambledSobolSeq{S,U}, n1::UInt32) where {S,U}
    d = ndims(s)
    s.x .= s.shift
    mask = one(UInt32)
        @inbounds for index = 1:s.l
        if n1 & mask != zero(UInt32)
          @inbounds for j = 1:d
                s.x[j] ⊻= s.v[j, index]
            end
        end
        mask <<= 1
    end
end

@inline function skipTo!(s::ScrambledSobolSeq{D, Burley{D}}, dim::Int, n1::Int, npoints::Int=0) where {D}
   s.counter = n1
   s
end

#skip! to position n for a given dimension dim
@inline function skipTo!(s::ScrambledSobolSeq{S,U}, dim::Int, n1::Int, npoints::Int=0) where {S,U}
    # Convert to Gray code
    n = (n1-1) << 1 #first point with index n1=1
    g = n ⊻ (n >> 1)
    j = dim
    s.x[j] = s.shift[j]

    @inbounds for index = 1:s.l
        if ((g >> index) & one(UInt32)) != 0
            s.x[j] ⊻= s.v[j, index]
        end
    end
    s.counter = n1
    s
end

@inline function ffz(x::Integer)
    # if x == 0 #not used as we check before if x is 0
    #     return 1
    # end
    return trailing_zeros(x) + 1
end
