#import Random: rand!
import AQFED.Math: norminv

export AbstractSeq, ScrambledSobolSeq, next!, nextn!, scramble, skip, skipTo
export Owen, FaureTezuka, OwenFaureTezuka, NoScrambling

include("ssoboldata.jl") #loads `sobol_a` and `sobol_minit` from Joe & Kuo.
include("ssobolrng.jl") #original RNG and RNG adapter for scrambling

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

unirand(s::Union{Owen,FaureTezuka,OwenFaureTezuka}) = unirand(s.rng)


# N iis the dimension of sequence being generated, S is scrambling type
mutable struct ScrambledSobolSeq{N,S} <: AbstractSeq{N}
    v::Array{UInt32,2} #array of size (sdim, log2(n))
    x::Array{UInt32,1}
    shift::Array{UInt32,1}
    l::UInt32
    counter::UInt32
    scrambling::S
end


ndims(s::AbstractSeq{N}) where {N} = N::Int

unirand(s::ScrambledSobolSeq) = unirand(s.scrambling)

@inline normalize(
    s::ScrambledSobolSeq{N,S},
    x::UInt32,
) where {N,S<:Union{NoScrambling,FaureTezuka}} = ldexp(Float64(x), -s.l % Int32)
@inline normalize(
    s::ScrambledSobolSeq{N,S},
    x::UInt32,
) where {N,S<:Union{Owen,OwenFaureTezuka}} = ldexp(Float64(x), -s.scrambling.maxd % Int32)

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

function ScrambledSobolSeq(dimension::Int, n::Int, scrambling::Scrambling)
    d = dimension
    (d < 0 || d > (length(sobol_a) + 1)) && error("invalid Sobol dimension")
    #special cases
    d == 0 && return (ScrambledSobolSeq{0})(v, UInt32[], UInt32[], zero(UInt32))
    l = 31
    if (n > 0)
    	l = fl1(n+1) #+1 as we skip the first point
    end
    x = Vector{UInt32}(undef, d)
    fill!(x, 0)
    counter = 0
    v = ones(UInt32, (d, l))
    for i = 1:l
        @inbounds v[1, i] = one(UInt32) << (l - i)
    end
    for j = 2:d
        a = sobol_a[j-1]
        s = floor(Int, log2(a)) #degree of poly
		a >>= 1
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
                @inbounds v[j, i] = v[j, i-s] ⊻ (v[j, i-s] >> s)
                for k = 1:s-1 
                    @inbounds v[j, i] ⊻= (((a >> (s - k - 1)) & one(UInt32)) * v[j, i-k])
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

@inline function next(s::ScrambledSobolSeq, ::Type{UInt32})
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

#next vector of uniform [0,1) numbers at counter containing all dimensions
@inline function next!(s::ScrambledSobolSeq, points::AbstractVector{<:AbstractFloat})
    next(s, UInt32)
    sx = s.x
    @inbounds for j = 1:ndims(s)
        # if sx[j] == 0 #may happen somewhere because of scrambling
        #     points[j] = normalize(s, one(UInt32)) / 2
        # else
            points[j] = normalize(s, sx[j])
        # end
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

#next vector for a given dimension (vertical) from counter to counter + length(points)
@inline function next!(s::ScrambledSobolSeq, dim::Int, points::AbstractVector{<:AbstractFloat})
    j = dim
    sx = s.x
    sv = s.v
    @inbounds for i = 1:length(points)
        if s.counter != 0
            c = ffz(s.counter)
            sx[j] ⊻= sv[j, c]
        end
        points[i] =
        #    (sx[j] == 0) ? normalize(s, one(UInt32)) / 2 :
            normalize(s, sx[j])
        s.counter += one(s.counter)
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

@inline function ffz(x::Integer)
    # if x == 0 #not used as we check before if x is 0
    #     return 1
    # end
    return trailing_zeros(x) + 1
end
