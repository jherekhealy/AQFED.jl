import Random: rand, seed!
import RandomNumbers: AbstractRNG #could redefine if best abstract type AbstractRNG{T<:Number} <: Random.AbstractRNG end

# This is a 64-bit version of the Mersenne-Twister Random Number Generator from
# T. Nishimura, "Tables of 64-bit Mersenne Twisters" ACM Transactions on Modeling and Computer Simulation 10. (2000) 348--357.
# this is the ID1 version, the same as the reference C code provided by T. Nishimura.

const NN = 312
const MM = 156
const MATRIX_A = 0xB5026F5AA96619E9
const UM = 0xFFFFFFFF80000000
const LM = 0x000000007FFFFFFF

mutable struct MersenneTwister64 <: AbstractRNG{UInt64}
    mt::Vector{UInt64}
    mti::Int
    function MersenneTwister64(x::Vector{UInt64}, i::Int)
        @assert length(x) == NN
        new(x, i)
    end
end

function MersenneTwister64(seed = UInt64(5489))
    r = MersenneTwister64(Vector{UInt64}(undef, NN), 1)
    seed!(r, seed)
end

function seed!(r::MersenneTwister64, seed::UInt64)
    r.mt[1] = seed
    for mti = 2:NN
        r.mt[mti] = (UInt64(6364136223846793005) * (r.mt[mti-1] ⊻ (r.mt[mti-1] >> 62)) + mti-1)
    end
    r.mti = 1 #for incr
    #r.mti = NN+1 #forbloack
    r
end

function seed!(r::MersenneTwister64, init_key::Array{UInt64})
    seed!(r, 19650218 % UInt64)
    i = 2
    j = 1
    k = NN > length(init_key) ? NN : length(init_key)
    mt = r.mt
    while (k != 0)
        mt[i] = (mt[i] ⊻ ((mt[i-1] ⊻ (mt[i-1] >> 62)) * 3935559000370003845)) + init_key[j] + (j - 1)
        i += 1
        j += 1
        if i > NN
            mt[1] = mt[NN]
            i = 2
        end
        if j > length(init_key)
            j = 1
        end
        k -= 1
    end
    k = NN - 1
    while (k != 0)
        mt[i] = (mt[i] ⊻ ((mt[i-1] ⊻ (mt[i-1] >> 62)) * 2862933555777941757)) - (i - 1)
        i += 1
        if (i > NN)
            mt[1] = mt[NN]
            i = 2
        end
        k -= 1
    end

    mt[1] = UInt64(1) << 63
    r.mti = 1 #for incr
    #r.mti = NN+1 #for block
    r
end

@inline mtMagic(y::UInt64) = ((y % Int64) << 63 >> 63) & MATRIX_A
#const mag01 = [UInt64(0), MATRIX_A]
#equivalent to @inline mtMagic(y::UInt64) = mag01[((y&1)%Int) + 1]

@inline function mtNext(r::MersenneTwister64)
    #mti should be init to NN+1 to start with block
    mt = r.mt
    if r.mti > NN
        @inbounds for mti=1:NN-MM
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] = mt[mti+MM] ⊻ (x >> 1) ⊻ mtMagic(x)
        end
        @inbounds for mti=NN-MM+1:NN-1
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] = mt[mti+(MM-NN)] ⊻ (x >> 1) ⊻ mtMagic(x)
        end
        x = (mt[NN] & UM) | (mt[1] & LM)
        mt[NN] = mt[MM] ⊻ (x >> 1) ⊻ mtMagic(x)
        r.mti = 1
    end
    x = mt[r.mti]
    r.mti+=1
    return x
end

@inline function mtNextIncremental(r::MersenneTwister64)
    mti = r.mti
    mt = r.mt
    local x
    @inbounds begin
        if mti <= NN - MM
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] = mt[mti+MM] ⊻ (x >> 1) ⊻ mtMagic(x)
            x = mt[mti]
            r.mti += 1
        elseif mti < NN
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] = mt[mti+(MM-NN)] ⊻ (x >> 1) ⊻ mtMagic(x)
            x = mt[mti]
            r.mti += 1
        else
            x = (mt[NN] & UM) | (mt[1] & LM)
            mt[NN] = mt[MM] ⊻ (x >> 1) ⊻ mtMagic(x)
            x = mt[NN]
            r.mti = 1
        end
    end
    return x
end

@inline function temper(x::UInt64)::UInt64
    x ⊻= (x >> 29) & 0x5555555555555555
    x ⊻= (x << 17) & 0x71D67FFFEDA60000
    x ⊻= (x << 37) & 0xFFF7EEE000000000
    x ⊻= (x >> 43)
    x
end

@inline function rand(r::MersenneTwister64, ::Type{UInt64})
    x = mtNextIncremental(r) #actually a little bit faster
    x = temper(x)
    x
end
