import Random: rand, seed!
import RandomNumbers: AbstractRNG #could redefine if best abstract type AbstractRNG{T<:Number} <: Random.AbstractRNG end

# This is a 64-bit version of the Mersenne-Twister Random Number Generator from
# T. Nishimura, "Tables of 64-bit Mersenne Twisters" ACM Transactions on Modeling and Computer Simulation 10. (2000) 348--357.
# this is the ID1 version, the same as the reference C code provided by T. Nishimura.

const NN = 312
const ID1_MM = 156
const ID1_MATRIX_A = 0xB5026F5AA96619E9
const UM = 0xFFFFFFFF80000000
const LM = 0x000000007FFFFFFF

const ID3_M0 = 63
const ID3_M1 = 151
const ID3_M2 = 224
const ID3_MATRIX_A = 0xB3815B624FC82E2F
const ID3_MASK_B = 0x599CFCBFCA660000
const ID3_MASK_C = 0xFFFAAFFE00000000
const ID3_UU = 26
const ID3_SS = 17
const ID3_TT = 33
const ID3_LL = 39

const ID1 = Val{:ID1}
const ID3 = Val{:ID3}

const IDS = Union{ID1,ID3}


mutable struct MersenneTwister64{T} <: AbstractRNG{UInt64}
    mt::Vector{UInt64}
    mti::Int
    function MersenneTwister64{T}(x::Vector{UInt64}, i::Int) where {T <: IDS}
        @assert length(x) == NN
        new(x, i)
    end
end


function MersenneTwister64(seed = UInt64(5489))
    r = MersenneTwister64{ID1}(Vector{UInt64}(undef, NN), 1)
    seed!(r, seed)
end


function MersenneTwister64ID3(seed = UInt64(5489))
    r = MersenneTwister64{ID3}(Vector{UInt64}(undef, NN), 1)
    seed!(r, seed)
end

function seed!(r::MersenneTwister64{T}, seed::UInt64) where {T}
    r.mt[1] = seed
    for mti = 2:NN
        r.mt[mti] =
            (UInt64(6364136223846793005) * (r.mt[mti-1] ⊻ (r.mt[mti-1] >> 62)) + mti - 1)
    end
    r.mti = 1 #for incr
    #r.mti = NN+1 #forbloack
    r
end

function seed!(r::MersenneTwister64{T}, init_key::Array{UInt64}) where {T}
    seed!(r, 19650218 % UInt64)
    i = 2
    j = 1
    k = NN > length(init_key) ? NN : length(init_key)
    mt = r.mt
    while (k != 0)
        mt[i] =
            (mt[i] ⊻ ((mt[i-1] ⊻ (mt[i-1] >> 62)) * 3935559000370003845)) +
            init_key[j] +
            (j - 1)
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

@inline mtMagicID1(y::UInt64) = ((y % Int64) << 63 >> 63) & ID1_MATRIX_A
#const mag01 = [UInt64(0), ID1_MATRIX_A]
#equivalent to @inline mtMagic(y::UInt64) = mag01[((y&1)%Int) + 1]

@inline mtMagicID3(y::UInt64) = ((y % Int64) << 63 >> 63) & ID3_MATRIX_A

@inline function mtNext(r::MersenneTwister64{ID1})
    #mti should be init to NN+1 to start with block
    mt = r.mt
    if r.mti > NN
        @inbounds for mti = 1:NN-ID1_MM
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] = mt[mti+ID1_MM] ⊻ (x >> 1) ⊻ mtMagicID1(x)
        end
        @inbounds for mti = NN-ID1_MM+1:NN-1
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] = mt[mti+(ID1_MM-NN)] ⊻ (x >> 1) ⊻ mtMagicID1(x)
        end
        x = (mt[NN] & UM) | (mt[1] & LM)
        mt[NN] = mt[ID1_MM] ⊻ (x >> 1) ⊻ mtMagicID1(x)
        r.mti = 1
    end
    x = mt[r.mti]
    r.mti += 1
    return x
end

@inline function mtNextIncremental(r::MersenneTwister64{ID1})
    mti = r.mti
    mt = r.mt
    local x
    @inbounds begin
        if mti <= NN - ID1_MM
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] = mt[mti+ID1_MM] ⊻ (x >> 1) ⊻ mtMagicID1(x)
            x = mt[mti]
            r.mti += 1
        elseif mti < NN
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] = mt[mti+(ID1_MM-NN)] ⊻ (x >> 1) ⊻ mtMagicID1(x)
            x = mt[mti]
            r.mti += 1
        else
            x = (mt[NN] & UM) | (mt[1] & LM)
            mt[NN] = mt[ID1_MM] ⊻ (x >> 1) ⊻ mtMagicID1(x)
            x = mt[NN]
            r.mti = 1
        end
    end
    return x
end

@inline function mtNext(r::MersenneTwister64{ID3})
    #mti should be init to NN+1 to start with block
    mt = r.mt
    if r.mti > NN
        @inbounds for mti = 1:NN-ID3_M2
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] =
                mt[mti+ID3_M0] ⊻ mt[mti+ID3_M1] ⊻ mt[mti+ID3_M2] ⊻ (x >> 1) ⊻ mtMagicID3(x)
        end
        @inbounds for mti = NN-ID2_M2+1:NN-ID3_M1
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] =
                mt[mti+ID3_M0] ⊻ mt[mti+ID3_M1] ⊻ mt[mti+ID3_M2-NN] ⊻ (x >> 1) ⊻
                mtMagicID3(x)
        end
        @inbounds for mti = NN-ID2_M1+1:NN-ID3_M0
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] =
                mt[mti+ID3_M0] ⊻ mt[mti+ID3_M1-NN] ⊻ mt[mti+ID3_M2-NN] ⊻ (x >> 1) ⊻
                mtMagicID3(x)
        end
        @inbounds for mti = NN-ID2_M0+1:NN-1
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] =
                mt[mti+ID3_M0-NN] ⊻ mt[mti+ID3_M1-NN] ⊻ mt[mti+ID3_M2-NN] ⊻ (x >> 1) ⊻
                mtMagicID3(x)
        end
        x = (mt[NN] & UM) | (mt[1] & LM)
        mt[NN] = mt[ID3_M0] ⊻ mt[ID3_M1] ⊻ mt[ID3_M2] ⊻ (x >> 1) ⊻ mtMagicID3(x)
        r.mti = 1
    end
    x = mt[r.mti]
    r.mti += 1
    return x
end

@inline function mtNextIncremental(r::MersenneTwister64{ID3})
    mti = r.mti
    mt = r.mt
    local x
    @inbounds begin
        if (mti <= NN - ID3_M2)
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] = (x >> 1) ⊻ mtMagicID3(x)
            mt[mti] ⊻= mt[mti+ID3_M0] ⊻ mt[mti+ID3_M1] ⊻ mt[mti+ID3_M2]
            x = mt[mti]
            r.mti += 1
        elseif (mti <= NN - ID3_M1)
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] = (x >> 1) ⊻ mtMagicID3(x)
            mt[mti] ⊻= mt[mti+ID3_M0] ⊻ mt[mti+ID3_M1] ⊻ mt[mti+ID3_M2-NN]
            x = mt[mti]
            r.mti += 1
        elseif (mti <= NN - ID3_M0)
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] = (x >> 1) ⊻ mtMagicID3(x)
            mt[mti] ⊻= mt[mti+ID3_M0] ⊻ mt[mti+ID3_M1-NN] ⊻ mt[mti+ID3_M2-NN]
            x = mt[mti]
            r.mti += 1
        elseif (mti < NN)
            x = (mt[mti] & UM) | (mt[mti+1] & LM)
            mt[mti] = (x >> 1) ⊻ mtMagicID3(x)
            mt[mti] ⊻= mt[mti+ID3_M0-NN] ⊻ mt[mti+ID3_M1-NN] ⊻ mt[mti+ID3_M2-NN]
            x = mt[mti]
            r.mti += 1
        else
            x = (mt[NN] & UM) | (mt[1] & LM)
            mt[NN] = (x >> 1) ⊻ mtMagicID3(x)
            mt[NN] ⊻= mt[ID3_M0] ⊻ mt[ID3_M1] ⊻ mt[ID3_M2]
            x = mt[NN]
            r.mti = 1
        end
    end
    return x
end
@inline function temper(::Type{ID1}, x::UInt64)::UInt64
    x ⊻= (x >> 29) & 0x5555555555555555
    x ⊻= (x << 17) & 0x71D67FFFEDA60000
    x ⊻= (x << 37) & 0xFFF7EEE000000000
    x ⊻= (x >> 43)
    x
end

@inline function temper(::Type{ID3}, x::UInt64)::UInt64
    x ⊻= (x >> ID3_UU)
    x ⊻= (x << ID3_SS) & ID3_MASK_B
    x ⊻= (x << ID3_TT) & ID3_MASK_C
    x ⊻= (x >> ID3_LL)
    x
end

@inline function rand(r::MersenneTwister64{T}, ::Type{UInt64}) where {T}
    x = mtNextIncremental(r) #actually a little bit faster
    x = temper(T, x)
    x
end
