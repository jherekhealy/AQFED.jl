import Random: rand, seed!
import RandomNumbers: AbstractRNG
export AbstractRNGSeq, ZRNGSeq, next!, nextn!, skipTo

@inline rand(r::AbstractRNG{UInt64}, ::Type{Float64}) = (Float64(Random.rand(r, UInt64)  >> 12 )+0.5) / 0x1p52
#@inline rand(r::AbstractRNG{UInt64}, ::Type{Float64}) = ldexp(Float64(Random.rand(r, UInt64)  >> 11 )+0.5, ~52)

@inline rand(r::AbstractRNG{UInt32}, ::Type{Float64}) = (Float64(Random.rand(r, UInt32) )+0.5) /0x1p32

@inline function skipTo!(
    rng::AbstractRNG{UInt32},
    n::UInt64,
)
    counter = n >> 2
    offset  = n & 0x03

    set_counter!(rng, (counter, 0, 0, 0))

    for _ in 1:offset
        rand(rng, UInt32)
    end

    return rng
end
