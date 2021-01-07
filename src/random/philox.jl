import Random: rand, seed!
import RandomNumbers: AbstractRNG
export AbstractRNGSeq, ZRNGSeq, next!, nextn!, skipTo

@inline rand(r::AbstractRNG{UInt64}, ::Type{Float64}) = (Float64(Random.rand(r, UInt64)  >> 12 )+0.5) / 0x1p52
#@inline rand(r::AbstractRNG{UInt64}, ::Type{Float64}) = ldexp(Float64(Random.rand(r, UInt64)  >> 11 )+0.5, ~52)

@inline rand(r::AbstractRNG{UInt32}, ::Type{Float64}) = (Float64(Random.rand(r, UInt32) )+0.5) /0x1p32
