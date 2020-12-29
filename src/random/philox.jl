import Random: rand, seed!

import RandomNumbers: AbstractRNG

#@inline rand(r::Philox4x{UInt64,R}, ::Type{Float64}) where {R} = (Float64(Random.rand(r, UInt64)  >> 11 )+0.5) / 9007199254740992.0
@inline rand(r::AbstractRNG{UInt64}, ::Type{Float64}) = (Float64(Random.rand(r, UInt64)  >> 11 )+0.5) / 0x1p53
#@inline rand(r::AbstractRNG{UInt64}, ::Type{Float64}) = ldexp(Float64(Random.rand(r, UInt64)  >> 11 )+0.5, ~52)

@inline rand(r::AbstractRNG{UInt32}, ::Type{Float64}) = (Float64(Random.rand(r, UInt32) )+0.5) /0x1p32
