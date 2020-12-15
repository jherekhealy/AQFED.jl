import Random: rand, seed!
import RandomNumbers: AbstractRNG

# Well1024a pseudo random number generator from "Improved Long-Period Generators Based on Linear Recurrences Modulo 2" f
# by Panneton and L'Ecuyer (2008)

const M1 = 3
const M2 = 24
const M3 = 10
const R = 32
const MASK_MODULO_32 = Int(0x0000001f)
mutable struct Well1024a <: AbstractRNG{UInt32}
    state::Vector{UInt32}
    stateIndex::Int
end

function Well1024a(seed::UInt32 = UInt32(20132109))
    r = Well1024a(Vector{UInt32}(undef, R), 0)
    seed!(r, seed)
end

function seed!(r::Well1024a, seed::UInt32)
    r.stateIndex = 0
    r.state[1] = seed
    @inbounds for j = 2:R
        r.state[j] = 0x6c078965 * (r.state[j-1] ⊻ (r.state[j-1] >> 30)) + (j - 1) % UInt32
    end
    r
end

@inline function rand(r::Well1024a, ::Type{UInt32})
    @inbounds begin
        sM1 = r.state[((r.stateIndex+M1)&MASK_MODULO_32)+1]
        sM2 = r.state[((r.stateIndex+M2)&MASK_MODULO_32)+1]
        sM3 = r.state[((r.stateIndex+M3)&MASK_MODULO_32)+1]
        z2 = sM2 ⊻ sM3 ⊻ (sM2 << 19) ⊻ (sM3 << 14)
        z1 = r.state[r.stateIndex+1] ⊻ sM1 ⊻ (sM1 >> 8)
        z0 = r.state[((r.stateIndex+31)&MASK_MODULO_32)+1]
        z12 = z1 ⊻ z2
        z3 = z12 ⊻ z0 ⊻ (z2 << 13) ⊻ (z0 << 11) ⊻ (z1 << 7)
        r.state[r.stateIndex+1] = z12
        r.stateIndex = (r.stateIndex + 31) & MASK_MODULO_32
        r.state[r.stateIndex+1] = z3
        z3
    end
end
