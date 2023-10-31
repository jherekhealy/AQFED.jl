import Random: rand, seed!
import RandomNumbers: AbstractRNG
export MRG32k3a

const mrg32m1 = 4294967087
const mrg32m2 = 4294944443
const mrg32a12 = 1403580
const mrg32a13 = 810728
const mrg32a21 = 527612
const mrg32a23 = 1370589
const mrg32seed = 12345
const mrg32corr1 = mrg32m1 * mrg32a13
const mrg32corr2 = mrg32m2 * mrg32a23
const mrg32norm = 0x1.000000d00000bp-32

mutable struct MRG32k3a <: AbstractRNG{UInt32}
    s::NTuple{6,Int64} #The seeds for s[1:3] must be integers in [0, m1 - 1] and not all 0.
    #The seeds for s[4:6] must be integers in [0, m2 - 1] and not all 0.
end

function MRG32k3a()
    return MRG32k3a((mrg32seed, mrg32seed, mrg32seed, mrg32seed, mrg32seed, mrg32seed))
end

@inline rand(rng::MRG32k3a, ::Type{Float64}) = rand(rng, UInt32) * mrg32norm

#63-bits of randomness
@inline function rand(rng::MRG32k3a, ::Type{UInt32})
    r = rng.s[3]  - rng.s[6]
    r -= mrg32m1 * ((r - 1) >> 63)

    # Component 1 - modulo op may be slow in julia?
    p1 = (mrg32a12 * rng.s[2] - mrg32a13 * rng.s[1] + mrg32corr1) % mrg32m1 

    # Component 2
    p2 = (mrg32a21 * rng.s[6] - mrg32a23 * rng.s[4] + mrg32corr2) % mrg32m2
    rng.s = (rng.s[2], rng.s[3], p1, rng.s[5], rng.s[6], p2)
    return convert(UInt32, r)
end

# mtx1 :: JumpMatrix Int64
# mtx1 = JM (0, 1, 0) (0, 0, 1) (m1 - a13n, a12, 0)
#
# mtx2 :: JumpMatrix Int64
# mtx2 = JM (0, 1, 0) (0, 0, 1) (m2 - a23n, 0, a21)
