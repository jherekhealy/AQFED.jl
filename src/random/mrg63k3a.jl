import Random: rand, seed!
import RandomNumbers: AbstractRNG


const mrg63norm = 1.0842021724855052e-19
const mrg63m1 = 9223372036854769163
const mrg63m2 = 9223372036854754679
const mrg63a12 = 1754669720
const mrg63q12 = 5256471877
const mrg63r12 = 251304723
const mrg63a13n = 3182104042
const mrg63q13 = 2898513661
const mrg63r13 = 394451401
const mrg63a21 = 31387477935
const mrg63q21 = 293855150
const mrg63r21 = 143639429
const mrg63a23n = 6199136374
const mrg63q23 = 1487847900
const mrg63r23 = 985240079
const mrg63seed = 123456789


mutable struct MRG63k3a <: AbstractRNG{UInt64}
    s::NTuple{6,Int64} #The seeds for s[1:3] must be integers in [0, m1 - 1] and not all 0.
    #The seeds for s[4:6] must be integers in [0, m2 - 1] and not all 0.
end

function MRG63k3a()
    return MRG63k3a((mrg63seed, mrg63seed, mrg63seed, mrg63seed, mrg63seed, mrg63seed))
end

@inline rand(rng::MRG63k3a, ::Type{Float64}) = rand(rng, UInt64) * mrg63norm

#63-bits of randomness
@inline function rand(rng::MRG63k3a, ::Type{UInt64})

    h = rng.s[1] ÷ mrg63q13
    p13 = mrg63a13n * (rng.s[1] - h * mrg63q13) - h * mrg63r13
    h = rng.s[2] ÷ mrg63q12
    p12 = mrg63a12 * (rng.s[2] - h * mrg63q12) - h * mrg63r12
    if (p13 < 0)
        p13 += mrg63m1
    end
    if (p12 < 0)
        p12 += mrg63m1 - p13
    else
        p12 -= p13
    end
    if (p12 < 0)
        p12 += mrg63m1
    end
    h = rng.s[4] ÷ mrg63q23
    p23 = mrg63a23n * (rng.s[4] - h * mrg63q23) - h * mrg63r23
    h = rng.s[6] ÷ mrg63q21
    p21 = mrg63a21 * (rng.s[6] - h * mrg63q21) - h * mrg63r21
    if (p23 < 0)
        p23 += mrg63m2
    end
    if (p21 < 0)
        p21 += mrg63m2 - p23
    else
        p21 -= p23
    end
    if (p21 < 0)
        p21 += mrg63m2
    end
    rng.s = (rng.s[2], rng.s[3], p12, rng.s[5], rng.s[6], p21)
    if (p12 > p21)
        return convert(UInt64, p12 - p21)
    end
    return convert(UInt64, p12 - p21 + mrg63m1)
end

# mtx1 :: JumpMatrix Int64
# mtx1 = JM (0, 1, 0) (0, 0, 1) (m1 - a13n, a12, 0)
#
# mtx2 :: JumpMatrix Int64
# mtx2 = JM (0, 1, 0) (0, 0, 1) (m2 - a23n, 0, a21)
