import Random: rand, seed!
import RandomNumbers: AbstractRNG

const M61 = UInt64(2305843009213693951)
const BITS = 61
const INV_MERSBASE = 0.4336808689942017736029811203479766845703E-18

mutable struct Mixmax{N,SPECIALMUL,SPECIAL} <: AbstractRNG{UInt64}
    v::Vector{UInt64}
    counter::Int
    sumtot::UInt64
    #skipMat::Array{UInt64}(undef, (128,17))
end

function Mixmax{N,SPECIALMUL,SPECIAL}(seed::UInt64 = UInt64(123)) where {N,SPECIALMUL,SPECIAL}
    r = Mixmax{N,SPECIALMUL,SPECIAL}(Vector{UInt64}(undef, N), 1, UInt64(0))
    seed!(r, seed)
    return r
end

function Mixmax17(seed::UInt64 = UInt64(123))
    return Mixmax{17,36,0}(seed)
end


function Mixmax240(seed::UInt64 = UInt64(123))
    return Mixmax{240,32,271828282}(seed)
end

function seed!(r::Mixmax{N,SPECIALMUL,SPECIAL}, seed::UInt64) where {N,SPECIALMUL,SPECIAL}
    sumtot = UInt64(0)
    ovflow = UInt64(0)
    if seed == 0
        throw(DomainError("A non-zero seed is required"))
    end
    l = seed

    for i = 1:N
        l *= UInt64(6364136223846793005)
        l = (l << 32) ⊻ (l >> 32)
        r.v[i] = l & M61
        sumtot += r.v[i]
        if sumtot < r.v[i]
            ovflow + 1
        end
    end
    r.counter = 1
    r.sumtot = modMersenne(modMersenne(sumtot) + (ovflow << 3))
end

@inline rand(r::Mixmax, ::Type{Float64}) = Float64(rand(r, UInt64)) * INV_MERSBASE

#actually 61 lowest bits
@inline rand(r::Mixmax, ::Type{UInt64}) = mixmax_get(r)


@inline function mixmax_get(r::Mixmax{N,SPECIALMUL,SPECIAL})  where {N,SPECIALMUL,SPECIAL}
    v = r.v
    if r.counter > N
        iterateRawVec(r)
        r.counter = 3
        return v[2] #first number skipped on purpose
    end
        result = r.v[r.counter]
        r.counter += 1
        result
end

@inline function iterateRawVec(r::Mixmax{N,SPECIALMUL,SPECIAL}) where {N,SPECIALMUL,SPECIAL}
    v = r.v
    temp2 = v[2]
    tempV = r.sumtot
    v[1] = tempV
    sumtotNew = v[1]
    ovflow = UInt64(0)
    tempP = UInt64(0)
    @inbounds for i = 2:N
        tempPO = mulWU(tempP, SPECIALMUL)
        tempP = modadd(tempP, v[i])
        tempV = modMersenne(tempV + tempP + tempPO)
        v[i] = tempV
        sumtotNew += tempV
        if (sumtotNew < tempV)
            ovflow += 1
        end
    end
    temp2 = modMulSpec(r, temp2)
    v[3] = modadd(v[3], temp2)
    sumtotNew += temp2
    if sumtotNew < temp2
        ovflow += 1
    end
    r.sumtot = modMersenne(modMersenne(sumtotNew) + (ovflow << 3))
end

@inline modMulSpec(r::Mixmax{240,SPECIALMUL,SPECIAL}, k::UInt64) where {SPECIALMUL,SPECIAL} = fmodmulM61(UInt64(0),UInt64(SPECIAL),k)

@inline function iterateRawVec(r::Mixmax{N,SPECIALMUL,0}) where {N,SPECIALMUL}
    v = r.v
    tempV = r.sumtot
    v[1] = tempV
    sumtotNew = v[1]
    ovflow = UInt64(0)
    tempP = UInt64(0)
    @inbounds for i = 2:N
        tempPO = mulWU(tempP, SPECIALMUL)
        tempP = modadd(tempP, v[i])
        tempV = modMersenne(tempV + tempP + tempPO)
        v[i] = tempV
        sumtotNew += tempV
        if (sumtotNew < tempV)
            ovflow += 1
        end
    end
    r.sumtot = modMersenne(modMersenne(sumtotNew) + (ovflow << 3))
end

@inline function modMersenne(k::UInt64)::UInt64
    return (k & M61) + (k >> BITS)
end

@inline function modadd(foo::UInt64, bar::UInt64)::UInt64
    k = (foo + bar)
    return modMersenne(k)
end

@inline function mulWU(k::UInt64, SPECIALMUL::Int)::UInt64
    return (((k) << (SPECIALMUL) & M61) | ((k) >> (BITS - SPECIALMUL)))
end

@inline function mod128(s::UInt128)::UInt64
    s1 = ((s % UInt64) & M61) + (((s >> 64)%UInt64) * 8) + ((s % UInt64) >> BITS)
    return modMersenne(s1)
end

@inline function fmodmulM61(cum::UInt64, a::UInt64, b::UInt64)::UInt64
    temp = UInt128(a) * UInt128(b) + cum
    return mod128(temp)
end
