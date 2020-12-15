import Random: rand, seed!
import RandomNumbers: AbstractRNG
using AbstractAlgebra #for skipping outside of skipMat

#####
# Mixmax Random Number Generators from "Spectral Test of the MIXMAX Random Number Generators" (2018) by Narek Martirosyan, Konstantin Savvidy and George Savvidy.
# In particular, this implements the N=17 and N=240 versions.
#
# Skipping ahead is possible, and relatively fast (~140 rand calls with nSkip=1000000) for N=17, thanks to a cache for nSkip < 16*2^30.
# The intent of skipping here is not to retrieve far away independent substreams, but to split a stream in blocks.
#
const M61 = UInt64(2305843009213693951)
const BITS = 61

mutable struct Mixmax{N,SPECIALMUL,SPECIAL} <: AbstractRNG{UInt64}
    v::Vector{UInt64}
    counter::Int
    sumtot::UInt64
    #skipMat::Array{UInt64}(undef, (128,17))
end

export Mixmax, Mixmax17, Mixmax240

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
    r.counter = 2
    r.sumtot = modMersenne(modMersenne(sumtot) + (ovflow << 3))
end

@inline rand(r::Mixmax, ::Type{Float64}) = Float64(rand(r, UInt64)) / M61

#actually 61 lowest bits
@inline rand(r::Mixmax, ::Type{UInt64}) = mixmax_get(r)


@inline function mixmax_get(r::Mixmax{N,SPECIALMUL,SPECIAL}) where {N,SPECIALMUL,SPECIAL}
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

@inline modMulSpec(r::Mixmax{240,SPECIALMUL,SPECIAL}, k::UInt64) where {SPECIALMUL,SPECIAL} =
    fmodmulM61(UInt64(0), UInt64(SPECIAL), k)

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
    s1 = ((s % UInt64) & M61) + (((s >> 64) % UInt64) * 8) + ((s % UInt64) >> BITS)
    return modMersenne(s1)
end

@inline function fmodmulM61(cum::UInt64, a::UInt64, b::UInt64)::UInt64
    temp = UInt128(a) * UInt128(b) + cum
    return mod128(temp)
end

@inline function skip(gen::Mixmax{N,SPECIALMUL,SPECIAL}, s::UInt64) where {N,SPECIALMUL,SPECIAL}
    n = s + gen.counter - 2
    k = trunc(UInt64, n / (N - 1))
    rk = n - k * (N - 1)
    nReal = k * N + rk + 1 #we skip one every 17.nReal is the effective n without skipping
    nMat = trunc(UInt64, nReal / N)
    #binary decomp of nMat
    _skipInternal(gen, nMat)
    gen.sumtot = 0
    for i = 1:N
        gen.sumtot = modadd(gen.sumtot, gen.v[i])
    end
    gen.counter = rk + 2
    gen
end

@inline function _skipInternal(gen::Mixmax{N,SPECIALMUL,SPECIAL}, nMat::UInt64) where {N,SPECIALMUL,SPECIAL}
    iStart = ceil(Int, log2(N))
    for i = 0:(iStart-1)
        if (nMat >> i) & UInt64(1) == UInt64(1)
            m = UInt64(1) << i
            for j=1:m
                iterateRawVec(gen)
            end
        end
    end
    F=GF(2^61-1)
    R,y = PolynomialRing(F,"y")
    local polyCoeffs
    if N == 17
        polyCoeffs = polyCoeffs17
    elseif N==240
        polyCoeffs = polyCoeffs240
    else
        throw(DomainError(string("coefficients for characteristic polynomial with N=",N," not found")))
    end
    cp = R(F.(polyCoeffs))
    eArray = zeros(Int, N)
    gCoeffs = zeros(F, 2^iStart)
    gCoeffs[2^(iStart-1)+1] = F(1)
    g = R(gCoeffs)
    iEnd = floor(Int,log2(nMat))
    @inbounds for i = iStart:iEnd
        g = g^2 % cp
        if (nMat >> i) & UInt64(1) == UInt64(1)
            #println("i ",i, " end ",iEnd," nMat ",nMat )
            #eArray = skipMat17[i-4]
             @inbounds for j=1:N
                  eArray[j] = g.coeffs[j].d
             end
            _skipWithCoeffs(gen, eArray)
        end
    end
end

@inline function _skipInternal(gen::Mixmax{17,SPECIALMUL,SPECIAL}, nMat::UInt64) where {SPECIALMUL,SPECIAL}
for i = 0:63
    if (nMat >> i) & UInt64(1) == UInt64(1)
        #println("i ",i, " nMat ",nMat )
        skipTwoPower(gen, i)
    end
end
end

@inline function skipTwoPower(gen::Mixmax{17,SPECIALMUL,SPECIAL}, m::Int64) where {SPECIALMUL,SPECIAL}
    if m <= 4 #2^(4+1) > N
        m2 = UInt64(1) << m
        for i = 1:m2
            iterateRawVec(gen)
        end
    else
        local skipMat = skipMat17
        if m - 4 > length(skipMat)
            throw(DomainError(string("Can not skip, 2^m too large where m=", m)))
        end
        eArray = skipMat17[m-4]
        _skipWithCoeffs(gen, eArray)
    end
end

@inline function _skipWithCoeffs(gen::Mixmax{N,SPECIALMUL,SPECIAL},eArray::Array{Int}) where {N, SPECIALMUL,SPECIAL}
    a2p = zeros(UInt64, N)
    @inbounds for i = 1:N
        @inbounds for j = 1:N
            a2p[j] = fmodmulM61(a2p[j], UInt64(eArray[i]), gen.v[j])
        end
        iterateRawVec(gen)
    end
    gen.v = a2p[1:N]
    gen.sumtot = 0
    for i = 1:N
        gen.sumtot = modadd(gen.sumtot, gen.v[i])
    end
end

function computeMixmaxCharPoly(gen::Mixmax{N,SPECIALMUL,SPECIAL}) where {N,SPECIALMUL,SPECIAL}
    # M=makeAsonovRealMatrix(Int64, 240, 271828282, 2^32+1)
    # M=makeAsonovRealMatrix(Int64, 240, 487013230256099140, 2^51+1)
    # M=makeAsonovRealMatrix(Int64, 256, 487013230256099064, 1)
    # M =makeAsonovRealMatrix(BigInt, 17, 0, 2^36+1)
    M =makeAsonovMatrix(BigInt, N, SPECIAL, 2^SPECIALMUL + 1)
    F=GF(2^61-1)
    R,y = PolynomialRing(F,"y")
    g = charpoly(R, matrix(F,M))
    return g
end

function makeAsonovMatrix(C, N::Int, s::T, m::T) where {T}
    A = zeros(C,N,N)
    for i=1:N
        for j=1:N
            A[i,j] += 1
        end
    end
    for i=2:N
        for j = 2:i
            A[i,j] = (i-j)*m + 2
        end
    end
    A[3,2] += s
    return A
end

include("mixmax17_skip.jl")
include("mixmax240_skip.jl")
