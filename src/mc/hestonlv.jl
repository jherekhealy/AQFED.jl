import AQFED.Math: norminv
import AQFED.TermStructure: LocalVolatilityModel, HestonModel, discountFactor, logForward, discreteLocalVarianceByLogmoneyness, DiscreteLocalVarianceSlice
import AQFED.Random: next!, nextn!, skipTo!, Chacha8SIMD, AbstractRNGSeq
using Statistics
using PPInterpolation

export StochasticLocalVolatilityModel, HestonLVModel, SLVLeverage, calibrateLeverage, conditionalVariance, simulateQE
export BinCalculator, BinSplineCalculator, NonConformCalculator, CompactSupportCalculator, LocalLinearCalculator

#A stochastic volatility model with a leverage L(t,S) such that it reproduces the vanilla prices of a local volatility model,
#L² = σ²_LV(t,S)/E[V|S_t=S] where V is the instantaneous variance. Fields: lv::LocalVolatilityModel, accessor stochasticVolatilityModel.
abstract type StochasticLocalVolatilityModel end

#Heston stochastic local volatility model: d ln S = (r-q-L²V/2)dt + L(t,S)√V dW, with V the Heston variance.
struct HestonLVModel{H<:HestonModel,L<:LocalVolatilityModel} <: StochasticLocalVolatilityModel
    heston::H
    lv::L
end
stochasticVolatilityModel(model::HestonLVModel) = model.heston

variatesPerStep(::HestonLVModel, ::QE) = 2
GridConfig(model::StochasticLocalVolatilityModel, specTimes::Vector{Float64}, timestepSize::Float64, method::PathGenerator; kwargs...) =
    GridConfig(stochasticVolatilityModel(model), specTimes, timestepSize, method; kwargs...)
ndims(model::HestonLVModel, specificTimes::Vector{Float64}, timestepSize::Float64) =
    ndims(model, GridConfig(model, specificTimes, timestepSize, QE()))

# ============================================================
# Estimators of x ↦ E[V | ln S = x] from sorted particles
# ============================================================

abstract type LocalComponentCalculator end
abstract type ConditionalVariance end

struct ConstantConditionalVariance{T} <: ConditionalVariance
    value::T
end
@inline (f::ConstantConditionalVariance)(x) = f.value

#constant time search of the last node <= z among sorted nodes, by a uniform table of the first candidate node:
#a binary search on the particle quantiles would be dominated by branch mispredictions in the path loops.
struct NodeLocator{T}
    x::Vector{T}
    xMin::T
    invDx::T
    first::Vector{Int32}
end

function NodeLocator(x::Vector{T}; cellsPerNode::Int=4) where {T}
    n = length(x)
    nCells = max(1, cellsPerNode * n)
    xMin = x[1]
    dx = (x[n] - xMin) / nCells
    invDx = dx > 0 ? 1 / dx : zero(T)
    first = Vector{Int32}(undef, nCells + 1)
    i = 1
    for c = 1:nCells+1
        z = xMin + (c - 1) * dx
        while i < n && x[i+1] <= z
            i += 1
        end
        first[c] = clamp(i, 1, max(n - 1, 1))
    end
    return NodeLocator(x, xMin, invDx, first)
end

#for x[1] <= z <= x[end], returns i in [1,n-1] (or 1 if n == 1) such that x[i] <= z < x[i+1], except at z = x[end].
@inline function locate(l::NodeLocator, z)
    x = l.x
    n = length(x)
    c = clamp(unsafe_trunc(Int, (z - l.xMin) * l.invDx), 0, length(l.first) - 1) + 1
    @inbounds i = Int(l.first[c])
    @inbounds while i > 1 && x[i] > z #only on rounding of the cell
        i -= 1
    end
    @inbounds while i < n - 1 && x[i+1] <= z
        i += 1
    end
    return i
end

#piecewise constant, value of the last node <= x.
struct StepConditionalVariance{T} <: ConditionalVariance
    locator::NodeLocator{T}
    value::Vector{T}
end
StepConditionalVariance(x::Vector{T}, value::Vector{T}) where {T} = StepConditionalVariance(NodeLocator(x), value)
@inline function (f::StepConditionalVariance)(x)
    l = f.locator
    @inbounds if x <= l.x[1]
        return f.value[1]
    elseif x >= l.x[end]
        return f.value[end]
    end
    return @inbounds f.value[locate(l, x)]
end

#piecewise linear ordinate + slope*x on [x[i],x[i+1]) (the last segment up to xMax), flat extrapolation outside [x[1],xMax].
struct LinearConditionalVariance{T} <: ConditionalVariance
    locator::NodeLocator{T}
    ordinate::Vector{T}
    slope::Vector{T}
    xMax::T
end
LinearConditionalVariance(x::Vector{T}, ordinate::Vector{T}, slope::Vector{T}, xMax::T) where {T} =
    LinearConditionalVariance(NodeLocator(x), ordinate, slope, xMax)
@inline function (f::LinearConditionalVariance)(x)
    l = f.locator
    z = clamp(x, l.xMin, f.xMax)
    i = z >= l.x[end] ? length(l.x) : locate(l, z)
    return @inbounds muladd(f.slope[i], z, f.ordinate[i])
end

#cubic spline with flat extrapolation, of E[V|x] or of 1/E[V|x] when isInverse.
struct SplineConditionalVariance{P,T} <: ConditionalVariance
    pp::P
    locator::NodeLocator{T}
    yMin::T
    yMax::T
    isInverse::Bool
end
@inline function (f::SplineConditionalVariance)(x)
    l = f.locator
    @inbounds y = if x <= l.x[1]
        f.yMin
    elseif x >= l.x[end]
        f.yMax
    else
        PPInterpolation.evaluate(f.pp, locate(l, x), x)
    end
    return f.isInverse ? inv(y) : y
end

function SplineConditionalVariance(x::Vector{T}, y::Vector{T}; isInverse::Bool=false) where {T}
    if length(x) < 2
        isempty(x) && throw(DomainError(0, "no particle to estimate the conditional variance"))
        #a two-node constant spline keeps the same type
        x = [x[1], x[1] + one(T)]
        y = [y[1], y[1]]
    end
    pp = makeCubicPP(x, y, PPInterpolation.FIRST_DERIVATIVE, zero(T), PPInterpolation.FIRST_DERIVATIVE, zero(T), PPInterpolation.HuynRational())
    return SplineConditionalVariance(pp, NodeLocator(x), y[1], y[end], isInverse)
end

#index of the last x[i] <= z searched from fromIndex (fromIndex if z <= x[fromIndex]).
function locateLowerIndex(x::AbstractVector, z, fromIndex::Int=1)
    @inbounds if z <= x[fromIndex]
        return fromIndex
    end
    return max(fromIndex, searchsortedlast(x, z, fromIndex + 1, length(x), Base.Order.Forward))
end

@inline function quarticKernelValue(x)
    u = 1 - x * x
    return u <= 0 ? zero(u) : u * u * 15 / 16
end

@inline function inverseQuarticKernelValue(x)
    b = 1 - sqrt(16 * x / 15)
    return b <= 0 ? one(b) : sqrt(b)
end

#gridSize bins of equal number of particles, the average variance of a bin applies from its first particle.
Base.@kwdef struct BinCalculator <: LocalComponentCalculator
    gridSize::Int = 64
end

#gridSize bins of equal number of particles, Huynh rational spline of the bin average variances at the bin median particles.
Base.@kwdef struct BinSplineCalculator <: LocalComponentCalculator
    gridSize::Int = 16
end

#non-conforming piecewise linear least squares regression on gridSize bins of equal number of particles, where a
#segment that is not positive at its ends is replaced by the bin average.
Base.@kwdef struct NonConformCalculator <: LocalComponentCalculator
    gridSize::Int = 32
end

#Nadaraya-Watson estimator with a quartic kernel of bandwidth bandwidthFactor*h*n^(-1/5), at max(maxGridSize, gridSize*√t)
#particle quantiles, interpolated by a Huynh rational spline of 1/E[V|x]. threshold truncates the kernel support.
Base.@kwdef struct CompactSupportCalculator <: LocalComponentCalculator
    gridSize::Int = 1
    maxGridSize::Int = 32
    bandwidthFactor::Float64 = 1.5
    threshold::Float64 = 1e-4
end

#local linear kernel regression with a quartic kernel at min(maxGridSize, gridSize*√t) particle quantiles, linearly
#interpolated. Nodes with less than minEffectiveParticles in their kernel window are skipped.
Base.@kwdef struct LocalLinearCalculator <: LocalComponentCalculator
    gridSize::Int = 50
    maxGridSize::Int = 100
    bandwidthFactor::Float64 = 2.0
    minEffectiveParticles::Int = 20
end

#x: particle log-spots in ascending order, v: their variances. bandwidth: a typical standard deviation of x.
function conditionalVariance(calc::BinCalculator, x::AbstractVector{T}, v::AbstractVector{T}, bandwidth, sqrtTime) where {T}
    n = length(x)
    gridSize = min(calc.gridSize, n)
    f = Vector{T}(undef, gridSize)
    e = Vector{T}(undef, gridSize)
    @inbounds for k = 1:gridSize
        iStart = ((k - 1) * n) ÷ gridSize + 1
        iEnd = (k * n) ÷ gridSize
        f[k] = x[iStart]
        sum = zero(T)
        for i = iStart:iEnd
            sum += max(v[i], zero(T))
        end
        e[k] = sum * gridSize / n
    end
    return StepConditionalVariance(f, e)
end

function conditionalVariance(calc::BinSplineCalculator, x::AbstractVector{T}, v::AbstractVector{T}, bandwidth, sqrtTime) where {T}
    n = length(x)
    gridSize = min(calc.gridSize, n)
    f = Vector{T}(undef, gridSize)
    e = Vector{T}(undef, gridSize)
    @inbounds for k = 1:gridSize
        iStart = ((k - 1) * n) ÷ gridSize
        iNext = (k * n) ÷ gridSize
        f[k] = x[(iStart+iNext)÷2+1]
        sum = zero(T)
        for i = iStart+1:iNext
            sum += max(v[i], zero(T))
        end
        e[k] = sum * gridSize / n
    end
    return SplineConditionalVariance(f, e)
end

function conditionalVariance(calc::NonConformCalculator, x::AbstractVector{T}, v::AbstractVector{T}, bandwidth, sqrtTime) where {T}
    n = length(x)
    nNodes = min(calc.gridSize, n)
    nodeSize = n ÷ nNodes
    t = Vector{T}(undef, nNodes)
    slope = Vector{T}(undef, nNodes)
    ordinate = Vector{T}(undef, nNodes)
    t[1] = x[1]
    iStart = 1
    @inbounds for k = 1:nNodes
        iEnd = k == nNodes ? n : k * nodeSize #last particle of the segment
        if k < nNodes
            t[k+1] = x[iEnd+1]
        end
        m = iEnd - iStart + 1
        sumx = zero(T)
        sumx2 = zero(T)
        sumy = zero(T)
        sumxy = zero(T)
        for j = iStart:iEnd
            xj = x[j]
            yj = max(v[j], zero(T))
            sumx += xj
            sumx2 += xj * xj
            sumy += yj
            sumxy += xj * yj
        end
        det = sumx2 * m - sumx * sumx
        β1 = (sumxy * m - sumy * sumx) / det
        β0 = (sumy * sumx2 - sumxy * sumx) / det
        xRight = k == nNodes ? x[iEnd] : x[iEnd+1]
        if !(β0 + β1 * x[iStart] > 0 && β0 + β1 * xRight > 0) #also catches a singular segment
            β1 = zero(T)
            β0 = sumy / m
        end
        slope[k] = β1
        ordinate[k] = β0
        iStart = iEnd + 1
    end
    return LinearConditionalVariance(t, ordinate, slope, x[n])
end

function conditionalVariance(calc::CompactSupportCalculator, x::AbstractVector{T}, v::AbstractVector{T}, bandwidth, sqrtTime) where {T}
    n = length(x)
    gridSize = trunc(Int, max(calc.maxGridSize, calc.gridSize * sqrtTime))
    h = calc.bandwidthFactor * bandwidth * T(n)^(-T(0.2))
    fEta = inverseQuarticKernelValue(h * calc.threshold) * h
    f = Vector{T}(undef, gridSize)
    invExpectation = Vector{T}(undef, gridSize)
    indexFirstParticle = 1
    nNodes = 0
    @inbounds for j = 0:gridSize-1
        iStart = (j * (n - 1)) ÷ gridSize
        iEnd = ((j + 1) * (n - 1)) ÷ gridSize
        fj = x[(iStart+iEnd)÷2+1]
        indexFirstParticle = locateLowerIndex(x, fj - fEta, indexFirstParticle)
        fHigh = fj + fEta
        num = zero(T)
        denom = zero(T)
        for i = indexFirstParticle:n
            fi = x[i]
            z = (fi - fj) / h
            dirac = quarticKernelValue(z) / h
            num += dirac
            vi = v[i]
            if vi > 0 #the variance can be negative because of the discretization
                denom += dirac * vi
            end
            if fi >= fHigh && (denom != 0 || z >= 1)
                break
            end
        end
        if denom > 0 && num > 0 #num can be zero if there is no particle in the kernel window
            nNodes += 1
            f[nNodes] = fj
            invExpectation[nNodes] = num / denom
        end
    end
    resize!(f, nNodes)
    resize!(invExpectation, nNodes)
    return SplineConditionalVariance(f, invExpectation, isInverse=true)
end

function conditionalVariance(calc::LocalLinearCalculator, x::AbstractVector{T}, v::AbstractVector{T}, bandwidth, sqrtTime) where {T}
    n = length(x)
    gridSize = clamp(round(Int, calc.gridSize * max(sqrtTime, T(0.1))), 4, max(4, min(calc.maxGridSize, n)))
    h = max(calc.bandwidthFactor * bandwidth * T(n)^(-T(0.2)), T(1e-8))
    f = T[]
    e = T[]
    @inbounds for k = 1:gridSize
        fj = x[clamp(1 + ((2k - 1) * n) ÷ (2gridSize), 1, n)]
        if !isempty(f) && fj <= f[end] + T(1e-10)
            continue
        end
        S0 = S1 = S2 = T0 = T1 = W2 = zero(T)
        for i = searchsortedfirst(x, fj - h):searchsortedlast(x, fj + h)
            dx = x[i] - fj
            w = quarticKernelValue(dx / h) / h
            vi = max(v[i], zero(T))
            S0 += w
            S1 += w * dx
            S2 += w * dx * dx
            T0 += w * vi
            T1 += w * dx * vi
            W2 += w * w
        end
        if S0 <= 0 || S0^2 / W2 < calc.minEffectiveParticles
            continue
        end
        det = S0 * S2 - S1 * S1
        value = det > T(1e-14) ? (S2 * T0 - S1 * T1) / det : zero(T)
        if !(value > 0) #fallback to the local constant estimate
            value = T0 / S0
        end
        if value > 0
            push!(f, fj)
            push!(e, max(value, T(1e-8)))
        end
    end
    if isempty(f)
        push!(f, x[(n+1)÷2])
        push!(e, max(mean(vi -> max(vi, zero(T)), v), T(1e-6)))
    end
    m = length(f)
    slope = zeros(T, m)
    ordinate = copy(e)
    for k = 1:m-1
        slope[k] = (e[k+1] - e[k]) / (f[k+1] - f[k])
        ordinate[k] = e[k] - slope[k] * f[k]
    end
    return LinearConditionalVariance(f, ordinate, slope, f[m])
end

#Sorts particles (x,v) by x into preallocated buffers. Bucket sort followed by an insertion sort, linear on smooth
#distributions, with a fallback to sortperm on very clustered particles.
struct ParticleSorter{T}
    x::Vector{T}
    v::Vector{T}
    counts::Vector{Int}
    perm::Vector{Int}
end
ParticleSorter(::Type{T}, n::Int) where {T} = ParticleSorter(Vector{T}(undef, n), Vector{T}(undef, n), zeros(Int, n + 1), Int[])

function sortParticles!(s::ParticleSorter{T}, x::AbstractVector{T}, v::AbstractVector{T}) where {T}
    n = length(x)
    xs, vs, counts = s.x, s.v, s.counts
    xMin, xMax = extrema(x)
    if !(xMax > xMin)
        copyto!(xs, x)
        copyto!(vs, v)
        return xs, vs
    end
    c = (n - 1) / (xMax - xMin) * (1 - eps(T))
    fill!(counts, 0)
    maxCount = 0
    @inbounds for i = 1:n
        b = unsafe_trunc(Int, (x[i] - xMin) * c) + 2
        counts[b] += 1
        maxCount = max(maxCount, counts[b])
    end
    if maxCount > 64
        resize!(s.perm, n)
        sortperm!(s.perm, x)
        @inbounds for i = 1:n
            xs[i] = x[s.perm[i]]
            vs[i] = v[s.perm[i]]
        end
        return xs, vs
    end
    @inbounds for b = 2:n
        counts[b] += counts[b-1]
    end
    @inbounds for i = 1:n
        b = unsafe_trunc(Int, (x[i] - xMin) * c) + 1
        p = counts[b] += 1
        xs[p] = x[i]
        vs[p] = v[i]
    end
    @inbounds for i = 2:n #insertion sort, the buckets are already in order
        xi = xs[i]
        if xs[i-1] > xi
            vi = vs[i]
            j = i - 1
            while j >= 1 && xs[j] > xi
                xs[j+1] = xs[j]
                vs[j+1] = vs[j]
                j -= 1
            end
            xs[j+1] = xi
            vs[j+1] = vi
        end
    end
    return xs, vs
end

# ============================================================
# Leverage calibration by the particle method and QE simulation
# ============================================================

#E[V | ln S(genTimes[k]) = x] for each step k of the simulation grid genTimes.
struct SLVLeverage{F<:AbstractVector}
    genTimes::Vector{Float64}
    conditionalVariances::F
end
conditionalVariance(l::SLVLeverage, k::Int, x) = l.conditionalVariances[k](x)

#the paths of a step use the precomputations of the stochastic volatility scheme, the conditional variance
#conditionalVariances[step] and the local variance of the surface.
struct SLVPrecalc{P,S,F}
    scheme::P
    surface::S
    conditionalVariances::F
end

#the particles of the calibration: a scheme state without observations, and their variances
_particle_state(::QE, ::Type{T}, m::Int) where {T} =
    QEState(Vector{T}(undef, m), Vector{T}(undef, m), Matrix{T}(undef, m, 0), T[], Vector{T}(undef, m), Vector{T}(undef, m))
_copy_particle_variance!(v::AbstractVector, lo::Int, st::QEState) = copyto!(v, lo, st.v, 1, length(st.v))

function step!(st::QEState{T,PV}, method::QE, model::HestonLVModel, common::CommonSimData{T}, precalc::SLVPrecalc, dim::Int) where {T,PV}
    localVariance = DiscreteLocalVarianceSlice(precalc.surface, common.genTimes[dim], common.genTimes[dim+1])
    #function barrier: the conditional variance type may change with the step
    _stepHestonLV!(st, method.withMartingale, model.heston, localVariance, common, precalc.scheme, dim, precalc.conditionalVariances[dim], eachindex(st.v))
end

#evolves the paths of indices range. localVariance: y ↦ local variance over the step at the log-moneyness y,
#expectedVariance: x ↦ E[V | ln S = x] at the start of the step.
function _stepHestonLV!(st::QEState{T}, withMartingale::Bool, model::HestonModel, localVariance::LV, common::CommonSimData{T}, precalc::QEPrecalc{T}, dim::Int, expectedVariance::F, range::AbstractUnitRange{Int}) where {T,LV,F}
    h = common.h[dim]
    lnf0 = common.lnf[dim]
    lndrift = common.lnf[dim+1] - lnf0
    ekdt = precalc.ekdt[dim]
    psiC = precalc.psiC
    κ, θ, σ, ρ = model.κ, model.θ, model.σ, model.ρ
    ρBar = common.ρBar
    ρσ = ρ / σ
    κρσ = κ * ρσ
    s2v = σ^2 * ekdt / κ * (1 - ekdt)
    s2c = θ * σ^2 / (2κ) * (1 - ekdt)^2
    hρBar2 = h * (1 - ρ^2) / 4
    @inbounds for j in range
        x = st.logpathValues[j]
        vold = st.v[j]
        L2 = localVariance(x - lnf0) / expectedVariance(x)
        L = sqrt(L2)
        #Andersen QE for V, the martingale correction includes the leverage
        A = h * L / 2 * (κρσ - L / 2) + ρσ * L + hρBar2 * L2
        B = A - 2ρσ * L
        C = ρσ * κ * θ * h * L
        m = θ + (vold - θ) * ekdt
        psi = (vold * s2v + s2c) / (m * m)
        u = st.u2[j]
        vnew = zero(T)
        mcorr = zero(T)
        if psi <= psiC
            psiInv = inv(psi)
            b2 = 2psiInv - 1 + sqrt(2psiInv * (2psiInv - 1))
            a = m / (1 + b2)
            b = sqrt(b2)
            zv = norminv(u)
            vnew = a * (b + zv)^2
            if withMartingale
                mcorr = C - A * b2 * a / (1 - 2A * a) + log(1 - 2A * a) / 2 - B * vold
            end
        else
            p = (psi - 1) / (psi + 1)
            invBeta = m / (1 - p)
            vnew = u <= p ? zero(T) : log((1 - p) / (1 - u)) * invBeta
            if withMartingale
                mcorr = C - log(p + (1 - p) / (1 - invBeta * A)) - B * vold
            end
        end
        vIntegral = (vnew + vold) / 2
        st.logpathValues[j] = x + lndrift + ρσ * L * (vnew - vold - κ * θ * h) + (κρσ - L / 2) * L * vIntegral * h +
                              ρBar * sqrt(vIntegral) * L * st.z1[j] + mcorr
        st.v[j] = vnew
    end
    return nothing
end

function _calibration_sim_data(model::StochasticVolatilityModel, spot::Float64, config::GridConfig)
    c = _common_sim_data(model, spot, nothing, config)
    return CommonSimData(c.specTimes, c.genTimes, c.h, c.sqrth, c.lnf, Int[], c.df, c.ρBar, c.nSteps, 0)
end

#E[V | ln S(t) = x] estimated from all the particles at the time t.
function estimateConditionalVariance(calculator::LocalComponentCalculator, sorter::ParticleSorter{T}, surface, x::AbstractVector{T}, v::AbstractVector{T}, t::T) where {T}
    t == 0 && return ConstantConditionalVariance(v[1]) #all the particles start at the spot
    xs, vs = sortParticles!(sorter, x, v)
    bandwidth = sqrt(discreteLocalVarianceByLogmoneyness(surface, zero(T), zero(T), t) * max(t, T(0.25)))
    return conditionalVariance(calculator, xs, vs, bandwidth, sqrt(t))
end

"""
    calibrateLeverage(rng, model::StochasticLocalVolatilityModel, spot, config::GridConfig, nSim; calculator=CompactSupportCalculator(), start=0, nThreads=Threads.nthreads())

Calibrate the leverage of the stochastic local volatility model on the simulation grid `config.genTimes` with `nSim`
particles: before each step, the conditional expectation E[V | ln S] of the instantaneous variance is estimated by
`calculator` from all the particles, then the particles are evolved with the same scheme as `simulateScheme`
(`QE` for `HestonLVModel`, `EAE` for `SchobelZhuLVModel`). `rng` provides `ndims(model, config)` dimensions and the particle
`j` uses its point `start+j`.
The particles are evolved by blocks on `nThreads` tasks (with copies of `rng`, which must support `skipTo!`), the result
does not depend on `nThreads`.
"""
function calibrateLeverage(
    rng,
    model::StochasticLocalVolatilityModel,
    spot::Float64,
    config::GridConfig,
    nSim::Int;
    calculator::LocalComponentCalculator=CompactSupportCalculator(),
    start::Int=0,
    withBB::Bool=config.method.withBB,
    cacheSize::Int=0,
    nThreads::Int=Threads.nthreads(),
)
    T = Float64
    method = config.method
    sv = stochasticVolatilityModel(model)
    common = _calibration_sim_data(sv, spot, config)
    precalc = SLVPrecalc(precompute_scheme(method, sv, common), model.lv.surface, ConditionalVariance[])
    sorter = ParticleSorter(T, nSim)
    isNested = ccall(:jl_in_threaded_region, Cint, ()) != 0 #@threads :static is not allowed inside @threads
    nBlocks = isNested ? 1 : clamp(min(nThreads, Threads.nthreads(), nSim ÷ 4096), 1, nSim)
    bounds = [_block_bounds(b, nSim, nBlocks) for b = 1:nBlocks]
    states = map(bounds) do (lo, hi)
        m = hi - lo + 1
        st = _particle_state(method, T, m)
        reset_state!(st, sv, common.lnf[1])
        st
    end
    rngs = [b == 1 ? rng : _clone_rng(rng) for b = 1:nBlocks]
    bbs = [_make_bb(method, common, withBB, cacheSize) for b = 1:nBlocks]
    x = Vector{T}(undef, nSim)
    v = Vector{T}(undef, nSim)
    for dim = 1:common.nSteps
        for (b, (lo, hi)) in enumerate(bounds)
            copyto!(x, lo, states[b].logpathValues, 1, hi - lo + 1)
            _copy_particle_variance!(v, lo, states[b])
        end
        push!(precalc.conditionalVariances, estimateConditionalVariance(calculator, sorter, precalc.surface, x, v, common.genTimes[dim]))
        advance!(b) = begin
            bb, cache = bbs[b]
            load_step_variates!(states[b], method, rngs[b], start + bounds[b][1] - 1, dim, common, withBB, bb, cache)
            step!(states[b], method, model, common, precalc, dim)
        end
        if nBlocks == 1
            advance!(1)
        else #a block stays on the same thread: its particles stay in the cache of a core between steps
            Threads.@threads :static for b = 1:nBlocks
                advance!(b)
            end
        end
    end
    return SLVLeverage(copy(common.genTimes), [f for f in precalc.conditionalVariances])
end

"""
    simulateScheme(rng, model::StochasticLocalVolatilityModel, spot, payoff, start, nSim, config::GridConfig; leverage=nothing, calculator, rngCalibration, nSimCalibration=nSim)

Price `payoff` under the stochastic local volatility model (by the `QE` scheme for `HestonLVModel`, `EAE` for `SchobelZhuLVModel`). The leverage is calibrated with `rngCalibration` and
`nSimCalibration` particles by `calibrateLeverage` (on `nThreads` threads) unless it is given (on the same grid).
"""
function simulateScheme(
    rng,
    model::StochasticLocalVolatilityModel,
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig;
    withBB::Bool=config.method.withBB,
    cacheSize::Int=0,
    leverage::Union{Nothing,SLVLeverage}=nothing,
    calculator::LocalComponentCalculator=CompactSupportCalculator(),
    rngCalibration=AbstractRNGSeq(Chacha8SIMD(), ndims(model, config)),
    nSimCalibration::Int=nSim,
    nThreads::Int=Threads.nthreads(),
)
    T = Float64
    if isnothing(leverage)
        leverage = calibrateLeverage(rngCalibration, model, spot, config, nSimCalibration; calculator=calculator, withBB=withBB, cacheSize=cacheSize, nThreads=nThreads)
    elseif leverage.genTimes != config.genTimes
        throw(ArgumentError("the leverage was calibrated on a different time grid"))
    end
    sv = stochasticVolatilityModel(model)
    common = _common_sim_data(sv, spot, payoff, config)
    precalc = SLVPrecalc(precompute_scheme(config.method, sv, common), model.lv.surface, leverage.conditionalVariances)
    st = init_scheme_state(config.method, T, payoff, nSim, common.nObs)
    reset_state!(st, sv, common.lnf[1])
    _evolve_scheme!(st, rng, start, config.method, model, common, precalc, withBB, cacheSize)
    _finalize_payoffs!(st.payoffValues, payoff, st.obsPath, common.df, nSim)
    return mc_convergence_detailed(st.payoffValues)
end

#calibrates the leverage once, then prices blocks of paths on all the threads (the RNG must support skipTo!).
function simulateSchemeParallel(
    rng,
    model::StochasticLocalVolatilityModel,
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig;
    withBB::Bool=config.method.withBB,
    cacheSize::Int=0,
    leverage::Union{Nothing,SLVLeverage}=nothing,
    calculator::LocalComponentCalculator=CompactSupportCalculator(),
    rngCalibration=AbstractRNGSeq(Chacha8SIMD(), ndims(model, config)),
    nSimCalibration::Int=nSim,
)
    if isnothing(leverage)
        leverage = calibrateLeverage(rngCalibration, model, spot, config, nSimCalibration; calculator=calculator, withBB=withBB, cacheSize=cacheSize)
    end
    return simulateParallelReuse(simulateScheme, rng, model, spot, payoff, start, nSim, config; withBB=withBB, cacheSize=cacheSize, leverage=leverage)
end
