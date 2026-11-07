import SpecialFunctions: besselix
export BackwardScript, ScriptEvent, ProductStateGrid, on, jump!, setState!, autocallScript, asianScript, volatilitySwapScript, discreteVolatilitySwapScript, carrLeeVolatilityPayoff

#Backward scripts, following the backward evaluation pseudo-codes of the book: a list of events, each applied at some dates to
#the value V(S, Q) where Q is an optional path dependent state (running average, accumulated quantity...), discretized on a grid
#(the columns of the finite difference structure). The events are listed in the backward order, as in the book listings,
#for example the autocall reads
#
#    BackwardScript(
#        on(maturity) do V, S, Q, k
#            V .= 1
#        end,
#        on(autocallTimes) do V, S, Q, k
#            @. V = ifelse(S >= K * S0, C[k], V)
#        end; nonSmoothPoints=[K * S0])

"""
    ScriptEvent(times, action!)

Event applied at each of the `times` by `action!(V, S, Q, k)`, where `V` is the matrix of values (asset × state), `S` the asset
grid, `Q` the state grid and `k` the index of the current time in `times`. Created with `on(times) do V, S, Q, k ... end`.
"""
struct ScriptEvent{F}
    times::Vector{Float64}
    action!::F
end

on(action!, times) = ScriptEvent(collect(Float64, vcat(times)), action!)

"""
    BackwardScript(events...; stateGrid=[0.0], nonSmoothPoints=Float64[])

Finite difference payoff defined by backward events. Events at the same time are applied in the order given. The price is
read at the first state `stateGrid[1]`: the state must either be reset by an event (a fixing), or be set at time 0 by
`setState!`.
"""
struct BackwardScript{TE<:Tuple,TQ<:AbstractVector} <: StructureDefinition
    events::TE
    stateGrid::TQ
    specialPoints::Vector{Float64}
end
BackwardScript(events::ScriptEvent...; stateGrid::AbstractVector=[0.0], nonSmoothPoints::AbstractVector=Float64[]) =
    BackwardScript(events, stateGrid isa ProductStateGrid ? stateGrid : collect(Float64, stateGrid), collect(Float64, nonSmoothPoints))

"""
    ProductStateGrid(grid1, grid2)

Two dimensional state (q₁, q₂) on the tensor product of two sorted grids. The state of column j is `Q[j]`, a tuple, with q₁
varying fastest. `jump!` interpolates in each dimension as for a single state.
"""
struct ProductStateGrid <: AbstractVector{Tuple{Float64,Float64}}
    grid1::Vector{Float64}
    grid2::Vector{Float64}
end
ProductStateGrid(grid1::AbstractVector, grid2::AbstractVector) = ProductStateGrid(collect(Float64, grid1), collect(Float64, grid2))
Base.size(Q::ProductStateGrid) = (length(Q.grid1) * length(Q.grid2),)
function Base.getindex(Q::ProductStateGrid, j::Int)
    i1, i2 = divrem(j - 1, length(Q.grid1))
    return (Q.grid1[i2+1], Q.grid2[i1+1])
end

#a time step may be skipped when it is shorter than 1e-8, see priceTRBDF2
const scriptTimeTolerance = 1e-8

observationTimes(p::BackwardScript) = sort(unique(vcat((e.times for e in p.events)...)))
nonSmoothPoints(p::BackwardScript) = p.specialPoints

makeFDMStructure(p::BackwardScript, underlying::AbstractArray{T}) where {T} =
    DefaultFDMStructure(zero(T), zeros(T, length(underlying), length(p.stateGrid)), zeros(T, length(underlying)))

#the solver evaluates the columns in reverse order: the events are applied to the whole matrix with the first column.
function evaluate(p::BackwardScript, s::DefaultFDMStructure{T}, S::AbstractArray{T}, columnIndex::Int) where {T}
    columnIndex == 1 || return nothing
    _applyEvents(s.currentTime, s.currentValue, S, p.stateGrid, p.events...)
end

#unrolled over the tuple of events, so that each action is compiled for its own closure type
_applyEvents(t, V, S, Q) = nothing
function _applyEvents(t, V, S, Q, event::ScriptEvent, events...)
    for (k, ti) in enumerate(event.times)
        if abs(t - ti) < scriptTimeTolerance
            event.action!(V, S, Q, k)
        end
    end
    _applyEvents(t, V, S, Q, events...)
end

"""
    jump!(V, S, Q, newState)

Continuity condition V(t⁻, S, Q) = V(t⁺, S, newState(S, Q)), by quadratic interpolation in the state, and linear extrapolation
outside of the state grid.
"""
function jump!(V::AbstractMatrix{T}, S::AbstractVector, Q::AbstractVector, newState) where {T}
    row = Vector{T}(undef, length(Q))
    @inbounds for i in eachindex(S)
        row .= @view V[i, :]
        for j in eachindex(Q)
            V[i, j] = _interpolateState(Q, row, newState(S[i], Q[j]))
        end
    end
    return V
end

"""
    setState!(V, S, Q, q0)

Sets all the states to the value at the state `q0`, typically at time 0 when the initial state is not `Q[1]`.
"""
setState!(V::AbstractMatrix, S::AbstractVector, Q::AbstractVector, q0) = jump!(V, S, Q, (s, q) -> q0)

#interpolation stencil: three node indices and weights, quadratic Lagrange inside the grid, linear extrapolation outside
function _stencil(Q::AbstractVector{T}, q) where {T}
    n = length(Q)
    n == 1 && return (1, 1, 1), (one(T), zero(T), zero(T))
    if q <= Q[1] || q >= Q[n] || n == 2
        j = q >= Q[n] ? n - 1 : 1
        w = (q - Q[j]) / (Q[j+1] - Q[j])
        return (j, j + 1, j + 1), (1 - w, w, zero(T))
    end
    j = clamp(searchsortedlast(Q, q), 2, n - 1)
    x0, x1, x2 = Q[j-1], Q[j], Q[j+1]
    return (j - 1, j, j + 1), ((q - x1) * (q - x2) / ((x0 - x1) * (x0 - x2)), (q - x0) * (q - x2) / ((x1 - x0) * (x1 - x2)),
        (q - x0) * (q - x1) / ((x2 - x0) * (x2 - x1)))
end

function _interpolateState(Q::AbstractVector, v::AbstractVector, q)
    idx, w = _stencil(Q, q)
    return w[1] * v[idx[1]] + w[2] * v[idx[2]] + w[3] * v[idx[3]]
end

#tensor product of the stencils of each state
function _interpolateState(Q::ProductStateGrid, v::AbstractVector, q::Tuple)
    n1 = length(Q.grid1)
    idx1, w1 = _stencil(Q.grid1, q[1])
    idx2, w2 = _stencil(Q.grid2, q[2])
    value = zero(eltype(v))
    for b = 1:3, a = 1:3
        value += w1[a] * w2[b] * v[(idx2[b]-1)*n1+idx1[a]]
    end
    return value
end

"""
    autocallScript(observationTimes, autocallLevels, coupons; reference=1.0, protectionLevel=0.0)

Autocall note on a notional of 1 (the backward version of `MonteCarlo.AutocallNote`): redeemed at `coupons[k]` at the first
observation time where the asset is above `autocallLevels[k]*reference`, otherwise at maturity 1, or the performance S/reference
if below `protectionLevel*reference`. The coupons are paid at the observation times, with the discounting of the PDE.
"""
function autocallScript(observationTimes::AbstractVector, autocallLevels::AbstractVector, coupons::AbstractVector; reference::Real=1.0, protectionLevel::Real=0.0)
    maturity = observationTimes[end]
    levels = autocallLevels .* reference
    return BackwardScript(
        on(maturity) do V, S, Q, k
            @. V = ifelse(S < protectionLevel * reference, S / reference, one(eltype(V)))
        end,
        on(observationTimes) do V, S, Q, k
            @. V = ifelse(S >= levels[k], coupons[k], V)
        end; nonSmoothPoints=filter(x -> 0 < x < Inf, unique(vcat(levels, protectionLevel * reference))))
end

"""
    asianScript(isCall, strike, fixingTimes, stateGrid; isFloatingStrike=false, maturity=fixingTimes[end])

Discretely monitored arithmetic Asian option: max(A-K, 0) for a fixed strike (Asian rate) or max(S_T-A, 0) for a floating
strike (Asian strike) at the `maturity`, with A the arithmetic average of the asset at the `fixingTimes` (up to the maturity).
The state is the running average, discretized on `stateGrid`, which should cover the likely range of the average.
"""
function asianScript(isCall::Bool, strike::Real, fixingTimes::AbstractVector, stateGrid::AbstractVector; isFloatingStrike::Bool=false, maturity::Real=fixingTimes[end])
    sign = isCall ? 1 : -1
    return BackwardScript(
        on(maturity) do V, S, Q, k
            if isFloatingStrike
                @. V = max(sign * (S - Q'), 0)
            else
                @. V = max(sign * (Q' - strike), zero(S))
            end
        end,
        on(fixingTimes) do V, S, Q, k
            #the running average after the k-th fixing
            jump!(V, S, Q, (s, q) -> q + (s - q) / k)
        end; stateGrid=stateGrid, nonSmoothPoints=isFloatingStrike ? Float64[] : [strike])
end

"""
    carrLeeVolatilityPayoff(S, forward)

European payoff h(S) = √(π/2) √(S/F) |x| (I₀(x/2) - I₁(x/2)), x = ln(S/F), with E[h(S_T)] = E[√⟨ln S⟩_T] when the volatility
is independent of the asset Brownian motion (Carr and Lee 2008).
"""
function carrLeeVolatilityPayoff(S::Real, forward::Real)
    x = max(log(S / forward), -700.0) #h grows like √(2|x|) when S → 0
    #scaled Bessel functions, to avoid the overflow for large |x|
    return sqrt(π / 2) * sqrt(S / forward) * abs(x) * (besselix(0, x / 2) - besselix(1, x / 2)) * exp(abs(x) / 2)
end

"""
    volatilitySwapScript(tte, forward; u=100/√tte)

Volatility swap on the synthetic payoff `u carrLeeVolatilityPayoff(S, forward)` at `tte`, by default the annualized volatility
in percent.
"""
volatilitySwapScript(tte::Real, forward::Real; u::Real=100 / sqrt(tte)) = BackwardScript(
    on(tte) do V, S, Q, k
        @. V = u * carrLeeVolatilityPayoff(S, forward)
    end; nonSmoothPoints=[forward])

"""
    discreteVolatilitySwapScript(fixingTimes, volatilityGrid, logSpotGrid; annualization=n/(t_n-t_0), u=100)

Volatility swap on the realized volatility u √(annualization/n Σᵢ ln²(Sᵢ/Sᵢ₋₁)) of the n returns between the `fixingTimes`
t₀ < t₁ < ... < tₙ, paid at tₙ. The states are the accumulated volatility √(Σ ln²(Sᵢ/Sᵢ₋₁)) on `volatilityGrid` (starting at 0),
and the log of the asset at the previous fixing on `logSpotGrid`. The continuity condition at the fixing tᵢ is
V(tᵢ⁻, S, v, y) = V(tᵢ⁺, S, √(v² + (ln S - y)²), ln S).
"""
function discreteVolatilitySwapScript(fixingTimes::AbstractVector, volatilityGrid::AbstractVector, logSpotGrid::AbstractVector;
    annualization::Real=(length(fixingTimes) - 1) / (fixingTimes[end] - fixingTimes[1]), u::Real=100)
    scale = u * sqrt(annualization / (length(fixingTimes) - 1))
    #ln S is clamped to the state grid, in particular at S = 0: this only changes the values far outside of the grid
    ylo, yhi = extrema(logSpotGrid)
    clampedLog(s) = clamp(log(s), ylo, yhi)
    return BackwardScript(
        on(fixingTimes[end]) do V, S, Q, k
            for j in eachindex(Q)
                V[:, j] .= scale * Q[j][1]
            end
        end,
        on(fixingTimes) do V, S, Q, k
            if k == 1 #the initial fixing starts the accumulation
                jump!(V, S, Q, (s, q) -> (0.0, clampedLog(s)))
            else
                jump!(V, S, Q, (s, q) -> (y = clampedLog(s); (sqrt(q[1]^2 + (y - q[2])^2), y)))
            end
        end; stateGrid=ProductStateGrid(volatilityGrid, logSpotGrid))
end
