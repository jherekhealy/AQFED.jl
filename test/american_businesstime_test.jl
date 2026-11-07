# American put under time-weighted ("business time") volatility.
#
# The implied volatility σ is quoted in calendar time: the total variance to expiry stays σ²T and the
# European price is unchanged. The variance is redistributed across calendar days with weight 1 on
# business days and `nonBusinessWeight` on week-ends and holidays, as in Jäckel (2020), "Time-weighted
# volatility", with a single quote pillar at the option expiry.
#
# Compared:
#  - calendar time (flat vol): PDE (TR-BDF2) vs Andersen-Lake.
#  - business time, exercise on business days only: PDE vs interval Andersen-Lake (one interval per
#    block of business days, solved backwards with the Bermudan exercise at the end of each block).
#  - business time, exercise on all days: PDE vs global piecewise Andersen-Lake (one Chebyshev piece
#    per block of business days / non-business days, so that the boundary may have a kink at each edge).
# With a zero week-end weight, the two exercise policies give the same price: over a week-end the
# asset moves deterministically, and for a put with r ≥ q exercising on Friday dominates exercising
# on Saturday or Sunday (for q > r this is not strictly true, but the difference is negligible).
# Andersen-Lake then skips the (almost) zero-variance part of its integrals, otherwise any level below
# the boundary solves its fixed-point equation on week-end days. A tiny positive weight such as 1e-6,
# as recommended by Jäckel, is treated in the same way.
# With a positive week-end weight, Andersen-Lake (all-day exercise) is only an approximation of
# business-day exercise, and its fixed-point iteration needs more iterations than usual.
#
# Run with: julia --project=test test/american_businesstime_test.jl

using AQFED, Test, Printf, Plots
using AQFED.TermStructure, AQFED.American, AQFED.FDM
import AQFED.TermStructure: varianceByLogmoneyness
import AQFED.FDM: observationTimes, nonSmoothPoints, makeFDMStructure, evaluate, isLowerBoundActive
import AQFED.American: ChebyshevPPCollocation, TanhSinhPPQuadrature, exerciseBoundary
import AQFED.Black: blackScholesFormula

const DAYS_PER_YEAR = 365.0

## Business-time clock. Time 0 is the start of day 0, and day d ≥ 0 covers the calendar days [d, d+1).
struct BusinessClock
    nonBusinessWeight::Float64
    holidays::Vector{Int} #day indices
    firstWeekday::Int #weekday of day 0: 0 for Monday, ..., 6 for Sunday
end
BusinessClock(nonBusinessWeight, holidays) = BusinessClock(nonBusinessWeight, holidays, 0)

isBusinessDay(c::BusinessClock, d::Int) = mod(d + c.firstWeekday, 7) < 5 && !(d in c.holidays)
dayWeight(c::BusinessClock, d::Int) = isBusinessDay(c, d) ? 1.0 : c.nonBusinessWeight

"weighted number of days elapsed at calendar time t (in years)"
function businessTime(c::BusinessClock, t)
    days = t * DAYS_PER_YEAR
    n = floor(Int, days)
    return sum(dayWeight(c, d) for d = 0:n-1; init=0.0) + dayWeight(c, n) * (days - n)
end

"true if t lies within a business day, the day edges included."
function isBusinessTime(c::BusinessClock, t)
    days = t * DAYS_PER_YEAR
    d = round(Int, days)
    if abs(days - d) < 1e-8
        return isBusinessDay(c, d - 1) || isBusinessDay(c, d)
    end
    return isBusinessDay(c, floor(Int, days))
end

"calendar times where the business/non-business status changes, strictly between 0 and τ."
function statusChangeTimes(c::BusinessClock, τ)
    nDays = ceil(Int, τ * DAYS_PER_YEAR - 1e-8)
    return [d / DAYS_PER_YEAR for d = 1:nDays-1 if isBusinessDay(c, d) != isBusinessDay(c, d - 1)]
end

## Variance surface with total variance σ²T at expiry T, accrued along the business clock.
struct BusinessTimeSurface <: VarianceSurface
    σ::Float64
    expiry::Float64
    clock::BusinessClock
end

function varianceByLogmoneyness(s::BusinessTimeSurface, y, t)
    t = max(t, 1e-10)
    return s.σ^2 * s.expiry * businessTime(s.clock, t) / (businessTime(s.clock, s.expiry) * t)
end

makeModel(surface, r, q) = TSBlackModel(surface, ConstantRateCurve(r), ConstantRateCurve(r - q))

## PDE payoff: American put which may only be exercised at business times (unless exerciseOnAllDays).
## It records the exercise boundary at each time step (NaN when exercise is not allowed).
struct BusinessDayAmericanPut <: StructureDefinition
    vanilla::VanillaAmerican{Float64}
    clock::BusinessClock
    exerciseOnAllDays::Bool
    stepsPerBusinessDay::Int #> 0: time grid in variance time, see observationTimes
    boundaryTimes::Vector{Float64}
    boundaryLevels::Vector{Float64}
end

BusinessDayAmericanPut(strike, expiry, clock; exerciseOnAllDays=false, stepsPerBusinessDay=0) =
    BusinessDayAmericanPut(VanillaAmerican(false, strike, expiry), clock, exerciseOnAllDays, stepsPerBusinessDay, Float64[], Float64[])

canExercise(p::BusinessDayAmericanPut, t) = p.exerciseOnAllDays || t == p.vanilla.timeToExpiry || isBusinessTime(p.clock, t)

#each day edge is on the PDE time grid, so that each time step sees a single day weight.
#With stepsPerBusinessDay > 0, the grid is in variance time instead: within each block of business days, the steps are
#clustered as (time to the end of the block)², where the boundary behaves as at expiry, and each gap is a single step.
function observationTimes(p::BusinessDayAmericanPut)
    τ = p.vanilla.timeToExpiry
    if p.stepsPerBusinessDay > 0
        edges = vcat(0.0, statusChangeTimes(p.clock, τ), τ)
        t = Float64[]
        for i = 1:2:length(edges)-1
            a, b = edges[i], edges[i+1]
            k = p.stepsPerBusinessDay * round(Int, (b - a) * DAYS_PER_YEAR)
            append!(t, [b - (b - a) * (1 - j / k)^2 for j = 1:k])
            i + 2 <= length(edges) && push!(t, edges[i+2])
        end
        return t
    end
    dayEdges = (1:floor(Int, τ * DAYS_PER_YEAR)) ./ DAYS_PER_YEAR
    return vcat(filter(t -> t < τ - 1e-8, dayEdges), τ)
end

nonSmoothPoints(p::BusinessDayAmericanPut) = nonSmoothPoints(p.vanilla)
makeFDMStructure(p::BusinessDayAmericanPut, S::AbstractArray) = makeFDMStructure(p.vanilla, S)
isLowerBoundActive(p::BusinessDayAmericanPut, s::AQFED.FDM.DefaultFDMStructure) = canExercise(p, s.currentTime)

function evaluate(p::BusinessDayAmericanPut, s::AQFED.FDM.DefaultFDMStructure{T}, S::AbstractArray{T}, columnIndex::Int) where {T}
    level = NaN
    if canExercise(p, s.currentTime)
        v = @view s.currentValue[:, columnIndex]
        @. v = max(v, s.lowerBound)
        level = putBoundaryLevel(S, v, s.lowerBound, p.vanilla.strike)
    end
    push!(p.boundaryTimes, s.currentTime)
    push!(p.boundaryLevels, level)
end

"put exercise boundary from the grid values: last node where V = payoff, refined with the smooth fit V - payoff ∝ (S - B)²."
function putBoundaryLevel(S, V, payoff, strike)
    i = findlast(j -> S[j] < strike && V[j] - payoff[j] <= sqrt(eps()) * strike, eachindex(S))
    if isnothing(i) || i + 2 > length(S)
        return NaN
    end
    g1, g2 = sqrt(V[i+1] - payoff[i+1]), sqrt(V[i+2] - payoff[i+2])
    return clamp(S[i+1] - g1 * (S[i+2] - S[i+1]) / (g2 - g1), S[i], S[i+1])
end

function pricePDE(payoff::BusinessDayAmericanPut, spot, model; M=2001, stepsPerDay=8)
    empty!(payoff.boundaryTimes)
    empty!(payoff.boundaryLevels)
    N = payoff.stepsPerBusinessDay > 0 ? 2 : ceil(Int, payoff.vanilla.timeToExpiry * DAYS_PER_YEAR * stepsPerDay) + 1
    pp = priceTRBDF2(payoff, spot, model, CapitalizedDividend{Float64}[], M=M, N=N)
    return pp(spot)
end

## Global Andersen-Lake with one Chebyshev piece (and one tanh-sinh quadrature piece) per business/non-business block.
## Exercise is allowed on all days, which is equivalent to business days only when the week-end weight is zero.
function makeAndersenLake(model, clock, τ; nC=7, nTS=31, nIter=32)
    taus = vcat(0.0, statusChangeTimes(clock, τ), τ)
    return AndersenLakeGRepresentation(model, τ, false, nIter=nIter,
        collocation=ChebyshevPPCollocation(taus, nC, isDiscontinuous=false),
        quadrature=TanhSinhPPQuadrature(taus, nTS, isDiscontinuous=false))
end

## Andersen-Lake with exercise on business days only: one interval per block of business days, solved backwards with the
## Bermudan exercise at the end of each block. Valuation and expiry must be on business days.
function makeIntervalAndersenLake(model, clock, τ)
    edges = vcat(0.0, statusChangeTimes(clock, τ), τ)
    return AndersenLakeIntervalRepresentation(model, [(edges[i], edges[i+1]) for i = 1:2:length(edges)-1])
end

## Example: valuation on a Monday, expiry on the Friday close of the 9th week, one holiday on a Friday (day 32).
spot, σ, r, q = 100.0, 0.3, 0.05, 0.0
expiryDays = 7 * 8 + 5
τ = expiryDays / DAYS_PER_YEAR
holidays = [32]
calendarClock = BusinessClock(1.0, holidays)
businessClock = BusinessClock(0.0, holidays)
partialClock = BusinessClock(0.25, holidays)
calendarModel = makeModel(BusinessTimeSurface(σ, τ, calendarClock), r, q)
businessModel = makeModel(BusinessTimeSurface(σ, τ, businessClock), r, q)
partialModel = makeModel(BusinessTimeSurface(σ, τ, partialClock), r, q)

@testset "AmericanBusinessTime" begin
    alCalendar = makeAndersenLake(calendarModel, calendarClock, τ)
    alBusiness, aliBusiness = makeAndersenLake(businessModel, businessClock, τ), makeIntervalAndersenLake(businessModel, businessClock, τ)
    alPartial, aliPartial = makeAndersenLake(partialModel, partialClock, τ), makeIntervalAndersenLake(partialModel, partialClock, τ)
    println("                    calendar time      |          business time, weight 0          |        business time, weight 0.25")
    @printf("%6s %9s | %9s %9s | %9s %9s %9s %9s | %9s %9s %9s %9s\n", "Strike", "European",
        "PDE", "AL", "PDE bus.", "AL bus.", "PDE all", "AL all", "PDE bus.", "AL bus.", "PDE all", "AL all")
    for strike in [90.0, 100.0, 110.0, 120.0]
        european = blackScholesFormula(false, strike, forward(calendarModel, spot, τ), σ^2 * τ, 1.0, discountFactor(calendarModel, τ))
        calendarPDE = pricePDE(BusinessDayAmericanPut(strike, τ, calendarClock, exerciseOnAllDays=true), spot, calendarModel)
        calendarAL = priceAmerican(AndersenLakeRepresentation(ConstantBlackModel(σ, r, q), τ, 1e-8, 7, 8, 15, 31), strike, spot)
        calendarALG = priceAmerican(alCalendar, strike, spot)
        businessPDE = pricePDE(BusinessDayAmericanPut(strike, τ, businessClock), spot, businessModel)
        businessALI = priceAmerican(aliBusiness, strike, spot)
        businessPDEAllDays = pricePDE(BusinessDayAmericanPut(strike, τ, businessClock, exerciseOnAllDays=true), spot, businessModel)
        businessAL = priceAmerican(alBusiness, strike, spot)
        partialPDE = pricePDE(BusinessDayAmericanPut(strike, τ, partialClock), spot, partialModel)
        partialALI = priceAmerican(aliPartial, strike, spot)
        partialPDEAllDays = pricePDE(BusinessDayAmericanPut(strike, τ, partialClock, exerciseOnAllDays=true), spot, partialModel)
        partialAL = priceAmerican(alPartial, strike, spot)
        @printf("%6.1f %9.5f | %9.5f %9.5f | %9.5f %9.5f %9.5f %9.5f | %9.5f %9.5f %9.5f %9.5f\n", strike, european, calendarPDE, calendarAL,
            businessPDE, businessALI, businessPDEAllDays, businessAL, partialPDE, partialALI, partialPDEAllDays, partialAL)
        @test isapprox(calendarPDE, calendarAL, atol=1e-4)
        @test isapprox(calendarALG, calendarAL, atol=1e-5)
        @test isapprox(businessPDE, businessPDEAllDays, atol=1e-5)
        @test isapprox(businessALI, businessPDE, atol=5e-5)
        @test isapprox(businessAL, businessPDE, atol=1e-4)
        @test isapprox(partialALI, partialPDE, atol=5e-5)
        @test isapprox(partialAL, partialPDEAllDays, atol=1e-4)
    end

    #Jäckel's tiny week-end weight, q > r with an expiry on a Monday, and a 4-day option from a Friday to a Monday close
    for (weight, rr, qq, strike, days, firstWeekday, hols) in [(1e-6, r, q, 110.0, expiryDays, 0, holidays),
        (0.0, 0.01, 0.05, 110.0, expiryDays + 3, 0, holidays), (0.0, r, q, 100.0, 4, 4, Int[])]
        T = days / DAYS_PER_YEAR
        clock = BusinessClock(weight, hols, firstWeekday)
        model = makeModel(BusinessTimeSurface(σ, T, clock), rr, qq)
        businessPDE = pricePDE(BusinessDayAmericanPut(strike, T, clock), spot, model, stepsPerDay=64)
        businessPDEAllDays = pricePDE(BusinessDayAmericanPut(strike, T, clock, exerciseOnAllDays=true), spot, model, stepsPerDay=64)
        businessALI = priceAmerican(makeIntervalAndersenLake(model, clock, T), strike, spot)
        businessAL = priceAmerican(makeAndersenLake(model, clock, T), strike, spot)
        @printf("weight=%g r=%.2f q=%.2f K=%.0f days=%d: PDE bus. %.6f, AL bus. %.6f, PDE all %.6f, AL all %.6f\n",
            weight, rr, qq, strike, days, businessPDE, businessALI, businessPDEAllDays, businessAL)
        @test isapprox(businessPDE, businessPDEAllDays, atol=1e-6)
        @test isapprox(businessALI, businessPDE, atol=5e-5)
        @test isapprox(businessAL, businessPDE, atol=1e-4)
    end

    #exercise boundaries for the at-the-money strike
    strike = 100.0
    calendarPayoff = BusinessDayAmericanPut(strike, τ, calendarClock, exerciseOnAllDays=true)
    businessPayoff = BusinessDayAmericanPut(strike, τ, businessClock)
    businessPayoffAllDays = BusinessDayAmericanPut(strike, τ, businessClock, exerciseOnAllDays=true)
    pricePDE(calendarPayoff, spot, calendarModel)
    pricePDE(businessPayoff, spot, businessModel)
    pricePDE(businessPayoffAllDays, spot, businessModel)
    t = collect(range(0.0, τ, length=4001))
    days = t .* DAYS_PER_YEAR
    nonBusinessBlocks = statusChangeTimes(businessClock, τ) .* DAYS_PER_YEAR #(start, end) pairs as valuation and expiry are business days
    pFull = plot(title="Put K=$strike, S=$spot, σ=$σ, r=$r", legend=:topleft)
    pZoom = plot(title="First two weeks, business time", legend=:topleft, xlims=(0, 15), ylims=(80.5, 84.5))
    for p in (pFull, pZoom)
        vspan!(p, nonBusinessBlocks, color=:grey, alpha=0.15, label="Week-end / holiday")
        plot!(p, xlabel="Calendar days from valuation (Monday)", ylabel="Exercise boundary")
    end
    plot!(pFull, calendarPayoff.boundaryTimes .* DAYS_PER_YEAR, calendarPayoff.boundaryLevels, label="Calendar time, PDE", color=1)
    plot!(pFull, days, exerciseBoundary(alCalendar, strike, t), label="Calendar time, Andersen-Lake", color=1, linestyle=:dash)
    plot!(pFull, businessPayoff.boundaryTimes .* DAYS_PER_YEAR, businessPayoff.boundaryLevels, label="Business time, PDE", color=2)
    plot!(pFull, days, exerciseBoundary(aliBusiness, strike, t), label="Business time, interval Andersen-Lake", color=2, linestyle=:dash)
    plot!(pZoom, businessPayoffAllDays.boundaryTimes .* DAYS_PER_YEAR, businessPayoffAllDays.boundaryLevels, label="PDE (all-day exercise)", color=:black)
    plot!(pZoom, businessPayoff.boundaryTimes .* DAYS_PER_YEAR, businessPayoff.boundaryLevels, label="PDE (business-day exercise)", color=2, linewidth=2)
    plot!(pZoom, days, exerciseBoundary(aliBusiness, strike, t), label="Interval Andersen-Lake (business-day exercise)", color=2, linestyle=:dash)
    plot!(pZoom, days, exerciseBoundary(alBusiness, strike, t), label="Global Andersen-Lake (all-day exercise)", color=3, linestyle=:dot)
    savefig(plot(pFull, pZoom, layout=(1, 2), size=(1200, 450), margin=4Plots.mm), "american_businesstime_boundary.pdf")
end
