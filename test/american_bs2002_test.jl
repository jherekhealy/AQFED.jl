using AQFED, Test, Printf, Statistics, BenchmarkTools
using AQFED.American, AQFED.TermStructure
import AQFED.Black: blackScholesFormula
import Random: MersenneTwister

@testset "BS2002PaperTables" begin
    #Bjerksund-Stensland (2002), Tables 1-4: (r, b, σ, T, S, isCall, flat c̄, two-step c, 2c-c̄)
    K = 100.0
    refs = [
        (0.08, -0.04, 0.2, 0.25, 100.0, true, 3.49, 3.51, 3.54),
        (0.08, -0.04, 0.4, 0.25, 120.0, true, 21.23, 21.26, 21.28),
        (0.08, -0.04, 0.2, 0.5, 110.0, true, 10.94, 10.98, 11.01),
        (0.08, 0.04, 0.4, 0.25, 80.0, false, 20.53, 20.55, 20.56),
        (0.08, 0.04, 0.2, 0.5, 100.0, false, 4.70, 4.74, 4.79),
        (0.08, 0.08, 0.4, 0.25, 90.0, false, 12.48, 12.54, 12.60),
        (0.08, 0.08, 0.2, 0.5, 100.0, false, 4.11, 4.15, 4.20),
        (0.08, 0.0, 0.2, 3.0, 80.0, false, 22.12, 22.14, 22.15),
        (0.08, 0.0, 0.2, 3.0, 120.0, true, 23.93, 23.95, 23.98),
        (0.08, -0.04, 0.2, 3.0, 100.0, false, 15.47, 15.49, 15.50),
    ]
    for (r, b, σ, T, S, isCall, cflat, c2, proxy) in refs
        model = ConstantBlackModel(σ, r, r - b)
        for (method, ref) in ((:flat, cflat), (:twoStep, c2), (:proxy, proxy))
            price = priceAmerican(BjerksundStensland2002Representation(model, T, isCall=isCall, method=method), K, S)
            @test isapprox(price, ref, atol=0.005 + 1e-10)
        end
    end
    #no early exercise of the call when b >= r
    model = ConstantBlackModel(0.2, 0.08, 0.0)
    euro = blackScholesFormula(true, 100.0, 90.0 * exp(0.08 * 3), 0.2^2 * 3, 1.0, exp(-0.08 * 3))
    @test isapprox(priceAmerican(BjerksundStensland2002Representation(model, 3.0, isCall=true), 100.0, 90.0), euro, atol=1e-14)
    #here bT + 2σ√T < 0 in the call frame: without the floor X_T >= B0 the price would be -20. It is now a (useless) lower bound 0.
    model = ConstantBlackModel(0.2, 0.2, 0.0)
    @test priceAmerican(BjerksundStensland2002Representation(model, 5.0), 100.0, 120.0) >= 0
end

@testset "JuZhongExhibits" begin
    #Ju and Zhong (1999), Exhibit 5 (puts) and Exhibit 6 (calls), column Mquad, K=100, T=3.
    K = 100.0
    mquad5 = [25.725, 20.185, 15.608, 11.905, 8.974, 22.148, 16.170, 11.700, 8.390, 5.968, 20.336, 13.471, 8.931, 5.920, 3.922, 20.000, 11.705, 6.956, 4.190, 2.551]
    i = 0
    for δ in (0.12, 0.08, 0.04, 0.0), S in 80.0:10:120
        i += 1
        @test isapprox(priceAmerican(JuZhongRepresentation(ConstantBlackModel(0.2, 0.08, δ), 3.0), K, S), mquad5[i], atol=6e-4)
    end
    mquad6 = [2.605, 5.182, 9.065, 14.430, 21.398, 11.336, 15.711, 20.760, 26.440, 32.709, 5.552, 8.868, 13.158, 18.458, 24.786, 12.177, 17.411, 23.402, 30.028, 37.177]
    i = 0
    for (σ, r, δ) in ((0.2, 0.03, 0.07), (0.4, 0.03, 0.07), (0.3, 0.0, 0.07), (0.3, 0.07, 0.03)), S in 80.0:10:120
        i += 1
        @test isapprox(priceAmerican(JuZhongRepresentation(ConstantBlackModel(σ, r, δ), 3.0, isCall=true), K, S), mquad6[i], atol=6e-4)
    end
end

## Benchmark definitions, shared with the note in mypapers/american_bs2002.
#Errors against Andersen-Lake-Offengenden (16,16,31,63). The implied volatility error is σ̂-σ where the reference
#American price at σ̂ matches the approximate price; it is only computed when the reference vega is >= 1.

hasEarlyExercise(isCall, r, q) = isCall ? q > 0 : r > 0

europeanPrice(isCall, σ, r, q, T, K, S) = blackScholesFormula(isCall, K, S * exp((r - q) * T), σ^2 * T, 1.0, exp(-r * T))

function referencePrice(isCall, σ, r, q, T, K, S)
    if !hasEarlyExercise(isCall, r, q)
        return europeanPrice(isCall, σ, r, q, T, K, S)
    end
    priceAmerican(AndersenLakeRepresentation(ConstantBlackModel(σ, r, q), T, 1e-12, 16, 16, 31, 63, isCall=isCall), K, S)
end

#Ju98 prices the call via McDonald-Schroder symmetry on the transformed model.
function makeJu98(σ, r, q, T, isCall; m=3, kw...)
    model = isCall ? TSBlackModel(FlatSurface(σ), ConstantRateCurve(q), ConstantRateCurve(q - r)) :
            TSBlackModel(FlatSurface(σ), ConstantRateCurve(r), ConstantRateCurve(r - q))
    return Ju98Representation(model, T; isCall=isCall, m=m, kw...)
end
juPrice(isCall, σ, r, q, T, K, S; kw...) = hasEarlyExercise(isCall, r, q) ? priceAmerican(makeJu98(σ, r, q, T, isCall; kw...), K, S) : europeanPrice(isCall, σ, r, q, T, K, S)
bsPrice(method, isCall, σ, r, q, T, K, S) = priceAmerican(BjerksundStensland2002Representation(ConstantBlackModel(σ, r, q), T, isCall=isCall, method=method), K, S)

const benchmarkMethods = [
    ("BS2002 flat", (args...) -> bsPrice(:flat, args...)),
    ("BS2002 two-step", (args...) -> bsPrice(:twoStep, args...)),
    ("BS2002 2c-c̄", (args...) -> bsPrice(:proxy, args...)),
    ("Ju-Zhong", (isCall, σ, r, q, T, K, S) -> priceAmerican(JuZhongRepresentation(ConstantBlackModel(σ, r, q), T, isCall=isCall), K, S)),
    ("Ju P3", (args...) -> juPrice(args...; m=3)),
    ("AL 7,8,15,31", (isCall, σ, r, q, T, K, S) -> hasEarlyExercise(isCall, r, q) ? priceAmerican(AndersenLakeRepresentation(ConstantBlackModel(σ, r, q), T, 1e-8, 7, 8, 15, 31, isCall=isCall), K, S) : europeanPrice(isCall, σ, r, q, T, K, S)),
]

struct OptionCase
    isCall::Bool
    σ::Float64
    r::Float64
    q::Float64
    T::Float64
    K::Float64
    S::Float64
end

#Bjerksund-Stensland (2002) Tables 1-4 (Barone-Adesi-Whaley parameters), options with early exercise.
function paperCases()
    cases = OptionCase[]
    K = 100.0
    for b in (-0.04, 0.04), (r, σ, T) in ((0.08, 0.2, 0.25), (0.12, 0.2, 0.25), (0.08, 0.4, 0.25), (0.08, 0.2, 0.5)), S in 80.0:10:120, isCall in (true, false)
        push!(cases, OptionCase(isCall, σ, r, r - b, T, K, S))
    end
    for (r, σ, T) in ((0.08, 0.2, 0.25), (0.12, 0.2, 0.25), (0.08, 0.4, 0.25), (0.08, 0.2, 0.5)), S in 80.0:10:120
        push!(cases, OptionCase(false, σ, r, 0.0, T, K, S))
    end
    for b in (-0.04, 0.0, 0.04), S in 80.0:10:120, isCall in (true, false)
        push!(cases, OptionCase(isCall, 0.2, 0.08, 0.08 - b, 3.0, K, S))
    end
    return filter(c -> hasEarlyExercise(c.isCall, c.r, c.q), cases)
end

#Broadie and Detemple (1996), p. 1222-1225: 2,500 random American calls.
function broadieDetempleCases(n=2500; seed=1996)
    rng = MersenneTwister(seed)
    map(1:n) do _
        σ = 0.1 + 0.5 * rand(rng)
        T = rand(rng) < 0.75 ? 0.1 + 0.9 * rand(rng) : 1.0 + 4 * rand(rng)
        S = 70.0 + 60 * rand(rng)
        q = 0.1 * rand(rng)
        r = rand(rng) < 0.8 ? 0.1 * rand(rng) : 0.0
        OptionCase(true, σ, r, q, T, 100.0, S)
    end
end

#Ju (1998), Table 3: 1,250 random American puts.
function juCases(n=1250; seed=1998)
    rng = MersenneTwister(seed)
    map(1:n) do _
        σ = 0.1 + 0.5 * rand(rng)
        T = 3 * rand(rng)
        S = 70.0 + 60 * rand(rng)
        r = 0.1 * rand(rng)
        q = 0.1 * rand(rng)
        OptionCase(false, σ, r, q, T, 100.0, S)
    end
end

const aloSpots = (25.0, 50.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0, 175.0, 200.0)
const aloVols = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
#Andersen, Lake and Offengenden (2016), Table 3: American puts with r = q.
aloEqualCases() = [OptionCase(false, σ, r, r, T, 100.0, S) for r in (0.02, 0.04, 0.06, 0.08, 0.10), S in aloSpots, T in (1 / 12, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0), σ in aloVols][:]
#Andersen, Lake and Offengenden (2016), Table 7: American puts, general case.
aloGeneralCases() = [OptionCase(false, σ, r, q, T, 100.0, S) for r in (0.02, 0.04, 0.06, 0.08, 0.10), q in (0.0, 0.04, 0.08, 0.12), S in aloSpots, T in (1 / 12, 0.25, 0.5, 0.75, 1.0), σ in aloVols][:]
#the same grid priced as calls (not in the paper), without the calls that have no early exercise (q = 0).
aloGeneralCallCases() = [OptionCase(true, c.σ, c.r, c.q, c.T, c.K, c.S) for c in aloGeneralCases() if c.q > 0]

intrinsic(c::OptionCase) = max(c.isCall ? c.S - c.K : c.K - c.S, 0.0)

#chord iterations on the reference price, with the reference vega at σ; clamped to [σ/4, 4σ].
function impliedVolError(price, refPrice, vega, c::OptionCase)
    isfinite(price) || return NaN
    try
        σi = clamp(c.σ + (price - refPrice) / vega, c.σ / 4, 4c.σ)
        for _ = 1:5
            diff = referencePrice(c.isCall, σi, c.r, c.q, c.T, c.K, c.S) - price
            abs(diff) < 1e-12 && break
            σi = clamp(σi - diff / vega, c.σ / 4, 4c.σ)
        end
        return σi - c.σ
    catch
        return NaN
    end
end

#returns reference prices, vegas, and for each method the prices and implied volatility errors (NaN when vega < 1).
function evaluateCases(cases; methods=benchmarkMethods)
    n = length(cases)
    refs, vegas = fill(NaN, n), fill(NaN, n)
    prices = fill(NaN, n, length(methods))
    volErrors = fill(NaN, n, length(methods))
    Threads.@threads for i = 1:n
        c = cases[i]
        args = (c.isCall, c.σ, c.r, c.q, c.T, c.K, c.S)
        try
            refs[i] = referencePrice(args...)
            h = 1e-3 * c.σ
            vegas[i] = (referencePrice(c.isCall, c.σ + h, c.r, c.q, c.T, c.K, c.S) - referencePrice(c.isCall, c.σ - h, c.r, c.q, c.T, c.K, c.S)) / (2h)
        catch
            continue
        end
        for (k, (_, f)) in enumerate(methods)
            prices[i, k] = try
                f(args...)
            catch
                NaN
            end
            if vegas[i] >= 1
                volErrors[i, k] = impliedVolError(prices[i, k], refs[i], vegas[i], c)
            end
        end
    end
    return refs, vegas, prices, volErrors
end

#RMSE, max abs error, RMS relative error; RMS, max and mean implied vol error in %, number of failures.
function errorStatistics(refs, vegas, prices, volErrors, mask)
    map(1:size(prices, 2)) do k
        pe = prices[mask, k] .- refs[mask]
        ok = isfinite.(pe)
        v = volErrors[mask.&(vegas.>=1), k] .* 100
        vok = filter(isfinite, v)
        (rmse=sqrt(mean(pe[ok] .^ 2)), maxAbs=maximum(abs.(pe[ok])), rmsre=sqrt(mean((pe[ok] ./ refs[mask][ok]) .^ 2)),
            volRMS=sqrt(mean(vok .^ 2)), volMax=maximum(abs.(vok)), volMean=mean(vok), failures=count(!, ok) + count(!isfinite, v))
    end
end

function printErrorTable(title, refs, vegas, prices, volErrors, mask; methods=benchmarkMethods)
    println(title, " (", count(mask), " options, ", count(mask .& (vegas .>= 1)), " with vega ≥ 1)")
    @printf("%-16s %10s %10s %10s %10s %10s %10s %5s\n", "method", "RMSE", "MaxAbs", "RMSRE", "RMS vol%", "Max vol%", "Mean vol%", "fail")
    for (k, s) in enumerate(errorStatistics(refs, vegas, prices, volErrors, mask))
        @printf("%-16s %10.2e %10.2e %10.2e %10.3f %10.3f %10.3f %5d\n", methods[k][1], s.rmse, s.maxAbs, s.rmsre, s.volRMS, s.volMax, s.volMean, s.failures)
    end
    println()
end

## Benchmark run
@testset "BS2002Benchmark" begin
    cases = paperCases()
    refs, vegas, prices, volErrors = evaluateCases(cases)
    printErrorTable("Bjerksund-Stensland 2002 parameter set", refs, vegas, prices, volErrors, trues(length(cases)))
    iFlat, iTwo = 1, 2
    @test maximum(abs.(prices[:, iTwo] .- refs)) < 0.08
    #two-step and flat are lower bounds (but the two-step strategy does not always dominate the flat one).
    @test all(prices[:, iTwo] .< refs .+ 1e-10)
    @test all(prices[:, iFlat] .< refs .+ 1e-10)

    #timings include the construction of the representation, for a single option.
    σ, r, q, T, K, S = 0.3, 0.05, 0.02, 1.0, 100.0, 95.0
    println("Time per option (put, σ=30%, r=5%, q=2%, T=1)")
    for (name, f) in benchmarkMethods
        t = @belapsed $f(false, $σ, $r, $q, $T, $K, $S)
        @printf("%-16s %10.3f μs\n", name, t * 1e6)
    end
end
