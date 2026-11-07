# Chapter 9: accuracy and speed of the Fourier pricers on Heston (tables of the sections on mild and challenging parameters
# and on the Andersen-Lake benchmark). The timings are only meaningful on a single thread:
#   julia --project=book -t1 book/run.jl --compute 'heston_(challenge|mild|al_benchmark)'

using CharFuncPricing, Statistics

const AL_MATURITIES = [0.0025, 0.1, 0.5, 2.0, 10.0, 30.0]
const AL_VARIANCES = [0.0001, 0.0025, 0.04, 0.25, 1.0]
const AL_KAPPAS = [0.01, 0.1, 0.5, 2.0]
const AL_SIGMAS = [0.0001, 0.1, 0.5, 1.0, 3.0]
const AL_RHOS = [-0.95, -0.5, -0.1, 0.0, 0.1, 0.5, 0.95]
const AL_STRIKES = [100.0001, 101.0, 110.0, 200.0, 1000.0, 10000.0]
const AL_ITM = length(AL_STRIKES) #the in-the-money options come first

function alBenchmarkSets()
    sets = NTuple{6,Float64}[]
    for τ in AL_MATURITIES, v0 in AL_VARIANCES, θ in AL_VARIANCES, κ in AL_KAPPAS, σ in AL_SIGMAS, ρ in AL_RHOS
        push!(sets, (τ, v0, θ, κ, σ, ρ))
    end
    return sets
end
#(forward, strike): in-the-money puts, then out-of-the-money puts of strike 100
alBenchmarkOptions() = vcat([(100.0, k) for k in AL_STRIKES], [(f, 100.0) for f in [100.0, AL_STRIKES...]])
alCharFunc((τ, v0, θ, κ, σ, ρ)) = DefaultCharFunc(HestonParams(v0, κ, θ, ρ, σ))

function alBenchmarkPrices(make, sets, options)
    prices = Matrix{Float64}(undef, length(sets), length(options))
    build, itm, otm = 0.0, 0.0, 0.0
    for (i, s) in enumerate(sets)
        τ = s[1]
        t0 = time_ns()
        pricer = make(alCharFunc(s), τ)
        t1 = time_ns()
        for j in 1:AL_ITM
            prices[i, j] = CharFuncPricing.priceEuropean(pricer, false, options[j][2], options[j][1], τ, 1.0)
        end
        t2 = time_ns()
        for j in AL_ITM+1:length(options)
            prices[i, j] = CharFuncPricing.priceEuropean(pricer, false, options[j][2], options[j][1], τ, 1.0)
        end
        t3 = time_ns()
        build += (t1 - t0) * 1e-9
        itm += (t2 - t1) * 1e-9
        otm += (t3 - t2) * 1e-9
    end
    return prices, build, itm, otm
end

function alBenchmarkRows(method, make, sets, options, reference)
    alBenchmarkPrices(make, sets[1:min(end, 100)], options) #compilation, not part of the timing
    prices, build, itm, otm = alBenchmarkPrices(make, sets, options)
    err = abs.(prices .- reference)
    rel = err ./ abs.(reference)
    n = length(sets)
    errItm, relItm = vec(err[:, 1:AL_ITM]), vec(rel[:, 1:AL_ITM])
    relAll = rel[abs.(reference).>1e-90] #some reference out-of-the-money prices are below the double precision threshold
    return [(Method=method, Options="ITM", MAE=maximum(errItm), RMSE=sqrt(mean(abs2, errItm)), MRE=maximum(relItm), RRMSE=sqrt(mean(abs2, relItm)), OptionsPerSec=AL_ITM * n / (build + itm)),
        (Method=method, Options="ALL", MAE=maximum(err), RMSE=sqrt(mean(abs2, err)), MRE=maximum(relAll), RRMSE=sqrt(mean(abs2, relAll)), OptionsPerSec=length(options) * n / (build + itm + otm))]
end

## Tables tbl:heston_al_set_itm and tbl:heston_al_set_all
@compute "heston_al_benchmark" begin
    sets, options = alBenchmarkSets(), alBenchmarkOptions()
    reference = Matrix{Float64}(undef, length(sets), length(options))
    Threads.@threads :dynamic for i in eachindex(sets)
        p = ALCharFuncPricer(alCharFunc(sets[i]), n=2000)
        reference[i, :] = [CharFuncPricing.priceEuropean(p, false, k, f, sets[i][1], 1.0) for (f, k) in options]
    end
    methods = [("Flinn-Transf.", (c, τ) -> AdaptiveFlinnCharFuncPricer(c, τ)),
        ("Flinn-Transf. 1e-11", (c, τ) -> AdaptiveFlinnCharFuncPricer(c, τ, qTol=1e-11)),
        ("Flinn 1e-10", (c, τ) -> FlinnCharFuncPricer(c, τ, tTol=1e-10, qTol=1e-10)),
        ("Andersen-Lake", (c, τ) -> ALCharFuncPricer(c, n=200)),
        ("Andersen-Lake n=1000", (c, τ) -> ALCharFuncPricer(c, n=1000)),
        ("Andersen-Lake GL", (c, τ) -> ALCharFuncPricer(c, ModlobQuadrature(1e-8))),
        ("Cos", (c, τ) -> makeCosCharFuncPricer(c, τ, 200, 8)),
        ("Cos-Adaptive", (c, τ) -> makeCosCharFuncPricer(c, τ, 8))]
    rows = NamedTuple[]
    for (method, make) in methods
        append!(rows, alBenchmarkRows(method, make, sets, options, reference))
        writetable("heston_al_benchmark.csv", DataFrame(rows))
    end
end

## Rows of the Flinn quadratures of tables tbl:heston_challenge_2 and tbl:heston_challenge_3. Case 1 has no table anymore, the text quotes its errors.
@compute "heston_challenge" begin
    cases = [(1, HestonParams(1.0, 2.0, 0.0025, 0.5, 0.1), 10.0, 10000.0),
        (2, HestonParams(0.0001, 0.1, 0.25, 0.95, 3.0), 10.0, 100.0),
        (3, HestonParams(0.0001, 2.0, 0.04, -0.5, 3.0), 0.1, 100.0)]
    out = DataFrame(Case=Int[], Method=String[], Strike=Float64[], Error=Float64[], Reference=Float64[])
    for (case, params, τ, forward) in cases
        cf = DefaultCharFunc(params)
        refPricer = ALCharFuncPricer(cf, n=4000)
        for (method, pricer) in (("Flinn", FlinnCharFuncPricer(cf, τ, tTol=1e-8, qTol=1e-8)), ("Flinn-Transf.", AdaptiveFlinnCharFuncPricer(cf, τ)))
            for k in AL_STRIKES
                ref = CharFuncPricing.priceEuropean(refPricer, false, k, forward, τ, 1.0)
                push!(out, (case, method, k, CharFuncPricing.priceEuropean(pricer, false, k, forward, τ, 1.0) - ref, ref))
            end
        end
    end
    writetable("heston_challenge.csv", out)
end

## Table tbl:heston_formulae_accuracy
function heston100Puts(make, strikes, τ)
    best, prices = Inf, zeros(length(strikes))
    for _ in 1:30
        t = @elapsed begin
            pricer = make()
            for (i, k) in enumerate(strikes)
                prices[i] = CharFuncPricing.priceEuropean(pricer, false, k, 1.0, τ, 1.0)
            end
        end
        best = min(best, t)
    end
    return prices, best
end

@compute "heston_mild" begin
    params = [HestonParams(0.1, 1.0, 0.1, -0.9, 1.0), HestonParams(0.073, 1.417, 0.082, -0.384, 0.591)]
    strikes = collect(range(0.4, 1.6, length=100))
    out = DataFrame(Set=Int[], Maturity=Float64[], Method=String[], RMSE=Float64[], TimeMs=Float64[])
    for (set, p) in enumerate(params), τ in (2 / 52, 2 / 12, 2.0)
        cf = DefaultCharFunc(p)
        refPricer = ALCharFuncPricer(cf, n=4000)
        ref = [CharFuncPricing.priceEuropean(refPricer, false, k, 1.0, τ, 1.0) for k in strikes]
        for (method, make) in (("Andersen-Lake", () -> ALCharFuncPricer(cf, n=200)), ("Cos", () -> makeCosCharFuncPricer(cf, τ, 200, 8)),
            ("Cos-Adaptive", () -> makeCosCharFuncPricer(cf, τ, 8)), ("Flinn-Truncated", () -> FlinnCharFuncPricer(cf, τ, tTol=1e-8, qTol=1e-8)),
            ("Flinn-Transformed", () -> AdaptiveFlinnCharFuncPricer(cf, τ)))
            prices, t = heston100Puts(make, strikes, τ)
            push!(out, (set, τ, method, sqrt(mean(abs2, prices .- ref)), 1e3t))
        end
    end
    writetable("heston_mild.csv", out)
end
