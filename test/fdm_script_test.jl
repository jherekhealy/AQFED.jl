using AQFED, Test
using AQFED.FDM, AQFED.TermStructure
import AQFED.Math: normcdf, GaussLegendre
import AQFED.Asian: priceAsianFixedStrike, priceAsianFloatingStrike
import AQFED.Basket: DeelstraBasketPricer

@testset "BackwardScript" begin
    σ, r, q, spot, T = 0.25, 0.03, 0.01, 100.0, 1.0
    df, F = exp(-r * T), spot * exp((r - q) * T)
    price(definition; M=401, N=201) = priceTRBDF2(definition, spot, F, σ^2 * T, df, CapitalizedDividend{Float64}[]; M=M, N=N)(spot)

    #single date autocall with the capital protection at the autocall level: a digital call and a share below the level
    level, coupon = 1.05, 1.08
    d1 = (log(F / (level * spot)) + σ^2 * T / 2) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    @test isapprox(price(autocallScript([T], [level], [coupon]; reference=spot, protectionLevel=level)), df * (coupon * normcdf(d2) + F / spot * normcdf(-d1)), atol=2e-4)
    #quarterly autocall, against MonteCarlo.AutocallNote with 2^18 paths: 0.97694972 (stderr 2.1e-4)
    times = [0.25, 0.5, 0.75, 1.0]
    @test isapprox(price(autocallScript(times, [1.0, 1.0, 1.05, 1.05], [1.02, 1.04, 1.06, 1.08]; reference=spot, protectionLevel=0.7); M=801, N=401), 0.97694972, atol=6e-4)

    #a discrete knock-out call written as a script is the built-in DiscreteKO
    koTimes = collect(range(T / 12, T, length=12))
    koScript = BackwardScript(
        on(T) do V, S, Q, k
            @. V = max(S - 100.0, 0)
        end,
        on(koTimes) do V, S, Q, k
            @. V = ifelse(S >= 130.0, 0.0, V)
        end; nonSmoothPoints=[100.0, 130.0])
    @test isapprox(price(koScript), price(DiscreteKO(VanillaEuropean(true, 100.0, T), 130.0, false, 0.0, koTimes)), rtol=1e-12)

    #volatility swap on the Carr-Lee synthetic payoff: σ in percent for a flat volatility
    @test isapprox(price(volatilitySwapScript(T, F)), 100σ * df, atol=2e-3)

    #monthly arithmetic Asian options, the state is the running average on 401 points
    fixings = collect(range(T / 12, T, length=12))
    forwards = spot .* exp.((r - q) .* fixings)
    deelstra = DeelstraBasketPricer(1, 3, GaussLegendre(101))
    stateGrid = collect(range(0.0, 300.0, length=401))
    for isCall in (true, false), strike in (90.0, 100.0, 110.0)
        ref = priceAsianFixedStrike(deelstra, isCall, strike, df, spot, forwards, σ^2 .* fixings, fill(1 / 12, 12))
        @test isapprox(price(asianScript(isCall, strike, fixings, stateGrid)), ref, atol=2e-3)
    end
    #floating strike on the average of the fixings before maturity
    for isCall in (true, false)
        ref = priceAsianFloatingStrike(deelstra, isCall, 1.0, df, spot, forwards, σ^2 .* fixings, vcat(fill(1 / 11, 11), 0.0))
        @test isapprox(price(asianScript(isCall, 0.0, fixings[1:11], stateGrid; isFloatingStrike=true, maturity=T)), ref, atol=2e-3)
    end
    #volatility swap on 12 monthly log returns, the states are the accumulated volatility and the log of the last fixing.
    #Exact reference for iid normal returns: E√X = 1/(2√π) ∫ (1 - E e^{-sX}) s^{-3/2} ds
    n = 12
    m, a2 = (r - q - σ^2 / 2) * T / n, σ^2 * T / n
    logLaplace(s) = n * (-log1p(2s * a2) / 2 - s * m^2 / (1 + 2s * a2))
    expectedVol = AQFED.Math.integrate(AQFED.Math.GaussKronrod(1e-12), s -> -expm1(logLaplace(s)) * s^(-3 / 2), 0.0, Inf) / (2sqrt(π))
    ref = 100 * df * sqrt(1 / T) * expectedVol
    script = discreteVolatilitySwapScript(collect(range(0.0, T, length=n + 1)), collect(range(0.0, 4σ, length=41)), collect(range(log(spot) - 5σ, log(spot) + 5σ, length=41)))
    @test isapprox(price(script), ref, atol=0.05)
    #carrLeeVolatilityPayoff is the integral of its second derivative
    h2(K) = (k = log(K / F) / 2; (K < F ? -1 : 1) * sqrt(π / (8K^3 * F)) * (AQFED.FDM.besselix(1, k) - AQFED.FDM.besselix(0, k)) * exp(abs(k)))
    hNumerical(S) = sqrt(π / 2) * abs(S - F) / F + (S > F ? AQFED.Math.integrate(AQFED.Math.GaussKronrod(1e-12), K -> h2(K) * (S - K), F, S) : AQFED.Math.integrate(AQFED.Math.GaussKronrod(1e-12), K -> h2(K) * (K - S), S, F))
    @test maximum(abs(carrLeeVolatilityPayoff(S, F) - hNumerical(S)) for S in (5.0, 50.0, 90.0, 110.0, 400.0, 2000.0)) < 1e-11
end
