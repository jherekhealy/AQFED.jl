import AQFED.Math: norminv
import Random: rand!, randn!, rand, randn
using Statistics
import AQFED.TermStructure:
    LocalVolatilityModel, VarianceSurfaceBySection, localVarianceByLogmoneyness

function simulate(
    rng,
    model::LocalVolatilityModel{S},
    payoff::VanillaOption,
    nSim::Int,
    nSteps::Int,
) where {S}
    tte = payoff.maturity
    df = exp(-model.r * tte)
    genTimes = LinRange(0.0, tte, ceil(Int, nSteps * tte) + 1)
    logpathValues = Vector{Float64}(undef, nSim)
    #z = Vector{Float64}(undef, nSim)
    t0 = genTimes[1]
    lnspot = log(model.spot)
    lnforward = lnspot
    logpathValues .= lnspot
    local payoffValues
    for t1 in genTimes[2:end]
        h = t1 - t0
        z = rand(rng, Float64, nSim)
        @. z = norminv(z)
        #z = randn(rng, Float64, nSim)
        lnforward += (model.r - model.q) * h
        # indexT0 = searchsortedlast(model.surface.expiries, t0)
        # indexT1 = searchsortedlast(model.surface.expiries, t1)

        # lvh = map(
        #     x -> localVarianceByLogmoneyness(model.surface, x - lnforward, t0, t1, indexT0, indexT1) * h,
        #     logpathValues,
        # )
        logmoneyness = @. logpathValues - lnforward
        lvh = localVarianceByLogmoneyness(model.surface, logmoneyness, t0, t1)
        lvh .*= h
        @. logpathValues += (model.r - model.q) * h - lvh / 2 + sqrt(lvh) * z
        if t1 == tte
            pathValues = lvh #reuse Var
            @. pathValues = exp(logpathValues)
            payoffValues = map(x -> evaluatePayoff(payoff, x, df), pathValues)
        end
        t0 = t1
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end
