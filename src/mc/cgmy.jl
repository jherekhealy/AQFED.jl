import AQFED.Math: norminv
using Statistics
using CharFuncPricing


function simulate(rng, model::CGMYParams{T}, r::T, q::T, spot::T, payoff::VanillaOption, nSim::Int, n::Int;) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = exp(-r * tte)
    dt = 1.0 / n
    lnspot = log(spot)
    lnf0 = lnspot + (r - q) * tte
    dimt = Int(floor(n * tte))
    genTimes = collect(range(0.0, stop=tte, length=(1 + dimt)))
    u = zeros(T, dimt)

    cf = DefaultCharFunc(model)
    cfPricer = makeCosCharFuncPricer(cf, dt, 1024 * 10, 16)
    logxmin = -sqrt(dt) * 10
    logxmax = -logxmin
    x = collect(range(logxmin, stop=logxmax, length=1024))
    y = exp.(x)
    cumDist = map(xi -> priceDigital(cfPricer, false, xi, 1.0, dt, 1.0), y)
    sort!(cumDist) #just in case there is some numerical error
    println(cumDist)
    specValues = zeros(T, 1)
    payoffValues = zeros(T, nSim)
    for sim = 1:nSim
        next!(rng, u)
        logpathValue = lnspot
        for i = 1:dimt
            l = searchsortedlast(cumDist, u[i])
            increment = if l <= 1
                x[1]
            elseif l >= length(x)
                x[end]
            else
                (u[i] * (x[l+1] - x[l]) + x[l] * cumDist[l+1] - x[l+1] * cumDist[l]) / (cumDist[l+1] - cumDist[l])
            end
            logpathValue += increment
        end
        specValues[1] = exp(logpathValue + lnf0 - lnspot)
        payoffValues[sim] = evaluatePayoffOnPath(payoff, specValues, df)
    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end