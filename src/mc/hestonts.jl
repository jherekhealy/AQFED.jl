using AQFED.TermStructure
import AQFED.Random: next!, nextn!, skipTo
import CharFuncPricing: HestonTSParams



function simulateFullTruncationIter(
    rng,
    model::HaganHestonPiecewiseParams,
    spot::T,
    forecastCurve::Curve,
    discountCurve::Curve,
    payoff::VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64;
    withBB=false,
) where {T}
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = discountFactor(discountCurve, tte)
    iLast = searchsortedfirst(model.startTime,tte)
    specTimesAll = sort(unique(vcat(specTimes, model.startTime[2:iLast-1])))
    genTimes = pathgenTimes(model, specTimesAll, timestepSize)
  local bb
    if withBB
        bb = BrownianBridgeConstruction(genTimes[2:end])
    end
    rhoFunc = PiecewiseConstantFunction(model.startTime, model.ρ)
    sigmaFunc = PiecewiseConstantFunction(model.startTime, model.σ)
    levFunc = PiecewiseConstantFunction(model.startTime, model.leverage)

    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, tte)
    z = Vector{Float64}(undef, (length(genTimes) - 1) * 2)
    u = Array{Float64}(undef, (2, length(genTimes) - 1))
    skipTo(rng, start)
    payoffMean = 0.0
    for sim ∈ 1:nSim
        if withBB
            nextn!(rng, z)
            transform!(bb, z, u)
        else
            nextn!(rng, z)
            t0 = genTimes[1]
            @inbounds for i ∈ 1:length(genTimes)-1
                dim = 2 * i - 1
                t1 = genTimes[i+1]
                u[1, i] = z[dim] * sqrt(t1 - t0)
                u[2, i] = z[dim+1] * sqrt(t1 - t0)
                t0 = t1
            end
        end
        t0 = genTimes[1]
        logpathValue = lnf0
        v = 1.0
        local payoffValue
        @inbounds for i ∈ 1:length(genTimes)-1
            t1 = genTimes[i+1]
            # dim = 2 * i - 1
            u1 = u[1, i]
            u2 = u[2, i]
            h = t1 - t0
            sqrtmv = sqrt(max(v, 0))
            lnf1 = logForward(forecastCurve, lnspot, t1)
            ρ = rhoFunc(t0)
            lev = levFunc(t0)
            ρBar = sqrt(1-ρ^2)
            logpathValue += lnf1 - lnf0 - 0.5 * (lev*sqrtmv)^2 * h + (lev*sqrtmv) * (u1 * ρBar + u2 * ρ)
            v += model.κ * (1.0 - sqrtmv^2) * h + sigmaFunc(t0) * sqrtmv * u2

            if t1 == tte
                pathValue = exp(logpathValue)
                payoffValue = evaluatePayoffOnPath(payoff, pathValue, df)
            end
            t0 = t1
            lnf0 = lnf1
        end

        payoffMean += payoffValue
    end
    payoffMean /= nSim
    return payoffMean, NaN
end
