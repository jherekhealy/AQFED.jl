using Optim
import AQFED.TermStructure: XSSVISection, varianceByLogmoneyness
export calibrateXSSVISection

ψplus(ρ, kstar, θstar) = -2 * ρ * kstar / (1 + abs(ρ)) + sqrt(4 * (ρ * kstar)^2 / (1 + abs(ρ))^2 + 4 * θstar / (1 + abs(ρ)))

function findZeroClosestIndex(ys)
    kstarbracket = searchsorted(ys, 0.0)
    if kstarbracket.start > length(ys)
        length(ys) - 1
    elseif abs(ys[kstarbracket.start]) < abs(ys[kstarbracket.stop])
        min(length(ys) - 1, max(2, kstarbracket.start))
    else
        min(length(ys) - 1, max(2, kstarbracket.stop))
    end
end


calibrateXSSVISection(tte, forward, xs, vols, weights; samplingLength=11, nLevel=6) = calibrateXSSVISection(tte, forward, zeros(LogmoneynessAxisTransformation, length(xs)), xs, vols, weights; samplingLength=samplingLength, nLevel=nLevel)


function calibrateXSSVISection(tte, forward, axisTransforms::Vector{U}, xs, vols, weights; samplingLength=11, nLevel=6) where {U<:AxisTransformation}
    ys = map((trans, x, vol) -> convertToLogmoneyness(trans, x, vol), axisTransforms, xs, vols)

    kstarIndex = findZeroClosestIndex(ys)

    kstar = ys[kstarIndex]
    θstar = vols[kstarIndex]^2 * tte


    #rhos = range(-1.0, stop=1.0, length=samplingLength)
    currentObj = 1e300
    ψ = 0.0
    ρ = 0.0
    minRho = -0.99
    maxRho = 0.99
    while nLevel > 0
        rhos = range(minRho, stop=maxRho, length=samplingLength)
        for rho in rhos
            upper = min(ψplus(rho, kstar, θstar), 4 / (1 + abs(rho)))
            ψlim = θstar / (rho * kstar)
            if ψlim > 0
                upper = min(upper, ψlim)
            end
            obj = function (ψ)
                section = XSSVISection(kstar, θstar, rho, ψ, tte, forward)
                sum = zero(vols[1])
                #  sum(w * (sqrt(varianceByLogmoneyness(section, k)) - vol)^2 for (k, vol, w) in zip(ys, vols, weights))
                for (k, vol, w, trans) in zip(xs, vols, weights, axisTransforms)
                    y = solveLogmoneyness(trans, k, z -> varianceByLogmoneyness(section, z))
                    sum += w * (sqrt(varianceByLogmoneyness(section, y)) - vol)^2
                end
                sum
            end
            res = Optim.optimize(obj, 1e-7, upper - 1e-7)
            if res.minimum < currentObj
                currentObj = res.minimum
                ψ = res.minimizer
                ρ = rho
                #println("better result ",res, " ",ρ, " ",ψ)
            end
        end
        rhoWidth = rhos[2] - rhos[1]
        minRho = max(-1.0, ρ - rhoWidth)
        maxRho = min(1.0, ρ + rhoWidth)
        nLevel -= 1
    end
    return XSSVISection(kstar, θstar, ρ, ψ, tte, forward)
end
