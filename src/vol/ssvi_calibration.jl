using Optim
import AQFED.TermStructure: XSSVISection, varianceByLogmoneyness
export calibrateXSSVISection

ψplus(ρ, kstar, θstar) = -2 * ρ * kstar / (1 + abs(ρ)) + sqrt(4 * (ρ * kstar)^2 / (1 + abs(ρ))^2 + 4 * θstar / (1 + abs(ρ)))

function calibrateXSSVISection(tte, forward, ys, vols, weights; samplingLength=11, nLevel=6)
    kstarbracket = searchsorted(ys, 0.0)
    kstarIndex = if abs(ys[kstarbracket.start]) < abs(ys[kstarbracket.stop])
        kstarbracket.start
    else
        kstarbracket.stop
    end
    kstar = ys[kstarIndex]
    θstar = vols[kstarIndex]^2 * tte


    rhos = range(-1.0, stop=1.0, length=samplingLength)
    currentObj = 1e300
    ψ = 0.0
    ρ = 0.0
    minRho = -1.0
    maxRho = 1.0
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
                sum(w * (sqrt(varianceByLogmoneyness(section, k)) - vol)^2 for (k, vol, w) in zip(ys, vols, weights))
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
