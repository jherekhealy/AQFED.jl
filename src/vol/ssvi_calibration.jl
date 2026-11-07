using Optim
import AQFED.TermStructure: XSSVISection, varianceByLogmoneyness, VarianceSurfaceBySection
export calibrateXSSVISection

# ------------------------------------------------------------------
# 1. Section Calibration
# ------------------------------------------------------------------

ψplus(ρ, kstar, θstar) = -2 * ρ * kstar / (1 + abs(ρ)) + sqrt(4 * (ρ * kstar)^2 / (1 + abs(ρ))^2 + 4 * θstar / (1 + abs(ρ)))

function findZeroClosestIndex(ys)
    kstarbracket = searchsorted(ys, 0.0)
    if kstarbracket.start > length(ys)
        return length(ys) - 1
    elseif abs(ys[kstarbracket.start]) < abs(ys[kstarbracket.stop])
        return min(length(ys) - 1, max(2, kstarbracket.start))
    else
        return min(length(ys) - 1, max(2, kstarbracket.stop))
    end
end

# Fixed outer signature forwarding
calibrateXSSVISection(tte, forward, xs, vols, weights; prevSection=nothing, samplingLength=11, nLevel=6) = 
    calibrateXSSVISection(tte, forward, zeros(LogmoneynessAxisTransformation, length(xs)), xs, vols, weights; 
                          prevSection=prevSection, samplingLength=samplingLength, nLevel=nLevel)

function calibrateXSSVISection(
    tte, forward, axisTransforms::Vector{U}, xs, vols, weights; 
    prevSection=nothing, samplingLength=11, nLevel=6
) where {U<:AxisTransformation}
    
    ys = map((trans, x, vol) -> convertToLogmoneyness(trans, x, vol), axisTransforms, xs, vols)
    kstarIndex = findZeroClosestIndex(ys)
    ystar = ys[kstarIndex]
    θstar = vols[kstarIndex]^2 * tte

    if prevSection !== nothing
        θstar = max(θstar, prevSection.θstar + 1e-8)
    end

    currentObj = 1e300
    ψ = 0.0
    ρ = 0.0
    minRho = -0.99
    maxRho = 0.99

    while nLevel > 0
        rhos = range(minRho, stop=maxRho, length=samplingLength)
        for rho in rhos
            upper = min(ψplus(rho, ystar, θstar), 4 / (1 + abs(rho)))
            ψlim = θstar / (rho * ystar)
            if ψlim > 0
                upper = min(upper, ψlim)
            end

            lower = 1e-7
            if prevSection !== nothing
                lower = max(lower, prevSection.ψ * (prevSection.θstar / θstar))
            end

            if lower >= upper
                lower = max(1e-7, upper - 1e-6)
            end

            obj = function (ψ_val)
                section = XSSVISection(ystar, θstar, rho, ψ_val, tte, forward)
                sum_err = zero(vols[1])
                for (k, vol, w, trans) in zip(xs, vols, weights, axisTransforms)
                    y = solveLogmoneyness(trans, k, z -> varianceByLogmoneyness(section, z))
                    sum_err += w * (sqrt(varianceByLogmoneyness(section, y)) - vol)^2
                end
                
                if prevSection !== nothing
                    for y in ys
                        w_prev = varianceByLogmoneyness(prevSection, y) * prevSection.tte
                        w_curr = varianceByLogmoneyness(section, y) * tte
                        if w_curr < w_prev
                            sum_err += 1e4 * (w_prev - w_curr)^2
                        end
                    end
                end
                return sum_err
            end

            res = Optim.optimize(obj, lower, upper - 1e-7)
            if res.minimum < currentObj
                currentObj = res.minimum
                ψ = res.minimizer
                ρ = rho
            end
        end

        rhoWidth = rhos[2] - rhos[1]
        minRho = max(-1.0, ρ - rhoWidth)
        maxRho = min(1.0, ρ + rhoWidth)
        nLevel -= 1
    end

    return XSSVISection(ystar, θstar, ρ, ψ, tte, forward)
end

# ------------------------------------------------------------------
# 2. Surface Parameter Surface Methods
# ------------------------------------------------------------------
export calibrateXSSVISection, ShortEndExtrapolation, FlatShapeExtrapolation, LinearPsiExtrapolation, PowerLawPsiExtrapolation

# Define Short-end Extrapolation Strategies
abstract type ShortEndExtrapolation end
struct FlatShapeExtrapolation <: ShortEndExtrapolation end  # Option A: Preserves smile shape as t -> 0
struct LinearPsiExtrapolation <: ShortEndExtrapolation end  # Option B: Flattens smile as t -> 0

# ------------------------------------------------------------------
# 1. Surface Evaluation Overloads
# ------------------------------------------------------------------

varianceByLogmoneyness(
    s::VarianceSurfaceBySection{XSSVISection},
    y::Real,
    t::Real,
    indexTime::Int = 0;
    shortEndStrategy::ShortEndExtrapolation = LinearPsiExtrapolation()
) = varianceByLogmoneynessParametric(s, y, t, indexTime, shortEndStrategy)

function varianceByLogmoneynessParametric(
    s::VarianceSurfaceBySection{XSSVISection},
    y::Real,
    t::Real,
    indexTime::Int,
    strategy::FlatShapeExtrapolation
)::Real
    n = length(s.expiries)

    # ------------------------------------------------------------------
    # 1. Short-end Extrapolation (t <= t_1) — Option A (Flat Shape)
    # Scaled total variance θ*(t) = θ*_1 * (t / T1), fixed shape (ρ, ψ, y*)
    # ------------------------------------------------------------------
    if t <= s.expiries[1]
        sec1 = s.sections[1]
        λ = max(1e-8, t / s.expiries[1])
        θstar_t = sec1.θstar * λ
        
        # Keep shape parameters identical to slice 1
        sec_t = XSSVISection(sec1.ystar, θstar_t, sec1.ρ, sec1.ψ, t, sec1.f)
        return varianceByLogmoneyness(sec_t, y)

    # ------------------------------------------------------------------
    # 2. Long-end Extrapolation (t >= t_N)
    # Slope-constrained linear extrapolation matching arXiv:1804.04924 §5.3
    # ------------------------------------------------------------------
    elseif t >= s.expiries[end]
        secN_prev = s.sections[end-1]
        secN = s.sections[end]
        tN_prev = s.expiries[end-1]
        tN = s.expiries[end]

        dt = tN - tN_prev
        w = (t - tN) / dt

        # Monotonic non-decreasing slope for total variance
        dθstar = (secN.θstar - secN_prev.θstar) / dt
        θstar_t = secN.θstar + max(0.0, dθstar) * (t - tN)

        # Extrapolate shape parameters with bounds clamping
        ρ_t = clamp(secN.ρ + w * (secN.ρ - secN_prev.ρ), -0.999, 0.999)
        ψ_t = max(1e-7, secN.ψ + w * (secN.ψ - secN_prev.ψ))
        ystar_t = secN.ystar + w * (secN.ystar - secN_prev.ystar)

        sec_t = XSSVISection(ystar_t, θstar_t, ρ_t, ψ_t, t, secN.f)
        return varianceByLogmoneyness(sec_t, y)

    # ------------------------------------------------------------------
    # 3. Intermediate Interpolation (t_1 < t < t_N)
    # Linear parameter interpolation across adjacent slices
    # ------------------------------------------------------------------
    else
        if indexTime == 0
            indexTime = searchsortedlast(s.expiries, t)
        end

        t0 = s.expiries[indexTime]
        t1 = s.expiries[indexTime+1]
        sec0 = s.sections[indexTime]
        sec1 = s.sections[indexTime+1]

        α = (t - t0) / (t1 - t0)

        θstar_t = (1.0 - α) * sec0.θstar + α * sec1.θstar
        ρ_t     = (1.0 - α) * sec0.ρ     + α * sec1.ρ
        ψ_t     = (1.0 - α) * sec0.ψ     + α * sec1.ψ
        ystar_t = (1.0 - α) * sec0.ystar + α * sec1.ystar
        fwd_t   = (1.0 - α) * sec0.f     + α * sec1.f

        sec_t = XSSVISection(ystar_t, θstar_t, ρ_t, ψ_t, t, fwd_t)
        return varianceByLogmoneyness(sec_t, y)
    end
end

# Implementation for Option B (Linear ψ Scaling)
function varianceByLogmoneynessParametric(
    s::VarianceSurfaceBySection{XSSVISection},
    y::Real,
    t::Real,
    indexTime::Int,
    strategy::LinearPsiExtrapolation
)::Real
    if t <= s.expiries[1]
        sec1 = s.sections[1]
        λ = max(1e-8, t / s.expiries[1])
        θstar_t = sec1.θstar * λ
        ψ_t = sec1.ψ * λ
        
        sec_t = XSSVISection(sec1.ystar, θstar_t, sec1.ρ, ψ_t, t, sec1.f)
        return varianceByLogmoneyness(sec_t, y)
    else
        return varianceByLogmoneynessParametric(s, y, t, indexTime, FlatShapeExtrapolation())
    end
end

struct PowerLawPsiExtrapolation <: ShortEndExtrapolation
    γ::Float64
    PowerLawPsiExtrapolation(γ=0.75) = new(γ)
end

function varianceByLogmoneynessParametric(
    s::VarianceSurfaceBySection{XSSVISection},
    y::Real,
    t::Real,
    indexTime::Int,
    strategy::PowerLawPsiExtrapolation
)::Real
    if t <= s.expiries[1]
        sec1 = s.sections[1]
        λ = max(1e-8, t / s.expiries[1])
        θstar_t = sec1.θstar * λ
        

        # Scaling ψ with exponent γ dampens skew gradient as t -> 0
        ψ_t = sec1.ψ * (λ^strategy.γ)
        #note: if theta = C1*t= C2*lambda, psi= theta^gamma
        
        
        sec_t = XSSVISection(sec1.ystar, θstar_t, sec1.ρ, ψ_t, t, sec1.f)
        return varianceByLogmoneyness(sec_t, y)
    else
        return varianceByLogmoneynessParametric(s, y, t, indexTime, FlatShapeExtrapolation())
    end
end