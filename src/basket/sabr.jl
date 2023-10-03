using AQFED.Math
import AQFED.Bachelier:bachelierFormula
import AQFED.TermStructure: SABRSection
export SABRBasketPricer, SingleStochasticVol

@enum CorrelationKind SingleStochasticVol ZeroCorrelation FullCorrelation
struct SABRBasketPricer{M}
end

#calibrate two smiles (eventually pre calibrated) against Black vols
#pass in the smiles. need also correlations dWi.dZj and dZi.dZj. We may assume dWidZi=rhosabri, and dWidZj=0 otherwise. and dZ.dZ = I. (helper method for this setting)
#other way: the Hagan single stoch vol way.
# plot case flat black vols, calibrated to 3 or 5/10 quotes. 0.75,1.0,1.25 1y? compare with quadrature 2 assets, price, IV plot? is IVbasket flat for quad?
# real calibration of 2 assets with smile. Show difference vs Black, ATM vs non ATM.
function priceEuropean(
    p::SABRBasketPricer{SingleStochasticVol},
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    variance::AbstractArray{SABRSection}, #vol^2 and tte.
    weight::AbstractArray{<:T},
    correlation::Matrix{TV} #correlation matrix contains only asset correl except if correlationKind = full where it is 2Nx2N
)::T where {T,TV}
    ρAv = zero(T)
    ρνAv = zero(T)
    νAv = zero(T)
    νμAv = zero(T)
    ν2Av = zero(T)
    ν2AAv = zero(T)
    Δ = zero(T)
    x = sum([wi * fi for (wi, fi) = zip(weight, forward)])
    for (wi, variancei) = zip(weight, variance)
        ρAv += variancei.ρ * wi * variancei.α
        ρνAv += variancei.ρ * wi * variancei.α * variancei.ν
        for (wj, variancej) = zip(weight, variance)
            Δ += correlation[i, j] * wi * wj * variancei.α * variancej.α
            νAv += wj * variancej.ν * variancej.α * wi * variancei.α * correlation[i, j]
            # νμAv += wj*variancej.ν*variancej.α*wi*variancei.α*correlation[i,j]        
            ν2Av += wj * variancej.ν * variancej.α * wi * variancei.ν * variancei.α * correlation[i, j]
            ν2AAv += ((variancej.ν + variancei.ν) / 2)^2 * variancej.α * wj * wi * variancei.α * correlation[i, j]
        end
    end
    Δ = sqrt(Δ)
    ρAv /= Δ
    ρνAv /= Δ
    νAv /= Δ^2
    νμAv /= Δ^2
    ν2Av /= Δ^2
    ν2AAv /= Δ^2
    η = ρAv * νAv
    g = νAv^2
    ϕ = ν2Av - νAv^2
    ϑ = νμAv
    κ = ρAv * ρνAv * νAv + 2 * ρAv^2 * ν2AAv - 3 * ρAv^2 * νAv^2
    Γ = ϕ - κ + 2ϑ
    α = Δ
    ν = sqrt(g + κ)
    ρ = η / ν
    tte = variance[1].tte
    σSBatm = α * (1 + ((2 - 3ρ^2) * ν^2 + 6Γ) * tte / 24)
    ξ = ν / α * (strike - x)
    y = log((sqrt(1 + 2ρξ + ξ^2) + ρ + ξ) / (1 + ρ))
    σSB = σSBatm * ν / α * (strike - x) / y

    price = bachelierFormula(isCall, strike, x, σSB, tte, discountFactor)
    return price
end