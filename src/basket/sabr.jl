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
    variance::AbstractArray{SABRSection{T,M}}, #vol^2 and tte.
    weight::AbstractArray{<:T},
    correlation::Matrix{TV} #correlation matrix contains only asset correl except if correlationKind = full where it is 2Nx2N
)::T where {T,TV,M}
    ρAv = zero(T)
    ρνAv = zero(T)
    νAv = zero(T)
    νμAv = zero(T)
    ν2Av = zero(T)
    ν2AAv = zero(T)
    Δ = zero(T)
    x = sum([wi * fi for (wi, fi) = zip(weight, forward)])
    for (i,wi) = enumerate(weight)
        p = variance[i].params
        ρAv += p.ρ * wi * p.α
        ρνAv += p.ρ * wi * p.α * p.ν
        for (j,wj) = enumerate(weight)
            pj = variance[j].params
            Δ += correlation[i, j] * wi * wj * p.α * pj.α
            νAv += wj * pj.ν * pj.α * wi * p.α * correlation[i, j]
            # νμAv += wj*pj.ν*pj.α*wi*p.α*correlation[i,j]        
            ν2Av += wj * pj.ν * pj.α * wi * p.ν * p.α * correlation[i, j]
            ν2AAv += ((pj.ν + p.ν) / 2)^2 * pj.α * wj * wi * p.α * correlation[i, j]
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
    ρ = min(1.0,max(-1.0,η / ν))
   tte = variance[1].tte
    σSBatm = α * (1 + ((2 - 3ρ^2) * ν^2 + 6Γ) * tte / 24)
    σSB = if strike == x 
        σSBatm
    else

    ξ = ν / α * (strike - x)
    y = log((sqrt(1 + 2*ρ*ξ + ξ^2) + ρ + ξ) / (1 + ρ))
    σSBatm * ν / α * (strike - x) / y
    end
    price = bachelierFormula(isCall, strike, x, σSB, tte, discountFactor)
    return price
end


function priceEuropean(
    p::SABRBasketPricer{FullCorrelation},
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    variance::AbstractArray{SABRSection{T,M}}, #vol^2 and tte.
    weight::AbstractArray{<:T},
    correlation::Matrix{TV} #correlation matrix contains only asset correl except if correlationKind = full where it is 2Nx2N
)::T where {T,TV,M}
#a naive implementation for checks
    η = zero(T)
    g = zero(T)
    Δ = zero(T)
    κ = zero(T)
    ϕ = zero(T)
    d = length(weight)
    x = sum([wi * fi for (wi, fi) = zip(weight, forward)])
    for (i,wi) = enumerate(weight)
        p = variance[i].params
        for (j,wj) = enumerate(weight)
            pj = variance[j].params            
            Δ += correlation[i,j] * wj*pj.α*wi*p.α
            for (k,wk) = enumerate(weight)
                pk = variance[k].params
                η += correlation[i,j+d]*pj.ν*correlation[j,k]*wk*pk.α*wj*pj.α*wi*p.α
                for (m,wm) = enumerate(weight)
                    pm = variance[m].params
                    g += correlation[i+d,j+d]*p.ν*pj.ν*correlation[i,k]*correlation[j,m]*wm*pm.α*wk*pk.α*wj*pj.α*wi*p.α
                    ϕ +=  correlation[i+d,j+d]*p.ν*pj.ν*(correlation[i,j]*correlation[k,m] - correlation[i,k]*correlation[j,m])*wm*pm.α*wk*pk.α*wj*pj.α*wi*p.α 
                    for (n,wn) = enumerate(weight)
                        pn = variance[n].params
                        for (s,ws) = enumerate(weight)
                            ps = variance[s].params
                            κ += correlation[i,j+d]*pj.ν*(correlation[k,i+d]*p.ν+correlation[k,j+d]*pj.ν+correlation[k,m+d]*pm.ν-3correlation[k,s+d]*ps.ν)*correlation[j,m]*correlation[n,s]* ws*ps.α*wn*pn.α*wm*pm.α*wk*pk.α*wj*pj.α*wi*p.α
                        end
                    end
                end
            end
        end
    end
    Δ = sqrt(Δ)
    η /= Δ^3
    g /= Δ^4
    ϕ /= Δ^4
    κ /= Δ^6
    Γ = ϕ - κ 
    α = Δ
    ν = sqrt(g + κ)
    ρ = min(1.0,max(-1.0,η / ν))
    tte = variance[1].tte
    σSBatm = α * (1 + ((2 - 3ρ^2) * ν^2 + 6Γ) * tte / 24)
    σSB = if strike == x 
        σSBatm
    else

    ξ = ν / α * (strike - x)
    y = log((sqrt(1 + 2*ρ*ξ + ξ^2) + ρ + ξ) / (1 + ρ))
    σSBatm * ν / α * (strike - x) / y
    end
    price = bachelierFormula(isCall, strike, x, σSB, tte, discountFactor)
    return price
end