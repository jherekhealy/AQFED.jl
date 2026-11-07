export futureValue, Dividend, CapitalizedDividend, DividendDriftCurve

struct Dividend{T}
    amount::T
    exDate::T
    payDate::T
    isProportional::Bool
    isKnown::Bool
end

struct CapitalizedDividend{T}
    dividend::Dividend{T}
    capitalizationFactor::T #growth factor from the dividend ex date up to a maturity τ
end

#Base.length(c::CapitalizedDividend) = 1

function futureValue(cd::CapitalizedDividend{T})::T where {T}
    return cd.dividend.amount*cd.capitalizationFactor
end

struct DividendDriftCurve{TX,T} <: Curve
    rateCurve::TX 
    dividends::Vector{Dividend{T}} #sorted asc
    spot::T #for cash
end
function discountFactor(c::DividendDriftCurve{TX,T}, time::TZ)::T where {T,TX,TZ}
    αs, βs = alphaBeta(c, time)
    #S/df = F
    return 1.0/(βs/discountFactor(c.rateCurve,time) - αs/c.spot)
end
    
function logDiscountFactor(c::DividendDriftCurve{TX,T}, time::TZ)::T where {T,TX,TZ}
    return log(discountFactor(c,time))
end

Base.broadcastable(p::Curve) = Ref(p)


function alphaBeta(c::DividendDriftCurve{TX,T},time::TZ) where {T,TX,TZ}
    βs = one(T)
    αs = zero(T)
    for dividend in c.dividends
        if dividend.exDate > time
            break
        end
        if dividend.isProportional
            βs *= one(T)-dividend.amount
        else
            αs += dividend.amount * discountFactor(c.rateCurve, dividend.exDate)/discountFactor(c.rateCurve, time)            
        end
    end
    return αs, βs
end

export DiscreteDividendMarket, withDividendAmount, withGrowthRate
#A Black-Scholes market with discrete dividends and constant continuously compounded rates:
#  r: discount rate, μ: growth rate of the asset excluding the discrete dividends (r minus repo/borrow minus continuous yield),
#  dividends: sorted by ex-date. A proportional dividend multiplies the spot by (1-amount) on the ex-date,
#  a cash dividend substracts its amount and is capitalized from its payment date in the forward.
struct DiscreteDividendMarket{T}
    spot::T
    r::T
    μ::T
    dividends::Vector{Dividend{T}}
    DiscreteDividendMarket(spot::T, r::T, μ::T, dividends::AbstractVector{Dividend{T}}) where {T} =
        new{T}(spot, r, μ, sort(collect(dividends), by = d -> d.exDate))
end
DiscreteDividendMarket(spot::T, r::T, μ::T) where {T} = DiscreteDividendMarket(spot, r, μ, Dividend{T}[])

discountFactor(m::DiscreteDividendMarket, τ) = exp(-m.r * τ)

function forward(m::DiscreteDividendMarket{T}, τ) where {T}
    growth = exp(m.μ * τ)
    f = m.spot * growth
    for d in m.dividends
        if d.exDate > τ
            break
        end
        if d.isProportional
            f *= one(T) - d.amount
        else
            f -= d.amount * growth * exp(-m.μ * d.payDate)
        end
    end
    return f
end

#same market where all the dividends of ex-date <= τ pay the given amount.
withDividendAmount(m::DiscreteDividendMarket{T}, amount::T, τ = typemax(T)) where {T} = DiscreteDividendMarket(m.spot, m.r, m.μ,
    [d.exDate <= τ ? Dividend(amount, d.exDate, d.payDate, d.isProportional, d.isKnown) : d for d in m.dividends])
withGrowthRate(m::DiscreteDividendMarket{T}, μ::T) where {T} = DiscreteDividendMarket(m.spot, m.r, μ, m.dividends)
