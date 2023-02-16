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