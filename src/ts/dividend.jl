export futureValue, Dividend, CapitalizedDividend

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


function futureValue(cd::CapitalizedDividend{T})::T where {T}
    return cd.dividend.amount*cd.capitalizationFactor
end
