export futureValue, Dividend, CapitalizedDividend

struct Dividend
    amount::Number
    exDate::Number
    payDate::Number
    isProportional::Bool
    isKnown::Bool
end

struct CapitalizedDividend
    dividend::Dividend
    capitalizationFactor::Number #growth factor from the dividend ex date up to a maturity τ
end


function futureValue(cd::CapitalizedDividend)::Number
    return cd.dividend.amount*cd.capitalizationFactor
end
