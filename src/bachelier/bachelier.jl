using AQFED.Math
export bachelierFormula
function bachelierFormula(isCall::Bool, strike::Number, forward::Number, vol::Number, tte::Number, discountDf::Number)
    sign = 1
    if !isCall
        sign = -1
    end
    if vol == 0
        return max(sign * (forward - strike), 0)
    end
    sqrtvar = sqrt(tte) * vol
    d = sign * (forward - strike) / sqrtvar
    if forward == strike
        return sqrtvar * Math.OneOverSqrt2Pi
    end
    Nd = normcdf(d)
    nd = normpdf(d)
    return sign * (forward - strike) * Nd + sqrtvar * nd
end
